"""Gemma's global layers (head size 512) on sm_120 GPUs, where flash attention (head size up to 256) and cuDNN's FROST
kernels (sm_100) do not run: one Triton flash-attention forward over every virtual sequence of a pack, bidirectional
within each, query and key boundaries read on the device (a CUDA graph replays it), each query head over its key-value
head. Inference only. Against PyTorch's memory-efficient kernel, the path it replaces there: about twice as fast at
8k-32k tokens, outputs within bf16 rounding of it.
"""

from __future__ import annotations

import math
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # no Triton (CPU-only installs): not supported
    triton = tl = None

#: Query rows and key rows a program takes per step (the head splits into four chunks), and its warps.
BLOCK_M, BLOCK_N, WARPS = 64, 64, 8

if triton is not None:

    @triton.jit
    def _forward(Q, K, V, Out, q_cu, k_cu, scale_log2, s_qt, s_qh, s_kt, s_kh, s_vt, s_vh, s_ot, s_oh,
                 GROUP: tl.constexpr, CHUNK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        block, head, seq = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        q_start, q_end = tl.load(q_cu + seq), tl.load(q_cu + seq + 1)
        k_start, k_end = tl.load(k_cu + seq), tl.load(k_cu + seq + 1)
        n_q, n_k = q_end - q_start, k_end - k_start
        kv = head // GROUP
        # a sequence's base pointers are 64-bit (a pack's keys pass 2^31 elements), the offsets inside it 32-bit
        qb = Q + q_start.to(tl.int64) * s_qt + head * s_qh
        kb = K + k_start.to(tl.int64) * s_kt + kv * s_kh
        vb = V + k_start.to(tl.int64) * s_vt + kv * s_vh
        ob = Out + q_start.to(tl.int64) * s_ot + head * s_oh
        m0 = block * BLOCK_M
        k_stop = tl.where(m0 < n_q, n_k, 0)  # a block past its sequence's queries reads no key
        rows = m0 + tl.arange(0, BLOCK_M)
        live = rows < n_q
        d = tl.arange(0, CHUNK)
        qp = qb + rows[:, None] * s_qt + d[None, :]
        q0 = tl.load(qp, mask=live[:, None], other=0.0)
        q1 = tl.load(qp + CHUNK, mask=live[:, None], other=0.0)
        q2 = tl.load(qp + 2 * CHUNK, mask=live[:, None], other=0.0)
        q3 = tl.load(qp + 3 * CHUNK, mask=live[:, None], other=0.0)
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        o0 = tl.zeros([BLOCK_M, CHUNK], dtype=tl.float32)
        o1 = tl.zeros([BLOCK_M, CHUNK], dtype=tl.float32)
        o2 = tl.zeros([BLOCK_M, CHUNK], dtype=tl.float32)
        o3 = tl.zeros([BLOCK_M, CHUNK], dtype=tl.float32)
        for n0 in range(0, k_stop, BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            seen = cols < n_k
            kp = kb + cols[:, None] * s_kt + d[None, :]
            s = tl.dot(q0, tl.trans(tl.load(kp, mask=seen[:, None], other=0.0)))
            s = tl.dot(q1, tl.trans(tl.load(kp + CHUNK, mask=seen[:, None], other=0.0)), s)
            s = tl.dot(q2, tl.trans(tl.load(kp + 2 * CHUNK, mask=seen[:, None], other=0.0)), s)
            s = tl.dot(q3, tl.trans(tl.load(kp + 3 * CHUNK, mask=seen[:, None], other=0.0)), s)
            s = tl.where(seen[None, :], s * scale_log2, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            p = p.to(tl.bfloat16)
            vp = vb + cols[:, None] * s_vt + d[None, :]
            o0 = tl.dot(p, tl.load(vp, mask=seen[:, None], other=0.0), o0 * alpha[:, None])
            o1 = tl.dot(p, tl.load(vp + CHUNK, mask=seen[:, None], other=0.0), o1 * alpha[:, None])
            o2 = tl.dot(p, tl.load(vp + 2 * CHUNK, mask=seen[:, None], other=0.0), o2 * alpha[:, None])
            o3 = tl.dot(p, tl.load(vp + 3 * CHUNK, mask=seen[:, None], other=0.0), o3 * alpha[:, None])
            m_i = m_new
        inv = 1.0 / l_i
        op = ob + rows[:, None] * s_ot + d[None, :]
        tl.store(op, (o0 * inv[:, None]).to(tl.bfloat16), mask=live[:, None])
        tl.store(op + CHUNK, (o1 * inv[:, None]).to(tl.bfloat16), mask=live[:, None])
        tl.store(op + 2 * CHUNK, (o2 * inv[:, None]).to(tl.bfloat16), mask=live[:, None])
        tl.store(op + 3 * CHUNK, (o3 * inv[:, None]).to(tl.bfloat16), mask=live[:, None])


def supported(device: torch.device) -> bool:
    """An sm_120 GPU with Triton (the kernel's tiles are sized for its shared memory; untested elsewhere)."""
    return triton is not None and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 12


def attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_cu: torch.Tensor, k_cu: torch.Tensor,
           max_q: int, scale: float) -> torch.Tensor:
    """Bidirectional attention per virtual sequence: (Lq, heads, D) bf16 queries over (Lk, kv heads, D) keys and
    values, sequence i's query rows q_cu[i]:q_cu[i+1] over its key rows k_cu[i]:k_cu[i+1] (int32 on the device),
    ``max_q`` at least its longest query range; (Lq, heads, D). Row offsets inside a sequence are 32-bit (its rows
    times the row stride stay below 2^31); a whole pack may exceed that."""
    if torch.is_grad_enabled():
        raise RuntimeError("triton_attention runs the forward only: call it under torch.inference_mode or no_grad")
    out = torch.empty_like(query)
    grid = (triton.cdiv(max_q, BLOCK_M), query.shape[1], q_cu.numel() - 1)
    _forward[grid](query, key, value, out, q_cu, k_cu, scale * math.log2(math.e), query.stride(0), query.stride(1),
                   key.stride(0), key.stride(1), value.stride(0), value.stride(1), out.stride(0), out.stride(1),
                   GROUP=query.shape[1] // key.shape[1], CHUNK=query.shape[2] // 4, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                   num_warps=WARPS, num_stages=1)
    return out


def enable(model: Any) -> None:
    """``attend`` as the global layers' attention of ``model`` (a ``Packed``); ValueError where it does not apply."""
    if not supported(model.lm.embed_tokens.weight.device):
        raise ValueError("triton_attention needs an sm_120 GPU")
    if model.causal_global:
        raise ValueError("triton_attention reads bidirectional checkpoints only")
    model.global_attend = attend
