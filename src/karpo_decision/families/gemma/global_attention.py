"""Gemma's global layers (head size 512, beyond flash attention's 256) on cuDNN's FROST SDPA kernels (cudnn-frontend
1.32, cutlass-dsl 4.8, Blackwell), one virtual sequence per call, the same attention as the memory-efficient path:

- forward: queries, keys and values padded at their end to a length bucket, cuDNN's padding mask over the real lengths,
  causal aligned at the end (a branch's queries follow its shared keys);
- backward: cuDNN's dense backward on a bucket padded at the FRONT on both sides, with zero queries, keys, values,
  outputs and output gradients and log-sum-exp 0 in the padding, and the forward's own outputs and log-sum-exp on the
  real rows. The real rows keep their pattern (aligned at the end); a zero key adds nothing to a query's gradient, a
  zero output gradient adds nothing to a key's or a value's, so the real gradients are the exact ones.

The backward's kernels are compiled per shape (seconds each; cuDNN keeps them on disk,
``CUDNN_FRONTEND_COMPILED_CACHE``), so lengths go to buckets (``bucket``) and
``python -m karpo_decision.families.gemma.global_attention`` compiles every bucket pair once, ahead of a run.
Against the memory-efficient path: several times faster (forward and backward), outputs and gradients within bf16
rounding of it.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
try:
    from cudnn.sdpa.fwd import torch_op as _op  # registers torch.ops.cudnn.sdpa_fwd / sdpa_bwd
except ImportError:  # no cudnn-frontend: the memory-efficient path serves
    _op = None
else:
    # One graph per bucket pair and direction, built once per process; the default 128 evicts within a training step.
    _op._GRAPH_CACHE_MAX = 1 << 16

HEAD = 512
# (step, up to): multiples of 256 (the kernels' own tile) to 4,096, then coarser; at most 1/8 extra rows past 4,096.
BUCKETS = ((256, 4096), (512, 8192), (1024, 16384), (2048, 32768), (4096, 1 << 30))


def usable(query: torch.Tensor) -> bool:
    return (_op is not None and query.is_cuda and query.shape[-1] == HEAD
            and query.dtype in (torch.bfloat16, torch.float16)
            and torch.cuda.get_device_capability(query.device)[0] == 10)


def bucket(n: int) -> int:
    for step, top in BUCKETS:
        if n <= top:
            return -(-n // step) * step
    raise ValueError(f"{n} rows is beyond the last bucket")


def buckets(longest: int) -> list[int]:
    """Every bucket up to the one holding ``longest`` rows."""
    out, n = [], 1
    while not out or out[-1] < bucket(longest):
        out.append(bucket(n))
        n = out[-1] + 1
    return out


def lengths(q_bounds: list[int], k_bounds: list[int], device: torch.device) -> torch.Tensor:
    """(sequences, 2, 4) int32: [i, 0, 0] query rows, [i, 1, 0] key rows; a row of four keeps each length 16-byte
    aligned, which cuDNN reads in place."""
    t = torch.zeros(len(q_bounds) - 1, 2, 4, dtype=torch.int32)
    t[:, 0, 0] = torch.tensor(q_bounds).diff()
    t[:, 1, 0] = torch.tensor(k_bounds).diff()
    return t.to(device)


def _bhsd(rows: torch.Tensor, size: int, front: bool) -> torch.Tensor:
    """(L, heads, D) rows padded with zeros to (1, size, heads, D), at its end (front padding) or start, as a
    (1, heads, size, D) view: rows stay heads-inner, the layout the kernels read without a staging copy. Only the
    padding is zeroed."""
    n = rows.shape[0]
    buf = rows.new_empty(1, size, *rows.shape[1:])
    real, pad = (buf[0, size - n:], buf[0, :size - n]) if front else (buf[0, :n], buf[0, n:])
    real.copy_(rows)
    pad.zero_()
    return buf.transpose(1, 2)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, q_len, k_len, causal, scale):
        lq, lk = query.shape[0], key.shape[0]
        sq, sk = bucket(lq), bucket(lk)
        out, lse = torch.ops.cudnn.sdpa_fwd(_bhsd(query, sq, False), _bhsd(key, sk, False), _bhsd(value, sk, False),
                                            scale, is_causal=causal, causal_bottom_right=causal, seq_len_q=q_len,
                                            seq_len_kv=k_len, return_lse=True)
        out = out[0, :, :lq].transpose(0, 1).contiguous()
        ctx.save_for_backward(query, key, value, out, lse[0, :, :lq, 0].contiguous())
        ctx.causal, ctx.scale = causal, scale
        return out

    @staticmethod
    def backward(ctx, grad):
        query, key, value, out, lse = ctx.saved_tensors
        lq, lk = query.shape[0], key.shape[0]
        sq, sk = bucket(lq), bucket(lk)
        stats = lse.new_zeros(1, lse.shape[0], sq, 1)
        stats[0, :, sq - lq:, 0] = lse
        dq, dk, dv = torch.ops.cudnn.sdpa_bwd(
            _bhsd(grad.contiguous(), sq, True), _bhsd(query, sq, True), _bhsd(key, sk, True), _bhsd(value, sk, True),
            _bhsd(out, sq, True), stats, ctx.scale, is_causal=ctx.causal, causal_bottom_right=ctx.causal)
        return (dq[0, :, sq - lq:].transpose(0, 1), dk[0, :, sk - lk:].transpose(0, 1),
                dv[0, :, sk - lk:].transpose(0, 1), None, None, None, None)


def attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_len: torch.Tensor, k_len: torch.Tensor,
           causal: bool, scale: float) -> torch.Tensor:
    """One virtual sequence: (Lq, heads, 512) queries over (Lk, kv heads, 512) keys and values, ``q_len`` / ``k_len``
    their lengths on the device (``lengths``); (Lq, heads, 512)."""
    return _Attention.apply(query, key, value, q_len, k_len, causal, scale)


def warm(longest: int, heads: int, kv_heads: int, causal: bool, part: int, parts: int, scale: float = 1.0) -> None:
    """Compile the backward of every bucket pair (queries never outnumber keys) up to ``longest`` rows, this process's
    share (``part`` of ``parts``); the kernels land in cuDNN's disk cache. ``scale`` is part of a graph's key: Gemma's
    attention scale is 1 (queries and keys are normed)."""
    pairs = [(sq, sk) for sk in buckets(longest) for sq in buckets(sk)][part::parts]
    t0 = time.time()
    for i, (sq, sk) in enumerate(pairs):
        q = torch.zeros(1, sq, heads, HEAD, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
        k = torch.zeros(1, sk, kv_heads, HEAD, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
        stats = torch.zeros(1, heads, sq, 1, dtype=torch.float32, device="cuda")
        torch.ops.cudnn.sdpa_bwd(q, q, k, k, q, stats, scale, is_causal=causal, causal_bottom_right=causal)
        if i % 20 == 0:
            print(f"part {part}: {i + 1}/{len(pairs)} ({sq}, {sk}) {time.time() - t0:.0f} s", flush=True)
    torch.cuda.synchronize()
    print(f"part {part}: {len(pairs)} bucket pairs in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=warm.__doc__)
    parser.add_argument("--longest", type=int, default=32768)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=16)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    a = parser.parse_args()
    warm(a.longest, a.heads, a.kv_heads, not a.bidirectional, a.part, a.parts, a.scale)
