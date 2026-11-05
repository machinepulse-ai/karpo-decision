"""Gemma's sliding-window layers (head size 256, causal by position, a query sees itself and the window - 1 keys
before it) over a pack's virtual sequences, queries and keys with their own boundaries (a branch's keys are the shared
block's tail, then its own rows; causal aligned at each sequence's end).

On Blackwell with flash-attention 4 installed (CuTe DSL kernels, ``flash-attn-4``), its varlen kernels; otherwise
torch's varlen flash attention (FlashAttention-2 kernels). FA4's dedicated d256 backward refuses windowed attention;
its generic SM100 d256 backward (``FLASH_ATTENTION_HD256_GENERIC_BWD=1``) takes it. Against float32, outputs and
gradients stay within bf16 rounding (as the FA2 path's).

FA4 runs as a torch operator (``karpo::fa4_sliding``, its backward ``karpo::fa4_sliding_backward``) rather than FA4's
own autograd function: a selectively recomputed layer then keeps its output (``model.packed.KEPT_OPS``) instead of
running the forward kernel again in backward.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor
from torch.nn.attention.varlen import varlen_attn

os.environ.setdefault("FLASH_ATTENTION_HD256_GENERIC_BWD", "1")
# FA4 compiles a kernel per configuration it picks from the shapes (GQA packing, splits, tiles), each in ~0.4 s, on
# every rank: compiled kernels persist on disk (file-locked, under FLASH_ATTENTION_CUTE_DSL_CACHE_DIR or the temporary
# directory) so a run or rank finds the ones any earlier one built.
os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
try:
    from flash_attn.cute import interface as _fa4
except ImportError:  # no flash-attn-4: torch's varlen attention serves
    _fa4 = None

HEAD = 256


@torch.library.custom_op("karpo::fa4_sliding", mutates_args=())
def _forward(query: Tensor, key: Tensor, value: Tensor, q_cu: Tensor, k_cu: Tensor, max_q: int, max_k: int,
             scale: float, window: int) -> tuple[Tensor, Tensor]:
    out, lse, *_ = _fa4._flash_attn_fwd(query, key, value, cu_seqlens_q=q_cu, cu_seqlens_k=k_cu, max_seqlen_q=max_q,
                                        max_seqlen_k=max_k, softmax_scale=scale, causal=True,
                                        window_size_left=window, window_size_right=0, return_lse=True)
    return out, lse


@_forward.register_fake
def _(query, key, value, q_cu, k_cu, max_q, max_k, scale, window):
    return query.new_empty(query.shape), query.new_empty(query.shape[1], query.shape[0], dtype=torch.float32)


@torch.library.custom_op("karpo::fa4_sliding_backward", mutates_args=())
def _backward(grad: Tensor, query: Tensor, key: Tensor, value: Tensor, out: Tensor, lse: Tensor, q_cu: Tensor,
              k_cu: Tensor, max_q: int, max_k: int, scale: float, window: int) -> tuple[Tensor, Tensor, Tensor]:
    dq, dk, dv, *_ = _fa4._flash_attn_bwd(query, key, value, out, grad, lse, scale, True, 0.0,
                                          window_size_left=window, window_size_right=0, cu_seqlens_q=q_cu,
                                          cu_seqlens_k=k_cu, max_seqlen_q=max_q, max_seqlen_k=max_k)
    return dq, dk, dv


@_backward.register_fake
def _(grad, query, key, value, out, lse, q_cu, k_cu, max_q, max_k, scale, window):
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


def _setup(ctx, inputs, output):
    query, key, value, q_cu, k_cu, ctx.max_q, ctx.max_k, ctx.scale, ctx.window = inputs
    ctx.save_for_backward(query, key, value, *output, q_cu, k_cu)


def _gradient(ctx, grad, _lse_grad):
    query, key, value, out, lse, q_cu, k_cu = ctx.saved_tensors
    dq, dk, dv = _backward(grad.contiguous(), query, key, value, out, lse, q_cu, k_cu, ctx.max_q, ctx.max_k,
                           ctx.scale, ctx.window)
    return dq, dk, dv, None, None, None, None, None, None


_forward.register_autograd(_gradient, setup_context=_setup)


def usable(query: torch.Tensor) -> bool:
    return (_fa4 is not None and query.is_cuda and query.shape[-1] == HEAD
            and query.dtype in (torch.bfloat16, torch.float16)
            and torch.cuda.get_device_capability(query.device)[0] == 10)


def attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_cu: torch.Tensor, k_cu: torch.Tensor,
           max_q: int, max_k: int, scale: float, window: int) -> torch.Tensor:
    """(T, heads, 256) queries over (Tk, kv heads, 256) keys and values (``q_cu`` / ``k_cu``: int32 boundaries),
    each query seeing at most ``window`` keys before its own position; (T, heads, 256)."""
    if usable(query):
        return _forward(query.contiguous(), key.contiguous(), value.contiguous(), q_cu, k_cu, max_q, max_k, scale,
                        window)[0]
    return varlen_attn(query, key, value, q_cu, k_cu, max_q, max_k, scale=scale, window_size=(window, 0),
                       enable_gqa=key.shape[1] != query.shape[1])
