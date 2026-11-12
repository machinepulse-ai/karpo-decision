"""Qwen3.5's pointwise work in a decoder layer, fused by ``torch.compile`` on GPU and eager elsewhere, with
Transformers' math: ``Qwen3_5RMSNorm`` (float32, ``1 + weight``), ``Qwen3_5RMSNormGated`` (normed, cast back, times the
weight, times silu of the gate in float32), the silu-gated MLP, ``apply_rotary_pos_emb`` (partial rotary), the Gated
DeltaNet's decay and write strength, the attention output gate and the residual sums.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    h = x.float()
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())).type_as(x)


def _add_rms_norm(h: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, ...]:
    """``h + x`` and its norm."""
    s = h + x
    return s, _rms_norm(s, weight, eps)


def _gated_norm(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    h = x.float()
    h = weight * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return (h * F.silu(gate.float())).to(x.dtype)


def _silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def _sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(gate)


def _decay(a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The Gated DeltaNet's log decay ``g`` and write strength ``beta``."""
    return -a_log.float().exp() * F.softplus(a.float() + dt_bias), b.sigmoid()


def _rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return apply_rotary_pos_emb(q, k, cos, sin)


def _on_gpu(fn: Callable[..., Any], dims: tuple[int | None, ...]) -> Callable[..., Any]:
    """``fn`` compiled with only the token dimension symbolic: ``dims[i]`` is that dimension of argument ``i`` (None:
    a parameter, every dimension static). ``dynamic=True`` made every dimension symbolic, head sizes included, and the
    kernels 3-5x slower (measured on Gemma's)."""
    compiled = torch.compile(fn)

    def run(*args: Any) -> Any:
        if not args[0].is_cuda:
            return fn(*args)
        for t, dim in zip(args, dims, strict=False):
            if isinstance(t, torch.Tensor):
                for d in range(t.dim()):
                    if d == dim and t.shape[d] > 1:
                        torch._dynamo.mark_dynamic(t, d)
                    else:
                        torch._dynamo.mark_static(t, d)
        return compiled(*args)

    run.__doc__ = fn.__doc__
    return run


# Token dimensions: hidden states and projections (1, T, ...); the Gated DeltaNet's gated norm over (T x heads, d);
# rotary queries and keys (1, heads, T, d) with cos and sin (1, T, d).
rms_norm = _on_gpu(_rms_norm, (1, None, None))
add_rms_norm = _on_gpu(_add_rms_norm, (1, 1, None, None))
gated_norm = _on_gpu(_gated_norm, (0, 0, None, None))
silu_mul = _on_gpu(_silu_mul, (1, 1))
sigmoid_mul = _on_gpu(_sigmoid_mul, (1, 1))
decay = _on_gpu(_decay, (1, 1, None, None))
rope = _on_gpu(_rope, (2, 2, 1, 1))


def norm(module: Any, x: torch.Tensor) -> torch.Tensor:
    """A ``Qwen3_5RMSNorm`` module applied to ``x``."""
    return rms_norm(x, module.weight, module.eps)


def residual_norm(h: torch.Tensor, x: torch.Tensor, module: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """``h + x`` and ``module(h + x)`` for a ``Qwen3_5RMSNorm`` module."""
    return add_rms_norm(h, x, module.weight, module.eps)


def mlp(module: Any, x: torch.Tensor) -> torch.Tensor:
    """A ``Qwen3_5MLP`` (silu) applied to ``x``."""
    return module.down_proj(silu_mul(module.gate_proj(x), module.up_proj(x)))
