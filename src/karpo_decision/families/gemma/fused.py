"""Gemma 4's pointwise work in a decoder layer, fused by ``torch.compile`` on GPU and eager elsewhere: the same math as
Transformers' ``Gemma4RMSNorm`` (in float32, a direct weight, no weight for the value norm), the ``gelu_pytorch_tanh``
gate and ``apply_rotary_pos_emb``, plus the residual sums and the layer scalar, each fused
into one kernel that reads and writes its tensors once.

Each is compiled for its tensors' static shapes but the sequence (dimension 1: a pack's rows, or its markers), which
varies from pack to pack. Compiled with every dimension symbolic, the kernels index with symbolic head counts and
sizes and run several times slower.
A query's or key's norm and rotation are one kernel (``norm_rope``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F


def _rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    h = x.float()
    out = h * torch.pow(h.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        out = out * weight.float()
    return out.type_as(x)


def _add_norm(h: torch.Tensor, x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    return h + _rms_norm(x, weight, eps)


def _add_scale(h: torch.Tensor, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (h + x) * scale.to(h.dtype)


def _gelu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.gelu(gate, approximate="tanh") * up


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=2)``."""
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


def _norm_rope(x: torch.Tensor, weight: torch.Tensor | None, eps: float, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    return _rope(_rms_norm(x, weight, eps), cos, sin)


def _shapes(*tensors: Any) -> None:
    """Dimension 1 (the sequence) of every tensor of more than one dimension dynamic, every other dimension static."""
    for t in tensors:
        if isinstance(t, torch.Tensor):
            for d in range(t.dim()):
                if d == 1 and t.shape[1] > 1:
                    torch._dynamo.mark_dynamic(t, 1)
                else:
                    torch._dynamo.mark_static(t, d)


def _on_gpu[T](fn: Callable[..., T], **options: Any) -> Callable[..., T]:
    compiled = torch.compile(fn, options=options or None)

    def run(x: torch.Tensor, *args: Any) -> T:
        if not x.is_cuda:
            return fn(x, *args)
        _shapes(x, *args)
        return compiled(x, *args)

    run.__doc__ = fn.__doc__
    return run


rms_norm, add_norm, add_scale, gelu_mul, rope, _fused_norm_rope = map(
    _on_gpu, (_rms_norm, _add_norm, _add_scale, _gelu_mul, _rope, _norm_rope))


def norm(module: Any, x: torch.Tensor) -> torch.Tensor:
    """A ``Gemma4RMSNorm`` module applied to ``x``."""
    return rms_norm(x, module.weight if module.with_scale else None, module.eps)


def norm_rope(module: Any, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``rope(norm(module, x), cos, sin)`` in one kernel (a query's or key's norm and rotation)."""
    return _fused_norm_rope(x, module.weight if module.with_scale else None, module.eps, cos, sin)


def residual_norm(h: torch.Tensor, module: Any, x: torch.Tensor) -> torch.Tensor:
    """``h + module(x)`` for a ``Gemma4RMSNorm`` module."""
    return add_norm(h, x, module.weight if module.with_scale else None, module.eps)


def mlp(module: Any, x: torch.Tensor) -> torch.Tensor:
    """A ``Gemma4TextMLP`` (gelu_pytorch_tanh) applied to ``x``."""
    return module.down_proj(gelu_mul(module.gate_proj(x), module.up_proj(x)))
