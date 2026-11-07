"""FP8 (e4m3) projections for serving: each weight row scaled into the e4m3 range once at load, each activation row
scaled as it arrives, products by ``torch._scaled_mm`` (rowwise scales, float32 accumulation, bf16 out). Faster on
GPUs with FP8 tensor cores; answers move by more than bf16 noise. An activation read by several projections is
quantized once, in the fused op that produces it (``norm``, ``mlp``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..families.gemma import fused

E4M3 = torch.float8_e4m3fn
E4M3_MAX = torch.finfo(E4M3).max
#: A decoder layer's projections (a global layer has no ``v_proj``: keys and values share one).
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _to_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of ``x`` scaled into the e4m3 range, and each row's float32 scale."""
    scale = x.abs().amax(-1, keepdim=True).float().clamp(min=1e-12) / E4M3_MAX
    return (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3), scale


_compiled = torch.compile(_to_e4m3, dynamic=True)


def to_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``_to_e4m3`` in one fused kernel on GPU."""
    return _compiled(x) if x.is_cuda else _to_e4m3(x)


@dataclass
class Quantized:
    """An activation's rows in e4m3 with one float32 scale each, quantized once for every projection that reads it."""

    values: torch.Tensor  # (N, K) e4m3
    scale: torch.Tensor  # (N, 1) float32
    shape: torch.Size  # the activation's shape (..., K), N = its rows
    dtype: torch.dtype

    @staticmethod
    def of(x: torch.Tensor) -> Quantized:
        values, scale = to_e4m3(x.reshape(-1, x.shape[-1]))
        return Quantized(values, scale, x.shape, x.dtype)

    def __getitem__(self, index: tuple) -> Quantized:
        """``q[:, rows]`` of a (1, T, K) activation: those rows' values and scales."""
        _, rows = index
        return Quantized(self.values[rows], self.scale[rows], torch.Size((1, rows.numel(), self.shape[-1])), self.dtype)


def _norm_e4m3(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    y = fused._rms_norm(x, weight, eps)
    return _to_e4m3(y.reshape(-1, y.shape[-1]))


def _gelu_mul_e4m3(gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y = fused._gelu_mul(gate, up)
    return _to_e4m3(y.reshape(-1, y.shape[-1]))


#: Eager's bf16 rounding of the op output is kept before the scale, as when the projection quantized it.
_norm_compiled = torch.compile(_norm_e4m3, dynamic=True, options={"emulate_precision_casts": True})
_gelu_mul_compiled = torch.compile(_gelu_mul_e4m3, dynamic=True, options={"emulate_precision_casts": True})


def norm(module: Any, x: torch.Tensor) -> Quantized:
    """A ``Gemma4RMSNorm`` applied to ``x`` and its rows quantized, in one kernel."""
    weight = module.weight if module.with_scale else None
    values, scale = (_norm_compiled if x.is_cuda else _norm_e4m3)(x, weight, module.eps)
    return Quantized(values, scale, x.shape, x.dtype)


def mlp(module: Any, x: Quantized) -> torch.Tensor:
    """A ``Gemma4TextMLP`` over a quantized input; ``down_proj``'s input is quantized as the gate is applied."""
    gate, up = module.gate_proj(x), module.up_proj(x)
    values, scale = (_gelu_mul_compiled if gate.is_cuda else _gelu_mul_e4m3)(gate, up)
    return module.down_proj(Quantized(values, scale, gate.shape, gate.dtype))


class Fp8Linear(torch.nn.Module):
    """A bias-free ``nn.Linear`` with an e4m3 weight and one scale per output feature."""

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        if linear.bias is not None:
            raise ValueError("FP8 projections have no bias")
        weight, scale = _to_e4m3(linear.weight.detach())
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale.t().contiguous())  # (1, out)
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x: torch.Tensor | Quantized) -> torch.Tensor:
        q = x if isinstance(x, Quantized) else Quantized.of(x)
        out = torch._scaled_mm(q.values, self.weight.t(), scale_a=q.scale, scale_b=self.scale, out_dtype=q.dtype)
        return out.reshape(*q.shape[:-1], self.out_features)


def quantize(model: torch.nn.Module) -> int:
    """Every decoder layer's projections to ``Fp8Linear``, in place, a layer at a time (each layer's bf16 weights freed
    as it goes). Returns how many were replaced."""
    replaced = 0
    for layer in model.layers:
        for parent in list(layer.modules()):
            for name, child in list(parent.named_children()):
                if name in PROJECTIONS and isinstance(child, torch.nn.Linear):
                    setattr(parent, name, Fp8Linear(child))
                    replaced += 1
        torch.cuda.empty_cache()
    model.norm_input, model.mlp = norm, mlp  # each activation quantized once, inside the fused op that makes it
    return replaced
