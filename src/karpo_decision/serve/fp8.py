"""FP8 (e4m3) projections for serving: each weight row scaled into the e4m3 range once at load, each activation row
scaled as it arrives, products by ``torch._scaled_mm`` (rowwise scales, float32 accumulation, bf16 out). Faster on
GPUs with FP8 tensor cores; answers move by more than bf16 noise."""

from __future__ import annotations

import torch

E4M3 = torch.float8_e4m3fn
E4M3_MAX = torch.finfo(E4M3).max
#: A decoder layer's projections (a global layer has no ``v_proj``: keys and values share one).
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _to_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of ``x`` scaled into the e4m3 range, and each row's float32 scale."""
    scale = x.abs().amax(-1, keepdim=True).float().clamp(min=1e-12) / E4M3_MAX
    return (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3), scale


_compiled = torch.compile(_to_e4m3)


def to_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``_to_e4m3`` in one fused kernel on GPU, compiled for any number of rows."""
    if not x.is_cuda:
        return _to_e4m3(x)
    if x.shape[0] > 1:
        torch._dynamo.mark_dynamic(x, 0)
    return _compiled(x)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, scale = to_e4m3(x.reshape(-1, self.in_features))
        out = torch._scaled_mm(values, self.weight.t(), scale_a=scale, scale_b=self.scale, out_dtype=x.dtype)
        return out.reshape(*x.shape[:-1], self.out_features)


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
    return replaced
