"""FP8 (e4m3) weights for a Qwen3.5 decoder's projections, for serving: each weight row scaled to the e4m3 range once
at load, each activation row scaled when it arrives, the product in FP8 with float32 accumulation
(``torch._scaled_mm``, rowwise scales) and a bf16 result. It halves the weights' memory and traffic and doubles the
matrix units' rate on GPUs with FP8 tensor cores (Ada, Hopper, Blackwell); answers move by more than bf16 noise, so a
checkpoint served this way is measured against its bf16 answers first.
"""

from __future__ import annotations

import torch

from ..families.qwen import Packed

E4M3 = torch.float8_e4m3fn
E4M3_MAX = torch.finfo(E4M3).max
#: Projections with fewer outputs (the Gated DeltaNet's per-head gates) stay bf16.
MIN_FEATURES = 256


def to_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of ``x`` scaled into the e4m3 range, and each row's scale (float32, one per row)."""
    scale = x.abs().amax(-1, keepdim=True).float().clamp(min=1e-12) / E4M3_MAX
    return (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3), scale


class Fp8Linear(torch.nn.Module):
    """A bias-free ``nn.Linear`` with an e4m3 weight and one scale per output feature."""

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        if linear.bias is not None:
            raise ValueError("FP8 projections have no bias here")
        weight, scale = to_e4m3(linear.weight.detach())
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.scale = torch.nn.Parameter(scale.t().contiguous(), requires_grad=False)  # (1, out)
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, self.in_features)
        values, scale = to_e4m3(rows)
        out = torch._scaled_mm(values, self.weight.t(), scale_a=scale, scale_b=self.scale, out_dtype=x.dtype)
        return out.reshape(*x.shape[:-1], self.out_features)


def quantize(model: Packed) -> int:
    """Every projection of every decoder layer with at least ``MIN_FEATURES`` outputs to FP8, in place, one layer at a
    time (the bf16 weights are freed as it goes). Returns how many were replaced."""
    replaced = 0
    for layer in model.layers:
        for parent in list(layer.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear) and child.out_features >= MIN_FEATURES:
                    setattr(parent, name, Fp8Linear(child))
                    replaced += 1
        torch.cuda.empty_cache()
    return replaced
