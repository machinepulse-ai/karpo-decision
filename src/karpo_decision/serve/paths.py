"""The opt-in serving paths each backbone ``model_type`` runs: ``fp8`` (``serve.fp8``), ``graphs`` (CUDA graphs under
FP8, ``serve.graphs``) and ``triton_attention`` (``families.gemma.triton_attention``). Read from the checkpoint's
files, before any weights load."""

from __future__ import annotations

from pathlib import Path

from ..config import DecisionConfig
from ..families import model_type

#: The names of the opt-in paths.
FP8, GRAPHS, TRITON_ATTENTION = "fp8", "graphs", "triton_attention"

PATHS: dict[str, frozenset[str]] = {
    "gemma4": frozenset({FP8, GRAPHS, TRITON_ATTENTION}),
    "gemma4_unified": frozenset(),
    "qwen3_5": frozenset(),
}


def require(checkpoint: str | Path, fp8: bool = False, triton_attention: bool = False) -> frozenset[str]:
    """The opt-in paths of the checkpoint's backbone; ValueError when ``fp8`` or ``triton_attention`` is asked of a
    backbone that does not run it."""
    backbone = DecisionConfig.load(checkpoint).backbone
    name = model_type(backbone or checkpoint)
    supported = PATHS.get(name, frozenset())
    for option, asked in ((FP8, fp8), (TRITON_ATTENTION, triton_attention)):
        if asked and option not in supported:
            raise ValueError(f"{option} is not supported for {name} checkpoints")
    return supported
