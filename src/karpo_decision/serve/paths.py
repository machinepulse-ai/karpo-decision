"""The opt-in serving paths each backbone ``model_type`` runs: ``fp8`` (``serve.fp8``, ``serve.qwen_fp8``), ``graphs``
(CUDA graphs under FP8: ``serve.graphs``, ``serve.qwen_graphs``), ``bf16_graphs`` (CUDA graphs on the bf16 path,
``serve.qwen_graphs``) and ``triton_attention`` (``families.gemma.triton_attention``). Read from the checkpoint's
files, before any weights load."""

from __future__ import annotations

from pathlib import Path

from ..config import DecisionConfig
from ..families import model_type

#: The names of the opt-in paths: ``GRAPHS`` is CUDA graphs under FP8, ``BF16_GRAPHS`` CUDA graphs on the bf16 path
#: (the ``graphs`` option); ``GRAPHS_OPTION`` and ``GRAPHS_PATH`` below are also named "graphs".
FP8, GRAPHS, BF16_GRAPHS, TRITON_ATTENTION = "fp8", "graphs", "bf16_graphs", "triton_attention"
#: The option name a refusal of ``BF16_GRAPHS`` reports.
GRAPHS_OPTION = "graphs"
#: The names of the paths a server runs, by FP8 and graphs.
PLAIN_PATH, GRAPHS_PATH, FP8_PATH, FP8_GRAPHS_PATH = "plain", "graphs", "fp8", "fp8-graphs"

GEMMA_PATHS = frozenset({FP8, GRAPHS, TRITON_ATTENTION})
PATHS: dict[str, frozenset[str]] = {
    "gemma4": GEMMA_PATHS,
    "gemma4_unified": GEMMA_PATHS,
    "qwen3_5": frozenset({FP8, GRAPHS, BF16_GRAPHS}),
}


def backbone_type(checkpoint: str | Path) -> str:
    """The ``model_type`` of the checkpoint's backbone: its ``backbone`` directory when it is not yet trained, else
    its own."""
    return model_type(DecisionConfig.load(checkpoint).backbone or checkpoint)


def require(checkpoint: str | Path, fp8: bool = False, triton_attention: bool = False,
            graphs: bool = False) -> frozenset[str]:
    """The opt-in paths of the checkpoint's backbone; ValueError when ``fp8``, ``triton_attention`` or ``graphs`` is
    asked of a backbone that does not run it."""
    name = backbone_type(checkpoint)
    supported = PATHS.get(name, frozenset())
    for option, path, asked in ((FP8, FP8, fp8), (TRITON_ATTENTION, TRITON_ATTENTION, triton_attention),
                                (GRAPHS_OPTION, BF16_GRAPHS, graphs)):
        if asked and path not in supported:
            raise ValueError(f"{option} is not supported for {name} checkpoints")
    return supported


def path_name(fp8: bool, captured: bool) -> str:
    """The name of the path a server runs: ``fp8`` for FP8 projections, ``captured`` for CUDA graphs."""
    if fp8:
        return FP8_GRAPHS_PATH if captured else FP8_PATH
    return GRAPHS_PATH if captured else PLAIN_PATH
