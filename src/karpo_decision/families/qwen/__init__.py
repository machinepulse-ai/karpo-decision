"""Qwen3.5 (``Qwen/Qwen3.5-0.8B`` to ``Qwen/Qwen3.5-9B``): a hybrid backbone (Gated DeltaNet linear-attention layers and
full-attention layers), images through its vision tower, the processor's chat template, no logit cap."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

from ...config import MIN_PIXELS
from .packed import Packed

MODEL_TYPE = "qwen3_5"
MARKER = "<|fim_suffix|>"
TEMPLATES = ("karpo-joint-v1", "karpo-joint-preview-v1")
EXTRA_HINT = "Qwen3.5 checkpoints need the qwen extra: pip install 'karpo-decision[qwen]'"

__all__ = ["MARKER", "MODEL_TYPE", "TEMPLATES", "Packed", "load_backbone", "load_processor", "logit_cap"]


def load_processor(directory: str | Path, max_pixels: int) -> Any:
    # Without torchvision the image processor resizes with another backend than the checkpoints were evaluated with.
    if any(importlib.util.find_spec(name) is None for name in ("torchvision", "PIL")):
        raise ImportError(EXTRA_HINT)
    processor = AutoProcessor.from_pretrained(str(directory))
    processor.tokenizer.padding_side = "left"
    processor.image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": max_pixels}
    return processor


def load_backbone(weights: str | Path, dtype: torch.dtype) -> Qwen3_5Model:
    return Qwen3_5Model.from_pretrained(str(weights), dtype=dtype, attn_implementation="sdpa")


def logit_cap(backbone: Any) -> None:
    return None
