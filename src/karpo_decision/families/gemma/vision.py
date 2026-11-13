"""Images for a Gemma 4 checkpoint with a vision tower (the 31B): each image as Gemma 4's image processor prepares it
(at most ``MAX_SOFT_TOKENS`` soft tokens) and the soft-token embeddings the backbone's pretrained vision tower makes of
it. ``blocks.py`` lays the soft tokens out in the shared block; ``packed.py`` writes the embeddings at their rows and
lets each image's soft tokens see their whole image in the sliding layers (Gemma 4 reads an image bidirectionally)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ...encoding import open_image

#: Soft tokens per image at most (Gemma 4's processor takes 70, 140, 280, 560 or 1120): 5,040 patches of 16 x 16
#: pixels pooled 3 x 3, about 1.29 MP.
MAX_SOFT_TOKENS = 560


@dataclass(frozen=True)
class ImageTokens:
    """The checkpoint's image placeholders: ``begin``, then ``soft`` once per soft token, then ``end``."""

    begin: int
    soft: int
    end: int


@dataclass(frozen=True)
class Images:
    """A request's images as the vision tower reads them, on CPU."""

    pixel_values: torch.Tensor  # (n, patches, 3 x 16 x 16) float32, padded
    position_ids: torch.Tensor  # (n, patches, 2), -1 on padding
    tokens: tuple[int, ...]  # soft tokens per image


class Vision:
    """Image preparation for a checkpoint whose backbone has a vision tower."""

    def __init__(self, tokens: ImageTokens, max_soft_tokens: int = MAX_SOFT_TOKENS) -> None:
        from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil

        self.tokens = tokens
        self.processor = Gemma4ImageProcessorPil(max_soft_tokens=max_soft_tokens)

    def _valid(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids != -1).all(-1).sum(-1)

    def prepare(self, values: Sequence[Any]) -> Images:
        """``values``: PIL images, data URLs or file paths (``encoding.open_image``)."""
        out = self.processor(images=[open_image(value) for value in values], return_tensors="pt")
        pooled = self._valid(out["image_position_ids"]) // self.processor.pooling_kernel_size ** 2
        return Images(out["pixel_values"], out["image_position_ids"], tuple(int(n) for n in pooled))

    def summary(self, position_ids: torch.Tensor) -> tuple[list[int], int]:
        """Each image's pixels as the vision tower reads them, and the soft tokens of them all."""
        valid = self._valid(position_ids)
        pixels = (valid * self.processor.patch_size ** 2).tolist()
        return pixels, int((valid // self.processor.pooling_kernel_size ** 2).sum())


def embed(backbone: Any, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """(soft tokens, hidden): every image's soft-token embeddings in order, from the backbone's vision tower and
    embedder (not scaled like the token embeddings, as Gemma 4 does it)."""
    param = next(backbone.vision_tower.parameters())
    features = backbone.get_image_features(pixel_values.to(param.device, param.dtype),
                                           position_ids.to(param.device), return_dict=True).pooler_output
    return torch.cat(list(features))
