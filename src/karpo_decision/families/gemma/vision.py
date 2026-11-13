"""Images for a Gemma 4 checkpoint with a vision tower (the 31B): each image as Gemma 4's image processor prepares it
(at most ``MAX_SOFT_TOKENS`` soft tokens) and the soft-token embeddings the backbone's pretrained vision tower makes of
it. ``blocks.py`` lays the soft tokens out in the shared block; ``packed.py`` writes the embeddings at their rows and
lets each image's soft tokens see their whole image in the sliding layers (Gemma 4 reads an image bidirectionally)."""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

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


@dataclass
class ImagePlan:
    """Each image's soft-token rows in a pack, for the sliding layers: queries ``[s, e)``, keys ``[k, e)`` (the window
    before the image, within its sequence, and the image) and which keys each query sees, on the device."""

    rows: torch.Tensor  # every soft token's row, in pack order
    spans: list[tuple[int, int, int]]  # (s, e, k) per image
    masks: list[torch.Tensor]  # (e - s, e - k) bool per image


def plan_images(input_ids: torch.Tensor, bounds: list[int], soft: int, window: int,
                device: torch.device) -> ImagePlan:
    """``input_ids`` on CPU; ``bounds`` the pack's sequence boundaries; ``window`` the sliding layers' (a token and the
    window - 1 before it). An image's soft tokens are one run (the layout puts ``begin`` and ``end`` around them)."""
    rows = (input_ids == soft).nonzero().flatten()
    breaks = ((rows[1:] != rows[:-1] + 1).nonzero().flatten() + 1).tolist()
    spans, masks = [], []
    for a, b in zip([0, *breaks], [*breaks, len(rows)], strict=True):
        s, e = int(rows[a]), int(rows[b - 1]) + 1
        if e - s > window:
            raise ValueError(f"an image of {e - s} soft tokens is longer than the sliding window ({window})")
        k = max(bounds[bisect.bisect_right(bounds, s) - 1], s - (window - 1))
        masks.append((torch.arange(k, e)[None, :] >= torch.arange(s, e)[:, None] - (window - 1)).to(device))
        spans.append((s, e, k))
    return ImagePlan(rows.to(device), spans, masks)


def see_whole(out: torch.Tensor, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, plan: ImagePlan,
              scale: float) -> torch.Tensor:
    """A sliding layer's ``out`` (rows, heads, head size) with each image's soft-token rows recomputed: such a row sees
    the window before it, as every token does, and its whole image (``query``, ``key``, ``value``: every row of the
    pack, keys not gathered)."""
    parts = []
    for (s, e, k), mask in zip(plan.spans, plan.masks, strict=True):
        q = query[s:e].transpose(0, 1)[None]
        kk, vv = key[k:e].transpose(0, 1)[None], value[k:e].transpose(0, 1)[None]
        seen = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, scale=scale,
                                              enable_gqa=kk.shape[1] != q.shape[1])
        parts.append(seen[0].transpose(0, 1))
    return out.index_copy(0, plan.rows, torch.cat(parts).to(out.dtype))
