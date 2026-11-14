"""Images for a Gemma 4 checkpoint with a vision tower (the 31B): each image as Gemma 4's image processor prepares it
(at most ``MAX_SOFT_TOKENS`` soft tokens) and the soft-token embeddings the backbone's pretrained vision tower makes of
it. A plan holds each image as it came (a data URL as its bytes) with its soft tokens, counted from its size alone, read
from its header (``Vision.measure``); its pixels are decoded once, only when its sequences are packed
(``Vision.read``), and dropped after the pass. ``blocks.py`` lays
the soft tokens out; ``packed.py`` writes the embeddings at their rows and lets each image's soft tokens see their
whole image in the sliding layers (Gemma 4 reads an image bidirectionally)."""

from __future__ import annotations

import bisect
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ...encoding import Encoded, ImageError, image_bytes, image_size, open_image

#: Soft tokens per image at most (Gemma 4's processor takes 70, 140, 280, 560 or 1120): 5,040 patches of 16 x 16
#: pixels pooled 3 x 3, about 1.29 MP.
MAX_SOFT_TOKENS = 560
PADDING = -1  # position id of a padding patch


@dataclass(frozen=True)
class ImageTokens:
    """The checkpoint's image placeholders: ``begin``, then ``soft`` once per soft token, then ``end``."""

    begin: int
    soft: int
    end: int


@dataclass(frozen=True)
class PlannedImage:
    """An image as a plan holds it: what ``encoding.open_image`` reads, and its soft tokens (``Vision.measure``)."""

    value: Any
    tokens: int


@dataclass(frozen=True)
class Images:
    """A request's images as the vision tower reads them, on CPU."""

    pixel_values: torch.Tensor  # (n, patches, 3 x 16 x 16) float32, padded
    position_ids: torch.Tensor  # (n, patches, 2), -1 on padding
    tokens: tuple[int, ...]  # soft tokens per image


def decode(value: Any) -> Any:
    """``value``'s pixels as a PIL image (``encoding.open_image``); ``ImageError`` when they do not decode."""
    try:
        return open_image(value)
    except (ValueError, OSError) as error:
        raise ImageError(f"an image does not decode: {error}") from error


class Vision:
    """Image preparation for a checkpoint whose backbone has a vision tower."""

    def __init__(self, tokens: ImageTokens, max_soft_tokens: int = MAX_SOFT_TOKENS) -> None:
        from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor

        self.tokens = tokens
        self.processor = Gemma4ImageProcessor(max_soft_tokens=max_soft_tokens)

    def _valid(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids != PADDING).all(-1).sum(-1)

    def prepare(self, values: Sequence[Any]) -> Images:
        """``values``: PIL images, image file bytes, data URLs or file paths, each decoded once (``decode``)."""
        out = self.processor(images=[decode(value) for value in values], return_tensors="pt")
        pooled = self._valid(out["image_position_ids"]) // self.processor.pooling_kernel_size ** 2
        return Images(out["pixel_values"], out["image_position_ids"], tuple(int(n) for n in pooled))

    def soft_tokens(self, height: int, width: int) -> int:
        """Soft tokens of an image of ``height`` x ``width`` pixels: the processor's own resize target (Transformers'
        ``get_aspect_ratio_preserving_size``) in patches, pooled."""
        from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size

        patch, pool = self.processor.patch_size, self.processor.pooling_kernel_size
        h, w = get_aspect_ratio_preserving_size(height=height, width=width, patch_size=patch,
                                                max_patches=self.processor.max_soft_tokens * pool ** 2,
                                                pooling_kernel_size=pool)
        return (h // patch) * (w // patch) // pool ** 2

    def measure(self, value: Any) -> PlannedImage:
        """``value``'s soft tokens from its size, read from its header alone (``encoding.image_size``): what is not an
        image, or a decompression bomb, fails here, before any pass; its pixels are decoded once, by ``prepare``. A data
        URL is kept as its bytes."""
        value = image_bytes(value)
        width, height = image_size(value)
        return PlannedImage(value, self.soft_tokens(height, width))

    def read(self, sequences: Sequence[Encoded]) -> list[Encoded]:
        """``sequences`` with the pixels of their planned images (``Encoded.images``): each image prepared on its own
        (``prepare``) and laid out in row order, a value several images share prepared once; a sequence that has its
        pixels already (``Decider.prepare``) is kept as it is."""
        prepared: dict[int, Images] = {}
        out = []
        for s in sequences:
            if not s.images or s.pixel_values is not None:
                out.append(s)
                continue
            parts = []
            for planned in s.images:
                key = id(planned.value)
                if key not in prepared:
                    prepared[key] = self.prepare([planned.value])
                if prepared[key].tokens[0] != planned.tokens:
                    raise ValueError(f"an image planned at {planned.tokens} soft tokens has "
                                     f"{prepared[key].tokens[0]}")
                parts.append(prepared[key])
            images = self.join(parts)
            out.append(dataclasses.replace(s, pixel_values=images.pixel_values,
                                           image_position_ids=images.position_ids))
        return out

    @staticmethod
    def join(parts: Sequence[Images]) -> Images:
        """Images prepared apart as one, in order (the processor pads every image to the same patch count)."""
        return Images(torch.cat([p.pixel_values for p in parts]), torch.cat([p.position_ids for p in parts]),
                      tuple(n for p in parts for n in p.tokens))

    def summary(self, images: Sequence[PlannedImage]) -> tuple[list[int], int]:
        """Each planned image's pixels as the vision tower reads them, and the soft tokens of them all."""
        per_token = self.processor.pooling_kernel_size ** 2 * self.processor.patch_size ** 2
        return [planned.tokens * per_token for planned in images], sum(planned.tokens for planned in images)


def embed(backbone: Any, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """(soft tokens, hidden): every image's soft-token embeddings in order, from the backbone's vision tower and
    embedder (not scaled like the token embeddings, as Gemma 4 does it)."""
    param = next(backbone.vision_tower.parameters())
    features = backbone.get_image_features(pixel_values.to(param.device, param.dtype),
                                           position_ids.to(param.device), return_dict=True).pooler_output
    return torch.cat(list(features))


@dataclass
class ImagePlan:
    """Each image's soft-token rows in a pack, for the sliding layers: queries ``[s, e)``, the rows they may see as keys
    (the window before the image within what its block reads, and the image itself) and which of those each query
    sees, on the device."""

    rows: torch.Tensor  # every soft token's row, in pack order
    spans: list[tuple[int, int]]  # (s, e) per image
    keys: list[torch.Tensor]  # key rows per image
    masks: list[torch.Tensor]  # (e - s, keys) bool per image


def plan_images(input_ids: torch.Tensor, bounds: list[int], segments: torch.Tensor, positions: torch.Tensor,
                soft: int, window: int, device: torch.device) -> ImagePlan:
    """``input_ids``, ``segments`` and ``positions`` on CPU, ``bounds`` the pack's sequence boundaries, ``window`` the
    sliding layers' (a token and the window - 1 before it). An image's soft tokens are one run inside one block (the
    layout puts ``begin`` and ``end`` around them). An image in a sequence's shared block (or its only block) sees the
    rows before it within the window; one in a branch sees the shared block's rows within the window by position and its
    branch's rows before it, as the branch's other tokens do (``blocks.py``), and its whole self in either case."""
    rows = (input_ids == soft).nonzero().flatten()
    if rows.numel() == 0:
        raise ValueError("a pack with images has no image soft tokens")
    breaks = ((rows[1:] != rows[:-1] + 1).nonzero().flatten() + 1).tolist()
    spans, keys, masks = [], [], []
    for a, b in zip([0, *breaks], [*breaks, len(rows)], strict=True):
        s, e = int(rows[a]), int(rows[b - 1]) + 1
        if e - s > window:
            raise ValueError(f"an image of {e - s} soft tokens is longer than the sliding window ({window})")
        start = bounds[bisect.bisect_right(bounds, s) - 1]
        if int(segments[s]) == 0:
            seen = torch.arange(max(start, s - (window - 1)), e)
        else:
            shared = int((segments[start:s] == 0).sum())  # the shared block's length: positions restart after it
            branch = s - (int(positions[s]) - shared)  # the branch's first row
            first = min(max(0, int(positions[s]) - (window - 1)), shared)  # the first shared position it sees
            seen = torch.cat([torch.arange(start + first, start + shared), torch.arange(branch, e)])
        masks.append((positions[seen][None, :] >= positions[s:e][:, None] - (window - 1)).to(device))
        keys.append(seen.to(device))
        spans.append((s, e))
    return ImagePlan(rows.to(device), spans, keys, masks)


def see_whole(out: torch.Tensor, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, plan: ImagePlan,
              scale: float) -> torch.Tensor:
    """A sliding layer's ``out`` (rows, heads, head size) with each image's soft-token rows recomputed: such a row sees
    the window before it, as every token does, and its whole image (``query``, ``key``, ``value``: every row of the
    pack, keys not gathered)."""
    parts = []
    for (s, e), rows, mask in zip(plan.spans, plan.keys, plan.masks, strict=True):
        q = query[s:e].transpose(0, 1)[None]
        kk, vv = (t.index_select(0, rows).transpose(0, 1)[None] for t in (key, value))
        seen = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, scale=scale,
                                              enable_gqa=kk.shape[1] != q.shape[1])
        parts.append(seen[0].transpose(0, 1))
    return out.index_copy(0, plan.rows, torch.cat(parts).to(out.dtype))
