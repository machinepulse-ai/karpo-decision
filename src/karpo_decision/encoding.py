"""Requests as token sequences: the prompt a template lays out (a joint layout, or a block layout: ``blocks.py``),
tokenized with the checkpoint's processor, and the positions of the readout markers.

``Encoder.plan`` encodes a request as one sequence. When it is longer than ``max_length``, a joint layout splits its
questions in halves until every part fits, and a block layout repeats the whole shared block in each part; every part
keeps the full state and images, so splitting never truncates what a question can read. All parts are encoded before
any forward pass: a single question that does not fit fails the whole request instead of returning partial answers.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pybase64
import torch

from .protocol import IMAGE_DATA_URL, Question, Request, describe, images_of, options, validate
from .templates import BLOCKS, Template

#: Fullwidth brackets that replace ``<`` and ``>`` of special and added tokens found in request text, so text can never
#: yield a marker, an image placeholder or a chat-template token.
_ESCAPED = ("\uff1c", "\uff1e")


class LengthError(ValueError):
    """One question, with the state and images, is longer than ``max_length`` tokens. Nothing is truncated."""


class TextOnlyError(ValueError):
    """The request has images; this checkpoint reads text only."""


class PlacementError(ValueError):
    """The request has inline images (``assets.py``) and this checkpoint's joint layout (Qwen3.5) reads images only
    before the text: the screenshot and ``images``."""


class ImageError(ValueError):
    """An image whose header reads (``image_size``) and whose pixels do not decode: truncated or corrupt data."""


@dataclass(frozen=True)
class Encoded:
    """One tokenized sequence: a request, or a part of its questions. Tensors are 1-D over tokens, on CPU."""

    keys: tuple[str, ...]  # question ids, in readout order
    counts: tuple[int, ...]  # option count of each question
    input_ids: torch.Tensor
    markers: torch.Tensor  # readout positions, one per question
    #: Joint layouts with images: which tokens are image tokens (1) and which text (0).
    mm_token_type_ids: torch.Tensor | None = None
    #: With images, the processor's pixel values of every image: a joint layout's, or Gemma 4's (images, patches, 768)
    #: in a block layout.
    pixel_values: torch.Tensor | None = None
    #: Joint layouts with images: each image's (t, h, w) grid of patches.
    image_grid_thw: torch.Tensor | None = None
    #: Gemma 4: each image's patch positions (n, patches, 2), -1 on padding (``families/gemma/vision.py``).
    image_position_ids: torch.Tensor | None = None
    #: Each token's position in the block layout (``blocks.py``; branches restart after the shared block) and segment
    #: (0 the shared block, k the k-th branch). None: positions 0..n-1, one segment.
    positions: torch.Tensor | None = None
    segment_ids: torch.Tensor | None = None
    #: Gemma 4 block layouts with images: each image in row order (``vision.PlannedImage``), its pixels prepared at
    #: pack time (``Vision.read`` fills ``pixel_values`` and ``image_position_ids``).
    images: tuple[Any, ...] | None = None

    @property
    def length(self) -> int:
        return int(self.input_ids.numel())


def image_bytes(value: Any) -> Any:
    """``value`` as a plan keeps it: a data URL as its bytes (its base64 read once), anything else unchanged."""
    if isinstance(value, str) and value.startswith(IMAGE_DATA_URL):
        return pybase64.b64decode(value.partition(",")[2], validate=True)
    return value


def image_size(value: Any) -> tuple[int, int]:
    """(width, height) of an image, image file bytes, a data URL or a file path, read from the header alone: no pixel
    is decoded. ``OSError`` for what is not an image, ``ValueError`` for a decompression bomb (as ``open_image``)."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.size
    value = image_bytes(value)
    try:
        with Image.open(io.BytesIO(value) if isinstance(value, bytes) else value) as image:
            return image.size
    except Image.DecompressionBombError as error:
        raise ValueError(str(error)) from error


def open_image(value: Any) -> Any:
    """A PIL image from image file bytes, a data URL, a file path or an image; ValueError or OSError when it does not
    decode (PIL's decompression bomb error is neither, so it becomes a ValueError)."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    value = image_bytes(value)
    try:
        with Image.open(io.BytesIO(value) if isinstance(value, bytes) else value) as image:
            return image.convert("RGB")
    except Image.DecompressionBombError as error:
        raise ValueError(str(error)) from error


class Encoder:
    def __init__(self, processor: Any, codes: Sequence[str], template: Template, marker: str) -> None:
        tokenizer = processor.tokenizer
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.convert_ids_to_tokens(ids[0]) != marker:
            raise ValueError(f"the readout marker {marker!r} must be a single existing token")
        self.processor = processor
        self.codes = tuple(codes)
        self.template = template
        self.marker = marker
        self.marker_id: int = ids[0]
        self.pad_id: int = tokenizer.pad_token_id
        # Every added token (``<think>``, ``<tool_call>``, the vision placeholders, ...), not only the named specials.
        self._specials = sorted({*tokenizer.all_special_tokens, *tokenizer.get_added_vocab(), marker}, key=len,
                                reverse=True)
        self.blocks = None
        if template.kind == BLOCKS:
            from .blocks import BlockLayout

            self.blocks = BlockLayout(self, template)

    def escape(self, text: str) -> str:
        for token in self._specials:
            text = text.replace(token, token.replace("<", _ESCAPED[0]).replace(">", _ESCAPED[1]))
        return text

    def text(self, state: Any, questions: Sequence[Question], images: int) -> str:
        messages = self.template.messages(describe(state), questions, self.codes, self.marker, images, self.escape)
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)

    def encode(self, request: Request) -> Encoded:
        """One (validated) request as a single sequence."""
        state, raw_images = images_of(request)
        images = [open_image(value) for value in raw_images]
        questions = list(request["questions"].values())
        text = self.text(state, questions, len(images))
        if images:
            inputs = self.processor(text=[text], images=images, return_tensors="pt")
        else:  # the processor's own result for text alone, without its image bookkeeping
            ids = self.processor.tokenizer(text, return_tensors="pt")["input_ids"]
            inputs = {"input_ids": ids, "mm_token_type_ids": torch.zeros_like(ids)}
        ids = inputs["input_ids"][0]
        markers = (ids == self.marker_id).nonzero().flatten()
        if len(markers) != len(questions):
            raise ValueError("readout markers do not match the questions after tokenization")
        mm = inputs.get("mm_token_type_ids")
        return Encoded(keys=tuple(request["questions"]), counts=tuple(len(options(q)[0]) for q in questions),
                       input_ids=ids, markers=markers, mm_token_type_ids=None if mm is None else mm[0],
                       pixel_values=inputs.get("pixel_values"), image_grid_thw=inputs.get("image_grid_thw"))

    def plan(self, request: Request, max_length: int | None) -> list[Encoded]:
        """The request as one sequence, its questions halved until every part fits ``max_length`` (a block template's
        layout: ``blocks.BlockLayout.plan``)."""
        validate(request, len(self.codes))
        if self.blocks is not None:
            return self.blocks.plan(request, max_length)
        from .assets import places_images

        if places_images(request):
            raise PlacementError("this checkpoint reads images at state.viewport_screenshot and images only")

        def split(items: list[tuple[str, Question]]) -> list[Encoded]:
            encoded = self.encode({**request, "questions": dict(items)})
            if max_length is None or encoded.length <= max_length:
                return [encoded]
            if len(items) == 1:
                raise LengthError(f"question {items[0][0]!r} needs {encoded.length} tokens with the state and images; "
                                  f"the limit is {max_length}")
            middle = len(items) // 2
            return split(items[:middle]) + split(items[middle:])

        return split(list(request["questions"].items()))
