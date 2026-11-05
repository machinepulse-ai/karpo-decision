"""Requests as token sequences: the block layout a template lays out (``blocks.py``), tokenized with the checkpoint's
processor, and the positions of the readout markers.

``Encoder.plan`` encodes a request as one sequence, or as several that each repeat the whole shared block when it is
longer than ``max_length``, so splitting never truncates what a question can read. All parts are encoded before any
forward pass: a single question that does not fit fails the whole request instead of returning partial answers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .protocol import Request, validate
from .templates import BlockTemplate

#: Brackets that replace ``<`` and ``>`` of special and added tokens found in request text, so text can never yield a
#: marker or a chat-template token.
_ESCAPED = ("＜", "＞")


class LengthError(ValueError):
    """One question, with the state, is longer than ``max_length`` tokens. Nothing is truncated."""


class TextOnlyError(ValueError):
    """The request has images; the block layout reads text only."""


@dataclass(frozen=True)
class Encoded:
    """One tokenized sequence: a request, or a part of its questions. Tensors are 1-D over tokens, on CPU."""

    keys: tuple[str, ...]  # question ids, in readout order
    counts: tuple[int, ...]  # option count of each question
    input_ids: torch.Tensor
    markers: torch.Tensor  # readout positions, one per question
    #: Each token's position in the block layout (``blocks.py``; branches restart after the shared block) and segment
    #: (0 the shared block, k the k-th branch). None: positions 0..n-1, one segment.
    positions: torch.Tensor | None = None
    segment_ids: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return int(self.input_ids.numel())


class Encoder:
    def __init__(self, processor: Any, codes: Sequence[str], template: BlockTemplate, marker: str) -> None:
        tokenizer = processor.tokenizer
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.convert_ids_to_tokens(ids[0]) != marker:
            raise ValueError(f"the readout marker {marker!r} must be a single existing token")
        self.processor = processor
        self.codes = tuple(codes)
        self.template = template
        self.marker = marker
        # Every added token (``<think>``, ``<tool_call>``, the vision placeholders, ...), not only the named specials.
        self._specials = sorted({*tokenizer.all_special_tokens, *tokenizer.get_added_vocab(), marker}, key=len,
                                reverse=True)
        from .blocks import BlockLayout

        self.blocks = BlockLayout(self, template)

    def escape(self, text: str) -> str:
        for token in self._specials:
            text = text.replace(token, token.replace("<", _ESCAPED[0]).replace(">", _ESCAPED[1]))
        return text

    def plan(self, request: Request, max_length: int | None) -> list[Encoded]:
        """The request's sequences, every one within ``max_length`` (``blocks.BlockLayout.plan``)."""
        validate(request, len(self.codes))
        return self.blocks.plan(request, max_length)
