"""Requests laid out in blocks (``templates.BlockTemplate``): a shared block and one branch per question, as token ids
with each token's position and segment, for a family whose attention keeps branches apart (Gemma).

- A request's legacy images (``protocol.images_of``: the screenshot and ``images``) sit in the shared block, after the
  context header and before the state text; its inline images (``assets.resolve``: asset references and image parts)
  sit where they occur, in the state text, or in a question's instructions or criteria inside that question's branch,
  where the shared block's list of questions shows ``IMAGE_PLACEHOLDER`` (``families/gemma/vision.py``). A checkpoint
  without a vision tower refuses images.
- A request with one question is one block (segment 0, positions 0..n-1): its state reads its question and options.
- With several questions, the shared block (segment 0: system turn, the list of every question's instructions, the
  state) comes once, then each question's branch (segment k for the k-th branch of the sequence) with positions
  restarting where the shared block ends. A branch reads the shared block and itself, the shared block only itself,
  no branch another (the family's attention, ``families/gemma/packed.py``). Which layout follows the request's own
  question count, never how many branches a sequence holds.
- A request too long for ``max_length`` is split into sequences that each repeat the whole shared block (its list
  still names every question) and hold the branches that fit; a single branch that does not fit with the shared block
  fails the request (nothing is truncated).

Pieces are tokenized on their own and concatenated as ids, so the label id an option shows is the id the answer slot
reads; the labels are checked to be one token after ``Answer: (`` and after ``\\n(``, the same token as the code alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .protocol import Question, Request, describe, options
from .templates import DEFAULT_INSTRUCTIONS, BlockTemplate

#: The line before each legacy image when a request has several (k from 1).
IMAGE_LABEL = "Image {}:\n"
#: What the shared block's list of questions shows for an image in a question's instructions.
IMAGE_PLACEHOLDER = "[image]"
#: The fewest ids an image takes (``begin``, one soft token, ``end``): with ``max_length`` it bounds the images the
#: state or one question may place before any is read.
MIN_IMAGE_IDS = 3


class Reader:
    """One request's inline images as its layout places them (``assets.Resolved``): each value read once
    (``vision.measure``), the images of the block being laid out kept in row order, and the request failing
    (``LengthError``) as soon as one block's inline images alone pass ``max_length``, before the next is read."""

    def __init__(self, layout: BlockLayout, vision: Any, resolved: Any, max_length: int | None) -> None:
        self.layout, self.vision, self.resolved, self.max_length = layout, vision, resolved, max_length
        self.read: dict[int, Any] = {}
        self.placed: list[Any] = []
        self.need = 0

    def image(self, k: int) -> list[int]:
        """Inline image ``k``'s ids, placed in the current block."""
        from .encoding import LengthError

        value = self.resolved.inline[k]
        if id(value) not in self.read:
            self.read[id(value)] = self.vision.measure(value)
        planned = self.read[id(value)]
        self.placed.append(planned)
        ids = self.layout.image(planned.tokens)
        self.need += len(ids)
        if self.max_length and self.need > self.max_length:
            raise LengthError(f"the request's images need {self.need} tokens; the limit is {self.max_length}")
        return ids

    def block(self) -> list[Any]:
        """The images placed since the last call, in row order; the next block starts."""
        placed, self.placed, self.need = self.placed, [], 0
        return placed


class BlockLayout:
    def __init__(self, encoder: Any, template: BlockTemplate) -> None:
        self.encoder = encoder
        self.template = template
        tokenizer = encoder.processor.tokenizer
        self.tokenize = lambda text: tokenizer.encode(text, add_special_tokens=False)
        before, after = template.chat(encoder.processor)
        self.prefix = self.tokenize(before)
        self.tail = self.tokenize(template.reply + after + template.slot)
        self.header, self.item, self.newline = (self.tokenize(t) for t in (template.header, template.item, "\n"))
        self.context, self.question, self.options = (self.tokenize(t) for t in
                                                     (template.context, template.question, template.options))
        self.open, self.close = self.tokenize("("), self.tokenize(") ")
        slot, opener = self.tokenize(template.slot), self.tokenize("\n(")
        if self.tail[-len(slot):] != slot or self.tokenize(after + template.slot)[-len(slot):] != slot:
            raise ValueError("the assistant opening merges with the answer slot")
        if len(self.open) != 1 or opener[-1:] != self.open:
            raise ValueError("'(' after a newline must be one token, the same as '(' alone")
        self.labels = []
        for code in encoder.codes:
            alone = self.tokenize(code)
            if (len(alone) != 1 or self.tokenize(template.slot + code) != [*slot, alone[0]]
                    or self.tokenize("\n(" + code) != [*opener, alone[0]]):
                raise ValueError(f"code {code!r} is not one token after {template.slot!r} and after '\\n('")
            self.labels.append(alone[0])

    def text(self, value: Any, reader: Reader | None = None) -> list[int]:
        """``describe(value)`` escaped and tokenized; with a ``reader``, each inline image (a sentinel) as its ids and
        the text between tokenized piece by piece."""
        if reader is None:
            return self.tokenize(self.encoder.escape(describe(value)))
        ids: list[int] = []
        for piece in reader.resolved.pieces(describe(value)):
            ids += reader.image(piece) if isinstance(piece, int) else self.tokenize(self.encoder.escape(piece))
        return ids

    def listed(self, question: Question, reader: Reader | None = None) -> list[int]:
        """A question's instructions as the shared block lists them: each inline image as ``IMAGE_PLACEHOLDER``."""
        if reader is None:
            return self.instructions(question)
        text = describe(question.get("instructions") or DEFAULT_INSTRUCTIONS)
        text = "".join(IMAGE_PLACEHOLDER if isinstance(p, int) else p for p in reader.resolved.pieces(text))
        return self.tokenize(self.encoder.escape(text))

    def instructions(self, question: Question) -> list[int]:
        return self.text(question.get("instructions") or DEFAULT_INSTRUCTIONS)

    def image(self, tokens: int) -> list[int]:
        """One image: ``begin``, ``soft`` x its soft tokens, ``end``."""
        marks = self.encoder.processor.vision.tokens
        return [marks.begin, *[marks.soft] * tokens, marks.end]

    def image_ids(self, tokens: Sequence[int]) -> list[int]:
        """The images' part of the shared block (``tokens``: each image's soft tokens): each image and a newline, after
        an ``IMAGE_LABEL`` line when there are several."""
        ids: list[int] = []
        for k, n in enumerate(tokens, start=1):
            if len(tokens) > 1:
                ids += self.tokenize(IMAGE_LABEL.format(k))
            ids += self.image(n) + self.newline
        return ids

    def measure_images(self, vision: Any, values: list[Any], max_length: int | None) -> list[Any]:
        """``values`` read one at a time (``vision.measure``): the processor scales even a tiny image up to its
        budget, so the request fails (``LengthError``) at the first image that takes the images' part of the shared
        block past ``max_length``, before the next is read."""
        from .encoding import LengthError

        planned, need = [], 0
        for k, value in enumerate(values, start=1):
            planned.append(vision.measure(value))
            need += len(self.image(planned[-1].tokens)) + len(self.newline)
            if len(values) > 1:
                need += len(self.tokenize(IMAGE_LABEL.format(k)))
            if max_length and need > max_length:
                raise LengthError(f"the request's images need {need} tokens; the limit is {max_length}")
        return planned

    def shared(self, state: Any, questions: Sequence[Question], images: Sequence[int] = (),
               reader: Reader | None = None) -> list[int]:
        """``images``: the legacy images' soft tokens; ``reader``: the request's inline images, if any."""
        ids = self.prefix + self.header
        for question in questions:
            ids += self.item + self.listed(question, reader) + self.newline
        return ids + self.context + self.image_ids(images) + self.text(state, reader)

    def branch(self, question: Question, reader: Reader | None = None) -> list[int]:
        _, texts = options(question)
        ids = self.question + self.text(question.get("instructions") or DEFAULT_INSTRUCTIONS, reader) + self.options
        for label, text in zip(self.labels, texts, strict=False):
            ids += self.open + [label] + self.close + self.text(text, reader) + self.newline
        return ids + self.tail

    def plan(self, request: Request, max_length: int | None) -> list[Any]:
        from .assets import resolve
        from .encoding import Encoded, LengthError, TextOnlyError

        resolved = resolve(request, max_length // MIN_IMAGE_IDS if max_length else None)
        vision = getattr(self.encoder.processor, "vision", None)
        if vision is None and (resolved.legacy or resolved.inline):
            raise TextOnlyError("this checkpoint reads text only")
        legacy = self.measure_images(vision, resolved.legacy, max_length) if resolved.legacy else []
        reader = Reader(self, vision, resolved, max_length) if resolved.inline else None
        items = list(resolved.questions.items())
        shared = self.shared(resolved.state, [q for _, q in items], [planned.tokens for planned in legacy], reader)
        images = legacy + (reader.block() if reader else [])
        branches = []
        for key, q in items:
            ids = self.branch(q, reader)
            branches.append((key, len(options(q)[0]), ids, reader.block() if reader else []))
        limit = max_length or float("inf")

        def sequence(ids: list[int], positions: list[int], segments: list[int], markers: list[int],
                     parts: list[tuple[str, int, list[int], list[Any]]]) -> Encoded:
            placed = images + [planned for part in parts for planned in part[3]]
            return Encoded(keys=tuple(part[0] for part in parts), counts=tuple(part[1] for part in parts),
                           input_ids=torch.tensor(ids), markers=torch.tensor(markers),
                           positions=torch.tensor(positions), segment_ids=torch.tensor(segments, dtype=torch.int16),
                           images=tuple(placed) or None)

        if len(items) == 1:
            ids = shared + branches[0][2]
            if len(ids) > limit:
                raise LengthError(f"question {items[0][0]!r} needs {len(ids)} tokens; the limit is {max_length}")
            return [sequence(ids, list(range(len(ids))), [0] * len(ids), [len(ids) - 1], branches)]
        n = len(shared)
        out, current = [], None
        for part in branches:
            ids = part[2]
            if n + len(ids) > limit:
                raise LengthError(f"question {part[0]!r} needs {n + len(ids)} tokens with the shared block; the limit "
                                  f"is {max_length}")
            if current is None or len(current[0]) + len(ids) > limit:
                current = (list(shared), list(range(n)), [0] * n, [], [])
                out.append(current)
            current[3].append(len(current[0]) + len(ids) - 1)
            current[4].append(part)
            current[0].extend(ids)
            current[1].extend(range(n, n + len(ids)))
            current[2].extend([len(current[4])] * len(ids))
        return [sequence(*c) for c in out]
