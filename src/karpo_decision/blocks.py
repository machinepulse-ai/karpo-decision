"""Requests laid out in blocks (``templates.BlockTemplate``): a shared block and one branch per question, as token ids
with each token's position and segment, for a family whose attention keeps branches apart (Gemma).

- A request's images (``protocol.images_of``) sit in the shared block, after the context header and before the state
  text (``families/gemma/vision.py``); a checkpoint without a vision tower refuses them.
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

from .protocol import Question, Request, describe, images_of, options
from .templates import DEFAULT_INSTRUCTIONS, BlockTemplate

#: The line before each image when a request has several (k from 1).
IMAGE_LABEL = "Image {}:\n"


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

    def text(self, value: Any) -> list[int]:
        return self.tokenize(self.encoder.escape(describe(value)))

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

    def shared(self, state: Any, questions: Sequence[Question], images: Sequence[int] = ()) -> list[int]:
        """``images``: the images' soft tokens."""
        ids = self.prefix + self.header
        for question in questions:
            ids += self.item + self.instructions(question) + self.newline
        return ids + self.context + self.image_ids(images) + self.text(state)

    def branch(self, question: Question) -> list[int]:
        _, texts = options(question)
        ids = self.question + self.instructions(question) + self.options
        for label, text in zip(self.labels, texts, strict=False):
            ids += self.open + [label] + self.close + self.text(text) + self.newline
        return ids + self.tail

    def plan(self, request: Request, max_length: int | None) -> list[Any]:
        from .encoding import Encoded, LengthError, TextOnlyError

        state, values = images_of(request)
        images = []
        if values:
            vision = getattr(self.encoder.processor, "vision", None)
            if vision is None:
                raise TextOnlyError("this checkpoint reads text only")
            images = self.measure_images(vision, values, max_length)
        items = list(request["questions"].items())
        shared = self.shared(state, [q for _, q in items], [planned.tokens for planned in images])
        branches = [(key, len(options(q)[0]), self.branch(q)) for key, q in items]
        limit = max_length or float("inf")

        def sequence(ids: list[int], positions: list[int], segments: list[int], markers: list[int],
                     parts: list[tuple[str, int, list[int]]]) -> Encoded:
            return Encoded(keys=tuple(k for k, _, _ in parts), counts=tuple(c for _, c, _ in parts),
                           input_ids=torch.tensor(ids), markers=torch.tensor(markers),
                           positions=torch.tensor(positions), segment_ids=torch.tensor(segments, dtype=torch.int16),
                           images=tuple(images) or None)

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
