"""Answering System One requests with a checkpoint: encode, pack, one forward pass per pack, read out, answer.

>>> decider = Decider.from_pretrained("path/to/checkpoint", device="cuda")
>>> decider.decide({"state": "...", "questions": {"next": {"type": "choice", "criteria": {...}}}})
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

from . import checkpoint as checkpoints
from .answers import answer
from .encoding import Encoded, Encoder
from .model.packed import pack
from .model.readout import per_question
from .protocol import Request
from .templates import template

#: The longest single sequence; a longer request is split into parts of its questions (each with the full state).
DEFAULT_MAX_LENGTH = 32_768
#: Tokens per packed forward pass. A sequence longer than this runs alone.
DEFAULT_TOKEN_BUDGET = 32_768


class Decider:
    def __init__(self, ckpt: checkpoints.Checkpoint, max_length: int = DEFAULT_MAX_LENGTH,
                 token_budget: int = DEFAULT_TOKEN_BUDGET) -> None:
        self.ckpt = ckpt
        self.config = ckpt.config
        self.encoder = Encoder(ckpt.processor, ckpt.config.codes, template(ckpt.config.template), ckpt.config.marker)
        self.model = ckpt.family.Packed(ckpt.backbone, ckpt.config.attention).eval()
        self.readout = ckpt.readout.eval()
        self.device = next(ckpt.backbone.parameters()).device
        self.max_length = max_length
        self.token_budget = token_budget

    @classmethod
    def from_pretrained(cls, directory: str | Path, device: str = "cuda", **kwargs: Any) -> Decider:
        return cls(checkpoints.load(directory, device), **kwargs)

    # ---------------------------------------------------------------- public API

    def plan(self, request: Request) -> list[Encoded]:
        """The sequences ``decide`` runs for a request: one, or parts of its questions when it is longer than
        ``max_length``. Raises ``RequestError`` outside the contract and ``LengthError`` when a question cannot fit."""
        return self.encoder.plan(request, self.max_length)

    def prepare(self, plan: list[Encoded]) -> list[Encoded]:
        """``plan`` with the pixels of the images its first pack reads (``Vision.read``), the rest left to their own
        packs: a server prepares one request's first pack while the GPU runs another."""
        vision = getattr(self.encoder.processor, "vision", None)
        if vision is None:
            return plan
        first = len(next(self._packs(list(enumerate(plan))), []))
        return vision.read(plan[:first]) + plan[first:]

    def decide(self, requests: Request | Sequence[Request],
               plans: Sequence[list[Encoded]] | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        """One response per request (``{"answers": {...}, "usage": {...}}``), in input order. ``plans``, when given,
        are the requests' ``plan``, already made (by a caller that sorts or screens requests before answering)."""
        single = isinstance(requests, dict)
        batch = [requests] if single else list(requests)
        responses = []
        for request, logits, parts in self._run(batch, plans):
            answers = {}
            for key, question in request["questions"].items():
                scaled = logits[key] / self.config.temperatures[question["type"]]
                answers[key] = answer(question, scaled.softmax(-1).tolist())
            responses.append({"answers": answers,
                              "usage": {"input_tokens": sum(p.length for p in parts), "output_tokens": 0,
                                        "sequences": len(parts)}})
        return responses[0] if single else responses

    def question_logits(self, requests: Sequence[Request],
                        plans: Sequence[list[Encoded]] | None = None) -> list[dict[str, torch.Tensor]]:
        """Each question's raw logits over its options (before temperature), per request."""
        return [logits for _, logits, _ in self._run(list(requests), plans)]

    # ---------------------------------------------------------------- internals

    def _run(self, requests: list[Request], plans: Sequence[list[Encoded]] | None = None,
             ) -> Iterator[tuple[Request, dict[str, torch.Tensor], list[Encoded]]]:
        plans = [self.plan(request) for request in requests] if plans is None else list(plans)
        parts = [(index, part) for index, plan in enumerate(plans) for part in plan]
        results: dict[tuple[int, str], torch.Tensor] = {}
        for group in self._packs(parts):
            rows = self.logits([part for _, part in group])
            for (index, part), logits in zip(group, rows, strict=True):
                results.update({(index, key): row for key, row in zip(part.keys, logits, strict=True)})
        for index, (request, plan) in enumerate(zip(requests, plans, strict=True)):
            yield request, {key: results[index, key] for key in request["questions"]}, plan

    def _packs(self, parts: list[tuple[int, Encoded]]) -> Iterator[list[tuple[int, Encoded]]]:
        """Consecutive parts filling at most ``token_budget`` tokens per pack (always at least one part)."""
        group: list[tuple[int, Encoded]] = []
        used = 0
        for item in parts:
            if group and used + item[1].length > self.token_budget:
                yield group
                group, used = [], 0
            group.append(item)
            used += item[1].length
        if group:
            yield group

    @torch.inference_mode()
    def logits(self, sequences: Sequence[Encoded]) -> list[list[torch.Tensor]]:
        """Per sequence, each question's logits over its options (CPU, float32). Planned images get their pixels here
        (``Vision.read``), for this pass only."""
        vision = getattr(self.encoder.processor, "vision", None)
        if vision is not None:
            sequences = vision.read(sequences)
        hidden = self.model(pack(sequences, self.ckpt.backbone, self.device))
        flat = self.readout(hidden).cpu()
        out, cursor = [], 0
        for s in sequences:
            out.append(per_question(flat[cursor:cursor + len(s.keys)], s.counts))
            cursor += len(s.keys)
        return out
