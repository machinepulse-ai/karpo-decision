"""The readout head: one row per answer code, applied to the final hidden state at each readout marker."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..codes import MAX_CODES


class Readout(torch.nn.Module):
    """``hidden @ weight.T``, soft-capped as ``cap * tanh(logits / cap)`` when ``cap`` is set (a head started from the
    rows of a language-model head that caps its logits keeps that cap; ``families.<name>.logit_cap``)."""

    def __init__(self, hidden: int, codes: int = MAX_CODES, dtype: torch.dtype = torch.float32,
                 cap: float | None = None) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(codes, hidden, dtype=dtype))
        self.cap = cap

    @classmethod
    def from_rows(cls, rows: torch.Tensor, dtype: torch.dtype = torch.float32, cap: float | None = None) -> Readout:
        """A head starting from given rows, e.g. the language-model-head rows of the code tokens."""
        head = cls(rows.shape[1], rows.shape[0], dtype, cap)
        with torch.no_grad():
            head.weight.copy_(rows.to(dtype))
        return head

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Code logits (float32) for each row of ``hidden`` (markers × hidden size)."""
        logits = (hidden.to(self.weight.dtype) @ self.weight.t()).float()
        return logits if self.cap is None else self.cap * torch.tanh(logits / self.cap)


def per_question(logits: torch.Tensor, counts: Sequence[int]) -> list[torch.Tensor]:
    """Each question's logits over its own options: the first ``count`` codes of its row."""
    if logits.shape[0] != len(counts):
        raise ValueError(f"{logits.shape[0]} readout rows for {len(counts)} questions")
    return [logits[i, :count] for i, count in enumerate(counts)]
