"""Answers in the System One shape, from a question's probabilities over its options.

- choice: ``{"type": "choice", "choice": <key>, "probabilities": {<key>: p}, "confidence": c}``, where ``c`` rescales
  the top probability so a uniform guess is 0 and certainty is 1;
- noul: ``{"type": "noul", "noul": P(true)}``;
- score: ``{"type": "score", "score": E[level], "probabilities": {...}, "legend": {...}, "confidence": c}``, where ``c``
  is 1 minus the expected distance from the most likely level, relative to a uniform distribution's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .protocol import Question, options


def normalized(probabilities: Sequence[float], expected: int) -> list[float]:
    """Check one finite, non-negative value per option with positive mass, and scale them to sum to one."""
    values = [float(v) for v in probabilities]
    if len(values) != expected:
        raise ValueError(f"expected {expected} option probabilities, got {len(values)}")
    if any(not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
        raise ValueError("option probabilities must be finite, non-negative and have positive mass")
    total = sum(values)
    return [v / total for v in values]


def answer(question: Question, probabilities: Sequence[float]) -> dict[str, Any]:
    keys, descriptions = options(question)
    values = normalized(probabilities, len(keys))
    if question["type"] == "noul":
        return {"type": "noul", "noul": values[1]}
    best = max(range(len(values)), key=values.__getitem__)
    distribution = dict(zip(keys, values, strict=True))
    if question["type"] == "choice":
        n = len(values)
        confidence = 1.0 if n == 1 else (values[best] - 1 / n) / (1 - 1 / n)
        return {"type": "choice", "choice": keys[best], "probabilities": distribution, "confidence": _clip(confidence)}
    distance = sum(p * abs(i - best) for i, p in enumerate(values))
    midpoint = (len(values) - 1) / 2
    uniform = sum(abs(i - midpoint) for i in range(len(values))) / len(values)
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(values)),
        "probabilities": distribution,
        "legend": dict(zip(keys, descriptions, strict=True)),
        "confidence": _clip(1.0 - distance / uniform),
    }


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
