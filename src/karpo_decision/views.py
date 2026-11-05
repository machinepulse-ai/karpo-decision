"""Readings of a request: the requests answered for it and how their answers make its own.

- ``joint``: the request itself, every question in one sequence;
- ``single``: each question in a request of its own (the state and that one question);
- ``reversed``: the request as given and again with the options of every choice question that can move
  (``train.order.movable``) in reverse order, the two probabilities of each option key averaged. A request with no
  movable question is read once (its second reading would be the same request).
"""

from __future__ import annotations

from typing import Any

from .answers import answer
from .protocol import Request, options

VIEWS = ("joint", "single", "reversed")


def views(request: Request, view: str) -> list[Request]:
    """The requests answered for one request under ``view``."""
    if view == "single":
        return [{"state": request["state"], "questions": {key: q}} for key, q in request["questions"].items()]
    if view == "reversed":
        from .train.order import movable

        moved = {key for key, q in request["questions"].items() if movable(q)}
        if not moved:
            return [request]
        flipped = {key: ({**q, "criteria": dict(reversed(list(q["criteria"].items())))} if key in moved else q)
                   for key, q in request["questions"].items()}
        return [request, {"state": request["state"], "questions": flipped}]
    return [request]


def joined(request: Request, responses: list[dict[str, Any]], view: str) -> dict[str, Any]:
    """One response for ``request`` from the responses of its ``views``."""
    usage = {k: sum(r["usage"][k] for r in responses) for k in responses[0]["usage"]}
    if view == "single":
        return {"answers": {k: v for r in responses for k, v in r["answers"].items()}, "usage": usage}
    if view == "reversed" and len(responses) == 2:
        answers = {}
        for key, question in request["questions"].items():
            first, second = (r["answers"][key] for r in responses)
            if question.get("type") != "choice":
                answers[key] = first
                continue
            keys = options(question)[0]
            answers[key] = answer(question, [(first["probabilities"][k] + second["probabilities"][k]) / 2
                                             for k in keys])
        return {"answers": answers, "usage": usage}
    return responses[0]
