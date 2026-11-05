"""The System One request contract: a state and typed questions about it.

A request is ``{"state": <text or JSON>, "questions": {<id>: <question>, ...}, "images": [...]}`` (images optional).
A question is one of:

- ``{"type": "choice", "instructions": ..., "criteria": {<option key>: <description or None>, ...}}``
- ``{"type": "noul", "instructions": ..., "criteria": {"false": ..., "true": ...}}`` (criteria optional)
- ``{"type": "score", "instructions": ..., "criteria": [<lowest level>, ..., <highest level>]}``

A caller may put a page screenshot in the state at ``VIEWPORT_SCREENSHOT``, as an OpenAI-style image part
``{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}``; ``images_of`` lifts it out so the model
reads it as an image rather than as text. Nothing here needs a GPU.
"""

from __future__ import annotations

import json
from typing import Any, Literal, NotRequired, TypedDict

#: The most options one question may have: one answer code each (see ``codes.MAX_CODES``).
MAX_OPTIONS = 255
#: Where a request may carry the page screenshot (a data URL) inside the state.
VIEWPORT_SCREENSHOT = "viewport_screenshot"
#: What a noul question's two options say when its criteria do not describe them.
NOUL_KEYS = ("false", "true")
NOUL_DEFAULTS = ("No / false", "Yes / true")
QUESTION_TYPES = ("choice", "noul", "score")

QuestionType = Literal["choice", "noul", "score"]
JSONValue = Any


class Question(TypedDict):
    type: QuestionType
    instructions: NotRequired[str]
    criteria: NotRequired[Any]


class Request(TypedDict):
    state: JSONValue
    questions: dict[str, Question]
    images: NotRequired[list[Any]]


class RequestError(ValueError):
    """The request does not follow the contract (unknown type, bad option count, no questions, ...)."""


def describe(value: JSONValue) -> str:
    """Text of a state, instruction or description: strings verbatim, anything else as JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def options(question: Question) -> tuple[list[str], list[str]]:
    """The option keys answers use and the text shown for each, in prompt order."""
    kind = question["type"]
    if kind == "choice":
        criteria = question["criteria"]
        return list(criteria), [key if text is None else f"{key}: {describe(text)}" for key, text in criteria.items()]
    if kind == "score":
        levels = question["criteria"]
        return [str(i) for i in range(len(levels))], [describe(level) for level in levels]
    given = question.get("criteria") or {}
    texts = [describe(given.get(key) or default) for key, default in zip(NOUL_KEYS, NOUL_DEFAULTS, strict=True)]
    return list(NOUL_KEYS), texts


def validate(request: Request, max_options: int = MAX_OPTIONS) -> None:
    """Raise ``RequestError`` unless the request follows the contract."""
    if not isinstance(request, dict) or "state" not in request or "questions" not in request:
        raise RequestError("A request needs 'state' and 'questions'")
    questions = request["questions"]
    if not isinstance(questions, dict) or not questions:
        raise RequestError("A request needs at least one question")
    for key, question in questions.items():
        if not isinstance(key, str) or not key:
            raise RequestError("Question ids must be nonempty strings")
        kind = question.get("type") if isinstance(question, dict) else None
        if kind not in QUESTION_TYPES:
            raise RequestError(f"Question {key!r}: type must be one of {', '.join(QUESTION_TYPES)}")
        if kind == "choice" and not isinstance(question.get("criteria"), dict):
            raise RequestError(f"Question {key!r}: choice criteria must map option keys to descriptions")
        if kind == "score" and not isinstance(question.get("criteria"), list):
            raise RequestError(f"Question {key!r}: score criteria must list the levels, lowest first")
        if kind == "noul" and not isinstance(question.get("criteria") or {}, dict):
            raise RequestError(f"Question {key!r}: noul criteria, when given, must map false and true to descriptions")
        count = len(options(question)[0])
        if not 1 <= count <= max_options:
            raise RequestError(f"Question {key!r}: {count} options; 1 to {max_options} are supported")
        if kind == "score" and count < 2:
            raise RequestError(f"Question {key!r}: a score needs at least two levels")


def image_source(value: Any) -> Any:
    """An image as ``encoding.open_image`` reads it: the URL of an ``image_url`` part, anything else unchanged."""
    if isinstance(value, dict) and value.get("type") == "image_url":
        return (value.get("image_url") or {}).get("url")
    return value


def images_of(request: Request) -> tuple[JSONValue, list[Any]]:
    """The state without its screenshot, and every image of the request (the screenshot first)."""
    state, images = request["state"], [image_source(value) for value in request.get("images") or []]
    if isinstance(state, dict) and state.get(VIEWPORT_SCREENSHOT):
        images.insert(0, image_source(state[VIEWPORT_SCREENSHOT]))
        state = {key: value for key, value in state.items() if key != VIEWPORT_SCREENSHOT}
    return state, images
