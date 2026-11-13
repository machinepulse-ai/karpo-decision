"""The System One request contract: a state and typed questions about it.

A request is ``{"state": <text or JSON>, "questions": {<id>: <question>, ...}, "images": [...], "assets": {...}}``
(images and assets optional).
A question is one of:

- ``{"type": "choice", "instructions": ..., "criteria": {<option key>: <description or None>, ...}}``
- ``{"type": "noul", "instructions": ..., "criteria": {"false": ..., "true": ...}}`` (criteria optional)
- ``{"type": "score", "instructions": ..., "criteria": [<lowest level>, ..., <highest level>]}``

A caller may put a page screenshot in the state at ``VIEWPORT_SCREENSHOT``, as an OpenAI-style image part
``{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}``; ``images_of`` lifts it out so the model
reads it as an image rather than as text. Images may also sit anywhere in the state, a question's instructions or its
criteria, as such parts or as the keys of ``assets`` (TypeSafe's System One; ``assets.py``). Nothing here needs a GPU.
"""

from __future__ import annotations

import json
from typing import Any, Literal, NotRequired, TypedDict

#: The most options one question may have: one answer code each (see ``codes.MAX_CODES``).
MAX_OPTIONS = 255
#: Where a request may carry the page screenshot (a data URL) inside the state.
VIEWPORT_SCREENSHOT = "viewport_screenshot"
#: The type of an OpenAI-style image part.
IMAGE_URL = "image_url"
#: How an inline image's URL starts.
IMAGE_DATA_URL = "data:image/"
#: The top-level map of a request's images by the key its text references them with (TypeSafe's System One).
ASSETS = "assets"
#: An asset's fields: its type (``IMAGE_ASSET``, or absent), and its image as base64 text with its format, as raw bytes
#: (Python callers; the server puts each multipart part here), or as the name of a multipart part.
ASSET_TYPE, IMAGE_ASSET = "type", "image"
DATA_BASE64, FORMAT, DATA, PART = "data_base64", "format", "data", "part"
#: The formats an asset with ``data_base64`` may name; the image's own bytes decide how it is read.
FORMATS = ("png", "jpg", "jpeg")
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
    assets: NotRequired[dict[str, Any]]


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
    if request.get(ASSETS) is not None:
        validate_assets(request[ASSETS])


def validate_assets(assets: Any) -> None:
    """Raise ``RequestError`` unless ``assets`` is a nonempty map of keys to image assets, each with ``part``, with
    ``data`` bytes, or with ``data_base64`` text and one of ``FORMATS``."""
    if not isinstance(assets, dict) or not assets:
        raise RequestError("'assets' must be a nonempty object")
    for key, asset in assets.items():
        if not isinstance(key, str) or not key:
            raise RequestError("Asset keys must be nonempty strings")
        if not isinstance(asset, dict) or asset.get(ASSET_TYPE) not in (None, IMAGE_ASSET):
            raise RequestError(f"Asset {key!r}: must be an object of type {IMAGE_ASSET!r}")
        if asset.get(PART) is not None:
            if not isinstance(asset[PART], str):
                raise RequestError(f"Asset {key!r}: 'part' must name a multipart part")
        elif not isinstance(asset.get(DATA), bytes):
            if not isinstance(asset.get(DATA_BASE64), str):
                raise RequestError(f"Asset {key!r}: needs 'data_base64' ('part' in a multipart request)")
            if asset.get(FORMAT) not in FORMATS:
                raise RequestError(f"Asset {key!r}: 'format' must be one of {', '.join(FORMATS)}")


def image_source(value: Any) -> Any:
    """An image as ``encoding.open_image`` reads it: the URL of an ``image_url`` part, anything else unchanged."""
    if isinstance(value, dict) and value.get("type") == IMAGE_URL:
        return (value.get(IMAGE_URL) or {}).get("url")
    return value


def images_of(request: Request) -> tuple[JSONValue, list[Any]]:
    """The state without its screenshot, and every image of the request (the screenshot first)."""
    state, images = request["state"], [image_source(value) for value in request.get("images") or []]
    if isinstance(state, dict) and state.get(VIEWPORT_SCREENSHOT):
        images.insert(0, image_source(state[VIEWPORT_SCREENSHOT]))
        state = {key: value for key, value in state.items() if key != VIEWPORT_SCREENSHOT}
    return state, images
