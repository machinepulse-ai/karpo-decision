"""Images a request places inline (TypeSafe's System One assets, OpenRouter's image parts): the keys of the request's
``assets`` wherever they occur in a string of the state, of a question's instructions or of its criteria, and
OpenAI-style ``image_url`` parts anywhere in those. ``resolve`` replaces each occurrence by a sentinel string, so the
layout renders the text as it renders any request and cuts it at the sentinels (``Resolved.pieces``); the screenshot
and ``images`` (``protocol.images_of``) stay legacy images. Nothing here reads pixels.

Matching follows TypeSafe: only string values are searched, never object keys; in each string the leftmost match
wins, the longest key where several start, and matches do not overlap (a string that is a key is one match). Each
asset must be referenced."""

from __future__ import annotations

import base64
import binascii
import re
import uuid
from dataclasses import dataclass
from typing import Any

from .protocol import ASSETS, DATA, DATA_BASE64, IMAGE_DATA_URL, IMAGE_URL, PART, Question, Request, images_of

#: Around a sentinel: Unicode private-use characters, which JSON rendering and the escaping leave as they are.
SENTINEL_OPEN, SENTINEL_CLOSE = chr(0xE000), chr(0xE001)
#: The fields of a question that may hold inline images.
SEARCHED = ("instructions", "criteria")


class AssetError(ValueError):
    """The request breaks a rule of its images: an asset no string references, a multipart part outside a multipart
    request, base64 that does not decode, an image part whose URL is not a data URL."""


@dataclass(frozen=True)
class Resolved:
    """A request with its inline images taken out (``resolve``)."""

    state: Any  # without the legacy screenshot; each inline image a sentinel
    questions: dict[str, Question]  # each inline image a sentinel
    legacy: list[Any]  # ``images_of``'s images, the screenshot first
    inline: list[Any]  # each inline image's bytes or data URL, by sentinel number
    pattern: re.Pattern[str] | None = None  # the sentinels' pattern; None without inline images

    def pieces(self, text: str) -> list[str | int]:
        """``text`` cut at its sentinels: its text and each inline image's number, in order, without empty text."""
        if self.pattern is None:
            return [text]
        return [int(p) if odd else p for odd, p in ((i % 2, p) for i, p in enumerate(self.pattern.split(text)))
                if odd or p]


class _Finder:
    """One request's inline images, found in order (``walk``)."""

    def __init__(self, assets: dict[str, Any]) -> None:
        self.assets = assets
        self.nonce = uuid.uuid4().hex
        keys = sorted(assets, key=len, reverse=True)
        self.keys = re.compile("|".join(map(re.escape, keys))) if keys else None
        self.inline: list[Any] = []
        self.data: dict[str, bytes] = {}  # each referenced asset's bytes, read once

    def walk(self, value: Any) -> Any:
        if isinstance(value, str):
            return value if self.keys is None else self.keys.sub(lambda m: self.place(self.asset(m.group())), value)
        if isinstance(value, dict):
            if value.get("type") == IMAGE_URL:
                return self.place(_data_url(value))
            return {key: self.walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.walk(item) for item in value]
        return value

    def place(self, source: Any) -> str:
        self.inline.append(source)
        return f"{SENTINEL_OPEN}{self.nonce}:{len(self.inline) - 1}{SENTINEL_CLOSE}"

    def asset(self, key: str) -> bytes:
        if key not in self.data:
            self.data[key] = _image_bytes(key, self.assets[key])
        return self.data[key]


def resolve(request: Request) -> Resolved:
    """``request`` (validated) with each inline image replaced by a sentinel; ``AssetError`` when it breaks a rule."""
    state, legacy = images_of(request)
    assets = request.get(ASSETS) or {}
    finder = _Finder(assets)
    state = finder.walk(state)
    questions = {key: {field: finder.walk(value) if field in SEARCHED else value for field, value in question.items()}
                 for key, question in request["questions"].items()}
    unused = [key for key in assets if key not in finder.data]
    if unused:
        raise AssetError(f"Asset {unused[0]!r} is not referenced in the state, instructions or criteria")
    if not finder.inline:
        return Resolved(state, questions, legacy, [])
    pattern = re.compile(f"{SENTINEL_OPEN}{finder.nonce}:(\\d+){SENTINEL_CLOSE}")
    return Resolved(state, questions, legacy, finder.inline, pattern)


def _image_bytes(key: str, asset: dict[str, Any]) -> bytes:
    if asset.get(PART) is not None:
        raise AssetError(f"Asset {key!r} names multipart part {asset[PART]!r}, which only a multipart/form-data "
                         "request to the server carries")
    if isinstance(asset.get(DATA), bytes):
        return asset[DATA]
    try:
        return base64.b64decode("".join(asset[DATA_BASE64].split()), validate=True)
    except (binascii.Error, ValueError) as error:
        raise AssetError(f"Asset {key!r}: data_base64 is not base64") from error


def _data_url(part: dict[str, Any]) -> str:
    inner = part.get(IMAGE_URL)
    url = inner.get("url") if isinstance(inner, dict) else None
    if not (isinstance(url, str) and url.startswith(IMAGE_DATA_URL)):
        raise AssetError("an image part's URL must be a data:image/... URL")
    return url
