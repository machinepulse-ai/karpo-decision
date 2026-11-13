"""Request bodies as TypeSafe's System One clients send them: compressed with one Content-Encoding (gzip, x-gzip or
zstd), and with images as multipart/form-data parts after the ``request`` JSON part. Nothing here needs a GPU."""

from __future__ import annotations

import email.parser
import email.policy
import sys
import zlib
from collections import Counter
from typing import Any

from ..protocol import ASSET_TYPE, ASSETS, DATA, IMAGE_ASSET, PART

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

#: Content-Encoding values read as gzip, as zstd, and as no compression.
GZIP_ENCODINGS = ("gzip", "x-gzip")
ZSTD_ENCODING = "zstd"
IDENTITY = "identity"
#: zlib's window bits for a gzip header and trailer around the deflate stream.
GZIP_WBITS = 16 + zlib.MAX_WBITS
MULTIPART = "multipart/form-data"
#: The multipart part that holds the request's JSON, first.
REQUEST_PART = "request"


class TooLarge(ValueError):
    """The body is larger than the limit, compressed or not."""


class UnsupportedEncoding(ValueError):
    """The body's Content-Encoding is not one layer of gzip, x-gzip or zstd."""


def inflate(body: bytes, encoding: str | None, limit: int) -> bytes:
    """``body`` under its Content-Encoding (None or identity: as it is): ``TooLarge`` past ``limit`` bytes,
    ``UnsupportedEncoding`` for any other coding or several, ValueError when it is malformed or cut short. Several
    gzip members or zstd frames are read one after another."""
    coding = (encoding or IDENTITY).strip().lower()
    if coding == IDENTITY:
        return body
    if coding in GZIP_ENCODINGS:
        make, errors = (lambda: zlib.decompressobj(GZIP_WBITS)), zlib.error
    elif coding == ZSTD_ENCODING:
        make, errors = zstd.ZstdDecompressor, zstd.ZstdError
    else:
        raise UnsupportedEncoding(f"Content-Encoding {encoding!r} is not supported: one of gzip, x-gzip or zstd")
    out, rest = bytearray(), body
    try:
        while rest:
            decompressor = make()
            out += decompressor.decompress(rest, limit + 1 - len(out))
            if len(out) > limit:
                raise TooLarge(f"the body is larger than {limit} bytes decompressed")
            if not decompressor.eof:
                raise ValueError(f"the {coding} body is cut short")
            rest = decompressor.unused_data
    except errors as error:
        raise ValueError(f"the {coding} body does not decompress: {error}") from error
    return bytes(out)


def parse_multipart(body: bytes, content_type: str) -> tuple[bytes, dict[str, bytes]]:
    """A multipart/form-data body's ``request`` part (the first) and its other parts by name; ValueError when it has
    no boundary or parts, a part without a name, a name twice, or another part first."""
    header = b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n"
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(header + body)
    if not message.is_multipart():
        raise ValueError("a multipart/form-data body needs a boundary and parts")
    parts: dict[str, bytes] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not isinstance(name, str) or not name:
            raise ValueError("every multipart part needs a name")
        if name in parts:
            raise ValueError(f"multipart part {name!r} appears twice")
        parts[name] = part.get_payload(decode=True) or b""
    if next(iter(parts), None) != REQUEST_PART:
        raise ValueError(f"the first multipart part must be {REQUEST_PART!r}, the request's JSON")
    return parts.pop(REQUEST_PART), parts


def claim(body: dict[str, Any], parts: dict[str, bytes]) -> dict[str, Any]:
    """``body`` with each asset that names a multipart part holding that part's bytes (``protocol.DATA``); ValueError
    unless every part is named by exactly one asset and every part an asset names is in the body."""
    assets = body.get(ASSETS)
    named = {key: asset[PART] for key, asset in assets.items()
             if isinstance(asset, dict) and isinstance(asset.get(PART), str)} if isinstance(assets, dict) else {}
    counts = Counter(named.values())
    for name in parts:
        if counts[name] != 1:
            raise ValueError(f"multipart part {name!r} must be named by exactly one asset, not {counts[name]}")
    for key, name in named.items():
        if name not in parts:
            raise ValueError(f"asset {key!r} names multipart part {name!r}, which the body does not have")
    if not named:
        return body
    claimed = {key: {ASSET_TYPE: IMAGE_ASSET, DATA: parts[name]} for key, name in named.items()}
    return {**body, ASSETS: {**assets, **claimed}}
