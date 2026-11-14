"""Request bodies as TypeSafe's System One clients send them: compressed with one Content-Encoding (gzip, x-gzip or
zstd), and with images as multipart/form-data parts after the ``request`` JSON part. Nothing here needs a GPU."""

from __future__ import annotations

import email.message
import re
import sys
import zlib
from collections import Counter
from typing import Any

from ..protocol import ASSETS, DATA, PART, validate_assets

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

#: The request headers this module reads.
CONTENT_ENCODING, CONTENT_TYPE = "Content-Encoding", "Content-Type"
#: Content-Encoding values read as gzip, as zstd, and as no compression.
GZIP_ENCODINGS = ("gzip", "x-gzip")
ZSTD_ENCODING = "zstd"
IDENTITY = "identity"
#: zlib's window bits for a gzip header and trailer around the deflate stream.
GZIP_WBITS = 16 + zlib.MAX_WBITS
MULTIPART = "multipart/form-data"
#: The multipart part that holds the request's JSON, first.
REQUEST_PART = "request"
#: Multipart syntax (RFC 2046): a boundary line is ``--``, the boundary, optional padding and CRLF; a part's headers end
#: at a blank line, or at the next boundary for a part of headers alone.
DASHES, CRLF = b"--", b"\r\n"
PADDING = re.compile(rb"[ \t]*")
#: The longest header block a part may have: a part's few header lines, never a pathological one.
MAX_PART_HEADER_BYTES = 8 * 1024
#: A part's form-data parameters, each a token name and a quoted (backslash escapes allowed) or token value; the
#: part's name is its ``name`` parameter (RFC 7578: UTF-8, no ``name*``). A Content-Transfer-Encoding beyond the
#: identity ones is ruled out.
DISPOSITION = re.compile(rb"^content-disposition:[ \t]*form-data[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
PARAMETER = re.compile(rb';[ \t]*([^\s=;"]+)[ \t]*=[ \t]*(?:"((?:[^"\\\r\n]|\\.)*+)"|([^;\s"]*))')
PART_NAME, PART_NAME_CHARSET = b"name", "utf-8"
ESCAPE = re.compile(rb"\\(.)")
TRANSFER_ENCODING = re.compile(rb"^content-transfer-encoding:[ \t]*(\S*)", re.IGNORECASE | re.MULTILINE)
IDENTITY_TRANSFERS = (b"7bit", b"8bit", b"binary")
#: How HTTP header bytes are read.
HEADER_CHARSET = "latin-1"
#: The most gzip members or zstd frames a body may hold: one is usual, and each copies the rest of the body.
MAX_MEMBERS = 16


class TooLarge(ValueError):
    """The body is larger than the limit, compressed or not."""


class UnsupportedEncoding(ValueError):
    """The body's Content-Encoding is not one layer of gzip, x-gzip or zstd."""


def inflate(body: bytes, encoding: str | None, limit: int) -> bytes:
    """``body`` under its Content-Encoding (None or identity: as it is): ``TooLarge`` past ``limit`` bytes,
    ``UnsupportedEncoding`` for any other coding or several, ValueError when it is malformed or cut short. Up to
    ``MAX_MEMBERS`` gzip members or zstd frames are read one after another."""
    coding = (encoding or IDENTITY).strip().lower()
    if coding == IDENTITY:
        return body
    if coding in GZIP_ENCODINGS:
        make, errors = (lambda: zlib.decompressobj(GZIP_WBITS)), zlib.error
    elif coding == ZSTD_ENCODING:
        make, errors = zstd.ZstdDecompressor, zstd.ZstdError
    else:
        raise UnsupportedEncoding(f"Content-Encoding {encoding!r} is not supported: one of gzip, x-gzip or zstd")
    out, rest, members = bytearray(), body, 0
    try:
        while rest:
            members += 1
            if members > MAX_MEMBERS:
                raise ValueError(f"the body holds more than {MAX_MEMBERS} gzip members or zstd frames")
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
    """A multipart/form-data body (RFC 7578) split at its boundary by byte searches, never line by line: its
    ``request`` part (the first) and its other parts by name. ValueError when it has no boundary or parts, a boundary
    line without CRLF, a part cut short, with headers past ``MAX_PART_HEADER_BYTES``, without a name or with a
    Content-Transfer-Encoding, a name twice, or another part first."""
    header = email.message.Message()
    header[CONTENT_TYPE] = content_type
    boundary = header.get_boundary()
    if not boundary:
        raise ValueError("a multipart/form-data body needs a boundary and parts")
    delimiter = CRLF + DASHES + boundary.encode(HEADER_CHARSET)
    first = delimiter[len(CRLF):]
    if body.startswith(first) and _closes_a_boundary_line(body, len(first)):
        at = len(first)  # the first boundary line opens the body
    else:
        at = _next_delimiter(body, delimiter, 0)
        if at < 0:
            raise ValueError("a multipart boundary line must end with CRLF" if first in body
                             else "a multipart/form-data body needs a boundary and parts")
        at += len(delimiter)
    parts: dict[str, bytes] = {}
    while not body.startswith(DASHES, at):
        at = PADDING.match(body, at).end()
        if not body.startswith(CRLF, at):
            raise ValueError("a multipart boundary line must end with CRLF")
        end = _next_delimiter(body, delimiter, at)
        if end < 0:
            raise ValueError("a multipart part is cut short: no boundary after it")
        blank = body.find(CRLF + CRLF, at, end)
        headers_end = blank if blank >= 0 else end
        if headers_end - at - len(CRLF) > MAX_PART_HEADER_BYTES:
            raise ValueError(f"a multipart part's headers are longer than {MAX_PART_HEADER_BYTES} bytes")
        headers = body[at + len(CRLF):headers_end]
        transfer = TRANSFER_ENCODING.search(headers)
        if transfer and transfer.group(1) and transfer.group(1).lower() not in IDENTITY_TRANSFERS:
            raise ValueError("a multipart part has a Content-Transfer-Encoding, which RFC 7578 rules out")
        key = _part_name(headers)
        if key in parts:
            raise ValueError(f"multipart part {key!r} appears twice")
        parts[key] = body[blank + 2 * len(CRLF):end] if blank >= 0 else b""
        at = end + len(delimiter)
    if next(iter(parts), None) != REQUEST_PART:
        raise ValueError(f"the first multipart part must be {REQUEST_PART!r}, the request's JSON")
    return parts.pop(REQUEST_PART), parts


def _closes_a_boundary_line(body: bytes, at: int) -> bool:
    """Whether the boundary that ends at ``at`` is a boundary line: the closing ``--``, or padding and CRLF."""
    return body.startswith(DASHES, at) or body.startswith(CRLF, PADDING.match(body, at).end())


def _next_delimiter(body: bytes, delimiter: bytes, start: int) -> int:
    """Where the next boundary line's CRLF is, from ``start``: the boundary inside content is no delimiter unless a
    boundary line follows it."""
    at = body.find(delimiter, start)
    while at >= 0 and not _closes_a_boundary_line(body, at + len(delimiter)):
        at = body.find(delimiter, at + 1)
    return at


def _part_name(headers: bytes) -> str:
    """The ``name`` parameter of a part's form-data Content-Disposition; ValueError when it has none."""
    disposition = DISPOSITION.search(headers)
    for match in PARAMETER.finditer(disposition.group(1)) if disposition else ():
        if match.group(1).lower() == PART_NAME:
            quoted = match.group(2)
            name = ESCAPE.sub(rb"\1", quoted) if quoted is not None else match.group(3)
            if name:
                return name.decode(PART_NAME_CHARSET)
            break
    raise ValueError("every multipart part needs a name")


def claim(body: dict[str, Any], parts: dict[str, bytes]) -> dict[str, Any]:
    """``body`` with each asset that names a multipart part holding that part's bytes (``protocol.DATA``) instead of
    the name: ``RequestError`` when the assets break the contract, ValueError unless every part is named by exactly
    one asset and every part an asset names is in the body."""
    assets = body.get(ASSETS)
    if assets is not None:
        validate_assets(assets)
    named = {key: asset[PART] for key, asset in (assets or {}).items() if asset.get(PART) is not None}
    counts = Counter(named.values())
    for name in parts:
        if counts[name] != 1:
            raise ValueError(f"multipart part {name!r} must be named by exactly one asset, not {counts[name]}")
    for key, name in named.items():
        if name not in parts:
            raise ValueError(f"asset {key!r} names multipart part {name!r}, which the body does not have")
    if not named:
        return body
    claimed = {key: {**{field: value for field, value in assets[key].items() if field != PART}, DATA: parts[name]}
               for key, name in named.items()}
    return {**body, ASSETS: {**assets, **claimed}}
