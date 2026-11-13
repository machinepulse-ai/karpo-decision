"""``serve.wire``: request bodies decompressed (one layer of gzip, x-gzip or zstd, within a limit) and multipart bodies
split into the request's JSON and its parts, each part claimed by one asset."""

import gzip
import sys

import pytest

from karpo_decision.protocol import RequestError
from karpo_decision.serve import wire

zstd = pytest.importorskip("compression.zstd" if sys.version_info >= (3, 14) else "backports.zstd")
LIMIT = 1000


@pytest.mark.parametrize("coding", ["gzip", "x-gzip", "GZIP"])
def test_gzip_bodies_inflate(coding):
    assert wire.inflate(gzip.compress(b"hello"), coding, LIMIT) == b"hello"


def test_zstd_bodies_inflate():
    assert wire.inflate(zstd.compress(b"hello"), "zstd", LIMIT) == b"hello"


def test_no_encoding_or_identity_keeps_the_body():
    assert wire.inflate(b"raw", None, LIMIT) == wire.inflate(b"raw", "identity", LIMIT) == b"raw"


def test_several_members_and_frames_are_read_in_turn():
    assert wire.inflate(gzip.compress(b"ab") + gzip.compress(b"cd"), "gzip", LIMIT) == b"abcd"
    assert wire.inflate(zstd.compress(b"ab") + zstd.compress(b"cd"), "zstd", LIMIT) == b"abcd"


@pytest.mark.parametrize("coding", ["br", "deflate", "gzip, zstd", "compress"])
def test_other_encodings_are_unsupported(coding):
    with pytest.raises(wire.UnsupportedEncoding):
        wire.inflate(b"x", coding, LIMIT)


@pytest.mark.parametrize(("coding", "data"), [
    ("gzip", b"not gzip"), ("zstd", b"not zstd"),
    ("gzip", gzip.compress(b"x" * 100)[:-6]), ("zstd", zstd.compress(b"x" * 100)[:-3])])
def test_malformed_or_cut_bodies_are_value_errors(coding, data):
    with pytest.raises(ValueError) as caught:
        wire.inflate(data, coding, LIMIT)
    assert not isinstance(caught.value, (wire.TooLarge, wire.UnsupportedEncoding))


@pytest.mark.parametrize(("coding", "compress"), [("gzip", gzip.compress), ("zstd", zstd.compress)])
def test_the_limit_holds_decompressed(coding, compress):
    assert wire.inflate(compress(b"x" * LIMIT), coding, LIMIT) == b"x" * LIMIT
    with pytest.raises(wire.TooLarge):
        wire.inflate(compress(b"x" * (LIMIT + 1)), coding, LIMIT)


def multipart(parts, boundary="b0undary"):
    chunks = []
    for name, data in parts:
        chunks += [f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode(), data, b"\r\n"]
    return f"multipart/form-data; boundary={boundary}", b"".join(chunks) + f"--{boundary}--\r\n".encode()


def test_multipart_gives_the_request_and_the_parts_by_name():
    binary = bytes(range(256)) + b"\r\n--b0undar\r\n\n"
    content_type, data = multipart([("request", b'{"state": "s"}'), ("asset-0", binary)])
    assert wire.parse_multipart(data, content_type) == (b'{"state": "s"}', {"asset-0": binary})


@pytest.mark.parametrize(("parts", "message"), [
    ([("asset-0", b"x"), ("request", b"{}")], "first multipart part"),
    ([("request", b"{}"), ("a", b"1"), ("a", b"2")], "twice"),
])
def test_multipart_rules(parts, message):
    content_type, data = multipart(parts)
    with pytest.raises(ValueError, match=message):
        wire.parse_multipart(data, content_type)


def test_multipart_needs_a_boundary_and_parts():
    with pytest.raises(ValueError, match="boundary"):
        wire.parse_multipart(b"--x\r\n\r\n{}\r\n--x--\r\n", "multipart/form-data")
    with pytest.raises(ValueError, match="boundary"):
        wire.parse_multipart(b"no parts at all", "multipart/form-data; boundary=x")


def test_claim_gives_each_part_naming_asset_its_bytes():
    body = {"state": "<a> <b> <c>", "assets": {"<a>": {"type": "image", "part": "p0"},
                                               "<b>": {"format": "png", "data_base64": "AA=="},
                                               "<c>": {"part": "p1", "format": "png", "data_base64": "AA=="}}}
    claimed = wire.claim(body, {"p0": b"img", "p1": b"two"})
    assert claimed["assets"] == {"<a>": {"type": "image", "data": b"img"}, "<b>": body["assets"]["<b>"],
                                 "<c>": {"format": "png", "data_base64": "AA==", "data": b"two"}}
    assert wire.claim({"state": "s"}, {}) == {"state": "s"}


def test_claim_checks_the_asset_fields_first():
    with pytest.raises(RequestError, match="of type 'image'"):
        wire.claim({"state": "<a>", "assets": {"<a>": {"type": "video", "part": "p0"}}}, {"p0": b"x"})
    with pytest.raises(RequestError, match="multipart part"):
        wire.claim({"state": "<a>", "assets": {"<a>": {"part": 3}}}, {"p0": b"x"})


@pytest.mark.parametrize(("assets", "parts", "message"), [
    (None, {"p0": b"x"}, "exactly one asset"),
    ({"<a>": {"part": "p0"}, "<b>": {"part": "p0"}}, {"p0": b"x"}, "exactly one asset"),
    ({"<a>": {"part": "p1"}}, {"p0": b"x"}, "exactly one asset"),
    ({"<a>": {"part": "p1"}}, {}, "does not have"),
])
def test_claim_rules(assets, parts, message):
    body = {"state": "s"} if assets is None else {"state": "s", "assets": assets}
    with pytest.raises(ValueError, match=message):
        wire.claim(body, parts)


@pytest.mark.parametrize(("coding", "compress"), [("gzip", gzip.compress), ("zstd", zstd.compress)])
def test_more_members_or_frames_than_the_limit_are_refused(coding, compress):
    """Each gzip member or zstd frame copies the rest of the body: their count is capped, so time stays linear."""
    assert wire.inflate(compress(b"x") * wire.MAX_MEMBERS, coding, LIMIT) == b"x" * wire.MAX_MEMBERS
    with pytest.raises(ValueError, match="members or zstd frames") as caught:
        wire.inflate(compress(b"x") * (wire.MAX_MEMBERS + 1), coding, LIMIT)
    assert not isinstance(caught.value, (wire.TooLarge, wire.UnsupportedEncoding))
