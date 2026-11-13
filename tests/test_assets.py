"""``assets.resolve``: a request's inline images (TypeSafe's assets, OpenRouter's image parts), the text cut at them,
and the rules a request breaks (``AssetError``, HTTP 400)."""

import base64

import pytest

from karpo_decision.assets import AssetError, resolve
from karpo_decision.protocol import describe

DATA = b"\x89PNG not decoded here"
URL = "data:image/png;base64," + base64.b64encode(DATA).decode()
PART = {"type": "image_url", "image_url": {"url": URL}}
QUESTION = {"type": "noul", "instructions": "Is the dialog open?"}


def asset(data=DATA):
    return {"type": "image", "format": "png", "data_base64": base64.b64encode(data).decode()}


def request(state, questions=None, assets=None):
    body = {"state": state, "questions": questions or {"q": QUESTION}}
    if assets is not None:
        body["assets"] = assets
    return body


def test_a_request_without_inline_images_is_unchanged():
    r = resolve(request({"page": "x"}))
    assert r.inline == [] and r.state == {"page": "x"} and r.questions == {"q": QUESTION}
    assert r.pieces("any text") == ["any text"]


def test_a_key_as_a_whole_string_and_inside_one_reads_the_asset_once():
    r = resolve(request({"a": "<k>", "b": "see <k> here"}, assets={"<k>": asset()}))
    assert r.inline == [DATA, DATA] and r.inline[0] is r.inline[1]
    assert r.pieces(r.state["a"]) == [0]
    assert r.pieces(r.state["b"]) == ["see ", 1, " here"]


def test_the_longest_key_wins_and_matches_do_not_overlap():
    r = resolve(request("img2 img img", assets={"img": asset(b"1"), "img2": asset(b"2")}))
    assert r.pieces(r.state) == [0, " ", 1, " ", 2]
    assert r.inline == [b"2", b"1", b"1"]


def test_object_keys_are_never_searched():
    with pytest.raises(AssetError, match="not referenced"):
        resolve(request({"<k>": "text"}, assets={"<k>": asset()}))


def test_instructions_and_every_kind_of_criteria_are_searched():
    questions = {"c": {"type": "choice", "instructions": ["Pick", "<a>"], "criteria": {"<x>": "<b>", "y": None}},
                 "n": {"type": "noul", "criteria": {"true": {"look": "<c>"}}},
                 "s": {"type": "score", "criteria": ["low", "<d>"]}}
    keys = {k: asset(k.encode()) for k in ("<a>", "<b>", "<c>", "<d>")}
    r = resolve(request("state", questions, keys))
    assert r.inline == [b"<a>", b"<b>", b"<c>", b"<d>"]
    assert list(r.questions["c"]["criteria"]) == ["<x>", "y"] and r.questions["c"]["criteria"]["y"] is None
    assert r.pieces(describe(r.questions["s"]["criteria"][1])) == [3]


def test_image_parts_anywhere_are_inline_images():
    r = resolve(request({"shots": [PART, {"deep": PART}]}, {"q": {**QUESTION, "instructions": [PART, "Same?"]}}))
    assert r.inline == [URL, URL, URL]


def test_the_screenshot_and_images_stay_legacy_images():
    r = resolve({"state": {"viewport_screenshot": PART, "page": "x"}, "questions": {"q": QUESTION}, "images": [URL]})
    assert r.inline == [] and r.legacy == [URL, URL] and r.state == {"page": "x"}


def test_sentinels_survive_json_rendering():
    r = resolve(request({"door": "<k>"}, assets={"<k>": asset()}))
    assert r.pieces(describe(r.state)) == ['{"door": "', 0, '"}']


def test_raw_bytes_and_wrapped_base64():
    encoded = base64.b64encode(DATA).decode()
    wrapped = {"format": "png", "data_base64": encoded[:8] + "\n" + encoded[8:]}
    r = resolve(request("<a> <b>", assets={"<a>": {"data": b"raw"}, "<b>": wrapped}))
    assert r.inline == [b"raw", DATA]


@pytest.mark.parametrize(("body", "message"), [
    (request("text", assets={"<k>": asset()}), "not referenced"),
    (request("<k>", assets={"<k>": {"type": "image", "part": "asset-0"}}), "multipart"),
    (request("<k>", assets={"<k>": {"format": "png", "data_base64": "%%%"}}), "not base64"),
    (request({"shot": {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}}), "data:image"),
    (request({"shot": {"type": "image_url", "image_url": "data:image/png;base64,AA"}}), "data:image"),
])
def test_rule_breaks_are_asset_errors(body, message):
    with pytest.raises(AssetError, match=message):
        resolve(body)


class Tokenizer:
    """Just enough of a tokenizer for an ``Encoder`` that never tokenizes."""

    pad_token_id = 0
    all_special_tokens = ["<m>"]

    def encode(self, text, add_special_tokens=False):
        return [1]

    def convert_ids_to_tokens(self, token):
        return "<m>"

    def get_added_vocab(self):
        return {}


def test_a_joint_layout_refuses_inline_images():
    from types import SimpleNamespace

    from karpo_decision.encoding import Encoder, PlacementError
    from karpo_decision.templates import JointTemplate

    encoder = Encoder(SimpleNamespace(tokenizer=Tokenizer()), ("A", "B"), JointTemplate(), "<m>")
    with pytest.raises(PlacementError, match="viewport_screenshot"):
        encoder.plan(request("<k>", assets={"<k>": asset()}), None)
    with pytest.raises(PlacementError):
        encoder.plan(request({"shot": PART}), None)
