import pytest

from karpo_decision.protocol import VIEWPORT_SCREENSHOT, RequestError, images_of, options, validate


def test_choice_options_keep_keys_and_describe_values():
    keys, texts = options({"type": "choice", "criteria": {"book": "Book now", "wait": None, "ask": {"why": "price"}}})
    assert keys == ["book", "wait", "ask"]
    assert texts == ["book: Book now", "wait", 'ask: {"why": "price"}']


def test_noul_options_default_descriptions():
    assert options({"type": "noul"}) == (["false", "true"], ["No / false", "Yes / true"])
    assert options({"type": "noul", "criteria": {"true": "Paid"}})[1] == ["No / false", "Paid"]


def test_score_options_are_level_indices():
    assert options({"type": "score", "criteria": ["bad", "ok", "good"]}) == (["0", "1", "2"], ["bad", "ok", "good"])


@pytest.mark.parametrize("request_, message", [
    ({"state": "s"}, "needs 'state' and 'questions'"),
    ({"state": "s", "questions": {}}, "at least one question"),
    ({"state": "s", "questions": {"q": {"type": "rank"}}}, "type must be"),
    ({"state": "s", "questions": {"q": {"type": "choice", "criteria": ["a"]}}}, "map option keys"),
    ({"state": "s", "questions": {"q": {"type": "score", "criteria": ["only"]}}}, "at least two levels"),
    ({"state": "s", "questions": {"q": {"type": "noul", "criteria": ["no", "yes"]}}}, "noul criteria"),
])
def test_validate_rejects(request_, message):
    with pytest.raises(RequestError, match=message):
        validate(request_)


def test_validate_rejects_too_many_options():
    criteria = {f"o{i}": None for i in range(256)}
    with pytest.raises(RequestError, match="256 options"):
        validate({"state": "s", "questions": {"q": {"type": "choice", "criteria": criteria}}})


def test_screenshot_is_lifted_out_of_the_state():
    state, images = images_of({"state": {"page": "x", VIEWPORT_SCREENSHOT: "data:image/jpeg;base64,AA"},
                               "questions": {}, "images": ["other"]})
    assert state == {"page": "x"}
    assert images == ["data:image/jpeg;base64,AA", "other"]


def test_screenshot_as_an_image_url_part_is_its_data_url():
    part = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA"}}
    state, images = images_of({"state": {"page": "x", VIEWPORT_SCREENSHOT: part}, "questions": {}, "images": [part]})
    assert state == {"page": "x"}
    assert images == ["data:image/jpeg;base64,AA", "data:image/jpeg;base64,AA"]


ASK = {"q": {"type": "noul"}}


@pytest.mark.parametrize(("assets", "message"), [
    ({}, "nonempty object"),
    ([], "nonempty object"),
    ({"": {"format": "png", "data_base64": "AA=="}}, "nonempty strings"),
    ({"k": "AA=="}, "of type 'image'"),
    ({"k": {"type": "video", "format": "png", "data_base64": "AA=="}}, "of type 'image'"),
    ({"k": {"type": "image"}}, "data_base64"),
    ({"k": {"type": "image", "data_base64": "AA=="}}, "format"),
    ({"k": {"type": "image", "format": "gif", "data_base64": "AA=="}}, "format"),
    ({"k": {"part": 3}}, "multipart part"),
])
def test_validate_rejects_assets(assets, message):
    with pytest.raises(RequestError, match=message):
        validate({"state": "k", "questions": ASK, "assets": assets})


@pytest.mark.parametrize("asset", [
    {"type": "image", "format": "jpg", "data_base64": "AA=="}, {"part": "asset-0"},
    {"type": "image", "part": "asset-0", "format": "nope"}, {"data": b"raw"}])
def test_validate_accepts_typesafe_assets(asset):
    validate({"state": "k", "questions": ASK, "assets": {"k": asset}})


def test_validate_accepts_no_assets():
    validate({"state": "k", "questions": ASK, "assets": None})


def test_a_box_question_is_an_unknown_type():
    with pytest.raises(RequestError, match="type must be"):
        validate({"state": "s", "questions": {"where": {"type": "box", "instructions": "Find the cat"}}})
