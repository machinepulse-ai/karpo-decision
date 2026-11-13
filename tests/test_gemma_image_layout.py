"""Where ``blocks.py`` puts a request's images, with the 31B's tokenizer (``KARPO_GEMMA_TOKENIZER``: a directory with
its tokenizer, ``config.json`` and ``decision_config.json``; skipped without it): in the shared block after the context
header and before the state text, ``Image k:`` before each when there are several; once per request; repeated in every
part of a split request; counted in its length, the request failing at the first image past it."""

import os

import pytest
import torch

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from karpo_decision.config import DecisionConfig  # noqa: E402
from karpo_decision.encoding import Encoder, LengthError, TextOnlyError  # noqa: E402
from karpo_decision.families import gemma  # noqa: E402
from karpo_decision.templates import template  # noqa: E402

DIRECTORY = os.environ.get("KARPO_GEMMA_TOKENIZER")
pytestmark = pytest.mark.skipif(not DIRECTORY, reason="KARPO_GEMMA_TOKENIZER is not set")

QUESTION = {"type": "choice", "instructions": "Which button submits the form?",
            "criteria": {"save": "The Save button", "cancel": "The Cancel button"}}
OTHER = {"type": "noul", "instructions": "Is a dialog open?"}
STATE = {"page": {"url": "https://example.com/form", "title": "Form"}}


@pytest.fixture(scope="module")
def encoder():
    config = DecisionConfig.load(DIRECTORY)
    return Encoder(gemma.load_processor(DIRECTORY, 0), config.codes, template(config.template), gemma.MARKER)


def picture(width=640, height=480):
    return Image.new("RGB", (width, height), (120, 60, 200))


def test_one_question_with_an_image_reads_as_gemma_4_lays_images_out(encoder):
    layout, marks = encoder.blocks, encoder.processor.vision.tokens
    (part,) = encoder.plan({"state": STATE, "questions": {"q": QUESTION}, "images": [picture()]}, None)
    n = int((part.input_ids == marks.soft).sum())
    want = (layout.prefix + layout.header + layout.item + layout.instructions(QUESTION) + layout.newline
            + layout.context + [marks.begin] + [marks.soft] * n + [marks.end] + layout.newline + layout.text(STATE)
            + layout.branch(QUESTION))
    assert part.input_ids.tolist() == want
    assert [planned.tokens for planned in part.images] == [n] and part.pixel_values is None
    assert 0 < n <= 560


def test_several_images_each_follow_an_image_label(encoder):
    marks = encoder.processor.vision.tokens
    request = {"state": STATE, "questions": {"q": QUESTION}, "images": [picture(), picture(300, 900)]}
    (part,) = encoder.plan(request, None)
    ids = part.input_ids.tolist()
    assert ids.count(marks.begin) == ids.count(marks.end) == 2
    starts = [i for i, token in enumerate(ids) if token == marks.begin]
    for k, start in enumerate(starts, start=1):
        label = encoder.blocks.tokenize(f"Image {k}:\n")
        assert ids[start - len(label):start] == label


def test_the_viewport_screenshot_comes_first(encoder):
    import base64
    import io

    buffer = io.BytesIO()
    picture(32, 32).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    state = {**STATE, "viewport_screenshot": {"type": "image_url", "image_url": {"url": url}}}
    (part,) = encoder.plan({"state": state, "questions": {"q": QUESTION}, "images": [picture(300, 900)]}, None)
    first = encoder.processor.vision.prepare([url, picture(300, 900)])
    assert part.images[0].value == url
    assert torch.equal(encoder.processor.vision.read([part])[0].pixel_values, first.pixel_values)


def test_several_questions_read_the_images_once_in_the_shared_block(encoder):
    marks = encoder.processor.vision.tokens
    (part,) = encoder.plan({"state": STATE, "questions": {"a": QUESTION, "b": OTHER}, "images": [picture()]}, None)
    soft = part.input_ids == marks.soft
    assert int((part.input_ids == marks.begin).sum()) == 1
    assert bool((part.segment_ids[soft] == 0).all())


def test_a_split_request_repeats_its_images_in_every_part(encoder):
    marks = encoder.processor.vision.tokens
    request = {"state": STATE, "questions": {"a": QUESTION, "b": OTHER}, "images": [picture()]}
    (whole,) = encoder.plan(request, None)
    shared = int((whole.segment_ids == 0).sum())
    parts = encoder.plan(request, shared + max(int((whole.segment_ids == k).sum()) for k in (1, 2)))
    assert len(parts) == 2
    assert all(int((p.input_ids == marks.begin).sum()) == 1 for p in parts)
    assert parts[0].images[0] is parts[1].images[0]


def test_images_count_in_the_length(encoder):
    request = {"state": STATE, "questions": {"q": QUESTION}, "images": [picture()]}
    (part,) = encoder.plan(request, None)
    with pytest.raises(LengthError):
        encoder.plan(request, part.length - 1)


def test_images_past_the_limit_fail_before_the_rest_are_read(encoder, monkeypatch):
    """Even a tiny image is scaled up to the processor's budget: images are read one at a time, and the first whose
    tokens alone pass ``max_length`` fails the request."""
    n = encoder.processor.vision.measure(picture()).tokens
    read, measure = [], gemma.vision.Vision.measure

    def counted(self, value):
        read.append(value)
        return measure(self, value)

    monkeypatch.setattr(gemma.vision.Vision, "measure", counted)
    request = {"state": STATE, "questions": {"q": QUESTION}, "images": [picture()] * 40}
    with pytest.raises(LengthError, match="images need"):
        encoder.plan(request, 2 * n)  # one image with its markers fits, the second passes the limit
    assert len(read) == 2


def test_text_requests_lay_out_as_before(encoder):
    (part,) = encoder.plan({"state": STATE, "questions": {"q": QUESTION}}, None)
    layout = encoder.blocks
    assert part.input_ids.tolist() == layout.shared(STATE, [QUESTION]) + layout.branch(QUESTION)
    assert part.images is None and part.pixel_values is None and part.image_position_ids is None


def test_a_checkpoint_without_vision_refuses_images(encoder):
    text_only = Encoder(gemma.Processor(encoder.processor.tokenizer, None), encoder.codes, encoder.template,
                        gemma.MARKER)
    with pytest.raises(TextOnlyError):
        text_only.plan({"state": STATE, "questions": {"q": QUESTION}, "images": [picture()]}, None)
