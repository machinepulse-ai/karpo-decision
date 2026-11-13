"""The packed forward with images against Transformers' own text model on the same input embeddings, on a tiny random
Gemma 4 with a vision tower on CPU (float64): rule A's masks, and each image's soft tokens seeing their whole image in
the sliding layers (the window is longer than an image, as in the 31B). One block and branches, two images, text
before an image longer than the window, two parts of one request in one pack."""

import contextlib

import pytest
import torch

pytest.importorskip("transformers.models.gemma4")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402
from test_gemma_blocks import rule_a, varlen_reference  # noqa: E402
from test_gemma_vision import SMALL, TOKENS, tiny_backbone  # noqa: E402

from karpo_decision.encoding import Encoded  # noqa: E402
from karpo_decision.families.gemma import packed as gp  # noqa: E402
from karpo_decision.families.gemma import sliding_attention, vision  # noqa: E402
from karpo_decision.model.packed import pack  # noqa: E402

WINDOW = 96  # longer than any image at 70 soft tokens
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def backbone():
    return tiny_backbone(WINDOW, types=("sliding_attention", "sliding_attention", "full_attention",
                                        "sliding_attention", "full_attention"))


@pytest.fixture
def cpu_kernels(monkeypatch):
    monkeypatch.setattr(sliding_attention, "varlen_attn", varlen_reference)
    monkeypatch.setattr(gp, "sdpa_kernel", lambda *_: contextlib.nullcontext())


def images(*sizes):
    return vision.Vision(TOKENS, max_soft_tokens=SMALL).prepare(
        [Image.new("RGB", size, (37 * i % 255, 90, 160)) for i, size in enumerate(sizes, start=1)])


def sequence(before: int, prepared, after: int, branches=(), vocab=280):
    """A block layout: ``before`` text tokens, each image as begin / soft x n / end, ``after`` text tokens, then the
    branches; one question when there are no branches."""
    torch.manual_seed(before + after + len(branches))
    ids = torch.randint(3, vocab, (before,)).tolist()
    for n in prepared.tokens:
        ids += [TOKENS.begin, *[TOKENS.soft] * n, TOKENS.end, int(torch.randint(3, vocab, (1,)))]
    ids += torch.randint(3, vocab, (after,)).tolist()
    shared = len(ids)
    pos, seg, markers = list(range(shared)), [0] * shared, []
    if not branches:
        markers = [shared - 1]
    for k, n in enumerate(branches, start=1):
        ids += torch.randint(3, vocab, (n,)).tolist()
        pos += range(shared, shared + n)
        seg += [k] * n
        markers.append(len(ids) - 1)
    keys = ("0",) if not branches else tuple(str(i) for i in range(len(branches)))
    return Encoded(keys=keys, counts=(2,) * len(keys), input_ids=torch.tensor(ids), markers=torch.tensor(markers),
                   positions=torch.tensor(pos), segment_ids=torch.tensor(seg, dtype=torch.int16),
                   pixel_values=prepared.pixel_values, image_position_ids=prepared.position_ids)


def text_only(n: int):
    torch.manual_seed(n)
    return Encoded(keys=("t",), counts=(2,), input_ids=torch.randint(3, 280, (n,)), markers=torch.tensor([n - 1]),
                   positions=torch.arange(n), segment_ids=torch.zeros(n, dtype=torch.int16))


def reference(model, seqs, bidir):
    """Transformers' text model, one sequence at a time, on input embeddings with the image features written in; rule
    A's masks, and in the sliding layers every soft token also sees its whole image."""
    lm, outs = model.language_model, []
    for s in seqs:
        seg, pos, ids = s.segment_ids.long(), s.positions, s.input_ids
        soft = ids == TOKENS.soft
        h = lm.embed_tokens(torch.where(soft, torch.zeros_like(ids), ids))
        if s.pixel_values is not None:
            h[soft] = vision.embed(model, s.pixel_values, s.image_position_ids).to(h.dtype)
        image = torch.cumsum((ids == TOKENS.begin).long(), 0).masked_fill(~soft, -1)
        same = (image[:, None] == image[None, :]) & (image[:, None] >= 0)
        masks = {"full_attention": rule_a(seg, pos, bidir=bidir),
                 "sliding_attention": rule_a(seg, pos, WINDOW) | same[None, None]}
        out = lm(inputs_embeds=h[None], position_ids=pos[None], attention_mask=masks, use_cache=False)
        outs.append(out.last_hidden_state[0, s.markers])
    return torch.cat(outs)


LAYOUTS = {
    "one-block": lambda: [sequence(12, images((640, 480)), 9)],
    "branches": lambda: [sequence(7, images((640, 480)), 5, branches=(6, 9, 4))],
    "two-images": lambda: [sequence(5, images((640, 480), (300, 900)), 8, branches=(5,))],
    "text-past-the-window": lambda: [sequence(WINDOW + 30, images((500, 500)), 4)],
    "two-parts-and-text": lambda: [sequence(6, images((640, 480)), 3, branches=(5,)),
                                   sequence(6, images((640, 480)), 3, branches=(7,)), text_only(15)],
}


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_packed_images_match_transformers(backbone, cpu_kernels, name):
    seqs = LAYOUTS[name]()
    packed = gp.Packed(backbone, "bidirectional")
    with torch.inference_mode():
        ours = packed(pack(seqs, None, CPU))
        theirs = reference(backbone, seqs, bidir=True)
    torch.testing.assert_close(ours, theirs, atol=1e-10, rtol=1e-9)


def test_the_images_change_the_answer(backbone, cpu_kernels):
    """The soft tokens' rows carry the images: another picture of the same size gives another readout."""
    a = sequence(6, images((640, 480)), 4)
    other = vision.Vision(TOKENS, max_soft_tokens=SMALL).prepare([Image.new("RGB", (640, 480), (250, 250, 10))])
    b = Encoded(**{**a.__dict__, "pixel_values": other.pixel_values, "image_position_ids": other.position_ids})
    packed = gp.Packed(backbone, "bidirectional")
    with torch.inference_mode():
        assert not torch.allclose(packed(pack([a], None, CPU)), packed(pack([b], None, CPU)))


def test_images_need_a_bidirectional_checkpoint(backbone, cpu_kernels):
    """A causal checkpoint's global layers would not let a soft token see the rest of its image."""
    with pytest.raises(ValueError, match="bidirectional"):
        gp.Packed(backbone, "causal")(pack([sequence(6, images((640, 480)), 4)], None, CPU))


def test_an_image_longer_than_the_window_is_refused(cpu_kernels):
    small = tiny_backbone(16)
    with pytest.raises(ValueError, match="sliding window"):
        gp.Packed(small, "bidirectional")(pack([sequence(3, images((640, 480)), 3)], None, CPU))


def test_soft_tokens_and_features_must_agree(backbone, cpu_kernels):
    s = sequence(6, images((640, 480)), 4)
    short = Encoded(**{**s.__dict__, "input_ids": torch.where(s.input_ids == TOKENS.soft, 5, s.input_ids)})
    short.input_ids[7] = TOKENS.soft
    with pytest.raises(ValueError, match="soft tokens"):
        gp.Packed(backbone, "bidirectional")(pack([short], None, CPU))
