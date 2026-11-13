"""Which packs the CUDA-graph path takes and what it answers, on CPU, where no bucket has a graph and the same padded
pass runs eagerly: buckets; single-block sequences padded to their bucket against the plain forward on the tiny random
Gemma 4 of ``test_gemma_blocks``; branched, several-sequence and too-long packs run the plain forward. ``fast`` on a
stub decider, the capture replaced: nothing off CUDA; otherwise the warmup request on the plain path, the capture, then
the request again through the graphs."""

import contextlib
import types

import pytest
import torch

pytest.importorskip("transformers.models.gemma4")
from test_gemma_blocks import model, sequence  # noqa: E402, F401  (module-scoped fixture)
from test_gemma_padded import kernels  # noqa: E402, F401  (reference kernels)

from karpo_decision.families.gemma import packed as gp  # noqa: E402
from karpo_decision.model.packed import pack  # noqa: E402
from karpo_decision.serve import graphs  # noqa: E402

CPU = torch.device("cpu")


def test_buckets_are_multiples_of_the_step_up_to_the_top():
    assert graphs.bucket(1) == graphs.STEP
    assert graphs.bucket(graphs.STEP) == graphs.STEP
    assert graphs.bucket(graphs.STEP + 1) == 2 * graphs.STEP
    assert graphs.bucket(graphs.MAX_TOKENS) == graphs.MAX_TOKENS
    assert graphs.bucket(graphs.MAX_TOKENS + 1) is None
    assert graphs.bucket(300, top=256) is None


def test_only_one_single_block_sequence_within_the_top_is_covered():
    torch.manual_seed(0)
    assert graphs.covered(pack([sequence(13, [])], None, CPU))
    assert not graphs.covered(pack([sequence(11, [5, 7])], None, CPU))
    assert not graphs.covered(pack([sequence(13, []), sequence(9, [])], None, CPU))
    assert not graphs.covered(pack([sequence(13, [])], None, CPU), top=12)


@pytest.mark.parametrize("attention", ["bidirectional", "causal"])
def test_graphs_answer_like_the_plain_forward(model, kernels, attention):  # noqa: F811
    torch.manual_seed(4)
    packed = gp.Packed(types.SimpleNamespace(language_model=model), attention)
    g = graphs.Graphs(packed, max_tokens=256)
    plain_forward, calls = packed.forward, []
    packed.forward = lambda p: calls.append(p) or plain_forward(p)
    layouts = [[sequence(13, [])], [sequence(128, [])], [sequence(129, [])],  # padded: 32, 128 (no padding), 160
               [sequence(300, [])], [sequence(11, [5, 7])], [sequence(9, []), sequence(5, [])]]  # plain
    with torch.no_grad():
        for seqs in layouts:
            want = plain_forward(pack(seqs, None, CPU))
            torch.testing.assert_close(g(pack(seqs, None, CPU)), want, atol=1e-10, rtol=1e-9)
    assert len(calls) == 3


def test_fast_keeps_the_plain_path_off_cuda():
    decided = []
    decider = types.SimpleNamespace(device=CPU, max_length=32768, model="packed", decide=decided.append)
    assert graphs.fast(decider) is False
    assert decider.model == "packed" and decided == []


@pytest.mark.parametrize("max_length", [1000, 32768])
def test_fast_answers_the_warmup_request_before_and_after_capture(monkeypatch, max_length):
    events = []
    packed = types.SimpleNamespace(lm=types.SimpleNamespace(embed_tokens=torch.nn.Embedding(1, 8)))
    decider = types.SimpleNamespace(device=torch.device("cuda"), max_length=max_length, model=packed)
    decider.decide = lambda request: events.append(("decide", request, decider.model))
    monkeypatch.setattr(graphs, "kernels_supported", lambda device: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(graphs.Graphs, "capture_all", lambda self: events.append("capture"))
    assert graphs.fast(decider) is True
    g = decider.model
    assert isinstance(g, graphs.Graphs) and g.model is packed and g.max_tokens == min(graphs.MAX_TOKENS, max_length)
    assert events == [("decide", graphs.WARMUP_REQUEST, packed), "capture", ("decide", graphs.WARMUP_REQUEST, g)]


def test_a_pack_with_images_is_not_covered():
    s = sequence(13, [])
    with_images = type(s)(**{**s.__dict__, "pixel_values": torch.zeros(1, 4, 3),
                             "image_position_ids": torch.zeros(1, 4, 2, dtype=torch.long)})
    assert not graphs.covered(pack([with_images], None, CPU))
