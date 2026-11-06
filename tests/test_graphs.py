"""Which packs the CUDA-graph path takes and what it answers, on CPU, where no bucket has a graph and the same padded
pass runs eagerly: buckets; single-block sequences padded to their bucket against the plain forward on the tiny random
Gemma 4 of ``test_gemma_blocks``; branched, several-sequence and too-long packs run the plain forward."""

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
    layouts = [[sequence(13, [])], [sequence(128, [])], [sequence(129, [])],  # padded: 128, 128 (no padding), 256
               [sequence(300, [])], [sequence(11, [5, 7])], [sequence(9, []), sequence(5, [])]]  # plain
    with torch.no_grad():
        for seqs in layouts:
            want = plain_forward(pack(seqs, None, CPU))
            torch.testing.assert_close(g(pack(seqs, None, CPU)), want, atol=1e-10, rtol=1e-9)
    assert len(calls) == 3
