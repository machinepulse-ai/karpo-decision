"""The CUDA-graph path on GPU with the real kernels, on a tiny random Gemma 4 in bf16: a replay equals the eager padded
pass bit for bit and stays within bf16 noise of the plain path; replays of other lengths in a bucket, and of a length
equal to its bucket, do not disturb each other; other packs run the plain path. With FP8 projections (``serve.fp8``),
a replay equals the eager padded pass bit for bit and stays near the plain FP8 forward."""

import sys
import types
from pathlib import Path

import pytest
import torch

from karpo_decision.families.gemma import packed as gp
from karpo_decision.model.packed import pack
from karpo_decision.serve import fp8, graphs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_gemma_blocks as cpu  # noqa: E402
from test_packed_gemma import build_tiny  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
GPU = torch.device("cuda")


def tiny(attention):
    torch.manual_seed(0)
    lm = build_tiny().to(GPU, torch.bfloat16)
    return gp.Packed(types.SimpleNamespace(language_model=lm), attention)


@pytest.mark.parametrize("attention", ["bidirectional", "causal"])
def test_replays_match_the_eager_padded_pass_and_the_plain_path(attention):
    packed = tiny(attention)
    eager = graphs.Graphs(packed, max_tokens=256)
    g = graphs.Graphs(packed, max_tokens=256)
    g.capture_all()
    assert sorted(g.graphs) == list(range(graphs.STEP, 257, graphs.STEP))
    torch.manual_seed(1)
    with torch.inference_mode():
        for length in (13, 128, 129, 150, 13):
            seqs = [cpu.sequence(length, [])]
            got = g(pack(seqs, None, GPU))
            assert torch.equal(got, eager(pack(seqs, None, GPU)))
            want = packed(pack(seqs, None, GPU)).float()
            assert ((got.float() - want).norm() / want.norm()) < 1e-2
    assert g.replays == 5


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
                    reason="needs FP8 tensor cores")
@pytest.mark.parametrize("attention", ["bidirectional", "causal"])
def test_fp8_replays_match_the_eager_padded_pass_and_the_plain_fp8_forward(attention):
    packed = tiny(attention)
    fp8.quantize(packed)
    eager = graphs.Graphs(packed, max_tokens=64)
    g = graphs.Graphs(packed, max_tokens=64)
    g.capture_all()
    assert sorted(g.graphs) == list(range(graphs.STEP, 65, graphs.STEP))
    torch.manual_seed(1)
    with torch.inference_mode():
        for length in (13, 32, 33, 50, 64, 13):
            seqs = [cpu.sequence(length, [])]
            got = g(pack(seqs, None, GPU))
            assert torch.equal(got, eager(pack(seqs, None, GPU)))
            want = packed(pack(seqs, None, GPU)).float()
            assert ((got.float() - want).norm() / want.norm()) < 5e-2
    assert g.replays == 6


def test_other_packs_run_the_plain_path():
    packed = tiny("bidirectional")
    g = graphs.Graphs(packed, max_tokens=256)
    g.capture_all()
    torch.manual_seed(2)
    with torch.inference_mode():
        for seqs in ([cpu.sequence(11, [5, 7])], [cpu.sequence(9, []), cpu.sequence(5, [])],
                     [cpu.sequence(300, [])]):
            assert torch.equal(g(pack(seqs, None, GPU)), packed(pack(seqs, None, GPU)))
    assert g.replays == 0
