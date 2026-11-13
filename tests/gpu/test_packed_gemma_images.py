"""The packed forward with images on a GPU, on a tiny random Gemma 4 with a vision tower in bfloat16 (``build_tiny``'s
text sizes in ``tests/gpu/test_packed_gemma.py``): the real kernels (varlen sliding attention, the global layers'
kernels, and the Triton global attention where it runs) against the same code with reference kernels, on the same
weights and activations, as ``test_packed_gemma.py`` does for text. ``tests/test_gemma_images.py`` checks the reference
kernels against Transformers on CPU."""

import contextlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_gemma_images as cpu  # noqa: E402

from karpo_decision.families.gemma import packed as gp  # noqa: E402
from karpo_decision.families.gemma import sliding_attention  # noqa: E402
from karpo_decision.families.gemma import triton_attention as ta  # noqa: E402
from karpo_decision.model.packed import pack  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")]
GPU = torch.device("cuda")


@pytest.fixture(scope="module")
def backbone():
    # build_tiny's sizes: a global head of 64 lets the Triton kernel compile
    return cpu.tiny_backbone(cpu.WINDOW, types=("sliding_attention", "sliding_attention", "full_attention",
                                                "sliding_attention", "full_attention"),
                             hidden=64, intermediate=96, head_dim=32, global_head_dim=64).to(GPU, torch.bfloat16)


@pytest.mark.parametrize("triton", [False, True])
@pytest.mark.parametrize("name", list(cpu.LAYOUTS))
def test_real_kernels_keep_the_reference_contract_with_images(backbone, name, triton, monkeypatch):
    if triton and not ta.supported(GPU):
        pytest.skip("Triton global attention runs on sm_120 only")
    seqs = cpu.LAYOUTS[name]()

    def run(with_triton):
        packed = gp.Packed(backbone, "bidirectional")
        if with_triton:
            ta.enable(packed, 32_768)
        with torch.inference_mode():
            return packed(pack(seqs, None, GPU)).float()

    got = run(triton)
    with monkeypatch.context() as m:  # the same bf16 weights and activations; only the kernels differ
        m.setattr(sliding_attention, "varlen_attn", cpu.varlen_reference)
        m.setattr(sliding_attention, "_fa4", None)
        m.setattr(gp, "sdpa_kernel", lambda *_: contextlib.nullcontext())
        want = run(False)
    assert ((got - want).norm() / want.norm()) < 1e-2
