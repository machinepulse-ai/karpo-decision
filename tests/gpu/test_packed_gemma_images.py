"""The packed forward with images on a GPU, the real kernels (varlen sliding attention, the global layers' kernels,
and the Triton global attention where it runs), against the dense-mask Transformers reference of
``tests/test_gemma_images.py`` on the tiny model in bfloat16."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_gemma_images as cpu  # noqa: E402

from karpo_decision.families.gemma import packed as gp  # noqa: E402
from karpo_decision.families.gemma import triton_attention as ta  # noqa: E402
from karpo_decision.model.packed import pack  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")]
GPU = torch.device("cuda")


@pytest.fixture(scope="module")
def backbone():
    return cpu.tiny_backbone(cpu.WINDOW, types=("sliding_attention", "sliding_attention", "full_attention",
                                                "sliding_attention", "full_attention")).to(GPU, torch.bfloat16)


def on_gpu(s):
    return type(s)(**{k: (v.to(GPU) if torch.is_tensor(v) else v) for k, v in s.__dict__.items()})


@pytest.mark.parametrize("triton", [False, True])
@pytest.mark.parametrize("name", list(cpu.LAYOUTS))
def test_packed_images_on_the_gpu_stay_near_the_reference(backbone, name, triton):
    if triton and not ta.supported(GPU):
        pytest.skip("Triton global attention runs on sm_120 only")
    seqs = cpu.LAYOUTS[name]()
    packed = gp.Packed(backbone, "bidirectional")
    if triton:
        ta.enable(packed, 32_768)
    with torch.inference_mode():
        ours = packed(pack(seqs, None, GPU)).float()
        theirs = cpu.reference(backbone, [on_gpu(s) for s in seqs], True).float()
    assert ((ours - theirs).norm() / theirs.norm()) < 2e-2
