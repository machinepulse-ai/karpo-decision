"""FP8 projections on GPU: an ``Fp8Linear`` stays near its bf16 ``Linear``; ``quantize`` replaces every decoder
projection of a tiny random Gemma 4, each staying near the bf16 ``Linear`` it replaced on the input that ``Linear``
gets in a packed forward, and the quantized forward runs. Its output is not compared with bf16's: the random model
amplifies any rounding."""

import sys
import types
from pathlib import Path

import pytest
import torch

from karpo_decision.families.gemma import packed as gp
from karpo_decision.model.packed import pack
from karpo_decision.serve import fp8

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_gemma_blocks as cpu  # noqa: E402
from test_packed_gemma import build_tiny  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9), reason="needs FP8 tensor cores")]
GPU = torch.device("cuda")


def test_fp8_linear_stays_near_bf16():
    torch.manual_seed(0)
    linear = torch.nn.Linear(512, 1024, bias=False, device=GPU, dtype=torch.bfloat16)
    x = torch.randn(37, 512, device=GPU, dtype=torch.bfloat16)
    want = linear(x).float()
    got = fp8.Fp8Linear(linear)(x).float()
    assert ((got - want).norm() / want.norm()) < 0.05


def test_quantize_replaces_every_projection_with_one_near_its_bf16_linear():
    torch.manual_seed(0)
    lm = build_tiny().to(GPU, torch.bfloat16)
    packed = gp.Packed(types.SimpleNamespace(language_model=lm), "bidirectional")
    torch.manual_seed(1)
    seqs = [cpu.sequence(13, []), cpu.sequence(11, [5, 7])]
    linears = {path: m for path, m in lm.named_modules() if path.rsplit(".", 1)[-1] in fp8.PROJECTIONS}
    inputs = {}  # each bf16 projection's input in the packed forward

    def keep(module, args):
        inputs[module] = args[0]

    hooks = [m.register_forward_pre_hook(keep) for m in linears.values()]
    with torch.inference_mode():
        want = packed(pack(seqs, None, GPU)).float()
    for hook in hooks:
        hook.remove()
    sliding = sum(t == gp.SLIDING for t in cpu.TYPES)
    assert fp8.quantize(packed) == 7 * sliding + 6 * (len(cpu.TYPES) - sliding)  # global: one key-value projection
    with torch.inference_mode():
        for path, linear in linears.items():
            replaced, x = lm.get_submodule(path), inputs[linear]
            y = linear(x).float()
            assert isinstance(replaced, fp8.Fp8Linear), path
            assert ((replaced(x).float() - y).norm() / y.norm()) < 0.05, path
        got = packed(pack(seqs, None, GPU)).float()
    assert got.shape == want.shape and torch.isfinite(got).all()
