"""The Triton kernels' switches off sm_120, on CPU: ``triton_attention.enable`` refuses the device."""

import types

import pytest
import torch

from karpo_decision.families.gemma import triton_attention as ta


def stub(device):
    weight = torch.zeros(1, device=device)
    return types.SimpleNamespace(lm=types.SimpleNamespace(embed_tokens=types.SimpleNamespace(weight=weight)),
                                 causal_global=False, global_attend=None)


def test_triton_attention_needs_sm_120():
    model = stub("cpu")
    with pytest.raises(ValueError, match="sm_120"):
        ta.enable(model)
    assert model.global_attend is None
