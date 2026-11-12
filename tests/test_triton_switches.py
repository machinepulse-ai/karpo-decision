"""``triton_attention`` on CPU, no Triton needed: ``enable`` refuses a device off sm_120 and a causal model, else
installs ``attend``, which refuses a query range past the kernel's 32-bit offsets."""

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


def test_enable_installs_attend_on_a_bidirectional_model(monkeypatch):
    monkeypatch.setattr(ta, "supported", lambda device: True)
    model = stub("cpu")
    ta.enable(model)
    assert model.global_attend is ta.attend


def test_enable_refuses_a_causal_model(monkeypatch):
    monkeypatch.setattr(ta, "supported", lambda device: True)
    model = stub("cpu")
    model.causal_global = True
    with pytest.raises(ValueError, match="bidirectional"):
        ta.enable(model)
    assert model.global_attend is None


def test_attend_refuses_a_query_range_past_32_bit_offsets():
    """The 31B's query rows are 32 heads x 512 = 2^14 elements, so 131,072 rows reach 2^31 (meta tensors: no memory,
    GPU or Triton)."""
    rows = 131_072
    query = torch.empty(rows, 32, 512, dtype=torch.bfloat16, device="meta")
    key = value = torch.empty(8, 4, 512, dtype=torch.bfloat16, device="meta")
    bounds = torch.empty(2, dtype=torch.int32, device="meta")
    with torch.inference_mode(), pytest.raises(ValueError, match="32-bit"):
        ta.attend(query, key, value, bounds, bounds, rows, 1.0)
