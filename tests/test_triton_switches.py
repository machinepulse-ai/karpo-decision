"""``triton_attention`` on CPU, no Triton needed: ``enable`` refuses a device off sm_120, a causal model and a
``max_length`` past the kernel's 32-bit offsets, else installs ``attend``, which refuses a query range past them."""

import types

import pytest
import torch

from karpo_decision.families.gemma import triton_attention as ta


def stub(device):
    """The 31B's shape where ``enable`` reads it: a sliding layer's query rows are 32 heads x 256, a global one's
    32 x 512."""
    weight = torch.zeros(1, device=device)
    layers = [types.SimpleNamespace(self_attn=types.SimpleNamespace(q_proj=types.SimpleNamespace(out_features=width)))
              for width in (32 * 256, 32 * 512)]
    return types.SimpleNamespace(lm=types.SimpleNamespace(embed_tokens=types.SimpleNamespace(weight=weight),
                                                          layers=layers),
                                 types=["sliding_attention", "full_attention"], causal_global=False, global_attend=None)


def test_triton_attention_needs_sm_120():
    model = stub("cpu")
    with pytest.raises(ValueError, match="sm_120"):
        ta.enable(model, 32_768)
    assert model.global_attend is None


def test_enable_installs_attend_on_a_bidirectional_model(monkeypatch):
    monkeypatch.setattr(ta, "supported", lambda device: True)
    model = stub("cpu")
    ta.enable(model, 32_768)
    assert model.global_attend is ta.attend


def test_enable_refuses_a_causal_model(monkeypatch):
    monkeypatch.setattr(ta, "supported", lambda device: True)
    model = stub("cpu")
    model.causal_global = True
    with pytest.raises(ValueError, match="bidirectional"):
        ta.enable(model, 32_768)
    assert model.global_attend is None


def test_enable_refuses_a_max_length_past_32_bit_offsets(monkeypatch):
    """A global query row of 2^14 elements: a sequence of 131,072 tokens reaches 2^31."""
    monkeypatch.setattr(ta, "supported", lambda device: True)
    model = stub("cpu")
    with pytest.raises(ValueError, match="up to 131071 tokens"):
        ta.enable(model, 131_072)
    assert model.global_attend is None
    ta.enable(model, 131_071)
    assert model.global_attend is ta.attend


def test_enable_reads_the_global_layers_query_rows_of_a_gemma_4_model(monkeypatch):
    """A tiny Gemma 4 text model: 4 heads of 8 in the sliding layers, of 16 in the global ones, so the limit is
    (2^31 - 1) // 64 tokens."""
    pytest.importorskip("transformers.models.gemma4")
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    from karpo_decision.families.gemma.packed import Packed

    config = Gemma4TextConfig(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                              num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=16,
                              num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=8,
                              layer_types=["sliding_attention", "full_attention"], hidden_size_per_layer_input=0)
    model = Packed(types.SimpleNamespace(language_model=Gemma4TextModel(config)), "bidirectional")
    monkeypatch.setattr(ta, "supported", lambda device: True)
    with pytest.raises(ValueError, match=f"up to {(2**31 - 1) // 64} tokens"):
        ta.enable(model, (2**31 - 1) // 64 + 1)
    ta.enable(model, (2**31 - 1) // 64)
    assert model.global_attend is ta.attend


def test_attend_refuses_a_query_range_past_32_bit_offsets():
    """The 31B's query rows are 32 heads x 512 = 2^14 elements, so 131,072 rows reach 2^31 (meta tensors: no memory,
    GPU or Triton)."""
    rows = 131_072
    query = torch.empty(rows, 32, 512, dtype=torch.bfloat16, device="meta")
    key = value = torch.empty(8, 4, 512, dtype=torch.bfloat16, device="meta")
    bounds = torch.empty(2, dtype=torch.int32, device="meta")
    with torch.inference_mode(), pytest.raises(ValueError, match="32-bit"):
        ta.attend(query, key, value, bounds, bounds, rows, 1.0)
