"""Qwen3.5 serving paths on CPU: which projections FP8 replaces."""

import importlib
import os

import pytest
import torch

PATH = os.environ.get("KARPO_TOKENIZER")
needs_config = pytest.mark.skipif(not PATH, reason="set KARPO_TOKENIZER to a Qwen3.5 checkpoint directory")


def tiny_backbone():
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    config = AutoConfig.from_pretrained(PATH)
    text = config.text_config
    text.num_hidden_layers, text.layer_types = 4, text.layer_types[:4]
    text.hidden_size, text.intermediate_size = 256, 512
    config.vision_config.depth = 1
    return Qwen3_5Model(config).eval()


@needs_config
def test_fp8_replaces_the_wide_projections_only(monkeypatch):
    from karpo_decision.families.qwen import Packed
    from karpo_decision.serve import qwen_fp8

    # Building Packed needs the CUDA kernels' package; the replacement never calls them.
    monkeypatch.setattr(importlib.import_module(Packed.__module__), "_kernels", lambda: (None, None, None))
    model = Packed(tiny_backbone(), "bidirectional")
    wide = [m for layer in model.layers for m in layer.modules()
            if isinstance(m, torch.nn.Linear) and m.out_features >= qwen_fp8.MIN_FEATURES]
    narrow = [m for layer in model.layers for m in layer.modules()
              if isinstance(m, torch.nn.Linear) and m.out_features < qwen_fp8.MIN_FEATURES]
    assert qwen_fp8.quantize(model) == len(wide) > 0
    left = [m for layer in model.layers for m in layer.modules() if isinstance(m, torch.nn.Linear)]
    assert len(left) == len(narrow) and all(m.out_features < qwen_fp8.MIN_FEATURES for m in left)


def test_buckets():
    from karpo_decision.serve.qwen_graphs import FINE_BUCKETS, bucket, sizes

    assert bucket(1, FINE_BUCKETS) == 64 and bucket(64, FINE_BUCKETS) == 64 and bucket(65, FINE_BUCKETS) == 128
    assert bucket(2049, FINE_BUCKETS) == 2176 and bucket(65537, FINE_BUCKETS) is None
    assert sizes(256, FINE_BUCKETS) == [64, 128, 192, 256]
    assert sizes(32768, FINE_BUCKETS)[-1] == 32768


def test_fast_does_nothing_off_cuda():
    import types

    from karpo_decision.serve import qwen_graphs

    decider = types.SimpleNamespace(device=torch.device("cpu"), model="packed", max_length=32768)
    assert qwen_graphs.fast(decider) is False and decider.model == "packed"
