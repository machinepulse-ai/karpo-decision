import importlib.util
import json

import pytest
import torch

from karpo_decision import families
from karpo_decision.config import DEFAULT_MARKER
from karpo_decision.model.readout import Readout


def _model_dir(tmp_path, model_type):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return tmp_path


def test_family_from_model_type(tmp_path):
    gemma = families.of(_model_dir(tmp_path, "gemma4"))
    assert gemma.MODEL_TYPE == "gemma4" and gemma.MARKER == DEFAULT_MARKER
    for name in ("load_processor", "load_backbone", "logit_cap", "Packed"):
        assert hasattr(gemma, name)


def test_unknown_model_type(tmp_path):
    with pytest.raises(ValueError, match="no model family"):
        families.of(_model_dir(tmp_path, "llama"))


def test_readout_cap():
    rows = torch.tensor([[10.0, 0.0], [0.0, -10.0]])
    hidden = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    plain = Readout.from_rows(rows)(hidden)
    capped = Readout.from_rows(rows, cap=30.0)(hidden)
    assert torch.equal(plain, hidden @ rows.t())
    assert torch.allclose(capped, 30.0 * torch.tanh(plain / 30.0))
    assert capped.abs().max() < 30.0


def test_gemma4_unified_is_the_gemma_family(tmp_path):
    gemma = families.of(_model_dir(tmp_path, "gemma4_unified"))
    assert gemma.MODEL_TYPE == "gemma4" and set(gemma.BACKBONES) == {"gemma4", "gemma4_unified"}


def test_model_type_reads_the_backbone_config(tmp_path):
    assert families.model_type(_model_dir(tmp_path, "gemma4_unified")) == "gemma4_unified"


def test_qwen3_5_is_the_qwen_family(tmp_path):
    qwen = families.of(_model_dir(tmp_path, "qwen3_5"))
    assert qwen.MODEL_TYPE == "qwen3_5" and qwen.MARKER == "<|fim_suffix|>"
    assert qwen.TEMPLATES == ("karpo-joint-v1", "karpo-joint-preview-v1")
    for name in ("load_processor", "load_backbone", "logit_cap", "Packed"):
        assert hasattr(qwen, name)


def test_qwen_processor_needs_the_extra(monkeypatch):
    from karpo_decision.families import qwen

    real = importlib.util.find_spec
    monkeypatch.setattr(qwen.importlib.util, "find_spec",
                        lambda name, *a: None if name == "torchvision" else real(name, *a))
    with pytest.raises(ImportError, match=r"karpo-decision\[qwen\]"):
        qwen.load_processor("/nonexistent", 1)
