"""Which opt-in serving paths a checkpoint's backbone runs (``serve.paths``), read before any weights load."""

import json

import pytest

from karpo_decision.config import DecisionConfig
from karpo_decision.serve import paths


def checkpoint(directory, model_type, backbone=None):
    DecisionConfig(template="karpo-blocks-v1", codes=("A",), token_ids=(1,), attention="bidirectional",
                   base_model="base", backbone=backbone).save(directory)
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(directory)


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_gemma_runs_every_path(tmp_path, model_type):
    assert paths.require(checkpoint(tmp_path, model_type), fp8=True, triton_attention=True) == {
        "fp8", "graphs", "triton_attention"}


def test_an_unlisted_backbone_runs_none(tmp_path):
    path = checkpoint(tmp_path, "llama")
    assert paths.require(path) == frozenset()
    with pytest.raises(ValueError, match="fp8 is not supported for llama checkpoints"):
        paths.require(path, fp8=True)
    with pytest.raises(ValueError, match="triton_attention is not supported for llama checkpoints"):
        paths.require(path, triton_attention=True)


def test_an_untrained_checkpoint_reads_its_backbone_directory(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "gemma4_unified"}))
    out = tmp_path / "out"
    out.mkdir()
    DecisionConfig(template="karpo-blocks-v1", codes=("A",), token_ids=(1,), attention="causal",
                   base_model="base", backbone=str(base)).save(out)
    assert "fp8" in paths.require(str(out), fp8=True)
