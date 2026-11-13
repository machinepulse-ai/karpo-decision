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


def test_gemma_runs_every_path(tmp_path):
    assert paths.require(checkpoint(tmp_path, "gemma4"), fp8=True, triton_attention=True) == {
        "fp8", "graphs", "triton_attention"}


def test_the_12b_runs_every_gemma_path(tmp_path):
    assert paths.require(checkpoint(tmp_path, "gemma4_unified"), fp8=True, triton_attention=True) == {
        paths.FP8, paths.GRAPHS, paths.TRITON_ATTENTION}


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
    (base / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    out = tmp_path / "out"
    out.mkdir()
    DecisionConfig(template="karpo-blocks-v1", codes=("A",), token_ids=(1,), attention="causal",
                   base_model="base", backbone=str(base)).save(out)
    assert "fp8" in paths.require(str(out), fp8=True)


def test_qwen_runs_fp8_and_graphs():
    assert paths.PATHS["qwen3_5"] == {paths.FP8, paths.GRAPHS, paths.BF16_GRAPHS}


def test_bf16_graphs_are_qwen_only(tmp_path):
    with pytest.raises(ValueError, match="graphs is not supported for gemma4 checkpoints"):
        paths.require(checkpoint(tmp_path, "gemma4"), graphs=True)


def test_path_names():
    assert [paths.path_name(f, c) for f, c in ((False, False), (False, True), (True, False), (True, True))] == [
        "plain", "graphs", "fp8", "fp8-graphs"]


def test_backbone_type(tmp_path):
    assert paths.backbone_type(checkpoint(tmp_path, "qwen3_5")) == "qwen3_5"
