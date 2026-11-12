"""Which path ``serve.server`` loads a checkpoint through: a Gemma 4 checkpoint on one GPU, plain by default, FP8
projections with ``fp8``, ``triton_attention`` after the quantization, before the graphs, then CUDA graphs unless
``fast`` is false, warmed up with the graphs' warmup request (text only). A Qwen3.5 checkpoint warms up with a
screenshot and is refused the opt-in paths. How it reads its ``--max-length``. No GPU and no weights: the loader, the
quantization and the capture are replaced."""

import json
import types

import pytest

from karpo_decision.config import DecisionConfig
from karpo_decision.decider import DEFAULT_MAX_LENGTH
from karpo_decision.protocol import images_of
from karpo_decision.serve import paths, server


def checkpoint(directory, model_type, template="karpo-blocks-v1"):
    DecisionConfig(template=template, codes=("A",), token_ids=(1,), attention="bidirectional",
                   base_model="base").save(directory)
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(directory)


def test_gemma_serves_on_one_gpu_through_the_plain_path(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    calls = []
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: calls.append((directory, device))
                                    or "plain"))
    decider, warmup = server.load_decider(path, 1, "cuda:0")
    assert decider == "plain" and calls == [(path, "cuda:0")]
    assert warmup is server.graphs.WARMUP_REQUEST and images_of(warmup)[1] == []


def test_gemma_refuses_more_than_one_gpu(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    monkeypatch.setattr(server.Decider, "from_pretrained", classmethod(lambda *_, **__: pytest.fail("loaded")))
    with pytest.raises(ValueError, match="one GPU"):
        server.load_decider(path, 2, "cuda:0")


def test_fp8_quantizes_the_decoder_projections_then_captures_graphs(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    decider = types.SimpleNamespace(model="packed")
    calls = []
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: decider))
    monkeypatch.setattr(server, "quantize", lambda model: calls.append(("fp8", model)))
    monkeypatch.setattr(server.graphs, "fast", lambda d: calls.append(("fast", d)) or True)
    assert server.load_decider(path, 1, "cuda:0", fp8=True)[0] is decider
    assert calls == [("fp8", "packed"), ("fast", decider)]


def test_fp8_without_fast_captures_no_graphs(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    decider = types.SimpleNamespace(model="packed")
    quantized = []
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: decider))
    monkeypatch.setattr(server, "quantize", lambda model: quantized.append(model))
    monkeypatch.setattr(server.graphs, "fast", lambda *_: pytest.fail("captured"))
    assert server.load_decider(path, 1, "cuda:0", fp8=True, fast=False)[0] is decider
    assert quantized == ["packed"]


def test_without_fp8_no_graphs_are_captured(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: "plain"))
    monkeypatch.setattr(server.graphs, "fast", lambda *_: pytest.fail("captured"))
    assert server.load_decider(path, 1, "cuda:0")[0] == "plain"


def test_triton_attention_comes_after_the_quantization_and_before_the_graphs(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    decider = types.SimpleNamespace(model="packed")
    calls = []
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: decider))
    monkeypatch.setattr(server, "quantize", lambda model: calls.append(("fp8", model)))
    monkeypatch.setattr(server, "enable_attention",
                        lambda model, max_length: calls.append(("attention", model, max_length)))
    monkeypatch.setattr(server.graphs, "fast", lambda d: calls.append(("fast", d)) or True)
    assert server.load_decider(path, 1, "cuda:0", fp8=True, triton_attention=True)[0] is decider
    assert calls == [("fp8", "packed"), ("attention", "packed", DEFAULT_MAX_LENGTH), ("fast", decider)]


def test_triton_attention_without_fp8(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    decider = types.SimpleNamespace(model="packed")
    calls = []
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: decider))
    monkeypatch.setattr(server, "enable_attention",
                        lambda model, max_length: calls.append(("attention", model, max_length)))
    monkeypatch.setattr(server.graphs, "fast", lambda *_: pytest.fail("captured"))
    assert server.load_decider(path, 1, "cuda:0", 65_536, triton_attention=True)[0] is decider
    assert calls == [("attention", "packed", 65_536)]


def test_gemma_reads_sequences_as_long_as_the_server_is_told(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    lengths = []
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", max_length=None, **_:
                                    lengths.append(max_length) or "plain"))
    server.load_decider(path, 1, "cuda:0")
    server.load_decider(path, 1, "cuda:0", 65_536)
    assert lengths == [DEFAULT_MAX_LENGTH, 65_536]


def test_the_server_takes_its_longest_sequence_from_max_length():
    assert server.arguments(["--checkpoint", "c"]).max_length == DEFAULT_MAX_LENGTH
    assert server.arguments(["--checkpoint", "c", "--max-length", "65536"]).max_length == 65_536


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_the_server_refuses_a_max_length_that_is_not_a_positive_integer(value, capsys):
    with pytest.raises(SystemExit):
        server.arguments(["--checkpoint", "c", "--max-length", value])
    err = capsys.readouterr().err
    assert "argument --max-length" in err and "is not a positive integer" in err


def test_a_path_the_backbone_does_not_run_is_refused_before_loading(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "llama")
    monkeypatch.setattr(server.Decider, "from_pretrained", classmethod(lambda *_, **__: pytest.fail("loaded")))
    with pytest.raises(ValueError, match="fp8 is not supported for llama checkpoints"):
        server.load_decider(path, 1, "cuda:0", fp8=True)


def test_fp8_without_graphs_for_the_backbone_captures_none(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4_unified")
    decider = types.SimpleNamespace(model="packed")
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: decider))
    monkeypatch.setattr(server, "quantize", lambda model: None)
    monkeypatch.setitem(paths.PATHS, "gemma4_unified", frozenset({"fp8", "triton_attention"}))
    monkeypatch.setattr(server.graphs, "fast", lambda *_: pytest.fail("captured"))
    assert server.load_decider(path, 1, "cuda:0", fp8=True)[0] is decider


def test_qwen_warms_up_with_a_screenshot(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    path = checkpoint(tmp_path, "qwen3_5", template="karpo-joint-preview-v1")
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: "plain"))
    decider, warmup = server.load_decider(path, 1, "cuda:0")
    assert decider == "plain" and len(images_of(warmup)[1]) == 1


@pytest.mark.parametrize("option", ["fp8", "triton_attention"])
def test_qwen_refuses_the_opt_in_paths(tmp_path, monkeypatch, option):
    path = checkpoint(tmp_path, "qwen3_5", template="karpo-joint-preview-v1")
    monkeypatch.setattr(server.Decider, "from_pretrained", classmethod(lambda *_, **__: pytest.fail("loaded")))
    with pytest.raises(ValueError, match=f"{option} is not supported for qwen3_5 checkpoints"):
        server.load_decider(path, 1, "cuda:0", **{option: True})
