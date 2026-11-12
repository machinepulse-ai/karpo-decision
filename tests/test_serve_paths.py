"""Which path ``serve.server`` loads a checkpoint through: a Gemma 4 checkpoint on its rank's GPU, plain by default, FP8
projections with ``fp8``, ``triton_attention`` after the quantization, before the graphs, then CUDA graphs unless
``fast`` is false, warmed up with the graphs' warmup request (text only). No GPU and no weights: the loader, the
quantization and the capture are replaced."""

import json
import types

import pytest

from karpo_decision.config import DecisionConfig
from karpo_decision.protocol import images_of
from karpo_decision.serve import server


def checkpoint(directory, model_type):
    DecisionConfig(template="karpo-blocks-v1", codes=("A",), token_ids=(1,), attention="bidirectional",
                   base_model="base").save(directory)
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(directory)


def test_gemma_serves_through_the_plain_path(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    calls = []
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: calls.append((directory, device))
                                    or "plain"))
    decider, warmup = server.load_decider(path, "cuda:0")
    assert decider == "plain" and calls == [(path, "cuda:0")]
    assert warmup is server.graphs.WARMUP_REQUEST and images_of(warmup)[1] == []


def test_fp8_quantizes_the_decoder_projections_then_captures_graphs(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    decider = types.SimpleNamespace(model="packed")
    calls = []
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: decider))
    monkeypatch.setattr(server, "quantize", lambda model: calls.append(("fp8", model)))
    monkeypatch.setattr(server.graphs, "fast", lambda d: calls.append(("fast", d)) or True)
    assert server.load_decider(path, "cuda:0", fp8=True)[0] is decider
    assert calls == [("fp8", "packed"), ("fast", decider)]


def test_fp8_without_fast_captures_no_graphs(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    decider = types.SimpleNamespace(model="packed")
    quantized = []
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: decider))
    monkeypatch.setattr(server, "quantize", lambda model: quantized.append(model))
    monkeypatch.setattr(server.graphs, "fast", lambda *_: pytest.fail("captured"))
    assert server.load_decider(path, "cuda:0", fp8=True, fast=False)[0] is decider
    assert quantized == ["packed"]


def test_without_fp8_no_graphs_are_captured(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: "plain"))
    monkeypatch.setattr(server.graphs, "fast", lambda *_: pytest.fail("captured"))
    assert server.load_decider(path, "cuda:0")[0] == "plain"


def test_triton_attention_comes_after_the_quantization_and_before_the_graphs(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    decider = types.SimpleNamespace(model="packed")
    calls = []
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: decider))
    monkeypatch.setattr(server, "quantize", lambda model: calls.append(("fp8", model)))
    monkeypatch.setattr(server, "enable_attention", lambda model: calls.append(("attention", model)))
    monkeypatch.setattr(server.graphs, "fast", lambda d: calls.append(("fast", d)) or True)
    assert server.load_decider(path, "cuda:0", fp8=True, triton_attention=True)[0] is decider
    assert calls == [("fp8", "packed"), ("attention", "packed"), ("fast", decider)]


def test_triton_attention_without_fp8(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    decider = types.SimpleNamespace(model="packed")
    calls = []
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: decider))
    monkeypatch.setattr(server, "enable_attention", lambda model: calls.append(("attention", model)))
    monkeypatch.setattr(server.graphs, "fast", lambda *_: pytest.fail("captured"))
    assert server.load_decider(path, "cuda:0", triton_attention=True)[0] is decider
    assert calls == [("attention", "packed")]


def test_runner_runs_one_request_on_its_device(monkeypatch):
    devices = []
    monkeypatch.setattr(server.torch.cuda, "set_device", devices.append)
    decider = types.SimpleNamespace(decide=lambda request, plans: {"request": request, "plans": plans})
    runner = server.Runner(decider, "cuda:3")
    assert runner.run({"q": {}}, ["sequence"]) == {"request": {"state": None, "questions": {"q": {}}},
                                                    "plans": [["sequence"]]}
    assert devices == ["cuda:3"]
