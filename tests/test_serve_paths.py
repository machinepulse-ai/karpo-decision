"""Which path ``serve.server`` loads a checkpoint through: a Gemma 4 checkpoint on its rank's GPU, plain by default, FP8
projections with ``fp8``, ``triton_attention`` after the quantization, before the graphs, then CUDA graphs unless
``fast`` is false, warmed up with the graphs' warmup request (text only). A Qwen3.5 checkpoint warms up with a
screenshot and runs FP8 and CUDA graphs through its own modules (``graphs`` alone: graphs on the bf16 path).
How it reads its ``--max-length``. No GPU and no weights: the loader, the
quantization and the capture are replaced."""

import json
import types

import pytest

from karpo_decision.config import DecisionConfig
from karpo_decision.decider import DEFAULT_MAX_LENGTH
from karpo_decision.protocol import images_of
from karpo_decision.serve import paths, server

QWEN_TEMPLATE = "karpo-joint-preview-v1"


def checkpoint(directory, model_type, template="karpo-blocks-v1"):
    DecisionConfig(template=template, codes=("A",), token_ids=(1,), attention="bidirectional",
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
    monkeypatch.setattr(server, "enable_attention",
                        lambda model, max_length: calls.append(("attention", model, max_length)))
    monkeypatch.setattr(server.graphs, "fast", lambda d: calls.append(("fast", d)) or True)
    assert server.load_decider(path, "cuda:0", fp8=True, triton_attention=True)[0] is decider
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
    assert server.load_decider(path, "cuda:0", 65_536, triton_attention=True)[0] is decider
    assert calls == [("attention", "packed", 65_536)]


def test_gemma_reads_sequences_as_long_as_the_server_is_told(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    lengths = []
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", max_length=None, **_:
                                    lengths.append(max_length) or "plain"))
    server.load_decider(path, "cuda:0")
    server.load_decider(path, "cuda:0", 65_536)
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
        server.load_decider(path, "cuda:0", fp8=True)


def test_fp8_without_graphs_for_the_backbone_captures_none(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4_unified")
    decider = types.SimpleNamespace(model="packed")
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: decider))
    monkeypatch.setattr(server, "quantize", lambda model: None)
    monkeypatch.setitem(paths.PATHS, "gemma4_unified", frozenset({"fp8", "triton_attention"}))
    monkeypatch.setattr(server.graphs, "fast", lambda *_: pytest.fail("captured"))
    assert server.load_decider(path, "cuda:0", fp8=True)[0] is decider


def test_qwen_warms_up_with_a_screenshot(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    path = checkpoint(tmp_path, "qwen3_5", template=QWEN_TEMPLATE)
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: "plain"))
    decider, warmup = server.load_decider(path, "cuda:0")
    assert decider == "plain" and len(images_of(warmup)[1]) == 1


def test_qwen_refuses_triton_attention(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "qwen3_5", template=QWEN_TEMPLATE)
    monkeypatch.setattr(server.Decider, "from_pretrained", classmethod(lambda *_, **__: pytest.fail("loaded")))
    with pytest.raises(ValueError, match="triton_attention is not supported for qwen3_5 checkpoints"):
        server.load_decider(path, "cuda:0", triton_attention=True)


def qwen_calls(tmp_path, monkeypatch):
    decider, calls = types.SimpleNamespace(model="packed"), []
    monkeypatch.setattr(server.Decider, "from_pretrained", classmethod(lambda cls, directory, **_: decider))
    monkeypatch.setattr(server.qwen_fp8, "quantize", lambda model: calls.append(("fp8", model)))
    monkeypatch.setattr(server.qwen_graphs, "fast", lambda d: calls.append(("graphs", d)) or True)
    monkeypatch.setattr(server.graphs, "fast", lambda *_: pytest.fail("Gemma graphs"))
    monkeypatch.setattr(server, "quantize", lambda *_: pytest.fail("Gemma FP8"))
    return checkpoint(tmp_path, "qwen3_5", template=QWEN_TEMPLATE), decider, calls


def test_qwen_graphs_capture_without_quantizing(tmp_path, monkeypatch):
    path, decider, calls = qwen_calls(tmp_path, monkeypatch)
    assert server.load_decider(path, "cuda:0", graphs=True)[0] is decider
    assert calls == [("graphs", decider)]


def test_qwen_graphs_without_fast_still_capture(tmp_path, monkeypatch):
    path, decider, calls = qwen_calls(tmp_path, monkeypatch)
    server.load_decider(path, "cuda:0", graphs=True, fast=False)
    assert calls == [("graphs", decider)]


def test_qwen_fp8_with_graphs_quantizes_then_captures_once(tmp_path, monkeypatch):
    path, decider, calls = qwen_calls(tmp_path, monkeypatch)
    server.load_decider(path, "cuda:0", fp8=True, graphs=True)
    assert calls == [("fp8", "packed"), ("graphs", decider)]


def test_qwen_fp8_quantizes_then_captures_graphs(tmp_path, monkeypatch):
    path, decider, calls = qwen_calls(tmp_path, monkeypatch)
    assert server.load_decider(path, "cuda:0", fp8=True)[0] is decider
    assert calls == [("fp8", "packed"), ("graphs", decider)]


def test_qwen_fp8_without_fast_only_quantizes(tmp_path, monkeypatch):
    path, decider, calls = qwen_calls(tmp_path, monkeypatch)
    server.load_decider(path, "cuda:0", fp8=True, fast=False)
    assert calls == [("fp8", "packed")]


def test_qwen_plain_captures_nothing(tmp_path, monkeypatch):
    path, decider, calls = qwen_calls(tmp_path, monkeypatch)
    server.load_decider(path, "cuda:0")
    assert calls == []


def test_gemma_refuses_bf16_graphs_before_loading(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    monkeypatch.setattr(server.Decider, "from_pretrained", classmethod(lambda *_, **__: pytest.fail("loaded")))
    with pytest.raises(ValueError, match="graphs is not supported for gemma4 checkpoints"):
        server.load_decider(path, "cuda:0", graphs=True)


def test_the_server_takes_graphs_from_its_flag():
    assert not server.arguments(["--checkpoint", "c"]).graphs
    assert server.arguments(["--checkpoint", "c", "--graphs"]).graphs


def test_runner_runs_one_request_on_its_device(monkeypatch):
    devices = []
    monkeypatch.setattr(server.torch.cuda, "set_device", devices.append)
    decider = types.SimpleNamespace(decide=lambda request, plans: {"request": request, "plans": plans})
    runner = server.Runner(decider, "cuda:3")
    assert runner.run({"q": {}}, ["sequence"]) == {"request": {"state": None, "questions": {"q": {}}},
                                                    "plans": [["sequence"]]}
    assert devices == ["cuda:3"]
