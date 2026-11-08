"""The Decision Index engine's path: plain by default, nothing answered at load; FP8 projections with ``fp8=true``, the
graphs' warmup request answered, then CUDA graphs unless ``fast=false`` or the graphs do not apply; a non-boolean
``fp8`` or ``fast`` is refused before loading. No GPU and no weights: the loader, the quantization and the capture are
replaced."""

import types

import pytest

pytest.importorskip("decision_index")
from karpo_decision.serve import engine  # noqa: E402


def decider():
    return types.SimpleNamespace(config=types.SimpleNamespace(template="karpo-blocks-v1", attention="bidirectional"),
                                 max_length=32768, model="packed")


@pytest.mark.parametrize(
    ("options", "applies", "path", "called"),
    [({"fp8": False}, True, "plain", []),
     ({"fp8": True}, True, "fp8-graphs", ["quantize", "decide", "fast"]),
     ({"fp8": True, "fast": False}, True, "fp8", ["quantize", "decide"]),
     ({"fp8": True}, False, "fp8", ["quantize", "decide", "fast"])],
    ids=["bf16", "fp8", "fp8-no-fast", "fp8-graphs-not-applicable"])
def test_engine_path(monkeypatch, options, applies, path, called):
    calls, loaded = [], decider()
    loaded.decide = lambda request: calls.append(("decide", request))
    monkeypatch.setattr(engine.Decider, "from_pretrained", classmethod(lambda cls, directory, **_: loaded))
    monkeypatch.setattr(engine, "quantize", lambda model: calls.append(("quantize", model)))
    monkeypatch.setattr(engine.graphs, "fast", lambda d: calls.append(("fast", d)) or applies)
    e = engine.DecisionIndexEngine(checkpoint="ckpt", device="cpu", **options)
    received = {"quantize": loaded.model, "decide": engine.graphs.WARMUP_REQUEST, "fast": loaded}
    assert e.path == path and e.provenance["path"] == path and calls == [(name, received[name]) for name in called]


def test_engine_refuses_a_non_boolean_fp8(monkeypatch):
    monkeypatch.setattr(engine.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, **_: pytest.fail("loaded")))
    with pytest.raises(ValueError, match="fp8"):
        engine.DecisionIndexEngine(checkpoint="ckpt", device="cpu", fp8="False")


def test_engine_refuses_a_non_boolean_fast(monkeypatch):
    monkeypatch.setattr(engine.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, **_: pytest.fail("loaded")))
    with pytest.raises(ValueError, match="fast"):
        engine.DecisionIndexEngine(checkpoint="ckpt", device="cpu", fast="False")
