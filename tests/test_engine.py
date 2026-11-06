"""The Decision Index engine's path: plain by default, FP8 projections with ``fp8=true``; a non-boolean ``fp8`` is
refused before loading. No GPU and no weights: the loader is replaced."""

import types

import pytest

pytest.importorskip("decision_index")
from karpo_decision.serve import engine  # noqa: E402


def decider():
    return types.SimpleNamespace(config=types.SimpleNamespace(template="karpo-blocks-v1", attention="bidirectional"),
                                 max_length=32768, model="packed")


@pytest.mark.parametrize(("fp8", "path", "quantized"), [(False, "plain", []), (True, "fp8", ["packed"])])
def test_engine_path(monkeypatch, fp8, path, quantized):
    calls = []
    monkeypatch.setattr(engine.Decider, "from_pretrained", classmethod(lambda cls, directory, **_: decider()))
    monkeypatch.setattr(engine, "quantize", lambda model: calls.append(model))
    e = engine.DecisionIndexEngine(checkpoint="ckpt", device="cpu", fp8=fp8)
    assert e.path == path and e.provenance["path"] == path and calls == quantized


def test_engine_refuses_a_non_boolean_fp8(monkeypatch):
    monkeypatch.setattr(engine.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, **_: pytest.fail("loaded")))
    with pytest.raises(ValueError, match="fp8"):
        engine.DecisionIndexEngine(checkpoint="ckpt", device="cpu", fp8="False")
