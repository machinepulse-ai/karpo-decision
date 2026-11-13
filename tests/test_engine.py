"""The Decision Index engine's path: plain by default, nothing answered at load; FP8 projections with ``fp8=true``, the
graphs' warmup request answered, then CUDA graphs unless ``fast=false`` or the graphs do not apply; a non-boolean
``fp8``, ``fast`` or ``triton_attention`` is refused before loading. No GPU and no weights: the loader, the
quantization and the capture are replaced. ``triton_attention`` enables the Triton global attention, for the
decider's ``max_length``, after the quantization and before the warmup."""

import sys
import types

import pytest

pytest.importorskip("decision_index")
from karpo_decision.encoding import TextOnlyError  # noqa: E402
from karpo_decision.serve import engine  # noqa: E402


def decider():
    return types.SimpleNamespace(config=types.SimpleNamespace(template="karpo-blocks-v1", attention="bidirectional"),
                                 max_length=32768, model="packed",
                                 encoder=types.SimpleNamespace(processor=types.SimpleNamespace(vision=None)))


@pytest.mark.parametrize(
    ("options", "applies", "path", "called", "kernels"),
    [({"fp8": False}, True, "plain", [], []),
     ({"fp8": True}, True, "fp8-graphs", ["quantize", "decide", "fast"], []),
     ({"fp8": True, "fast": False}, True, "fp8", ["quantize", "decide"], []),
     ({"fp8": True}, False, "fp8", ["quantize", "decide", "fast"], []),
     ({"triton_attention": True}, True, "plain", ["attention"], ["triton-attention"]),
     ({"fp8": True, "triton_attention": True}, True, "fp8-graphs", ["quantize", "attention", "decide", "fast"],
      ["triton-attention"])],
    ids=["bf16", "fp8", "fp8-no-fast", "fp8-graphs-not-applicable", "bf16-triton-attention", "fp8-triton-attention"])
def test_engine_path(monkeypatch, options, applies, path, called, kernels):
    monkeypatch.setattr(engine, "require", lambda checkpoint, **_: frozenset({"fp8", "graphs", "triton_attention"}))
    calls, loaded = [], decider()
    loaded.decide = lambda request: calls.append(("decide", request))
    monkeypatch.setattr(engine.Decider, "from_pretrained", classmethod(lambda cls, directory, **_: loaded))
    monkeypatch.setattr(engine, "quantize", lambda model: calls.append(("quantize", model)))
    monkeypatch.setattr(engine, "enable_attention",
                        lambda model, max_length: calls.append(("attention", (model, max_length))))
    monkeypatch.setattr(engine.graphs, "fast", lambda d: calls.append(("fast", d)) or applies)
    monkeypatch.setattr(engine.torch.cuda, "get_device_name", lambda *_: "test-gpu")
    e = engine.DecisionIndexEngine(checkpoint="ckpt", device="cpu", **options)
    received = {"quantize": loaded.model, "attention": (loaded.model, loaded.max_length),
                "decide": engine.graphs.WARMUP_REQUEST, "fast": loaded}
    assert e.path == path and e.provenance["path"] == path and calls == [(name, received[name]) for name in called]
    assert e.kernels == kernels and e.provenance["kernels"] == kernels and e.runtime()["kernels"] == kernels


def test_engine_without_graphs_stays_on_fp8(monkeypatch):
    monkeypatch.setattr(engine, "require", lambda checkpoint, **_: frozenset({"fp8", "triton_attention"}))
    calls, loaded = [], decider()
    loaded.decide = lambda request: calls.append("decide")
    monkeypatch.setattr(engine.Decider, "from_pretrained", classmethod(lambda cls, directory, **_: loaded))
    monkeypatch.setattr(engine, "quantize", lambda model: calls.append("quantize"))
    monkeypatch.setattr(engine.graphs, "fast", lambda d: pytest.fail("graphs.fast called"))
    monkeypatch.setattr(engine.torch.cuda, "get_device_name", lambda *_: "test-gpu")
    e = engine.DecisionIndexEngine(checkpoint="ckpt", device="cpu", fp8=True)
    assert e.path == "fp8" and calls == ["quantize", "decide"]


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


def test_engine_refuses_a_non_boolean_triton_attention(monkeypatch):
    monkeypatch.setattr(engine.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, **_: pytest.fail("loaded")))
    with pytest.raises(ValueError, match="triton_attention"):
        engine.DecisionIndexEngine(checkpoint="ckpt", device="cpu", triton_attention="true")


def test_engine_refuses_a_path_the_backbone_does_not_run(monkeypatch):
    def refuse(checkpoint, **_):
        raise ValueError("fp8 is not supported for qwen3_5 checkpoints")

    monkeypatch.setattr(engine, "require", refuse)
    monkeypatch.setattr(engine.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, **_: pytest.fail("loaded")))
    with pytest.raises(ValueError, match="not supported"):
        engine.DecisionIndexEngine(checkpoint="ckpt", device="cpu", fp8=True)


def vision_decider(tokens=63):
    loaded = decider()
    loaded.encoder = types.SimpleNamespace(processor=types.SimpleNamespace(
        vision=types.SimpleNamespace(summary=lambda position_ids: ([1_000_000], tokens))))
    loaded.plans = []

    def plan(request):
        loaded.plans.append(request)
        if request.get("images") == ["broken"]:
            raise OSError("cannot identify image file")
        return [types.SimpleNamespace(image_position_ids="positions" if request.get("images") else None)]

    loaded.plan = plan
    loaded.decide = lambda readings, plans: [{"answers": {"q": {"choice": "a"}}, "usage": {"input_tokens": 1}}
                                             for _ in readings]
    return loaded


def test_engine_reads_the_kits_images_and_reports_them(monkeypatch):
    loaded = vision_decider()
    monkeypatch.setattr(engine, "require", lambda checkpoint, **_: frozenset())
    monkeypatch.setattr(engine.Decider, "from_pretrained", classmethod(lambda cls, directory, **_: loaded))
    images_module = types.SimpleNamespace(MAX_PIXELS=1_638_400, pil=lambda ref, root, cap: f"{root}/{ref}@{cap}")
    monkeypatch.setitem(sys.modules, "decision_index.vision.images", images_module)
    monkeypatch.setitem(sys.modules, "decision_index.vision", types.SimpleNamespace(images=images_module))
    e = engine.DecisionIndexEngine(checkpoint="ckpt", device="cpu", image_root="suite")
    response, meta = e({}, {"q": {"type": "noul", "instructions": "Blank?"}}, images=["img/0.jpg"])
    assert loaded.plans[0]["images"] == ["suite/img/0.jpg@1638400"]
    assert meta == {"image_pixels": [1_000_000], "image_tokens": 63}
    _, text_meta = e({}, {"q": {"type": "noul", "instructions": "Blank?"}})
    assert text_meta is None


def test_engine_answers_unsupported_for_an_image_it_cannot_read(monkeypatch):
    loaded = vision_decider()
    monkeypatch.setattr(engine, "require", lambda checkpoint, **_: frozenset())
    monkeypatch.setattr(engine.Decider, "from_pretrained", classmethod(lambda cls, directory, **_: loaded))
    images_module = types.SimpleNamespace(MAX_PIXELS=1_638_400, pil=lambda ref, root, cap: ref)
    monkeypatch.setitem(sys.modules, "decision_index.vision.images", images_module)
    monkeypatch.setitem(sys.modules, "decision_index.vision", types.SimpleNamespace(images=images_module))
    e = engine.DecisionIndexEngine(checkpoint="ckpt", device="cpu")
    with pytest.raises(engine.Unsupported):
        e({}, {"q": {"type": "noul", "instructions": "Blank?"}}, images=["broken"])


def test_engine_answers_unsupported_for_images_on_a_checkpoint_without_vision(monkeypatch):
    """The 12B (``vision`` None): its plan refuses images with ``TextOnlyError``."""
    loaded = decider()

    def plan(request):
        if request.get("images"):
            raise TextOnlyError("this checkpoint reads text only")
        return [types.SimpleNamespace(image_position_ids=None)]

    loaded.plan = plan
    monkeypatch.setattr(engine, "require", lambda checkpoint, **_: frozenset())
    monkeypatch.setattr(engine.Decider, "from_pretrained", classmethod(lambda cls, directory, **_: loaded))
    images_module = types.SimpleNamespace(MAX_PIXELS=1_638_400, pil=lambda ref, root, cap: ref)
    monkeypatch.setitem(sys.modules, "decision_index.vision.images", images_module)
    monkeypatch.setitem(sys.modules, "decision_index.vision", types.SimpleNamespace(images=images_module))
    e = engine.DecisionIndexEngine(checkpoint="ckpt", device="cpu")
    with pytest.raises(engine.Unsupported, match="TextOnlyError"):
        e({}, {"q": {"type": "noul", "instructions": "Blank?"}}, images=["img/0.jpg"])
