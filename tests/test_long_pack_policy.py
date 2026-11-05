"""model.packed.PackedLayers: a pack above long_tokens keeps the share long_tokens / tokens of the layers keep_every
keeps, spread evenly, the others recomputed whole; a pack at or below it, and every pack without long_tokens, keep
them all. The choice follows the pack, not the layer index alone."""
import types

import torch

from karpo_decision.model import packed as mp


class Layers(mp.PackedLayers):
    def layer(self, layer, h, cos, sin, p):
        return h


def policies(monkeypatch, tokens, long_tokens=100, keep_every=1):
    seen = []
    monkeypatch.setattr(mp, "checkpoint", lambda fn, *a, **k: seen.append("kept" if "context_fn" in k else "whole")
                        or fn(*a))
    layers = Layers([torch.nn.Identity() for _ in range(6)], recompute="selective", keep_every=keep_every,
                    long_tokens=long_tokens)
    h = torch.zeros(1, requires_grad=True)
    for i in range(6):
        layers.run(i, h, None, None, types.SimpleNamespace(tokens=tokens))
    return seen


def test_a_pack_up_to_long_tokens_keeps_every_layer(monkeypatch):
    assert policies(monkeypatch, 100) == ["kept"] * 6


def test_a_longer_pack_keeps_the_share_long_tokens_over_tokens_spread_evenly(monkeypatch):
    assert policies(monkeypatch, 200) == ["whole", "kept"] * 3
    assert policies(monkeypatch, 150) == ["whole", "kept", "kept", "whole", "kept", "kept"]
    assert policies(monkeypatch, 101).count("kept") == 5
    assert policies(monkeypatch, 700) == ["whole"] * 6


def test_the_share_is_of_the_layers_keep_every_keeps(monkeypatch):
    assert policies(monkeypatch, 100, keep_every=2) == ["kept", "whole"] * 3
    assert policies(monkeypatch, 200, keep_every=2).count("kept") == 1


def test_without_long_tokens_every_pack_the_same(monkeypatch):
    assert policies(monkeypatch, 10_000, long_tokens=0) == ["kept"] * 6
