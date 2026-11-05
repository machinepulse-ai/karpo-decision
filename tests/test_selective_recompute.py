import types

import torch

from karpo_decision.model import packed as mp


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = torch.nn.Linear(16, 32), torch.nn.Linear(32, 16)
        self.norm = torch.nn.LayerNorm(16)

    def step(self, h, sin):
        return h + self.b(torch.nn.functional.gelu(self.a(self.norm(h))) * sin)


class Layers(mp.PackedLayers):
    def layer(self, layer, h, cos, sin, p):
        return layer.step(h, sin)


def test_selective_recompute_gives_the_same_gradients_as_keeping_everything():
    torch.manual_seed(0)
    toys = [Toy() for _ in range(4)]
    h0, sin = torch.randn(5, 16), torch.randn(5, 1)

    def grads(**policy):
        for t in toys:
            t.zero_grad()
        layers = Layers(toys, **policy)
        h = h0
        for i in range(len(toys)):
            h = layers.run(i, h, None, sin, types.SimpleNamespace(tokens=h.shape[0]))
        h.square().sum().backward()
        return [q.grad.clone() for t in toys for q in t.parameters()]

    want = grads()
    for policy in ({"recompute": "selective", "keep_every": 1}, {"recompute": "selective", "keep_every": 2},
                   {"recompute": "layer", "checkpoint_every": 1}):
        for g, w in zip(grads(**policy), want, strict=True):
            torch.testing.assert_close(g, w)


def test_selective_keeps_the_matmuls_and_recomputes_the_rest():
    kept = mp._kept_ops()
    assert torch.ops.aten.mm.default in kept and torch.ops.aten.addmm.default in kept
    assert torch.ops.aten.gelu.default not in kept
