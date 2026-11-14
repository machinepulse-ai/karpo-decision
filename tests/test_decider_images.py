"""``Decider.logits`` gives planned images their pixels (``Vision.read``) right before it packs the sequences."""

import types

import torch

from karpo_decision import decider as module
from karpo_decision.encoding import Encoded


def test_logits_reads_the_images_of_the_sequences_it_packs(monkeypatch):
    s = Encoded(keys=("q",), counts=(2,), input_ids=torch.arange(4), markers=torch.tensor([3]))
    read, packed = [], []
    vision = types.SimpleNamespace(read=lambda seqs: read.append(list(seqs)) or list(seqs))
    d = module.Decider.__new__(module.Decider)
    d.encoder = types.SimpleNamespace(processor=types.SimpleNamespace(vision=vision))
    d.ckpt, d.device = types.SimpleNamespace(backbone=None), torch.device("cpu")
    d.model = lambda p: torch.zeros(1, 8)
    d.readout = lambda hidden: torch.zeros(1, 2)
    monkeypatch.setattr(module, "pack", lambda seqs, backbone, device: packed.append(seqs))
    (logits,) = d.logits([s])
    assert read == [[s]] and packed == [[s]] and logits[0].shape == (2,)


def planned(n: int, length: int = 10) -> list[Encoded]:
    return [Encoded(keys=(str(k),), counts=(2,), input_ids=torch.arange(length), markers=torch.tensor([length - 1]))
            for k in range(n)]


def test_prepare_reads_the_first_pack_and_leaves_the_rest_to_theirs():
    plan, read = planned(3), []
    vision = types.SimpleNamespace(read=lambda seqs: read.append(list(seqs)) or [f"read {s.keys[0]}" for s in seqs])
    d = module.Decider.__new__(module.Decider)
    d.encoder = types.SimpleNamespace(processor=types.SimpleNamespace(vision=vision))
    d.token_budget = 25  # two sequences of 10 tokens fill the first pack
    assert d.prepare(plan) == ["read 0", "read 1", plan[2]] and read == [plan[:2]]


def test_prepare_leaves_a_plan_alone_on_a_checkpoint_without_vision():
    d = module.Decider.__new__(module.Decider)
    d.encoder = types.SimpleNamespace(processor=types.SimpleNamespace(vision=None))
    d.token_budget = 25
    plan = planned(2)
    assert d.prepare(plan) is plan
