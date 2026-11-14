"""``Decider.logits`` gives planned images their pixels (``Vision.read``) right before it packs the sequences."""

import types

import torch

from karpo_decision import decider as module
from karpo_decision.encoding import Encoded

#: A sequence's tokens, a pack's token budget that two of them fill, and the image processor's smallest budget.
LENGTH, BUDGET, SMALL = 10, 25, 70


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


def planned(n: int, length: int = LENGTH) -> list[Encoded]:
    return [Encoded(keys=(str(k),), counts=(2,), input_ids=torch.arange(length), markers=torch.tensor([length - 1]))
            for k in range(n)]


def test_prepare_reads_the_first_pack_and_leaves_the_rest_to_theirs():
    plan, read = planned(3), []
    vision = types.SimpleNamespace(read=lambda seqs: read.append(list(seqs)) or [f"read {s.keys[0]}" for s in seqs])
    d = module.Decider.__new__(module.Decider)
    d.encoder = types.SimpleNamespace(processor=types.SimpleNamespace(vision=vision))
    d.token_budget = BUDGET  # two sequences fill the first pack
    assert d.prepare(plan) == ["read 0", "read 1", plan[2]] and read == [plan[:2]]


def test_prepare_leaves_a_plan_alone_on_a_checkpoint_without_vision():
    d = module.Decider.__new__(module.Decider)
    d.encoder = types.SimpleNamespace(processor=types.SimpleNamespace(vision=None))
    d.token_budget = BUDGET
    plan = planned(2)
    assert d.prepare(plan) is plan


def test_a_prepared_plan_packs_the_same_pixels(monkeypatch):
    """With the real image processor: deciding a prepared plan hands every pack exactly the tensors it would make."""
    from PIL import Image

    from karpo_decision.families.gemma import vision

    v = vision.Vision(vision.ImageTokens(begin=290, soft=291, end=292), max_soft_tokens=SMALL)
    plan = [Encoded(keys=(str(k),), counts=(2,), input_ids=torch.arange(LENGTH), markers=torch.tensor([LENGTH - 1]),
                    images=(v.measure(Image.new("RGB", (640, 480), (40 * k, 90, 160))),)) for k in range(3)]
    packs = []

    def pack(seqs, backbone, device):
        packs.append([s.pixel_values for s in seqs])
        return seqs

    monkeypatch.setattr(module, "pack", pack)
    d = module.Decider.__new__(module.Decider)
    d.encoder = types.SimpleNamespace(processor=types.SimpleNamespace(vision=v))
    d.ckpt, d.device, d.token_budget = types.SimpleNamespace(backbone=None), torch.device("cpu"), BUDGET
    d.model = lambda seqs: torch.zeros(len(seqs), 8)
    d.readout = lambda hidden: torch.zeros(hidden.shape[0], 2)
    request = {"state": None, "questions": {s.keys[0]: {"type": "noul"} for s in plan}}
    d.question_logits([request], plans=[plan])
    lazy, packs[:] = list(packs), []
    d.question_logits([request], plans=[d.prepare(plan)])
    assert [len(p) for p in packs] == [len(p) for p in lazy] == [2, 1]
    assert all(torch.equal(a, b) for got, want in zip(packs, lazy, strict=True) for a, b in zip(got, want, strict=True))
