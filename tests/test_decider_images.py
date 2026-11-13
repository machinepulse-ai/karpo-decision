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
