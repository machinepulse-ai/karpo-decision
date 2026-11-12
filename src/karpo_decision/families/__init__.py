"""Model families: everything that depends on which pretrained model the decision model is built on, behind one small
interface; encoding, templates, training, answering and the readout head stay the same for all of them.

A family is a subpackage with:

- ``MODEL_TYPE``: the ``model_type`` of the backbone's ``config.json``, which picks the family;
- ``MARKER``: the readout marker, an existing special token request text never produces;
- ``TEMPLATES``: the template ids its attention reads (joint layouts for Qwen, block layouts for Gemma);
- ``load_processor(directory, max_pixels)``: the processor (tokenizer, chat template, images when the family reads
  them);
- ``load_backbone(weights, dtype)``: the backbone module;
- ``logit_cap(backbone)``: the soft cap the backbone's language-model head puts on its logits (None when it has none),
  kept by a readout head started from that head's rows;
- ``Packed(backbone, attention, checkpoint_every=0, recompute="layer", offload=False, keep_every=0)``: the packed
  forward pass (``model.packed.PackedLayers``).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType

#: model_type -> subpackage
FAMILIES = {"qwen3_5": "qwen", "gemma4": "gemma", "gemma4_unified": "gemma"}


def model_type(weights: str | Path) -> str:
    """The ``model_type`` of the backbone whose ``config.json`` is in ``weights``."""
    return json.loads((Path(weights) / "config.json").read_text())["model_type"]


def of(weights: str | Path) -> ModuleType:
    """The family of the backbone whose ``config.json`` is in ``weights``."""
    name = model_type(weights)
    if name not in FAMILIES:
        raise ValueError(f"no model family for model_type {name!r}; known: {', '.join(FAMILIES)}")
    return importlib.import_module(f".{FAMILIES[name]}", __name__)
