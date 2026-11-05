"""Checkpoints: a directory with ``decision_config.json``, ``readout.safetensors`` (``{"weight": [codes, hidden]}``),
the processor files, and the backbone's weights (or, for an untrained initialization, the config's ``backbone``
path to them). The backbone's ``config.json`` picks its model family (``families``), which loads the processor and
the backbone and gives the packed forward pass.

``initialize`` starts a model from a base checkpoint of any family: its answer codes from the tokenizer, its readout
rows from the base language-model head's rows of those code tokens (the input embeddings when they are tied), causal
attention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from . import families
from .codes import answer_codes
from .config import DEFAULT_MAX_PIXELS, Attention, DecisionConfig
from .model.readout import Readout
from .templates import BlockTemplate

READOUT_FILE = "readout.safetensors"
READOUT_KEY = "weight"
#: The base model's language-model head tensor, whose code-token rows start the readout; a base with tied embeddings
#: has none and its input embeddings are the head.
LM_HEAD = "lm_head.weight"
TIED_HEADS = (
    "model.language_model.embed_tokens.weight", "model.embed_tokens.weight", "language_model.embed_tokens.weight",
)
WEIGHTS_INDEX = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"


@dataclass
class Checkpoint:
    config: DecisionConfig
    processor: Any
    backbone: torch.nn.Module
    readout: Readout
    directory: Path
    family: ModuleType


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def load_processor(directory: str | Path, max_pixels: int) -> Any:
    """The processor of the base model in ``directory`` (its family's)."""
    return families.of(directory).load_processor(directory, max_pixels)


def load(directory: str | Path, device: str | torch.device = "cuda", dtype: torch.dtype = torch.bfloat16) -> Checkpoint:
    directory = Path(directory)
    config = DecisionConfig.load(directory)
    weights = Path(config.backbone) if config.backbone else directory
    family = families.of(weights)
    processor = family.load_processor(directory if (directory / "tokenizer.json").exists() else weights,
                                      config.max_pixels)
    tokenizer = processor.tokenizer
    for code, token_id in zip(config.codes, config.token_ids, strict=True):
        if tokenizer.encode(code, add_special_tokens=False) != [token_id]:
            raise ValueError(f"code {code!r} is not token {token_id} of this tokenizer")
    backbone = family.load_backbone(weights, dtype)
    readout = Readout.from_rows(load_file(str(directory / READOUT_FILE))[READOUT_KEY], _dtype(config.readout_dtype),
                                family.logit_cap(backbone))
    device = torch.device(device)
    return Checkpoint(config, processor, backbone.to(device), readout.to(device), directory, family)


def lm_head_rows(base: str | Path, token_ids: list[int]) -> torch.Tensor:
    """The base model's language-model-head rows of ``token_ids`` (its input embeddings when they are tied), read from
    its safetensors file or shards."""
    base = Path(base)
    if (base / WEIGHTS_INDEX).exists():
        files = json.loads((base / WEIGHTS_INDEX).read_text())["weight_map"]
    else:
        with safe_open(str(base / SINGLE_FILE), "pt") as single:
            files = dict.fromkeys(single.keys(), SINGLE_FILE)
    name = next((key for key in (LM_HEAD, *TIED_HEADS) if key in files), None)
    if name is None:
        raise ValueError(f"{base} has neither {LM_HEAD} nor tied input embeddings")
    with safe_open(str(base / files[name]), "pt") as shard:
        return shard.get_tensor(name)[token_ids].clone()


def initialize(base: str | Path, out: str | Path, attention: Attention = "causal",
               template: str = BlockTemplate.id) -> DecisionConfig:
    """A trainable starting point in ``out`` from a base checkpoint (the backbone stays where it is)."""
    base, out = Path(base), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    family = families.of(base)
    processor = family.load_processor(base, DEFAULT_MAX_PIXELS)
    codes, token_ids = answer_codes(processor.tokenizer)
    config = DecisionConfig(template=template, codes=tuple(codes), token_ids=tuple(token_ids), attention=attention,
                            base_model=str(base), marker=family.MARKER, backbone=str(base.resolve()))
    save_file({READOUT_KEY: lm_head_rows(base, token_ids).float().contiguous()}, str(out / READOUT_FILE))
    processor.save_pretrained(str(out))
    config.save(out)
    return config


def save(checkpoint: Checkpoint, out: str | Path) -> None:
    """A self-contained checkpoint: backbone weights, readout, processor and config in ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint.backbone.save_pretrained(str(out))
    save_file({READOUT_KEY: checkpoint.readout.weight.detach().float().cpu().contiguous()}, str(out / READOUT_FILE))
    checkpoint.processor.save_pretrained(str(out))
    replace(checkpoint.config, backbone=None).save(out)


if __name__ == "__main__":
    import sys

    print(initialize(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or ["causal"])))
