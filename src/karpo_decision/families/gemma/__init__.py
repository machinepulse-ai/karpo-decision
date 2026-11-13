"""Gemma 4 (``google/gemma-4-31B-it``), trained the way the
board's top Gemma entries are (Sieve-31B-v3): text only (the vision tower is loaded and frozen, requests with images
are refused), the tokenizer as the processor, block layouts only (``karpo-blocks-v1``: the answer is read at the
``Answer: (`` slot ending each branch, as the model's own next-token logits of the option labels, with the soft cap of
its tied language-model head, ``final_logit_softcapping`` 30). Only the decoder's linear projections train: the token
embeddings (tied with the head), every norm and the readout stay as pretrained.
In a bidirectional model the global layers see what a token may see in both directions and the sliding-window layers
stay causal inside their window (``packed.py``). ``MARKER`` only satisfies the encoder's check and is escaped from
request text; block layouts place no marker token. The 12B (``google/gemma-4-12B-it``, ``gemma4_unified``) has the
same text layers and no vision tower (images go through an embedder), and loads as its own model class."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4Model
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedModel

from .. import model_type
from .packed import Packed

MODEL_TYPE = "gemma4"
MARKER = "<mask>"
TEMPLATES = ("karpo-blocks-v1",)
#: The trained parameters: every decoder layer's linear projections (Sieve-31B-v3's choice, 29,286,727,680 for the 31B).
TRAINED = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
#: The backbone class of each ``model_type`` the family reads.
BACKBONES = {"gemma4": Gemma4Model, "gemma4_unified": Gemma4UnifiedModel}

__all__ = ["MARKER", "MODEL_TYPE", "TEMPLATES", "Packed", "TextProcessor", "load_backbone", "load_processor",
           "logit_cap"]


class TextProcessor:
    """The tokenizer in the processor's place: chat template, tokenization of text alone, saving."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def save_pretrained(self, directory: str) -> None:
        self.tokenizer.save_pretrained(directory)

    def __call__(self, text: list[str], images: Any = None, return_tensors: str | None = None, **_: Any) -> Any:
        """Text alone, as the processor would tokenize it (the chat template already wrote ``<bos>``)."""
        if images:
            raise ValueError("the Gemma family reads text only")
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors=return_tensors)["input_ids"]
        return {"input_ids": ids, "mm_token_type_ids": torch.zeros_like(ids)}


def load_processor(directory: str | Path, max_pixels: int) -> TextProcessor:
    tokenizer = AutoTokenizer.from_pretrained(str(directory))
    tokenizer.padding_side = "left"
    return TextProcessor(tokenizer)


def load_backbone(weights: str | Path, dtype: torch.dtype) -> Gemma4Model | Gemma4UnifiedModel:
    backbone = BACKBONES[model_type(weights)].from_pretrained(str(weights), dtype=dtype, attn_implementation="sdpa")
    for name, param in backbone.named_parameters():  # the vision tower, embeddings and norms stay as pretrained
        param.requires_grad_(name.startswith("language_model.layers.") and name.rsplit(".", 2)[-2] in TRAINED)
    return backbone


def logit_cap(backbone: Any) -> float | None:
    return backbone.config.get_text_config().final_logit_softcapping
