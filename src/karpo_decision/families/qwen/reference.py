"""The reference forward pass: one left-padded batch through the unmodified Transformers backbone.

It is the computation the packed path must reproduce (up to bf16 noise); tests and the equivalence tools compare
the two. A bidirectional model's full-attention layers get the padding mask alone (every real token sees every
other); the Gated DeltaNet layers keep transformers' recurrent padding mask, so they stay causal. RoPE positions
are shifted by the left padding here and restart at 0 in the packed path; attention only sees position
differences, so the two agree.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from transformers.masking_utils import create_recurrent_attention_mask

from ...config import Attention
from ...encoding import Encoded


@contextmanager
def bidirectional_masks(language_model: torch.nn.Module) -> Iterator[None]:
    """While active, the full-attention layers of ``language_model`` attend to every non-padding token."""

    def masks(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any]:
        if args:
            raise ValueError("the bidirectional mask hook needs keyword inputs")
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("bidirectional readout has no KV cache")
        padding = kwargs.get("attention_mask")
        if isinstance(padding, dict):
            return args, kwargs
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            embeddings = module.embed_tokens(kwargs["input_ids"])
        if padding is None:
            padding = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        kwargs["attention_mask"] = {
            "full_attention": padding[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(config=module.config, inputs_embeds=embeddings,
                                                                attention_mask=padding),
        }
        return args, kwargs

    handle = language_model.register_forward_pre_hook(masks, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def left_pad(sequences: Sequence[Encoded], pad_id: int) -> dict[str, torch.Tensor]:
    """The processor's batch layout (left padding) built from single-sequence encodings."""
    width = max(s.length for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, s in enumerate(sequences):
        ids[row, width - s.length:] = s.input_ids
        mask[row, width - s.length:] = 1
    inputs = {"input_ids": ids, "attention_mask": mask}
    typed = [s.mm_token_type_ids for s in sequences if s.mm_token_type_ids is not None]
    if typed:
        types = torch.zeros((len(sequences), width), dtype=typed[0].dtype)
        for row, s in enumerate(sequences):
            if s.mm_token_type_ids is not None:
                types[row, width - s.length:] = s.mm_token_type_ids
        inputs["mm_token_type_ids"] = types
    pixels = [s.pixel_values for s in sequences if s.pixel_values is not None]
    if pixels:
        inputs["pixel_values"] = torch.cat(pixels)
        inputs["image_grid_thw"] = torch.cat([s.image_grid_thw for s in sequences if s.image_grid_thw is not None])
    return inputs


def reference_hidden(backbone: Any, sequences: Sequence[Encoded], attention: Attention, pad_id: int) -> torch.Tensor:
    """(M, hidden) final hidden states at every marker, sequences in order, markers in question order."""
    device = next(backbone.parameters()).device
    inputs = {key: value.to(device) for key, value in left_pad(sequences, pad_id).items()}
    if attention == "bidirectional":
        with bidirectional_masks(backbone.language_model):
            hidden = backbone(**inputs, use_cache=False).last_hidden_state
    else:
        hidden = backbone(**inputs, use_cache=False).last_hidden_state
    width = inputs["input_ids"].shape[1]
    rows = torch.cat([torch.full((len(s.keys),), i) for i, s in enumerate(sequences)]).to(device)
    cols = torch.cat([s.markers + width - s.length for s in sequences]).to(device)
    return hidden[rows, cols]
