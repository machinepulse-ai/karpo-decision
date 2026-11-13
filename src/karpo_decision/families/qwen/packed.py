"""The packed forward pass over a Qwen3.5 backbone's text layers (a hybrid: Gated DeltaNet
linear-attention layers and full-attention layers, three to one). Serving and training both use it; it is
differentiable. Sequences stay independent of each other:

- full-attention layers: varlen attention inside each sequence, bidirectional (window (-1, -1)) or causal
  (window (-1, 0)), never across sequences;
- linear-attention (Gated DeltaNet) layers: flash-linear-attention's chunked gated delta rule with the sequence
  boundaries (its state restarts at every sequence), after a depthwise causal convolution that never reads taps from
  before a sequence's start (flash-linear-attention's kernel with the boundaries; ``packed_conv`` for padded rows);
- positions restart at 0 for every sequence; a sequence with images gets the backbone's M-RoPE positions;
- a bidirectional model computes its last layer (a full-attention layer) only at the markers: attention for the
  marker rows over every token, then the MLP for those rows. A causal model needs each marker's own prefix, which
  varlen attention cannot express for a subset of rows, so it computes the last layer for every token.

The layers are re-expressed with the backbone's own modules and weights; the Transformers model is not patched,
so the reference path (``reference.py``) over the same backbone stays the unmodified computation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN

from ...config import Attention
from ...model.packed import Pack, PackedLayers
from . import fused

#: Attention windows (left, right) of the full-attention layers: -1 is unbounded.
WINDOWS = {"bidirectional": (-1, -1), "causal": (-1, 0)}
LINEAR, FULL = "linear_attention", "full_attention"


def _kernels() -> tuple[Any, Any, Any]:
    try:
        from fla.modules.conv.causal_conv1d import causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("the packed path needs flash-linear-attention: pip install 'karpo-decision[qwen]'") from error
    from torch.nn.attention.varlen import varlen_attn

    return chunk_gated_delta_rule, varlen_attn, causal_conv1d


def packed_conv(x: torch.Tensor, weight: torch.Tensor, seqpos: torch.Tensor) -> torch.Tensor:
    """Depthwise causal convolution over (1, C, T) whose taps never reach before a sequence's start (fp32 sums)."""
    taps = weight.shape[-1]
    xf = x.float()
    out = xf * weight[:, taps - 1, None].float()
    for shift in range(1, taps):
        shifted = F.pad(xf, (shift, 0))[..., : x.shape[-1]]
        out = out + shifted * (seqpos >= shift).float() * weight[:, taps - 1 - shift, None].float()
    return out.to(x.dtype)


class Packed(PackedLayers):
    """A Qwen3.5 backbone's text layers over a ``Pack``; returns the normalized hidden state at every marker."""

    def __init__(self, backbone: Any, attention: Attention, checkpoint_every: int = 0, recompute: str = "layer",
                 offload: bool = False, keep_every: int = 0, long_tokens: int = 0) -> None:
        super().__init__(backbone.language_model.layers, checkpoint_every, recompute, offload, keep_every, long_tokens)
        self.backbone = backbone
        self.lm = backbone.language_model
        self.window = WINDOWS[attention]
        self.prune_last = attention == "bidirectional"
        if self.prune_last and self.layers[-1].block_type != FULL:
            raise ValueError("pruning the last layer needs a full-attention last layer")
        self.image_token_id = backbone.config.image_token_id
        self._delta_rule, self._varlen, self._conv = _kernels()

    # ---------------------------------------------------------------- token mixers

    def linear_attention(self, mod: Any, x: torch.Tensor, p: Pack) -> torch.Tensor:
        """Qwen3_5GatedDeltaNet.forward without cache or padding, over the pack."""
        length = x.shape[1]
        if mod.conv1d.bias is not None:
            raise ValueError("unexpected convolution bias")
        z = mod.in_proj_z(x).reshape(1, length, -1, mod.head_v_dim)
        b, a = mod.in_proj_b(x), mod.in_proj_a(x)
        if p.cu_cpu is not None:  # packed rows: the convolution and its activation in one kernel, cut at every start
            mixed, _ = self._conv(mod.in_proj_qkv(x), mod.conv1d.weight.squeeze(1), activation=mod.activation,
                                  cu_seqlens=p.plan, cu_seqlens_cpu=p.cu_cpu)
        else:
            mixed = packed_conv(mod.in_proj_qkv(x).transpose(1, 2), mod.conv1d.weight.squeeze(1), p.seqpos)
            mixed = ACT2FN[mod.activation](mixed).transpose(1, 2)
        query, key, value = torch.split(mixed, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
        query = query.reshape(1, length, -1, mod.head_k_dim)
        key = key.reshape(1, length, -1, mod.head_k_dim)
        value = value.reshape(1, length, -1, mod.head_v_dim)
        g, beta = fused.decay(a, b, mod.A_log, mod.dt_bias)
        repeat = mod.num_v_heads // mod.num_k_heads
        if repeat > 1:
            query, key = query.repeat_interleave(repeat, dim=2), key.repeat_interleave(repeat, dim=2)
        bounds = {} if p.cu_cpu is None else {"cu_seqlens": p.plan, "cu_seqlens_cpu": p.cu_cpu}
        rows = 1 if p.cu_cpu is not None else p.segments  # padded segments: one batch row each

        def split(t: torch.Tensor) -> torch.Tensor:
            return t if rows == 1 else t.reshape(rows, length // rows, *t.shape[2:])

        core, _ = self._delta_rule(split(query), split(key), split(value), g=split(g), beta=split(beta),
                                   initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True, **bounds)
        core = fused.gated_norm(core.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim), mod.norm.weight,
                                mod.norm.variance_epsilon)
        return mod.out_proj(core.reshape(1, length, -1))

    def _qkv(self, attn: Any, x: torch.Tensor, rows: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        """Gated queries for ``rows`` (all rows when None), and keys and values for every row."""
        length, hd = x.shape[1], attn.head_dim
        xq = x if rows is None else x[:, rows]
        n = xq.shape[1]
        query, gate = torch.chunk(attn.q_proj(xq).view(1, n, -1, hd * 2), 2, dim=-1)
        query = fused.norm(attn.q_norm, query).transpose(1, 2)
        key = fused.norm(attn.k_norm, attn.k_proj(x).view(1, length, -1, hd)).transpose(1, 2)
        value = attn.v_proj(x).view(1, length, -1, hd).transpose(1, 2)
        return query, key, value, gate.reshape(1, n, -1)

    def full_attention(self, attn: Any, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, p: Pack) -> torch.Tensor:
        query, key, value, gate = self._qkv(attn, x, None)
        query, key = fused.rope(query, key, cos, sin)
        q, k, v = (t[0].transpose(0, 1) for t in (query, key, value))  # (1, H, T, D) -> (T, H, D)
        out = self._varlen(q, k, v, p.cu, p.cu, p.max_length, p.max_length, scale=attn.scaling,
                           window_size=self.window, enable_gqa=k.shape[1] != q.shape[1])
        return attn.o_proj(fused.sigmoid_mul(out.reshape(1, x.shape[1], -1), gate))

    def last_layer_at_markers(self, layer: Any, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                              p: Pack) -> torch.Tensor:
        """The last (full-attention) layer for the marker rows only: their queries over every token's keys."""
        attn = layer.self_attn
        x = fused.norm(layer.input_layernorm, h)
        query, key, value, gate = self._qkv(attn, x, p.markers)
        query, _ = fused.rope(query, query, cos[:, p.markers], sin[:, p.markers])
        _, key = fused.rope(key, key, cos, sin)
        q, k, v = (t[0].transpose(0, 1).contiguous() for t in (query, key, value))
        out = self._varlen(q, k, v, p.marker_cu, p.cu, p.max_markers, p.max_length, scale=attn.scaling,
                           window_size=WINDOWS["bidirectional"], enable_gqa=k.shape[1] != q.shape[1])
        out = attn.o_proj(fused.sigmoid_mul(out.reshape(1, p.markers.numel(), -1), gate))
        hs, y = fused.residual_norm(h[:, p.markers], out, layer.post_attention_layernorm)
        return hs + self._mlp(layer, y)

    # ---------------------------------------------------------------- layers

    @staticmethod
    def _mlp(layer: Any, y: torch.Tensor) -> torch.Tensor:
        """The MLP over the post-attention-normed ``y``."""
        return fused.mlp(layer.mlp, y)

    def layer(self, layer: Any, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, p: Pack) -> torch.Tensor:
        x = fused.norm(layer.input_layernorm, h)
        if layer.block_type == LINEAR:
            x = self.linear_attention(layer.linear_attn, x, p)
        else:
            x = self.full_attention(layer.self_attn, x, cos, sin, p)
        h, y = fused.residual_norm(h, x, layer.post_attention_layernorm)
        return h + self.mlp_branch(self._mlp, layer, y)

    def embed(self, p: Pack) -> torch.Tensor:
        h = self.lm.embed_tokens(p.input_ids)[None]
        if p.pixel_values is None:
            return h
        features = self.backbone.get_image_features(p.pixel_values, p.image_grid_thw, return_dict=True).pooler_output
        features = torch.cat(list(features)).to(h.dtype)
        slots = (p.input_ids == self.image_token_id)[None, :, None].expand_as(h)
        return h.masked_scatter(slots, features)

    def body(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, p: Pack) -> torch.Tensor:
        """The layers that run on every token: all of them, or all but the last when it runs at the markers only."""
        for index in range(len(self.layers) - 1 if self.prune_last else len(self.layers)):
            h = self.run(index, h, cos, sin, p)
        return h

    def head(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, p: Pack) -> torch.Tensor:
        """(M, hidden) final-normalized hidden states at the markers, from the body's output."""
        h = self.layers[-1].packed_forward(h, cos, sin, p, markers_only=True) if self.prune_last else h[:, p.markers]
        return self.lm.norm(h)[0]

    def forward(self, p: Pack) -> torch.Tensor:
        """(M, hidden) final-normalized hidden states at the markers, in pack order."""
        if p.segment_ids is not None:
            raise ValueError("the Qwen family reads joint layouts, not block layouts")
        # The plan: the boundaries in int64 on the device, one tensor for every layer. fla caches its chunk plans by
        # tensor identity, so a fresh ``p.cu.long()`` per layer would plan again on the host twice a layer.
        p.plan = None if p.cu_cpu is None else p.cu.long()
        h = self.embed(p)
        cos, sin = self.lm.rotary_emb(h, p.rope)
        return self.head(self.body(h, cos, sin, p), cos, sin, p)
