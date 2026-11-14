"""The packed forward pass over a Gemma 4 backbone's text layers (the dense 31B: 50 sliding-window layers, window
1,024, head size 256; 10 global layers, head size 512, keys and values from one projection), over block layouts
(``karpo-blocks-v1``, ``blocks.py``). Sequences laid end to end never attend across each other; the final hidden state
is taken at the readout markers (each branch's last token). Training and serving both use it; it is differentiable.

Attention follows Sieve-31B-v3's rule A, by segments (``Pack.segment_ids``) and positions (``Pack.seqpos``):
- a sequence of one block (a request with one question) attends within itself: global layers bidirectionally (or
  causally), sliding-window layers causally within the window;
- in a branched sequence the shared block attends to itself only; a branch to the shared block and itself, never to
  another branch. Global layers: bidirectionally within what a token may see (or causally, aligned at the end).
  Sliding-window layers: causally within the window by position, so a branch sees the shared block's last 1,023
  tokens before its own first token.

``block_plan`` turns a pack into virtual sequences once (reused by every layer and by recomputation): every row is a
query of exactly one virtual sequence, in pack order, and each virtual sequence gathers its keys (the shared block's
rows, then its own) with ``index_select``, whose backward sums each branch's gradient into the shared rows. Sliding
layers run one varlen flash-attention call with separate query and key boundaries and the window (1023, 0)
(``sliding_attention``: flash-attention 4 on Blackwell, FlashAttention-2 elsewhere); global
layers (head size beyond flash attention's 256) run cuDNN's FROST kernels per virtual sequence on Blackwell
(``global_attention``), PyTorch's memory-efficient attention elsewhere. A bidirectional model computes its last layer
(a global layer) only at the markers, each over its own virtual sequence.
A pack padded for a CUDA graph (``PaddedPlan``: one single-block sequence and its padding) runs its global layers as one
varlen memory-efficient call (``global_attention.padded``). With ``global_attend`` set (``triton_attention.enable``,
sm_120), every global layer runs one call over all virtual sequences instead.

The layers are re-expressed with the backbone's own modules and weights (``Gemma4TextDecoderLayer``: sandwich norms,
q/k/v norms, the layer scalar kept in the activations' dtype); the Transformers model is not patched. Per-layer
embeddings, shared key-value layers and expert blocks (other Gemma 4 sizes) are refused.

A pack with images (``families/gemma/vision.py``) gets the vision tower's soft-token embeddings at the soft tokens'
rows; in the sliding layers each image's soft tokens are recomputed after the varlen call so that they see their whole
image (Gemma 4's bidirectional images; an image is no longer than the window), an image in a branch with what the
branch sees.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

from ...config import Attention
from ...model.packed import Pack, PackedLayers
from . import fused, global_attention, sliding_attention, vision

SLIDING, GLOBAL = "sliding_attention", "full_attention"


@dataclass
class BlockPlan:
    """A pack's virtual sequences: query boundaries over the pack's rows, and per layer kind the key rows gathered
    for each (global: the whole shared block; sliding: its last window - 1 rows), with their boundaries."""

    q_cu: list[int]
    q_cu_t: torch.Tensor
    max_q: int
    g_rows: torch.Tensor | None  # None: every virtual sequence's keys are its own rows (no branches)
    g_cu: list[int]
    g_cu_t: torch.Tensor  # g_cu on the device (``Packed.global_attend``); q_cu_t itself without branches
    s_rows: torch.Tensor | None
    s_cu_t: torch.Tensor
    max_s: int
    m_cu: list[int]  # marker boundaries over the virtual sequences (markers in pack order)
    m_cu_t: torch.Tensor  # m_cu on the device
    g_lens: torch.Tensor  # per virtual sequence its query and key lengths on the device (global_attention.lengths)
    m_lens: torch.Tensor  # the same with the markers as queries
    images: vision.ImagePlan | None = None  # a pack with images (``vision.plan_images``)


@dataclass
class PaddedPlan:
    """One single-block sequence of L rows padded to ``size`` rows (``serve.graphs``): the real rows and the padding are
    two sequences for attention, ``cu`` = [0, L, size] on the device."""

    cu: torch.Tensor
    size: int
    g_rows: None = None
    s_rows: None = None
    images: None = None

    @property
    def q_cu_t(self) -> torch.Tensor:
        return self.cu

    @property
    def s_cu_t(self) -> torch.Tensor:
        return self.cu

    @property
    def max_q(self) -> int:
        return self.size

    @property
    def max_s(self) -> int:
        return self.size


def block_plan(p: Pack, window: int) -> BlockPlan:
    """The virtual sequences of ``p``'s block layouts (``window``: the sliding layers' window, a token and the
    window - 1 before it). Its device tensors are all built here, before any layer is queued: a host-to-device copy
    made mid-forward waits for the GPU work ahead of it."""
    bounds = p.cu_cpu.tolist()
    segments = p.segment_ids_cpu
    q_cu, g_cu, s_cu = [0], [0], [0]
    g_rows, s_rows = [], []
    branched = False
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        seg = segments[a:b]
        changes = ((seg[1:] != seg[:-1]).nonzero().flatten() + 1).tolist()
        starts, ends = [0, *changes], [*changes, b - a]
        labels = [int(seg[i]) for i in starts]
        if labels[0] != 0 or labels[1:] != list(range(1, len(labels))):
            raise ValueError(f"segments {labels} are not a shared block followed by branches 1, 2, ...")
        shared = ends[0]
        own = torch.arange(a, a + shared)
        q_cu.append(a + shared)
        g_rows.append(own)
        s_rows.append(own)
        g_cu.append(g_cu[-1] + shared)
        s_cu.append(s_cu[-1] + shared)
        tail = own[max(0, shared - (window - 1)):]
        for start, end in zip(starts[1:], ends[1:], strict=True):
            branched = True
            rows = torch.arange(a + start, a + end)
            q_cu.append(a + end)
            g_rows.append(torch.cat([own, rows]))
            s_rows.append(torch.cat([tail, rows]))
            g_cu.append(g_cu[-1] + shared + len(rows))
            s_cu.append(s_cu[-1] + len(tail) + len(rows))
    device = p.input_ids.device
    markers = p.markers.cpu()
    m_cu = torch.searchsorted(markers, torch.tensor(q_cu)).tolist()
    for qb, ma, mb in zip(q_cu[1:], m_cu[:-1], m_cu[1:], strict=True):
        if mb - ma > 1 or (mb > ma and int(markers[ma]) != qb - 1):
            raise ValueError("a block's readout marker must be its last token, one per branch")
    lengths = [b - a for a, b in zip(q_cu[:-1], q_cu[1:], strict=True)]
    s_lengths = [b - a for a, b in zip(s_cu[:-1], s_cu[1:], strict=True)]
    q_cu_t = torch.tensor(q_cu, dtype=torch.int32, device=device)
    return BlockPlan(
        q_cu=q_cu, q_cu_t=q_cu_t, max_q=max(lengths),
        g_rows=torch.cat(g_rows).to(device) if branched else None, g_cu=g_cu,
        g_cu_t=torch.tensor(g_cu, dtype=torch.int32, device=device) if branched else q_cu_t,
        s_rows=torch.cat(s_rows).to(device) if branched else None,
        s_cu_t=torch.tensor(s_cu, dtype=torch.int32, device=device), max_s=max(s_lengths), m_cu=m_cu,
        m_cu_t=torch.tensor(m_cu, dtype=torch.int32, device=device),
        g_lens=global_attention.lengths(q_cu, g_cu, device), m_lens=global_attention.lengths(m_cu, g_cu, device))


class Packed(PackedLayers):
    """A Gemma 4 backbone's text layers over a ``Pack`` of block layouts; returns the normalized hidden state at every
    marker."""

    def __init__(self, backbone: Any, attention: Attention, checkpoint_every: int = 0, recompute: str = "layer",
                 offload: bool = False, keep_every: int = 0, long_tokens: int = 0) -> None:
        super().__init__(backbone.language_model.layers, checkpoint_every, recompute, offload, keep_every, long_tokens)
        self.backbone = backbone
        self.lm = backbone.language_model
        config = self.lm.config
        if config.hidden_size_per_layer_input or config.num_kv_shared_layers or config.enable_moe_block:
            raise ValueError("per-layer embeddings, shared key-value layers and expert blocks are not supported")
        if config.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(f"the fused MLP is gelu_pytorch_tanh, not {config.hidden_activation}")
        self.types = list(config.layer_types)
        self.causal_global = attention == "causal"
        self.window = config.sliding_window
        self.sliding = (config.sliding_window - 1, 0)
        self.prune_last = attention == "bidirectional"
        if self.prune_last and self.types[-1] != GLOBAL:
            raise ValueError("pruning the last layer needs a global last layer")
        #: The norm before a layer's attention and MLP, and the MLP; ``serve.fp8.quantize`` swaps in quantizing ones.
        self.norm_input, self.mlp = fused.norm, fused.mlp
        #: The global layers' attention over every virtual sequence at once (``triton_attention.attend``'s signature);
        #: None: cuDNN or memory-efficient attention per virtual sequence (one varlen call for a ``PaddedPlan``).
        self.global_attend: Callable[..., torch.Tensor] | None = None

    # ---------------------------------------------------------------- attention

    @staticmethod
    def _kv(attn: Any, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(1, T, heads, head size) keys (normed, rotated) and values (normed) of every row."""
        length, hd = x.shape[1], attn.head_dim
        raw = attn.k_proj(x).view(1, length, -1, hd)
        value = attn.v_proj(x).view(1, length, -1, hd) if attn.v_proj is not None else raw
        return fused.norm_rope(attn.k_norm, raw, cos, sin), fused.norm(attn.v_norm, value)

    @staticmethod
    def _q(attn: Any, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return fused.norm_rope(attn.q_norm, attn.q_proj(x).view(1, x.shape[1], -1, attn.head_dim), cos, sin)

    @staticmethod
    def _global(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_bounds: list[int], k_bounds: list[int],
                lens: torch.Tensor, causal: bool, scale: float) -> torch.Tensor:
        """Attention per virtual sequence: query rows [q_bounds[i], q_bounds[i+1]) over key rows
        [k_bounds[i], k_bounds[i+1]) (``lens``: those lengths on the device); causal aligns the last query with the last
        key (a branch's queries follow its shared keys); (rows, heads × head size)."""
        outs = []
        if global_attention.usable(query):
            spans = zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:], strict=True)
            for i, (qa, qb, ka, kb) in enumerate(spans):
                if qb > qa:
                    outs.append(global_attention.attend(query[0, qa:qb], key[0, ka:kb], value[0, ka:kb], lens[i, 0, :1],
                                                        lens[i, 1, :1], causal, scale))
            out = torch.cat(outs)
            return out.reshape(out.shape[0], -1)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            for qa, qb, ka, kb in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:], strict=True):
                if qb == qa:
                    continue
                q, k, v = (t[:, a:b].transpose(1, 2) for t, a, b in ((query, qa, qb), (key, ka, kb), (value, ka, kb)))
                mask = causal_lower_right(qb - qa, kb - ka) if causal and qb - qa != kb - ka else None
                outs.append(F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale,
                                                           is_causal=causal and mask is None,
                                                           enable_gqa=k.shape[1] != q.shape[1]).transpose(1, 2))
        out = torch.cat(outs, dim=1)
        return out.reshape(out.shape[1], -1)

    @staticmethod
    def _gather(key: torch.Tensor, value: torch.Tensor, rows: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if rows is None:
            return key, value
        return key.index_select(1, rows), value.index_select(1, rows)

    def attention(self, attn: Any, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, p: Pack) -> torch.Tensor:
        plan = p.plan
        query = self._q(attn, x, cos, sin)
        key, value = self._kv(attn, x, cos, sin)
        if attn.is_sliding:
            whole = (key[0], value[0]) if plan.images is not None else None
            key, value = self._gather(key, value, plan.s_rows)
            out = sliding_attention.attend(query[0], key[0], value[0], plan.q_cu_t, plan.s_cu_t, plan.max_q,
                                           plan.max_s, attn.scaling, self.sliding[0])
            if plan.images is not None:
                out = vision.see_whole(out, query[0], whole[0], whole[1], plan.images, attn.scaling)
            out = out.reshape(x.shape[1], -1)
        elif self.global_attend is not None:
            if isinstance(plan, PaddedPlan):
                out = self.global_attend(query[0], key[0], value[0], plan.cu, plan.cu, plan.size, attn.scaling)
            else:
                key, value = self._gather(key, value, plan.g_rows)
                out = self.global_attend(query[0], key[0], value[0], plan.q_cu_t, plan.g_cu_t, plan.max_q,
                                         attn.scaling)
            out = out.reshape(x.shape[1], -1)
        elif isinstance(plan, PaddedPlan):
            out = global_attention.padded(query, key, value, plan.cu, plan.size, self.causal_global, attn.scaling)
            out = out.reshape(x.shape[1], -1)
        else:
            key, value = self._gather(key, value, plan.g_rows)
            out = self._global(query, key, value, plan.q_cu, plan.g_cu, plan.g_lens, self.causal_global, attn.scaling)
        return attn.o_proj(out[None])

    # ---------------------------------------------------------------- layers

    def _mlp(self, layer: Any, h: torch.Tensor) -> torch.Tensor:
        x = self.norm_input(layer.pre_feedforward_layernorm, h)
        return fused.norm(layer.post_feedforward_layernorm, self.mlp(layer.mlp, x))

    def layer(self, layer: Any, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, p: Pack) -> torch.Tensor:
        attn = self.attention(layer.self_attn, self.norm_input(layer.input_layernorm, h), cos, sin, p)
        h = fused.residual_norm(h, layer.post_attention_layernorm, attn)
        return fused.add_scale(h, self.mlp_branch(self._mlp, layer, h), layer.layer_scalar)

    def last_layer_at_markers(self, layer: Any, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                              p: Pack) -> torch.Tensor:
        """The last (global) layer for the marker rows only: each over the keys of its own virtual sequence."""
        plan = p.plan
        attn = layer.self_attn
        x = self.norm_input(layer.input_layernorm, h)
        query = self._q(attn, x[:, p.markers], cos[:, p.markers], sin[:, p.markers])
        key, value = self._gather(*self._kv(attn, x, cos, sin), plan.g_rows)
        if self.global_attend is not None:
            out = self.global_attend(query[0], key[0], value[0], plan.m_cu_t, plan.g_cu_t, 1,
                                     attn.scaling).reshape(query.shape[1], -1)
        else:
            out = self._global(query, key, value, plan.m_cu, plan.g_cu, plan.m_lens, False, attn.scaling)
        hs = fused.residual_norm(h[:, p.markers], layer.post_attention_layernorm, attn.o_proj(out[None]))
        return fused.add_scale(hs, self._mlp(layer, hs), layer.layer_scalar)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(1, T, hidden) scaled token embeddings."""
        return self.lm.embed_tokens(input_ids)[None]

    def rope(self, h: torch.Tensor, seqpos: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Per layer type, the rotary tables (cos, sin) at ``seqpos``."""
        return {t: self.lm.rotary_emb(h, seqpos[None], t) for t in set(self.types)}

    def body(self, h: torch.Tensor, rope: dict[str, tuple[torch.Tensor, torch.Tensor]], p: Pack) -> torch.Tensor:
        """The layers that run on every token (all but the last in a bidirectional model)."""
        for index in range(len(self.layers) - 1 if self.prune_last else len(self.layers)):
            h = self.run(index, h, *rope[self.types[index]], p)
        return h

    def head(self, h: torch.Tensor, rope: dict[str, tuple[torch.Tensor, torch.Tensor]], p: Pack) -> torch.Tensor:
        """(M, hidden) final-normalized hidden states at the markers, in pack order."""
        if self.prune_last:
            h = self.layers[-1].packed_forward(h, *rope[self.types[-1]], p, markers_only=True)
        else:
            h = h[:, p.markers]
        return self.lm.norm(h)[0]

    def forward(self, p: Pack) -> torch.Tensor:
        """(M, hidden) final-normalized hidden states at the markers, in pack order."""
        if p.cu_cpu is None or p.segment_ids_cpu is None:
            raise ValueError("the Gemma family reads block layouts (karpo-blocks-v1)")
        p.plan = block_plan(p, self.window)
        if p.pixel_values is not None:
            if self.causal_global:
                raise ValueError("images are supported on bidirectional checkpoints only, the ones the image path was "
                                 "validated on")
            p.plan.images = vision.plan_images(p.input_ids.cpu(), p.cu_cpu.tolist(), p.segment_ids_cpu, p.seqpos.cpu(),
                                               self.backbone.config.image_token_id, self.window, p.input_ids.device)
        h = self.embed(p.input_ids)
        if p.pixel_values is not None:
            features = vision.embed(self.backbone, p.pixel_values, p.image_position_ids)
            if features.shape[0] != p.plan.images.rows.numel():
                raise ValueError(f"{p.plan.images.rows.numel()} soft tokens for {features.shape[0]} image features")
            h = h.index_copy(1, p.plan.images.rows, features[None].to(h.dtype))
        rope = self.rope(h, p.seqpos)
        return self.head(self.body(h, rope, p), rope, p)
