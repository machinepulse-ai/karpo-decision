"""What every family's packed forward pass shares: sequences laid end to end into one padding-free ``Pack``, and the
layer loop around a family's own layer code (``families/<name>/packed.py``).

A family's ``Packed`` subclasses ``PackedLayers`` and supplies ``layer`` (one decoder layer over every token),
``last_layer_at_markers`` (the last layer for the marker rows only) and ``forward``. The base gives each decoder layer a
``packed_forward(h, cos, sin, pack, markers_only=False)`` method, so per-layer wrappers act on it (training registers
it with FSDP2 so a sharded layer gathers its weights for it, and recomputes it under activation checkpointing), and
runs the layers with the training run's recompute policy. The module tree stays the backbone's own.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from ..encoding import Encoded

#: The M-RoPE axes: time, height, width.
MROPE_AXES = 3


@dataclass
class Pack:
    """Sequences laid end to end, on the model's device."""

    input_ids: torch.Tensor  # (T,)
    seqpos: torch.Tensor  # (T,) position inside the token's own sequence
    cu: torch.Tensor  # (n + 1,) int32 sequence boundaries
    #: The same, int64 on CPU (the family plans its attention with it).
    cu_cpu: torch.Tensor | None
    max_length: int
    markers: torch.Tensor  # (M,) positions of every readout marker in the pack
    marker_cu: torch.Tensor  # (n + 1,) int32 marker boundaries per sequence
    max_markers: int
    #: Block layouts only (``blocks.py``): each token's segment (0 a sequence's shared block, k its k-th branch), on
    #: the device and on CPU (the family plans its attention with it); ``seqpos`` then holds the layout's positions.
    segment_ids: torch.Tensor | None = None
    segment_ids_cpu: torch.Tensor | None = None
    #: Joint layouts only: (3, 1, T) M-RoPE positions, and the images' pixels and grids.
    rope: torch.Tensor | None = None
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    #: Qwen: right-padded segments laid end to end when ``cu_cpu`` is None (one batch row each in the delta rule).
    segments: int = 1
    #: A family's plan of the pack's attention, computed once by its forward pass and reused by every layer (and by
    #: recomputation in backward).
    plan: Any = None

    @property
    def tokens(self) -> int:
        return int(self.input_ids.numel())


def positions(backbone: Any, sequence: Encoded) -> torch.Tensor:
    """(3, T) M-RoPE positions of one joint sequence: plain 0..T-1 on every axis unless it has images."""
    if sequence.image_grid_thw is None:
        return torch.arange(sequence.length).expand(MROPE_AXES, -1)
    rope, _ = backbone.get_rope_index(sequence.input_ids[None], sequence.mm_token_type_ids[None],
                                      image_grid_thw=sequence.image_grid_thw)
    return rope[:, 0]


def pack(sequences: Sequence[Encoded], backbone: Any, device: torch.device) -> Pack:
    lengths = [s.length for s in sequences]
    blocks = [s.segment_ids is not None for s in sequences]
    if any(blocks) and not all(blocks):
        raise ValueError("a pack mixes block layouts with plain sequences")
    segment_ids = torch.cat([s.segment_ids for s in sequences]).to(torch.int32) if all(blocks) else None
    seqpos = (torch.cat([s.positions for s in sequences]) if all(blocks)
              else torch.cat([torch.arange(n) for n in lengths]))
    cu_cpu = F.pad(torch.tensor(lengths).cumsum(0), (1, 0))
    starts = cu_cpu[:-1].tolist()
    marker_counts = [len(s.keys) for s in sequences]
    pixels = [s.pixel_values for s in sequences if s.pixel_values is not None]
    grids = [s.image_grid_thw for s in sequences if s.image_grid_thw is not None]
    rope = None if all(blocks) else torch.cat([positions(backbone, s) for s in sequences], dim=1)[:, None, :]
    return Pack(
        input_ids=torch.cat([s.input_ids for s in sequences]).to(device, non_blocking=True),
        seqpos=seqpos.to(device, non_blocking=True),
        cu=cu_cpu.to(device, torch.int32, non_blocking=True),
        cu_cpu=cu_cpu,
        max_length=max(lengths),
        markers=torch.cat([s.markers + start for s, start in zip(sequences, starts, strict=True)]).to(device),
        marker_cu=F.pad(torch.tensor(marker_counts).cumsum(0), (1, 0)).to(device, torch.int32),
        max_markers=max(marker_counts),
        segment_ids=None if segment_ids is None else segment_ids.to(device, non_blocking=True),
        segment_ids_cpu=segment_ids,
        rope=None if rope is None else rope.to(device, non_blocking=True),
        pixel_values=torch.cat(pixels).to(device) if pixels else None,
        image_grid_thw=torch.cat(grids).to(device) if grids else None,
    )


#: Ops whose outputs a selectively recomputed layer keeps for backward: the matmuls and the attention kernels, the
#: costly part; norms, rotations, activations, gathers and copies run again in backward from those (``karpo.*``: our
#: own operators, registered when their module is imported).
KEPT_OPS = ("aten.mm.default", "aten.addmm.default", "aten.bmm.default", "torch_attn._varlen_attn.default",
            "aten._efficient_attention_forward.default", "aten._scaled_dot_product_efficient_attention.default",
            "aten._flash_attention_forward.default", "cudnn.sdpa_fwd.default", "karpo.fa4_sliding.default")


def _kept_ops() -> set:
    kept = set()
    for name in KEPT_OPS:
        namespace, op, overload = name.split(".")
        with contextlib.suppress(AttributeError, RuntimeError):  # a kernel this build or environment does not have
            kept.add(getattr(getattr(getattr(torch.ops, namespace), op), overload))
    return kept


def _selective() -> Any:
    kept = _kept_ops()

    def policy(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
        return CheckpointPolicy.MUST_SAVE if op in kept else CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy)


class PackedLayers(torch.nn.Module):
    """The decoder layers' ``packed_forward`` methods and the recompute policy of the layers that run on every token.

    ``recompute``: "layer" recomputes whole layers in backward (every ``checkpoint_every``-th, or all but every
    ``keep_every``-th), "mlp" only each layer's MLP (``mlp_branch``), "selective"
    recomputes whole layers as "layer" does and runs the remaining layers keeping only their matmul and attention
    outputs (``KEPT_OPS``; the cheap rest runs again in backward). ``offload`` keeps the activations of layers that are
    not recomputed in pinned host memory instead of on the GPU. Recomputation does not save and restore the random
    number generators: the layers draw no random numbers (no dropout).

    A pack of more than ``long_tokens`` tokens keeps about the activations of a ``long_tokens`` pack: of the layers
    ``keep_every`` keeps, the share ``long_tokens / tokens`` (spread evenly) keep theirs and the others are recomputed
    whole, so its memory stays near a ``long_tokens`` pack's whatever its length. Chosen per pack before its
    forward; the collectives are the same either way.
    """

    def __init__(self, layers: Sequence[torch.nn.Module], checkpoint_every: int = 0, recompute: str = "layer",
                 offload: bool = False, keep_every: int = 0, long_tokens: int = 0) -> None:
        super().__init__()
        if recompute not in ("layer", "mlp", "selective"):
            raise ValueError(f"recompute must be layer, mlp or selective, not {recompute}")
        self.layers = list(layers)
        self.checkpoint_every = checkpoint_every
        self.recompute = recompute
        self.offload = offload
        self.keep_every = keep_every
        self.long_tokens = long_tokens
        for layer in self.layers:
            layer.packed_forward = MethodType(self._packed_layer, layer)

    def layer(self, layer: Any, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, p: Pack) -> torch.Tensor:
        raise NotImplementedError

    def last_layer_at_markers(self, layer: Any, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                              p: Pack) -> torch.Tensor:
        raise NotImplementedError

    def _packed_layer(self, layer: Any, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, p: Pack,
                      markers_only: bool = False) -> torch.Tensor:
        return self.last_layer_at_markers(layer, h, cos, sin, p) if markers_only else self.layer(layer, h, cos, sin, p)

    def mlp_branch(self, mlp: Callable[[Any, torch.Tensor], torch.Tensor], layer: Any, h: torch.Tensor) -> torch.Tensor:
        """``mlp(layer, h)``, recomputed in backward under ``recompute="mlp"``."""
        if self.recompute == "mlp" and torch.is_grad_enabled():
            return checkpoint(mlp, layer, h, use_reentrant=False, preserve_rng_state=False)
        return mlp(layer, h)

    def recomputed_whole(self, index: int, tokens: int) -> bool:
        """Whether layer ``index`` runs again whole in backward on a pack of ``tokens`` tokens."""
        if not self.keep_every:
            return bool(self.checkpoint_every) and index % self.checkpoint_every == 0
        if index % self.keep_every:
            return True
        if not self.long_tokens or tokens <= self.long_tokens:
            return False
        kept, k = -(-len(self.layers) // self.keep_every), index // self.keep_every
        share = kept * self.long_tokens // tokens  # of the kept layers, these keep theirs: the k-th iff it starts one
        return (k + 1) * share // kept == k * share // kept

    def run(self, index: int, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, p: Pack) -> torch.Tensor:
        """Layer ``index`` over every token, under the recompute policy."""
        layer = self.layers[index]
        grad = torch.is_grad_enabled()
        if self.recompute in ("layer", "selective") and self.recomputed_whole(index, p.tokens) and grad:
            return checkpoint(layer.packed_forward, h, cos, sin, p, use_reentrant=False, preserve_rng_state=False)
        if self.recompute == "selective" and grad:
            return checkpoint(layer.packed_forward, h, cos, sin, p, use_reentrant=False, context_fn=_selective,
                              preserve_rng_state=False)
        if self.offload and grad:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                return layer.packed_forward(h, cos, sin, p)
        return layer.packed_forward(h, cos, sin, p)
