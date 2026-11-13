"""Bucketed CUDA graphs over the packed forward pass: a request replays the graph of its length bucket instead of
launching every kernel of every layer from the CPU.

A request (one sequence of length L) is right-padded to its bucket, and the bucket's graph runs every layer that
runs on every token (``qwen.Packed.body``). The padding is a sequence of its own for attention (``cu`` is
[0, L, bucket], read from device memory on every replay), and the causal convolution and delta rule run the whole
bucket as one sequence, where the padding comes after every real token and so never reaches one (``Pack.cu_cpu`` is
None). Padding rows hold whatever the input buffers held before, always finite values; no real row reads them.

With ``segments`` > 1, a pack of up to that many sequences (the readings of one request, read in one pass) replays a
graph of n segments, each sequence right-padded to the bucket of the longest: ``cu`` is [0, L1, P, P + L2, 2P, ...],
the convolution's positions restart at every segment and the delta rule runs one batch row per segment.

Outside the graph: the embeddings before it (with the vision tower, whose inputs vary in size), and the head after it
(the last layer at the markers for a bidirectional model, the final norm), on the real rows only. A pack with no graph
(more sequences than ``segments``, or longer than the largest bucket) runs the layers directly.

Every graph writes the body's output and the rotary tables into buffers shared by all buckets, so a graph keeps none of
its intermediate tensors after capture and the buckets share one memory pool the size of the largest bucket's pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..families.qwen import Packed
from ..model.packed import Pack

#: (step, up to): 256-token buckets up to 4k, 512 up to 16k, 1,024 up to 32k.
BUCKETS = ((256, 4096), (512, 16384), (1024, 32768))
#: Passes before a capture: the first compiles the layers and tunes the kernels for the bucket's length.
WARMUP = 2

Buckets = tuple[tuple[int, int], ...]


def bucket(length: int, buckets: Buckets = BUCKETS) -> int | None:
    """The padded length of a sequence, or None beyond the largest bucket."""
    for step, top in buckets:
        if length <= top:
            return -(-length // step) * step
    return None


def sizes(max_tokens: int, buckets: Buckets = BUCKETS) -> list[int]:
    """Every bucket up to ``max_tokens``."""
    out, start = [], 0
    for step, top in buckets:
        out += range(start + step, min(top, max_tokens) + 1, step)
        start = top
    return out


@dataclass
class Graph:
    graph: torch.cuda.CUDAGraph
    cu: torch.Tensor  # (2n + 1,) int32: [0, L1, P, P + L2, 2P, ...]
    seqpos: torch.Tensor  # (nP,) position inside each segment; the graph reads it on every replay


class CudaGraphs:
    """A ``Packed`` whose packs of up to ``segments`` sequences replay the CUDA graph of their bucket (callable
    like it). Graphs of n > 1 segments cover buckets up to ``segment_tokens`` (default ``max_tokens``)."""

    def __init__(self, model: Packed, max_tokens: int, buckets: Buckets = BUCKETS, warmup: int = WARMUP,
                 segments: int = 1, segment_tokens: int | None = None) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.buckets = buckets
        self.warmup = warmup
        self.segments = segments
        self.segment_tokens = min(segment_tokens or max_tokens, max_tokens)
        width = max(max_tokens, segments * self.segment_tokens)
        embed = model.lm.embed_tokens.weight
        # Input buffers shared by every graph (each reads its leading slice).
        self.hidden = torch.zeros(1, width, embed.shape[1], dtype=embed.dtype, device=embed.device)
        self.rope = torch.zeros(3, 1, width, dtype=torch.long, device=embed.device)
        # Output buffers shared by every graph: the body's output and the rotary tables the head reads.
        self.out = torch.zeros_like(self.hidden)
        cos, _ = model.lm.rotary_emb(self.hidden[:, :1], self.rope[..., :1])
        self.cos = torch.zeros(1, width, *cos.shape[2:], dtype=cos.dtype, device=cos.device)
        self.sin = torch.zeros_like(self.cos)
        self.graphs: dict[tuple[int, int], Graph] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.replays = 0

    def padded(self, n: int, size: int, cu: torch.Tensor) -> Pack:
        """The pack a graph runs: n segments of ``size`` tokens for the causal layers, 2n sequences for attention."""
        device = cu.device
        markers = torch.zeros(0, dtype=torch.long, device=device)
        return Pack(input_ids=markers, rope=self.rope[..., :n * size],
                    seqpos=torch.arange(size, device=device).repeat(n), cu=cu, cu_cpu=None, max_length=size,
                    markers=markers, marker_cu=cu[:1], max_markers=0, pixel_values=None, image_grid_thw=None,
                    segments=n)

    @torch.inference_mode()
    def capture(self, size: int, n: int = 1) -> None:
        bounds = [0] + [edge for i in range(n) for edge in ((i + 1) * size, (i + 1) * size)]
        cu = torch.tensor(bounds, dtype=torch.int32, device=self.hidden.device)
        p, hidden, tokens = self.padded(n, size, cu), self.hidden[:, :n * size], n * size

        def run() -> None:
            cos, sin = self.model.lm.rotary_emb(hidden, p.rope)
            self.out[:, :tokens].copy_(self.model.body(hidden, cos, sin, p))
            self.cos[:, :tokens].copy_(cos)
            self.sin[:, :tokens].copy_(sin)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                run()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            run()
        self.graphs[n, size] = Graph(graph, cu, p.seqpos)

    def capture_all(self) -> None:
        """Every bucket up to ``max_tokens`` (n > 1 segments: up to ``segment_tokens``), the most tokens first (the
        shared pool is then sized once)."""
        todo = [(n * size, n, size) for n in range(1, self.segments + 1)
                for size in sizes(self.max_tokens if n == 1 else self.segment_tokens, self.buckets)]
        for _, n, size in sorted(todo, reverse=True):
            self.capture(size, n)

    def key(self, lengths: list[int]) -> tuple[int, int] | None:
        """The graph a pack of sequences of these lengths replays, or None."""
        size = bucket(max(lengths), self.buckets)
        key = (len(lengths), size)
        return key if size is not None and key in self.graphs else None

    def covers(self, length: int) -> bool:
        return self.key([length]) is not None

    def __call__(self, p: Pack) -> torch.Tensor:
        """(M, hidden) final-normalized hidden states at the markers, as ``Packed`` gives them."""
        lengths = p.cu_cpu.diff().tolist()
        key = self.key(lengths)
        if key is None:
            return self.model(p)
        n, size = key
        g = self.graphs[key]
        embedded = self.model.embed(p)
        if n == 1:
            self.hidden[:, :lengths[0]].copy_(embedded)
            self.rope[..., :lengths[0]].copy_(p.rope)
            g.cu[1].fill_(lengths[0])
        else:
            starts = p.cu_cpu.tolist()
            for i, (start, length) in enumerate(zip(starts, lengths, strict=False)):
                self.hidden[:, i * size:i * size + length].copy_(embedded[:, start:start + length])
                self.rope[..., i * size:i * size + length].copy_(p.rope[..., start:start + length])
            bounds = [0] + [edge for i, length in enumerate(lengths) for edge in (i * size + length, (i + 1) * size)]
            g.cu.copy_(torch.tensor(bounds, dtype=torch.int32))
        g.graph.replay()
        self.replays += 1
        if n == 1:
            rows = slice(0, lengths[0])
            return self.model.head(self.out[:, rows], self.cos[:, rows], self.sin[:, rows], p)
        real = torch.cat([torch.arange(i * size, i * size + length) for i, length in enumerate(lengths)])
        real = real.to(self.out.device, non_blocking=True)
        return self.model.head(self.out[:, real], self.cos[:, real], self.sin[:, real], p)


#: (step, up to): 64-token buckets up to 2k (most requests are a few hundred tokens), then coarser.
FINE_BUCKETS = ((64, 2048), (128, 4096), (256, 8192), (512, 16384), (1024, 32768), (2048, 65536))
#: Recompilations allowed per compiled layer as request lengths vary.
RECOMPILE_LIMIT = 16


def fast(decider: Any) -> bool:
    """The layers the graphs run compiled, then a CUDA graph captured for every bucket up to ``decider.max_length``
    (minutes); False off CUDA. A bidirectional model's last layer runs at the markers only, outside the graphs, and
    stays eager so that no request recompiles it."""
    if decider.device.type != "cuda":
        return False
    model = decider.model
    torch._dynamo.config.recompile_limit = RECOMPILE_LIMIT
    for layer in model.layers[:-1] if model.prune_last else model.layers:
        layer.packed_forward = torch.compile(layer.packed_forward, dynamic=True)
    graphs = CudaGraphs(model, min(decider.max_length, FINE_BUCKETS[-1][1]), FINE_BUCKETS, WARMUP)
    with torch.cuda.device(decider.device):
        graphs.capture_all()
    decider.model = graphs
    return True
