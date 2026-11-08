"""CUDA graphs for single-question requests on one GPU. A pack of one single-block sequence (a request with one
question) of at most ``MAX_TOKENS`` tokens is padded to its bucket (a multiple of ``STEP``) and replays that bucket's
graph of the embedding and the layers that run on every token; the last layer at the marker, the final norm and the
readout run as in the plain path. Any other pack runs the plain path.

    from karpo_decision.serve.graphs import fast
    fast(decider)  # captures every bucket at load (minutes); False where it does not apply
"""

from __future__ import annotations

import torch

from ..decider import Decider
from ..families.gemma import global_attention, sliding_attention
from ..families.gemma.packed import Packed, PaddedPlan, block_plan
from ..model.packed import Pack

#: Bucket step: fine enough that padding costs little GEMM work.
STEP = 32
#: The longest sequence with a graph; a longer one gains little (its GPU time dwarfs the launches) and runs plain.
MAX_TOKENS = 8192
#: Passes before a capture: the first compiles and tunes kernels for the bucket's length.
WARMUP = 2
#: Answered before capture, so the fused kernels compile on it as on the plain path (a kernel's reduction order follows
#: the shapes of its first call), and once after, so the first request compiles nothing.
WARMUP_REQUEST = {"state": {"page": {"url": "about:blank"}},
                  "questions": {"blank": {"type": "noul", "instructions": "Is the page blank?"}}}


def bucket(length: int, top: int = MAX_TOKENS) -> int | None:
    """The padded length of a sequence, or None beyond ``top``."""
    size = -(-length // STEP) * STEP
    return size if length > 0 and size <= top else None


def covered(p: Pack, top: int = MAX_TOKENS) -> bool:
    """One single-block sequence (a request with one question) of at most ``top`` tokens."""
    return (p.cu_cpu is not None and len(p.cu_cpu) == 2 and p.segment_ids_cpu is not None
            and not bool(p.segment_ids_cpu.any()) and p.markers.numel() == 1 and p.max_length <= top)


def kernels_supported(device: torch.device) -> bool:
    """False where the ``blackwell`` extra's kernels would run: the padded pass is written for the memory-efficient and
    FA2 kernels."""

    def probe(head: int) -> torch.Tensor:
        return torch.empty(1, 1, head, dtype=torch.bfloat16, device=device)

    return not (global_attention.usable(probe(global_attention.HEAD))
                or sliding_attention.usable(probe(sliding_attention.HEAD)))


class Graphs:
    """``Packed`` (callable like it) whose single-question packs replay the graph of their bucket; a bucket without a
    graph runs the same padded pass eagerly."""

    def __init__(self, model: Packed, max_tokens: int = MAX_TOKENS, warmup: int = WARMUP) -> None:
        self.model, self.max_tokens, self.warmup = model, max_tokens, warmup
        weight = model.lm.embed_tokens.weight
        self.input_ids = torch.zeros(max_tokens, dtype=torch.long, device=weight.device)
        self.seqpos = torch.arange(max_tokens, device=weight.device)
        self.cu = torch.zeros(3, dtype=torch.int32, device=weight.device)  # [0, L, P], read on every replay
        self.out = torch.zeros(1, max_tokens, weight.shape[1], dtype=weight.dtype, device=weight.device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.replays = 0

    def _pack(self, size: int) -> Pack:
        empty = torch.zeros(0, dtype=torch.long, device=self.cu.device)
        return Pack(input_ids=self.input_ids[:size], seqpos=self.seqpos[:size], cu=self.cu, cu_cpu=None,
                    max_length=size, markers=empty, marker_cu=self.cu[:1], max_markers=0,
                    plan=PaddedPlan(self.cu, size))

    def _run(self, size: int) -> None:
        """A bucket's padded pass; its rows land in ``out``."""
        p = self._pack(size)
        h = self.model.embed(p.input_ids)
        self.out[:, :size].copy_(self.model.body(h, self.model.rope(h, p.seqpos), p))

    @torch.inference_mode()
    def capture(self, size: int, pool: tuple) -> None:
        self.cu.copy_(torch.tensor([0, size, size], dtype=torch.int32))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                self._run(size)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            self._run(size)
        self.graphs[size] = graph

    def capture_all(self) -> None:
        """Every bucket up to ``max_tokens``, the longest first (the shared memory pool is sized once)."""
        pool = torch.cuda.graph_pool_handle()
        for size in range(self.max_tokens // STEP * STEP, 0, -STEP):
            self.capture(size, pool)

    def __call__(self, p: Pack) -> torch.Tensor:
        """(M, hidden) final-normalized hidden states at the markers, as ``Packed`` gives them."""
        size = bucket(p.max_length, self.max_tokens) if covered(p, self.max_tokens) else None
        if size is None:
            return self.model(p)
        p.plan = block_plan(p, self.model.window)
        length = p.max_length
        self.input_ids[:length].copy_(p.input_ids)
        self.cu[1].fill_(length)
        self.cu[2].fill_(size)
        graph = self.graphs.get(size)
        if graph is None:
            self._run(size)
        else:
            graph.replay()
            self.replays += 1
        h = self.out[:, :length]
        last = self.model.types[-1]
        return self.model.head(h, {last: self.model.lm.rotary_emb(h, p.seqpos[None], last)}, p)


def fast(decider: Decider, max_tokens: int = MAX_TOKENS) -> bool:
    """``Graphs`` in front of ``decider.model``, every bucket captured; False (the plain path kept) off CUDA or where
    the ``blackwell`` extra's kernels would run."""
    if decider.device.type != "cuda" or not kernels_supported(decider.device):
        return False
    decider.decide(WARMUP_REQUEST)
    graphs = Graphs(decider.model, min(max_tokens, decider.max_length))
    with torch.cuda.device(decider.device):
        graphs.capture_all()
    decider.model = graphs
    decider.decide(WARMUP_REQUEST)
    return True
