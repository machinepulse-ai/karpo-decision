"""The Decision Index kit's engine for a decision checkpoint: what the kit's runner loads to answer the suite in
process, one request at a time, and what the leaderboard's maintainers time.

    python -m decision_index run --engine karpo_decision.serve.engine:DecisionIndexEngine \
        --option checkpoint=path/to/checkpoint --out runs/karpo-decision

Options: ``checkpoint`` (required); ``view`` (default ``joint``; ``single`` or ``reversed``: ``karpo_decision.views``);
``max_length`` (default ``DEFAULT_MAX_LENGTH``); ``device`` (default ``cuda``); ``fp8`` (default false: the decoder
projections in FP8, ``serve.fp8``; answers move by more than bf16 noise); ``fast`` (default true: with ``fp8``,
single-question requests replay CUDA graphs captured at load, ``serve.graphs``; not with the ``blackwell`` extra's
kernels); ``triton_attention`` (default false: the global layers' attention in one Triton kernel, sm_120 GPUs,
bidirectional checkpoints and a ``max_length`` under 131,072 (31B) only, ``families.gemma.triton_attention``; answers
move by bf16 noise). A path the checkpoint's backbone does not run is refused before loading (``serve.paths``).
A request outside the contract, or with a question that does not fit ``max_length`` tokens with
its state, is ``Unsupported``.
"""

from __future__ import annotations

import time
from importlib.metadata import version
from typing import Any

import torch
from decision_index.engines import Engine, Unsupported

from ..decider import DEFAULT_MAX_LENGTH, Decider
from ..encoding import LengthError
from ..families.gemma.triton_attention import enable as enable_attention
from ..protocol import RequestError
from ..views import VIEWS, joined, views
from . import graphs
from .fp8 import quantize
from .paths import require


class DecisionIndexEngine(Engine):
    name = "karpo-decision"
    latency = ("Device-synchronized in-process request wall time: prompt, tokenization and plan, the forward passes, "
               "readout and answers; excludes loading, graph capture and warmup.")

    def __init__(self, checkpoint: str, view: str = "joint", max_length: int = DEFAULT_MAX_LENGTH,
                 device: str = "cuda", fp8: bool = False, fast: bool = True, triton_attention: bool = False,
                 **options: Any) -> None:
        super().__init__(checkpoint=checkpoint, view=view, max_length=max_length, device=device, fp8=fp8, fast=fast,
                         triton_attention=triton_attention, **options)
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        for name, value in (("fp8", fp8), ("fast", fast), ("triton_attention", triton_attention)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false, not {value!r}")
        supported = require(checkpoint, fp8=fp8, triton_attention=triton_attention)
        if torch.device(device).index is not None:
            torch.cuda.set_device(torch.device(device))
        decider = Decider.from_pretrained(checkpoint, device=device, max_length=int(max_length))
        if fp8:
            quantize(decider.model)
        if triton_attention:
            enable_attention(decider.model, decider.max_length)
        if fp8:
            decider.decide(graphs.WARMUP_REQUEST)  # the fused kernels compile on it with or without graphs
        graphed = fp8 and fast and "graphs" in supported and graphs.fast(decider)
        self.decider = decider
        self.view = view
        self.path = "fp8-graphs" if graphed else "fp8" if fp8 else "plain"
        self.kernels = ["triton-attention"] if triton_attention else []
        self.provenance = {"kind": "decision-checkpoint", "checkpoint": checkpoint, "path": self.path, "view": view,
                           "template": decider.config.template, "attention": decider.config.attention,
                           "kernels": self.kernels}
        #: Milliseconds the last request spent in encoding (template, tokenizer, plan).
        self.last_encode_ms = 0.0

    def __call__(self, state: Any, questions: dict[str, Any]) -> tuple[dict[str, Any], None]:
        request = {"state": state, "questions": questions}
        start = time.perf_counter()
        readings = views(request, self.view)
        try:
            plans = [self.decider.plan(reading) for reading in readings]
        except (RequestError, LengthError) as error:
            raise Unsupported(f"{type(error).__name__}: {error}") from error
        self.last_encode_ms = (time.perf_counter() - start) * 1000
        return {"model": self.name, **joined(request, self.decider.decide(readings, plans), self.view)}, None

    def runtime(self) -> dict[str, Any]:
        return {"gpu": torch.cuda.get_device_name(), "path": self.path, "kernels": self.kernels, "view": self.view,
                "max_length": self.decider.max_length,
                **{name: version(name) for name in ("torch", "transformers")}}

    def synchronize(self) -> None:
        torch.cuda.synchronize()
