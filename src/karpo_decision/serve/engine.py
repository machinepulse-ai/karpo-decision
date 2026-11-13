"""The Decision Index kit's engine for a decision checkpoint: what the kit's runner loads to answer the suite in
process, one request at a time, and what the leaderboard's maintainers time.

    python -m decision_index run --engine karpo_decision.serve.engine:DecisionIndexEngine \
        --option checkpoint=path/to/checkpoint --out runs/karpo-decision

Options: ``checkpoint`` (required); ``view`` (default ``joint``; ``single`` or ``reversed``: ``karpo_decision.views``);
``max_length`` (default ``DEFAULT_MAX_LENGTH``); ``device`` (default ``cuda``); ``fp8`` (default false, every
checkpoint: the decoder projections in FP8, ``serve.fp8`` or ``serve.qwen_fp8``; answers move by more than bf16 noise);
``fast`` (default true; with ``fp8`` alone, false skips the CUDA graphs captured at load: ``serve.graphs`` for
single-question text-only requests, not with the ``blackwell`` extra's kernels, or ``serve.qwen_graphs``); ``graphs``
(default false, Qwen3.5 checkpoints: CUDA graphs on the bf16 path); ``triton_attention`` (default false, Gemma
checkpoints: the global layers' attention in one Triton kernel, sm_120 GPUs, bidirectional checkpoints and a
``max_length`` within the checkpoint's limit, 131,072 for the 31B, only; ``families.gemma.triton_attention``; answers
move by bf16 noise); ``image_root`` (default none: the directory the kit's image references are relative to, read with
the kit's loader and its pixel cap). A path the checkpoint's backbone does not run is refused before loading
(``serve.paths``).

A request outside the contract, with an image that cannot be read, or with a question that does not fit ``max_length``
tokens with its state and images is ``Unsupported``, and so is a request with images on the 12B, which reads text only.
On the 31B a request with images (the vision suite) runs through the backbone's vision tower
(``families/gemma/vision.py``) and its meta reports the images' pixels and soft tokens; a checkpoint without that Gemma
vision unit returns meta None (Qwen3.5 reads images through its own layout).
"""

from __future__ import annotations

import time
from importlib.metadata import version
from typing import Any

import torch
from decision_index.engines import Engine, Unsupported

from ..decider import DEFAULT_MAX_LENGTH, Decider
from ..encoding import LengthError, TextOnlyError
from ..families.gemma import vision
from ..families.gemma.triton_attention import enable as enable_attention
from ..families.qwen import MODEL_TYPE as QWEN
from ..protocol import RequestError
from ..views import VIEWS, joined, views
from . import graphs, qwen_fp8, qwen_graphs
from .fp8 import quantize
from .paths import FP8, GRAPHS, GRAPHS_OPTION, TRITON_ATTENTION, backbone_type, path_name, require

#: ``graphs`` is also an option of the engine, which would hide the module in ``__init__``.
gemma_graphs = graphs


class DecisionIndexEngine(Engine):
    name = "karpo-decision"
    latency = ("Device-synchronized in-process request wall time: prompt, tokenization and plan, the forward passes, "
               "readout and answers (with images, also reading them and the vision tower); excludes loading, graph "
               "capture and warmup.")

    def __init__(self, checkpoint: str, view: str = "joint", max_length: int = DEFAULT_MAX_LENGTH,
                 device: str = "cuda", fp8: bool = False, fast: bool = True, triton_attention: bool = False,
                 graphs: bool = False, image_root: str | None = None, **options: Any) -> None:
        super().__init__(checkpoint=checkpoint, view=view, max_length=max_length, device=device, fp8=fp8, fast=fast,
                         triton_attention=triton_attention, graphs=graphs, image_root=image_root, **options)
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        for name, value in ((FP8, fp8), ("fast", fast), (TRITON_ATTENTION, triton_attention), (GRAPHS_OPTION, graphs)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false, not {value!r}")
        supported = require(checkpoint, fp8=fp8, triton_attention=triton_attention, graphs=graphs)
        qwen = backbone_type(checkpoint) == QWEN
        if torch.device(device).index is not None:
            torch.cuda.set_device(torch.device(device))
        decider = Decider.from_pretrained(checkpoint, device=device, max_length=int(max_length))
        if fp8:
            (qwen_fp8.quantize if qwen else quantize)(decider.model)
        if triton_attention:
            enable_attention(decider.model, decider.max_length)
        if qwen:
            captured = (graphs or (fp8 and fast)) and qwen_graphs.fast(decider)
        else:
            if fp8:
                decider.decide(gemma_graphs.WARMUP_REQUEST)  # the fused kernels compile on it with or without graphs
            captured = fp8 and fast and GRAPHS in supported and gemma_graphs.fast(decider)
        self.decider = decider
        self.view = view
        self.image_root = image_root
        self.vision = getattr(decider.encoder.processor, "vision", None)
        self.path = path_name(fp8, bool(captured))
        self.kernels = ["triton-attention"] if triton_attention else []
        self.provenance = {"kind": "decision-checkpoint", "checkpoint": checkpoint, "path": self.path, "view": view,
                           "template": decider.config.template, "attention": decider.config.attention,
                           "kernels": self.kernels,
                           "images": (f"Gemma 4's image processor at {vision.MAX_SOFT_TOKENS} soft tokens per image "
                                      "at most, after the kit's pixel cap; the backbone's vision tower; the soft "
                                      "tokens in the shared block, each image whole in the sliding layers")
                           if self.vision else None}
        #: Milliseconds the last request spent in encoding (template, tokenizer, plan).
        self.last_encode_ms = 0.0

    def __call__(self, state: Any, questions: dict[str, Any], images: Any = (),
                 prompt: Any = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
        request = {"state": state, "questions": questions}
        start = time.perf_counter()
        try:
            if images:
                from decision_index.vision import images as kit_images

                request["images"] = [kit_images.pil(ref, self.image_root, kit_images.MAX_PIXELS) for ref in images]
            readings = views(request, self.view)
            plans = [self.decider.plan(reading) for reading in readings]
        except (RequestError, LengthError, TextOnlyError, OSError) as error:
            raise Unsupported(f"{type(error).__name__}: {error}") from error
        self.last_encode_ms = (time.perf_counter() - start) * 1000
        response = {"model": self.name, **joined(request, self.decider.decide(readings, plans), self.view)}
        if not images or self.vision is None:
            return response, None
        pixels, tokens = self.vision.summary(plans[0][0].image_position_ids)
        return response, {"image_pixels": pixels, "image_tokens": tokens}

    def runtime(self) -> dict[str, Any]:
        return {"gpu": torch.cuda.get_device_name(), "path": self.path, "kernels": self.kernels, "view": self.view,
                "max_length": self.decider.max_length,
                **{name: version(name) for name in ("torch", "transformers")}}

    def synchronize(self) -> None:
        torch.cuda.synchronize()
