"""Qwen3.5 FP8 projections and CUDA graphs on a small random backbone (CUDA, flash-linear-attention)."""

import importlib.util
import inspect
import os

import pytest
import torch

from karpo_decision.encoding import Encoded
from karpo_decision.families.qwen import Packed
from karpo_decision.model.packed import pack

PATH = os.environ.get("KARPO_TOKENIZER")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(importlib.util.find_spec("fla") is None, reason="needs flash-linear-attention"),
    pytest.mark.skipif(not PATH, reason="set KARPO_TOKENIZER to a Qwen3.5 checkpoint directory"),
]
LAYERS = 8
#: Plain-text token ids used for random sequences (well below the special tokens).
TOKENS = (1000, 50_000)


def build_backbone():
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    config = AutoConfig.from_pretrained(PATH)
    text = config.text_config
    text.hidden_size, text.intermediate_size = 256, 512
    text.num_attention_heads, text.num_key_value_heads = 2, 1
    text.linear_num_key_heads, text.linear_num_value_heads = 2, 4
    text.linear_key_head_dim = text.linear_value_head_dim = 64
    text.num_hidden_layers, text.layer_types = LAYERS, text.layer_types[:LAYERS]
    vision = config.vision_config
    vision.depth, vision.hidden_size, vision.intermediate_size, vision.num_heads = 1, 64, 128, 2
    vision.out_hidden_size = text.hidden_size
    torch.manual_seed(0)
    return Qwen3_5Model(config).to("cuda", torch.bfloat16).eval()


@pytest.fixture(scope="module")
def backbone():
    return build_backbone()


def sequences(lengths, questions):
    generator = torch.Generator().manual_seed(1)
    out = []
    for n, k in zip(lengths, questions, strict=True):
        ids = torch.randint(*TOKENS, (n,), generator=generator)
        markers = torch.sort(torch.randperm(n - 1, generator=generator)[: k - 1] + 1).values
        markers = torch.cat([markers, torch.tensor([n - 1])])
        out.append(Encoded(keys=tuple(f"q{i}" for i in range(k)), counts=(4,) * k, input_ids=ids, markers=markers))
    return out


def test_fp8_reads_out_close_to_bf16():
    from karpo_decision.serve import qwen_fp8

    backbone = build_backbone()  # its own: quantizing is in place
    seqs = sequences([300, 700], [2, 3])
    device = torch.device("cuda")
    with torch.inference_mode():
        expected = Packed(backbone, "bidirectional")(pack(seqs, backbone, device)).float()
        model = Packed(backbone, "bidirectional")
        qwen_fp8.quantize(model)
        got = model(pack(seqs, backbone, device)).float()
    cosine = torch.nn.functional.cosine_similarity(got, expected, dim=-1)
    assert cosine.min().item() > 0.98, cosine


def image_sequences():
    """Real processor encodings of requests with an image, mixed with a text-only one (M-RoPE, feature order)."""
    from PIL import Image

    from karpo_decision.checkpoint import load_processor
    from karpo_decision.codes import answer_codes
    from karpo_decision.config import DEFAULT_MAX_PIXELS
    from karpo_decision.encoding import Encoder
    from karpo_decision.families.qwen import MARKER
    from karpo_decision.templates import JointTemplate

    processor = load_processor(PATH, DEFAULT_MAX_PIXELS)
    encoder = Encoder(processor, answer_codes(processor.tokenizer)[0], JointTemplate(), MARKER)
    generator = torch.Generator().manual_seed(2)

    def image(width, height):
        pixels = torch.randint(0, 255, (height, width, 3), generator=generator, dtype=torch.uint8).numpy()
        return Image.fromarray(pixels)

    question = {"type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": "second", "c": None}}
    requests = [
        {"state": {"page": "checkout", "viewport_screenshot": image(320, 256)}, "questions": {"q": question}},
        {"state": "text only", "questions": {"q": question, "r": {"type": "noul"}}},
        {"state": "two images", "questions": {"q": question}, "images": [image(256, 320), image(288, 288)]},
    ]
    return [encoder.encode(r) for r in requests]


@pytest.mark.parametrize("attention", ["bidirectional", "causal"])
def test_cuda_graphs_match_the_packed_pass(backbone, attention):
    """A request replayed from its bucket's CUDA graph (right-padded, or exactly a bucket long) reads out as the
    packed pass does, with an image too."""
    from karpo_decision.serve.qwen_graphs import CudaGraphs

    model = Packed(backbone, attention)
    graphs = CudaGraphs(model, 1024)
    graphs.capture_all()
    device = torch.device("cuda")
    for count, seq in enumerate([*sequences([37, 256, 700], [1, 3, 2]), image_sequences()[0]], start=1):
        with torch.inference_mode():
            p = pack([seq], backbone, device)
            expected, got = model(p).float(), graphs(p).float()
        assert graphs.replays == count
        cosine = torch.nn.functional.cosine_similarity(got, expected, dim=-1)
        assert cosine.min().item() > 0.999, (seq.length, cosine)


@pytest.mark.parametrize("attention", ["bidirectional", "causal"])
def test_cuda_graphs_read_two_sequences_in_one_pass(backbone, attention):
    """Two sequences (the readings of one request) replay one graph of two segments, each right-padded to the bucket
    of the longer, and read out as the packed pass does; a sequence alone still replays a one-segment graph."""
    from karpo_decision.serve.qwen_graphs import CudaGraphs

    model = Packed(backbone, attention)
    graphs = CudaGraphs(model, 1024, segments=2)
    graphs.capture_all()
    device = torch.device("cuda")
    seqs = sequences([37, 300, 700, 256, 511], [1, 2, 3, 1, 2])
    for count, group in enumerate([seqs[:2], seqs[2:4], seqs[4:]], start=1):
        with torch.inference_mode():
            p = pack(group, backbone, device)
            expected, got = model(p).float(), graphs(p).float()
        assert graphs.replays == count
        cosine = torch.nn.functional.cosine_similarity(got, expected, dim=-1)
        assert cosine.min().item() > 0.999, ([s.length for s in group], cosine)


def test_fast_replays_graphs_and_matches_the_plain_pass():
    import types

    from karpo_decision.serve.qwen_graphs import CudaGraphs, fast

    # A backbone of its own: every Packed rebinds packed_forward on the backbone's layers, and fast() compiles them.
    backbone = build_backbone()
    device = torch.device("cuda")
    seq = sequences([300], [2])
    with torch.inference_mode():
        expected = Packed(backbone, "bidirectional")(pack(seq, backbone, device)).float()
    model = Packed(backbone, "bidirectional")
    decider = types.SimpleNamespace(device=device, model=model, max_length=1024)
    # torch.compile returns a wrapper function, so the compiled layer's packed_forward is no longer a bound method.
    assert inspect.ismethod(model.layers[0].packed_forward)
    assert fast(decider) and isinstance(decider.model, CudaGraphs)
    assert not inspect.ismethod(model.layers[0].packed_forward)
    with torch.inference_mode():
        got = decider.model(pack(seq, backbone, device)).float()
    assert decider.model.replays == 1
    assert torch.nn.functional.cosine_similarity(got, expected, dim=-1).min().item() > 0.999
