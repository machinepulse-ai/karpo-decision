"""The packed path against the reference path on a small random Qwen3.5 backbone (CUDA, flash-linear-attention).

KARPO_TOKENIZER points at a Qwen3.5 checkpoint directory whose config is shrunk here (width, heads, layers) while
keeping what the kernels and M-RoPE depend on (head size, rotary sections, the 3-linear-1-full layer pattern).
"""

import importlib.util
import os

import pytest
import torch

from karpo_decision.encoding import Encoded
from karpo_decision.families.qwen import Packed
from karpo_decision.families.qwen.reference import reference_hidden
from karpo_decision.model.packed import pack
from karpo_decision.model.readout import Readout

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
PAD = 0


@pytest.fixture(scope="module")
def backbone():
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
    model = Qwen3_5Model(config).to("cuda", torch.bfloat16).eval()
    return model


def sequences(lengths, questions):
    generator = torch.Generator().manual_seed(1)
    out = []
    for n, k in zip(lengths, questions, strict=True):
        ids = torch.randint(*TOKENS, (n,), generator=generator)
        markers = torch.sort(torch.randperm(n - 1, generator=generator)[: k - 1] + 1).values
        markers = torch.cat([markers, torch.tensor([n - 1])])
        out.append(Encoded(keys=tuple(f"q{i}" for i in range(k)), counts=(4,) * k, input_ids=ids, markers=markers))
    return out


@pytest.mark.parametrize("attention", ["bidirectional", "causal"])
def test_packed_matches_reference(backbone, attention):
    seqs = sequences([37, 211, 64, 130], [1, 3, 2, 5])
    with torch.inference_mode():
        expected = reference_hidden(backbone, seqs, attention, PAD).float()
        got = Packed(backbone, attention)(pack(seqs, backbone, torch.device("cuda"))).float()
    assert got.shape == expected.shape
    cosine = torch.nn.functional.cosine_similarity(got, expected, dim=-1)
    assert cosine.min().item() > 0.999, cosine
    assert ((got - expected).abs().max() / expected.abs().max()).item() < 0.03


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
def test_packed_matches_reference_with_images(backbone, attention):
    seqs = image_sequences()
    assert seqs[0].pixel_values is not None and seqs[2].image_grid_thw.shape[0] == 2
    with torch.inference_mode():
        expected = reference_hidden(backbone, seqs, attention, PAD).float()
        got = Packed(backbone, attention)(pack(seqs, backbone, torch.device("cuda"))).float()
    cosine = torch.nn.functional.cosine_similarity(got, expected, dim=-1)
    assert cosine.min().item() > 0.999, cosine


def test_sequences_do_not_see_each_other(backbone):
    """A sequence's readouts are the same alone and packed with others."""
    seqs = sequences([50, 90, 70], [2, 2, 2])
    model = Packed(backbone, "bidirectional")
    device = torch.device("cuda")
    with torch.inference_mode():
        together = model(pack(seqs, backbone, device)).float()
        alone = model(pack(seqs[1:2], backbone, device)).float()
    assert torch.allclose(together[2:4], alone, atol=2e-2, rtol=2e-2)


def test_gradients_reach_every_kind_of_layer(backbone):
    model = Packed(backbone, "bidirectional", checkpoint_every=1)
    # Nonzero rows: a zero head passes no gradient back to the backbone.
    head = Readout.from_rows(torch.randn(255, backbone.config.text_config.hidden_size) * 0.02).cuda()
    backbone.train()
    try:
        backbone.zero_grad(set_to_none=True)
        logits = head(model(pack(sequences([40, 60], [2, 1]), backbone, torch.device("cuda"))))
        logits[:, :4].log_softmax(-1)[:, 0].sum().backward()
        layers = backbone.language_model.layers
        for param in (layers[0].linear_attn.in_proj_qkv.weight, layers[0].linear_attn.A_log,
                      layers[3].self_attn.q_proj.weight, layers[-1].mlp.down_proj.weight):
            assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    finally:
        backbone.eval()


def test_packed_convolution_kernel_matches_the_reference_convolution():
    """flash-linear-attention's causal convolution with the sequence boundaries (the packed path) against
    ``packed_conv`` (taps cut by position), with the activation, forward and backward."""
    from fla.modules.conv.causal_conv1d import causal_conv1d

    from karpo_decision.families.qwen.packed import packed_conv

    torch.manual_seed(0)
    lengths = [1, 3, 700, 4, 2048, 333]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    seqpos = torch.cat([torch.arange(n) for n in lengths]).cuda()
    x = torch.randn(1, int(cu[-1]), 640, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = (0.3 * torch.randn(640, 4, device="cuda")).to(torch.bfloat16).requires_grad_()
    grad = torch.randn_like(x)

    def run(fn):
        x.grad = weight.grad = None
        out = fn()
        out.backward(grad)
        return out.float(), x.grad.float(), weight.grad.float()

    got = run(lambda: causal_conv1d(x, weight, activation="silu", cu_seqlens=cu.cuda(), cu_seqlens_cpu=cu)[0])
    want = run(lambda: torch.nn.functional.silu(packed_conv(x.transpose(1, 2), weight, seqpos)).transpose(1, 2))
    for g, w in zip(got, want, strict=True):
        assert ((g - w).abs().max() / w.abs().max()).item() < 2e-2
