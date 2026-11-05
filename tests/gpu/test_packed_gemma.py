"""The Gemma packed path on GPU, with the real kernels.

- cuDNN's global attention (``global_attention``, Blackwell, head size 512): outputs and gradients per virtual
  sequence against the memory-efficient kernel, within the memory-efficient kernel's own bf16 distance from float32;
  lengths off and on the buckets, a branch's queries after a longer shared block, causal and bidirectional.
- Kernel contracts (a tiny random Gemma 4, bf16): the packed forward and backward with the real varlen flash attention
  and memory-efficient attention against the same code with reference kernels in float32 (``tests/test_gemma_blocks``
  checks those against Transformers with rule A masks on CPU). This pins varlen's end alignment of a shorter query run
  and the window, and causal_lower_right in the memory-efficient kernel.
- Real weights (KARPO_GEMMA, a Gemma 4 base checkpoint directory, ``google/gemma-4-31B-it``): requests in the block
  layout (one block; a shared block longer than the sliding window with branches; short branches) packed together
  through ``families.gemma.Packed`` against ``Gemma4Model``'s own forward one sequence at a time with rule A masks.
  Hidden states at the markers agree (relative error and cosine), the readout picks the same option, and the readout is
  the language-model head's rows of the label tokens with the head's logit cap.
"""

import contextlib
import functools
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from karpo_decision import checkpoint
from karpo_decision.encoding import Encoder
from karpo_decision.families.gemma import Packed, global_attention, sliding_attention
from karpo_decision.families.gemma import packed as gp
from karpo_decision.families.gemma.packed import GLOBAL, SLIDING
from karpo_decision.model.packed import pack
from karpo_decision.templates import BlockTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_gemma_blocks as cpu  # noqa: E402

BASE = os.environ.get("KARPO_GEMMA")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
needs_base = pytest.mark.skipif(not BASE, reason="set KARPO_GEMMA to a Gemma 4 base checkpoint directory")


@pytest.mark.parametrize("attention", ["bidirectional", "causal"])
@pytest.mark.parametrize("layout", cpu.PACKS)
def test_real_kernels_keep_the_reference_contract(attention, layout, monkeypatch):
    torch.manual_seed(0)
    lm = build_tiny()
    torch.manual_seed(1)
    seqs = [cpu.sequence(shared, branches) for shared, branches in layout]

    def run(dtype):
        model = lm.to("cuda", dtype)
        model.zero_grad()
        out = Packed(types.SimpleNamespace(language_model=model), attention)(pack(seqs, None, torch.device("cuda")))
        (out.float() * torch.linspace(-1, 1, out.numel(), device="cuda").view_as(out)).sum().backward()
        return out.float(), {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}

    got, got_grads = run(torch.bfloat16)
    with monkeypatch.context() as m:  # the same bf16 weights and activations; only the kernels differ
        m.setattr(sliding_attention, "varlen_attn", cpu.varlen_reference)
        m.setattr(sliding_attention, "_fa4", None)
        m.setattr(gp, "sdpa_kernel", lambda *_: contextlib.nullcontext())
        want, want_grads = run(torch.bfloat16)
    assert ((got - want).norm() / want.norm()) < 1e-2
    for name in ("layers.0.self_attn.q_proj.weight", "layers.2.self_attn.k_proj.weight", "embed_tokens.weight"):
        g, w = got_grads[name], want_grads[name]
        assert ((g - w).norm() / w.norm()) < 3e-2, name


def test_flash_attention_4_sliding_matches_float32():
    """FA4's windowed varlen attention (head size 256, 32 q / 16 kv heads, window 1023, queries and keys with their
    own boundaries, causal aligned at each sequence's end) against float32, no further from it than FA2."""
    q0 = torch.zeros(1, 1, 256, dtype=torch.bfloat16, device="cuda")
    if not sliding_attention.usable(q0):
        pytest.skip("needs flash-attn-4 and a Blackwell GPU")
    q_len = [1500, 120, 300, 1023, 1024, 1025, 40, 2500, 7]
    k_len = [1500, 1143, 1323, 1023, 1024, 1025, 40, 2500, 1030]  # 2nd, 3rd, last: branches after a shared tail
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(sum(n), h, 256, generator=g, device="cuda", dtype=torch.bfloat16)
               for n, h in ((q_len, 32), (k_len, 16), (k_len, 16)))
    cu = [torch.tensor([0, *torch.tensor(n).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
          for n in (q_len, k_len)]

    def run(attend, dtype):
        xs = [t.to(dtype).clone().requires_grad_(True) for t in (q, k, v)]
        out = attend(*xs)
        (out.float() * torch.linspace(-1, 1, out.numel(), device="cuda").view_as(out)).sum().backward()
        return [out.float(), *(x.grad.float() for x in xs)]

    def fa(*xs):
        return sliding_attention.attend(*xs, cu[0], cu[1], max(q_len), max(k_len), 256 ** -0.5, 1023)

    def ref(*xs):
        return cpu.varlen_reference(*xs, cu[0], cu[1], max(q_len), max(k_len), scale=256 ** -0.5,
                                    window_size=(1023, 0), enable_gqa=True)

    def fa2(*xs):
        return sliding_attention.varlen_attn(*xs, cu[0], cu[1], max(q_len), max(k_len), scale=256 ** -0.5,
                                             window_size=(1023, 0), enable_gqa=True)

    want = run(ref, torch.float32)
    for got, base, w in zip(run(fa, torch.bfloat16), run(fa2, torch.bfloat16), want, strict=True):
        err, err_fa2 = ((got - w).norm() / w.norm()).item(), ((base - w).norm() / w.norm()).item()
        assert err < max(1.5 * err_fa2, 5e-3), (err, err_fa2)


def test_selective_recompute_keeps_flash_attention_4_output(monkeypatch):
    """Under the selective recompute policy FA4's forward kernel runs once per step (its output is kept, not
    recomputed in backward), and the gradients match those without recompute."""
    from torch.utils.checkpoint import checkpoint

    from karpo_decision.model.packed import _selective

    q0 = torch.zeros(1, 1, 256, dtype=torch.bfloat16, device="cuda")
    if not sliding_attention.usable(q0):
        pytest.skip("needs flash-attn-4 and a Blackwell GPU")
    calls = []
    forward = sliding_attention._fa4._flash_attn_fwd

    @functools.wraps(forward)  # keeps its compile_cache attribute, which FA4 reads through the module name
    def counted(*args, **kwargs):
        calls.append(1)
        return forward(*args, **kwargs)

    monkeypatch.setattr(sliding_attention._fa4, "_flash_attn_fwd", counted)
    n = [700, 1300]
    cu = torch.tensor([0, n[0], sum(n)], dtype=torch.int32, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(sum(n), 32, 256, generator=g, device="cuda", dtype=torch.bfloat16)

    def layer(x):  # a pointwise op before the kernel (recomputed) and after it
        return (sliding_attention.attend(x * 2, x[:, :16], x[:, 16:], cu, cu, max(n), max(n), 0.0625, 1023) * 3).sum()

    grads = []
    for recompute in (False, True):
        xs = x.clone().requires_grad_(True)
        loss = checkpoint(layer, xs, use_reentrant=False, context_fn=_selective) if recompute else layer(xs)
        loss.backward()
        grads.append(xs.grad)
    assert len(calls) == 2  # once per run: the recomputed run reuses the kept output
    # FA4's backward sums GQA key/value gradients with atomics: equal up to their order, not bit for bit
    assert ((grads[1].float() - grads[0].float()).norm() / grads[0].float().norm()) < 1e-3


@pytest.mark.parametrize("causal", [True, False])
def test_cudnn_global_attention_matches_memory_efficient(causal):
    if not global_attention.usable(torch.zeros(1, 1, 512, dtype=torch.bfloat16, device="cuda")):
        pytest.skip("needs cudnn-frontend and a Blackwell GPU")
    heads, kv_heads, d = 32, 16, 512
    spans = [(700, 1900), (1500, 1500), (333, 2047), (37, 4100), (4097, 4097), (900, 9000), (256, 256)]
    g = torch.Generator(device="cuda").manual_seed(0)
    lens = global_attention.lengths([0, *torch.tensor([q for q, _ in spans]).cumsum(0).tolist()],
                                    [0, *torch.tensor([k for _, k in spans]).cumsum(0).tolist()], torch.device("cuda"))

    def tensors(lq, lk):
        return [torch.randn(n, h, d, device="cuda", dtype=torch.bfloat16, generator=g)
                for n, h in ((lq, heads), (lk, kv_heads), (lk, kv_heads), (lq, heads))]

    def memeff(q, k, v, dtype):
        a, b, c = (t.to(dtype).transpose(0, 1)[None] for t in (q, k, v))
        mask = gp.causal_lower_right(a.shape[2], b.shape[2]) if causal and a.shape[2] != b.shape[2] else None
        with gp.sdpa_kernel(gp.SDPBackend.EFFICIENT_ATTENTION):
            o = gp.F.scaled_dot_product_attention(a, b, c, attn_mask=mask, is_causal=causal and mask is None,
                                                  scale=d ** -0.5, enable_gqa=True)
        return o[0].transpose(0, 1)

    def grads(fn, q, k, v, do):
        q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
        out = fn(q, k, v)
        out.backward(do.to(out.dtype))
        return [out.float(), q.grad.float(), k.grad.float(), v.grad.float()]

    def rel(a, b):
        return ((a - b).abs().max() / b.abs().max()).item()

    for i, (lq, lk) in enumerate(spans):
        q, k, v, do = tensors(lq, lk)
        exact = grads(lambda *t: memeff(*t, torch.float32), q, k, v, do)
        bf16 = grads(lambda *t: memeff(*t, torch.bfloat16), q, k, v, do)
        ql, kl = lens[i, 0, :1], lens[i, 1, :1]
        got = grads(lambda q, k, v, ql=ql, kl=kl: global_attention.attend(q, k, v, ql, kl, causal, d ** -0.5),
                    q, k, v, do)
        for name, x, b, w in zip(("out", "dq", "dk", "dv"), got, bf16, exact, strict=True):
            assert rel(x, w) < 2 * rel(b, w) + 2e-3, (lq, lk, name, rel(x, w), rel(b, w))


def build_tiny():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    config = Gemma4TextConfig(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=len(cpu.TYPES),
                              num_attention_heads=4, num_key_value_heads=2, head_dim=32, global_head_dim=64,
                              num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=cpu.WINDOW,
                              layer_types=cpu.TYPES, hidden_size_per_layer_input=0, final_logit_softcapping=30.0)
    config._attn_implementation = "sdpa"
    lm = Gemma4TextModel(config).float()
    with torch.no_grad():
        for p in lm.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return lm


def requests() -> list[dict]:
    long_state = " ".join(f"Row {i}: item {i * 7 % 13} costs {i * 3} dollars and ships in {i % 5 + 1} days."
                          for i in range(260))
    return [
        {"state": "The page shows a search box and a 'Book now' button.",
         "questions": {"next": {"type": "choice", "instructions": "What should the agent do next?",
                                "criteria": {"type": "Type the destination", "book": "Click Book now",
                                             "ask": "Ask the user"}}}},
        {"state": long_state,
         "questions": {"cheap": {"type": "choice", "instructions": "Which row is cheapest?",
                                 "criteria": {"r0": "Row 0", "r5": "Row 5", "r9": "Row 9"}},
                       "done": {"type": "noul", "instructions": "Is the list complete?"},
                       "ships": {"type": "choice", "instructions": "Which row ships fastest?",
                                 "criteria": {"r1": "Row 1", "r4": "Row 4", "r6": "Row 6"}}}},
        {"state": {"user": "find me a flight to Rome", "history": ["opened site", "typed Rome"]},
         "questions": {"q": {"type": "score", "instructions": "How far along is the task?",
                             "criteria": ["not started", "halfway", "done"]},
                       "next": {"type": "noul", "instructions": "Should the agent search now?"}}},
    ]


def reference(backbone, encoded, bidirectional: bool, window: int) -> torch.Tensor:
    rows = []
    for s in encoded:
        seg, pos = s.segment_ids.long().cuda(), s.positions.cuda()
        masks = {GLOBAL: cpu.rule_a(seg, pos, bidir=bidirectional), SLIDING: cpu.rule_a(seg, pos, window)}
        out = backbone.language_model(input_ids=s.input_ids[None].cuda(), position_ids=pos[None], use_cache=False,
                                      attention_mask=masks)
        rows.append(out.last_hidden_state[0, s.markers.cuda()])
    return torch.cat(rows)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    work = tmp_path_factory.mktemp("gemma-init")
    checkpoint.initialize(BASE, work, template=BlockTemplate.id)
    return checkpoint.load(work, "cuda")


@needs_base
def test_readout_is_the_heads_label_rows_with_its_cap(ckpt):
    assert ckpt.family.MODEL_TYPE == "gemma4" and ckpt.config.template == BlockTemplate.id
    head = ckpt.backbone.language_model.embed_tokens.weight  # tied
    ids = torch.tensor(ckpt.config.token_ids, device=head.device)
    torch.testing.assert_close(ckpt.readout.weight.float(), head[ids].float())
    assert ckpt.readout.cap == ckpt.backbone.config.get_text_config().final_logit_softcapping


@needs_base
@pytest.mark.parametrize("attention", ["causal", "bidirectional"])
@torch.inference_mode()
def test_packed_blocks_match_transformers(ckpt, attention):
    encoder = Encoder(ckpt.processor, ckpt.config.codes, BlockTemplate(), ckpt.family.MARKER)
    encoded = [part for r in requests() for part in encoder.plan(r, None)]
    window = ckpt.backbone.language_model.config.sliding_window
    assert max(s.segment_ids.tolist().count(0) for s in encoded) > window
    got = Packed(ckpt.backbone, attention).eval()(pack(encoded, ckpt.backbone, torch.device("cuda"))).float()
    want = reference(ckpt.backbone, encoded, attention == "bidirectional", window).float()
    # Both are bf16 passes through 60 layers with different kernels: their hidden states differ by bf16 rounding (1-5%
    # relative against a float32 Transformers pass: ours as close to it as Transformers' own bf16 pass,
    # label probabilities within 0.0011 of float32, Transformers' bf16 within 0.0025). What is read must agree.
    assert torch.nn.functional.cosine_similarity(got, want, dim=-1).min() > 0.995
    a, b = ckpt.readout(got.to(torch.bfloat16)), ckpt.readout(want.to(torch.bfloat16))
    counts = [c for s in encoded for c in s.counts]
    for i, c in enumerate(counts):
        pa, pb = a[i, :c].float().softmax(-1), b[i, :c].float().softmax(-1)
        assert pa.argmax() == pb.argmax() and (pa - pb).abs().max() < 1e-2
