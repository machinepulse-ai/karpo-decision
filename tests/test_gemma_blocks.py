"""The Gemma family's packed forward over block layouts against an unmodified Transformers forward with explicit masks
(Sieve-31B-v3's rule A, as its SegMask.dense builds them), on a tiny random Gemma 4 text model on CPU: hidden states
at the markers and gradients, bidirectional and causal, window boundaries (shared blocks one short of, at and one past
the window), one-block sequences beside branched ones. The CUDA kernels are replaced by reference implementations of
their contracts (varlen flash attention aligns a shorter query run with the end of its keys); the GPU tests compare
the kernels themselves."""

import contextlib
import types

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers.models.gemma4")
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig  # noqa: E402
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel  # noqa: E402

from karpo_decision.encoding import Encoded  # noqa: E402
from karpo_decision.families.gemma import packed as gp  # noqa: E402
from karpo_decision.families.gemma import sliding_attention  # noqa: E402
from karpo_decision.model.packed import pack  # noqa: E402

WINDOW = 8
TYPES = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention"]


def varlen_reference(q, k, v, cu_q, cu_k, max_q, max_k, scale=None, window_size=(-1, -1), enable_gqa=False):
    """varlen flash attention's contract: per sequence, query i sees key j when i + (Lk - Lq) - left <= j <= i + (Lk -
    Lq) + right (right 0: causal, aligned at the end)."""
    left, right = window_size
    outs = []
    for qa, qb, ka, kb in zip(cu_q[:-1].tolist(), cu_q[1:].tolist(), cu_k[:-1].tolist(), cu_k[1:].tolist(),
                              strict=True):
        i = torch.arange(qb - qa, device=q.device)[:, None] + (kb - ka) - (qb - qa)
        j = torch.arange(kb - ka, device=q.device)[None, :]
        mask = (j <= i + right) & (j >= i - left)
        outs.append(F.scaled_dot_product_attention(q[qa:qb].transpose(0, 1)[None], k[ka:kb].transpose(0, 1)[None],
                                                   v[ka:kb].transpose(0, 1)[None], attn_mask=mask, scale=scale,
                                                   enable_gqa=enable_gqa)[0].transpose(0, 1))
    return torch.cat(outs)


@pytest.fixture
def cpu_kernels(monkeypatch):
    monkeypatch.setattr(sliding_attention, "varlen_attn", varlen_reference)
    monkeypatch.setattr(gp, "sdpa_kernel", lambda *_: contextlib.nullcontext())


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = Gemma4TextConfig(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=len(TYPES),
                              num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=16,
                              num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=WINDOW,
                              layer_types=TYPES, hidden_size_per_layer_input=0, final_logit_softcapping=30.0)
    config._attn_implementation = "sdpa"
    lm = Gemma4TextModel(config).double()  # float64: the two paths agree to rounding, so the tolerances can be tight
    with torch.no_grad():
        for p in lm.parameters():
            p.add_(0.05 * torch.randn_like(p))  # norms and scalars away from 1 so a wrong rule shows
        for layer in lm.layers:
            layer.layer_scalar.fill_(0.9)
    return lm


def sequence(shared, branches, vocab=97):
    """A block layout: one block of ``shared`` tokens (no branches), or a shared block and branches."""
    ids = torch.randint(3, vocab, (shared + sum(branches),))
    if not branches:
        return Encoded(keys=("0",), counts=(2,), input_ids=ids, markers=torch.tensor([shared - 1]),
                       positions=torch.arange(shared), segment_ids=torch.zeros(shared, dtype=torch.int16))
    pos, seg, markers, at = list(range(shared)), [0] * shared, [], shared
    for k, n in enumerate(branches, start=1):
        pos += range(shared, shared + n)
        seg += [k] * n
        at += n
        markers.append(at - 1)
    return Encoded(keys=tuple(str(i) for i in range(len(branches))), counts=(2,) * len(branches), input_ids=ids,
                   markers=torch.tensor(markers), positions=torch.tensor(pos),
                   segment_ids=torch.tensor(seg, dtype=torch.int16))


def rule_a(seg, pos, window=None, bidir=False):
    """Sieve's SegMask.dense for one sequence: [1, 1, L, L], True where a query may see a key."""
    qs, qp, ks, kp = seg[:, None], pos[:, None], seg[None, :], pos[None, :]
    ok = (ks == 0) | (ks == qs)
    if not bidir:
        ok &= kp <= qp
        if window:
            ok &= (qp - kp) < window
    return ok[None, None]


def reference(lm, seqs, bidir):
    """Transformers' own forward, one sequence at a time, with rule A's masks; final hidden states at the markers."""
    outs = []
    for s in seqs:
        seg, pos = s.segment_ids.long(), s.positions
        masks = {"full_attention": rule_a(seg, pos, bidir=bidir), "sliding_attention": rule_a(seg, pos, WINDOW)}
        h = lm(input_ids=s.input_ids[None], position_ids=pos[None], attention_mask=masks, use_cache=False)
        outs.append(h.last_hidden_state[0, s.markers])
    return torch.cat(outs)


PACKS = [
    [(13, []), (11, [5, 7, 3])],
    [(WINDOW - 2, [4, 9]), (WINDOW - 1, [3]), (WINDOW, [6, 2]), (WINDOW + 1, [10, 1, 4])],  # shared block vs window
    [(3, [12, 15])],  # branches longer than the window
]


@pytest.mark.parametrize("attention", ["bidirectional", "causal"])
@pytest.mark.parametrize("layout", PACKS)
def test_packed_blocks_match_transformers_with_rule_a(model, cpu_kernels, attention, layout):
    torch.manual_seed(1)
    seqs = [sequence(shared, branches) for shared, branches in layout]
    packed = gp.Packed(types.SimpleNamespace(language_model=model), attention)
    p = pack(seqs, None, torch.device("cpu"))
    model.zero_grad()
    ours = packed(p)
    (ours * torch.linspace(-1, 1, ours.numel(), dtype=ours.dtype).view_as(ours)).sum().backward()
    ours_grads = {n: q.grad.clone() for n, q in model.named_parameters() if q.grad is not None}
    model.zero_grad()
    theirs = reference(model, seqs, attention == "bidirectional")
    (theirs * torch.linspace(-1, 1, theirs.numel(), dtype=theirs.dtype).view_as(theirs)).sum().backward()
    torch.testing.assert_close(ours, theirs, atol=1e-10, rtol=1e-9)
    params = dict(model.named_parameters())
    for name in ("layers.0.self_attn.q_proj.weight", "layers.2.self_attn.k_proj.weight",
                 "layers.3.mlp.down_proj.weight", "embed_tokens.weight"):
        param = params[name]
        # Gemma's norms compute in float32 inside a float64 model; a shared row's gradient sums its branches in another
        # order than the masked forward does, so the paths agree to float32 rounding of the largest gradient.
        torch.testing.assert_close(ours_grads[name], param.grad, atol=1e-6 * float(param.grad.abs().max()), rtol=1e-5)


def test_a_branch_never_sees_another(model, cpu_kernels):
    """Changing one branch's tokens leaves the shared block's and the other branches' readouts as they were."""
    torch.manual_seed(2)
    packed = gp.Packed(types.SimpleNamespace(language_model=model), "bidirectional")
    a = sequence(10, [4, 6, 5])
    b = Encoded(**{**a.__dict__, "input_ids": a.input_ids.clone()})
    b.input_ids[10 + 4: 10 + 4 + 6] = torch.randint(3, 97, (6,))
    with torch.no_grad():
        ra, rb = packed(pack([a], None, torch.device("cpu"))), packed(pack([b], None, torch.device("cpu")))
    torch.testing.assert_close(ra[[0, 2]], rb[[0, 2]])
    assert not torch.allclose(ra[1], rb[1])


def test_the_plan_refuses_scrambled_segments(model):
    s = sequence(6, [3, 3])
    bad = Encoded(**{**s.__dict__, "segment_ids": torch.tensor([0] * 6 + [2] * 3 + [1] * 3, dtype=torch.int16)})
    with pytest.raises(ValueError):
        gp.block_plan(pack([bad], None, torch.device("cpu")), WINDOW)
