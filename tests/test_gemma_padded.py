"""The padded pass of the CUDA-graph path (``PaddedPlan``) against the plain packed forward, on the tiny random Gemma 4
of ``test_gemma_blocks`` on CPU with reference kernels: one single-block sequence padded to a bucket, bidirectional and
causal, lengths inside, one short of and at the bucket; the padding's token ids change nothing."""

import contextlib
import types

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers.models.gemma4")
from test_gemma_blocks import model, sequence, varlen_reference  # noqa: E402, F401  (module-scoped fixture)

from karpo_decision.families.gemma import global_attention, sliding_attention  # noqa: E402
from karpo_decision.families.gemma import packed as gp  # noqa: E402
from karpo_decision.model.packed import Pack, pack  # noqa: E402

SIZE = 24
CPU = torch.device("cpu")


def padded_reference(query, key, value, cu, size, causal, scale):
    """``global_attention.padded``'s contract: each sequence ``cu`` bounds attends within itself, causal from its
    first row."""
    out = torch.zeros_like(query)
    bounds = cu.tolist()
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if b > a:
            q, k, v = (t[0, a:b].transpose(0, 1)[None] for t in (query, key, value))
            out[0, a:b] = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=causal,
                                                         enable_gqa=k.shape[1] != q.shape[1])[0].transpose(0, 1)
    return out


@pytest.fixture
def kernels(monkeypatch):
    monkeypatch.setattr(sliding_attention, "varlen_attn", varlen_reference)
    monkeypatch.setattr(global_attention, "padded", padded_reference)
    monkeypatch.setattr(gp, "sdpa_kernel", lambda *_: contextlib.nullcontext())


def padded_forward(packed, s, size, pad_ids):
    """What the graph path computes for one single-block sequence padded to ``size`` rows, run eagerly."""
    n = s.length
    cu = torch.tensor([0, n, size], dtype=torch.int32)
    padded = Pack(input_ids=torch.cat([s.input_ids, pad_ids[:size - n]]), seqpos=torch.arange(size), cu=cu,
                  cu_cpu=None, max_length=size, markers=torch.zeros(0, dtype=torch.long), marker_cu=cu[:1],
                  max_markers=0, plan=gp.PaddedPlan(cu, size))
    h = packed.embed(padded.input_ids)
    h = packed.body(h, packed.rope(h, padded.seqpos), padded)[:, :n]
    real = pack([s], None, CPU)
    real.plan = gp.block_plan(real, packed.window)
    last = packed.types[-1]
    return packed.head(h, {last: packed.lm.rotary_emb(h, real.seqpos[None], last)}, real)


@pytest.mark.parametrize("attention", ["bidirectional", "causal"])
@pytest.mark.parametrize("length", [5, 9, SIZE - 1, SIZE])
def test_padded_pass_matches_the_plain_forward(model, kernels, attention, length):  # noqa: F811
    torch.manual_seed(3)
    s = sequence(length, [])
    packed = gp.Packed(types.SimpleNamespace(language_model=model), attention)
    with torch.no_grad():
        plain = packed(pack([s], None, CPU))
        ours = padded_forward(packed, s, SIZE, torch.randint(3, 97, (SIZE,)))
        other = padded_forward(packed, s, SIZE, torch.randint(3, 97, (SIZE,)))
    torch.testing.assert_close(ours, plain, atol=1e-10, rtol=1e-9)
    assert torch.equal(other, ours)
