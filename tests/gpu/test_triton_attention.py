"""The Triton global attention (``families.gemma.triton_attention``) on an sm_120 GPU, at the 31B's head size and the
tiny model's: one sequence against float32 math, with logits of order one and in the hundreds; a varlen pack with
branches (queries and keys with their own boundaries) and an empty sequence against PyTorch's memory-efficient
attention per virtual sequence; keys beginning past 2^31 elements into the pack (31B only); equal outputs from
repeated calls; no gradients. Through ``Packed.global_attend`` (``triton_attention.enable``), on a tiny random Gemma 4,
with a spy on the hook: the packed forward stays near the plain path and calls the hook with each pack's longest
virtual sequence (past one 64-row tile too); CUDA graph replays equal the eager padded pass, stay near the plain path,
and reach the hook through the padded plan; a causal model is refused."""

import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from karpo_decision.families.gemma import packed as gp
from karpo_decision.families.gemma import triton_attention as ta
from karpo_decision.model.packed import pack
from karpo_decision.serve import graphs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_gemma_blocks as cpu  # noqa: E402
from test_packed_gemma import build_tiny  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not ta.supported(torch.device("cuda", 0))
                                                  if torch.cuda.is_available() else True, reason="needs an sm_120 GPU")]
GPU = torch.device("cuda")
SHAPES = pytest.mark.parametrize(("heads", "kv_heads", "head"), [(32, 4, 512), (4, 1, 64)], ids=["31b", "tiny"])


def tensors(rows, keys, heads, kv_heads, head):
    q = torch.randn(rows, heads, head, device=GPU, dtype=torch.bfloat16) / 8
    k = torch.randn(keys, kv_heads, head, device=GPU, dtype=torch.bfloat16) / 8
    v = torch.randn(keys, kv_heads, head, device=GPU, dtype=torch.bfloat16)
    return q, k, v


def bounds(cu):
    return torch.tensor(cu, dtype=torch.int32, device=GPU)


def memory_efficient(q, k, v, q_cu, k_cu):
    outs = []
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        for qa, qb, ka, kb in zip(q_cu[:-1], q_cu[1:], k_cu[:-1], k_cu[1:], strict=True):
            if qb > qa:
                outs.append(F.scaled_dot_product_attention(
                    q[None, qa:qb].transpose(1, 2), k[None, ka:kb].transpose(1, 2), v[None, ka:kb].transpose(1, 2),
                    scale=1.0, enable_gqa=True).transpose(1, 2)[0])
    return torch.cat(outs)


@SHAPES
def test_one_sequence_matches_float32_math(heads, kv_heads, head):
    torch.manual_seed(0)
    q, k, v = tensors(300, 300, heads, kv_heads, head)
    with torch.inference_mode():
        got = ta.attend(q, k, v, bounds([0, 300]), bounds([0, 300]), 300, 1.0).float()
    rep = heads // kv_heads
    qf, kf, vf = (t.float().transpose(0, 1) for t in (q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)))
    want = (torch.softmax(qf @ kf.transpose(1, 2), -1) @ vf).transpose(0, 1)
    assert ((got - want).norm() / want.norm()) < 1e-2


@SHAPES
def test_branched_varlen_matches_the_memory_efficient_path(heads, kv_heads, head):
    """A shared block of 1,000 rows and branches of 37 and 120 rows (each over the shared keys, then its own), and an
    empty virtual sequence between them."""
    torch.manual_seed(1)
    shared, b1, b2 = 1000, 37, 120
    q, kk, vv = tensors(shared + b1 + b2, shared + b1 + b2, heads, kv_heads, head)
    rows = torch.cat([torch.arange(shared), torch.arange(shared), torch.arange(shared, shared + b1),
                      torch.arange(shared), torch.arange(shared + b1, shared + b1 + b2)]).to(GPU)
    k, v = kk.index_select(0, rows), vv.index_select(0, rows)
    q_cu = [0, shared, shared + b1, shared + b1, shared + b1 + b2]
    k_cu = [0, shared, 2 * shared + b1, 2 * shared + b1, 3 * shared + b1 + b2]
    with torch.inference_mode():
        got = ta.attend(q, k, v, bounds(q_cu), bounds(k_cu), shared, 1.0).float()
        want = memory_efficient(q, k, v, q_cu, k_cu).float()
    assert ((got - want).norm() / want.norm()) < 1e-2


def test_refuses_gradients():
    q, k, v = tensors(8, 8, 4, 1, 64)
    with pytest.raises(RuntimeError, match="forward"):
        ta.attend(q, k, v, bounds([0, 8]), bounds([0, 8]), 8, 1.0)


def test_keys_past_32_bit_offsets_match_the_memory_efficient_path():
    """The 31B's key rows are 2,048 elements, so row 2^20 starts at element 2^31. A first sequence holds 2^20 keys and
    no queries, a second 64 queries over the 300 keys after them (only those are filled in)."""
    torch.manual_seed(2)
    start, rows, keys = 1 << 20, 64, 300
    q, k_tail, v_tail = tensors(rows, keys, 32, 4, 512)
    k, v = (torch.empty(start + keys, 4, 512, device=GPU, dtype=torch.bfloat16) for _ in range(2))
    k[start:], v[start:] = k_tail, v_tail
    with torch.inference_mode():
        got = ta.attend(q, k, v, bounds([0, 0, rows]), bounds([0, start, start + keys]), rows, 1.0).float()
        want = memory_efficient(q, k[start:], v[start:], [0, rows], [0, keys]).float()
    assert ((got - want).norm() / want.norm()) < 1e-2


@SHAPES
def test_repeated_calls_are_bitwise_equal(heads, kv_heads, head):
    torch.manual_seed(3)
    q, k, v = tensors(500, 700, heads, kv_heads, head)
    q_cu, k_cu = bounds([0, 200, 500]), bounds([0, 300, 700])
    with torch.inference_mode():
        assert torch.equal(ta.attend(q, k, v, q_cu, k_cu, 300, 1.0), ta.attend(q, k, v, q_cu, k_cu, 300, 1.0))


@SHAPES
def test_large_logits_match_float32_math(heads, kv_heads, head):
    """Queries and keys without the usual 1/8, at scale 8: logits reach the hundreds, past float32's exp unless the
    kernel subtracts a running maximum."""
    torch.manual_seed(4)
    q, k, v = tensors(300, 300, heads, kv_heads, head)
    q, k = q * 8, k * 8
    with torch.inference_mode():
        got = ta.attend(q, k, v, bounds([0, 300]), bounds([0, 300]), 300, 8.0).float()
    rep = heads // kv_heads
    qf, kf, vf = (t.float().transpose(0, 1) for t in (q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)))
    want = (torch.softmax(8.0 * (qf @ kf.transpose(1, 2)), -1) @ vf).transpose(0, 1)
    assert ((got - want).norm() / want.norm()) < 1e-2


def tiny(attention="bidirectional"):
    torch.manual_seed(0)
    lm = build_tiny().to(GPU, torch.bfloat16)
    return gp.Packed(types.SimpleNamespace(language_model=lm), attention)


def test_packed_forward_with_the_hook_stays_near_the_plain_path():
    """A spy on the hook: it is called by the body's global layers with the longest virtual sequence as ``max_q`` and by
    the markers-only layer with 1, so a disconnected hook or an undersized grid shows (the layouts include rows past
    one 64-row tile)."""
    packed = tiny()
    torch.manual_seed(2)
    layouts = [[cpu.sequence(13, [])], [cpu.sequence(11, [5, 7])], [cpu.sequence(9, []), cpu.sequence(40, [6])],
               [cpu.sequence(150, [])], [cpu.sequence(70, [30, 90])]]
    longest = [13, 11, 40, 150, 90]
    with torch.inference_mode():
        want = [packed(pack(seqs, None, GPU)).float() for seqs in layouts]
        ta.enable(packed)
        assert packed.global_attend is ta.attend
        packed.global_attend = spy = mock.Mock(wraps=ta.attend)
        got, max_q = [], []
        for seqs in layouts:
            spy.reset_mock()
            got.append(packed(pack(seqs, None, GPU)).float())
            max_q.append([call.args[5] for call in spy.call_args_list])
    for g, w in zip(got, want, strict=True):
        assert ((g - w).norm() / w.norm()) < 2e-2
    for calls, n in zip(max_q, longest, strict=True):
        assert len(calls) >= 2, "the hook was not called"
        assert set(calls[:-1]) == {n} and calls[-1] == 1, calls


def test_graph_replays_with_the_hook_equal_the_eager_padded_pass():
    """Replays equal the eager padded pass and stay near the plain forward; a spy shows the padded plan reaching the
    hook during capture, with the graphs' boundary buffer for both bounds and the bucket's size as ``max_q``."""
    packed = tiny()
    torch.manual_seed(3)
    batches = [[cpu.sequence(length, [])] for length in (13, 64, 65, 150, 13)]
    with torch.inference_mode():
        want = [packed(pack(seqs, None, GPU)).float() for seqs in batches]
    ta.enable(packed)
    packed.global_attend = spy = mock.Mock(wraps=ta.attend)
    eager = graphs.Graphs(packed, max_tokens=256)
    g = graphs.Graphs(packed, max_tokens=256)
    g.capture_all()
    calls = spy.call_args_list  # the warmup and captured passes of every bucket
    assert calls, "the padded plan did not call the hook"
    assert all(call.args[3] is g.cu and call.args[4] is g.cu for call in calls)
    assert {call.args[5] for call in calls} == set(range(graphs.STEP, 257, graphs.STEP))
    with torch.inference_mode():
        for seqs, w in zip(batches, want, strict=True):
            got = g(pack(seqs, None, GPU))
            assert torch.equal(got, eager(pack(seqs, None, GPU)))
            assert ((got.float() - w).norm() / w.norm()) < 2e-2
    assert g.replays == 5


def test_enable_refuses_a_causal_model():
    with pytest.raises(ValueError, match="bidirectional"):
        ta.enable(tiny("causal"))
