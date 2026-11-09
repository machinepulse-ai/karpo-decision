"""The Triton global attention (``families.gemma.triton_attention``) on an sm_120 GPU, at the 31B's head size and the
tiny model's: one sequence against float32 math; a varlen pack with branches (queries and keys with their own
boundaries) and an empty sequence against PyTorch's memory-efficient attention per virtual sequence; no gradients."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from karpo_decision.families.gemma import triton_attention as ta

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
