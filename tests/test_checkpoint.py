import torch
from safetensors.torch import save_file

from karpo_decision.checkpoint import LM_HEAD, lm_head_rows


def test_lm_head_rows_from_a_single_file(tmp_path):
    head = torch.arange(12.0).reshape(6, 2)
    save_file({LM_HEAD: head}, str(tmp_path / "model.safetensors"))
    assert lm_head_rows(tmp_path, [4, 1]).tolist() == [[8.0, 9.0], [2.0, 3.0]]


def test_lm_head_rows_fall_back_to_tied_embeddings(tmp_path):
    embeddings = torch.arange(8.0).reshape(4, 2)
    save_file({"model.language_model.embed_tokens.weight": embeddings}, str(tmp_path / "model.safetensors"))
    assert lm_head_rows(tmp_path, [3]).tolist() == [[6.0, 7.0]]
