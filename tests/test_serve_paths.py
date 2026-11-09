"""How ``serve.server`` loads a checkpoint (a Gemma 4 checkpoint on one GPU, plain and text only) and reads its
``--max-length``. No GPU and no weights: the loader is replaced."""

import json

import pytest

from karpo_decision.config import DecisionConfig
from karpo_decision.decider import DEFAULT_MAX_LENGTH
from karpo_decision.protocol import images_of
from karpo_decision.serve import server


def checkpoint(directory, model_type):
    DecisionConfig(template="karpo-blocks-v1", codes=("A",), token_ids=(1,), attention="bidirectional",
                   base_model="base").save(directory)
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(directory)


def test_gemma_serves_on_one_gpu_through_the_plain_path(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    calls = []
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", **_: calls.append((directory, device))
                                    or "plain"))
    decider, warmup = server.load_decider(path, 1, "cuda:0")
    assert decider == "plain" and calls == [(path, "cuda:0")]
    assert images_of(warmup)[1] == []


def test_gemma_refuses_more_than_one_gpu(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    monkeypatch.setattr(server.Decider, "from_pretrained", classmethod(lambda *_, **__: pytest.fail("loaded")))
    with pytest.raises(ValueError, match="one GPU"):
        server.load_decider(path, 2, "cuda:0")


def test_gemma_reads_sequences_as_long_as_the_server_is_told(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "gemma4")
    lengths = []
    monkeypatch.setattr(server.Decider, "from_pretrained",
                        classmethod(lambda cls, directory, device="cuda", max_length=None, **_:
                                    lengths.append(max_length) or "plain"))
    server.load_decider(path, 1, "cuda:0")
    server.load_decider(path, 1, "cuda:0", 65_536)
    assert lengths == [DEFAULT_MAX_LENGTH, 65_536]


def test_the_server_takes_its_longest_sequence_from_max_length():
    assert server.arguments(["--checkpoint", "c"]).max_length == DEFAULT_MAX_LENGTH
    assert server.arguments(["--checkpoint", "c", "--max-length", "65536"]).max_length == 65_536


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_the_server_refuses_a_max_length_that_is_not_a_positive_integer(value, capsys):
    with pytest.raises(SystemExit):
        server.arguments(["--checkpoint", "c", "--max-length", value])
    err = capsys.readouterr().err
    assert "argument --max-length" in err and "is not a positive integer" in err
