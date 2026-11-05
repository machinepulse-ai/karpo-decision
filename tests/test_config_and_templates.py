import pytest

from karpo_decision.codes import MAX_CODES, candidates
from karpo_decision.config import DecisionConfig
from karpo_decision.templates import BlockTemplate, template

CODES = tuple(candidates()[:MAX_CODES])


def config(**kwargs):
    base = dict(template=BlockTemplate.id, codes=CODES, token_ids=tuple(range(MAX_CODES)), attention="bidirectional",
                base_model="google/gemma-4-31B-it")
    return DecisionConfig(**{**base, **kwargs})


def test_config_round_trips(tmp_path):
    saved = config(temperatures={"choice": 1.3, "noul": 1.1, "score": 0.9})
    saved.save(tmp_path)
    assert DecisionConfig.load(tmp_path) == saved


@pytest.mark.parametrize("kwargs", [{"attention": "sideways"}, {"token_ids": (1, 2)},
                                    {"temperatures": {"choice": 1.0}}, {"format_version": 9}])
def test_config_rejects(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_codes_start_with_single_letters():
    assert candidates()[:3] == ["A", "B", "C"] and candidates()[26:28] == ["AA", "AB"]


def test_unknown_template():
    with pytest.raises(ValueError, match="unknown template"):
        template("karpo-blocks-v0")
