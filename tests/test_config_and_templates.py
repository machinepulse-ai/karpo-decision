import pytest

from karpo_decision.codes import MAX_CODES, candidates
from karpo_decision.config import DecisionConfig
from karpo_decision.templates import BlockTemplate, JointTemplate, PreviewTemplate, template

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


def test_joint_template_numbers_questions_and_puts_a_marker_after_each():
    questions = [{"type": "choice", "instructions": "Next?", "criteria": {"click": "Click it", "stop": None}},
                 {"type": "noul", "instructions": "Done?"}]
    messages = template(JointTemplate.id).messages("the page", questions, CODES, "<M>", images=1, escape=str)
    system, user = messages
    assert system["role"] == "system" and user["content"][0] == {"type": "image"}
    text = user["content"][-1]["text"]
    assert text.startswith("State:\nthe page")
    assert text.count("<M>") == 2
    assert "Question 1:\nNext?\nOptions:\nA: click: Click it\nB: stop\n<M>" in text
    assert "Question 2:\nDone?\nOptions:\nA: No / false\nB: Yes / true\n<M>" in text


def test_preview_template_lists_the_instructions_before_the_state():
    questions = [{"type": "choice", "instructions": "Next?", "criteria": {"click": "Click it", "stop": None}},
                 {"type": "noul"}]
    joint = template(JointTemplate.id).messages("the page", questions, CODES, "<M>", images=0, escape=str)
    system, user = template(PreviewTemplate.id).messages("the page", questions, CODES, "<M>", images=0, escape=str)
    text = user["content"][-1]["text"]
    assert text == "Questions:\n1. Next?\n2. Choose the best matching option.\n\n" + joint[1]["content"][-1]["text"]
    assert text.count("<M>") == 2 and "questions are listed first" in system["content"]


def test_template_kinds():
    assert [template(i).kind for i in (JointTemplate.id, PreviewTemplate.id, BlockTemplate.id)] == [
        "joint", "joint", "blocks"]
