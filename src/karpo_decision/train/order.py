"""Which choice questions may be read with their options in another order (``movable``), as ``views`` reads them.

What never moves: score and noul questions; a single option; options that point at others ("all of the above",
"both A and C", "both a and b", "option 2", "(B)"); ordered options (numbers, dates, ranges, stars, scales); a question
whose instructions name options by place or write its options out ("A: …" lines)."""

from __future__ import annotations

import re
from typing import Any

# Options that point at other options. A letter counts as a capital standing alone ("for a", "Ford", "has part(s)",
# "only a few" point nowhere), or in lower case where nothing else can be meant: "both a and b", "(b) and (c)", an
# option that is only letters ("a & b").
POINTING = re.compile(r"(?i:\b(all|none|both|neither|either)\s+(of\s+)?(the\s+)?(above|below|following|these|"
                      r"options?|choices?)\b|\b(of|the)\s+above\b|\b(options?|choices?)\s+[a-j0-9]\b"
                      r"|\bthe\s+(former|latter)\b(?!\s+[a-z])"
                      r"|\b(both|either|neither)\s+\(?[a-j]\)?\s+(and|or|nor)\s+\(?[a-j]\)?(?![a-z])"
                      r"|\A\s*\(?[a-j]\)?(\s*(,|&|and|or)\s*\(?[a-j]\)?)+[\s.]*\Z)"
                      r"|\(\s*[a-j]\s*\)\s*(?:,|&|and|or)\s*\(\s*[a-j]\s*\)"
                      r"|\b[A-J]\s*[,&]\s*[A-J]\b|\b[A-J]\s+(?i:and|or)\s+[A-J]\b|\(\s*[A-J]\s*\)"
                      r"|\b[A-J]\s+(?i:only)\b|(?i:\bonly)\s+[A-J]\b")
# Instructions that name options by their place ("(R)-1-phenylethanol" does not).
BY_PLACE = re.compile(r"\b(first|second|third|fourth|fifth|last)\s+(option|choice)\b|\boption\s*#?\s*[A-Z0-9]\b"
                      r"|\(\s*[A-J]\s*\)")
INTEGER = re.compile(r"^\s*[-+]?\d+\s*$")
SCALE = re.compile(r"\b(very|slightly|somewhat|moderately|extremely|strongly|not at all|low|medium|high|lowest|"
                   r"highest|never|rarely|sometimes|often|always|poor|fair|good|excellent|worst|best|least|most|minor|"
                   r"major|mild|moderate|severe|stars?|points?|level|degree|disagree|agree|neutral|increases?|"
                   r"decreases?)\b", re.I)


def text_of(key: str, text: Any) -> str:
    return str(text) if text not in (None, "") else key


def ordered(texts: list[str]) -> bool:
    """Options whose order means something: a run of consecutive integers (a rating scale; computed answers such as
    $345 or 23 move freely), or short options that are (almost) all scale words (stars, low / medium / high)."""
    if len(texts) < 3:
        return False
    if all(INTEGER.match(t) for t in texts):
        values = [int(t) for t in texts]
        steps = [b - a for a, b in zip(values, values[1:], strict=False)]
        return all(step == 1 for step in steps) or all(step == -1 for step in steps)
    return max(len(t) for t in texts) <= 40 and sum(bool(SCALE.search(t)) for t in texts) >= len(texts) - 1


def lists_own_options(instructions: str, criteria: dict[str, Any]) -> bool:
    """The stem writes two or more of the question's options as "K: option" lines (an option known by its first six
    words), which a reorder of the criteria would contradict."""
    def head(text: Any) -> str:
        return r"\s+".join(re.escape(word) for word in str(text).split()[:6])
    return sum(bool(re.search(rf"(?m)^\s*\(?{re.escape(k)}\s*[:.)]\s*{head(v)}", instructions))
               for k, v in criteria.items() if v not in (None, "")) >= 2


def movable(question: dict[str, Any]) -> bool:
    if question.get("type") != "choice" or not isinstance(question.get("criteria"), dict):
        return False
    criteria = question["criteria"]
    instructions = str(question.get("instructions") or "")
    if len(criteria) < 2 or BY_PLACE.search(instructions) or lists_own_options(instructions, criteria):
        return False
    texts = [text_of(k, v) for k, v in criteria.items()]
    return not any(POINTING.search(t) for t in texts) and not ordered(texts)
