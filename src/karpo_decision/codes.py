"""The answer codes: the labels options get in the prompt (``A``, ``B``, ..., ``AA``, ...), one readout row each.

A code qualifies when it is a single token of its own, no earlier code took that token, and it stays that single
token right after the assistant turn opens (so the code's token embedding and its language-model-head row are the
ones the base model knows). The first ``MAX_CODES`` that qualify, in candidate order, are the model's codes; the
readout head starts from the language-model-head rows of their tokens.
"""

from __future__ import annotations

import string
from typing import Any

from .protocol import MAX_OPTIONS

MAX_CODES = MAX_OPTIONS
#: The user turn used to see how a code tokenizes right after the assistant turn opens.
_PROBE = "Choose an option."


def candidates() -> list[str]:
    """Every code considered, in preference order: one letter, then two letters."""
    letters = string.ascii_uppercase
    return [*letters, *(first + second for first in letters for second in letters)]


def answer_codes(tokenizer: Any) -> tuple[list[str], list[int]]:
    """The model's ``MAX_CODES`` answer codes and their token ids for ``tokenizer``."""
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": _PROBE}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    codes, ids, taken = [], [], set()
    for code in candidates():
        alone = tokenizer.encode(code, add_special_tokens=False)
        if len(alone) != 1 or alone[0] in taken:
            continue
        if tokenizer.encode(prefix + code, add_special_tokens=False) != [*prefix_ids, alone[0]]:
            continue
        codes.append(code)
        ids.append(alone[0])
        taken.add(alone[0])
        if len(codes) == MAX_CODES:
            return codes, ids
    raise ValueError(f"the tokenizer yields only {len(codes)} usable answer codes; the model needs {MAX_CODES}")
