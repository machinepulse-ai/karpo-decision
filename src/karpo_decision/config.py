"""``decision_config.json``: everything besides the weights that a checkpoint's answers depend on."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .protocol import QUESTION_TYPES

CONFIG_FILE = "decision_config.json"
FORMAT_VERSION = 1
#: How the full-attention (Qwen) or global (Gemma) layers see the prompt. The linear-attention and sliding-window
#: layers are causal in both.
Attention = Literal["causal", "bidirectional"]
ATTENTIONS = ("causal", "bidirectional")
#: The readout marker of Gemma checkpoints (``families.gemma.MARKER``): an existing special token that request text
#: can never produce.
DEFAULT_MARKER = "<mask>"
#: Image pixel bounds; the base model's vision tower reads images between these sizes.
MIN_PIXELS = 65_536
DEFAULT_MAX_PIXELS = 1_638_400


@dataclass(frozen=True)
class DecisionConfig:
    template: str
    codes: tuple[str, ...]
    token_ids: tuple[int, ...]  # each code's token: where the readout rows started
    attention: Attention
    base_model: str
    marker: str = DEFAULT_MARKER
    #: Softmax temperature per question type, fitted on held-out data after training.
    temperatures: dict[str, float] = field(default_factory=lambda: dict.fromkeys(QUESTION_TYPES, 1.0))
    max_pixels: int = DEFAULT_MAX_PIXELS
    readout_dtype: Literal["float32", "bfloat16"] = "float32"
    #: Where the backbone weights are when they are not in the checkpoint directory (an untrained initialization).
    backbone: str | None = None
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.attention not in ATTENTIONS:
            raise ValueError(f"attention must be one of {ATTENTIONS}, got {self.attention!r}")
        if len(self.codes) != len(self.token_ids):
            raise ValueError("every code needs its token id")
        if set(self.temperatures) != set(QUESTION_TYPES) or any(t <= 0 for t in self.temperatures.values()):
            raise ValueError(f"temperatures must be positive, one per question type {QUESTION_TYPES}")
        if self.format_version != FORMAT_VERSION:
            raise ValueError(f"unsupported decision_config format {self.format_version}")

    @classmethod
    def load(cls, directory: str | Path) -> DecisionConfig:
        data = json.loads((Path(directory) / CONFIG_FILE).read_text())
        return cls(**{**data, "codes": tuple(data["codes"]), "token_ids": tuple(data["token_ids"])})

    def save(self, directory: str | Path) -> None:
        data = asdict(self)
        data["codes"], data["token_ids"] = list(self.codes), list(self.token_ids)
        (Path(directory) / CONFIG_FILE).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
