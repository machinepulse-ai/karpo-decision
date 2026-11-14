# karpo-decision

The inference code of Karpo's decision model: a System One request in, typed decisions out, read by a trained
readout from a Gemma 4 or Qwen3.5 backbone. It answers through `karpo_decision.decider.Decider`, serves over HTTP with
`karpo_decision.serve.server` (`POST /v1/systemone`, `GET /healthz`), and runs the
[Decision Index](https://github.com/apolinario/decision-index) kit's suite with
`karpo_decision.serve.engine:DecisionIndexEngine` (the kit installed separately).

Models:

| Model | Backbone | Reads |
|---|---|---|
| [karpo-decision-31B](https://huggingface.co/MachinePulseAI/karpo-decision-31B) | Gemma 4 31B | text, images |
| [karpo-decision-12B](https://huggingface.co/MachinePulseAI/karpo-decision-12B) | Gemma 4 12B | text |
| [karpo-decision-9B](https://huggingface.co/MachinePulseAI/karpo-decision-9B) | Qwen3.5 9B | text, a screenshot |
| [karpo-decision-4B](https://huggingface.co/MachinePulseAI/karpo-decision-4B) | Qwen3.5 4B | text, a screenshot |
| [karpo-decision-2B](https://huggingface.co/MachinePulseAI/karpo-decision-2B) | Qwen3.5 2B | text, a screenshot |
| [karpo-decision-0.8B](https://huggingface.co/MachinePulseAI/karpo-decision-0.8B) | Qwen3.5 0.8B | text, a screenshot |

## Install

```bash
pip install "karpo-decision @ git+https://github.com/machinepulse-ai/karpo-decision"
```

The Qwen3.5 checkpoints need the `qwen` extra:

```bash
pip install "karpo-decision[qwen] @ git+https://github.com/machinepulse-ai/karpo-decision"
```

On Blackwell GPUs, the `blackwell` extra adds faster attention kernels for Gemma.

## Use

```python
from huggingface_hub import snapshot_download
from karpo_decision.decider import Decider

decider = Decider.from_pretrained(snapshot_download("MachinePulseAI/karpo-decision-31B"), device="cuda")
response = decider.decide({
    "state": "Order 1182: the customer asks to return a blender bought 45 days ago. Policy: returns within 30 days.",
    "questions": {
        "eligible": {"type": "noul", "instructions": "Is the return eligible under the policy?"},
        "action": {"type": "choice", "instructions": "What should the agent do?",
                   "criteria": {"refund": "Issue a full refund", "decline": "Politely decline the return",
                                "escalate": "Escalate to a supervisor"}},
    },
})
print(response["answers"])
```

The 31B and the Qwen3.5 checkpoints also read a screenshot at `state.viewport_screenshot`, as an `image_url` part
with a `data:image/...` URL. The 31B reads more images at `images` (data URLs; in Python also PIL images or file
paths); each image takes up to 560 tokens of `--max-length` plus a few markers.

## Serving

```bash
torchrun --standalone --nproc_per_node 1 -m karpo_decision.serve.server --checkpoint <checkpoint dir> --port 18440
```

It listens on `127.0.0.1` by default and has no authentication or TLS of its own.
`--fp8` and `--graphs` run on every checkpoint, alone or together; `--triton-attention` on the Gemma checkpoints.
`--fp8` runs the decoder projections in FP8 (answers move by more than bf16 noise); without `--graphs` it is slower
than the plain path on the Qwen3.5 checkpoints.
`--graphs` replays CUDA graphs captured at startup (2-7 minutes; the layers are compiled first on Qwen3.5). Gemma:
single-question text-only requests up to the Gemma graphs' limit, not with the `blackwell` extra's kernels. Qwen3.5:
any request read as one sequence (several questions or a screenshot included) up to min(`--max-length`, 65,536)
tokens. Other passes run without the graphs.
`--triton-attention` runs the global layers' attention in one Triton kernel on sm_120 GPUs, twice as fast; requests of
16k-32k tokens take 1.2-1.4x less time (answers move by bf16 noise).

`--max-length` sets the longest sequence in tokens (default 32,768; with `--triton-attention` the limit depends on the
checkpoint, 131,072 tokens for the 31B); a question that does not fit gets HTTP 422.

With Docker:

```bash
docker build -t karpo-decision .
docker run --gpus all -v <checkpoint dir>:/checkpoint -p 18440:18440 karpo-decision \
    torchrun --standalone --nproc_per_node 1 -m karpo_decision.serve.server --checkpoint /checkpoint --host 0.0.0.0
```

### Opt-in paths by checkpoint

Same top answer as the plain path, and latency per request once the kernels are compiled (median / mean), one request
at a time on one GPU, over 2,215 Decision Index requests:

| Checkpoint | Plain | `--graphs` | `--fp8` | `--fp8 --graphs` | `--triton-attention` |
|---|---|---|---|---|---|
| 31B | 114 / 510 ms | 100.0%, 114 / 509 ms | 99.4%, 82 / 367 ms | 99.4%, 77 / 354 ms | 99.8%, 114 / 460 ms |
| 12B | 57 / 214 ms | 99.9%, 46 / 204 ms | 99.1%, 61 / 166 ms | 99.1%, 31 / 147 ms | 99.9%, 54 / 190 ms |
| 9B | 71 / 155 ms | 99.9%, 31 / 130 ms | 99.6%, 106 / 192 ms | 99.5%, 21 / 93 ms | |
| 4B | 72 / 123 ms | 99.9%, 19 / 87 ms | 99.5%, 108 / 166 ms | 99.5%, 14 / 68 ms | |
| 2B | 52 / 72 ms | 99.9%, 11 / 41 ms | 99.2%, 82 / 105 ms | 99.3%, 9 / 34 ms | |
| 0.8B | 53 / 65 ms | 99.9%, 8 / 31 ms | 99.1%, 79 / 93 ms | 99.0%, 8 / 27 ms | |

On the Qwen3.5 checkpoints kernels compile on first use: in a first pass over these requests the mean latency was up
to 2.7x higher.

## Decision Index

```bash
python -m decision_index run --engine karpo_decision.serve.engine:DecisionIndexEngine \
    --option checkpoint=<checkpoint dir> --out runs/karpo-decision-31B
```

`--option fp8=true` is `--fp8`.
`--option triton_attention=true` is `--triton-attention`.
`--option graphs=true` is `--graphs`.
`--option image_root=<dir>` is the directory the vision suite's image references are relative to.

## Layout

```
src/karpo_decision/
  protocol.py answers.py   the System One request contract and the answer shapes
  config.py codes.py       a checkpoint's decision config and its answer codes
  templates.py blocks.py   how a request becomes text: one sequence (Qwen) or a shared block and branches (Gemma)
  encoding.py              tokens, segments and plans of a request
  checkpoint.py            a decision checkpoint on disk: config, processor, backbone, readout
  decider.py               answering: encode, pack, one forward pass per pack, read out
  views.py                 other readings of a request (single, reversed) and how their answers join
  model/ families/         the packed forward pass and the readout; the Gemma 4 and Qwen3.5 backbones (families/)
  serve/                   the HTTP server, the Decision Index kit's engine, the FP8 projections and the CUDA graphs
```

## Tests

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -e '.[dev]'
.venv/bin/python -m pytest -q          # GPU, Gemma-base, KARPO_TOKENIZER (Qwen3.5) and KARPO_GEMMA_TOKENIZER (a
                                       # directory with the 31B's tokenizer and configs) tests skip without them
.venv/bin/ruff check src tests
```

## License

Apache-2.0 ([LICENSE](LICENSE)).
