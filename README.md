# karpo-decision

The inference code of Karpo's decision model: a System One request in, typed decisions out, read by a trained
readout from a Gemma 4 backbone. It answers through `karpo_decision.decider.Decider`, serves over HTTP with
`karpo_decision.serve.server` (`POST /v1/systemone`, `GET /healthz`), and runs the
[Decision Index](https://github.com/apolinario/decision-index) kit's suite with
`karpo_decision.serve.engine:DecisionIndexEngine` (the kit installed separately).

Model: [MachinePulseAI/karpo-decision-31B](https://huggingface.co/MachinePulseAI/karpo-decision-31B).

## Install

```bash
pip install "karpo-decision @ git+https://github.com/machinepulse-ai/karpo-decision"
```

On Blackwell GPUs, the `blackwell` extra adds faster attention kernels.

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

## Serving

```bash
torchrun --standalone --nproc_per_node 1 -m karpo_decision.serve.server --checkpoint <checkpoint dir> --port 18440
```

It listens on `127.0.0.1` by default and has no authentication or TLS of its own.
`--fp8` runs the decoder projections in FP8 (answers move by more than bf16 noise) and replays CUDA graphs captured
at startup (a few minutes) for single-question requests (not with the `blackwell` extra's kernels); `--no-fast` skips
the graphs.
`--triton-attention` runs the global layers' attention in one Triton kernel on sm_120 GPUs, twice as fast; requests of
16k-32k tokens take 1.2-1.4x less time (answers move by bf16 noise).

With Docker:

```bash
docker build -t karpo-decision .
docker run --gpus all -v <checkpoint dir>:/checkpoint -p 18440:18440 karpo-decision \
    torchrun --standalone --nproc_per_node 1 -m karpo_decision.serve.server --checkpoint /checkpoint --host 0.0.0.0
```

## Decision Index

```bash
python -m decision_index run --engine karpo_decision.serve.engine:DecisionIndexEngine \
    --option checkpoint=<checkpoint dir> --out runs/karpo-decision-31B
```

`--option fp8=true` runs the FP8 path of `--fp8`, graphs included (`--option fast=false` without them).
`--option triton_attention=true` is `--triton-attention`.

## Layout

```
src/karpo_decision/
  protocol.py answers.py   the System One request contract and the answer shapes
  config.py codes.py       a checkpoint's decision config and its answer codes
  templates.py blocks.py   how a request becomes text: a shared block and one branch per question
  encoding.py              tokens, segments and plans of a request
  checkpoint.py            a decision checkpoint on disk: config, processor, backbone, readout
  decider.py               answering: encode, pack, one forward pass per pack, read out
  views.py                 other readings of a request (single, reversed) and how their answers join
  model/ families/         the packed forward pass and the readout; the Gemma 4 backbone (families/gemma)
  serve/                   the HTTP server, the Decision Index kit's engine, the FP8 projections and the CUDA graphs
```

## Tests

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -e '.[dev]'
.venv/bin/python -m pytest -q          # tests needing a GPU or a Gemma base skip without them
.venv/bin/ruff check src tests
```

## License

Apache-2.0 ([LICENSE](LICENSE)).
