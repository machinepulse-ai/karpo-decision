# karpo-decision

The inference code of Karpo's decision model: a System One request in, typed decisions out, read by a trained
readout from a Gemma 4 backbone. It answers through `karpo_decision.decider.Decider`, serves over HTTP with
`karpo_decision.serve.server` (`POST /v1/systemone`, `GET /healthz`), and runs the
[Decision Index](https://github.com/apolinario/decision-index) kit's suite with
`karpo_decision.serve.engine:DecisionIndexEngine` (the kit installed separately).

The released checkpoint is [MachinePulseAI/karpo-decision-31B](https://huggingface.co/MachinePulseAI/karpo-decision-31B)
(Gemma 4 31B).

## Install

```bash
pip install "karpo-decision @ git+https://github.com/machinepulse-ai/karpo-decision"
```

Python 3.12+. On Blackwell GPUs the `blackwell` extra adds Gemma's faster attention kernels (cuDNN, flash-attention 4);
without it the memory-efficient and FlashAttention-2 paths serve.

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

The request contract is in `protocol.py`, the answer shapes in `answers.py`.

## Serving

```bash
torchrun --standalone --nproc_per_node 1 -m karpo_decision.serve.server --checkpoint <checkpoint dir> --port 18440
```

The server runs on one GPU (`--nproc_per_node 1`). It listens on `127.0.0.1` unless `--host` says otherwise
(`--host 0.0.0.0` accepts other machines); it checks no keys and speaks no TLS itself, so that address belongs behind
something that does.

The image is `Dockerfile`: Python 3.12 and `requirements-serve.txt` (the pinned environment the model was evaluated
in), then this package:

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
  serve/                   the HTTP server and the Decision Index kit's engine
```

## Tests

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -e '.[dev]'
.venv/bin/python -m pytest -q          # tests needing a GPU or a Gemma base skip without them
.venv/bin/ruff check src tests
```

## License

Apache-2.0 ([LICENSE](LICENSE)).
