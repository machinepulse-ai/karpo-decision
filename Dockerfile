# The serving image of a decision checkpoint, with the pinned dependencies of requirements-serve.txt. The command is
# torchrun ... -m karpo_decision.serve.server, with the checkpoint mounted into the container (README).
FROM python:3.12-slim

# Triton builds its kernel launchers with the system C compiler at run time: torch.compile of Gemma's pointwise work
# (families/gemma/fused.py) needs one.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
WORKDIR /opt/karpo-decision

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
