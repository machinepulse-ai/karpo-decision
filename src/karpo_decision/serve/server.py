"""System One over HTTP for a decision checkpoint, under torchrun, one replica per GPU.

    torchrun --nproc_per_node 1 -m karpo_decision.serve.server --checkpoint path/to/checkpoint --port 18440

Every rank loads the checkpoint on its own GPU and serves one request at a time. Rank 0 serves ``POST /v1/systemone``
(``{"state", "questions"}`` in, ``{"id", "model", "answers", "usage"}`` out; a screenshot as an ``image_url`` part at
``state.viewport_screenshot`` and more images at the top-level ``images``, as data URLs) and ``GET /healthz``, on
loopback by default (no TLS or keys of its own), once every replica is up. With ``--nproc_per_node N`` it hands each
request to the rank with the fewest in flight, itself included; rank r listens on loopback port ``port + r``. The rank
that runs a request encodes it: one outside the contract gets 400, and one with a question too long for the model
gets 422. The ``Server-Timing`` header gives the request's encode, share (the hop through rank 0) and model (passes,
readout, answers) times in milliseconds. Each request is logged by the rank that ran it, as one JSON line without
content.

A checkpoint serves through the ``Decider``, warmed up with a text request for a block layout and a screenshot for a
Qwen3.5 checkpoint. The 31B (a Gemma checkpoint with a vision tower, ``families/gemma/vision.py``) and Qwen3.5 read
images, and the 12B reads text only. The 31B alone runs the opt-in paths, which ``serve.paths`` refuses for the others
at startup: ``--fp8`` runs its decoder projections in FP8 (``serve.fp8``) and replays CUDA graphs captured at load for
single-question text-only requests (``serve.graphs``, skipped with ``--no-fast`` or the ``blackwell`` extra's
kernels), and ``--triton-attention`` runs the global layers' attention in one Triton kernel (sm_120 GPUs).

``--max-length`` is the longest sequence in tokens (default ``DEFAULT_MAX_LENGTH``; with ``--triton-attention``, under
131,072 for the 31B).
"""

from __future__ import annotations

import argparse
import base64
import http.client
import io
import json
import os
import re
import signal
import threading
import time
import uuid
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import torch

from ..config import DecisionConfig
from ..decider import DEFAULT_MAX_LENGTH, Decider
from ..encoding import Encoded, LengthError
from ..families.gemma.triton_attention import enable as enable_attention
from ..protocol import IMAGE_DATA_URL, IMAGE_URL, VIEWPORT_SCREENSHOT, Request, RequestError, images_of
from ..templates import BLOCKS, template
from . import graphs
from .fp8 import quantize
from .paths import GRAPHS, require

MAX_BODY_BYTES = 4 * 1024 * 1024
#: The screenshot warm-up: a blank page as a JPEG data URL in the state, with one question.
SCREENSHOT_WARMUP_SIZE = (768, 480)
SCREENSHOT_WARMUP_GREY = (200, 200, 200)
SCREENSHOT_WARMUP_JPEG_QUALITY = 75
SCREENSHOT_WARMUP_PAGE = {"url": "about:blank"}
SCREENSHOT_WARMUP_QUESTIONS = {"blank": {"type": "noul", "instructions": "Is the page blank?"}}
LOOPBACK_HOST = "127.0.0.1"
DECIDE_PATH = "/v1/systemone"
HEALTH_PATH = "/healthz"
SERVER_TIMING_HEADER = "Server-Timing"
#: The encode and model times in a replica's ``Server-Timing``.
TIMING = re.compile(r"(encode|model);dur=([0-9.]+)")
#: How long rank 0 waits for a replica's answer: requests queue on the replica behind its running one.
FORWARD_TIMEOUT_SECONDS = 600
HEALTH_PROBE_TIMEOUT_SECONDS = 5
HEALTH_POLL_INTERVAL_SECONDS = 0.1
#: How long rank 0 waits for every replica to listen: a load and the graph capture take a few minutes.
REPLICA_STARTUP_TIMEOUT_SECONDS = 3600
RANK = int(os.environ.get("LOCAL_RANK", "0"))


def log(**fields: Any) -> None:
    print(json.dumps({"rank": RANK, **fields}), flush=True)


def screenshot_request() -> Request:
    """A tiny request with a page-sized screenshot: the first image a process reads sets up the vision tower's kernels
    (seconds), so the server reads one before it listens."""
    from PIL import Image

    jpeg = io.BytesIO()
    Image.new("RGB", SCREENSHOT_WARMUP_SIZE, SCREENSHOT_WARMUP_GREY).save(jpeg, "JPEG",
                                                                         quality=SCREENSHOT_WARMUP_JPEG_QUALITY)
    screenshot = {"type": IMAGE_URL,
                  IMAGE_URL: {"url": IMAGE_DATA_URL + "jpeg;base64," + base64.b64encode(jpeg.getvalue()).decode()}}
    return {"state": {"page": SCREENSHOT_WARMUP_PAGE, VIEWPORT_SCREENSHOT: screenshot},
            "questions": SCREENSHOT_WARMUP_QUESTIONS}


def warmup_request(checkpoint: str) -> Request:
    """The request that warms a checkpoint up: text for a block layout, a screenshot for a joint layout
    (Qwen)."""
    if template(DecisionConfig.load(checkpoint).template).kind == BLOCKS:
        return graphs.WARMUP_REQUEST
    return screenshot_request()


def load_decider(checkpoint: str, device: torch.device, max_length: int = DEFAULT_MAX_LENGTH,
                 fp8: bool = False, fast: bool = True, triton_attention: bool = False) -> tuple[Decider, Request]:
    """The ``Decider``, reading sequences of up to ``max_length`` tokens, and the request that warms it up, on its
    GPU. ``fp8``: the decoder projections in FP8, and single-question requests replaying CUDA graphs unless ``fast``
    is false, the ``blackwell`` extra's kernels run or the backbone has no graphs. ``triton_attention``: the global
    layers' attention in one Triton kernel (``families.gemma.triton_attention``). A path the backbone does not run is
    refused before loading (``serve.paths``)."""
    supported = require(checkpoint, fp8=fp8, triton_attention=triton_attention)
    decider = Decider.from_pretrained(checkpoint, device=device, max_length=max_length)
    if fp8:
        quantize(decider.model)
    if triton_attention:
        enable_attention(decider.model, max_length)
    if fp8 and fast and GRAPHS in supported:
        graphs.fast(decider)
    return decider, warmup_request(checkpoint)


def scrub(node: Any) -> Any:
    """The body with every lone UTF-16 surrogate replaced: page text cut inside an emoji arrives as half a surrogate
    pair, which ``json.loads`` keeps and the tokenizer refuses."""
    if isinstance(node, str):
        try:
            node.encode("utf-8")
            return node
        except UnicodeEncodeError:
            return node.encode("utf-8", "replace").decode("utf-8")
    if isinstance(node, dict):
        return {scrub(key): scrub(item) for key, item in node.items()}
    if isinstance(node, list):
        return [scrub(item) for item in node]
    return node


def check_images(request: Request) -> None:
    """Images come inline, a data URL each: a path would read the server's own files."""
    for image in images_of(request)[1]:
        if not (isinstance(image, str) and image.startswith(IMAGE_DATA_URL)):
            raise RequestError("an image must be a data:image/... URL")


class Runner:
    """Runs one request at a time on this rank's GPU."""

    def __init__(self, decider: Decider, device: torch.device) -> None:
        self.decider, self.device = decider, device
        self.lock = threading.Lock()

    def run(self, questions: dict[str, Any], plan: list[Encoded]) -> dict[str, Any]:
        with self.lock:
            torch.cuda.set_device(self.device)
            return self.decider.decide({"state": None, "questions": questions}, plans=[plan])


class Replicas:
    """Rank 0's view of the replicas: slot 0 is this process, slot r the replica on loopback port ``ports[r]``. Each
    request goes to the slot with the fewest requests in flight, ties in turn."""

    def __init__(self, ports: list[int]) -> None:
        self.ports = ports
        self.busy = [0] * len(ports)
        self.turn = 0
        self.lock = threading.Lock()

    def pick(self) -> int:
        with self.lock:
            count = len(self.busy)
            slot = min(range(count), key=lambda r: (self.busy[r], (r - self.turn) % count))
            self.busy[slot] += 1
            self.turn = (slot + 1) % count
            return slot

    def done(self, slot: int) -> None:
        with self.lock:
            self.busy[slot] -= 1

    def forward(self, slot: int, body: bytes) -> tuple[int, bytes, str | None]:
        """The replica's status, body and ``Server-Timing`` for the request body, sent as it came."""
        connection = http.client.HTTPConnection(LOOPBACK_HOST, self.ports[slot], timeout=FORWARD_TIMEOUT_SECONDS)
        try:
            connection.request("POST", DECIDE_PATH, body)
            response = connection.getresponse()
            return response.status, response.read(), response.getheader(SERVER_TIMING_HEADER)
        finally:
            connection.close()

    def wait(self, timeout: float) -> None:
        """Returns once every replica answers ``/healthz``: each listens only after its load and warmup."""
        deadline = time.monotonic() + timeout
        for slot in range(1, len(self.ports)):
            while True:
                connection = http.client.HTTPConnection(LOOPBACK_HOST, self.ports[slot],
                                                        timeout=HEALTH_PROBE_TIMEOUT_SECONDS)
                try:
                    connection.request("GET", HEALTH_PATH)
                    if connection.getresponse().status == 200:
                        break
                except OSError:
                    pass
                finally:
                    connection.close()
                if time.monotonic() > deadline:
                    raise TimeoutError(f"replica {slot} did not come up")
                time.sleep(HEALTH_POLL_INTERVAL_SECONDS)


def listen_address(host: str, port: int, rank: int) -> tuple[str, int]:
    """Rank 0 listens where the server was asked to; replica r on loopback, r ports after it."""
    return (host, port) if rank == 0 else (LOOPBACK_HOST, port + rank)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256

    def __init__(self, address: tuple[str, int], runner: Runner, name: str, info: dict[str, Any],
                 replicas: Replicas | None = None) -> None:
        self.runner, self.name, self.info, self.replicas = runner, name, info, replicas
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True
    timeout = 60
    server: Server

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send(self, status: int, body: dict[str, Any], timing: str | None = None) -> None:
        self._send_bytes(status, json.dumps(body, allow_nan=False).encode(), timing)

    def _send_bytes(self, status: int, data: bytes, timing: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if timing:
            self.send_header(SERVER_TIMING_HEADER, timing)
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, close: bool = False) -> None:
        if close:
            self.close_connection = True
        self._send(status, {"error": {"message": message}})
        log(event="request", status=status, error=message[:200])

    def do_GET(self) -> None:
        if urlsplit(self.path).path == HEALTH_PATH:
            self._send(200, {"ok": True, "model": self.server.name, **self.server.info})
        else:
            self._error(404, "not found")

    def do_POST(self) -> None:
        start = time.perf_counter()
        if urlsplit(self.path).path != DECIDE_PATH:
            return self._error(404, "not found", close=True)
        length = self.headers.get("Content-Length", "")
        if not length.isdigit():
            return self._error(411, "Content-Length is required", close=True)
        if int(length) > MAX_BODY_BYTES:
            return self._error(413, f"body is larger than {MAX_BODY_BYTES} bytes", close=True)
        raw = self.rfile.read(int(length))
        replicas = self.server.replicas
        slot = replicas.pick() if replicas else 0
        try:
            if slot:
                self._forward(slot, raw, start)
            else:
                self._serve(raw, start)
        finally:
            if replicas:
                replicas.done(slot)

    def _forward(self, slot: int, raw: bytes, start: float) -> None:
        """The replica's answer as it is; ``share`` is what the hop through this rank added."""
        try:
            status, data, timing = self.server.replicas.forward(slot, raw)
        except Exception as error:  # noqa: BLE001 - one failed request must not stop the server
            log(event="request", status=500, error=type(error).__name__, stage="forward", replica=slot)
            return self._send(500, {"error": {"message": "inference failed"}})
        spent = dict(TIMING.findall(timing or ""))
        if len(spent) == 2:
            encode_ms, model_ms = float(spent["encode"]), float(spent["model"])
            shared = max((time.perf_counter() - start) * 1000 - encode_ms - model_ms, 0.0)
            timing = f"encode;dur={encode_ms:.1f}, share;dur={shared:.1f}, model;dur={model_ms:.1f}"
        self._send_bytes(status, data, timing)

    def _serve(self, raw: bytes, start: float) -> None:
        decider = self.server.runner.decider
        try:
            body = scrub(json.loads(raw))
            if not isinstance(body, dict):
                raise RequestError("the body must be a JSON object")
            request: Request = {"state": body.get("state", ""), "questions": body.get("questions")}
            if body.get("images"):
                request["images"] = body["images"]
            encoding = time.perf_counter()
            check_images(request)
            plan = decider.plan(request)
        except LengthError as error:  # a question that does not fit: the model declines it
            return self._error(422, str(error))
        except (ValueError, OSError) as error:  # json, the contract, an image that does not decode
            return self._error(400, str(error))
        except Exception as error:  # noqa: BLE001 - answer instead of dropping the connection
            log(event="request", status=500, error=type(error).__name__, stage="encode")
            return self._send(500, {"error": {"message": "the request could not be encoded"}})
        running = time.perf_counter()
        try:
            response = self.server.runner.run(request["questions"], plan)
        except Exception as error:  # noqa: BLE001 - one failed request must not stop the server
            log(event="request", status=500, error=type(error).__name__)
            return self._send(500, {"error": {"message": "inference failed"}})
        done = time.perf_counter()
        shared = 0.0
        encode_ms, model_ms = (running - encoding) * 1000, (done - running) * 1000
        timing = f"encode;dur={encode_ms:.1f}, share;dur={shared:.1f}, model;dur={model_ms:.1f}"
        tokens = response["usage"]["input_tokens"]
        self._send(200, {"id": f"decision-{uuid.uuid4().hex}", "model": self.server.name,
                         "answers": response["answers"], "usage": {"input_tokens": tokens, "output_tokens": 0}}, timing)
        log(event="request", status=200, questions=len(response["answers"]), tokens=tokens, sequences=len(plan),
            encode_ms=round(encode_ms, 1), share_ms=round(shared, 1), model_ms=round(model_ms, 1),
            total_ms=round((done - start) * 1000, 1))


def positive(text: str) -> int:
    if not text.isdecimal() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"{text} is not a positive integer")
    return int(text)


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default=LOOPBACK_HOST)
    parser.add_argument("--port", type=int, default=18440)
    parser.add_argument("--name", default="karpo-decision")
    parser.add_argument("--fp8", action="store_true", help="the decoder projections in FP8")
    parser.add_argument("--no-fast", action="store_true", help="with --fp8: no CUDA graphs")
    parser.add_argument("--triton-attention", action="store_true",
                        help="the global layers' attention in one Triton kernel (sm_120)")
    parser.add_argument("--max-length", type=positive, default=DEFAULT_MAX_LENGTH,
                        help="the longest sequence in tokens")
    return parser.parse_args(argv)


def main() -> None:
    args = arguments()
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    started = time.time()
    decider, warmup = load_decider(args.checkpoint, device, args.max_length, fp8=args.fp8,
                                   fast=not args.no_fast, triton_attention=args.triton_attention)
    loaded = time.time()
    decider.decide(warmup)
    info = {"ranks": world, "gpu": torch.cuda.get_device_name(device), "attention": decider.config.attention,
            "path": "fp8-graphs" if isinstance(decider.model, graphs.Graphs) else "fp8" if args.fp8 else "plain",
            "kernels": ["triton-attention"] if args.triton_attention else [], "max_length": decider.max_length}
    log(event="loaded", seconds=round(loaded - started), warmup_seconds=round(time.time() - loaded, 1), **info)
    replicas = None
    if local == 0 and world > 1:
        replicas = Replicas([args.port + rank for rank in range(world)])
        replicas.wait(timeout=REPLICA_STARTUP_TIMEOUT_SECONDS)
    address = listen_address(args.host, args.port, local)
    server = Server(address, Runner(decider, device), args.name, info, replicas)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(event="listening", host=address[0], port=address[1], model=args.name)
    stop.wait()
    server.shutdown()
    server.server_close()
    os._exit(0)


if __name__ == "__main__":
    main()
