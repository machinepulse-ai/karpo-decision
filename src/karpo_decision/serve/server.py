"""System One over HTTP for a decision checkpoint, under torchrun on one GPU.

    torchrun --nproc_per_node 1 -m karpo_decision.serve.server --checkpoint path/to/checkpoint --port 18440

Rank 0 serves ``POST /v1/systemone`` (``{"state", "questions"}`` in, ``{"id", "model", "answers", "usage"}`` out; a
screenshot as an ``image_url`` part at ``state.viewport_screenshot``) and ``GET /healthz``, on loopback by default (no
TLS or keys of its own). Rank 0 encodes each request, so one outside the contract gets 400 and never
reaches the other ranks; it hands the encoded sequences to them over a CPU (gloo) group, and every rank runs the same
passes, one request at a time. The ``Server-Timing`` header gives the request's encode, share (the hand-over) and
model (passes, readout, answers) times in milliseconds. Each request is logged as one JSON line without content.

A Gemma 4 checkpoint serves on one GPU (``--nproc_per_node 1``) through the ``Decider``, its decoder projections in
FP8 with ``--fp8`` (``serve.fp8``), single-question requests then replaying CUDA graphs captured at load
(``serve.graphs``, not with the ``blackwell`` extra's kernels; ``--no-fast`` skips them), warmed up with a text
request, since Gemma reads text only; ``--triton-attention`` runs the global layers' attention in one Triton kernel
(sm_120 GPUs).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import torch

from ..decider import Decider
from ..encoding import Encoded
from ..families.gemma.triton_attention import enable as enable_attention
from ..protocol import Request, RequestError, images_of
from . import graphs
from .fp8 import quantize

MAX_BODY_BYTES = 4 * 1024 * 1024
#: How long rank 0 waits for a replica's answer: requests queue on the replica behind its running one.
FORWARD_TIMEOUT = 600
RANK = int(os.environ.get("LOCAL_RANK", "0"))


def log(**fields: Any) -> None:
    print(json.dumps({"rank": RANK, **fields}), flush=True)


def text_warmup_request() -> Request:
    """A tiny request without images, for a family that reads text only (Gemma)."""
    return graphs.WARMUP_REQUEST


def load_decider(checkpoint: str, device: torch.device, fp8: bool = False,
                 fast: bool = True, triton_attention: bool = False) -> tuple[Decider, Request]:
    """The ``Decider`` and the request that warms it up: a Gemma 4 checkpoint runs on its GPU, its decoder projections
    in FP8 when ``fp8`` is true, and reads text only. With ``fp8``, single-question requests replay CUDA graphs unless
    ``fast`` is false or the ``blackwell`` extra's kernels run. ``triton_attention``: the global layers' attention in
    one Triton kernel (``families.gemma.triton_attention``)."""
    decider = Decider.from_pretrained(checkpoint, device=device)
    if fp8:
        quantize(decider.model)
    if triton_attention:
        enable_attention(decider.model)
    if fp8 and fast:
        graphs.fast(decider)
    return decider, text_warmup_request()


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
        if not (isinstance(image, str) and image.startswith("data:image/")):
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
        connection = http.client.HTTPConnection("127.0.0.1", self.ports[slot], timeout=FORWARD_TIMEOUT)
        try:
            connection.request("POST", "/v1/systemone", body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, response.read(), response.getheader("Server-Timing")
        finally:
            connection.close()

    def wait(self, timeout: float) -> None:
        """Returns once every replica answers ``/healthz``: each listens only after its load and warmup."""
        deadline = time.monotonic() + timeout
        for slot in range(1, len(self.ports)):
            while True:
                connection = http.client.HTTPConnection("127.0.0.1", self.ports[slot], timeout=5)
                try:
                    connection.request("GET", "/healthz")
                    if connection.getresponse().status == 200:
                        break
                except OSError:
                    pass
                finally:
                    connection.close()
                if time.monotonic() > deadline:
                    raise TimeoutError(f"replica {slot} did not come up")
                time.sleep(0.1)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256

    def __init__(self, address: tuple[str, int], runner: Runner, name: str, info: dict[str, Any]) -> None:
        self.runner, self.name, self.info = runner, name, info
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True
    timeout = 60
    server: Server

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send(self, status: int, body: dict[str, Any], timing: str | None = None) -> None:
        data = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if timing:
            self.send_header("Server-Timing", timing)
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, close: bool = False) -> None:
        if close:
            self.close_connection = True
        self._send(status, {"error": {"message": message}})
        log(event="request", status=status, error=message[:200])

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/healthz":
            self._send(200, {"ok": True, "model": self.server.name, **self.server.info})
        else:
            self._error(404, "not found")

    def do_POST(self) -> None:
        start = time.perf_counter()
        if urlsplit(self.path).path != "/v1/systemone":
            return self._error(404, "not found", close=True)
        length = self.headers.get("Content-Length", "")
        if not length.isdigit():
            return self._error(411, "Content-Length is required", close=True)
        if int(length) > MAX_BODY_BYTES:
            return self._error(413, f"body is larger than {MAX_BODY_BYTES} bytes", close=True)
        decider = self.server.runner.decider
        try:
            body = scrub(json.loads(self.rfile.read(int(length))))
            if not isinstance(body, dict):
                raise RequestError("the body must be a JSON object")
            request: Request = {"state": body.get("state", ""), "questions": body.get("questions")}
            if body.get("images"):
                request["images"] = body["images"]
            encoding = time.perf_counter()
            check_images(request)
            plan = decider.plan(request)
        except (ValueError, OSError) as error:  # json, the contract, a question too long, an image that does not decode
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
                         "answers": response["answers"], "usage": {"inputTokens": tokens, "outputTokens": 0}}, timing)
        log(event="request", status=200, questions=len(response["answers"]), tokens=tokens, sequences=len(plan),
            encode_ms=round(encode_ms, 1), share_ms=round(shared, 1), model_ms=round(model_ms, 1),
            total_ms=round((done - start) * 1000, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18440)
    parser.add_argument("--name", default="karpo-decision")
    parser.add_argument("--fp8", action="store_true", help="the decoder projections in FP8")
    parser.add_argument("--no-fast", action="store_true", help="with --fp8: no CUDA graphs")
    parser.add_argument("--triton-attention", action="store_true",
                        help="the global layers' attention in one Triton kernel (sm_120)")
    args = parser.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    started = time.time()
    decider, warmup = load_decider(args.checkpoint, device, fp8=args.fp8, fast=not args.no_fast,
                                   triton_attention=args.triton_attention)
    loaded = time.time()
    decider.decide(warmup)
    info = {"ranks": world, "gpu": torch.cuda.get_device_name(device), "attention": decider.config.attention,
            "path": "fp8-graphs" if isinstance(decider.model, graphs.Graphs) else "fp8" if args.fp8 else "plain",
            "kernels": ["triton-attention"] if args.triton_attention else []}
    log(event="loaded", seconds=round(loaded - started), warmup_seconds=round(time.time() - loaded, 1), **info)
    server = Server((args.host, args.port), Runner(decider, device), args.name, info)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(event="listening", host=args.host, port=args.port, model=args.name)
    stop.wait()
    server.shutdown()
    server.server_close()
    os._exit(0)


if __name__ == "__main__":
    main()
