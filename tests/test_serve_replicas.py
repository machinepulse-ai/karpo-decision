"""Rank 0's replicas: the least busy slot serves, ties rotate, a forwarded request comes back as the replica
answered it. No GPU: the replicas are plain HTTP servers."""

import http.client
import json
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from karpo_decision.serve import server


class Replica(BaseHTTPRequestHandler):
    """Answers ``/v1/systemone`` with the status and body its server holds, and ``/healthz`` with 200."""

    def log_message(self, format, *args):
        pass

    def _send(self, status, data, timing=None):
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        if timing:
            self.send_header("Server-Timing", timing)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, b"{}")

    def do_POST(self):
        self.server.bodies.append(self.rfile.read(int(self.headers["Content-Length"])))
        self._send(self.server.status, self.server.body, "encode;dur=1.0, share;dur=0.0, model;dur=2.0")


@pytest.fixture
def replica():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Replica)
    httpd.status, httpd.body, httpd.bodies = 200, b'{"answers": {}}', []
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def closed_port():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Replica)
    port = httpd.server_address[1]
    httpd.server_close()
    return port


def test_pick_takes_the_least_busy_slot_and_rotates_ties():
    replicas = server.Replicas([1, 2, 3])
    assert [replicas.pick(), replicas.pick(), replicas.pick()] == [0, 1, 2]
    replicas.done(1)
    assert replicas.pick() == 1
    for slot in (0, 1, 2):
        replicas.done(slot)
    assert replicas.pick() == 2  # the turn after slot 1, not slot 0 again


def test_forward_returns_what_the_replica_answered(replica):
    replicas = server.Replicas([0, replica.server_address[1]])
    replica.status, replica.body = 400, b'{"error": {"message": "no questions"}}'
    assert replicas.forward(1, b'{"state": "s"}') == (400, replica.body,
                                                      "encode;dur=1.0, share;dur=0.0, model;dur=2.0")
    assert replica.bodies == [b'{"state": "s"}']


def test_forward_to_a_replica_that_is_gone_raises():
    replicas = server.Replicas([0, closed_port()])
    with pytest.raises(OSError):
        replicas.forward(1, b"{}")


def test_wait_returns_once_every_replica_answers(replica):
    server.Replicas([0, replica.server_address[1]]).wait(timeout=5)


def test_wait_times_out_on_a_replica_that_never_listens():
    with pytest.raises(TimeoutError, match="replica 1"):
        server.Replicas([0, closed_port()]).wait(timeout=0.3)


class FakeRunner:
    """Stands in for the GPU: plans a request as one sequence and answers it."""

    def __init__(self):
        self.decider = types.SimpleNamespace(plan=lambda request: ["sequence"])
        self.ran = 0

    def run(self, questions, plan):
        self.ran += 1
        return {"answers": {name: "local" for name in questions}, "usage": {"input_tokens": 3}}


@pytest.fixture
def rank0(replica):
    """Rank 0 with one replica behind it: (post, runner, replicas, replica)."""
    runner, replicas = FakeRunner(), server.Replicas([0, replica.server_address[1]])
    httpd = server.Server(("127.0.0.1", 0), runner, "model", {"ranks": 2}, replicas)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def post(body):
        connection = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
        connection.request("POST", "/v1/systemone", body)
        response = connection.getresponse()
        result = response.status, json.loads(response.read()), response.getheader("Server-Timing")
        connection.close()
        return result

    yield post, runner, replicas, replica
    httpd.shutdown()
    httpd.server_close()


BODY = json.dumps({"state": "s", "questions": {"q": {"type": "choice", "options": ["a", "b"]}}}).encode()


def test_rank_0_serves_the_first_request_itself_and_forwards_the_next(rank0):
    post, runner, replicas, replica = rank0
    status, body, timing = post(BODY)
    assert status == 200 and body["answers"] == {"q": "local"} and runner.ran == 1 and replica.bodies == []
    status, body, timing = post(BODY)
    assert status == 200 and body == {"answers": {}} and runner.ran == 1 and replica.bodies == [BODY]
    names = [part.split(";")[0] for part in timing.split(", ")]
    assert names == ["encode", "share", "model"] and "encode;dur=1.0" in timing and "model;dur=2.0" in timing
    assert replicas.busy == [0, 0]


def test_a_replica_error_comes_back_as_it_is(rank0):
    post, runner, replicas, replica = rank0
    replica.status, replica.body = 400, b'{"error": {"message": "no questions"}}'
    post(BODY)
    assert post(BODY)[:2] == (400, {"error": {"message": "no questions"}})
    assert replicas.busy == [0, 0]


def test_a_replica_that_is_gone_answers_500(rank0):
    post, runner, replicas, replica = rank0
    replicas.ports[1] = closed_port()
    post(BODY)
    assert post(BODY)[:2] == (500, {"error": {"message": "inference failed"}})
    assert replicas.busy == [0, 0]


def test_without_replicas_every_request_is_served_here():
    runner = FakeRunner()
    httpd = server.Server(("127.0.0.1", 0), runner, "model", {"ranks": 1})
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        for _ in range(3):
            connection = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
            connection.request("POST", "/v1/systemone", BODY)
            assert connection.getresponse().status == 200
            connection.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert runner.ran == 3


def test_replicas_listen_on_loopback_ports_after_the_public_one():
    assert server.listen_address("0.0.0.0", 18440, 0) == ("0.0.0.0", 18440)
    assert server.listen_address("0.0.0.0", 18440, 3) == ("127.0.0.1", 18443)
