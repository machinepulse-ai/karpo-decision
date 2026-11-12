"""Rank 0's replicas: the least busy slot serves, ties rotate, a forwarded request comes back as the replica
answered it. No GPU: the replicas are plain HTTP servers."""

import threading
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
