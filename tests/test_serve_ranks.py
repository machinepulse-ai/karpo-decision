"""Rank 0 runs each request on the rank with the fewest in flight, ties in turn, over a channel per rank. The channels
are fakes, except in the last test: four CPU processes in a gloo group, the runners still fakes."""

import multiprocessing
import os
import socket
import threading
import time
import types

import pytest

from karpo_decision.encoding import ImageError
from karpo_decision.serve import server

LOOPBACK = "127.0.0.1"
WORLD = 4
REQUESTS = 24
CALLERS = 8
RUN_SECONDS = 0.05
GROUP_TIMEOUT_SECONDS = 120


class FakeRunner:
    def __init__(self, name="rank 0"):
        self.decider, self.name, self.ran = types.SimpleNamespace(), name, []

    def run(self, questions, plan):
        self.ran.append(questions)
        return {"by": self.name, "questions": questions, "plan": plan}


class FakeChannel:
    """Answers as its rank would: the response, or the error's name and message."""

    def __init__(self, name, error=None):
        self.name, self.error, self.sent, self.lock = name, error, [], threading.Lock()

    def call(self, message):
        self.sent.append(message)
        return ((None, self.error) if self.error else ({"by": self.name, "questions": message[0]}, None)), 1.5

    def send(self, message):
        self.sent.append(message)

    def receive(self):
        return server.READY


def test_pick_takes_the_least_busy_slot_and_rotates_ties():
    ranks = server.Ranks(FakeRunner(), [FakeChannel("rank 1"), FakeChannel("rank 2")])
    assert [ranks.pick(), ranks.pick(), ranks.pick()] == [0, 1, 2]
    ranks.done(1)
    assert ranks.pick() == 1
    for slot in (0, 1, 2):
        ranks.done(slot)
    assert ranks.pick() == 2  # the turn after slot 1, not slot 0 again


def test_requests_go_to_rank_0_and_to_the_other_ranks_in_turn():
    runner, channel = FakeRunner(), FakeChannel("rank 1")
    ranks = server.Ranks(runner, [channel])
    assert ranks.run({"a": 1}, ["plan"]) == ({"by": "rank 0", "questions": {"a": 1}, "plan": ["plan"]}, 0.0)
    assert ranks.run({"b": 2}, ["plan"]) == ({"by": "rank 1", "questions": {"b": 2}}, 1.5)
    assert runner.ran == [{"a": 1}] and channel.sent == [({"b": 2}, ["plan"])] and ranks.busy == [0, 0]


def test_a_rank_that_failed_raises_and_frees_its_slot():
    ranks = server.Ranks(FakeRunner(), [FakeChannel("rank 1", error=("OutOfMemoryError", "CUDA out of memory"))])
    ranks.run({}, [])
    with pytest.raises(RuntimeError, match="rank 1: OutOfMemoryError"):
        ranks.run({}, [])
    assert ranks.busy == [0, 0]


def test_an_image_another_rank_cannot_decode_is_an_image_error_here():
    failure = (ImageError.__name__, "an image does not decode: image file is truncated")
    ranks = server.Ranks(FakeRunner(), [FakeChannel("rank 1", error=failure)])
    ranks.run({}, [])
    with pytest.raises(ImageError, match="truncated"):
        ranks.run({}, [])
    assert ranks.busy == [0, 0]


def test_without_other_ranks_every_request_runs_here():
    runner = FakeRunner()
    ranks = server.Ranks(runner, [])
    for _ in range(3):
        assert ranks.run({}, [])[1] == 0.0
    assert len(runner.ran) == 3


def test_stop_reaches_every_rank():
    channels = [FakeChannel("rank 1"), FakeChannel("rank 2")]
    server.Ranks(FakeRunner(), channels).stop()
    assert [channel.sent for channel in channels] == [[server.STOP], [server.STOP]]


class SlowRunner:
    """Takes a moment, as a pass would, and names the process that ran."""

    def __init__(self, rank):
        self.decider, self.rank = None, rank

    def run(self, questions, plan):
        if questions.get("fail"):
            raise ValueError("refused")
        if questions.get("image"):
            raise ImageError("an image does not decode: image file is truncated")
        started = time.time()
        time.sleep(RUN_SECONDS)
        return {"rank": self.rank, "questions": questions, "plan": plan, "span": (started, time.time())}


def one_rank(rank, port, results):
    """One process of the group: rank 0 fires the requests from several threads, the others follow."""
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(WORLD), MASTER_ADDR=LOOPBACK, MASTER_PORT=str(port))
    dist = server.dist
    dist.init_process_group(server.CHANNEL_BACKEND, timeout=server.CHANNEL_TIMEOUT)
    groups = [dist.new_group([0, other], backend=server.CHANNEL_BACKEND, timeout=server.CHANNEL_TIMEOUT)
              for other in range(1, WORLD)]
    runner = SlowRunner(rank)
    if rank != 0:
        server.follow(runner, groups[rank - 1], rank)
        return
    ranks = server.Ranks(runner, [server.Channel(group, other) for other, group in enumerate(groups, start=1)])
    ranks.ready()
    answers, lock = [], threading.Lock()

    def caller(first):
        for number in range(first, REQUESTS, CALLERS):
            response, _ = ranks.run({"n": number}, [number])
            with lock:
                answers.append((number, response))

    threads = [threading.Thread(target=caller, args=(first,)) for first in range(CALLERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = ranks.channels[0].call(({"fail": True}, []))[0]
    image = ranks.channels[0].call(({"image": True}, []))[0]
    after = ranks.channels[0].call(({"n": -1}, [-1]))[0]
    ranks.stop()
    results.put({"answers": answers, "failed": failed, "image": image, "after": after})


def test_four_processes_share_the_requests_over_a_gloo_group():
    with socket.socket() as probe:
        probe.bind((LOOPBACK, 0))
        port = probe.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [context.Process(target=one_rank, args=(rank, port, results)) for rank in range(WORLD)]
    for process in processes:
        process.start()
    try:
        result = results.get(timeout=GROUP_TIMEOUT_SECONDS)
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
    answers = dict(result["answers"])
    assert sorted(answers) == list(range(REQUESTS))
    assert all(answers[n]["questions"] == {"n": n} and answers[n]["plan"] == [n] for n in answers)
    assert {answer["rank"] for answer in answers.values()} == set(range(WORLD))
    spans = [(answer["span"], answer["rank"]) for answer in answers.values()]
    # Two ranks ran at the same moment.
    assert any(a[0] < b[1] and b[0] < a[1] for a, first in spans for b, second in spans if first != second)
    assert result["failed"] == (None, ("ValueError", "refused"))
    assert result["image"] == (None, (ImageError.__name__, "an image does not decode: image file is truncated"))
    after, error = result["after"]
    # The rank kept following after its failure.
    assert error is None and (after["rank"], after["questions"], after["plan"]) == (1, {"n": -1}, [-1])
