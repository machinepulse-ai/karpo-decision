"""``Runner.run`` gives a request's first pack its pixels (``Decider.prepare``) before the GPU is free: the next
request's are made while one runs, and at most ``PREPARED_AHEAD`` requests hold prepared pixels."""

import threading
import time
import types

import pytest

from karpo_decision.serve import server

#: How long a test waits for a thread before calling it stuck, and how long it lets a blocked thread show itself.
WAIT_SECONDS, SETTLE_SECONDS = 10.0, 0.2


class Decider:
    def __init__(self, ranks_lock: list, gate: threading.Event) -> None:
        self.ranks_lock, self.gate = ranks_lock, gate
        self.prepared: list[tuple[int, bool]] = []  # (request, whether the GPU lock was held meanwhile)

    def prepare(self, plan):
        self.prepared.append((plan[0], self.ranks_lock[0].locked()))
        return plan

    def decide(self, request, plans):
        self.gate.wait(WAIT_SECONDS)
        return {"answers": {}, "usage": {"input_tokens": 0}}


def until(condition, seconds=WAIT_SECONDS):
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        time.sleep(0.01)
    return condition()


def test_the_next_request_prepares_while_one_runs_and_no_more(monkeypatch):
    monkeypatch.setattr(server.torch.cuda, "set_device", lambda device: None)
    lock, gate = [], threading.Event()
    decider = Decider(lock, gate)
    ranks = server.Runner(decider, "cpu")
    lock.append(ranks.lock)
    first = threading.Thread(target=ranks.run, args=({}, [0]))
    first.start()
    assert until(ranks.lock.locked)  # the first request runs, and waits on the gate holding the GPU lock
    rest = [threading.Thread(target=ranks.run, args=({}, [k])) for k in (1, 2)]
    for thread in rest:
        thread.start()
    assert until(lambda: len(decider.prepared) == server.PREPARED_AHEAD)
    time.sleep(SETTLE_SECONDS)
    assert len(decider.prepared) == server.PREPARED_AHEAD  # the third waits for a free slot before preparing
    assert decider.prepared[1][1]  # the second prepared while the first held the GPU lock
    gate.set()
    for thread in (first, *rest):
        thread.join(WAIT_SECONDS)
    assert sorted(request for request, _ in decider.prepared) == [0, 1, 2]


def test_another_rank_gets_the_plan_and_the_gpu_the_prepared_one(monkeypatch):
    """Prepared pixels are not pickled for another rank, which reads its own."""
    sent, decided = [], []
    monkeypatch.setattr(server.torch.cuda, "set_device", lambda device: None)
    decider = types.SimpleNamespace(prepare=lambda plan: ["prepared", *plan],
                                    decide=lambda request, plans: decided.append(plans[0]) or {})
    channel = types.SimpleNamespace(call=lambda message: (sent.append(message[1]) or ({}, None), 0.0))
    ranks = server.Ranks(server.Runner(decider, "cpu"), [channel])
    ranks.run({}, ["plan"])  # here, on slot 0
    ranks.run({}, ["plan"])  # on the other rank
    assert decided == [["prepared", "plan"]] and sent == [["plan"]]


def test_a_failing_prepare_frees_its_slot(monkeypatch):
    def prepare(plan):
        raise ValueError("an image planned at 9 soft tokens has 8")

    ranks = server.Runner(types.SimpleNamespace(prepare=prepare), "cpu")

    def fail_more_than_the_slots():
        for _ in range(server.PREPARED_AHEAD + 1):
            with pytest.raises(ValueError):
                ranks.run({}, ["plan"])

    thread = threading.Thread(target=fail_more_than_the_slots)
    thread.start()
    thread.join(WAIT_SECONDS)
    assert not thread.is_alive()  # a slot kept by a failure would block the last call
