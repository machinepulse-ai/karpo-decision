"""``Ranks.run`` gives a request's first pack its pixels (``Decider.prepare``) before the GPU is free: the next
request's are made while one runs, and at most ``PREPARED_AHEAD`` requests hold prepared pixels."""

import threading
import time

from karpo_decision.serve import server


class Decider:
    def __init__(self, ranks_lock: list, gate: threading.Event) -> None:
        self.ranks_lock, self.gate = ranks_lock, gate
        self.prepared: list[tuple[int, bool]] = []  # (request, whether the GPU lock was held meanwhile)

    def prepare(self, plan):
        self.prepared.append((plan[0], self.ranks_lock[0].locked()))
        return plan

    def decide(self, request, plans):
        self.gate.wait(10)
        return {"answers": {}, "usage": {"input_tokens": 0}}


def until(condition, seconds=10.0):
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        time.sleep(0.01)
    return condition()


def test_the_next_request_prepares_while_one_runs_and_no_more(monkeypatch):
    monkeypatch.setattr(server.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(server.dist, "broadcast_object_list", lambda *args, **kwargs: None)
    lock, gate = [], threading.Event()
    decider = Decider(lock, gate)
    ranks = server.Ranks(decider, None, "cpu")
    lock.append(ranks.lock)
    first = threading.Thread(target=ranks.run, args=({}, [0]))
    first.start()
    assert until(ranks.lock.locked)  # the first request runs, and waits on the gate holding the GPU lock
    rest = [threading.Thread(target=ranks.run, args=({}, [k])) for k in (1, 2)]
    for thread in rest:
        thread.start()
    assert until(lambda: len(decider.prepared) == server.PREPARED_AHEAD)
    time.sleep(0.2)
    assert len(decider.prepared) == server.PREPARED_AHEAD  # the third waits for a free slot before preparing
    assert decider.prepared[1][1]  # the second prepared while the first held the GPU lock
    gate.set()
    for thread in (first, *rest):
        thread.join(10)
    assert sorted(request for request, _ in decider.prepared) == [0, 1, 2]
