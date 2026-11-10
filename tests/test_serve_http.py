"""POST /v1/systemone over real HTTP on CPU: status codes and usage keys. The ranks and the decider are fakes."""

import http.client
import json
import threading

import pytest

from karpo_decision.encoding import LengthError
from karpo_decision.serve import server


class FakeDecider:
    def plan(self, request):
        if "long" in request["questions"]:
            raise LengthError("question 'long' needs 9 tokens; the limit is 8")
        return ["plan"]


class FakeRanks:
    decider = FakeDecider()

    def run(self, questions, plan):
        return {"answers": {key: {"answer": "A"} for key in questions}, "usage": {"input_tokens": 7}}, 0.0


@pytest.fixture
def post():
    httpd = server.Server(("127.0.0.1", 0), FakeRanks(), "test", {})
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def send(body):
        connection = http.client.HTTPConnection(*httpd.server_address, timeout=10)
        connection.request("POST", "/v1/systemone", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    yield send
    httpd.shutdown()
    httpd.server_close()


QUESTION = {"type": "noul", "instructions": "Is the page blank?"}


def test_usage_reports_input_and_output_tokens_in_snake_case(post):
    status, body = post({"state": "page", "questions": {"blank": QUESTION}})
    assert status == 200
    assert body["usage"] == {"input_tokens": 7, "output_tokens": 0}


def test_a_question_too_long_answers_422(post):
    status, body = post({"state": "page", "questions": {"long": QUESTION}})
    assert status == 422
    assert "the limit is 8" in body["error"]["message"]


def test_a_body_that_is_not_an_object_still_answers_400(post):
    status, body = post([1])
    assert status == 400
    assert "JSON object" in body["error"]["message"]
