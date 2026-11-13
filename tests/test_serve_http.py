"""POST /v1/systemone over real HTTP on CPU: status codes as TypeSafe's System One gives them, usage keys, compressed
and multipart bodies. The ranks and the decider are fakes; the decider checks the contract and the assets."""

import gzip
import http.client
import json
import sys
import threading

import pytest

from karpo_decision.assets import resolve
from karpo_decision.encoding import LengthError
from karpo_decision.protocol import validate
from karpo_decision.serve import server, wire

zstd = pytest.importorskip("compression.zstd" if sys.version_info >= (3, 14) else "backports.zstd")


class FakeDecider:
    def plan(self, request):
        validate(request)
        resolve(request)
        if "long" in request["questions"]:
            raise LengthError("question 'long' needs 9 tokens; the limit is 8")
        if request.get("images") == ["data:image/png;base64,broken"]:
            raise OSError("cannot identify image file")
        self.seen = request
        return ["plan"]


class FakeRanks:
    decider = FakeDecider()

    def run(self, questions, plan):
        return {"answers": {key: {"answer": "A"} for key in questions}, "usage": {"input_tokens": 7}}, 0.0


@pytest.fixture
def post():
    httpd = server.Server(("127.0.0.1", 0), FakeRanks(), "test", {})
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def send(body, headers=None, data=None):
        connection = http.client.HTTPConnection(*httpd.server_address, timeout=10)
        content = json.dumps(body).encode() if data is None else data
        connection.request("POST", "/v1/systemone", content, {"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    yield send
    httpd.shutdown()
    httpd.server_close()


QUESTION = {"type": "noul", "instructions": "Is the page blank?"}
BODY = {"state": "page", "questions": {"blank": QUESTION}}


def test_usage_reports_input_and_output_tokens_in_snake_case(post):
    status, body = post(BODY)
    assert status == 200
    assert body["usage"] == {"input_tokens": 7, "output_tokens": 0}


def test_a_question_too_long_answers_422(post):
    status, body = post({"state": "page", "questions": {"long": QUESTION}})
    assert status == 422
    assert "the limit is 8" in body["error"]["message"]


def test_a_request_outside_the_contract_answers_422(post):
    status, body = post({"state": "page", "questions": {"where": {"type": "box", "instructions": "Find the cat"}}})
    assert status == 422 and "type must be" in body["error"]["message"]
    status, body = post([1])
    assert status == 422 and "JSON object" in body["error"]["message"]


def test_a_body_that_is_not_json_answers_400(post):
    status, _ = post(None, data=b"{not json")
    assert status == 400


PNG = "data:image/png;base64,iVBORw0KGgo="


def test_an_image_request_reaches_the_decider(post):
    status, body = post({"state": {"viewport_screenshot": {"type": "image_url", "image_url": {"url": PNG}}},
                         "questions": {"q": QUESTION}, "images": [PNG]})
    assert status == 200 and FakeRanks.decider.seen["images"] == [PNG]


def test_an_image_path_is_refused(post):
    status, body = post({"state": "", "questions": {"q": QUESTION}, "images": ["/etc/passwd"]})
    assert status == 400 and "data:image" in body["error"]["message"]


def test_an_image_that_does_not_decode_is_a_bad_request(post):
    status, _ = post({"state": "", "questions": {"q": QUESTION}, "images": ["data:image/png;base64,broken"]})
    assert status == 400


ASSET = {"type": "image", "format": "png", "data_base64": "iVBORw0KGgo="}


def test_assets_reach_the_decider(post):
    status, _ = post({"state": {"shot": "<a>"}, "questions": {"q": QUESTION}, "assets": {"<a>": ASSET}})
    assert status == 200 and FakeRanks.decider.seen["assets"] == {"<a>": ASSET}


def test_asset_rules_answer_400_and_asset_fields_422(post):
    status, body = post({"state": "no reference", "questions": {"q": QUESTION}, "assets": {"<a>": ASSET}})
    assert status == 400 and "not referenced" in body["error"]["message"]
    status, _ = post({"state": "<a>", "questions": {"q": QUESTION}, "assets": {}})
    assert status == 422
    status, _ = post({"state": "<a>", "questions": {"q": QUESTION}, "assets": {"<a>": {"data_base64": "AA=="}}})
    assert status == 422


@pytest.mark.parametrize(("coding", "compress"), [("gzip", gzip.compress), ("x-gzip", gzip.compress),
                                                  ("zstd", zstd.compress)])
def test_compressed_bodies_are_read(post, coding, compress):
    status, _ = post(None, {"Content-Encoding": coding}, compress(json.dumps(BODY).encode()))
    assert status == 200 and FakeRanks.decider.seen["state"] == "page"


def test_compression_errors(post):
    assert post(None, {"Content-Encoding": "br"}, b"x")[0] == 415
    assert post(None, {"Content-Encoding": "gzip, zstd"}, b"x")[0] == 415
    assert post(None, {"Content-Encoding": "gzip"}, b"not gzip")[0] == 400
    assert post(None, {"Content-Encoding": "gzip"}, gzip.compress(b" " * (server.MAX_BODY_BYTES + 1)))[0] == 413


def test_a_content_length_past_the_limit_answers_413(post):
    assert post(None, {"Content-Length": str(server.MAX_BODY_BYTES + 1)}, b"")[0] == 413


def multipart(request, parts, boundary="b0undary"):
    chunks = []
    for name, data in [("request", json.dumps(request).encode()), *parts]:
        chunks += [f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n'
                   f"Content-Type: application/octet-stream\r\n\r\n".encode(), data, b"\r\n"]
    chunks.append(f"--{boundary}--\r\n".encode())
    return {"Content-Type": f"multipart/form-data; boundary={boundary}"}, b"".join(chunks)


def test_a_multipart_body_gives_each_asset_its_part(post):
    headers, data = multipart({"state": "<a>", "questions": {"q": QUESTION},
                               "assets": {"<a>": {"type": "image", "part": "asset-0"}}}, [("asset-0", b"\x89PNG..")])
    status, _ = post(None, {**headers, "Content-Encoding": "zstd"}, zstd.compress(data))
    assert status == 200
    assert FakeRanks.decider.seen["assets"] == {"<a>": {"type": "image", "data": b"\x89PNG.."}}


def test_multipart_rules_answer_400(post):
    headers, data = multipart({"state": "s", "questions": {"q": QUESTION}}, [("asset-0", b"x")])
    status, body = post(None, headers, data)
    assert status == 400 and "exactly one asset" in body["error"]["message"]
    status, _ = post({"state": "<a>", "questions": {"q": QUESTION}, "assets": {"<a>": {"part": "asset-0"}}})
    assert status == 400


def test_a_text_state_with_an_object_typed_image_url_stays_text(post):
    status, _ = post({"state": {"form": [{"name": "avatar", "type": "image_url"}]}, "questions": {"q": QUESTION}})
    assert status == 200


def test_a_claimed_asset_of_another_type_answers_422(post):
    headers, data = multipart({"state": "<a>", "questions": {"q": QUESTION},
                               "assets": {"<a>": {"type": "video", "part": "asset-0"}}}, [("asset-0", b"x")])
    assert post(None, headers, data)[0] == 422


def test_a_body_of_too_many_gzip_members_answers_400(post):
    assert post(None, {"Content-Encoding": "gzip"}, gzip.compress(b" ") * (wire.MAX_MEMBERS + 1))[0] == 400
