"""TypeSafe's Python SDK (``typesafe-sdk``, a dev dependency) against the server over real HTTP on CPU: an image asset
goes as a multipart/form-data part of a zstd-compressed body, the decider gets its bytes at each reference, and the SDK
reads the response. The ranks and the decider are fakes."""

import threading

import pytest

typesafe_sdk = pytest.importorskip("typesafe_sdk")
httpx2 = pytest.importorskip("httpx2")

from karpo_decision.assets import resolve  # noqa: E402
from karpo_decision.protocol import validate  # noqa: E402
from karpo_decision.serve import server  # noqa: E402

#: PNG's signature (the SDK labels the part image/png) and bytes enough for a compressed body; never decoded here.
IMAGE = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 8
ANSWERS = {"open": {"type": "noul", "noul": 0.75},
           "tone": {"type": "choice", "choice": "calm", "probabilities": {"calm": 0.9, "angry": 0.1},
                    "confidence": 0.8}}


class Decider:
    def plan(self, request):
        validate(request)
        self.seen, self.resolved = request, resolve(request)
        return ["plan"]


class Ranks:
    decider = Decider()

    def run(self, questions, plan):
        return {"answers": ANSWERS, "usage": {"input_tokens": 600}}, 0.0


@pytest.fixture
def client():
    httpd = server.Server(("127.0.0.1", 0), Ranks(), "karpo-decision", {})
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    with typesafe_sdk.TypeSafeClient(api_key="unused", base_url=f"http://{host}:{port}", compression="zstd",
                                     http_client=httpx2.Client(trust_env=False)) as sdk:
        yield sdk
    httpd.shutdown()
    httpd.server_close()


def test_the_sdk_sends_an_image_asset_and_reads_the_answers(client):
    image = typesafe_sdk.Image(IMAGE)
    response = client.system_one(
        state={"screenshot": image, "page": "checkout"},
        questions={"open": {"type": "noul", "instructions": f"Is the dialog in {image} open?"},
                   "tone": {"type": "choice", "criteria": {"calm": None, "angry": None}}})
    assert response.nouls["open"].noul == 0.75 and response.choices["tone"].choice == "calm"
    assert response.usage.input_tokens == 600
    assert Ranks.decider.seen["assets"] == {str(image): {"type": "image", "data": IMAGE}}
    assert Ranks.decider.resolved.inline == [IMAGE, IMAGE]
