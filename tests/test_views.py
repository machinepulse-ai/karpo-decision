"""Every view keeps a request's images (and any other key) beside its state and questions."""

from karpo_decision.views import views

CHOICE = {"type": "choice", "instructions": "Pick one", "criteria": {"a": "A", "b": "B"}}
NOUL = {"type": "noul", "instructions": "Yes?"}


def test_every_view_keeps_the_images():
    request = {"state": {"page": "x"}, "questions": {"c": CHOICE, "n": NOUL}, "images": ["data:image/png;base64,AA"]}
    for view in ("joint", "single", "reversed"):
        readings = views(request, view)
        assert readings and all(r["images"] == request["images"] for r in readings), view
