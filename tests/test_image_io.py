"""``encoding.image_bytes`` and ``encoding.image_size``: a data URL's base64 read once, an image's size read from its
header alone, and what fails there (not an image, a decompression bomb, bad base64)."""

import base64
import io

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from karpo_decision.encoding import ImageError, image_bytes, image_size, open_image  # noqa: E402

SIZE = (300, 900)


def png(size=SIZE):
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 200, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def data_url(raw):
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def test_image_bytes_reads_a_data_url_and_keeps_anything_else():
    raw, picture = png(), Image.new("RGB", (2, 2))
    assert image_bytes(data_url(raw)) == raw
    assert image_bytes(raw) is raw and image_bytes("some/path.png") == "some/path.png"
    assert image_bytes(picture) is picture


def test_image_size_reads_bytes_data_urls_paths_and_images(tmp_path):
    raw = png()
    path = tmp_path / "tall.png"
    path.write_bytes(raw)
    values = (raw, data_url(raw), str(path), Image.open(io.BytesIO(raw)))
    assert [image_size(value) for value in values] == [SIZE] * len(values)


def test_image_size_reads_the_header_only():
    truncated = png()[: len(png()) // 2]
    assert image_size(truncated) == SIZE  # the header reads
    with pytest.raises(OSError):
        open_image(truncated)  # the pixels do not


def test_image_size_refuses_what_is_not_an_image_and_a_decompression_bomb(monkeypatch):
    with pytest.raises(OSError):
        image_size(b"not an image")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)  # PIL raises DecompressionBombError past twice this
    with pytest.raises(ValueError):
        image_size(png())


def test_bad_base64_in_a_data_url_is_a_value_error():
    with pytest.raises(ValueError):
        image_bytes("data:image/png;base64,%%%")


def test_image_error_is_a_value_error():
    assert issubclass(ImageError, ValueError)
