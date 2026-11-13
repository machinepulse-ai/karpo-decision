"""``families.gemma.vision`` on CPU: images prepared from PIL images, data URLs and paths (soft tokens as Gemma 4's
image processor counts them) and bad ones refused, embeddings from a tiny random vision tower, and which checkpoints get
``vision``."""

import base64
import io
import json

import pytest
import torch

pytest.importorskip("PIL")
pytest.importorskip("transformers.models.gemma4")
from PIL import Image  # noqa: E402
from transformers.models.gemma4.configuration_gemma4 import (  # noqa: E402
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)
from transformers.models.gemma4.modeling_gemma4 import Gemma4Model  # noqa: E402

from karpo_decision.encoding import open_image  # noqa: E402
from karpo_decision.families import gemma  # noqa: E402
from karpo_decision.families.gemma import vision  # noqa: E402

TOKENS = vision.ImageTokens(begin=290, soft=291, end=292)
#: The smallest budget Gemma 4's processor takes: images of at most 70 soft tokens keep the CPU tests fast.
SMALL = 70


def tiny_backbone(window: int, types=("sliding_attention", "full_attention"), *, hidden: int = 32,
                  intermediate: int = 48, head_dim: int = 8, global_head_dim: int = 16) -> Gemma4Model:
    """A tiny random Gemma 4 with a one-layer vision tower, float64 (the tests compare paths to rounding); the text
    model's sizes are keyword arguments."""
    torch.manual_seed(0)
    text = Gemma4TextConfig(vocab_size=300, hidden_size=hidden, intermediate_size=intermediate,
                            num_hidden_layers=len(types), num_attention_heads=4, num_key_value_heads=2,
                            head_dim=head_dim, global_head_dim=global_head_dim, num_global_key_value_heads=1,
                            attention_k_eq_v=True, sliding_window=window, layer_types=list(types),
                            hidden_size_per_layer_input=0, final_logit_softcapping=30.0)
    tower = Gemma4VisionConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=1, num_attention_heads=2,
                               num_key_value_heads=2, head_dim=16, patch_size=16, pooling_kernel_size=3,
                               position_embedding_size=64)
    config = Gemma4Config(text_config=text.to_dict(), vision_config=tower.to_dict(), boi_token_id=TOKENS.begin,
                          image_token_id=TOKENS.soft, eoi_token_id=TOKENS.end, audio_config=None)
    config.get_text_config()._attn_implementation = "sdpa"
    model = Gemma4Model(config).double().eval()
    with torch.no_grad():
        for p in model.language_model.parameters():
            p.add_(0.05 * torch.randn_like(p))
        for layer in model.language_model.layers:
            layer.layer_scalar.fill_(0.9)
    return model


def data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_prepare_reads_pil_images_data_urls_and_paths(tmp_path):
    red = Image.new("RGB", (640, 480), (200, 30, 30))
    path = tmp_path / "tall.png"
    Image.new("RGB", (300, 900), (10, 200, 30)).save(path)
    images = vision.Vision(TOKENS, max_soft_tokens=SMALL).prepare([red, data_url(red), str(path)])
    assert images.pixel_values.shape[0] == images.position_ids.shape[0] == 3
    assert images.tokens[0] == images.tokens[1]
    valid = (images.position_ids != -1).all(-1).sum(-1)
    assert list(images.tokens) == (valid // 9).tolist()
    assert all(0 < n <= SMALL for n in images.tokens)


def test_the_default_budget_is_560_soft_tokens():
    images = vision.Vision(TOKENS).prepare([Image.new("RGB", (4000, 3000))])
    assert vision.MAX_SOFT_TOKENS == 560
    assert 280 < images.tokens[0] <= vision.MAX_SOFT_TOKENS


def test_summary_gives_pixels_per_image_and_soft_tokens():
    v = vision.Vision(TOKENS, max_soft_tokens=SMALL)
    images = v.prepare([Image.new("RGB", (640, 480)), Image.new("RGB", (300, 900))])
    pixels, tokens = v.summary(images.position_ids)
    valid = (images.position_ids != -1).all(-1).sum(-1)
    assert pixels == (valid * 16 * 16).tolist() and tokens == sum(images.tokens)


def test_a_data_url_that_is_not_an_image_is_refused():
    bad = "data:image/png;base64," + base64.b64encode(b"not an image").decode()
    with pytest.raises((ValueError, OSError)):
        vision.Vision(TOKENS, max_soft_tokens=SMALL).prepare([bad])


def test_a_data_url_without_a_payload_is_refused():
    with pytest.raises((ValueError, OSError)):
        open_image("data:image/png")


def test_a_decompression_bomb_is_refused(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)  # PIL raises DecompressionBombError past twice this
    with pytest.raises((ValueError, OSError)):
        open_image(data_url(Image.new("RGB", (8, 8))))


def test_embed_gives_one_row_per_soft_token_in_image_order():
    backbone = tiny_backbone(window=128)
    v = vision.Vision(TOKENS, max_soft_tokens=SMALL)
    images = v.prepare([Image.new("RGB", (640, 480), (200, 30, 30)), Image.new("RGB", (300, 900), (10, 200, 30))])
    with torch.inference_mode():
        both = vision.embed(backbone, images.pixel_values, images.position_ids)
        second = vision.embed(backbone, images.pixel_values[1:], images.position_ids[1:])
    assert both.shape == (sum(images.tokens), 32) and both.dtype == torch.float64
    torch.testing.assert_close(both[images.tokens[0]:], second)


@pytest.mark.parametrize(("config", "reads_images"), [
    ({"model_type": "gemma4", "vision_config": {"model_type": "gemma4_vision"}, "boi_token_id": 7,
      "image_token_id": 8, "eoi_token_id": 9}, True),
    ({"model_type": "gemma4_unified", "vision_config": None, "boi_token_id": 7, "image_token_id": 8,
      "eoi_token_id": 9}, False),
    ({"model_type": "gemma4"}, False),
    (None, False)])
def test_vision_of_reads_the_checkpoints_config(tmp_path, config, reads_images):
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config))
    found = gemma.vision_of(tmp_path)
    assert (found is not None) == reads_images
    if reads_images:
        assert found.tokens == vision.ImageTokens(begin=7, soft=8, end=9)
