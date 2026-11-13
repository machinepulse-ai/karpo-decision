"""``pack`` lays the image tensors of its sequences end to end, in sequence order; none without images."""

import torch

from karpo_decision.encoding import Encoded
from karpo_decision.model.packed import pack


def sequence(images: int):
    n = 6
    return Encoded(keys=("q",), counts=(2,), input_ids=torch.arange(n), markers=torch.tensor([n - 1]),
                   positions=torch.arange(n), segment_ids=torch.zeros(n, dtype=torch.int16),
                   pixel_values=torch.full((images, 4, 3), float(images)) if images else None,
                   image_position_ids=torch.full((images, 4, 2), images) if images else None)


def test_pack_lays_image_tensors_end_to_end_in_sequence_order():
    p = pack([sequence(2), sequence(0), sequence(1)], None, torch.device("cpu"))
    assert p.pixel_values.shape == (3, 4, 3) and p.image_position_ids.shape == (3, 4, 2)
    assert p.pixel_values[:, 0, 0].tolist() == [2.0, 2.0, 1.0]


def test_a_pack_without_images_has_none():
    p = pack([sequence(0), sequence(0)], None, torch.device("cpu"))
    assert p.pixel_values is None and p.image_position_ids is None
