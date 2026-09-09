import numpy as np
import pytest
from PIL import Image
from experiments.parallel_method_protocol_v1 import (
    CORE_ATTACKS, Attack, latent_to_rgb_matrix, reference_to_observed,
    render_attack, rectify_once,
)

pytestmark = pytest.mark.unit


def test_identity_and_translation_recovery():
    values = np.zeros((19, 21, 3), dtype=np.uint8)
    values[7:11, 8:12] = 255
    source = Image.fromarray(values)
    unchanged, H = render_attack(source, CORE_ATTACKS[0])
    assert np.array_equal(np.asarray(unchanged), values)
    assert np.array_equal(H, np.eye(3))
    # The inverse translation must recover an interior marker, and catches H sign.
    observed = Image.fromarray(np.roll(values, 3, axis=1))
    restored = rectify_once(observed, reference_to_observed(source.size, translation=(3, 0)))
    assert np.array_equal(np.asarray(restored), values)


def test_clockwise_rotation_and_center():
    values = np.zeros((21, 21, 3), dtype=np.uint8)
    values[10, 14] = 255
    rotated, H = render_attack(Image.fromarray(values), Attack("quarter", 90))
    assert np.all(np.asarray(rotated)[14, 10] >= 254)
    assert np.allclose(H @ [10, 10, 1], [10, 10, 1])
    assert np.max(np.abs(np.asarray(rectify_once(rotated, H)).astype(int) - values)) <= 2


def test_noise_is_reproducible_not_blur():
    source = Image.new("RGB", (128, 128), (128, 128, 128))
    first, _ = render_attack(source, CORE_ATTACKS[5], noise_seed=14)
    second, _ = render_attack(source, CORE_ATTACKS[5], noise_seed=14)
    values = np.asarray(first).astype(float) / 255
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert .019 < values.std() < .021
    assert len(CORE_ATTACKS) == 9


def test_latent_pixel_centres_and_geometry_conjugation():
    S = latent_to_rgb_matrix((512, 512), (64, 64))
    assert np.allclose(S @ [0, 0, 1], [3.5, 3.5, 1])
    H = reference_to_observed((512, 512), 10, .75)
    latent_H = np.linalg.inv(S) @ H @ S
    assert np.allclose(latent_H, reference_to_observed((64, 64), 10, .75))
