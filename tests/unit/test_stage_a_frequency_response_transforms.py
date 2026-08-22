from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

from experiments.stage_a_frequency_response.attack_transforms import apply_condition, public_noise_domain
from experiments.stage_a_frequency_response.protocol import CONDITIONS


def _image() -> Image.Image:
    yy, xx = np.mgrid[:32, :40]
    return Image.fromarray(np.stack(((xx * 5 + yy * 3) % 256, (xx * 7 + yy) % 256, (xx + yy * 11) % 256), axis=-1).astype(np.uint8), mode="RGB")


def _domain(condition: str) -> str:
    return public_noise_domain(protocol_id="standalone-lf-hf-frequency-response-v1", condition=condition, unit_id="frequency-response-0001", source_id="frequency-response-source-7001", generation_seed=1701121, height=32, width=40)


@pytest.mark.unit
def test_all_finite_conditions_are_ordinary_rgb_and_noise_is_public_deterministic() -> None:
    image = _image()
    original = np.asarray(image).copy()
    for condition in CONDITIONS:
        domain = _domain(condition) if condition.startswith("gaussian_noise_") else None
        result = apply_condition(image, condition, noise_domain=domain)
        assert result.mode == "RGB" and result.size == image.size and np.asarray(result).dtype == np.uint8
        if condition == "identity":
            np.testing.assert_array_equal(np.asarray(result), original)
    first = np.asarray(apply_condition(image, "gaussian_noise_std_0_01", noise_domain=_domain("gaussian_noise_std_0_01")))
    second = np.asarray(apply_condition(image, "gaussian_noise_std_0_01", noise_domain=_domain("gaussian_noise_std_0_01")))
    np.testing.assert_array_equal(first, second)
    assert all(token not in _domain("gaussian_noise_std_0_01") for token in ("key", "method", "prompt", "outcome"))
    np.testing.assert_array_equal(np.asarray(image), original)


@pytest.mark.unit
def test_transform_rejects_nonordinary_or_missing_noise_domain() -> None:
    with pytest.raises(ValueError, match="uint8"):
        apply_condition(np.zeros((8, 8, 3), dtype=np.float32), "jpeg_q90")
    with pytest.raises(ValueError, match="requires its public domain"):
        apply_condition(_image(), "gaussian_noise_std_0_02")
