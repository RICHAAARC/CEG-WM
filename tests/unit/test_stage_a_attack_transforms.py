from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image
import pytest

from experiments.stage_a.attack_transforms import (
    ATTACK_IDS,
    CONDITION_ORDER,
    IDENTITY_REFERENCE,
    apply_attack,
    public_noise_domain,
)


def _image() -> Image.Image:
    yy, xx = np.mgrid[:32, :40]
    pixels = np.stack(
        ((xx * 7 + yy * 3) % 256, (xx * 2 + yy * 11) % 256, (xx * 13 + yy) % 256),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _domain(unit_id: str = "attack-comp-0001") -> str:
    return public_noise_domain(
        protocol_id="cegwm-stage-a-hf-lf-attack-complementarity-v1",
        attack_id="gaussian_noise_std_0_01",
        unit_id=unit_id,
        source_id="attack-prompt-6001",
        generation_seed=1317317,
        height=32,
        width=40,
    )


@pytest.mark.unit
def test_attack_set_separates_identity_reference_from_three_attacks() -> None:
    assert CONDITION_ORDER == (
        "identity_reference",
        "jpeg_q75",
        "gaussian_blur_sigma_1",
        "gaussian_noise_std_0_01",
    )
    assert IDENTITY_REFERENCE not in ATTACK_IDS
    assert len(ATTACK_IDS) == 3


@pytest.mark.unit
def test_frozen_attacks_return_same_shape_rgb8_without_mutating_input() -> None:
    image = _image()
    before = np.asarray(image).copy()
    for attack_id in ATTACK_IDS:
        attacked = apply_attack(
            image,
            attack_id,
            noise_domain=_domain() if attack_id == "gaussian_noise_std_0_01" else None,
        )
        pixels = np.asarray(attacked)
        assert attacked.mode == "RGB" and attacked.size == image.size
        assert pixels.dtype == np.uint8 and pixels.shape == before.shape
        assert np.any(pixels != before)
    np.testing.assert_array_equal(np.asarray(image), before)


@pytest.mark.unit
def test_public_noise_is_deterministic_shared_and_unit_separated() -> None:
    image = _image()
    first = np.asarray(
        apply_attack(image, "gaussian_noise_std_0_01", noise_domain=_domain())
    )
    second = np.asarray(
        apply_attack(image, "gaussian_noise_std_0_01", noise_domain=_domain())
    )
    changed = np.asarray(
        apply_attack(
            image,
            "gaussian_noise_std_0_01",
            noise_domain=_domain("attack-comp-0002"),
        )
    )
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert "key" not in _domain() and "method" not in _domain()


@pytest.mark.unit
def test_jpeg_parameters_and_blur_are_fixed_by_output_determinism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _image()
    save_calls: list[dict[str, object]] = []
    original_save = Image.Image.save

    def recording_save(self: Image.Image, fp: object, format: str | None = None, **params: object) -> None:
        save_calls.append({"format": format, **params})
        original_save(self, fp, format=format, **params)

    monkeypatch.setattr(Image.Image, "save", recording_save)
    jpeg_first = apply_attack(image, "jpeg_q75")
    jpeg_second = apply_attack(image, "jpeg_q75")
    blur_first = apply_attack(image, "gaussian_blur_sigma_1")
    blur_second = apply_attack(image, "gaussian_blur_sigma_1")
    assert hashlib.sha256(np.asarray(jpeg_first).tobytes()).digest() == hashlib.sha256(
        np.asarray(jpeg_second).tobytes()
    ).digest()
    assert all(
        call == {
            "format": "JPEG",
            "quality": 75,
            "subsampling": 2,
            "optimize": False,
            "progressive": False,
            "exif": b"",
            "icc_profile": None,
        }
        for call in save_calls
    )
    np.testing.assert_array_equal(np.asarray(blur_first), np.asarray(blur_second))

    source = np.asarray(image, dtype=np.float64) / 255.0
    coordinates = np.arange(-3, 4, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(coordinates))
    kernel /= kernel.sum()
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(source, ((3, 3), (3, 3), (0, 0)), mode="reflect"),
        (7, 7),
        axis=(0, 1),
    )
    expected = np.einsum("ijcxy,x,y->ijc", windows, kernel, kernel)
    expected_rgb8 = np.rint(np.clip(expected, 0.0, 1.0) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(np.asarray(blur_first), expected_rgb8)


@pytest.mark.unit
def test_attack_inputs_and_noise_identity_fail_closed() -> None:
    with pytest.raises(TypeError, match="Pillow RGB"):
        apply_attack(np.zeros((8, 8, 3), dtype=np.uint8), "jpeg_q75")
    with pytest.raises(TypeError, match="Pillow RGB"):
        apply_attack(Image.new("L", (8, 8)), "jpeg_q75")
    with pytest.raises(ValueError, match="requires"):
        apply_attack(_image(), "gaussian_noise_std_0_01")
    with pytest.raises(ValueError, match="not frozen"):
        apply_attack(_image(), "identity_reference")
    with pytest.raises(ValueError, match="only defined"):
        public_noise_domain(
            protocol_id="p",
            attack_id="jpeg_q75",
            unit_id="u",
            source_id="s",
            generation_seed=1,
            height=8,
            width=8,
        )
