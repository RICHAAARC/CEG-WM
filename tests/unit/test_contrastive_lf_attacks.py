from __future__ import annotations

import pytest
import torch

from experiments.attacks.contrastive_lf_branch_attribution import (
    ATTACK_IDS,
    apply_contrastive_lf_attack,
    validate_jpeg_capability,
)


def _image() -> torch.Tensor:
    return torch.arange(3 * 12 * 12, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(1, 3, 12, 12).contiguous()


@pytest.mark.unit
def test_stage_a_attacks_are_exact_and_deterministic() -> None:
    assert ATTACK_IDS == (
        "identity",
        "jpeg_quality_70",
        "gaussian_blur_sigma_1",
        "gaussian_noise_sigma_0_01",
    )
    assert len(validate_jpeg_capability()) == 5
    image = _image()
    for attack_id in ATTACK_IDS:
        first = apply_contrastive_lf_attack(
            image,
            attack_id,
            source_cluster_id="1" * 64,
            generation_seed=202608200000,
        )
        second = apply_contrastive_lf_attack(
            image,
            attack_id,
            source_cluster_id="1" * 64,
            generation_seed=202608200000,
        )
        assert first.attack_id == attack_id
        assert torch.equal(first.image_rgb8, second.image_rgb8)
        assert first.image_rgb8.dtype is torch.uint8
        assert first.image_rgb8.is_contiguous()
        assert first.attack_identity == second.attack_identity


@pytest.mark.unit
def test_noise_seed_is_public_and_cluster_bound() -> None:
    image = _image()
    first = apply_contrastive_lf_attack(
        image,
        "gaussian_noise_sigma_0_01",
        source_cluster_id="1" * 64,
        generation_seed=202608200000,
    )
    second = apply_contrastive_lf_attack(
        image,
        "gaussian_noise_sigma_0_01",
        source_cluster_id="2" * 64,
        generation_seed=202608200000,
    )
    assert first.attacked_rgb8_digest != second.attacked_rgb8_digest
