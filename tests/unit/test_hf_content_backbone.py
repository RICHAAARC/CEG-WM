from dataclasses import replace
from hashlib import sha256
from math import sqrt
from struct import pack

import pytest

from main.content_chain.detector import (
    ContentDetectorError,
    content_detector,
)
from main.content_chain.embedder import (
    ContentEmbedderError,
    content_embedder,
)
from main.content_chain.hf_carrier import (
    HfCarrierError,
    _sparse_tail,
    _zero_padded_average_5x5,
    hf_carrier,
)
from main.content_chain.hf_detector import (
    HfDetectionObservation,
    HfDetectorError,
    hf_detector,
)
from main.shared.key_schedule import (
    derive_wrong_key_material,
    identify_root_key,
)

HF_TEST_ROOT = "ceg-wm-hf-batch-two-root-π"
HF_TEST_SHAPE = (1, 2, 8, 8)


def _carrier():
    return hf_carrier(HF_TEST_ROOT, HF_TEST_SHAPE)


def _latent_values(size: int) -> tuple[float, ...]:
    return tuple(((index % 17) - 8) / 9.0 for index in range(size))


def _float32_digest(values: tuple[float, ...]) -> str:
    return sha256(b"".join(pack(">f", value) for value in values)).hexdigest()


@pytest.mark.unit
def test_hf_sparse_support() -> None:
    carrier = hf_carrier(HF_TEST_ROOT, HF_TEST_SHAPE)
    assert len(carrier.support_indices) == 26
    assert carrier.support_indices == (
        3,
        5,
        15,
        19,
        20,
        25,
        29,
        42,
        47,
        53,
        59,
        60,
        66,
        71,
        83,
        88,
        89,
        101,
        102,
        107,
        111,
        112,
        116,
        121,
        126,
        127,
    )
    support = set(carrier.support_indices)
    assert all(
        value != 0.0 if index in support else value == 0.0
        for index, value in enumerate(carrier.template)
    )


@pytest.mark.unit
def test_hf_template_normalization_order_and_unit_l2() -> None:
    carrier = hf_carrier(HF_TEST_ROOT, (1, 1, 4, 4))
    template_l2 = sqrt(sum(value * value for value in carrier.template))
    assert template_l2 == pytest.approx(1.0, abs=2e-6)
    assert carrier.direction == carrier.template
    assert sum(carrier.template) / len(carrier.template) != pytest.approx(
        0.0,
        abs=1e-5,
    )
    assert sum(value != 0.0 for value in carrier.template) == 4


@pytest.mark.unit
def test_hf_direct_score_time_centering() -> None:
    carrier = _carrier()
    observation = HfDetectionObservation.from_public_image_encoding(
        tuple(value + 2.75 for value in carrier.template),
        carrier.shape,
    )
    shifted_observation = HfDetectionObservation.from_public_image_encoding(
        tuple(value - 6.5 for value in carrier.template),
        carrier.shape,
    )
    score = hf_detector(observation, HF_TEST_ROOT).hf_score
    shifted_score = hf_detector(shifted_observation, HF_TEST_ROOT).hf_score
    assert score > 0.999
    assert shifted_score == pytest.approx(score, abs=2e-6)


@pytest.mark.unit
def test_hf_carrier_zero_padding_and_sparse_tie_order() -> None:
    pooled = _zero_padded_average_5x5((25.0,), (1, 1, 1, 1))
    assert pooled == (1.0,)
    sparse, support = _sparse_tail((2.0, -2.0, 1.0, -1.0, 0.0, 0.0))
    assert support == (0, 1)
    assert sparse == (2.0, -2.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.unit
def test_hf_carrier_consumes_mask_before_unit_direction() -> None:
    full = _carrier()
    active = set(full.support_indices[:3])
    mask = tuple(1.0 if index in active else 0.0 for index in range(len(full.template)))
    masked = hf_carrier(HF_TEST_ROOT, HF_TEST_SHAPE, mask_hf=mask)
    assert masked.template == full.template
    assert masked.mask_digest != full.mask_digest
    assert masked.direction_digest != full.direction_digest
    assert sqrt(sum(value * value for value in masked.direction)) == pytest.approx(
        1.0,
        abs=2e-6,
    )
    assert all(
        value == 0.0 for index, value in enumerate(masked.direction)
        if index not in active
    )
    with pytest.raises(HfCarrierError, match="zero"):
        hf_carrier(
            HF_TEST_ROOT,
            HF_TEST_SHAPE,
            mask_hf=(0.0,) * len(full.template),
        )


@pytest.mark.unit
def test_hf_only_content_embedder_owns_common_total_budget() -> None:
    full = _carrier()
    active = set(full.support_indices[:5])
    mask = tuple(1.0 if index in active else 0.0 for index in range(len(full.template)))
    masked = hf_carrier(HF_TEST_ROOT, HF_TEST_SHAPE, mask_hf=mask)
    latent = _latent_values(len(full.direction))

    full_embedding = content_embedder(latent, full)
    masked_embedding = content_embedder(latent, masked)
    assert full_embedding.mode == masked_embedding.mode == "hf_only"
    assert full_embedding.target_relative_l2 == pytest.approx(0.012)
    assert masked_embedding.target_total_norm == full_embedding.target_total_norm
    assert sqrt(
        sum(value * value for value in full_embedding.delta_content)
    ) == pytest.approx(full_embedding.target_total_norm, rel=2e-6)
    assert sqrt(
        sum(value * value for value in masked_embedding.delta_content)
    ) == pytest.approx(masked_embedding.target_total_norm, rel=2e-6)
    assert not hasattr(full_embedding, "realized_relative_l2")
    assert not hasattr(full_embedding, "delta_content_actual")


@pytest.mark.unit
def test_hf_only_embedder_fails_on_zero_latent_and_zero_direction() -> None:
    carrier = _carrier()
    with pytest.raises(ContentEmbedderError, match="latent"):
        content_embedder((0.0,) * len(carrier.direction), carrier)

    zero_direction = (0.0,) * len(carrier.direction)
    forged = replace(
        carrier,
        direction=zero_direction,
        direction_digest=_float32_digest(zero_direction),
    )
    with pytest.raises(ContentEmbedderError, match="direction"):
        content_embedder(_latent_values(len(carrier.direction)), forged)


@pytest.mark.unit
def test_hf_detector_blind_reconstruction_and_wrong_key() -> None:
    carrier = _carrier()
    observation = HfDetectionObservation.from_public_image_encoding(
        tuple(value + 1.25 for value in carrier.template),
        carrier.shape,
    )
    registered_identity = identify_root_key(HF_TEST_ROOT)
    wrong_key = derive_wrong_key_material(
        registered_identity.root_key_public_digest,
        0,
    )

    registered = hf_detector(observation, HF_TEST_ROOT)
    wrong = hf_detector(observation, wrong_key)
    assert registered.hf_score > 0.999
    assert wrong.hf_score < 0.5
    assert registered.key_role == "registered"
    assert registered.wrong_key_index is None
    assert wrong.key_role == "wrong"
    assert wrong.wrong_key_index == 0
    assert registered.template_digest != wrong.template_digest
    assert registered.root_key_public_digest == wrong.root_key_public_digest
    assert not hasattr(observation, "reference_image")
    assert not hasattr(observation, "embed_record")
    assert not hasattr(observation, "callback_latent")


@pytest.mark.unit
def test_hf_detector_rejects_non_public_or_zero_centered_observation() -> None:
    with pytest.raises(HfDetectorError, match="public-image"):
        hf_detector((1.0, 2.0), HF_TEST_ROOT)  # type: ignore[arg-type]
    constant = HfDetectionObservation.from_public_image_encoding(
        (3.0,) * 16,
        (1, 1, 4, 4),
    )
    with pytest.raises(HfDetectorError, match="zero centered"):
        hf_detector(constant, HF_TEST_ROOT)


@pytest.mark.unit
def test_hf_only_content_detector_preserves_branch_result_and_identity() -> None:
    carrier = _carrier()
    observation = HfDetectionObservation.from_public_image_encoding(
        carrier.template,
        carrier.shape,
    )
    branch = hf_detector(observation, HF_TEST_ROOT)
    result = content_detector(branch)
    assert result.content_score == branch.hf_score
    assert result.hf_result is branch
    assert result.detector_identity != branch.detector_identity
    assert result.formal_mode == "hf_only"
    assert result.lf_score is None
    assert result.combined_score is None
    assert result.diagnostic_combination is None
    with pytest.raises(ContentDetectorError):
        content_detector(0.5)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("shape", "mask"),
    [
        ((2, 1, 4, 4), None),
        ((1, 1, 0, 4), None),
        ((1, 1, 4, 4), (1.0,) * 15),
        ((1, 1, 4, 4), (float("nan"),) * 16),
        ((1, 1, 4, 4), (1.5,) * 16),
    ],
)
def test_hf_carrier_fails_closed_on_invalid_shape_or_mask(
    shape: tuple[int, ...],
    mask: tuple[float, ...] | None,
) -> None:
    with pytest.raises(HfCarrierError):
        hf_carrier(HF_TEST_ROOT, shape, mask_hf=mask)
