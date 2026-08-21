from __future__ import annotations

import numpy as np
import pytest

from cegwm.shared.bands import make_frequency_band_masks
from cegwm.shared.numerics import (
    assert_relative_l2_budget,
    masked_normalized_correlation,
    project_sum_to_relative_l2_budget,
    relative_l2_measurement,
)


@pytest.mark.unit
def test_lf_hf_masks_match_rfft_shape_and_never_overlap() -> None:
    masks = make_frequency_band_masks(
        64,
        80,
        lf_min_radius=0.04,
        lf_max_radius=0.24,
        hf_min_radius=0.58,
    )

    assert masks.lf.shape == (64, 41)
    assert masks.hf.shape == (64, 41)
    assert np.any(masks.lf)
    assert np.any(masks.hf)
    assert not np.any(masks.lf & masks.hf)
    assert not masks.lf[0, 0]
    assert not masks.hf[0, 0]


@pytest.mark.unit
def test_band_builder_rejects_touching_or_overlapping_roles() -> None:
    with pytest.raises(ValueError, match="lf_max < hf_min"):
        make_frequency_band_masks(
            64,
            64,
            lf_min_radius=0.04,
            lf_max_radius=0.58,
            hf_min_radius=0.58,
        )


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_total_budget_is_enforced_after_actual_dtype_cast(dtype: type[np.floating]) -> None:
    base = np.linspace(0.2, 1.2, 4096, dtype=dtype).reshape(64, 64)
    lf_delta = np.full(base.shape, 0.08, dtype=np.float64)
    hf_delta = np.tile(np.array([-0.06, 0.06], dtype=np.float64), 2048).reshape(base.shape)

    candidate, measurement = project_sum_to_relative_l2_budget(
        base,
        [lf_delta, hf_delta],
        0.012,
    )

    assert candidate.dtype == base.dtype
    assert 0.0 < measurement.relative_l2 <= 0.012
    assert measurement == assert_relative_l2_budget(base, candidate, 0.012)
    assert measurement == relative_l2_measurement(base, candidate)


@pytest.mark.unit
def test_budget_rejects_separate_dtype_or_shape_accounting() -> None:
    base = np.ones((8, 8), dtype=np.float16)
    with pytest.raises(ValueError, match="same actual dtype"):
        relative_l2_measurement(base, base.astype(np.float32))
    with pytest.raises(ValueError, match="match the base shape"):
        project_sum_to_relative_l2_budget(base, [np.ones((4, 4))], 0.012)


@pytest.mark.unit
def test_blind_content_score_uses_only_observation_carrier_and_mask() -> None:
    carrier = np.array([-1.0, 1.0, -1.0, 1.0])
    mask = np.array([True, True, True, True])

    assert masked_normalized_correlation(carrier * 3.0, carrier, mask) == pytest.approx(1.0)
    assert masked_normalized_correlation(-carrier, carrier, mask) == pytest.approx(-1.0)
