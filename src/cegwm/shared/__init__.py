"""Small numerical building blocks shared by the Stage-A carriers."""

from cegwm.shared.bands import FrequencyBandMasks, make_frequency_band_masks
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from cegwm.shared.numerics import (
    BudgetMeasurement,
    assert_relative_l2_budget,
    masked_normalized_correlation,
    project_sum_to_relative_l2_budget,
    relative_l2_measurement,
)
from cegwm.shared.prg import prg_bytes, prg_normal, prg_rademacher

__all__ = [
    "BudgetMeasurement",
    "FrequencyBandMasks",
    "assert_relative_l2_budget",
    "make_frequency_band_masks",
    "masked_normalized_correlation",
    "normalize_detection_key",
    "prg_bytes",
    "prg_normal",
    "prg_rademacher",
    "project_sum_to_relative_l2_budget",
    "public_key_digest",
    "relative_l2_measurement",
]
