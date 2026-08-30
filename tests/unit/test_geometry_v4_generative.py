from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cegwm.method.geometry_v4_generative import measure_final_rgb, rgb_only_anchor_score, write_final_latent_anchor
from cegwm.protocol.geometry_v4_generative import CALLBACK_STEP_INDEX, LUMA_PEAK_CAP, LUMA_RMS_CAP, load_g0_g1_contract

ROOT = Path(__file__).resolve().parents[2]
KEY, WRONG = "0123456789abcdef", "fedcba9876543210"


@pytest.mark.unit
def test_contract_freezes_sole_placement_budget_and_rosters() -> None:
    contract = load_g0_g1_contract(ROOT)
    assert contract["identity"]["callback_step_index_zero_based"] == CALLBACK_STEP_INDEX == 19
    assert tuple(contract["g0"]["seeds"]) == (5101, 5102, 5103, 5104)
    assert tuple(contract["residual_budget"]["global_local_energy_shares"]) == (.4, .6)
    assert LUMA_RMS_CAP == 2 / 255 and LUMA_PEAK_CAP == 8 / 255


@pytest.mark.unit
def test_writer_is_keyed_deterministic_and_rejects_non_latent_input() -> None:
    latents = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    first = write_final_latent_anchor(latents, KEY)
    assert torch.equal(first, write_final_latent_anchor(latents, KEY))
    assert not torch.equal(first, write_final_latent_anchor(latents, WRONG))
    with pytest.raises(ValueError, match="NCHW"):
        write_final_latent_anchor(torch.zeros((4, 8, 8)), KEY)


@pytest.mark.unit
def test_rgb_observability_is_rgb_key_only_and_fail_closed_on_equal_keys() -> None:
    clean = np.full((64, 64, 3), .5, dtype=np.float64)
    marked = clean.copy()
    yy, xx = np.mgrid[:64, :64]
    marked += (0.0001 * np.cos(2 * np.pi * 8 * xx / 64))[..., None]
    observation = measure_final_rgb(clean, marked, KEY, WRONG, lambda rgb, key: float(rgb.mean()))
    assert observation.luma_rms <= LUMA_RMS_CAP and observation.luma_peak <= LUMA_PEAK_CAP
    assert np.isfinite(rgb_only_anchor_score(marked, KEY))
    with pytest.raises(ValueError, match="must differ"):
        measure_final_rgb(clean, marked, KEY, KEY, lambda rgb, key: 0.0)
