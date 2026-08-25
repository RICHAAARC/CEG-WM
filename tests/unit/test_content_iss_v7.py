from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct

import pytest
import torch

from cegwm.method import content_adaptive_v3 as v3
from cegwm.method import content_iss_v7 as v7
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
)


def _lf_assets() -> FrozenLFPublicAssets:
    assets = object.__new__(FrozenLFPublicAssets)
    object.__setattr__(assets, "candidate_id", LF_BALANCED_BLOCKS_CARRIER_METHOD_ID)
    object.__setattr__(assets, "detector_statistic_id", LF_BLOCKNORM_DETECTOR_STATISTIC_ID)
    object.__setattr__(
        assets, "evaluated_candidate_id", LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID
    )
    return assets


def _fit() -> v7.ISSFit:
    return v7.fit_iss_gain_target(tuple(
        v7.ISSDevelopmentMeasurement(
            -0.2 + index / 1000,
            0.1 + index / 1000,
            0.2 + index / 1000,
        )
        for index in range(32)
    ))


@pytest.mark.unit
def test_v7_lf_score_uses_the_single_frozen_ordinary_scorer_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        v7,
        "score_lf_image",
        lambda image, key, assets: calls.append((image, key, assets)) or 0.25,
    )
    assets = _lf_assets()
    assert v7.score_content_v7_lf("ordinary-rgb", b"key", assets) == 0.25
    assert calls == [("ordinary-rgb", b"key", assets)]
    assert v7.ISS_SCORER_CALLABLE_ID == "cegwm.method.lf.score_lf_image"


@pytest.mark.unit
def test_v7_fit_uses_registered_median_rank28_and_exact_margin() -> None:
    fit = _fit()
    assert fit.gain_g == pytest.approx(0.3, abs=1e-16)
    assert fit.competition_rank_28 == pytest.approx(0.227, abs=1e-16)
    assert fit.target_m == pytest.approx(0.227 + math.ldexp(1.0, -12), abs=0.0)
    with pytest.raises(ValueError, match="gain must be finite and positive"):
        v7.fit_iss_gain_target(
            [v7.ISSDevelopmentMeasurement(0.0, 0.0, 0.0)] * 32
        )
    invalid = [v7.ISSDevelopmentMeasurement(0.2, 0.3, 0.1)] * 32
    with pytest.raises(ValueError, match="include the registered host"):
        v7.fit_iss_gain_target(invalid)


@pytest.mark.unit
def test_v7_runtime_asset_is_canonical_binary64_sidecar_bound_and_clamped(
    tmp_path: Path,
) -> None:
    key = v7.derive_development_key(b"content-v7-root-key-material")
    asset = v7.build_iss_asset("1" * 40, key, _fit())
    assert tuple(asset.payload) == tuple(sorted(asset.payload))
    assert (
        struct.unpack(">d", bytes.fromhex(asset.payload["gain_g_be_hex"]))[0]
        == asset.gain_g
    )
    path = tmp_path / v7.ISS_ASSET_FILENAME
    sidecar = tmp_path / f"{v7.ISS_ASSET_FILENAME}.sha256"
    path.write_bytes(asset.json_bytes)
    digest = hashlib.sha256(asset.json_bytes).hexdigest()
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    loaded = v7.load_iss_asset(path, sidecar)
    assert loaded == asset
    assert v7.iss_beta(loaded.target_m - 0.5 * loaded.gain_g, loaded) == 1.0
    assert v7.iss_beta(
        loaded.target_m - 1.5 * loaded.gain_g, loaded
    ) == pytest.approx(1.5)
    assert v7.iss_beta(loaded.target_m - 3.0 * loaded.gain_g, loaded) == 2.0
    sidecar.write_text(f"{'0' * 64}  {path.name}\n", encoding="ascii")
    with pytest.raises(ValueError, match="sidecar binding"):
        v7.load_iss_asset(path, sidecar)


@pytest.mark.unit
def test_v7_beta_scales_only_lf_before_the_shared_projector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = torch.linspace(-1.0, 1.0, 64, dtype=torch.float64).reshape(1, 1, 8, 8)
    lf_delta = torch.ones_like(base)
    hf_delta = torch.where(
        torch.arange(base.numel()).reshape(base.shape) % 2 == 0,
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(-1.0, dtype=torch.float64),
    )
    calls: list[tuple[object, ...]] = []

    def branch_deltas(*args: object) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(args)
        return lf_delta, hf_delta

    monkeypatch.setattr(v7, "_content_v3_branch_deltas", branch_deltas)
    allocation = v3.ContentAllocation((1.0,) * 16, (1.0,) * 16, 0.4, 0.6, (0.1,) * 6)
    one, one_measurement = v7.embed_content_v7(
        base, b"key", object(), object(), allocation, 1.0
    )
    two, two_measurement = v7.embed_content_v7(
        base, b"key", object(), object(), allocation, 2.0
    )
    assert len(calls) == 2
    assert one_measurement.combined_budget.relative_l2 <= 0.012
    assert two_measurement.combined_budget.relative_l2 <= 0.012
    assert (
        two_measurement.lf_effective_relative_l2
        / two_measurement.hf_effective_relative_l2
    ) == pytest.approx(
        2.0
        * one_measurement.lf_effective_relative_l2
        / one_measurement.hf_effective_relative_l2,
        rel=1e-12,
    )
    assert not torch.equal(one, two)
