from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cegwm.method import content_iss_v8 as v8
from cegwm.method import content_adaptive_v2 as base_v2
from cegwm.method.content_adaptive_v2 import ContentAllocation
from cegwm.method.lf import FrozenLFPublicAssets
from cegwm.shared.numerics import BudgetMeasurement


def _fit() -> v8.ISSFit:
    return v8.fit_iss_gain_target(
        v8.ISSDevelopmentMeasurement(
            -0.2 + index / 1000,
            0.1 + index / 1000,
            0.2 + index / 1000,
        )
        for index in range(32)
    )


@pytest.mark.unit
def test_fit_is_exact_median_gain_rank28_margin_and_fails_closed() -> None:
    fit = _fit()
    assert fit.gain_g == pytest.approx(0.3, abs=1e-16)
    assert fit.competition_rank_28 == pytest.approx(0.227, abs=1e-16)
    assert fit.target_m == fit.competition_rank_28 + math.ldexp(1.0, -12)
    with pytest.raises(ValueError, match="exactly 32"):
        v8.fit_iss_gain_target(())
    with pytest.raises(ValueError, match="positive"):
        v8.fit_iss_gain_target(
            [v8.ISSDevelopmentMeasurement(0.1, 0.1, 0.1)] * 32
        )


@pytest.mark.unit
def test_h_is_only_the_ordinary_lf_score_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, object, object]] = []
    assets = object.__new__(FrozenLFPublicAssets)
    monkeypatch.setattr(
        v8,
        "score_lf_image",
        lambda image, key, received: calls.append((image, key, received)) or 0.25,
    )
    assert v8.content_v8_h("ordinary-rgb", b"k" * 32, assets) == 0.25
    assert calls == [("ordinary-rgb", b"k" * 32, assets)]


@pytest.mark.unit
def test_beta_scales_completed_lf_delta_only_before_shared_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocation = ContentAllocation(
        (1.0,) * 16, (1.0,) * 16, 0.4, 0.6, (0.0,) * 6
    )
    latent = torch.ones((1, 2, 4, 4), dtype=torch.float32)
    lf = torch.full_like(latent, 2.0, dtype=torch.float64)
    hf = torch.full_like(latent, 3.0, dtype=torch.float64)
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    monkeypatch.setattr(
        v8, "_content_v8_branch_deltas",
        lambda *args: (lf.clone(), hf.clone()),
    )

    def project(base: torch.Tensor, received_lf: torch.Tensor, received_hf: torch.Tensor):
        captured.append((received_lf.clone(), received_hf.clone()))
        return (
            base + 0.01,
            BudgetMeasurement(str(base.dtype), 1.0, 0.01, 0.01),
            BudgetMeasurement(str(base.dtype), 1.0, 0.006, 0.006),
            BudgetMeasurement(str(base.dtype), 1.0, 0.004, 0.004),
        )

    monkeypatch.setattr(v8, "_project_content_v8_deltas", project)
    _, measurement = v8.embed_content_v8(
        latent, b"k" * 32, object(), object(), allocation, 1.5
    )
    assert torch.equal(captured[0][0], lf * 1.5)
    assert torch.equal(captured[0][1], hf)
    assert measurement.probe_evaluation_count == 64


@pytest.mark.unit
def test_beta_one_is_exactly_the_v2_spatial_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tuple(0.5 + index / 15.0 for index in range(16))
    allocation = ContentAllocation(weights, tuple(reversed(weights)), 0.4, 0.6, (0.0,) * 6)
    latents = torch.linspace(
        0.1, 3.2, 32, dtype=torch.float32
    ).reshape(1, 2, 4, 4)
    lf = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32).reshape_as(latents)
    hf = torch.linspace(1.0, -0.5, 32, dtype=torch.float32).reshape_as(latents)
    assets = SimpleNamespace(injection_step_index=18)
    for module in (v8, base_v2):
        monkeypatch.setattr(module, "reconstruct_lf_carrier", lambda *args, **kwargs: lf.clone())
        monkeypatch.setattr(module, "reconstruct_hf_carrier", lambda *args, **kwargs: hf.clone())
    expected, expected_measurement = base_v2.embed_content_adaptive(
        latents, b"k" * 32, assets, assets, allocation
    )
    received, received_measurement = v8.embed_content_v8(
        latents, b"k" * 32, assets, assets, allocation, 1.0
    )
    assert torch.equal(received, expected)
    assert received_measurement == expected_measurement


@pytest.mark.unit
def test_runtime_asset_is_canonical_binary64_protocol_bound_and_sidecar_bound(
    tmp_path: Path,
) -> None:
    key = v8.derive_development_key(b"content-v8-root-key")
    protocol_digest = "a" * 64
    asset = v8.build_iss_asset("1" * 40, protocol_digest, key, _fit())
    assert tuple(asset.payload) == tuple(sorted(asset.payload))
    path = tmp_path / f"{v8.ISS_ASSET_ROLE_ID}.json"
    sidecar = path.with_name(f"{path.name}.sha256")
    path.write_bytes(asset.json_bytes)
    digest = hashlib.sha256(asset.json_bytes).hexdigest()
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    loaded = v8.load_iss_asset(
        path, sidecar, expected_protocol_digest=protocol_digest
    )
    assert loaded == asset
    assert v8.iss_beta(loaded.target_m - 0.5 * loaded.gain_g, loaded) == 1.0
    assert v8.iss_beta(loaded.target_m - 1.5 * loaded.gain_g, loaded) == pytest.approx(1.5)
    assert v8.iss_beta(loaded.target_m - 3.0 * loaded.gain_g, loaded) == 2.0
    with pytest.raises(ValueError, match="protocol binding"):
        v8.load_iss_asset(path, sidecar, expected_protocol_digest="b" * 64)
