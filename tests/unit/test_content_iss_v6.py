from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from cegwm.method import content_adaptive_v3 as v3
from cegwm.method import content_iss_v6 as v6
from cegwm.method.content_whitening_v4 import FrozenContentV4LFPublicAssets, load_frozen_content_v4_whitening_asset
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
)

_ROOT = Path(__file__).resolve().parents[2]


class _Processor:
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pixels = np.asarray(image.resize((64, 64)), dtype=np.float32) / 255.0
        return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).contiguous()


class _VAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(scaling_factor=0.5, shift_factor=0.1)

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        mode = torch.cat([pixels] * 5 + [pixels.mean(dim=1, keepdim=True)], dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: mode))


def _assets() -> FrozenContentV4LFPublicAssets:
    carrier = FrozenLFPublicAssets(
        _VAE(),
        _Processor(),
        "stabilityai/stable-diffusion-3.5-medium:image_processor",
        LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    return FrozenContentV4LFPublicAssets(
        carrier,
        load_frozen_content_v4_whitening_asset(_ROOT),
    )


def _fit() -> v6.ISSFit:
    measurements = tuple(
        v6.ISSDevelopmentMeasurement(
            host_score=-0.2 + index / 1000,
            beta_one_score=0.1 + index / 1000,
            competition_score=0.2 + index / 1000,
        )
        for index in range(32)
    )
    return v6.fit_iss_gain_target(measurements)


@pytest.mark.unit
def test_phi_u_h_are_normalized_float64_and_equal_current_v4_score() -> None:
    assert v6.CONTENT_V6_METHOD_ID == "content_v6_detector_domain_iss_lf_adaptive_hf_v1"
    assert v6.CONTENT_V6_EVALUATED_CANDIDATE_ID == (
        "content_v6_detector_domain_iss_lf_adaptive_hf_semantic_gate_v1"
    )
    y, x = np.mgrid[:64, :64]
    pixels = np.stack(((3*x+5*y)%256, (7*x+11*y+13)%256, (x*x+3*y)%256), axis=-1).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    assets = _assets()
    key = b"content-v6-phi-equality-key"
    phi = v6.content_v6_phi(image, assets)
    template = v6.content_v6_u(key, assets)
    score = v6.content_v6_h(image, key, assets)
    assert phi.dtype == torch.float64 and template.dtype == torch.float64
    assert phi.is_contiguous() and template.is_contiguous()
    assert float(torch.linalg.vector_norm(phi)) == pytest.approx(1.0, abs=2e-15)
    assert float(torch.linalg.vector_norm(template)) == pytest.approx(1.0, abs=2e-15)
    assert score == pytest.approx(float(torch.dot(phi, template)), abs=0.0)


@pytest.mark.unit
def test_fit_formula_uses_median_gain_rank28_and_exact_margin() -> None:
    fit = _fit()
    expected_gain = ((0.1 + 15/1000) - (-0.2 + 15/1000) + (0.1 + 16/1000) - (-0.2 + 16/1000)) / 2
    expected_q = 0.2 + 27/1000
    assert fit.gain_g == pytest.approx(expected_gain, abs=1e-16)
    assert fit.competition_rank_28 == pytest.approx(expected_q, abs=1e-16)
    assert fit.target_m == pytest.approx(expected_q + math.ldexp(1.0, -12), abs=0.0)
    bad = [v6.ISSDevelopmentMeasurement(0.0, 0.0, 0.0)] * 32
    with pytest.raises(ValueError, match="gain must be finite and positive"):
        v6.fit_iss_gain_target(bad)


@pytest.mark.unit
def test_asset_binary64_sidecar_and_beta_boundaries_fail_closed(tmp_path: Path) -> None:
    key = v6.derive_development_key(b"content-v6-root-key-material")
    asset = v6.build_iss_asset("1" * 40, key, _fit())
    assert tuple(asset.payload) == tuple(sorted(asset.payload))
    assert struct.unpack(">d", bytes.fromhex(asset.payload["gain_g_be_hex"]))[0] == asset.gain_g
    path = tmp_path / f"{v6.ISS_ASSET_ROLE_ID}.json"
    sidecar = tmp_path / f"{v6.ISS_ASSET_ROLE_ID}.json.sha256"
    path.write_bytes(asset.json_bytes)
    digest = hashlib.sha256(asset.json_bytes).hexdigest()
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    loaded = v6.load_iss_asset(path, sidecar)
    assert loaded == asset
    low_host = loaded.target_m - 0.5 * loaded.gain_g
    middle_host = loaded.target_m - 1.5 * loaded.gain_g
    high_host = loaded.target_m - 3.0 * loaded.gain_g
    assert v6.iss_beta(low_host, loaded) == 1.0
    assert v6.iss_beta(middle_host, loaded) == pytest.approx(1.5)
    assert v6.iss_beta(high_host, loaded) == 2.0
    for invalid in (True, math.nan, math.inf):
        with pytest.raises((TypeError, ValueError)):
            v6.iss_beta(invalid, loaded)
    sidecar.write_text(f"{'0'*64}  {path.name}\n", encoding="ascii")
    with pytest.raises(ValueError, match="sidecar binding"):
        v6.load_iss_asset(path, sidecar)
    with pytest.raises(ValueError, match="rank-28 rule"):
        v6.build_iss_asset("1" * 40, key, v6.ISSFit(0.2, 0.4, 0.1))


@pytest.mark.unit
def test_frozen_v6_gain_target_asset_is_the_accepted_exact_pair() -> None:
    asset_path = _ROOT / v6.ISS_ASSET_REPO_PATH
    sidecar_path = _ROOT / v6.ISS_ASSET_SIDECAR_REPO_PATH
    raw = asset_path.read_bytes()
    sidecar = sidecar_path.read_bytes()
    assert len(raw) == 1152
    assert len(sidecar) == 101
    assert hashlib.sha256(raw).hexdigest() == v6.ISS_ASSET_SHA256
    assert hashlib.sha256(sidecar).hexdigest() == v6.ISS_ASSET_SIDECAR_SHA256
    assert sidecar == f"{v6.ISS_ASSET_SHA256}  {asset_path.name}\n".encode("ascii")
    asset = v6.load_frozen_content_v6_iss_asset(_ROOT)
    assert asset.payload["producer_exact"] == "70d4147ceb9832acf7511b2e68edf0c47e453229"
    assert asset.gain_g == pytest.approx(0.01216927948727384, abs=0.0)
    assert asset.target_m == pytest.approx(0.013417310725870352, abs=0.0)


@pytest.mark.unit
def test_v6_beta_scales_only_lf_before_the_unchanged_common_projector(
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

    monkeypatch.setattr(v6, "_content_v3_branch_deltas", branch_deltas)
    allocation = v3.ContentAllocation(
        (1.0,) * 16, (1.0,) * 16, 0.4, 0.6, (0.1,) * 6
    )
    one, one_measurement = v6.embed_content_v6(
        base, b"key", object(), object(), allocation, 1.0
    )
    two, two_measurement = v6.embed_content_v6(
        base, b"key", object(), object(), allocation, 2.0
    )
    assert len(calls) == 2
    assert one_measurement.combined_budget.relative_l2 <= 0.012
    assert two_measurement.combined_budget.relative_l2 <= 0.012
    assert one_measurement.lf_branch_share == two_measurement.lf_branch_share == 0.4
    assert one_measurement.hf_branch_share == two_measurement.hf_branch_share == 0.6
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
    for invalid in (0.999, 2.001, math.nan, True):
        with pytest.raises((TypeError, ValueError)):
            v6.embed_content_v6(base, b"key", object(), object(), allocation, invalid)
