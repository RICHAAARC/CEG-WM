from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest

from cegwm.method import content_weighted_joint_v9 as v9


def _pairs() -> tuple[v9.LFHFScorePair, ...]:
    return tuple(
        v9.LFHFScorePair(
            -0.6 + index / 1000.0,
            -0.45 + ((index * 37) % 997) / 1200.0,
        )
        for index in range(1056)
    )


def _asset() -> v9.WeightedJointAsset:
    return v9.build_calibration_asset(
        producer_exact="1" * 40,
        protocol_digest="2" * 64,
        public_key_digest="3" * 64,
        fit=v9.fit_weighted_joint_calibration(_pairs()),
    )


@pytest.mark.unit
def test_binary64_fit_uses_exact_order_count_ddof1_and_paired_rho() -> None:
    pairs = _pairs()
    fit = v9.fit_weighted_joint_calibration(pairs)
    lf = [pair.lf for pair in pairs]
    hf = [pair.hf for pair in pairs]
    expected_mu_lf = math.fsum(lf) / len(lf)
    expected_mu_hf = math.fsum(hf) / len(hf)
    centered_lf = [value - expected_mu_lf for value in lf]
    centered_hf = [value - expected_mu_hf for value in hf]
    m2_lf = math.fsum(value * value for value in centered_lf)
    m2_hf = math.fsum(value * value for value in centered_hf)
    expected_rho = math.fsum(
        left * right for left, right in zip(centered_lf, centered_hf, strict=True)
    ) / math.sqrt(m2_lf * m2_hf)
    assert fit.mu_lf == expected_mu_lf
    assert fit.mu_hf == expected_mu_hf
    assert fit.sigma_lf == math.sqrt(m2_lf / 1055)
    assert fit.sigma_hf == math.sqrt(m2_hf / 1055)
    assert fit.rho == expected_rho
    assert fit.sigma_lf == statistics.stdev(lf)
    with pytest.raises(ValueError, match="exactly 1056"):
        v9.fit_weighted_joint_calibration(pairs[:-1])
    degenerate = (v9.LFHFScorePair(0.0, 0.0),) * 1056
    with pytest.raises(ValueError, match="variances"):
        v9.fit_weighted_joint_calibration(degenerate)
    with pytest.raises(TypeError):
        v9.fit_weighted_joint_calibration((v9.LFHFScorePair(True, 0.0),) * 1056)


@pytest.mark.unit
def test_asset_is_stable_create_only_payload_with_exact_five_f64_values(tmp_path: Path) -> None:
    asset = _asset()
    assert tuple(asset.payload) == tuple(sorted(asset.payload))
    numeric = {
        "mu_lf_be_hex", "sigma_lf_be_hex", "mu_hf_be_hex", "sigma_hf_be_hex",
        "rho_be_hex",
    }
    assert len(numeric) == 5 and numeric <= set(asset.payload)
    assert asset.json_bytes == json.dumps(
        asset.payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    forbidden = {"prompt", "token", "image", "latent", "raw_scores", "private_state"}
    assert forbidden.isdisjoint(asset.payload)
    path = tmp_path / "asset.json"
    sidecar = tmp_path / "asset.json.sha256"
    path.write_bytes(asset.json_bytes)
    digest = hashlib.sha256(asset.json_bytes).hexdigest()
    sidecar.write_bytes(f"{digest}  {path.name}\n".encode("ascii"))
    assert v9.load_calibration_asset(path, sidecar) == asset
    sidecar.write_bytes(f"{'0' * 64}  {path.name}\n".encode("ascii"))
    with pytest.raises(ValueError, match="sidecar binding"):
        v9.load_calibration_asset(path, sidecar)


@pytest.mark.unit
def test_weights_apply_only_to_z_and_branch_diagnostics_cannot_veto_weighted_gates() -> None:
    asset = _asset()
    fit = asset.fit
    lf, hf = 0.12, -0.03
    denominator = math.sqrt(
        0.25**2 + 0.75**2 + 2 * 0.25 * 0.75 * fit.rho
    )
    expected = (
        0.25 * ((lf - fit.mu_lf) / fit.sigma_lf)
        + 0.75 * ((hf - fit.mu_hf) / fit.sigma_hf)
    ) / denominator
    assert v9.weighted_joint_score(lf, hf, asset) == expected
    registered = v9.LFHFScorePair(-0.1, 0.9)
    wrong = tuple(v9.LFHFScorePair(0.8, -0.8) for _ in range(16))
    null = v9.LFHFScorePair(0.7, -0.7)
    evidence = v9.weighted_gate_evidence(registered, wrong, null, asset)
    assert evidence.weighted_gate_a is True
    assert evidence.weighted_gate_b is True
    assert evidence.lf_gate_a_diagnostic is False
    assert evidence.lf_gate_b_diagnostic is False
    assert evidence.hf_gate_a_diagnostic is True
    assert evidence.hf_gate_b_diagnostic is True
    with pytest.raises(TypeError):
        v9.weighted_joint_score(True, 0.0, asset)
