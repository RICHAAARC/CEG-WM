"""content paired-null calibration and fixed weighted joint statistic."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cegwm.method.content_iss import (
    ISS_ASSET_SHA256 as ISS_ASSET_SHA256,
    ISS_ASSET_SIDECAR_SHA256 as ISS_ASSET_SIDECAR_FILE_SHA256,
)
from cegwm.method.content_whitening import (
    ASSET_SHA256 as V4_WHITENING_ASSET_SHA256,
    ASSET_SIDECAR_SHA256 as V4_WHITENING_ASSET_SIDECAR_FILE_SHA256,
    CONTENT_WHITENING_LF_SCORER_ID,
)
from cegwm.protocol.content_calibration import (
    CONTENT_CALIBRATION_BASE_EXACT,
    CONTENT_CALIBRATION_CALIBRATION_ASSET_ROLE_ID,
    CONTENT_CALIBRATION_CALIBRATION_ASSET_SCHEMA_ID,
    CONTENT_CALIBRATION_CALIBRATION_COUNT,
    CONTENT_CALIBRATION_CALIBRATION_MANIFEST_SHA256,
    CONTENT_CALIBRATION_CALIBRATION_PROMPT_LIST_SHA256,
    CONTENT_CALIBRATION_CALIBRATION_PROTOCOL_ID,
    CONTENT_CALIBRATION_EVALUATED_CANDIDATE_ID,
    CONTENT_CALIBRATION_METHOD_ID,
    CONTENT_CALIBRATION_PAIRED_NULL_SCORE_COUNT,
)

LF_WEIGHT = 0.25
HF_WEIGHT = 0.75
HF_SCORER_ID = "frozen_hf_final_rgb_public_vae_global_normalized_correlation"
CALIBRATION_PAIR_ORDER = (
    "per_unit_candidate_wrong_00_to_15_then_primary_null_registered_then_"
    "primary_null_wrong_00_to_15_each_lf_then_hf"
)
CALIBRATION_STATISTIC_ID = "binary64_fsum_mean_ddof1_std_pearson_rho_v1"
JOINT_OPERATOR_ID = "content_v9_correlation_adjusted_fixed_weight_stouffer_v1"
_HEX16 = re.compile(r"[0-9a-f]{16}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class LFHFScorePair:
    lf: float
    hf: float


@dataclass(frozen=True, slots=True)
class CalibrationFit:
    mu_lf: float
    sigma_lf: float
    mu_hf: float
    sigma_hf: float
    rho: float


@dataclass(frozen=True, slots=True)
class WeightedJointAsset:
    payload: Mapping[str, Any]
    json_bytes: bytes

    @property
    def fit(self) -> CalibrationFit:
        return CalibrationFit(
            _decode_f64(self.payload["mu_lf_be_hex"], "mu_lf"),
            _decode_f64(self.payload["sigma_lf_be_hex"], "sigma_lf", positive=True),
            _decode_f64(self.payload["mu_hf_be_hex"], "mu_hf"),
            _decode_f64(self.payload["sigma_hf_be_hex"], "sigma_hf", positive=True),
            _decode_f64(self.payload["rho_be_hex"], "rho"),
        )


@dataclass(frozen=True, slots=True)
class WeightedGateEvidence:
    weighted_gate_a: bool
    weighted_gate_b: bool
    lf_gate_a_diagnostic: bool
    lf_gate_b_diagnostic: bool
    hf_gate_a_diagnostic: bool
    hf_gate_b_diagnostic: bool


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _finite_score(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar) or not -1.0 <= scalar <= 1.0:
        raise ValueError(f"{name} must be finite in [-1, 1]")
    return scalar


def _pair(value: Any, name: str) -> LFHFScorePair:
    if not isinstance(value, LFHFScorePair):
        raise TypeError(f"{name} must be an LFHFScorePair")
    return LFHFScorePair(
        _finite_score(value.lf, f"{name}.lf"),
        _finite_score(value.hf, f"{name}.hf"),
    )


def fit_weighted_joint_calibration(pairs: Iterable[LFHFScorePair]) -> CalibrationFit:
    """Fit binary64 mean, ddof=1 standard deviations, and paired rho."""

    received = tuple(_pair(value, f"pair[{index}]") for index, value in enumerate(pairs))
    if len(received) != CONTENT_CALIBRATION_PAIRED_NULL_SCORE_COUNT:
        raise ValueError("content calibration requires exactly 1056 ordered LF/HF pairs")
    count = len(received)
    lf_values = tuple(value.lf for value in received)
    hf_values = tuple(value.hf for value in received)
    mu_lf = math.fsum(lf_values) / count
    mu_hf = math.fsum(hf_values) / count
    lf_centered = tuple(value - mu_lf for value in lf_values)
    hf_centered = tuple(value - mu_hf for value in hf_values)
    m2_lf = math.fsum(value * value for value in lf_centered)
    m2_hf = math.fsum(value * value for value in hf_centered)
    if not math.isfinite(m2_lf) or not math.isfinite(m2_hf) or m2_lf <= 0.0 or m2_hf <= 0.0:
        raise ValueError("content calibration variances must be finite and positive")
    sigma_lf = math.sqrt(m2_lf / (count - 1))
    sigma_hf = math.sqrt(m2_hf / (count - 1))
    rho = math.fsum(
        left * right for left, right in zip(lf_centered, hf_centered, strict=True)
    ) / math.sqrt(m2_lf * m2_hf)
    if not all(math.isfinite(value) for value in (mu_lf, sigma_lf, mu_hf, sigma_hf, rho)):
        raise ValueError("content calibration statistics must be finite")
    if sigma_lf <= 0.0 or sigma_hf <= 0.0 or not -1.0 <= rho <= 1.0:
        raise ValueError("content calibration sigma/rho domain differs")
    _joint_denominator(rho)
    return CalibrationFit(mu_lf, sigma_lf, mu_hf, sigma_hf, rho)


def _joint_denominator(rho: Any) -> float:
    correlation = _finite_score(rho, "rho")
    value = math.sqrt(
        LF_WEIGHT * LF_WEIGHT
        + HF_WEIGHT * HF_WEIGHT
        + 2.0 * LF_WEIGHT * HF_WEIGHT * correlation
    )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("content joint denominator must be finite and positive")
    return value


def weighted_joint_score(lf_score: Any, hf_score: Any, asset: WeightedJointAsset) -> float:
    """Apply fixed weights only after frozen per-branch standardization."""

    if not isinstance(asset, WeightedJointAsset):
        raise TypeError("content weighted score requires a frozen calibration asset")
    lf = _finite_score(lf_score, "lf_score")
    hf = _finite_score(hf_score, "hf_score")
    fit = asset.fit
    z_lf = (lf - fit.mu_lf) / fit.sigma_lf
    z_hf = (hf - fit.mu_hf) / fit.sigma_hf
    value = (LF_WEIGHT * z_lf + HF_WEIGHT * z_hf) / _joint_denominator(fit.rho)
    if not math.isfinite(value):
        raise ValueError("content weighted joint score must be finite")
    return value


def weighted_gate_evidence(
    candidate_registered: LFHFScorePair,
    candidate_wrong: Sequence[LFHFScorePair],
    primary_null_registered: LFHFScorePair,
    asset: WeightedJointAsset,
) -> WeightedGateEvidence:
    """Evaluate weighted formal gates; branch gates are diagnostics only."""

    registered = _pair(candidate_registered, "candidate_registered")
    null = _pair(primary_null_registered, "primary_null_registered")
    wrong = tuple(_pair(value, f"candidate_wrong[{index}]") for index, value in enumerate(candidate_wrong))
    if len(wrong) != 16:
        raise ValueError("content weighted Gate A requires exactly 16 wrong keys")
    registered_joint = weighted_joint_score(registered.lf, registered.hf, asset)
    wrong_joint = tuple(weighted_joint_score(value.lf, value.hf, asset) for value in wrong)
    null_joint = weighted_joint_score(null.lf, null.hf, asset)
    return WeightedGateEvidence(
        registered_joint > max(wrong_joint),
        registered_joint > null_joint,
        registered.lf > max(value.lf for value in wrong),
        registered.lf > null.lf,
        registered.hf > max(value.hf for value in wrong),
        registered.hf > null.hf,
    )


def _encode_f64(value: Any, name: str, *, positive: bool = False) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar) or (positive and scalar <= 0.0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return struct.pack(">d", scalar).hex()


def _decode_f64(value: Any, name: str, *, positive: bool = False) -> float:
    if not isinstance(value, str) or _HEX16.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase binary64 big-endian hex")
    scalar = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(scalar) or (positive and scalar <= 0.0):
        raise ValueError(f"{name} binary64 value is invalid")
    return scalar


def _validate_asset_payload(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict) or tuple(value) != tuple(sorted(value)):
        raise ValueError("content calibration asset fields must use stable order")
    required = {
        "asset_role_id", "base_exact", "calibration_manifest_sha256",
        "calibration_pair_count", "calibration_pair_order",
        "calibration_prompt_list_sha256", "calibration_protocol_digest",
        "calibration_protocol_id", "calibration_public_key_digest",
        "evaluated_candidate_id", "hf_scorer_id", "hf_weight_be_hex",
        "fit_unit_count", "iss_asset_sha256", "iss_asset_sidecar_file_sha256",
        "joint_formula", "lf_scorer_id", "lf_weight_be_hex",
        "method_id", "mu_hf_be_hex", "mu_lf_be_hex", "producer_exact",
        "rho_be_hex", "schema_version", "sigma_hf_be_hex", "sigma_lf_be_hex",
        "statistic_id", "value_dtype", "whitening_asset_sha256",
        "whitening_asset_sidecar_file_sha256",
    }
    if set(value) != required:
        raise ValueError("content calibration asset fields differ")
    expected = {
        "asset_role_id": CONTENT_CALIBRATION_CALIBRATION_ASSET_ROLE_ID,
        "base_exact": CONTENT_CALIBRATION_BASE_EXACT,
        "calibration_manifest_sha256": CONTENT_CALIBRATION_CALIBRATION_MANIFEST_SHA256,
        "calibration_pair_count": CONTENT_CALIBRATION_PAIRED_NULL_SCORE_COUNT,
        "calibration_pair_order": CALIBRATION_PAIR_ORDER,
        "calibration_prompt_list_sha256": CONTENT_CALIBRATION_CALIBRATION_PROMPT_LIST_SHA256,
        "calibration_protocol_id": CONTENT_CALIBRATION_CALIBRATION_PROTOCOL_ID,
        "evaluated_candidate_id": CONTENT_CALIBRATION_EVALUATED_CANDIDATE_ID,
        "fit_unit_count": CONTENT_CALIBRATION_CALIBRATION_COUNT,
        "hf_scorer_id": HF_SCORER_ID,
        "iss_asset_sha256": ISS_ASSET_SHA256,
        "iss_asset_sidecar_file_sha256": ISS_ASSET_SIDECAR_FILE_SHA256,
        "joint_formula": "J=(0.25*z_lf+0.75*z_hf)/sqrt(0.25^2+0.75^2+2*0.25*0.75*rho)",
        "lf_scorer_id": CONTENT_WHITENING_LF_SCORER_ID,
        "method_id": CONTENT_CALIBRATION_METHOD_ID,
        "schema_version": CONTENT_CALIBRATION_CALIBRATION_ASSET_SCHEMA_ID,
        "statistic_id": CALIBRATION_STATISTIC_ID,
        "value_dtype": "IEEE-754_binary64_big_endian_hex",
        "whitening_asset_sha256": V4_WHITENING_ASSET_SHA256,
        "whitening_asset_sidecar_file_sha256": (
            V4_WHITENING_ASSET_SIDECAR_FILE_SHA256
        ),
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise ValueError("content calibration asset frozen identity differs")
    if value.get("lf_weight_be_hex") != _encode_f64(LF_WEIGHT, "lf_weight"):
        raise ValueError("content LF weight differs")
    if value.get("hf_weight_be_hex") != _encode_f64(HF_WEIGHT, "hf_weight"):
        raise ValueError("content HF weight differs")
    for name in ("producer_exact",):
        if not isinstance(value.get(name), str) or _HEX40.fullmatch(value[name]) is None:
            raise ValueError(f"content {name} differs")
    for name in ("calibration_protocol_digest", "calibration_public_key_digest"):
        if not isinstance(value.get(name), str) or _HEX64.fullmatch(value[name]) is None:
            raise ValueError(f"content {name} differs")
    fit = CalibrationFit(
        _decode_f64(value["mu_lf_be_hex"], "mu_lf"),
        _decode_f64(value["sigma_lf_be_hex"], "sigma_lf", positive=True),
        _decode_f64(value["mu_hf_be_hex"], "mu_hf"),
        _decode_f64(value["sigma_hf_be_hex"], "sigma_hf", positive=True),
        _decode_f64(value["rho_be_hex"], "rho"),
    )
    if not -1.0 <= fit.rho <= 1.0:
        raise ValueError("content calibration rho differs")
    _joint_denominator(fit.rho)
    return value


def build_calibration_asset(
    *, producer_exact: str, protocol_digest: str, public_key_digest: str,
    fit: CalibrationFit,
) -> WeightedJointAsset:
    if not isinstance(fit, CalibrationFit):
        raise TypeError("content asset requires a CalibrationFit")
    payload = {
        "asset_role_id": CONTENT_CALIBRATION_CALIBRATION_ASSET_ROLE_ID,
        "base_exact": CONTENT_CALIBRATION_BASE_EXACT,
        "calibration_manifest_sha256": CONTENT_CALIBRATION_CALIBRATION_MANIFEST_SHA256,
        "calibration_pair_count": CONTENT_CALIBRATION_PAIRED_NULL_SCORE_COUNT,
        "calibration_pair_order": CALIBRATION_PAIR_ORDER,
        "calibration_prompt_list_sha256": CONTENT_CALIBRATION_CALIBRATION_PROMPT_LIST_SHA256,
        "calibration_protocol_digest": protocol_digest,
        "calibration_protocol_id": CONTENT_CALIBRATION_CALIBRATION_PROTOCOL_ID,
        "calibration_public_key_digest": public_key_digest,
        "evaluated_candidate_id": CONTENT_CALIBRATION_EVALUATED_CANDIDATE_ID,
        "fit_unit_count": CONTENT_CALIBRATION_CALIBRATION_COUNT,
        "hf_scorer_id": HF_SCORER_ID,
        "hf_weight_be_hex": _encode_f64(HF_WEIGHT, "hf_weight"),
        "iss_asset_sha256": ISS_ASSET_SHA256,
        "iss_asset_sidecar_file_sha256": ISS_ASSET_SIDECAR_FILE_SHA256,
        "joint_formula": "J=(0.25*z_lf+0.75*z_hf)/sqrt(0.25^2+0.75^2+2*0.25*0.75*rho)",
        "lf_scorer_id": CONTENT_WHITENING_LF_SCORER_ID,
        "lf_weight_be_hex": _encode_f64(LF_WEIGHT, "lf_weight"),
        "method_id": CONTENT_CALIBRATION_METHOD_ID,
        "mu_hf_be_hex": _encode_f64(fit.mu_hf, "mu_hf"),
        "mu_lf_be_hex": _encode_f64(fit.mu_lf, "mu_lf"),
        "producer_exact": producer_exact,
        "rho_be_hex": _encode_f64(fit.rho, "rho"),
        "schema_version": CONTENT_CALIBRATION_CALIBRATION_ASSET_SCHEMA_ID,
        "sigma_hf_be_hex": _encode_f64(fit.sigma_hf, "sigma_hf", positive=True),
        "sigma_lf_be_hex": _encode_f64(fit.sigma_lf, "sigma_lf", positive=True),
        "statistic_id": CALIBRATION_STATISTIC_ID,
        "value_dtype": "IEEE-754_binary64_big_endian_hex",
        "whitening_asset_sha256": V4_WHITENING_ASSET_SHA256,
        "whitening_asset_sidecar_file_sha256": (
            V4_WHITENING_ASSET_SIDECAR_FILE_SHA256
        ),
    }
    validated = _validate_asset_payload(dict(sorted(payload.items())))
    raw = stable_json_bytes(validated)
    return WeightedJointAsset(validated, raw)


def load_calibration_asset(path: str | Path, sidecar_path: str | Path) -> WeightedJointAsset:
    asset_path = Path(path)
    raw = asset_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if Path(sidecar_path).read_bytes() != f"{digest}  {asset_path.name}\n".encode("ascii"):
        raise ValueError("content calibration asset sidecar binding differs")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("content calibration asset must be UTF-8 JSON") from error
    validated = _validate_asset_payload(payload)
    if stable_json_bytes(validated) != raw:
        raise ValueError("content calibration asset must use stable JSON")
    return WeightedJointAsset(validated, raw)


__all__ = [
    "CALIBRATION_PAIR_ORDER", "CALIBRATION_STATISTIC_ID", "CalibrationFit",
    "HF_SCORER_ID", "HF_WEIGHT", "JOINT_OPERATOR_ID", "LFHFScorePair", "LF_WEIGHT",
    "WeightedGateEvidence", "WeightedJointAsset", "build_calibration_asset",
    "fit_weighted_joint_calibration", "load_calibration_asset", "stable_json_bytes",
    "weighted_gate_evidence", "weighted_joint_score",
]
