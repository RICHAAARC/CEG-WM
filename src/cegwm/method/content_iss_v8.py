"""Content V8 ordinary-detector ISS fit and V2 spatial writer."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from cegwm.method.content_adaptive_v2 import (
    COMBINED_RELATIVE_L2,
    PROBE_EVALUATION_COUNT,
    ContentAdaptiveMeasurement,
    ContentAllocation,
    _branch_transformed_delta,
    _relative_l2,
)
from cegwm.method.hf import FrozenHFPublicAssets, reconstruct_hf_carrier
from cegwm.method.lf import (
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
    reconstruct_lf_carrier,
    score_lf_image,
)
from cegwm.protocol.content_chain_v8 import (
    V8_DEVELOPMENT_KEY_DOMAIN,
    V8_DEVELOPMENT_MANIFEST_SHA256,
)
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from cegwm.shared.numerics import BudgetMeasurement
from cegwm.shared.prg import prg_bytes

CONTENT_V8_METHOD_ID = "content_v8_v2_spatial_lf_detector_domain_iss_v1"
CONTENT_V8_EVALUATED_CANDIDATE_ID = (
    "content_v8_v2_spatial_lf_detector_domain_iss_semantic_gate_v1"
)
CONTENT_V8_LF_WRITE_ID = (
    "beta_times_v2_A_LF_times_normalize_v2_lf_tile_weights_hadamard_c_LF_v1"
)
CONTENT_V8_HF_WRITE_ID = "unchanged_content_v2_spatial_hf_preprojection_delta_v1"
CONTENT_V8_PROJECTOR_ID = "shared_actual_dtype_relative_l2_at_most_0.012_v1"
CONTENT_V8_HOST_SCORER_SYMBOL = "cegwm.method.lf.score_lf_image"
ISS_ASSET_SCHEMA_ID = "cegwm_content_v8_v2_spatial_lf_iss_gain_target_asset_v1"
ISS_ASSET_ROLE_ID = "content_v8_v2_spatial_lf_iss_gain_target_v1"
ISS_DEVELOPMENT_COUNT = 32
ISS_WRONG_KEY_COUNT = 16
ISS_WRONG_KEY_DOMAIN = "stage-a/content-adaptive-v2-external-wrong-key/v1"
ISS_BETA_DEVELOPMENT = 1.0
ISS_BETA_MIN = 1.0
ISS_BETA_MAX = 2.0
ISS_MARGIN_DELTA = math.ldexp(1.0, -12)

_EXACT = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{16}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ASSET_FIELDS = (
    "schema_version",
    "asset_role_id",
    "protocol_digest",
    "development_manifest_sha256",
    "development_public_key_digest",
    "fit_sample_count",
    "gain_formula",
    "target_formula",
    "controller_formula",
    "beta_development_be_hex",
    "margin_delta_be_hex",
    "gain_g_be_hex",
    "target_m_be_hex",
    "host_scorer_symbol",
    "lf_detector_statistic_id",
    "lf_write_id",
    "hf_write_id",
    "projector_id",
    "producer_exact",
)


@dataclass(frozen=True, slots=True)
class ISSDevelopmentMeasurement:
    host_score: float
    beta_one_score: float
    competition_score: float


@dataclass(frozen=True, slots=True)
class ISSFit:
    gain_g: float
    target_m: float
    competition_rank_28: float


@dataclass(frozen=True, slots=True)
class ISSAsset:
    payload: Mapping[str, Any]
    json_bytes: bytes

    @property
    def gain_g(self) -> float:
        return _decode_f64(self.payload["gain_g_be_hex"], "gain_g", positive=True)

    @property
    def target_m(self) -> float:
        return _decode_f64(self.payload["target_m_be_hex"], "target_m")


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def derive_development_key(root_key: str | bytes | bytearray | memoryview) -> bytes:
    return prg_bytes(normalize_detection_key(root_key), V8_DEVELOPMENT_KEY_DOMAIN, 32)


def derive_wrong_keys(key: str | bytes | bytearray | memoryview) -> tuple[bytes, ...]:
    normalized = normalize_detection_key(key)
    return tuple(
        prg_bytes(normalized, f"{ISS_WRONG_KEY_DOMAIN}/index={index}", 32)
        for index in range(ISS_WRONG_KEY_COUNT)
    )


def _finite_score(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar) or not -1.0 <= scalar <= 1.0:
        raise ValueError(f"{name} must be finite in [-1, 1]")
    return scalar


def content_v8_h(
    image: Any,
    detection_key: str | bytes | bytearray | memoryview,
    lf_assets: FrozenLFPublicAssets,
) -> float:
    """Use exactly the registered ordinary blind LF scorer; no alternate transform."""

    if not isinstance(lf_assets, FrozenLFPublicAssets):
        raise TypeError("Content V8 h requires FrozenLFPublicAssets")
    return _finite_score(
        score_lf_image(image, detection_key, lf_assets),
        "ordinary registered LF score",
    )


def fit_iss_gain_target(measurements: Iterable[ISSDevelopmentMeasurement]) -> ISSFit:
    received = tuple(measurements)
    if len(received) != ISS_DEVELOPMENT_COUNT:
        raise ValueError("Content V8 ISS fit requires exactly 32 measurements")
    gains: list[float] = []
    competitions: list[float] = []
    for item in received:
        if not isinstance(item, ISSDevelopmentMeasurement):
            raise TypeError("Content V8 ISS fit requires frozen measurement values")
        host = _finite_score(item.host_score, "host_score")
        beta_one = _finite_score(item.beta_one_score, "beta_one_score")
        competition = _finite_score(item.competition_score, "competition_score")
        if competition < host:
            raise ValueError("competition must include ordinary registered host score")
        gains.append(beta_one - host)
        competitions.append(competition)
    ordered = sorted(gains)
    gain = (ordered[15] + ordered[16]) / 2.0
    rank_28 = sorted(competitions)[27]
    target = rank_28 + ISS_MARGIN_DELTA
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("Content V8 fitted gain must be finite and positive")
    if not (-1.0 <= rank_28 < target < 1.0):
        raise ValueError("Content V8 fitted target must satisfy -1 <= q < m < 1")
    return ISSFit(gain, target, rank_28)


def iss_beta(host_score: Any, asset: ISSAsset) -> float:
    if not isinstance(asset, ISSAsset):
        raise TypeError("Content V8 controller requires an ISSAsset")
    host = _finite_score(host_score, "host_score")
    raw = (asset.target_m - host) / asset.gain_g
    if not math.isfinite(raw):
        raise ValueError("Content V8 raw beta must be finite")
    return min(ISS_BETA_MAX, max(ISS_BETA_MIN, raw))


def _content_v8_branch_deltas(
    latents: torch.Tensor,
    detection_key: str | bytes | bytearray | memoryview,
    hf_assets: FrozenHFPublicAssets,
    lf_assets: FrozenLFPublicAssets,
    allocation: ContentAllocation,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hf_assets.injection_step_index != 18 or lf_assets.injection_step_index != 18:
        raise ValueError("Content V8 base carriers must both use step 18")
    base_l2 = torch.linalg.vector_norm(latents.to(torch.float64))
    if not bool(torch.isfinite(base_l2)) or float(base_l2.item()) <= 0.0:
        raise ValueError("Content V8 writer requires a finite nonzero latent")
    hf_carrier = reconstruct_hf_carrier(
        detection_key, tuple(latents.shape), hf_assets,
        dtype=torch.float32, device=latents.device,
    )
    lf_carrier = reconstruct_lf_carrier(
        detection_key, tuple(latents.shape), lf_assets,
        dtype=torch.float32, device=latents.device,
    )
    # These are the completed V2 spatial preprojection deltas:
    # A_branch(content) * normalize(tile_weights(content) hadamard carrier).
    hf_delta = _branch_transformed_delta(
        hf_carrier,
        allocation.hf_tile_weights,
        base_l2 * COMBINED_RELATIVE_L2 * allocation.hf_branch_share,
    )
    lf_delta = _branch_transformed_delta(
        lf_carrier,
        allocation.lf_tile_weights,
        base_l2 * COMBINED_RELATIVE_L2 * allocation.lf_branch_share,
    )
    return lf_delta, hf_delta


def _project_content_v8_deltas(
    latents: torch.Tensor,
    lf_delta: torch.Tensor,
    hf_delta: torch.Tensor,
) -> tuple[
    torch.Tensor,
    BudgetMeasurement,
    BudgetMeasurement,
    BudgetMeasurement,
]:
    base64 = latents.to(torch.float64)

    def candidate_at(scale: float) -> torch.Tensor:
        return (base64 + scale * (lf_delta + hf_delta)).to(latents.dtype)

    low, high = 0.0, 2.0
    best = latents.detach().clone()
    measurement = _relative_l2(latents, best)
    for _ in range(96):
        middle = (low + high) / 2.0
        trial = candidate_at(middle)
        trial_measurement = _relative_l2(latents, trial)
        if trial_measurement.relative_l2 <= COMBINED_RELATIVE_L2:
            low, best, measurement = middle, trial, trial_measurement
        else:
            high = middle
    lf_actual = _relative_l2(latents, (base64 + low * lf_delta).to(latents.dtype))
    hf_actual = _relative_l2(latents, (base64 + low * hf_delta).to(latents.dtype))
    if measurement.perturbation_l2 == 0.0 or measurement.relative_l2 > COMBINED_RELATIVE_L2:
        raise RuntimeError("Content V8 actual-dtype joint write is zero or over budget")
    if lf_actual.perturbation_l2 == 0.0 or hf_actual.perturbation_l2 == 0.0:
        raise RuntimeError("Content V8 actual-dtype branches must both remain nonzero")
    return best, measurement, lf_actual, hf_actual


def embed_content_v8(
    latents: torch.Tensor,
    detection_key: str | bytes | bytearray | memoryview,
    hf_assets: FrozenHFPublicAssets,
    lf_assets: FrozenLFPublicAssets,
    allocation: ContentAllocation,
    beta: Any,
) -> tuple[torch.Tensor, ContentAdaptiveMeasurement]:
    """Scale only the completed V2 LF delta by beta, then jointly project."""

    if (
        not isinstance(latents, torch.Tensor)
        or latents.ndim != 4
        or not latents.dtype.is_floating_point
    ):
        raise TypeError("Content V8 embedding requires floating NCHW latents")
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("Content V8 latents must be finite")
    if not isinstance(allocation, ContentAllocation):
        raise TypeError("Content V8 embedding requires ContentAllocation")
    if not isinstance(beta, (int, float)) or isinstance(beta, bool):
        raise TypeError("Content V8 beta must be a real scalar")
    multiplier = float(beta)
    if not math.isfinite(multiplier) or not ISS_BETA_MIN <= multiplier <= ISS_BETA_MAX:
        raise ValueError("Content V8 beta must be finite in [1, 2]")
    lf_delta, hf_delta = _content_v8_branch_deltas(
        latents, detection_key, hf_assets, lf_assets, allocation
    )
    # ISS acts after weighting and normalization, on LF only. HF is byte-for-byte
    # the unchanged V2 preprojection tensor.
    embedded, budget, lf_actual, hf_actual = _project_content_v8_deltas(
        latents, lf_delta * multiplier, hf_delta
    )
    return embedded, ContentAdaptiveMeasurement(
        budget,
        lf_actual.relative_l2,
        hf_actual.relative_l2,
        allocation.lf_branch_share,
        allocation.hf_branch_share,
        *allocation.counterfactual_effects,
        PROBE_EVALUATION_COUNT,
    )


def _encode_f64(value: Any, name: str, *, positive: bool = False) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar) or (positive and scalar <= 0.0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return struct.pack(">d", scalar).hex()


def _decode_f64(word: Any, name: str, *, positive: bool = False) -> float:
    if not isinstance(word, str) or _HEX64.fullmatch(word) is None:
        raise ValueError(f"{name} must be lowercase binary64 hex")
    value = struct.unpack(">d", bytes.fromhex(word))[0]
    if not math.isfinite(value) or (positive and value <= 0.0):
        raise ValueError(f"{name} binary64 value is invalid")
    return value


def _validate_asset_payload(value: Any, expected_protocol_digest: str | None) -> Mapping[str, Any]:
    if not isinstance(value, dict) or tuple(value) != tuple(sorted(_ASSET_FIELDS)):
        raise ValueError("Content V8 ISS asset fields or stable order differ")
    expected = {
        "schema_version": ISS_ASSET_SCHEMA_ID,
        "asset_role_id": ISS_ASSET_ROLE_ID,
        "development_manifest_sha256": V8_DEVELOPMENT_MANIFEST_SHA256,
        "fit_sample_count": ISS_DEVELOPMENT_COUNT,
        "gain_formula": "median_of_32_beta1_ordinary_registered_minus_ordinary_registered_host",
        "target_formula": "rank28_of_max_ordinary_registered_host_and_16_beta1_ordinary_wrong_scores_plus_2^-12",
        "controller_formula": "beta_equals_clamp_(m-h)/g_inclusive_1_to_2",
        "host_scorer_symbol": CONTENT_V8_HOST_SCORER_SYMBOL,
        "lf_detector_statistic_id": LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        "lf_write_id": CONTENT_V8_LF_WRITE_ID,
        "hf_write_id": CONTENT_V8_HF_WRITE_ID,
        "projector_id": CONTENT_V8_PROJECTOR_ID,
    }
    if any(value.get(name) != item for name, item in expected.items()):
        raise ValueError("Content V8 ISS asset frozen identity differs")
    for name in ("protocol_digest", "development_public_key_digest"):
        if not isinstance(value.get(name), str) or _SHA256.fullmatch(value[name]) is None:
            raise ValueError(f"Content V8 {name} differs")
    if expected_protocol_digest is not None and value["protocol_digest"] != expected_protocol_digest:
        raise ValueError("Content V8 asset protocol binding differs")
    if not isinstance(value.get("producer_exact"), str) or _EXACT.fullmatch(value["producer_exact"]) is None:
        raise ValueError("Content V8 producer exact differs")
    if _decode_f64(value.get("beta_development_be_hex"), "beta_development") != 1.0:
        raise ValueError("Content V8 development beta differs")
    if _decode_f64(value.get("margin_delta_be_hex"), "margin_delta", positive=True) != ISS_MARGIN_DELTA:
        raise ValueError("Content V8 margin differs")
    _decode_f64(value.get("gain_g_be_hex"), "gain_g", positive=True)
    target = _decode_f64(value.get("target_m_be_hex"), "target_m")
    if not -1.0 < target < 1.0:
        raise ValueError("Content V8 target domain differs")
    return value


def build_iss_asset(
    producer_exact: str,
    protocol_digest: str,
    development_key: bytes,
    fit: ISSFit,
) -> ISSAsset:
    if not isinstance(producer_exact, str) or _EXACT.fullmatch(producer_exact) is None:
        raise ValueError("producer exact must be lowercase 40-hex")
    if not isinstance(protocol_digest, str) or _SHA256.fullmatch(protocol_digest) is None:
        raise ValueError("protocol digest must be lowercase SHA256")
    if not isinstance(fit, ISSFit):
        raise TypeError("Content V8 asset requires ISSFit")
    if (
        not math.isfinite(fit.gain_g)
        or fit.gain_g <= 0.0
        or not math.isfinite(fit.competition_rank_28)
        or not -1.0 <= fit.competition_rank_28 < fit.target_m < 1.0
        or fit.target_m != fit.competition_rank_28 + ISS_MARGIN_DELTA
    ):
        raise ValueError("Content V8 fit is inconsistent with the frozen rule")
    payload = {
        "asset_role_id": ISS_ASSET_ROLE_ID,
        "beta_development_be_hex": _encode_f64(ISS_BETA_DEVELOPMENT, "beta_development"),
        "controller_formula": "beta_equals_clamp_(m-h)/g_inclusive_1_to_2",
        "development_manifest_sha256": V8_DEVELOPMENT_MANIFEST_SHA256,
        "development_public_key_digest": public_key_digest(development_key),
        "fit_sample_count": ISS_DEVELOPMENT_COUNT,
        "gain_formula": "median_of_32_beta1_ordinary_registered_minus_ordinary_registered_host",
        "gain_g_be_hex": _encode_f64(fit.gain_g, "gain_g", positive=True),
        "hf_write_id": CONTENT_V8_HF_WRITE_ID,
        "host_scorer_symbol": CONTENT_V8_HOST_SCORER_SYMBOL,
        "lf_detector_statistic_id": LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        "lf_write_id": CONTENT_V8_LF_WRITE_ID,
        "margin_delta_be_hex": _encode_f64(ISS_MARGIN_DELTA, "margin_delta", positive=True),
        "producer_exact": producer_exact,
        "projector_id": CONTENT_V8_PROJECTOR_ID,
        "protocol_digest": protocol_digest,
        "schema_version": ISS_ASSET_SCHEMA_ID,
        "target_formula": "rank28_of_max_ordinary_registered_host_and_16_beta1_ordinary_wrong_scores_plus_2^-12",
        "target_m_be_hex": _encode_f64(fit.target_m, "target_m"),
    }
    validated = _validate_asset_payload(payload, protocol_digest)
    return ISSAsset(validated, stable_json_bytes(validated))


def load_iss_asset(
    path: str | Path,
    sidecar_path: str | Path,
    *,
    expected_protocol_digest: str | None = None,
) -> ISSAsset:
    asset_path = Path(path)
    raw = asset_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if Path(sidecar_path).read_bytes() != f"{digest}  {asset_path.name}\n".encode("ascii"):
        raise ValueError("Content V8 ISS asset sidecar binding differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Content V8 ISS asset must be UTF-8 JSON") from error
    validated = _validate_asset_payload(value, expected_protocol_digest)
    if stable_json_bytes(validated) != raw:
        raise ValueError("Content V8 ISS asset must use canonical stable JSON")
    return ISSAsset(validated, raw)


__all__ = [
    "CONTENT_V8_EVALUATED_CANDIDATE_ID", "CONTENT_V8_HOST_SCORER_SYMBOL",
    "CONTENT_V8_METHOD_ID", "ISS_ASSET_ROLE_ID", "ISSAsset",
    "ISSDevelopmentMeasurement", "ISSFit", "build_iss_asset", "content_v8_h",
    "derive_development_key", "derive_wrong_keys", "embed_content_v8",
    "fit_iss_gain_target", "iss_beta", "load_iss_asset", "stable_json_bytes",
]
