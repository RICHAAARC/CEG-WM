"""Content V6 detector-domain ISS mathematics and future public asset contract."""

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

from cegwm.method.content_adaptive_v3 import (
    COMBINED_RELATIVE_L2,
    PROBE_EVALUATION_COUNT,
    ContentAdaptiveMeasurement,
    ContentAllocation,
    _content_v3_branch_deltas,
    _relative_l2,
)
from cegwm.method.content_whitening_v4 import (
    ASSET_SHA256 as V4_WHITENING_ASSET_SHA256,
    CONTENT_V4_LF_SCORER_ID,
    FrozenContentV4LFPublicAssets,
    WHITENING_SHAPE,
    _detection_observation,
    _detrended_dct,
    _ring_masks,
    decode_whitening_weights,
    score_content_v4_lf_image,
)
from cegwm.method.lf import reconstruct_lf_carrier
from cegwm.protocol.content_chain_v6 import (
    V6_DEVELOPMENT_MANIFEST_SHA256,
    V6_DEVELOPMENT_PROMPT_LIST_SHA256,
    V6_PERSONAL_SPEC_SHA256,
)
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from cegwm.shared.prg import prg_bytes

CONTENT_V6_METHOD_ID = "content_v6_detector_domain_iss_lf_adaptive_hf_v1"
CONTENT_V6_EVALUATED_CANDIDATE_ID = (
    "content_v6_detector_domain_iss_lf_adaptive_hf_semantic_gate_v1"
)
ISS_ASSET_SCHEMA_ID = "cegwm_content_v6_iss_gain_target_asset_v1"
ISS_ASSET_ROLE_ID = "content_v6_iss_gain_target_v1"
ISS_ASSET_REPO_PATH = "configs/content_chain/assets/content_v6_iss_gain_target_v1.json"
ISS_ASSET_SIDECAR_REPO_PATH = f"{ISS_ASSET_REPO_PATH}.sha256"
ISS_ASSET_SHA256 = "d66ff88640a3d1a020646cfde3face7502282bf835c9d3fb746b518dfb02c231"
ISS_ASSET_SIDECAR_SHA256 = (
    "27094d56994bc6f5d93564bad79ddd9ce8218d2d193786f4816535ee1e7f6538"
)
ISS_LF_PREPROJECTION_CONTROLLER_ID = (
    "content_v6_detector_domain_iss_lf_preprojection_multiplier_v1"
)
ISS_DEVELOPMENT_KEY_DOMAIN = "stage-a/content-v6-iss-development-key/v1"
ISS_WRONG_KEY_DOMAIN = "stage-a/content-adaptive-v2-external-wrong-key/v1"
ISS_DEVELOPMENT_COUNT = 32
ISS_WRONG_KEY_COUNT = 16
ISS_BETA_DEVELOPMENT = 1.0
ISS_BETA_MIN = 1.0
ISS_BETA_MAX = 2.0
ISS_MARGIN_DELTA = math.ldexp(1.0, -12)
ISS_SCORE_EQUIVALENCE_ABSOLUTE_TOLERANCE = 1e-12

_EXACT = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{16}")
_ASSET_FIELDS = (
    "schema_version",
    "asset_role_id",
    "personal_spec_sha256",
    "development_manifest_sha256",
    "development_prompt_list_sha256",
    "development_public_key_digest",
    "fit_sample_count",
    "gain_formula",
    "target_formula",
    "controller_formula",
    "beta_development_be_hex",
    "margin_delta_be_hex",
    "gain_g_be_hex",
    "target_m_be_hex",
    "v4_lf_scorer_id",
    "v4_whitening_asset_sha256",
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
        return _decode_f64(self.payload["gain_g_be_hex"], "gain_g")

    @property
    def target_m(self) -> float:
        return _decode_f64(self.payload["target_m_be_hex"], "target_m")


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def derive_development_key(root_key: str | bytes | bytearray | memoryview) -> bytes:
    return prg_bytes(normalize_detection_key(root_key), ISS_DEVELOPMENT_KEY_DOMAIN, 32)


def derive_development_wrong_keys(development_key: bytes) -> tuple[bytes, ...]:
    normalized = normalize_detection_key(development_key)
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


def _normalized_vector(vector: torch.Tensor, name: str) -> torch.Tensor:
    received = vector.detach().to(device="cpu", dtype=torch.float64).contiguous().reshape(-1)
    if not bool(torch.isfinite(received).all()):
        raise ValueError(f"{name} must be finite")
    norm = torch.linalg.vector_norm(received)
    if not bool(torch.isfinite(norm)) or float(norm.item()) <= 0.0:
        raise ValueError(f"{name} norm must be finite and positive")
    return received / norm


def _whitened_vector(values: torch.Tensor, assets: FrozenContentV4LFPublicAssets) -> torch.Tensor:
    transformed = _detrended_dct(values.to(device="cpu", dtype=torch.float64))
    weights = decode_whitening_weights(assets.whitening_asset).to(torch.float64)
    parts: list[torch.Tensor] = []
    for channel in range(WHITENING_SHAPE[0]):
        for band, mask in enumerate(_ring_masks()):
            parts.append(transformed[0, channel][mask] * weights[channel, band])
    return torch.cat(parts)


def content_v6_phi(image: Any, assets: FrozenContentV4LFPublicAssets) -> torch.Tensor:
    """Return normalized V4 public detector-domain observation features."""

    if not isinstance(assets, FrozenContentV4LFPublicAssets):
        raise TypeError("Content V6 Phi requires frozen V4 LF public assets")
    return _normalized_vector(_whitened_vector(_detection_observation(image, assets), assets), "Phi")


def content_v6_u(
    detection_key: str | bytes | bytearray | memoryview,
    assets: FrozenContentV4LFPublicAssets,
) -> torch.Tensor:
    """Return normalized keyed V4 whitened LF template features."""

    if not isinstance(assets, FrozenContentV4LFPublicAssets):
        raise TypeError("Content V6 template requires frozen V4 LF public assets")
    carrier = reconstruct_lf_carrier(
        detection_key,
        (1, 16, 64, 64),
        assets.carrier_assets,
        dtype=torch.float32,
        device="cpu",
    ).contiguous()
    return _normalized_vector(_whitened_vector(carrier, assets), "u")


def content_v6_h(
    image: Any,
    detection_key: str | bytes | bytearray | memoryview,
    assets: FrozenContentV4LFPublicAssets,
) -> float:
    """Compute h=<Phi(I),u_k> and require exact V4 scorer equivalence."""

    score = float(torch.dot(content_v6_phi(image, assets), content_v6_u(detection_key, assets)).item())
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        raise ValueError("Content V6 detector-domain score must be finite in [-1, 1]")
    v4_score = score_content_v4_lf_image(image, detection_key, assets)
    if abs(score - v4_score) > ISS_SCORE_EQUIVALENCE_ABSOLUTE_TOLERANCE:
        raise RuntimeError("Content V6 detector-domain score differs from V4 LF scorer")
    return score


def fit_iss_gain_target(
    measurements: Iterable[ISSDevelopmentMeasurement],
) -> ISSFit:
    """Fit the frozen median paired gain and rank-28 competition target."""

    received = tuple(measurements)
    if len(received) != ISS_DEVELOPMENT_COUNT:
        raise ValueError("Content V6 ISS fit requires exactly 32 development measurements")
    gains: list[float] = []
    competitions: list[float] = []
    for item in received:
        if not isinstance(item, ISSDevelopmentMeasurement):
            raise TypeError("Content V6 ISS measurements must use the frozen type")
        host = _finite_score(item.host_score, "host_score")
        beta_one = _finite_score(item.beta_one_score, "beta_one_score")
        competition = _finite_score(item.competition_score, "competition_score")
        if competition < host:
            raise ValueError("competition score must include the plain registered host score")
        gains.append(beta_one - host)
        competitions.append(competition)
    ordered_gains = sorted(gains)
    gain = (ordered_gains[15] + ordered_gains[16]) / 2.0
    rank_28 = sorted(competitions)[27]
    target = rank_28 + ISS_MARGIN_DELTA
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("Content V6 ISS fitted gain must be finite and positive")
    if not (-1.0 <= rank_28 < target < 1.0):
        raise ValueError("Content V6 ISS target must satisfy -1 <= q < m < 1")
    return ISSFit(gain, target, rank_28)


def iss_beta(host_score: Any, asset: ISSAsset) -> float:
    """Return total LF multiplier clamp((m-h)/g, 1, 2)."""

    if not isinstance(asset, ISSAsset):
        raise TypeError("Content V6 ISS controller requires a frozen public asset")
    host = _finite_score(host_score, "host_score")
    gain = asset.gain_g
    target = asset.target_m
    if not math.isfinite(gain) or gain <= 0.0 or not math.isfinite(target):
        raise ValueError("Content V6 ISS asset controller values are invalid")
    raw = (target - host) / gain
    if not math.isfinite(raw):
        raise ValueError("Content V6 ISS raw beta must be finite")
    return min(ISS_BETA_MAX, max(ISS_BETA_MIN, raw))


def embed_content_v6(
    latents: torch.Tensor,
    detection_key: str | bytes | bytearray | memoryview,
    hf_assets: Any,
    lf_assets: Any,
    allocation: ContentAllocation,
    beta: Any,
) -> tuple[torch.Tensor, ContentAdaptiveMeasurement]:
    """Apply beta to only the V4 LF preprojection delta, then project jointly."""

    if (
        not isinstance(latents, torch.Tensor)
        or latents.ndim != 4
        or not latents.dtype.is_floating_point
    ):
        raise TypeError("Content V6 embedding requires floating NCHW callback latents")
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("Content V6 callback latents must be finite")
    if not isinstance(allocation, ContentAllocation):
        raise TypeError("Content V6 embedding requires a real ContentAllocation")
    if not isinstance(beta, (int, float)) or isinstance(beta, bool):
        raise TypeError("beta must be a real scalar")
    multiplier = float(beta)
    if not math.isfinite(multiplier) or not ISS_BETA_MIN <= multiplier <= ISS_BETA_MAX:
        raise ValueError("Content V6 beta must be finite in [1, 2]")

    base64 = latents.to(torch.float64)
    lf_v4, hf_v4 = _content_v3_branch_deltas(
        latents, detection_key, hf_assets, lf_assets, allocation
    )
    lf_delta = lf_v4 * multiplier
    hf_delta = hf_v4
    if any(
        float(torch.linalg.vector_norm(delta).item()) == 0.0
        for delta in (lf_delta, hf_delta)
    ):
        raise RuntimeError("both Content V6 branches must be nonzero")

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
        raise RuntimeError("Content V6 actual-dtype embedding is zero or over budget")
    if lf_actual.perturbation_l2 == 0.0 or hf_actual.perturbation_l2 == 0.0:
        raise RuntimeError("both actual-dtype Content V6 branches must remain nonzero")
    return best, ContentAdaptiveMeasurement(
        measurement,
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
        raise ValueError(f"{name} must be lowercase 16-hex binary64")
    value = struct.unpack(">d", bytes.fromhex(word))[0]
    if not math.isfinite(value) or (positive and value <= 0.0):
        raise ValueError(f"{name} binary64 value is invalid")
    return value


def _validate_asset_payload(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict) or tuple(value) != tuple(sorted(_ASSET_FIELDS)):
        raise ValueError("Content V6 ISS asset fields or stable order differ")
    expected = {
        "schema_version": ISS_ASSET_SCHEMA_ID,
        "asset_role_id": ISS_ASSET_ROLE_ID,
        "personal_spec_sha256": V6_PERSONAL_SPEC_SHA256,
        "development_manifest_sha256": V6_DEVELOPMENT_MANIFEST_SHA256,
        "development_prompt_list_sha256": V6_DEVELOPMENT_PROMPT_LIST_SHA256,
        "fit_sample_count": ISS_DEVELOPMENT_COUNT,
        "gain_formula": "median_of_32_paired_registered_lf_beta_one_minus_plain_scores",
        "target_formula": "rank_28_of_max_plain_registered_and_16_beta_one_wrong_scores_plus_2^-12",
        "controller_formula": "beta_equals_clamp_total_multiplier_of_(m-h)/g_inclusive_1_to_2",
        "v4_lf_scorer_id": CONTENT_V4_LF_SCORER_ID,
        "v4_whitening_asset_sha256": V4_WHITENING_ASSET_SHA256,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("Content V6 ISS asset frozen identity differs")
    digest = value.get("development_public_key_digest")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Content V6 development public key digest differs")
    producer = value.get("producer_exact")
    if not isinstance(producer, str) or _EXACT.fullmatch(producer) is None:
        raise ValueError("Content V6 ISS producer exact differs")
    if _decode_f64(value.get("beta_development_be_hex"), "beta_development") != 1.0:
        raise ValueError("Content V6 ISS development beta differs")
    if _decode_f64(value.get("margin_delta_be_hex"), "margin_delta", positive=True) != ISS_MARGIN_DELTA:
        raise ValueError("Content V6 ISS margin delta differs")
    gain = _decode_f64(value.get("gain_g_be_hex"), "gain_g", positive=True)
    target = _decode_f64(value.get("target_m_be_hex"), "target_m")
    if not -1.0 < target < 1.0 or not math.isfinite(gain):
        raise ValueError("Content V6 ISS gain/target domain differs")
    return value


def build_iss_asset(
    producer_exact: str,
    development_key: bytes,
    fit: ISSFit,
) -> ISSAsset:
    if not isinstance(producer_exact, str) or _EXACT.fullmatch(producer_exact) is None:
        raise ValueError("producer exact must be lowercase 40-hex")
    if not isinstance(fit, ISSFit):
        raise TypeError("Content V6 ISS asset requires an ISSFit")
    if (
        not math.isfinite(fit.competition_rank_28)
        or not -1.0 <= fit.competition_rank_28 < fit.target_m < 1.0
        or fit.target_m != fit.competition_rank_28 + ISS_MARGIN_DELTA
    ):
        raise ValueError("Content V6 ISS fit target is inconsistent with frozen rank-28 rule")
    payload = {
        "asset_role_id": ISS_ASSET_ROLE_ID,
        "beta_development_be_hex": _encode_f64(ISS_BETA_DEVELOPMENT, "beta_development"),
        "controller_formula": "beta_equals_clamp_total_multiplier_of_(m-h)/g_inclusive_1_to_2",
        "development_manifest_sha256": V6_DEVELOPMENT_MANIFEST_SHA256,
        "development_prompt_list_sha256": V6_DEVELOPMENT_PROMPT_LIST_SHA256,
        "development_public_key_digest": public_key_digest(development_key),
        "fit_sample_count": ISS_DEVELOPMENT_COUNT,
        "gain_formula": "median_of_32_paired_registered_lf_beta_one_minus_plain_scores",
        "gain_g_be_hex": _encode_f64(fit.gain_g, "gain_g", positive=True),
        "margin_delta_be_hex": _encode_f64(ISS_MARGIN_DELTA, "margin_delta", positive=True),
        "personal_spec_sha256": V6_PERSONAL_SPEC_SHA256,
        "producer_exact": producer_exact,
        "schema_version": ISS_ASSET_SCHEMA_ID,
        "target_formula": "rank_28_of_max_plain_registered_and_16_beta_one_wrong_scores_plus_2^-12",
        "target_m_be_hex": _encode_f64(fit.target_m, "target_m"),
        "v4_lf_scorer_id": CONTENT_V4_LF_SCORER_ID,
        "v4_whitening_asset_sha256": V4_WHITENING_ASSET_SHA256,
    }
    validated = _validate_asset_payload(payload)
    raw = stable_json_bytes(validated)
    return ISSAsset(validated, raw)


def load_iss_asset(path: str | Path, sidecar_path: str | Path) -> ISSAsset:
    asset_path = Path(path)
    raw = asset_path.read_bytes()
    sidecar = Path(sidecar_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if sidecar != f"{digest}  {asset_path.name}\n".encode("ascii"):
        raise ValueError("Content V6 ISS asset sidecar binding differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Content V6 ISS asset must be UTF-8 JSON") from error
    validated = _validate_asset_payload(value)
    if stable_json_bytes(validated) != raw:
        raise ValueError("Content V6 ISS asset must use stable JSON encoding")
    return ISSAsset(validated, raw)


def load_frozen_content_v6_iss_asset(repo_root: str | Path) -> ISSAsset:
    root = Path(repo_root)
    asset_path = root / ISS_ASSET_REPO_PATH
    sidecar_path = root / ISS_ASSET_SIDECAR_REPO_PATH
    if hashlib.sha256(asset_path.read_bytes()).hexdigest() != ISS_ASSET_SHA256:
        raise ValueError("frozen Content V6 ISS asset SHA differs")
    if hashlib.sha256(sidecar_path.read_bytes()).hexdigest() != ISS_ASSET_SIDECAR_SHA256:
        raise ValueError("frozen Content V6 ISS sidecar file SHA differs")
    return load_iss_asset(asset_path, sidecar_path)


__all__ = [
    "CONTENT_V6_EVALUATED_CANDIDATE_ID",
    "CONTENT_V6_METHOD_ID",
    "ISS_ASSET_REPO_PATH",
    "ISS_ASSET_ROLE_ID",
    "ISS_ASSET_SCHEMA_ID",
    "ISS_ASSET_SHA256",
    "ISS_ASSET_SIDECAR_REPO_PATH",
    "ISS_ASSET_SIDECAR_SHA256",
    "ISS_BETA_MAX",
    "ISS_BETA_MIN",
    "ISS_LF_PREPROJECTION_CONTROLLER_ID",
    "ISSDevelopmentMeasurement",
    "ISSFit",
    "ISSAsset",
    "build_iss_asset",
    "content_v6_h",
    "content_v6_phi",
    "content_v6_u",
    "derive_development_key",
    "derive_development_wrong_keys",
    "embed_content_v6",
    "fit_iss_gain_target",
    "iss_beta",
    "load_frozen_content_v6_iss_asset",
    "load_iss_asset",
    "stable_json_bytes",
]
