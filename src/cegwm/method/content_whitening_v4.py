"""Clean-null whitening fit for Content V4."""

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

from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
    reconstruct_lf_carrier,
)
from cegwm.runtime.observation import encode_final_rgb_image

FIT_MANIFEST_REPO_PATH = (
    "configs/content_chain/content_v4_clean_null_whitening_fit_v1.json"
)
FIT_MANIFEST_SCHEMA_ID = "cegwm_content_v4_clean_null_whitening_fit_manifest_v1"
ASSET_SCHEMA_ID = "cegwm_content_v4_clean_null_whitening_operator_asset_v1"
ASSET_ROLE_ID = "content_v4_clean_null_whitening_operator_v1"
ASSET_REPO_PATH = f"configs/content_chain/assets/{ASSET_ROLE_ID}.json"
ASSET_SIDECAR_REPO_PATH = f"{ASSET_REPO_PATH}.sha256"
ASSET_SHA256 = "a7021dd8b98bc4282b98ed5d1fe276236d99a3c9e80b9bdce015d28cf715633f"
ASSET_SIDECAR_SHA256 = "c900cce0980348eeadcf07d782b6169c4d46ac55d7154db0fc0a0a878cce0ced"
ASSET_PRODUCER_EXACT = "79f67646595bd99cc8b066cad0e4b12e96a22cbb"
CONTENT_V4_METHOD_ID = "content_v4_clean_null_whitened_lf_adaptive_hf_v1"
CONTENT_V4_EVALUATED_CANDIDATE_ID = (
    "content_v4_clean_null_whitened_lf_adaptive_hf_semantic_gate_v1"
)
CONTENT_V4_LF_SCORER_ID = "content_v4_whitened_lf_dct_matched_cosine_v1"
OBSERVATION_CONTRACT_ID = (
    "final_rgb_current_processor_sd35_vae_posterior_mode_float32_1x16x64x64_v1"
)
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
FIT_UNIT_COUNT = 32
FIT_HEIGHT = 512
FIT_WIDTH = 512
OBSERVATION_SHAPE = (1, 16, 64, 64)
OBSERVATION_STRIDE = (65536, 4096, 64, 1)
CHEBYSHEV_RING_BOUNDS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64))
CHEBYSHEV_RING_COUNTS = (3, 12, 48, 192, 768, 3072)
RIDGE_MULTIPLIER_POWER_OF_TWO = -10
WHITENING_SHAPE = (16, 6)
WHITENING_ORDER = "channel_major_band_minor"
WHITENING_WORD_COUNT = 96

_EXACT_40 = re.compile(r"[0-9a-f]{40}")
_WORD_8 = re.compile(r"[0-9a-f]{8}")
_MANIFEST_KEYS = {"schema_version", "entries"}
_ENTRY_KEYS = {"unit_id", "prompt", "generation_seed"}
_ASSET_KEYS = {
    "schema_version",
    "observation_contract_id",
    "whitening_shape",
    "whitening_order",
    "whitening_words_be_hex",
    "fit_sample_count",
    "producer_exact",
}


@dataclass(frozen=True, slots=True)
class FitEntry:
    unit_id: str
    prompt: str
    generation_seed: int


@dataclass(frozen=True, slots=True)
class FitManifest:
    entries: tuple[FitEntry, ...]


@dataclass(frozen=True, slots=True)
class WhiteningFit:
    words_be_hex: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WhiteningAsset:
    payload: Mapping[str, Any]
    json_bytes: bytes


@dataclass(frozen=True, slots=True)
class FrozenContentV4LFPublicAssets:
    """Current public VAE/carrier assets plus the fixed key-independent W."""

    carrier_assets: FrozenLFPublicAssets
    whitening_asset: WhiteningAsset

    def __post_init__(self) -> None:
        carrier = self.carrier_assets
        if not isinstance(carrier, FrozenLFPublicAssets):
            raise TypeError("Content V4 LF carrier assets must be frozen public assets")
        if (
            carrier.candidate_id != LF_BALANCED_BLOCKS_CARRIER_METHOD_ID
            or carrier.detector_statistic_id != LF_BLOCKNORM_DETECTOR_STATISTIC_ID
            or carrier.evaluated_candidate_id
            != LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID
        ):
            raise ValueError("Content V4 requires the current balanced LF carrier")
        _validate_asset_payload(self.whitening_asset.payload)


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    received = set(value)
    if received != expected:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        raise ValueError(f"{name} fields differ: missing={missing}, extra={extra}")


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty exact text")
    return value


def _parse_entry(value: Any, ordinal: int) -> FitEntry:
    if not isinstance(value, dict):
        raise ValueError("fit manifest entry must be an object")
    _require_exact_keys(value, _ENTRY_KEYS, "fit manifest entry")
    seed = value["generation_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 2026081000 + ordinal:
        raise ValueError("fit manifest generation seeds must be ordered 2026081000 through 2026081031")
    return FitEntry(
        unit_id=_require_text(value["unit_id"], "unit_id"),
        prompt=_require_text(value["prompt"], "prompt"),
        generation_seed=seed,
    )


def load_fit_manifest(path: str | Path) -> FitManifest:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("fit manifest must be UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("fit manifest must be an object")
    _require_exact_keys(value, _MANIFEST_KEYS, "fit manifest")
    if value["schema_version"] != FIT_MANIFEST_SCHEMA_ID:
        raise ValueError("fit manifest schema version differs")
    entries_value = value["entries"]
    if not isinstance(entries_value, list) or len(entries_value) != FIT_UNIT_COUNT:
        raise ValueError("fit manifest must contain exactly 32 entries")
    entries = tuple(_parse_entry(item, ordinal) for ordinal, item in enumerate(entries_value))
    unit_ids = tuple(item.unit_id for item in entries)
    generation_bindings = tuple((item.prompt, item.generation_seed) for item in entries)
    if len(set(unit_ids)) != FIT_UNIT_COUNT or len(set(generation_bindings)) != FIT_UNIT_COUNT:
        raise ValueError("fit manifest unit identities and generation bindings must be unique")
    return FitManifest(entries)


def _validate_observations(observations: Iterable[torch.Tensor]) -> torch.Tensor:
    values = tuple(observations)
    if len(values) != FIT_UNIT_COUNT:
        raise ValueError("whitening fit requires exactly 32 observations")
    for value in values:
        if not isinstance(value, torch.Tensor):
            raise TypeError("whitening observation must be a torch Tensor")
        if tuple(value.shape) != OBSERVATION_SHAPE:
            raise ValueError("whitening observation shape differs")
        if value.device.type != "cpu" or value.dtype != torch.float32:
            raise TypeError("whitening observation must be CPU float32")
        if tuple(value.stride()) != OBSERVATION_STRIDE or not value.is_contiguous():
            raise ValueError("whitening observation must be materialized in exact C order")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("whitening observation must be finite")
    return torch.stack([value[0] for value in values], dim=0).to(torch.float64)


def _dct_matrix() -> torch.Tensor:
    n = torch.arange(64, dtype=torch.float64)
    k = torch.arange(64, dtype=torch.float64).unsqueeze(1)
    matrix = torch.cos((math.pi / 64.0) * (n + 0.5) * k)
    matrix[0] *= math.sqrt(1.0 / 64.0)
    matrix[1:] *= math.sqrt(2.0 / 64.0)
    return matrix


def _ring_masks() -> tuple[torch.Tensor, ...]:
    axis = torch.arange(64)
    radius = torch.maximum(axis[:, None], axis[None, :])
    return tuple(
        (radius >= lower) & (radius < upper)
        for lower, upper in CHEBYSHEV_RING_BOUNDS
    )


def _detrended_dct(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 4 or tuple(values.shape[-2:]) != (64, 64):
        raise ValueError("detrended DCT input must be NCHW with 64 by 64 fields")
    if values.dtype != torch.float64 or values.device.type != "cpu":
        raise TypeError("detrended DCT input must be CPU float64")
    coordinate = (2.0 * torch.arange(64, dtype=torch.float64) - 63.0) / 63.0
    y = coordinate[:, None].expand(64, 64)
    x = coordinate[None, :].expand(64, 64)
    constant = values.mean(dim=(-2, -1), keepdim=True)
    centered = values - constant
    slope_x = torch.sum(centered * x, dim=(-2, -1), keepdim=True) / torch.sum(x.square())
    slope_y = torch.sum(centered * y, dim=(-2, -1), keepdim=True) / torch.sum(y.square())
    residual = centered - (slope_x * x + slope_y * y)
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("detrended observations must be finite")
    dct = _dct_matrix()
    return torch.matmul(torch.matmul(dct, residual), dct.transpose(0, 1))


def fit_whitening_operator(observations: Iterable[torch.Tensor]) -> WhiteningFit:
    values = _validate_observations(observations)
    coefficients = _detrended_dct(values)
    masks = _ring_masks()
    energy_columns = []
    for mask, count in zip(masks, CHEBYSHEV_RING_COUNTS, strict=True):
        energy_columns.append(
            coefficients[..., mask].square().sum(dim=(0, 2))
            / (FIT_UNIT_COUNT * count)
        )
    energy = torch.stack(energy_columns, dim=1)
    if tuple(energy.shape) != WHITENING_SHAPE or not bool(torch.isfinite(energy).all()):
        raise ValueError("whitening band energies must be finite 16 by 6")
    if bool((energy < 0.0).any()):
        raise ValueError("whitening band energies must be nonnegative")
    counts = torch.tensor(CHEBYSHEV_RING_COUNTS, dtype=torch.float64)
    energy_global = float((torch.sum(energy * counts) / (16 * 4095)).item())
    if not math.isfinite(energy_global) or energy_global <= 0.0:
        raise ValueError("global whitening energy must be finite and positive")
    ridge = math.ldexp(energy_global, RIDGE_MULTIPLIER_POWER_OF_TWO)
    weights64 = torch.rsqrt(energy + ridge)
    if not bool(torch.isfinite(weights64).all()) or bool((weights64 <= 0.0).any()):
        raise ValueError("whitening weights must be finite and positive")
    weights32 = weights64.to(torch.float32).contiguous()
    words = tuple(struct.pack(">f", float(value)).hex() for value in weights32.reshape(-1))
    return WhiteningFit(_validate_words(words))


def _validate_words(words: Iterable[str]) -> tuple[str, ...]:
    received = tuple(words)
    if len(received) != WHITENING_WORD_COUNT:
        raise ValueError("whitening asset must contain exactly 96 words")
    for word in received:
        if not isinstance(word, str) or _WORD_8.fullmatch(word) is None:
            raise ValueError("whitening words must be lowercase 8-hex big-endian values")
        value = struct.unpack(">f", bytes.fromhex(word))[0]
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("whitening words must encode finite positive float32 values")
    return received


def _validate_producer_exact(producer_exact: Any) -> str:
    if not isinstance(producer_exact, str) or _EXACT_40.fullmatch(producer_exact) is None:
        raise ValueError("producer exact must be a lowercase 40-character revision")
    return producer_exact


def _validate_asset_payload(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("whitening asset must be an object")
    _require_exact_keys(value, _ASSET_KEYS, "whitening asset")
    if value["schema_version"] != ASSET_SCHEMA_ID:
        raise ValueError("whitening asset schema version differs")
    if value["observation_contract_id"] != OBSERVATION_CONTRACT_ID:
        raise ValueError("whitening asset observation contract differs")
    if value["whitening_shape"] != list(WHITENING_SHAPE):
        raise ValueError("whitening asset shape differs")
    if value["whitening_order"] != WHITENING_ORDER:
        raise ValueError("whitening asset order differs")
    if value["fit_sample_count"] != FIT_UNIT_COUNT:
        raise ValueError("whitening asset fit sample count differs")
    _validate_producer_exact(value["producer_exact"])
    words = value["whitening_words_be_hex"]
    if not isinstance(words, list):
        raise ValueError("whitening asset words must be a list")
    _validate_words(words)
    return value


def build_whitening_asset(
    producer_exact: str,
    words: Iterable[str],
) -> WhiteningAsset:
    payload = {
        "schema_version": ASSET_SCHEMA_ID,
        "observation_contract_id": OBSERVATION_CONTRACT_ID,
        "whitening_shape": list(WHITENING_SHAPE),
        "whitening_order": WHITENING_ORDER,
        "whitening_words_be_hex": list(_validate_words(words)),
        "fit_sample_count": FIT_UNIT_COUNT,
        "producer_exact": _validate_producer_exact(producer_exact),
    }
    _validate_asset_payload(payload)
    return WhiteningAsset(payload, stable_json_bytes(payload))


def load_whitening_asset(path: str | Path) -> WhiteningAsset:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("whitening asset must be UTF-8 JSON") from error
    payload = _validate_asset_payload(value)
    if stable_json_bytes(payload) != raw:
        raise ValueError("whitening asset must use stable JSON encoding")
    return WhiteningAsset(payload, raw)


def load_frozen_content_v4_whitening_asset(repo_root: str | Path) -> WhiteningAsset:
    """Load the sole repository W only when its JSON and sidecar bytes are exact."""

    root = Path(repo_root)
    asset_path = root / ASSET_REPO_PATH
    sidecar_path = root / ASSET_SIDECAR_REPO_PATH
    asset_bytes = asset_path.read_bytes()
    sidecar_bytes = sidecar_path.read_bytes()
    if hashlib.sha256(asset_bytes).hexdigest() != ASSET_SHA256:
        raise ValueError("Content V4 whitening asset SHA differs")
    if hashlib.sha256(sidecar_bytes).hexdigest() != ASSET_SIDECAR_SHA256:
        raise ValueError("Content V4 whitening sidecar SHA differs")
    expected_sidecar = f"{ASSET_SHA256}  {Path(ASSET_REPO_PATH).name}\n".encode("ascii")
    if sidecar_bytes != expected_sidecar:
        raise ValueError("Content V4 whitening sidecar content differs")
    asset = load_whitening_asset(asset_path)
    if asset.payload["producer_exact"] != ASSET_PRODUCER_EXACT:
        raise ValueError("Content V4 whitening producer exact differs")
    return asset


def decode_whitening_weights(asset: WhiteningAsset) -> torch.Tensor:
    """Decode the exact channel-major/band-minor W as CPU float32."""

    if not isinstance(asset, WhiteningAsset):
        raise TypeError("Content V4 whitening asset type differs")
    payload = _validate_asset_payload(asset.payload)
    if stable_json_bytes(payload) != asset.json_bytes:
        raise ValueError("Content V4 whitening asset bytes differ from its payload")
    words = _validate_words(payload["whitening_words_be_hex"])
    weights = torch.tensor(
        [struct.unpack(">f", bytes.fromhex(word))[0] for word in words],
        dtype=torch.float32,
    ).reshape(WHITENING_SHAPE)
    if not weights.is_contiguous() or not bool(torch.isfinite(weights).all()):
        raise ValueError("decoded Content V4 whitening weights are invalid")
    return weights


def _detection_observation(image: Any, assets: FrozenContentV4LFPublicAssets) -> torch.Tensor:
    carrier = assets.carrier_assets
    observation = encode_final_rgb_image(image, carrier.image_processor, carrier.vae)
    observation = observation.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(observation.shape) != OBSERVATION_SHAPE:
        raise ValueError("Content V4 final-image observation shape differs")
    if tuple(observation.stride()) != OBSERVATION_STRIDE:
        raise ValueError("Content V4 final-image observation order differs")
    if not bool(torch.isfinite(observation).all()):
        raise ValueError("Content V4 final-image observation must be finite")
    return observation


def score_content_v4_lf_image(
    image: Any,
    detection_key: str | bytes | bytearray | memoryview,
    frozen_public_assets: FrozenContentV4LFPublicAssets,
) -> float:
    """Blind matched cosine after public VAE, detrend, DCT, and fixed W."""

    if not isinstance(frozen_public_assets, FrozenContentV4LFPublicAssets):
        raise TypeError("Content V4 LF score requires frozen public assets")
    observation = _detection_observation(image, frozen_public_assets)
    carrier = reconstruct_lf_carrier(
        detection_key,
        OBSERVATION_SHAPE,
        frozen_public_assets.carrier_assets,
        dtype=torch.float32,
        device="cpu",
    ).contiguous()
    if tuple(carrier.shape) != OBSERVATION_SHAPE or not bool(torch.isfinite(carrier).all()):
        raise ValueError("Content V4 reconstructed LF carrier is invalid")
    observation_dct = _detrended_dct(observation.to(torch.float64))
    carrier_dct = _detrended_dct(carrier.to(torch.float64))
    weights = decode_whitening_weights(frozen_public_assets.whitening_asset).to(torch.float64)
    masks = _ring_masks()
    observation_parts: list[torch.Tensor] = []
    carrier_parts: list[torch.Tensor] = []
    for channel in range(WHITENING_SHAPE[0]):
        for band, mask in enumerate(masks):
            weight = weights[channel, band]
            observation_parts.append(observation_dct[0, channel][mask] * weight)
            carrier_parts.append(carrier_dct[0, channel][mask] * weight)
    observation_vector = torch.cat(observation_parts)
    carrier_vector = torch.cat(carrier_parts)
    observation_norm = torch.linalg.vector_norm(observation_vector)
    carrier_norm = torch.linalg.vector_norm(carrier_vector)
    denominator = observation_norm * carrier_norm
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= 0.0:
        raise ValueError("Content V4 LF matched-cosine denominator must be positive")
    score = float(torch.dot(observation_vector, carrier_vector).div(denominator).item())
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        raise ValueError("Content V4 LF matched cosine must be finite in [-1, 1]")
    return score


__all__ = [
    "ASSET_PRODUCER_EXACT",
    "ASSET_REPO_PATH",
    "ASSET_ROLE_ID",
    "ASSET_SCHEMA_ID",
    "ASSET_SHA256",
    "ASSET_SIDECAR_REPO_PATH",
    "ASSET_SIDECAR_SHA256",
    "CHEBYSHEV_RING_BOUNDS",
    "CHEBYSHEV_RING_COUNTS",
    "CONTENT_V4_EVALUATED_CANDIDATE_ID",
    "CONTENT_V4_LF_SCORER_ID",
    "CONTENT_V4_METHOD_ID",
    "FIT_HEIGHT",
    "FIT_MANIFEST_REPO_PATH",
    "FIT_UNIT_COUNT",
    "FIT_WIDTH",
    "MODEL_ID",
    "OBSERVATION_CONTRACT_ID",
    "OBSERVATION_SHAPE",
    "OBSERVATION_STRIDE",
    "WHITENING_ORDER",
    "WHITENING_SHAPE",
    "WHITENING_WORD_COUNT",
    "FitEntry",
    "FitManifest",
    "FrozenContentV4LFPublicAssets",
    "WhiteningAsset",
    "WhiteningFit",
    "build_whitening_asset",
    "decode_whitening_weights",
    "fit_whitening_operator",
    "load_fit_manifest",
    "load_frozen_content_v4_whitening_asset",
    "load_whitening_asset",
    "score_content_v4_lf_image",
    "stable_json_bytes",
]
