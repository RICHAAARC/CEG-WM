"""Frozen clean-null whitening fit for the future Content V4 LF candidate."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

FIT_MANIFEST_REPO_PATH = (
    "configs/content_chain/content_v4_clean_null_whitening_fit_v1.json"
)
FIT_MANIFEST_SCHEMA_ID = "cegwm_content_v4_clean_null_whitening_fit_manifest_v1"
FIT_PROTOCOL_ID = "cegwm-stage-a-content-v4-clean-null-whitening-fit-v1"
FIT_RUN_PREFIX = "content-v4-whitening-fit"
ASSET_SCHEMA_ID = "cegwm_content_v4_clean_null_whitening_operator_asset_v1"
ASSET_ROLE_ID = "content_v4_clean_null_whitening_operator_v1"
CONTENT_V4_METHOD_ID = "content_v4_clean_null_whitened_lf_adaptive_hf_v1"
CONTENT_V4_EVALUATED_CANDIDATE_ID = (
    "content_v4_clean_null_whitened_lf_adaptive_hf_semantic_gate_v1"
)
RESERVED_FUTURE_CLEAN_PROTOCOL_ID = (
    "cegwm-stage-a-content-v4-whitened-lf-adaptive-hf-clean-v1"
)
RESERVED_FUTURE_RECORD_CONTRACT_ID = (
    "content_v4_whitened_lf_adaptive_hf_record_v1"
)
OBSERVATION_CONTRACT_ID = (
    "final_rgb_current_processor_sd35_vae_posterior_mode_float32_1x16x64x64_v1"
)
AUTHORIZED_BASE_EXACT = "aa2ff4476901033bff2564d93298889a9967303c"
ARCHIVE_MANIFEST_PATH = (
    "../CEG-WM-Archive/configs/experiments/lf_whitening_null_fit_manifest.json"
)
ARCHIVE_MANIFEST_SHA256 = (
    "5d7388a92c98aa5fb1996369bae8de65360e2d25fa7569400135753257bb6e86"
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
WHITENING_WORD_COUNT = 96

V3_FORMAL_ROSTER_SHA256 = (
    "dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88"
)
V3_FORMAL_DENY_TUPLES = (
    ("A violin maker carving a maple bridge beside a sunlit window", 1213061, 512, 512),
    ("A night market noodle stall reflected in rain-polished pavement", 1238321, 512, 512),
    ("Scientific illustration of desert succulents and layered roots", 1263581, 512, 512),
    ("A potter arranging glazed bowls on rough cedar shelves", 1288843, 512, 512),
    ("A satellite technician inspecting an antenna above a coastal plain", 1314103, 512, 512),
    ("A mountain hare crossing dark heather after a light snowfall", 1339367, 512, 512),
    ("Architectural photograph of a brick museum with long arcades", 1364627, 512, 512),
    ("An ecologist labeling moss samples in a compact field station", 1389887, 512, 512),
)
V3_CANARY_EXACT = "943813ee6e667361353a9eaaf096b21a00e18398"
V3_CANARY_ID = "content-v3-unweighted-lf-full-runtime-non-roster-canary-v1"
V3_CANARY_UNIT_ID = "content-v3-unweighted-lf-canary-0001"
V3_CANARY_SOURCE_ID = "content-v3-unweighted-lf-canary-prompt-9001"
V3_CANARY_DENY_TUPLE = (
    "A book conservator examining an illuminated manuscript under neutral studio light",
    1415149,
    512,
    512,
)

_HEX_64 = re.compile(r"[0-9a-f]{64}")
_EXACT_40 = re.compile(r"[0-9a-f]{40}")
_WORD_8 = re.compile(r"[0-9a-f]{8}")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "fit_protocol_id",
    "asset_role_id",
    "method_id",
    "evaluated_candidate_id",
    "reserved_future_clean_protocol_id",
    "reserved_future_record_contract_id",
    "observation_contract_id",
    "authorized_base_exact",
    "archive_source",
    "generation",
    "future_v4_formal_roster_rule",
    "entries",
}
_ENTRY_KEYS = {
    "cluster_ordinal",
    "cluster_identity",
    "prompt",
    "prompt_digest",
    "generation_seed",
    "image_lineage_identity",
    "image_lineage_digest",
    "split",
    "role_id",
}


@dataclass(frozen=True, slots=True)
class FitEntry:
    cluster_ordinal: int
    cluster_identity: str
    prompt: str
    prompt_digest: str
    generation_seed: int
    image_lineage_identity: str
    image_lineage_digest: str
    split: str
    role_id: str

    @property
    def generation_tuple(self) -> tuple[str, int, int, int]:
        return (self.prompt, self.generation_seed, FIT_HEIGHT, FIT_WIDTH)


@dataclass(frozen=True, slots=True)
class FitManifest:
    repo_path: str
    raw_sha256: str
    entries: tuple[FitEntry, ...]


@dataclass(frozen=True, slots=True)
class FitProtocolBinding:
    payload: Mapping[str, Any]
    digest: str
    run_id: str
    producer_exact: str


@dataclass(frozen=True, slots=True)
class WhiteningFit:
    words_be_hex: tuple[str, ...]
    energy_global: float
    ridge: float


@dataclass(frozen=True, slots=True)
class WhiteningAsset:
    payload: Mapping[str, Any]
    json_bytes: bytes
    digest: str


def stable_json_bytes(value: Any) -> bytes:
    """Return the one canonical UTF-8 JSON representation used by this asset."""

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
    if value["cluster_ordinal"] != ordinal:
        raise ValueError("fit manifest cluster ordinals must be ordered 0 through 31")
    seed = value["generation_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 2026081000 + ordinal:
        raise ValueError("fit manifest generation seeds must be ordered 2026081000 through 2026081031")
    cluster_identity = _require_text(value["cluster_identity"], "cluster_identity")
    prompt = _require_text(value["prompt"], "prompt")
    prompt_digest = value["prompt_digest"]
    if (
        not isinstance(prompt_digest, str)
        or _HEX_64.fullmatch(prompt_digest) is None
        or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_digest
    ):
        raise ValueError("fit manifest prompt digest differs")
    image_lineage_digest = value["image_lineage_digest"]
    if not isinstance(image_lineage_digest, str) or _HEX_64.fullmatch(image_lineage_digest) is None:
        raise ValueError("fit manifest image lineage digest must be lowercase SHA256")
    if value["image_lineage_identity"] != "clean_public_rgb8_to_vae_observation":
        raise ValueError("fit manifest image lineage identity differs")
    if value["split"] != "development" or value["role_id"] != "lf_whitening_null_fit":
        raise ValueError("fit manifest split or role differs")
    return FitEntry(
        cluster_ordinal=ordinal,
        cluster_identity=cluster_identity,
        prompt=prompt,
        prompt_digest=prompt_digest,
        generation_seed=seed,
        image_lineage_identity=value["image_lineage_identity"],
        image_lineage_digest=image_lineage_digest,
        split=value["split"],
        role_id=value["role_id"],
    )


def load_fit_manifest(path: str | Path) -> FitManifest:
    """Load the frozen current-project copy of public fit identities."""

    path = Path(path)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("fit manifest must be UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("fit manifest must be an object")
    _require_exact_keys(value, _TOP_LEVEL_KEYS, "fit manifest")
    expected_scalars = {
        "schema_version": FIT_MANIFEST_SCHEMA_ID,
        "fit_protocol_id": FIT_PROTOCOL_ID,
        "asset_role_id": ASSET_ROLE_ID,
        "method_id": CONTENT_V4_METHOD_ID,
        "evaluated_candidate_id": CONTENT_V4_EVALUATED_CANDIDATE_ID,
        "reserved_future_clean_protocol_id": RESERVED_FUTURE_CLEAN_PROTOCOL_ID,
        "reserved_future_record_contract_id": RESERVED_FUTURE_RECORD_CONTRACT_ID,
        "observation_contract_id": OBSERVATION_CONTRACT_ID,
        "authorized_base_exact": AUTHORIZED_BASE_EXACT,
        "future_v4_formal_roster_rule": "deny_every_ordered_fit_prompt_seed_height_width_tuple",
    }
    for name, expected in expected_scalars.items():
        if value[name] != expected:
            raise ValueError(f"fit manifest {name} differs")
    if value["archive_source"] != {
        "path": ARCHIVE_MANIFEST_PATH,
        "raw_sha256": ARCHIVE_MANIFEST_SHA256,
    }:
        raise ValueError("fit manifest Archive provenance differs")
    if value["generation"] != {
        "model_id": MODEL_ID,
        "vae_id": f"{MODEL_ID}:vae",
        "image_processor_id": f"{MODEL_ID}:image_processor",
        "inference_steps": 20,
        "height": FIT_HEIGHT,
        "width": FIT_WIDTH,
        "condition": "clean_plain_final_rgb_no_watermark_no_key",
    }:
        raise ValueError("fit manifest generation contract differs")
    entries_value = value["entries"]
    if not isinstance(entries_value, list) or len(entries_value) != FIT_UNIT_COUNT:
        raise ValueError("fit manifest must contain exactly 32 entries")
    entries = tuple(_parse_entry(item, ordinal) for ordinal, item in enumerate(entries_value))
    if entries[0].cluster_identity != "alabaster_alabaster" or entries[-1].cluster_identity != "frosted_frosted":
        raise ValueError("fit manifest endpoint identities differ")
    identities = tuple(item.cluster_identity for item in entries)
    tuples = tuple(item.generation_tuple for item in entries)
    if len(set(identities)) != FIT_UNIT_COUNT or len(set(tuples)) != FIT_UNIT_COUNT:
        raise ValueError("fit manifest identities and generation tuples must be unique")
    if set(tuples).intersection(V3_FORMAL_DENY_TUPLES):
        raise ValueError("fit manifest overlaps the V3 formal roster")
    if V3_CANARY_DENY_TUPLE in tuples:
        raise ValueError("fit manifest overlaps the authenticated V3 canary")
    return FitManifest(
        repo_path=FIT_MANIFEST_REPO_PATH,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        entries=entries,
    )


def _fit_spec_payload() -> dict[str, Any]:
    return {
        "observation_shape": list(OBSERVATION_SHAPE),
        "observation_order": "C",
        "observation_dtype": "float32",
        "detrend": {
            "rule": "per_observation_per_channel_affine_plane_least_squares",
            "coordinates": "axis_i_equals_(2i_minus_63)_over_63_for_i_0_through_63",
        },
        "transform": {
            "name": "orthonormal_2d_dct_ii",
            "basis": "D[k,n]=alpha[k]*cos(pi*(n+0.5)*k/64),alpha[0]=sqrt(1/64),alpha[k>0]=sqrt(2/64)",
            "dc": "excluded",
            "ring_metric": "chebyshev_max_frequency_index",
            "ring_bounds_lower_inclusive_upper_exclusive": [list(item) for item in CHEBYSHEV_RING_BOUNDS],
            "ring_counts": list(CHEBYSHEV_RING_COUNTS),
        },
        "energy_rule": "E[c,b]=sum_squared_ring_coefficients/(32*ring_count[b])",
        "global_energy_rule": "E_global=sum_over_c_b(ring_count[b]*E[c,b])/(16*4095)",
        "ridge_rule": "ridge=E_global*2^-10",
        "weight_rule": "W[c,b]=(E[c,b]+ridge)^-1/2",
        "materialization": "float32_round_to_nearest_ties_to_even",
        "word_encoding": "96_lowercase_8_hex_ieee754_binary32_big_endian_channel_major_band_minor",
    }


def bind_fit_protocol(manifest: FitManifest, producer_exact: str) -> FitProtocolBinding:
    """Bind the deterministic pre-W run identity before observations are produced."""

    if not isinstance(manifest, FitManifest):
        raise TypeError("fit protocol requires a validated FitManifest")
    if not isinstance(producer_exact, str) or _EXACT_40.fullmatch(producer_exact) is None:
        raise ValueError("producer exact must be a lowercase 40-character revision")
    ordered_bindings = [
        {
            key: value
            for key, value in asdict(item).items()
            if key != "prompt"
        }
        for item in manifest.entries
    ]
    payload: dict[str, Any] = {
        "fit_protocol_id": FIT_PROTOCOL_ID,
        "asset_role_id": ASSET_ROLE_ID,
        "method_id": CONTENT_V4_METHOD_ID,
        "evaluated_candidate_id": CONTENT_V4_EVALUATED_CANDIDATE_ID,
        "producer_exact": producer_exact,
        "generation": {
            "model_id": MODEL_ID,
            "vae_id": f"{MODEL_ID}:vae",
            "image_processor_id": f"{MODEL_ID}:image_processor",
            "inference_steps": 20,
            "height": FIT_HEIGHT,
            "width": FIT_WIDTH,
            "condition": "clean_plain_final_rgb_no_watermark_no_key",
        },
        "observation_contract_id": OBSERVATION_CONTRACT_ID,
        "fit_manifest": {
            "path": manifest.repo_path,
            "raw_sha256": manifest.raw_sha256,
            "archive_source_path": ARCHIVE_MANIFEST_PATH,
            "archive_source_raw_sha256": ARCHIVE_MANIFEST_SHA256,
            "ordered_bindings": ordered_bindings,
        },
        "fit_specification": _fit_spec_payload(),
        "disjointness": {
            "v3_formal_roster_sha256": V3_FORMAL_ROSTER_SHA256,
            "v3_canary_exact": V3_CANARY_EXACT,
            "v3_canary_id": V3_CANARY_ID,
            "v3_canary_unit_id": V3_CANARY_UNIT_ID,
            "v3_canary_source_id": V3_CANARY_SOURCE_ID,
            "future_v4_formal_rule": "deny_every_ordered_fit_prompt_seed_height_width_tuple",
        },
        "scientific_denominator": 0,
        "claim_ceiling": "engineering_asset_fit_only_no_scores_gates_threshold_or_scientific_result",
    }
    digest = hashlib.sha256(stable_json_bytes(payload)).hexdigest()
    run_id = f"{FIT_RUN_PREFIX}-{digest[:12]}-{producer_exact[:12]}"
    return FitProtocolBinding(payload, digest, run_id, producer_exact)


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
    masks = tuple((radius >= lower) & (radius < upper) for lower, upper in CHEBYSHEV_RING_BOUNDS)
    counts = tuple(int(mask.sum().item()) for mask in masks)
    if counts != CHEBYSHEV_RING_COUNTS or sum(counts) != 4095:
        raise RuntimeError("internal Chebyshev ring construction differs")
    if any(bool(mask[0, 0]) for mask in masks):
        raise RuntimeError("DCT DC coefficient must be excluded")
    return masks


def fit_whitening_operator(observations: Iterable[torch.Tensor]) -> WhiteningFit:
    """Fit the frozen 16-channel by 6-band clean-null whitening operator."""

    values = _validate_observations(observations)
    coordinate = (2.0 * torch.arange(64, dtype=torch.float64) - 63.0) / 63.0
    y = coordinate[:, None].expand(64, 64)
    x = coordinate[None, :].expand(64, 64)
    constant = values.mean(dim=(-2, -1), keepdim=True)
    centered = values - constant
    denominator_x = torch.sum(x.square())
    denominator_y = torch.sum(y.square())
    if not bool(torch.isfinite(denominator_x)) or not bool(torch.isfinite(denominator_y)):
        raise RuntimeError("affine detrend denominators must be finite")
    if float(denominator_x) <= 0.0 or float(denominator_y) <= 0.0:
        raise RuntimeError("affine detrend denominators must be positive")
    slope_x = torch.sum(centered * x, dim=(-2, -1), keepdim=True) / denominator_x
    slope_y = torch.sum(centered * y, dim=(-2, -1), keepdim=True) / denominator_y
    residual = centered - (slope_x * x + slope_y * y)
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("detrended observations must be finite")
    dct = _dct_matrix()
    coefficients = torch.matmul(torch.matmul(dct, residual), dct.transpose(0, 1))
    masks = _ring_masks()
    energy_columns = []
    for mask, count in zip(masks, CHEBYSHEV_RING_COUNTS, strict=True):
        denominator = FIT_UNIT_COUNT * count
        if denominator <= 0:
            raise RuntimeError("ring energy denominator must be positive")
        energy_columns.append(coefficients[..., mask].square().sum(dim=(0, 2)) / denominator)
    energy = torch.stack(energy_columns, dim=1)
    if tuple(energy.shape) != (16, 6) or not bool(torch.isfinite(energy).all()):
        raise ValueError("whitening band energies must be finite 16 by 6")
    if bool((energy < 0.0).any()):
        raise ValueError("whitening band energies must be nonnegative")
    counts = torch.tensor(CHEBYSHEV_RING_COUNTS, dtype=torch.float64)
    global_denominator = 16 * 4095
    if global_denominator <= 0:
        raise RuntimeError("global energy denominator must be positive")
    energy_global_tensor = torch.sum(energy * counts) / global_denominator
    energy_global = float(energy_global_tensor.item())
    if not math.isfinite(energy_global) or energy_global <= 0.0:
        raise ValueError("global whitening energy must be finite and positive")
    ridge = math.ldexp(energy_global, RIDGE_MULTIPLIER_POWER_OF_TWO)
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("whitening ridge must be finite and positive")
    weights64 = torch.rsqrt(energy + ridge)
    if not bool(torch.isfinite(weights64).all()) or bool((weights64 <= 0.0).any()):
        raise ValueError("whitening weights must be finite and positive")
    weights32 = weights64.to(torch.float32).contiguous()
    words = tuple(struct.pack(">f", float(value)).hex() for value in weights32.reshape(-1))
    _validate_words(words)
    return WhiteningFit(words, energy_global, ridge)


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


def build_whitening_asset(binding: FitProtocolBinding, words: Iterable[str]) -> WhiteningAsset:
    """Build the public stable-JSON asset whose digest includes the final words."""

    if not isinstance(binding, FitProtocolBinding):
        raise TypeError("whitening asset requires a FitProtocolBinding")
    received = _validate_words(words)
    payload = {
        "schema_version": ASSET_SCHEMA_ID,
        "asset_role_id": ASSET_ROLE_ID,
        "fit_protocol_id": FIT_PROTOCOL_ID,
        "fit_protocol_digest": binding.digest,
        "run_id": binding.run_id,
        "producer_exact": binding.producer_exact,
        "method_id": CONTENT_V4_METHOD_ID,
        "evaluated_candidate_id": CONTENT_V4_EVALUATED_CANDIDATE_ID,
        "fit_contract": binding.payload,
        "whitening_words_be_hex_channel_major_band_minor": list(received),
        "claim_ceiling": "public_engineering_asset_only_actual_W_requires_authorized_user_execution",
    }
    raw = stable_json_bytes(payload)
    return WhiteningAsset(payload, raw, hashlib.sha256(raw).hexdigest())


__all__ = [
    "ARCHIVE_MANIFEST_PATH",
    "ARCHIVE_MANIFEST_SHA256",
    "ASSET_ROLE_ID",
    "CHEBYSHEV_RING_BOUNDS",
    "CHEBYSHEV_RING_COUNTS",
    "CONTENT_V4_EVALUATED_CANDIDATE_ID",
    "CONTENT_V4_METHOD_ID",
    "FIT_HEIGHT",
    "FIT_MANIFEST_REPO_PATH",
    "FIT_PROTOCOL_ID",
    "FIT_UNIT_COUNT",
    "FIT_WIDTH",
    "MODEL_ID",
    "OBSERVATION_CONTRACT_ID",
    "OBSERVATION_SHAPE",
    "RESERVED_FUTURE_CLEAN_PROTOCOL_ID",
    "RESERVED_FUTURE_RECORD_CONTRACT_ID",
    "V3_CANARY_DENY_TUPLE",
    "V3_CANARY_EXACT",
    "V3_CANARY_ID",
    "V3_FORMAL_DENY_TUPLES",
    "WHITENING_WORD_COUNT",
    "FitEntry",
    "FitManifest",
    "FitProtocolBinding",
    "WhiteningAsset",
    "WhiteningFit",
    "bind_fit_protocol",
    "build_whitening_asset",
    "fit_whitening_operator",
    "load_fit_manifest",
    "stable_json_bytes",
]
