#!/usr/bin/env python3
"""Prepare/freeze BlindDetection-V1 assets without silently running models."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable
import zipfile

from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cegwm.method.blind_detection import (  # noqa: E402
    BLIND_DEV_DENOMINATOR,
    BLIND_DEV_DISJOINT_FROM,
    BLIND_PRODUCTION_RUNTIME_ID,
    BLIND_STATISTIC_ID,
    BLIND_THRESHOLD_SCHEMA_ID,
    BlindCalibrationRoster,
    BlindCalibrationUnit,
    BlindThresholdAsset,
    build_threshold_asset,
    candidate_tau_blind,
    decode_binary64,
    encode_binary64,
    load_threshold_asset,
    stable_json_bytes,
)
from cegwm.method.content_iss import (  # noqa: E402
    ISS_ASSET_ROLE_ID,
    ISS_ASSET_SCHEMA_ID,
    ISSAsset,
    stable_json_bytes as iss_stable_json_bytes,
)
from cegwm.method.content_whitening import (  # noqa: E402
    ASSET_PRODUCER_EXACT as WHITENING_ASSET_PRODUCER_EXACT,
    ASSET_SCHEMA_ID as WHITENING_ASSET_SCHEMA_ID,
    CONTENT_WHITENING_LF_SCORER_ID,
    FIT_UNIT_COUNT as WHITENING_FIT_UNIT_COUNT,
    OBSERVATION_CONTRACT_ID as WHITENING_OBSERVATION_CONTRACT_ID,
    WHITENING_ORDER,
    WHITENING_SHAPE,
    FrozenContentWhiteningLFPublicAssets,
    WhiteningAsset,
    stable_json_bytes as whitening_stable_json_bytes,
)
from cegwm.geometry_v7.syncseal import (  # noqa: E402
    SYNCSEAL_TORCHSCRIPT_URL,
    SyncSealTorchScript,
    download_official_syncseal_torchscript,
)
from cegwm.geometry_v7.r1a import condition_by_id, render_r1a_attack  # noqa: E402
from cegwm.method.content_weighted_joint import (  # noqa: E402
    HF_SCORER_ID,
    WeightedJointAsset,
    stable_json_bytes as weighted_stable_json_bytes,
)
from cegwm.protocol.content_chain import CONTENT_CHAIN_PUBLIC_KEY_DIGEST  # noqa: E402
from cegwm.runtime.blind_detection import (  # noqa: E402
    BLIND_PREPROCESS_ID,
    BLIND_SCORER_ID,
    BlindEmbeddingAssets,
    BlindProductionAssets,
    detect_watermark,
    embed_watermark,
    run_development_calibration,
    run_development_full_system_replay,
)
from cegwm.runtime.content_weighted_joint_sd35 import ContentCalibrationAssets  # noqa: E402
from cegwm.runtime.content_iss_sd35 import (  # noqa: E402
    ContentISSEvaluationAssets,
    run_content_iss_evaluation_pair,
)
from cegwm.runtime.diffusers_sd35 import run_sd35_plain  # noqa: E402
from cegwm.runtime.observation import require_ordinary_rgb_image  # noqa: E402
from cegwm.shared.keys import normalize_detection_key, public_key_digest  # noqa: E402


CALIBRATION_RESULT_SCHEMA = "cegwm_blind_detection_v1_calibration_result_v2"
CALIBRATION_CLAIM_CEILING = (
    "engineering_N_dev_256_threshold_calibration_only; science_denominator=0; "
    "not_fixed_FPR_production_reliability_or_paper_evidence"
)
CONTENT_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
ROSTER_REPO_PATH = Path(
    "configs/blind_detection/blind_detection_v1_dev_roster_256.json"
)
COLAB_ASSETS_REPO_PATH = Path(
    "configs/blind_detection/blind_detection_v1_colab_assets.json"
)
WEIGHTED_ASSET_REPO_PATH = Path(
    "configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json"
)
WHITENING_ASSET_REPO_PATH = Path(
    "configs/content_chain/assets/content_v4_clean_null_whitening_operator_v1.json"
)
ISS_ASSET_REPO_PATH = Path(
    "configs/content_chain/assets/content_v6_iss_gain_target_v1.json"
)
REPOSITORY_THRESHOLD_REPO_PATH = Path(
    "configs/blind_detection/assets/blind_detection_v1_thresholds.json"
)
WEIGHTED_ASSET_PRODUCER_EXACT = "c38522dcab6cb173cedf8415cee2fd30998222ba"
THRESHOLD_CALIBRATION_PRODUCER_EXACT = "920561bd264075628d52dcb70deef842284b6a75"
FROZEN_TAU_BLIND = 1.1328391433063743
FROZEN_TAU_BLIND_HEX = "3ff2201bf0021293"
ROOT_KEY_ENV = "CEG_WM_ROOT_KEY"
HF_TOKEN_ENV = "HF_TOKEN"
_PROMPT_SOURCES = (
    (
        "content_v6_iss_development_v1",
        "configs/content_chain/content_v6_iss_development_v1.jsonl",
    ),
    (
        "content_v9_calibration_v1",
        "configs/content_chain/content_v9_calibration_v1.jsonl",
    ),
)
_DEVELOPMENT_SEEDS = (2026101000, 2026101001, 2026101002, 2026101003)
CALLBACK_N4_CONFIG_REPO_PATH = Path(
    "configs/blind_detection/blind_detection_v1_callback_n4.json"
)
CALLBACK_N4_RESULT_SCHEMA = "cegwm_blind_detection_v1_callback_n4_result_v1"
CALLBACK_N4_TAU = 1.1328391433063743
CALLBACK_N4_TAU_HEX = "3ff2201bf0021293"
CALLBACK_N4_COVERAGE = (
    "direct_positive",
    "geometry_recovered_positive",
    "direct_negative",
    "geometry_recovered_negative",
)


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


@dataclass(frozen=True, slots=True)
class BlindGenerationUnit:
    calibration_unit: BlindCalibrationUnit
    prompt: str
    seed: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class CallbackN4Unit:
    case_id: str
    coverage: str
    source_stratum: str
    prompt: str
    prompt_source_path: str
    prompt_source_unit_id: str
    seed: int
    watermarked: bool
    attacked: bool
    height: int = 512
    width: int = 512


@dataclass(frozen=True, slots=True)
class CallbackN4EmbeddingRequest:
    prompt: str
    seed: int
    height: int
    width: int


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"prompt source line {line_number} is not JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"prompt source line {line_number} must be an object")
            rows.append(row)
    return tuple(rows)


def load_roster_inputs(
    repo_root: str | Path = REPO_ROOT,
) -> tuple[BlindCalibrationRoster, tuple[BlindGenerationUnit, ...], dict[str, Any]]:
    """Expand the committed seed-major 64-prompt by four-seed roster."""

    root = Path(repo_root)
    payload = _read_json(root / ROSTER_REPO_PATH)
    required = {
        "denominator",
        "dimensions",
        "disjoint_from",
        "exclusive_reservation",
        "geometry_v7_pair_sources",
        "ordering",
        "prompt_sources",
        "seeds",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("blind development roster fields differ")
    if payload["denominator"] != BLIND_DEV_DENOMINATOR:
        raise ValueError("blind development denominator differs")
    if payload["dimensions"] != {"height": 512, "width": 512}:
        raise ValueError("blind development dimensions differ")
    if tuple(payload["disjoint_from"]) != BLIND_DEV_DISJOINT_FROM:
        raise ValueError("blind development disjointness declaration differs")
    if payload["exclusive_reservation"] != "blind_detection_v1_development_only":
        raise ValueError("blind development exclusive reservation differs")
    expected_geometry_sources = (
        {
            "path": "configs/content_chain/content_adaptive_dual_branch_v2_clean.jsonl",
            "selection": "first_4_in_source_order",
        },
        {
            "path": "configs/content_chain/content_v6_iss_clean.jsonl",
            "selection": "all_in_source_order",
        },
    )
    if tuple(payload["geometry_v7_pair_sources"]) != expected_geometry_sources:
        raise ValueError("blind development Geometry-V7 exclusions differ")
    if payload["ordering"] != "seed_major_then_prompt_source_then_source_order":
        raise ValueError("blind development ordering differs")
    if tuple(payload["seeds"]) != _DEVELOPMENT_SEEDS:
        raise ValueError("blind development seed set differs")
    expected_sources = tuple(
        {"expected_count": 32, "source_id": source_id, "path": path}
        for source_id, path in _PROMPT_SOURCES
    )
    if tuple(payload["prompt_sources"]) != expected_sources:
        raise ValueError("blind development prompt sources differ")

    prompt_rows: list[tuple[str, int, dict[str, Any]]] = []
    prompt_identities: set[tuple[str, str]] = set()
    prompt_texts: set[str] = set()
    for source_id, relative_path in _PROMPT_SOURCES:
        rows = _read_jsonl(root / relative_path)
        if len(rows) != 32:
            raise ValueError(f"blind prompt source {source_id} must contain exactly 32 rows")
        for ordinal, row in enumerate(rows, 1):
            required_row = {"height", "prompt", "seed", "source_id", "split", "unit_id", "width"}
            if set(row) != required_row:
                raise ValueError(f"blind prompt source {source_id} row fields differ")
            if (
                not isinstance(row["prompt"], str)
                or not row["prompt"].strip()
                or not isinstance(row["source_id"], str)
                or not row["source_id"]
                or row["height"] != 512
                or row["width"] != 512
            ):
                raise ValueError(f"blind prompt source {source_id} row {ordinal} differs")
            identity = (source_id, row["source_id"])
            if identity in prompt_identities or row["prompt"] in prompt_texts:
                raise ValueError("blind development prompt identities must be unique")
            prompt_identities.add(identity)
            prompt_texts.add(row["prompt"])
            prompt_rows.append((source_id, ordinal, row))

    expanded: list[BlindGenerationUnit] = []
    for seed_index, seed in enumerate(_DEVELOPMENT_SEEDS, 1):
        for source_id, source_ordinal, row in prompt_rows:
            stratum = f"seed_{seed_index:02d}__{source_id}"
            base_id = f"{source_id}:{row['source_id']}:seed={seed}"
            unit_id = f"blind-dev-{seed_index:02d}-{source_id}-{source_ordinal:04d}"
            image_ref = f"generated://{base_id}"
            expanded.append(
                BlindGenerationUnit(
                    BlindCalibrationUnit(unit_id, stratum, image_ref, base_id),
                    row["prompt"], seed, 512, 512,
                )
            )
    roster = BlindCalibrationRoster(
        tuple(unit.calibration_unit for unit in expanded),
        tuple(payload["disjoint_from"]),
    )
    if len(expanded) != BLIND_DEV_DENOMINATOR:
        raise RuntimeError("blind development roster expansion differs")
    if len({unit.calibration_unit.image_ref for unit in expanded}) != BLIND_DEV_DENOMINATOR:
        raise ValueError("blind development logical image references must be unique")
    occupied_geometry_pairs: set[tuple[str, int]] = set()
    for descriptor in expected_geometry_sources:
        rows = _read_jsonl(root / descriptor["path"])
        selected = rows[:4] if descriptor["selection"] == "first_4_in_source_order" else rows
        for row in selected:
            prompt = row.get("prompt")
            seed = row.get("seed")
            if not isinstance(prompt, str) or not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError("Geometry-V7 exclusion pair fields differ")
            occupied_geometry_pairs.add((prompt, seed))
    development_pairs = {(unit.prompt, unit.seed) for unit in expanded}
    if len(development_pairs) != BLIND_DEV_DENOMINATOR:
        raise ValueError("blind development prompt-seed pairs must be unique")
    if development_pairs & occupied_geometry_pairs:
        raise ValueError("blind development overlaps Geometry-V7 prompt-seed pairs")
    summary = {
        "denominator": BLIND_DEV_DENOMINATOR,
        "dimensions": dict(payload["dimensions"]),
        "disjoint_from": list(payload["disjoint_from"]),
        "exclusive_reservation": payload["exclusive_reservation"],
        "geometry_v7_excluded_pair_count": len(occupied_geometry_pairs),
        "ordering": payload["ordering"],
        "prompt_sources": list(payload["prompt_sources"]),
        "seeds": list(payload["seeds"]),
        "source_strata": dict(
            sorted(Counter(unit.calibration_unit.source_stratum for unit in expanded).items())
        ),
    }
    return roster, tuple(expanded), summary


def load_roster(repo_root: str | Path = REPO_ROOT) -> BlindCalibrationRoster:
    return load_roster_inputs(repo_root)[0]


def load_runtime_config(repo_root: str | Path = REPO_ROOT) -> dict[str, str]:
    payload = _read_json(Path(repo_root) / COLAB_ASSETS_REPO_PATH)
    required = {
        "content_model_id",
        "device",
        "syncseal_filename",
        "syncseal_url",
        "weighted_joint_asset_path",
        "weighted_joint_asset_producer_exact",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("blind calibration asset configuration fields differ")
    expected = {
        "content_model_id": CONTENT_MODEL_ID,
        "device": "cuda",
        "syncseal_filename": "syncmodel.jit.pt",
        "syncseal_url": SYNCSEAL_TORCHSCRIPT_URL,
        "weighted_joint_asset_path": str(WEIGHTED_ASSET_REPO_PATH),
        "weighted_joint_asset_producer_exact": WEIGHTED_ASSET_PRODUCER_EXACT,
    }
    if payload != expected:
        raise ValueError("blind calibration asset configuration differs")
    return dict(payload)


def load_callback_n4_config(
    repo_root: str | Path = REPO_ROOT,
) -> tuple[dict[str, Any], tuple[CallbackN4Unit, ...]]:
    """Load the committed four-case smoke roster before any model or score."""

    root = Path(repo_root)
    payload = _read_json(root / CALLBACK_N4_CONFIG_REPO_PATH)
    required = {
        "attack_condition_id",
        "attack_condition_source",
        "cases",
        "denominator",
        "residual_strength_multiplier",
        "schema_version",
        "science_denominator",
        "tau_blind_be_hex",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("callback N=4 configuration fields differ")
    expected_header = {
        "attack_condition_id": "core_rotation_pos15",
        "attack_condition_source": "src/cegwm/geometry_v7/r1a.py:R1A_CORE_CONDITIONS",
        "denominator": 4,
        "residual_strength_multiplier": 0.75,
        "schema_version": "cegwm_blind_detection_v1_callback_n4_config_v1",
        "science_denominator": 0,
        "tau_blind_be_hex": CALLBACK_N4_TAU_HEX,
    }
    if any(payload.get(name) != value for name, value in expected_header.items()):
        raise ValueError("callback N=4 frozen configuration differs")
    if encode_binary64(CALLBACK_N4_TAU, "callback_tau") != CALLBACK_N4_TAU_HEX:
        raise RuntimeError("callback N=4 frozen tau literal differs")
    attack_spec = condition_by_id(payload["attack_condition_id"])
    if attack_spec.condition_id != "core_rotation_pos15":
        raise ValueError("callback N=4 attack condition differs")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != 4:
        raise ValueError("callback N=4 requires exactly four committed cases")
    required_case = {
        "attacked",
        "case_id",
        "coverage",
        "height",
        "prompt",
        "prompt_source_path",
        "prompt_source_unit_id",
        "seed",
        "source_stratum",
        "watermarked",
        "width",
    }
    units: list[CallbackN4Unit] = []
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict) or set(raw) != required_case:
            raise ValueError(f"callback N=4 case[{index}] fields differ")
        if raw["coverage"] != CALLBACK_N4_COVERAGE[index]:
            raise ValueError("callback N=4 coverage order differs")
        if raw["case_id"] != f"blind-callback-n4-{index + 1:02d}":
            raise ValueError("callback N=4 case identity differs")
        if raw["height"] != 512 or raw["width"] != 512:
            raise ValueError("callback N=4 dimensions differ")
        if raw["prompt_source_path"] != _PROMPT_SOURCES[0][1]:
            raise ValueError("callback N=4 prompt source differs")
        if raw["prompt_source_unit_id"] != f"content-v6-iss-dev-{index + 1:04d}":
            raise ValueError("callback N=4 prompt-source identity differs")
        expected_stratum = f"content_v6_iss_development_v1__{raw['coverage']}"
        if raw["source_stratum"] != expected_stratum:
            raise ValueError("callback N=4 source stratum differs")
        expected_flags = (
            (True, False),
            (True, True),
            (False, False),
            (False, True),
        )[index]
        if (raw["watermarked"], raw["attacked"]) != expected_flags:
            raise ValueError("callback N=4 generation/attack layout differs")
        if raw["seed"] != _DEVELOPMENT_SEEDS[index]:
            raise ValueError("callback N=4 seed order differs")
        units.append(CallbackN4Unit(**raw))
    frozen_source = {
        row["unit_id"]: row for row in _read_jsonl(root / _PROMPT_SOURCES[0][1])
    }
    for unit in units:
        source = frozen_source.get(unit.prompt_source_unit_id)
        if source is None or source.get("prompt") != unit.prompt:
            raise ValueError("callback N=4 prompt does not match its frozen Git source")
    if len({unit.case_id for unit in units}) != 4:
        raise ValueError("callback N=4 case identities must be unique")
    if len({unit.prompt for unit in units}) != 4:
        raise ValueError("callback N=4 prompts must be distinct")
    if len({unit.seed for unit in units}) != 4:
        raise ValueError("callback N=4 seeds must be distinct")
    if len({(unit.prompt, unit.seed) for unit in units}) != 4:
        raise ValueError("callback N=4 prompt-seed units must be distinct")
    return dict(payload), tuple(units)


def load_weighted_asset_semantic(repo_root: str | Path) -> WeightedJointAsset:
    """Load the Git-bound weighted asset without a sidecar/hash integrity gate."""

    payload = _read_json(Path(repo_root) / WEIGHTED_ASSET_REPO_PATH)
    if not isinstance(payload, dict):
        raise ValueError("weighted-joint asset must be a JSON object")
    expected_semantics = {
        "asset_role_id": "content_v9_calibrated_weighted_joint_v1",
        "hf_scorer_id": HF_SCORER_ID,
        "hf_weight_be_hex": "3fe8000000000000",
        "joint_formula": (
            "J=(0.25*z_lf+0.75*z_hf)/sqrt(0.25^2+0.75^2+2*0.25*0.75*rho)"
        ),
        "lf_scorer_id": "content_v4_whitened_lf_dct_matched_cosine_v1",
        "lf_weight_be_hex": "3fd0000000000000",
        "method_id": "content_v9_v6_calibrated_weighted_joint_v1",
        "producer_exact": WEIGHTED_ASSET_PRODUCER_EXACT,
        "schema_version": "cegwm_content_v9_calibrated_weighted_joint_asset_v1",
        "statistic_id": "binary64_fsum_mean_ddof1_std_pearson_rho_v1",
        "value_dtype": "IEEE-754_binary64_big_endian_hex",
    }
    if any(payload.get(name) != value for name, value in expected_semantics.items()):
        raise ValueError("weighted-joint semantic identity differs")
    asset = WeightedJointAsset(dict(payload), weighted_stable_json_bytes(payload))
    fit = asset.fit
    if not all(math.isfinite(value) for value in (
        fit.mu_lf, fit.sigma_lf, fit.mu_hf, fit.sigma_hf, fit.rho,
    )) or fit.sigma_lf <= 0.0 or fit.sigma_hf <= 0.0 or not -1.0 <= fit.rho <= 1.0:
        raise ValueError("weighted-joint calibration fit differs")
    return asset


def load_whitening_asset_semantic(repo_root: str | Path) -> WhiteningAsset:
    """Parse only the computation-bearing semantics of the Git whitening JSON."""

    payload = _read_json(Path(repo_root) / WHITENING_ASSET_REPO_PATH)
    if not isinstance(payload, dict):
        raise ValueError("content-whitening asset must be a JSON object")
    expected_semantics = {
        "observation_contract_id": WHITENING_OBSERVATION_CONTRACT_ID,
        "schema_version": WHITENING_ASSET_SCHEMA_ID,
        "whitening_order": WHITENING_ORDER,
        "whitening_shape": list(WHITENING_SHAPE),
    }
    if any(payload.get(name) != value for name, value in expected_semantics.items()):
        raise ValueError("content-whitening computation semantics differ")
    normalized = {
        "fit_sample_count": WHITENING_FIT_UNIT_COUNT,
        "observation_contract_id": payload["observation_contract_id"],
        "producer_exact": WHITENING_ASSET_PRODUCER_EXACT,
        "schema_version": payload["schema_version"],
        "whitening_order": payload["whitening_order"],
        "whitening_shape": payload["whitening_shape"],
        "whitening_words_be_hex": payload.get("whitening_words_be_hex"),
    }
    return WhiteningAsset(normalized, whitening_stable_json_bytes(normalized))


def load_iss_asset_semantic(repo_root: str | Path) -> ISSAsset:
    """Parse the ISS controller without using historical integrity metadata."""

    payload = _read_json(Path(repo_root) / ISS_ASSET_REPO_PATH)
    if not isinstance(payload, dict):
        raise ValueError("content ISS asset must be a JSON object")
    expected_semantics = {
        "asset_role_id": ISS_ASSET_ROLE_ID,
        "controller_formula": "beta_equals_clamp_total_multiplier_of_(m-h)/g_inclusive_1_to_2",
        "schema_version": ISS_ASSET_SCHEMA_ID,
        "v4_lf_scorer_id": CONTENT_WHITENING_LF_SCORER_ID,
    }
    if any(payload.get(name) != value for name, value in expected_semantics.items()):
        raise ValueError("content ISS controller semantics differ")
    beta = decode_binary64(payload.get("beta_development_be_hex"), "beta_development")
    margin = decode_binary64(payload.get("margin_delta_be_hex"), "margin_delta")
    gain = decode_binary64(payload.get("gain_g_be_hex"), "gain_g")
    target = decode_binary64(payload.get("target_m_be_hex"), "target_m")
    if beta != 1.0 or margin <= 0.0 or gain <= 0.0 or not -1.0 < target < 1.0:
        raise ValueError("content ISS controller numeric domain differs")
    normalized = {
        "gain_g_be_hex": payload["gain_g_be_hex"],
        "target_m_be_hex": payload["target_m_be_hex"],
    }
    return ISSAsset(normalized, iss_stable_json_bytes(normalized))


def build_production_runtime(
    repo_root: Path,
    config: dict[str, str],
    *,
    hf_token: str,
    runtime_root: Path,
) -> tuple[Any, BlindProductionAssets]:
    """Construct the real typed detector and fresh official SyncSeal download."""

    if not isinstance(hf_token, str) or not hf_token.strip():
        raise RuntimeError("HF_TOKEN is required to load public content assets")
    from experiments import content_unweighted_engine

    pipeline, embed_assets = content_unweighted_engine._load_pipeline_and_assets(
        config["content_model_id"], hf_token
    )
    whitening_asset = load_whitening_asset_semantic(repo_root)
    lf_public_assets = FrozenContentWhiteningLFPublicAssets(
        embed_assets.lf_public_assets, whitening_asset
    )
    evaluation_assets = ContentISSEvaluationAssets(
        embed_assets, lf_public_assets, load_iss_asset_semantic(repo_root)
    )
    content_assets = ContentCalibrationAssets(evaluation_assets)
    weighted_asset = load_weighted_asset_semantic(repo_root)
    checkpoint = runtime_root / config["syncseal_filename"]
    download_official_syncseal_torchscript(checkpoint)
    if not checkpoint.is_file():
        raise RuntimeError("official SyncSeal checkpoint download did not produce a file")
    geometry = SyncSealTorchScript.from_file(checkpoint, device=config["device"])
    return pipeline, BlindProductionAssets(content_assets, weighted_asset, geometry)


def load_repository_threshold_asset(
    repo_root: str | Path = REPO_ROOT,
) -> BlindThresholdAsset:
    """Load the landed engineering threshold by repository path and semantics."""

    asset = load_threshold_asset(Path(repo_root) / REPOSITORY_THRESHOLD_REPO_PATH)
    expected = {
        "decision_rule": "positive_iff_m_strictly_greater_than_tau_blind",
        "denominator": BLIND_DEV_DENOMINATOR,
        "producer_exact": THRESHOLD_CALIBRATION_PRODUCER_EXACT,
        "production_runtime_id": BLIND_PRODUCTION_RUNTIME_ID,
        "replay_false_positives": 0,
        "replay_kind": "fresh_full_system_pre_geometry_post_same_roster_images_key_runtime",
        "schema_version": BLIND_THRESHOLD_SCHEMA_ID,
        "statistic_id": BLIND_STATISTIC_ID,
        "tau_blind_be_hex": FROZEN_TAU_BLIND_HEX,
    }
    if any(asset.payload.get(name) != value for name, value in expected.items()):
        raise ValueError("repository blind threshold method semantics differ")
    if asset.tau_blind != FROZEN_TAU_BLIND:
        raise ValueError("repository blind threshold binary64 value differs")
    replay = asset.payload["full_system_replay_rows"]
    if any(
        not isinstance(row, dict)
        or row.get("roster_index") != index
        or row.get("method_complete") is not True
        or row.get("operational_error") is not None
        or row.get("positive") is not False
        for index, row in enumerate(replay)
    ):
        raise ValueError("repository blind threshold replay semantics differ")
    return asset


def build_production_detection_runtime(
    repo_root: Path,
    config: dict[str, str],
    *,
    hf_token: str,
    runtime_root: Path,
) -> tuple[Any, BlindProductionAssets]:
    """Build the real production runtime with the landed repository threshold."""

    pipeline, base_assets = build_production_runtime(
        repo_root, config, hf_token=hf_token, runtime_root=runtime_root
    )
    threshold = load_repository_threshold_asset(repo_root)
    return pipeline, BlindProductionAssets(
        base_assets.content_assets,
        base_assets.weighted_joint_asset,
        base_assets.geometry_backend,
        threshold,
    )


class GenerationBlocked(RuntimeError):
    def __init__(self, records: Iterable[dict[str, Any]]) -> None:
        self.records = tuple(records)
        failures = sum(record["status"] != "GENERATED" for record in self.records)
        super().__init__(f"{failures}/256 ordinary RGB generations failed")


def generate_development_images(
    generation_units: Iterable[BlindGenerationUnit],
    pipeline: Any,
    *,
    device: str,
) -> tuple[dict[str, Image.Image], tuple[dict[str, Any], ...]]:
    """Generate every fixed unit once through callback-free ``run_sd35_plain``."""

    images: dict[str, Image.Image] = {}
    records: list[dict[str, Any]] = []
    units = tuple(generation_units)
    if len(units) != BLIND_DEV_DENOMINATOR:
        raise ValueError("blind generation requires exactly 256 fixed units")
    for index, unit in enumerate(units):
        record = {
            "error": None,
            "roster_index": index,
            "source_stratum": unit.calibration_unit.source_stratum,
            "status": "OPERATIONAL_BLOCKED",
            "unit_id": unit.calibration_unit.unit_id,
        }
        try:
            generator = torch.Generator(device=device).manual_seed(unit.seed)
            image = require_ordinary_rgb_image(
                run_sd35_plain(
                    pipeline,
                    unit.prompt,
                    height=unit.height,
                    width=unit.width,
                    generator=generator,
                )
            )
            images[unit.calibration_unit.image_ref] = image.copy()
            record["status"] = "GENERATED"
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
    if len(images) != BLIND_DEV_DENOMINATOR:
        raise GenerationBlocked(records)
    return images, tuple(records)


class ThresholdFreezeBlocked(RuntimeError):
    """Carry every attempted fixed-denominator row when threshold output is blocked."""

    def __init__(self, cause: Exception, calibration_rows, replay_rows) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.calibration_rows = tuple(calibration_rows)
        self.replay_rows = tuple(replay_rows)
        self.status = "METHOD_FAILED" if "0/256" in str(cause) else "OPERATIONAL_BLOCKED"


def evaluate_threshold_with_runtime(
    roster: BlindCalibrationRoster,
    key: bytes,
    public_assets: BlindProductionAssets,
    image_loader,
    *,
    producer_exact: str,
    replay_image_loader=None,
):
    """Evaluate fixed calibration and fresh replay without writing any artifact."""

    if type(public_assets) is not BlindProductionAssets:
        raise TypeError("threshold freeze requires BlindProductionAssets")
    if public_assets.threshold_asset is not None:
        raise ValueError("threshold calibration runtime must not contain an earlier threshold")
    rows = run_development_calibration(roster, key, public_assets, image_loader)
    try:
        tau_blind = candidate_tau_blind(rows, roster)
    except Exception as error:
        raise ThresholdFreezeBlocked(error, rows, ()) from error
    replay = run_development_full_system_replay(
        roster,
        key,
        public_assets,
        image_loader if replay_image_loader is None else replay_image_loader,
        tau_blind,
    )
    try:
        asset = build_threshold_asset(
            rows, roster, replay, producer_exact=producer_exact,
            calibration_key_digest=public_key_digest(key),
        )
    except Exception as error:
        raise ThresholdFreezeBlocked(error, rows, replay) from error
    return tuple(rows), tuple(replay), asset


def freeze_threshold_with_runtime(
    roster: BlindCalibrationRoster,
    key: bytes,
    public_assets: BlindProductionAssets,
    image_loader,
    output_path: str | Path,
    *,
    producer_exact: str,
) -> Path:
    """Run fresh calibration and full-system replay before create-only output."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError("blind threshold output is create-only")
    _, _, asset = evaluate_threshold_with_runtime(
        roster, key, public_assets, image_loader, producer_exact=producer_exact
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as sink:
        sink.write(asset.json_bytes)
    return output


def _verify_producer_checkout(producer_exact: str) -> None:
    if not isinstance(producer_exact, str) or len(producer_exact) != 40 or any(
        character not in "0123456789abcdef" for character in producer_exact
    ):
        raise ValueError("producer exact must be lowercase 40-hex")
    head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != producer_exact:
        raise RuntimeError("calibration checkout does not match producer exact")
    status = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True
    )
    if status:
        raise RuntimeError("calibration checkout must be clean")


def _calibration_rows_payload(rows) -> list[dict[str, Any]]:
    payload = []
    for row in rows:
        z_hex = None
        if row.method_complete and row.pre_score is not None:
            try:
                z_hex = encode_binary64(row.z, "z")
            except (TypeError, ValueError):
                z_hex = None
        payload.append(
            {
                "geometry_status": row.geometry_outcome,
                "method_complete": row.method_complete,
                "operational_error": row.operational_error,
                "post_m_be_hex": (
                    None if row.post_score is None else encode_binary64(row.post_score, "post_m")
                ),
                "pre_m_be_hex": (
                    None if row.pre_score is None else encode_binary64(row.pre_score, "pre_m")
                ),
                "roster_index": row.roster_index,
                "source_stratum": row.source_stratum,
                "unit_id": row.unit_id,
                "z_be_hex": z_hex,
            }
        )
    return payload


def _replay_rows_payload(rows) -> list[dict[str, Any]]:
    return [
        {
            "method_complete": row.method_complete,
            "operational_error": row.operational_error,
            "positive": row.positive,
            "post_m_be_hex": (
                None if row.post_score is None else encode_binary64(row.post_score, "post_m")
            ),
            "pre_m_be_hex": (
                None if row.pre_score is None else encode_binary64(row.pre_score, "pre_m")
            ),
            "recovered": row.recovered,
            "roster_index": row.roster_index,
            "route": row.route,
            "source_stratum": row.source_stratum,
            "unit_id": row.unit_id,
        }
        for row in rows
    ]


def _input_summary(
    roster: BlindCalibrationRoster,
    roster_summary: dict[str, Any],
    key: bytes,
) -> dict[str, Any]:
    return {
        "calibration_key_digest": public_key_digest(key),
        "disjoint_from": list(roster.disjoint_from),
        "exclusive_reservation": roster_summary["exclusive_reservation"],
        "ordering": roster_summary["ordering"],
        "prompt_sources": roster_summary["prompt_sources"],
        "roster_repo_path": str(ROSTER_REPO_PATH),
        "seeds": roster_summary["seeds"],
        "source_strata": roster_summary["source_strata"],
    }


def _config_summary(config: dict[str, str]) -> dict[str, Any]:
    return {
        "automatic_retries": 0,
        "content_model_id": config["content_model_id"],
        "decision_rule": "positive_iff_m_strictly_greater_than_tau_blind",
        "device": config["device"],
        "geometry_route": "Geometry-Direct_once_per_current_RGB",
        "preprocess_id": BLIND_PREPROCESS_ID,
        "production_runtime_id": BLIND_PRODUCTION_RUNTIME_ID,
        "scorer_id": BLIND_SCORER_ID,
        "statistic_id": BLIND_STATISTIC_ID,
        "syncseal_filename": config["syncseal_filename"],
        "syncseal_url": config["syncseal_url"],
        "weighted_asset_repo_path": str(WEIGHTED_ASSET_REPO_PATH),
        "weighted_asset_producer_exact": WEIGHTED_ASSET_PRODUCER_EXACT,
    }


def _base_calibration_result(*, producer_exact: str) -> dict[str, Any]:
    return {
        "calibration_rows": [],
        "candidate_tau_blind_be_hex": None,
        "claim_ceiling": CALIBRATION_CLAIM_CEILING,
        "config_summary": None,
        "denominator": BLIND_DEV_DENOMINATOR,
        "error": None,
        "fresh_replay_false_positives": None,
        "fresh_replay_rows": [],
        "fresh_replay_zero_of_256": False,
        "frozen_tau_blind_be_hex": None,
        "generation_records": [],
        "input_summary": None,
        "producer_exact": producer_exact,
        "schema_version": CALIBRATION_RESULT_SCHEMA,
        "science_denominator": 0,
        "stage": "preflight",
        "status": "OPERATIONAL_BLOCKED",
        "threshold_candidate_ready": False,
    }


def calibrate_and_record(
    runtime_root: str | Path,
    threshold_candidate_path: str | Path,
    result_output_path: str | Path,
    *,
    producer_exact: str,
) -> tuple[Path, Path | None, str]:
    """Run one formal calibration attempt and retain success or failure create-only."""

    threshold_candidate = Path(threshold_candidate_path)
    result_output = Path(result_output_path)
    if threshold_candidate.exists():
        raise FileExistsError("blind threshold candidate is create-only")
    if result_output.exists():
        raise FileExistsError("blind calibration result is create-only")
    runtime_work = Path(runtime_root)
    if runtime_work.exists():
        raise FileExistsError("blind calibration runtime root is create-only")
    result = _base_calibration_result(producer_exact=producer_exact)
    rows = ()
    replay = ()
    generation_records = ()
    root_key = os.environ.pop(ROOT_KEY_ENV, "")
    hf_token = os.environ.pop(HF_TOKEN_ENV, "")
    key = b""
    try:
        _verify_producer_checkout(producer_exact)
        result["stage"] = "committed_inputs"
        config = load_runtime_config(REPO_ROOT)
        result["config_summary"] = _config_summary(config)
        roster, generation_units, roster_summary = load_roster_inputs(REPO_ROOT)
        result["stage"] = "secret_binding"
        if not isinstance(root_key, str) or not root_key.strip():
            raise RuntimeError("CEG_WM_ROOT_KEY is required")
        if not isinstance(hf_token, str) or not hf_token.strip():
            raise RuntimeError("HF_TOKEN is required")
        key = normalize_detection_key(root_key)
        root_key = ""
        if public_key_digest(key) != CONTENT_CHAIN_PUBLIC_KEY_DIGEST:
            raise RuntimeError("content chain public key identity differs")
        result["input_summary"] = _input_summary(roster, roster_summary, key)
        result["stage"] = "runtime_construction"
        runtime_work.mkdir(parents=True, exist_ok=False)
        pipeline, public_assets = build_production_runtime(
            REPO_ROOT,
            config,
            hf_token=hf_token,
            runtime_root=runtime_work,
        )
        hf_token = ""
        result["stage"] = "ordinary_rgb_generation"
        cached_images, generation_records = generate_development_images(
            generation_units, pipeline, device=config["device"]
        )

        def calibration_image_loader(image_ref: str) -> Image.Image:
            return cached_images[image_ref].copy()

        result["stage"] = "calibration_and_replay"
        rows, replay, asset = evaluate_threshold_with_runtime(
            roster,
            key,
            public_assets,
            calibration_image_loader,
            producer_exact=producer_exact,
        )
        result["candidate_tau_blind_be_hex"] = asset.payload["tau_blind_be_hex"]
        result["frozen_tau_blind_be_hex"] = asset.payload["tau_blind_be_hex"]
        result["fresh_replay_false_positives"] = 0
        result["fresh_replay_zero_of_256"] = True
        result["stage"] = "threshold_candidate_write"
        threshold_candidate.parent.mkdir(parents=True, exist_ok=True)
        with threshold_candidate.open("xb") as sink:
            sink.write(asset.json_bytes)
        result["status"] = "CALIBRATION_COMPLETE_THRESHOLD_CANDIDATE_READY"
        result["stage"] = "terminal"
        result["threshold_candidate_ready"] = True
    except GenerationBlocked as blocked:
        generation_records = blocked.records
        result["status"] = "OPERATIONAL_BLOCKED"
        result["error"] = str(blocked)
    except ThresholdFreezeBlocked as blocked:
        rows = blocked.calibration_rows
        replay = blocked.replay_rows
        result["status"] = blocked.status
        result["error"] = str(blocked)
        if rows and all(row.method_complete for row in rows):
            try:
                result["candidate_tau_blind_be_hex"] = encode_binary64(
                    max(row.z for row in rows), "candidate_tau_blind"
                )
            except (TypeError, ValueError):
                pass
        if replay:
            result["fresh_replay_false_positives"] = sum(row.positive for row in replay)
    except Exception as error:
        result["status"] = "OPERATIONAL_BLOCKED"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        root_key = ""
        hf_token = ""
        key = b""
    result["calibration_rows"] = _calibration_rows_payload(rows)
    result["fresh_replay_rows"] = _replay_rows_payload(replay)
    result["generation_records"] = list(generation_records)
    result_output.parent.mkdir(parents=True, exist_ok=True)
    with result_output.open("xb") as sink:
        sink.write(stable_json_bytes(result))
    return (
        result_output,
        threshold_candidate if result["threshold_candidate_ready"] else None,
        result["status"],
    )


def _walk_weighted_scores(value: Any) -> Iterable[float]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "weighted_joint" and isinstance(child, (int, float)) and not isinstance(child, bool):
                scalar = float(child)
                if math.isfinite(scalar):
                    yield scalar
            yield from _walk_weighted_scores(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_weighted_scores(child)


def diagnose_existing_artifacts(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Read-only compact diagnostics; these are not calibration or paper evidence."""

    records = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            records.append({"path": str(path), "present": False})
            continue
        payload = _read_json(path)
        scores = tuple(_walk_weighted_scores(payload))
        records.append(
            {
                "path": str(path),
                "present": True,
                "status": payload.get("status") if isinstance(payload, dict) else None,
                "finite_weighted_joint_count": len(scores),
                "finite_weighted_joint_min": min(scores) if scores else None,
                "finite_weighted_joint_max": max(scores) if scores else None,
            }
        )
    return {
        "classification": "read_only_engineering_diagnostic_not_calibration_or_paper_evidence",
        "artifacts": records,
    }


def discover_callback_n4_threshold(
    calibration_runs_root: str | Path,
    detection_key: bytes,
) -> BlindThresholdAsset:
    """Find the unique successful threshold by ZIP member content semantics."""

    root = Path(calibration_runs_root)
    if not root.is_dir():
        raise FileNotFoundError("calibration-runs root is absent")
    candidates: list[BlindThresholdAsset] = []
    archives = tuple(sorted(root.glob("*.zip")))
    if not archives:
        raise FileNotFoundError("calibration-runs contains no terminal ZIP")
    for archive_path in archives:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            payloads: list[Any] = []
            for member in archive.infolist():
                if member.is_dir():
                    continue
                try:
                    payloads.append(json.loads(archive.read(member)))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
        completed = tuple(
            payload for payload in payloads
            if isinstance(payload, dict)
            and payload.get("status")
                == "CALIBRATION_COMPLETE_THRESHOLD_CANDIDATE_READY"
            and payload.get("threshold_candidate_ready") is True
            and payload.get("denominator") == BLIND_DEV_DENOMINATOR
            and payload.get("frozen_tau_blind_be_hex") == CALLBACK_N4_TAU_HEX
        )
        thresholds = []
        for payload in payloads:
            try:
                thresholds.append(
                    BlindThresholdAsset(payload, stable_json_bytes(payload))
                )
            except (TypeError, ValueError):
                continue
        if len(completed) == 1 and len(thresholds) == 1:
            threshold = thresholds[0]
            if (
                threshold.payload["schema_version"] == BLIND_THRESHOLD_SCHEMA_ID
                and threshold.payload["production_runtime_id"]
                    == BLIND_PRODUCTION_RUNTIME_ID
                and threshold.payload["denominator"] == BLIND_DEV_DENOMINATOR
                and threshold.payload["tau_blind_be_hex"] == CALLBACK_N4_TAU_HEX
                and threshold.tau_blind == CALLBACK_N4_TAU
                and threshold.payload["calibration_key_digest"]
                    == public_key_digest(detection_key)
            ):
                candidates.append(threshold)
    if len(candidates) != 1:
        raise RuntimeError(
            "callback N=4 requires exactly one semantically matching completed threshold"
        )
    return candidates[0]


class _CallbackN4ContentBackend:
    """Adapt the accepted paired content runtime to the public embed API."""

    def __init__(self, pipeline: Any, assets: ContentISSEvaluationAssets) -> None:
        self._pipeline = pipeline
        self._assets = assets

    def embed_content(
        self, request: CallbackN4EmbeddingRequest, detection_key: bytes
    ) -> Image.Image:
        if not isinstance(request, CallbackN4EmbeddingRequest):
            raise TypeError("callback N=4 content request differs")
        pair = run_content_iss_evaluation_pair(
            self._pipeline,
            request.prompt,
            detection_key,
            self._assets,
            height=request.height,
            width=request.width,
            seed=request.seed,
        )
        watermarked = require_ordinary_rgb_image(pair.image).copy()
        primary_null = pair.primary_null
        del primary_null
        del pair
        return watermarked


def _prepare_callback_n4_current_rgb(
    unit: CallbackN4Unit,
    pipeline: Any,
    detection_key: bytes,
    embedding_assets: BlindEmbeddingAssets,
    *,
    device: str,
) -> Image.Image:
    """Generate one predeclared case exactly once and return only current RGB."""

    if unit.watermarked:
        request = CallbackN4EmbeddingRequest(
            unit.prompt, unit.seed, unit.height, unit.width
        )
        try:
            generated = embed_watermark(request, detection_key, embedding_assets)
        except Exception as error:
            raise RuntimeError(f"embed:{type(error).__name__}: {error}") from error
        del request
    else:
        try:
            generator = torch.Generator(device=device).manual_seed(unit.seed)
            generated = run_sd35_plain(
                pipeline,
                unit.prompt,
                height=unit.height,
                width=unit.width,
                generator=generator,
            )
        except Exception as error:
            raise RuntimeError(f"generation:{type(error).__name__}: {error}") from error
        del generator
    current_rgb = require_ordinary_rgb_image(generated).copy()
    del generated
    if unit.attacked:
        try:
            attacked = render_r1a_attack(
                current_rgb, condition_by_id("core_rotation_pos15")
            )
        except Exception as error:
            raise RuntimeError(f"attack:{type(error).__name__}: {error}") from error
        del current_rgb
        current_rgb = require_ordinary_rgb_image(attacked).copy()
        del attacked
    return current_rgb


def detect_callback_n4_current_rgb(
    current_rgb: Image.Image,
    detection_key: bytes,
    public_assets: BlindProductionAssets,
):
    """Closure-free single-image handoff; no generation context crosses it."""

    return detect_watermark(current_rgb, detection_key, public_assets)


def _callback_n4_row_matches(coverage: str, record: Any) -> bool:
    if not record.method_complete or record.operational_error is not None:
        return False
    if coverage == "direct_negative":
        return record.positive is False
    expected = {
        "direct_positive": ("DIRECT_POSITIVE", True, False),
        "geometry_recovered_positive": ("GEOMETRY_RECOVERED", True, True),
        "geometry_recovered_negative": ("GEOMETRY_RECOVERED", False, True),
    }
    return (record.route, record.positive, record.recovered) == expected[coverage]


def _callback_n4_empty_row(index: int, coverage: str) -> dict[str, Any]:
    return {
        "case_id": f"blind-callback-n4-{index + 1:02d}",
        "coverage": coverage,
        "current_rgb_file": None,
        "method_complete": False,
        "operational_error": None,
        "positive": False,
        "post_m_be_hex": None,
        "pre_m_be_hex": None,
        "recovered": False,
        "route": "ERROR_FAIL_CLOSED",
        "source_stratum": None,
    }


def _write_callback_n4_current_rgb(
    output_dir: Path, index: int, case_id: str, image: Image.Image
) -> str:
    name = f"current_rgb_{index + 1:02d}_{case_id}.png"
    path = output_dir / name
    with path.open("xb") as sink:
        require_ordinary_rgb_image(image).save(sink, format="PNG")
    return name


def run_callback_n4(
    calibration_runs_root: str | Path,
    runtime_root: str | Path,
    current_rgb_output_dir: str | Path,
    result_output_path: str | Path,
    *,
    producer_exact: str,
) -> tuple[Path, str, tuple[str, ...]]:
    """Run each fixed real N=4 case once and retain every row."""

    result_output = Path(result_output_path)
    runtime_work = Path(runtime_root)
    image_output = Path(current_rgb_output_dir)
    if result_output.exists():
        raise FileExistsError("callback N=4 result is create-only")
    root_key = os.environ.pop(ROOT_KEY_ENV, "")
    hf_token = os.environ.pop(HF_TOKEN_ENV, "")
    detection_key = b""
    rows = [
        _callback_n4_empty_row(index, coverage)
        for index, coverage in enumerate(CALLBACK_N4_COVERAGE)
    ]
    result: dict[str, Any] = {
        "automatic_retries": 0,
        "claim_ceiling": "engineering_real_single_image_callback_n4_only",
        "denominator": 4,
        "error": None,
        "mismatched_case_ids": [],
        "operational_case_ids": [],
        "producer_exact": producer_exact,
        "records": rows,
        "schema_version": CALLBACK_N4_RESULT_SCHEMA,
        "science_denominator": 0,
        "stage": "preflight",
        "status": "OPERATIONAL_BLOCKED",
        "tau_blind_be_hex": CALLBACK_N4_TAU_HEX,
    }
    try:
        _verify_producer_checkout(producer_exact)
        if runtime_work.exists():
            raise FileExistsError("callback N=4 runtime root is create-only")
        if image_output.exists():
            raise FileExistsError("callback N=4 current-RGB output is create-only")
        config, units = load_callback_n4_config(REPO_ROOT)
        if not isinstance(root_key, str) or not root_key.strip():
            raise RuntimeError("CEG_WM_ROOT_KEY is required")
        if not isinstance(hf_token, str) or not hf_token.strip():
            raise RuntimeError("HF_TOKEN is required")
        detection_key = normalize_detection_key(root_key)
        root_key = ""
        if public_key_digest(detection_key) != CONTENT_CHAIN_PUBLIC_KEY_DIGEST:
            raise RuntimeError("content chain public key identity differs")
        result["stage"] = "threshold_selection"
        threshold = discover_callback_n4_threshold(
            calibration_runs_root, detection_key
        )
        result["stage"] = "runtime_construction"
        runtime_work.mkdir(parents=True, exist_ok=False)
        runtime_config = load_runtime_config(REPO_ROOT)
        pipeline, base_assets = build_production_runtime(
            REPO_ROOT,
            runtime_config,
            hf_token=hf_token,
            runtime_root=runtime_work,
        )
        hf_token = ""
        public_assets = BlindProductionAssets(
            base_assets.content_assets,
            base_assets.weighted_joint_asset,
            base_assets.geometry_backend,
            threshold,
        )
        embedding_assets = BlindEmbeddingAssets(
            _CallbackN4ContentBackend(
                pipeline, base_assets.content_assets.iss_assets
            ),
            base_assets.geometry_backend,
            config["residual_strength_multiplier"],
        )
        image_output.mkdir(parents=True, exist_ok=False)
        current_images: list[Image.Image | None] = [None, None, None, None]
        result["stage"] = "fixed_four_preparation"
        for index in range(4):
            unit = units[index]
            row = rows[index]
            row["source_stratum"] = unit.source_stratum
            try:
                current_rgb = _prepare_callback_n4_current_rgb(
                    unit,
                    pipeline,
                    detection_key,
                    embedding_assets,
                    device=runtime_config["device"],
                )
                case_id = unit.case_id
                try:
                    row["current_rgb_file"] = _write_callback_n4_current_rgb(
                        image_output, index, case_id, current_rgb
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"current_rgb_io:{type(error).__name__}: {error}"
                    ) from error
                current_images[index] = current_rgb
            except Exception as error:
                row["operational_error"] = f"{type(error).__name__}: {error}"
        del units
        del config
        del pipeline
        del embedding_assets
        del base_assets
        del runtime_config
        del threshold
        unit = None
        current_rgb = None
        case_id = None
        result["stage"] = "fixed_four_detection"
        for index in range(4):
            row = rows[index]
            current_rgb = current_images[index]
            if current_rgb is None:
                continue
            try:
                try:
                    record = detect_callback_n4_current_rgb(
                        current_rgb, detection_key, public_assets
                    )
                except Exception as error:
                    raise RuntimeError(f"detect:{type(error).__name__}: {error}") from error
                coverage = row["coverage"]
                row.update(
                    {
                        "method_complete": record.method_complete,
                        "operational_error": record.operational_error,
                        "positive": record.positive,
                        "post_m_be_hex": (
                            None
                            if record.post is None
                            else encode_binary64(record.post.value, "post_m")
                        ),
                        "pre_m_be_hex": (
                            None
                            if record.pre is None
                            else encode_binary64(record.pre.value, "pre_m")
                        ),
                        "recovered": record.recovered,
                        "route": record.route,
                    }
                )
                if not _callback_n4_row_matches(coverage, record):
                    if record.method_complete and record.operational_error is None:
                        result["mismatched_case_ids"].append(row["case_id"])
            except Exception as error:
                row["operational_error"] = f"{type(error).__name__}: {error}"
            finally:
                current_images[index] = None
                del current_rgb
        operational = [
            row["case_id"]
            for row in rows
            if not row["method_complete"] or row["operational_error"] is not None
        ]
        result["operational_case_ids"] = operational
        result["status"] = (
            "OPERATIONAL_BLOCKED"
            if operational
            else "METHOD_FAILED"
            if result["mismatched_case_ids"]
            else "CALLBACK_N4_PASSED"
        )
        result["stage"] = "terminal"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["operational_case_ids"] = [row["case_id"] for row in rows]
        for row in rows:
            row["operational_error"] = result["error"]
    finally:
        root_key = ""
        hf_token = ""
        detection_key = b""
    result_output.parent.mkdir(parents=True, exist_ok=True)
    with result_output.open("xb") as sink:
        sink.write(stable_json_bytes(result))
    return result_output, result["status"], tuple(result["mismatched_case_ids"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("calibrate-and-freeze")
    freeze.add_argument("--producer-exact", required=True)
    freeze.add_argument("--runtime-root", required=True)
    freeze.add_argument("--candidate-output", required=True)
    freeze.add_argument("--result-output", required=True)
    diagnose = sub.add_parser("diagnose-existing")
    diagnose.add_argument("artifacts", nargs="+")
    callback = sub.add_parser("callback-n4")
    callback.add_argument("--producer-exact", required=True)
    callback.add_argument("--calibration-runs-root", required=True)
    callback.add_argument("--runtime-root", required=True)
    callback.add_argument("--current-rgb-output-dir", required=True)
    callback.add_argument("--result-output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "calibrate-and-freeze":
        result, threshold, status = calibrate_and_record(
            args.runtime_root,
            args.candidate_output,
            args.result_output,
            producer_exact=args.producer_exact,
        )
        print(
            "CEGWM_BLIND_DETECTION_V1 "
            + stable_json_bytes(
                {
                    "denominator": 256,
                    "disjoint_from": list(BLIND_DEV_DISJOINT_FROM),
                    "result_output": str(result),
                    "status": status,
                    "threshold_candidate": None if threshold is None else str(threshold),
                }
            ).decode("ascii")
        )
        return 0 if threshold is not None else 2 if status == "METHOD_FAILED" else 3
    if args.command == "callback-n4":
        output, status, mismatches = run_callback_n4(
            args.calibration_runs_root,
            args.runtime_root,
            args.current_rgb_output_dir,
            args.result_output,
            producer_exact=args.producer_exact,
        )
        print(
            "CEGWM_BLIND_DETECTION_V1 "
            + stable_json_bytes(
                {
                    "denominator": 4,
                    "mismatched_case_ids": list(mismatches),
                    "output": str(output),
                    "status": status,
                }
            ).decode("ascii")
        )
        return 0 if status == "CALLBACK_N4_PASSED" else 2 if status == "METHOD_FAILED" else 3
    diagnostic = diagnose_existing_artifacts(args.artifacts)
    print("CEGWM_BLIND_DETECTION_V1 " + stable_json_bytes(diagnostic).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
