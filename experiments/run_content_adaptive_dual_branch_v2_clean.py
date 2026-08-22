"""Real clean Stage-A runner for v2 content-adaptive dual-branch evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping
import zipfile

import numpy as np
import torch

from cegwm.method.content_adaptive_v2 import (
    COUNTERFACTUAL_EFFECT_FIELDS,
    JOINT_EVALUATED_CANDIDATE_ID,
)
from cegwm.method.hf import FrozenHFPublicAssets, score_hf_image
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
    score_lf_image,
)
from cegwm.protocol.content_chain_v2 import (
    ContentChainProtocol,
    load_content_adaptive_dual_branch_v2_clean_protocol,
)
from cegwm.runtime.content_adaptive_sd35_v2 import (
    ContentEmbedAssets,
    load_dino_content_assets,
    run_sd35_content_adaptive,
)
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline, run_sd35_plain
from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from cegwm.shared.prg import prg_bytes

KEY_ENV = "CEG_WM_ROOT_KEY"
TOKEN_ENV = "HF_TOKEN"
EXECUTION_SCOPE_ID = "content_adaptive_dual_branch_v2_semantic_gate_engineering_and_stage_a_evaluation_v1"
COMPLETE_EXECUTION = "complete_for_content_adaptive_dual_branch_v2_semantic_gate_evaluation"
INCOMPLETE_EXECUTION = "incomplete_operational_execution"
ARMS = (JOINT_EVALUATED_CANDIDATE_ID, f"primary_null__{JOINT_EVALUATED_CANDIDATE_ID}")
BRANCHES = ("lf", "hf", "joint")
RECORD_CONTRACT_ID = "content_adaptive_dual_branch_v2_semantic_gate_record_v1"
RECORD_FIELDS = (
    "run_id", "unit_id", "source_cluster_id", "arm", "condition",
    "code_revision", "config_digest", "key_public_digest", "status",
    "failure_reason", "scores", "metrics", "record_contract_id",
)
_SCORE_FIELDS = tuple(
    f"{branch}__{label}"
    for branch in BRANCHES
    for label in ("registered", *(f"wrong_{index:02d}" for index in range(16)))
)
_CANDIDATE_METRIC_FIELDS = (
    "combined_relative_l2", "lf_effective_relative_l2", "hf_effective_relative_l2",
    "lf_branch_share", "hf_branch_share", *COUNTERFACTUAL_EFFECT_FIELDS,
    "minimum_counterfactual_effect", "probe_evaluation_count", "paired_rgb_psnr_db",
)
_NULL_METRIC_FIELDS = ("paired_rgb_psnr_db",)
_PUBLIC_OPERATIONAL_ERROR_CLASSES = (
    "FileNotFoundError", "ImportError", "MemoryError", "OSError",
    "OutOfMemoryError", "RuntimeError", "TimeoutError", "TypeError", "ValueError",
    "OtherOperationalError",
)
STATE_SCHEMA_ID = "content_adaptive_dual_branch_v2_resumable_state_v1"
CHECKPOINT_INTERVAL_HOURS = 2.0
FIXED_UNIT_COUNT = 8
RECORDS_PER_UNIT = 2
FIXED_RECORD_COUNT = 16
_STATE_FIELDS = (
    "state_schema_id", "identity", "checkpoint_sequence",
    "checkpoint_time_anchor_unix_seconds", "committed_unit_count", "records",
)
_IDENTITY_FIELDS = (
    "run_id", "exact", "execution_scope_id", "protocol_id", "protocol_digest",
    "public_key_digest", "model_id", "ordered_roster", "ordered_arms",
    "record_contract_id", "fixed_unit_count", "records_per_unit",
    "fixed_record_count", "checkpoint_interval_hours",
)


def _finite_real(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validate_content_v2_record(record: Mapping[str, Any]) -> None:
    """Validate the narrow public JSON contract owned only by this v2 runner."""

    if tuple(record) != RECORD_FIELDS:
        raise ValueError("content v2 record fields or order differ from the frozen contract")
    if record["record_contract_id"] != RECORD_CONTRACT_ID:
        raise ValueError("content v2 record contract identity differs")
    for name in (
        "run_id", "unit_id", "source_cluster_id", "arm", "condition",
        "code_revision", "config_digest", "key_public_digest", "status",
    ):
        if not isinstance(record[name], str) or not record[name].strip():
            raise ValueError(f"content v2 record {name} must be nonempty text")
    if record["arm"] not in ARMS or record["condition"] != "clean":
        raise ValueError("content v2 record arm or condition differs")
    if re.fullmatch(r"[0-9a-f]{40}", record["code_revision"]) is None:
        raise ValueError("content v2 record revision must be an exact lowercase commit")
    if any(
        re.fullmatch(r"[0-9a-f]{64}", record[name]) is None
        for name in ("config_digest", "key_public_digest")
    ):
        raise ValueError("content v2 record digest identity differs")
    status = record["status"]
    failure_reason = record["failure_reason"]
    scores = record["scores"]
    metrics = record["metrics"]
    if not isinstance(scores, dict) or not isinstance(metrics, dict):
        raise TypeError("content v2 record scores and metrics must be plain JSON objects")
    if status == "operational_failure":
        if failure_reason not in _PUBLIC_OPERATIONAL_ERROR_CLASSES:
            raise ValueError("operational failure requires one finite public error class")
        if scores or metrics:
            raise ValueError("operational failure scores and metrics must be empty")
        return
    if status != "success" or failure_reason is not None:
        raise ValueError("successful records require null failure_reason")
    if tuple(scores) != _SCORE_FIELDS:
        raise ValueError("successful record scores differ from the exact 3-by-17 fields")
    if any(not -1.0 <= _finite_real(value, name) <= 1.0 for name, value in scores.items()):
        raise ValueError("successful record scores must lie in [-1, 1]")
    expected_metrics = (
        _CANDIDATE_METRIC_FIELDS if record["arm"] == ARMS[0] else _NULL_METRIC_FIELDS
    )
    if set(metrics) != set(expected_metrics):
        raise ValueError("successful record metrics differ from the exact arm contract")
    finite_metrics = {name: _finite_real(metrics[name], name) for name in expected_metrics}
    if finite_metrics["paired_rgb_psnr_db"] < 0.0:
        raise ValueError("paired_rgb_psnr_db must be nonnegative")
    if record["arm"] == ARMS[1]:
        return
    if not 0.0 <= finite_metrics["combined_relative_l2"] <= 0.012:
        raise ValueError("combined_relative_l2 escaped the frozen budget")
    for name in ("lf_effective_relative_l2", "hf_effective_relative_l2"):
        if not 0.0 < finite_metrics[name] <= 0.012:
            raise ValueError(f"{name} must be positive within the frozen budget")
    lf_share = finite_metrics["lf_branch_share"]
    hf_share = finite_metrics["hf_branch_share"]
    if not (
        0.0 < lf_share < 1.0
        and 0.0 < hf_share < 1.0
        and math.isclose(lf_share + hf_share, 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("recorded public branch shares are invalid")
    effects = [finite_metrics[name] for name in COUNTERFACTUAL_EFFECT_FIELDS]
    if any(value < 0.0 for value in effects):
        raise ValueError("recorded counterfactual effects must be nonnegative")
    if finite_metrics["minimum_counterfactual_effect"] != min(effects):
        raise ValueError("recorded minimum counterfactual effect differs")
    if finite_metrics["probe_evaluation_count"] != 64.0:
        raise ValueError("recorded probe evaluation count must be exactly 64")


def _content_v2_record(
    *,
    run_id: str,
    unit_id: str,
    source_cluster_id: str,
    arm: str,
    condition: str,
    code_revision: str,
    config_digest: str,
    key_public_digest: str,
    status: str,
    failure_reason: str | None = None,
    scores: Mapping[str, float] | None = None,
    metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    record = {
        "run_id": run_id,
        "unit_id": unit_id,
        "source_cluster_id": source_cluster_id,
        "arm": arm,
        "condition": condition,
        "code_revision": code_revision,
        "config_digest": config_digest,
        "key_public_digest": key_public_digest,
        "status": status,
        "failure_reason": failure_reason,
        "scores": dict(scores or {}),
        "metrics": dict(metrics or {}),
        "record_contract_id": RECORD_CONTRACT_ID,
    }
    _validate_content_v2_record(record)
    return record


def _public_operational_error_class(error: Exception) -> str:
    name = type(error).__name__
    return name if name in _PUBLIC_OPERATIONAL_ERROR_CLASSES else "OtherOperationalError"


def _git_exact(repo_root: Path, expected_exact: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected_exact) is None:
        raise ValueError("expected exact must be a lowercase 40-character revision")
    exact = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if exact != expected_exact:
        raise RuntimeError("resolved revision differs from approved execution exact")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout
    if status:
        raise RuntimeError("execution checkout must be clean")
    return exact


def _load_protocol(repo_root: Path) -> ContentChainProtocol:
    root = repo_root / "configs" / "content_chain"
    return load_content_adaptive_dual_branch_v2_clean_protocol(
        root / "content_adaptive_dual_branch_v2_clean_v1.json",
        root / "content_adaptive_dual_branch_v2_clean.jsonl",
    )


def _load_pipeline_and_assets(model_id: str, token: str) -> tuple[Any, ContentEmbedAssets]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_real_content_adaptive_execution")
    pipeline = load_sd35_pipeline(model_id, torch_dtype=torch.float16, token=token)
    pipeline.to("cuda")
    vae = getattr(pipeline, "vae", None)
    processor = getattr(pipeline, "image_processor", None)
    hf = FrozenHFPublicAssets(
        vae=vae,
        image_processor=processor,
        image_processor_id=f"{model_id}:image_processor",
    )
    lf = FrozenLFPublicAssets(
        vae=vae,
        image_processor=processor,
        image_processor_id=f"{model_id}:image_processor",
        candidate_id=LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        detector_statistic_id=LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        evaluated_candidate_id=LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    dino_model, dino_processor = load_dino_content_assets(token=token)
    dino_model.to("cuda")
    dino_model.eval()
    return pipeline, ContentEmbedAssets(dino_model, dino_processor, hf, lf)


def _wrong_keys(key: bytes, protocol: ContentChainProtocol) -> tuple[bytes, ...]:
    keying = protocol.config["keying"]
    return tuple(
        prg_bytes(key, f"{keying['wrong_key_derivation_domain']}/index={index}", 32)
        for index in range(keying["wrong_key_count"])
    )


def _blind_scores(
    image: Any,
    key: bytes,
    wrong_keys: tuple[bytes, ...],
    hf_public_assets: FrozenHFPublicAssets,
    lf_public_assets: FrozenLFPublicAssets,
) -> dict[str, dict[str, float]]:
    ordinary_image = require_ordinary_rgb_image(image)
    if not isinstance(hf_public_assets, FrozenHFPublicAssets):
        raise TypeError("blind HF score requires FrozenHFPublicAssets")
    if not isinstance(lf_public_assets, FrozenLFPublicAssets):
        raise TypeError("blind LF score requires FrozenLFPublicAssets")
    if len(wrong_keys) != 16 or any(not isinstance(item, bytes) for item in wrong_keys):
        raise ValueError("blind score requires exactly 16 normalized external wrong keys")
    lf = {"registered": float(score_lf_image(ordinary_image, key, lf_public_assets))}
    hf = {"registered": float(score_hf_image(ordinary_image, key, hf_public_assets))}
    for index, wrong_key in enumerate(wrong_keys):
        label = f"wrong_{index:02d}"
        lf[label] = float(score_lf_image(ordinary_image, wrong_key, lf_public_assets))
        hf[label] = float(score_hf_image(ordinary_image, wrong_key, hf_public_assets))
    joint = {label: min(lf[label], hf[label]) for label in lf}
    values = {"lf": lf, "hf": hf, "joint": joint}
    if not all(math.isfinite(value) for branch in values.values() for value in branch.values()):
        raise ValueError("nonfinite_blind_score")
    return values


def _flat_scores(values: dict[str, dict[str, float]]) -> dict[str, float]:
    expected_labels = {"registered", *(f"wrong_{index:02d}" for index in range(16))}
    if set(values) != set(BRANCHES) or any(set(values[branch]) != expected_labels for branch in BRANCHES):
        raise ValueError("blind score fields differ from the fixed 3-by-17 roster")
    return {
        f"{branch}__{label}": float(values[branch][label])
        for branch in BRANCHES
        for label in ("registered", *(f"wrong_{index:02d}" for index in range(16)))
    }


def _psnr(first: Any, second: Any) -> float:
    first_pixels = np.asarray(first, dtype=np.float64) / 255.0
    second_pixels = np.asarray(second, dtype=np.float64) / 255.0
    if first_pixels.shape != second_pixels.shape:
        raise ValueError("paired_image_shape_mismatch")
    mse = float(np.mean(np.square(first_pixels - second_pixels)))
    if not math.isfinite(mse) or mse <= 0.0:
        raise ValueError("paired_psnr_requires_finite_nonidentical_images")
    value = -10.0 * math.log10(mse)
    if not math.isfinite(value):
        raise ValueError("paired_psnr_not_finite")
    return value


def _candidate_aggregate_metrics(
    unit_id: str,
    measurement: Any,
    paired_rgb_psnr_db: float,
    *,
    share_sum_absolute_tolerance: float,
) -> dict[str, Any]:
    """Pass through the fixed public aggregates and reject invalid identities."""

    lf_share = measurement.lf_branch_share
    hf_share = measurement.hf_branch_share
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in (lf_share, hf_share)
    ):
        raise ValueError("public branch shares must be finite real scalars")
    if not 0.0 < float(lf_share) < 1.0 or not 0.0 < float(hf_share) < 1.0:
        raise ValueError("public branch shares must be strictly between zero and one")
    if not math.isclose(
        float(lf_share) + float(hf_share),
        1.0,
        rel_tol=0.0,
        abs_tol=share_sum_absolute_tolerance,
    ):
        raise ValueError("public branch shares do not sum to one within the frozen tolerance")
    for name in COUNTERFACTUAL_EFFECT_FIELDS:
        value = getattr(measurement, name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    if measurement.probe_evaluation_count != 64:
        raise ValueError("v2 candidate measurement must contain exactly 64 probe evaluations")
    result = {
        "unit_id": unit_id,
        "combined_relative_l2": measurement.combined_budget.relative_l2,
        "lf_effective_relative_l2": measurement.lf_effective_relative_l2,
        "hf_effective_relative_l2": measurement.hf_effective_relative_l2,
        "lf_branch_share": lf_share,
        "hf_branch_share": hf_share,
        "minimum_counterfactual_effect": measurement.minimum_counterfactual_effect,
        "probe_evaluation_count": measurement.probe_evaluation_count,
        "paired_rgb_psnr_db": paired_rgb_psnr_db,
    }
    for name in COUNTERFACTUAL_EFFECT_FIELDS:
        result[name] = getattr(measurement, name)
    return result


def _branch_share_population_summary(
    unit_metrics: list[dict[str, Any]],
    expected_unit_ids: tuple[str, ...],
    *,
    rc: int,
    share_sum_absolute_tolerance: float,
    population_std_absolute_tolerance: float,
) -> tuple[float | None, float | None, bool, bool]:
    """Compute two independent fixed-roster ddof=0 summaries only for valid RC0."""

    unavailable = (None, None, False, False)
    if rc != 0 or len(unit_metrics) != 8 or len(expected_unit_ids) != 8:
        return unavailable
    try:
        received_unit_ids = tuple(metric["unit_id"] for metric in unit_metrics)
    except KeyError:
        return unavailable
    if received_unit_ids != expected_unit_ids:
        return unavailable
    try:
        lf_values = np.asarray([metric["lf_branch_share"] for metric in unit_metrics], dtype=np.float64)
        hf_values = np.asarray([metric["hf_branch_share"] for metric in unit_metrics], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return unavailable
    if (
        lf_values.shape != (8,)
        or hf_values.shape != (8,)
        or not np.all(np.isfinite(lf_values))
        or not np.all(np.isfinite(hf_values))
    ):
        return unavailable
    if (
        not np.all((0.0 < lf_values) & (lf_values < 1.0))
        or not np.all((0.0 < hf_values) & (hf_values < 1.0))
        or not np.allclose(
            lf_values + hf_values,
            np.ones(8, dtype=np.float64),
            rtol=0.0,
            atol=share_sum_absolute_tolerance,
        )
    ):
        return unavailable
    lf_mean = float(np.sum(lf_values) / 8.0)
    lf_population_std = math.sqrt(float(np.sum(np.square(lf_values - lf_mean)) / 8.0))
    hf_mean = float(np.sum(hf_values) / 8.0)
    hf_population_std = math.sqrt(float(np.sum(np.square(hf_values - hf_mean)) / 8.0))
    lf_reference = float(np.std(lf_values, ddof=0))
    hf_reference = float(np.std(hf_values, ddof=0))
    if not all(math.isfinite(value) for value in (lf_population_std, hf_population_std)):
        return unavailable
    if not (
        math.isclose(
            lf_population_std,
            lf_reference,
            rel_tol=0.0,
            abs_tol=population_std_absolute_tolerance,
        )
        and math.isclose(
            hf_population_std,
            hf_reference,
            rel_tol=0.0,
            abs_tol=population_std_absolute_tolerance,
        )
        and math.isclose(
            lf_population_std,
            hf_population_std,
            rel_tol=0.0,
            abs_tol=population_std_absolute_tolerance,
        )
    ):
        return unavailable
    supports_nonidentical = lf_population_std > 0.0 and hf_population_std > 0.0
    return lf_population_std, hf_population_std, supports_nonidentical, True


def _gate_evidence(records: list[dict[str, Any]], unit_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    by_unit: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_unit.setdefault(record["unit_id"], {})[record["arm"]] = record
    gates: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        gate_a = 0
        gate_b = 0
        for transaction in by_unit.values():
            joint = transaction[ARMS[0]]["scores"]
            primary_null = transaction[ARMS[1]]["scores"]
            registered = float(joint[f"{branch}__registered"])
            wrong = [float(joint[f"{branch}__wrong_{index:02d}"]) for index in range(16)]
            gate_a += int(registered > max(wrong))
            gate_b += int(registered > float(primary_null[f"{branch}__registered"]))
        gates[branch] = {
            "gate_a_pass_units": gate_a,
            "gate_b_pass_units": gate_b,
            "gate_a_pass": gate_a >= 7,
            "gate_b_pass": gate_b >= 7,
            "strict_ties_fail": True,
        }
    budget_units = sum(metric["combined_relative_l2"] <= 0.012 for metric in unit_metrics)
    nonzero_units = sum(
        metric["lf_effective_relative_l2"] > 0.0 and metric["hf_effective_relative_l2"] > 0.0
        for metric in unit_metrics
    )
    response_units = sum(
        all(
            isinstance(metric.get(name), (int, float))
            and not isinstance(metric.get(name), bool)
            and math.isfinite(float(metric[name]))
            and float(metric[name]) >= 0.0
            for name in COUNTERFACTUAL_EFFECT_FIELDS
        )
        for metric in unit_metrics
    )
    probe_count_units = sum(metric.get("probe_evaluation_count") == 64 for metric in unit_metrics)
    share_units = sum(
        isinstance(metric.get("lf_branch_share"), (int, float))
        and isinstance(metric.get("hf_branch_share"), (int, float))
        and not isinstance(metric.get("lf_branch_share"), bool)
        and not isinstance(metric.get("hf_branch_share"), bool)
        and math.isclose(
            float(metric["lf_branch_share"]) + float(metric["hf_branch_share"]),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for metric in unit_metrics
    )
    psnr_units = sum(metric["paired_rgb_psnr_db"] >= 30.0 for metric in unit_metrics)
    pass_all = (
        len(records) == 16
        and len(unit_metrics) == 8
        and all(item[gate] for item in gates.values() for gate in ("gate_a_pass", "gate_b_pass"))
        and budget_units == nonzero_units == response_units == probe_count_units == share_units == psnr_units == 8
    )
    return {
        "branches": gates,
        "combined_budget_pass_units": budget_units,
        "both_nonzero_branches_pass_units": nonzero_units,
        "baseline_differenced_probe_response_pass_units": response_units,
        "probe_evaluation_count_64_pass_units": probe_count_units,
        "public_branch_share_valid_pass_units": share_units,
        "paired_rgb_psnr_pass_units": psnr_units,
        "all_predeclared_gates_pass": pass_all,
        "formal_fpr_claim": False,
    }


def _now() -> float:
    return time.time()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _same_json_bytes(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """Compare only frozen public identity/history with JSON type and order fidelity."""

    return _json_bytes(first) == _json_bytes(second)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _read_json_bytes(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)


def _public_identity(
    protocol: ContentChainProtocol,
    *,
    exact: str,
    key_digest: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "exact": exact,
        "execution_scope_id": EXECUTION_SCOPE_ID,
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol.protocol_digest,
        "public_key_digest": key_digest,
        "model_id": protocol.config["generation_runtime"]["model_id"],
        "ordered_roster": [
            [unit.unit_id, unit.source_id]
            for unit in protocol.roster
        ],
        "ordered_arms": list(ARMS),
        "record_contract_id": RECORD_CONTRACT_ID,
        "fixed_unit_count": FIXED_UNIT_COUNT,
        "records_per_unit": RECORDS_PER_UNIT,
        "fixed_record_count": FIXED_RECORD_COUNT,
        "checkpoint_interval_hours": CHECKPOINT_INTERVAL_HOURS,
    }


def _new_state(identity: dict[str, Any], now: float) -> dict[str, Any]:
    return {
        "state_schema_id": STATE_SCHEMA_ID,
        "identity": identity,
        "checkpoint_sequence": 0,
        "checkpoint_time_anchor_unix_seconds": _finite_real(now, "checkpoint time anchor"),
        "committed_unit_count": 0,
        "records": [],
    }


def _validate_state(
    state: Any,
    identity: dict[str, Any],
    protocol: ContentChainProtocol,
) -> dict[str, Any]:
    if not isinstance(state, dict) or tuple(state) != _STATE_FIELDS:
        raise ValueError("resumable state fields or order differ")
    if state["state_schema_id"] != STATE_SCHEMA_ID:
        raise ValueError("resumable state schema identity differs")
    received_identity = state["identity"]
    if not isinstance(received_identity, dict) or tuple(received_identity) != _IDENTITY_FIELDS:
        raise ValueError("resumable public identity fields or order differ")
    if not _same_json_bytes(received_identity, identity):
        raise ValueError("resumable public identity differs")
    sequence = state["checkpoint_sequence"]
    committed = state["committed_unit_count"]
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise ValueError("checkpoint sequence must be a nonnegative integer")
    if not isinstance(committed, int) or isinstance(committed, bool) or not 0 <= committed <= 8:
        raise ValueError("committed unit count differs from the fixed roster")
    anchor = _finite_real(
        state["checkpoint_time_anchor_unix_seconds"], "checkpoint time anchor"
    )
    if anchor < 0.0:
        raise ValueError("checkpoint time anchor must be nonnegative")
    records = state["records"]
    if not isinstance(records, list) or len(records) != committed * RECORDS_PER_UNIT:
        raise ValueError("committed records are not an exact two-record unit prefix")
    for unit_index in range(committed):
        unit = protocol.roster[unit_index]
        transaction = records[unit_index * 2 : unit_index * 2 + 2]
        for arm_index, record in enumerate(transaction):
            if not isinstance(record, dict):
                raise TypeError("committed record must be a plain JSON object")
            _validate_content_v2_record(record)
            if (
                record["run_id"] != identity["run_id"]
                or record["unit_id"] != unit.unit_id
                or record["source_cluster_id"] != unit.source_id
                or record["arm"] != ARMS[arm_index]
                or record["code_revision"] != identity["exact"]
                or record["config_digest"] != identity["protocol_digest"]
                or record["key_public_digest"] != identity["public_key_digest"]
            ):
                raise ValueError("committed record prefix identity or order differs")
        statuses = tuple(record["status"] for record in transaction)
        if statuses not in (("success", "success"), ("operational_failure", "operational_failure")):
            raise ValueError("committed unit transaction status is incomplete")
        if statuses[0] == "operational_failure" and (
            transaction[0]["failure_reason"] != transaction[1]["failure_reason"]
        ):
            raise ValueError("committed unit failure classes differ")
    return state


def _write_local_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".state-stage-", dir=path.parent) as staging:
        staged = Path(staging) / "state.json"
        staged.write_bytes(_json_bytes(state))
        os.replace(staged, path)


def _zip_bytes(members: tuple[tuple[str, bytes], ...], path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload)


def _copy_create_only(source: Path, destination: Path) -> None:
    opened = False
    try:
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            opened = True
            shutil.copyfileobj(incoming, outgoing)
    except BaseException:
        if opened:
            destination.unlink(missing_ok=True)
        raise


def _publish_pair(
    *,
    local_run_root: Path,
    sink_run_root: Path,
    archive_name: str,
    members: tuple[tuple[str, bytes], ...],
) -> None:
    sink_run_root.mkdir(parents=True, exist_ok=True)
    archive_destination = sink_run_root / archive_name
    checksum_destination = sink_run_root / f"{archive_name}.sha256"
    if archive_destination.exists() or checksum_destination.exists():
        raise FileExistsError("create-only artifact destination already exists")
    created: list[Path] = []
    with tempfile.TemporaryDirectory(prefix=".artifact-stage-", dir=local_run_root) as staging:
        staging_root = Path(staging)
        staged_archive = staging_root / archive_name
        staged_checksum = staging_root / f"{archive_name}.sha256"
        _zip_bytes(members, staged_archive)
        digest = hashlib.sha256(staged_archive.read_bytes()).hexdigest()
        staged_checksum.write_text(
            f"{digest}  {archive_name}\n", encoding="ascii"
        )
        try:
            _copy_create_only(staged_archive, archive_destination)
            created.append(archive_destination)
            _copy_create_only(staged_checksum, checksum_destination)
            created.append(checksum_destination)
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise


def _read_checkpoint_pair(archive_path: Path, checksum_path: Path) -> dict[str, Any]:
    checksum_fields = checksum_path.read_text(encoding="ascii").strip().split()
    if (
        len(checksum_fields) != 2
        or re.fullmatch(r"[0-9a-f]{64}", checksum_fields[0]) is None
        or checksum_fields[1] != archive_path.name
    ):
        raise ValueError("checkpoint checksum sidecar differs")
    payload = archive_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != checksum_fields[0]:
        raise ValueError("checkpoint checksum validation failed")
    with zipfile.ZipFile(archive_path, "r") as archive:
        if archive.namelist() != ["state.json"]:
            raise ValueError("checkpoint ZIP must contain exactly state.json")
        return _read_json_bytes(archive.read("state.json"))


def _load_sink_checkpoint(
    sink_run_root: Path,
    identity: dict[str, Any],
    protocol: ContentChainProtocol,
) -> dict[str, Any] | None:
    if not sink_run_root.exists():
        return None
    run_id = identity["run_id"]
    pattern = re.compile(re.escape(run_id) + r"\.checkpoint-([0-9]{4})\.zip")
    archives: dict[int, Path] = {}
    sidecars: dict[int, Path] = {}
    for path in sink_run_root.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            archives[int(match.group(1))] = path
            continue
        checksum_match = pattern.fullmatch(path.name.removesuffix(".sha256"))
        if path.name.endswith(".zip.sha256") and checksum_match:
            sidecars[int(checksum_match.group(1))] = path
    if set(archives) != set(sidecars):
        raise ValueError("sink checkpoint ZIP and SHA pairs are incomplete")
    if not archives:
        return None
    sequences = sorted(archives)
    if sequences != list(range(sequences[-1] + 1)):
        raise ValueError("sink checkpoint sequence is not contiguous from zero")
    previous: dict[str, Any] | None = None
    for sequence in sequences:
        state = _validate_state(
            _read_checkpoint_pair(archives[sequence], sidecars[sequence]), identity, protocol
        )
        if state["checkpoint_sequence"] != sequence + 1:
            raise ValueError("sink checkpoint metadata sequence differs from its name")
        if previous is not None:
            previous_records = previous["records"]
            records = state["records"]
            if (
                len(records) <= len(previous_records)
                or not _same_json_bytes(
                    {"records": records[: len(previous_records)]},
                    {"records": previous_records},
                )
            ):
                raise ValueError("sink checkpoint history diverges or rolls back")
        previous = state
    return previous


def _terminal_pair_presence(sink_run_root: Path, run_id: str) -> None:
    archive = sink_run_root / f"{run_id}.zip"
    checksum = sink_run_root / f"{run_id}.zip.sha256"
    if archive.exists() or checksum.exists():
        raise FileExistsError("terminal artifact pair already exists; no terminal reconstruction")


def _resolve_state(
    *,
    local_state_path: Path,
    sink_run_root: Path,
    identity: dict[str, Any],
    protocol: ContentChainProtocol,
    now: float,
) -> dict[str, Any]:
    _terminal_pair_presence(sink_run_root, identity["run_id"])
    local = None
    if local_state_path.exists():
        local = _validate_state(
            _read_json_bytes(local_state_path.read_bytes()), identity, protocol
        )
    sink = _load_sink_checkpoint(sink_run_root, identity, protocol)
    if local is None and sink is None:
        state = _new_state(identity, now)
        _write_local_state(local_state_path, state)
        return state
    if local is None:
        assert sink is not None
        _write_local_state(local_state_path, sink)
        return sink
    if sink is None:
        if local["checkpoint_sequence"] != 0:
            raise ValueError("local checkpoint metadata would roll back missing sink history")
        return local
    local_sequence = local["checkpoint_sequence"]
    sink_sequence = sink["checkpoint_sequence"]
    if sink_sequence > local_sequence:
        if not _same_json_bytes({"records": sink["records"]}, {"records": local["records"]}):
            raise ValueError("checkpoint crash reconciliation requires identical records")
        _write_local_state(local_state_path, sink)
        return sink
    if sink_sequence < local_sequence:
        raise ValueError("sink checkpoint metadata rolls back local publication history")
    if len(sink["records"]) > len(local["records"]):
        raise ValueError("sink checkpoint has an uncommitted longer history")
    if not _same_json_bytes(
        {"records": local["records"][: len(sink["records"])]},
        {"records": sink["records"]},
    ):
        raise ValueError("local and sink checkpoint histories diverge")
    if not _same_json_bytes(
        {
            "checkpoint_time_anchor_unix_seconds": local[
                "checkpoint_time_anchor_unix_seconds"
            ]
        },
        {
            "checkpoint_time_anchor_unix_seconds": sink[
                "checkpoint_time_anchor_unix_seconds"
            ]
        },
    ):
        raise ValueError("local and sink checkpoint time metadata differs")
    return local


def _progress(identity: dict[str, Any], committed: int, phase: str) -> None:
    print("CEGWM_PROGRESS " + json.dumps({
        "run_id": identity["run_id"],
        "committed": committed,
        "fixed_total": FIXED_UNIT_COUNT,
        "phase": phase,
    }, separators=(",", ":")), flush=True)


def _summary(identity: dict[str, Any], committed: int, rc: int) -> None:
    print("CEGWM_SUMMARY " + json.dumps({
        "run_id": identity["run_id"],
        "committed": committed,
        "fixed_total": FIXED_UNIT_COUNT,
        "rc": rc,
        "phase": "terminal",
    }, separators=(",", ":")), flush=True)


def _derive_result(
    records: list[dict[str, Any]],
    protocol: ContentChainProtocol,
    identity: dict[str, Any],
) -> dict[str, Any]:
    unit_metrics: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, unit in enumerate(protocol.roster):
        transaction = records[index * 2 : index * 2 + 2]
        if transaction[0]["status"] == "operational_failure":
            failures.append({
                "unit_id": unit.unit_id,
                "status": "failed",
                "error_type": transaction[0]["failure_reason"],
            })
        else:
            unit_metrics.append({"unit_id": unit.unit_id, **transaction[0]["metrics"]})
    rc = 0 if not failures and len(records) == 16 and len(unit_metrics) == 8 else 2
    aggregate_contract = protocol.config["aggregate_measurement"]
    lf_share_std, hf_share_std, supports_nonidentical, summary_valid = (
        _branch_share_population_summary(
            unit_metrics,
            tuple(unit.unit_id for unit in protocol.roster),
            rc=rc,
            share_sum_absolute_tolerance=aggregate_contract[
                "branch_share_sum_absolute_tolerance"
            ],
            population_std_absolute_tolerance=aggregate_contract[
                "population_std_absolute_tolerance"
            ],
        )
    )
    if rc == 0 and not summary_valid:
        rc = 1
        lf_share_std = hf_share_std = None
        supports_nonidentical = False
    gates = _gate_evidence(records, unit_metrics) if rc == 0 else None
    return {
        "rc": rc,
        "completeness": COMPLETE_EXECUTION if rc == 0 else INCOMPLETE_EXECUTION,
        "scientific_outcome_allowed": rc == 0,
        "scientific_status": "not_adjudicated" if rc == 0 else "not_evaluable",
        "execution_scope_id": EXECUTION_SCOPE_ID,
        "exact": identity["exact"],
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol.protocol_digest,
        "public_key_digest": identity["public_key_digest"],
        "fixed_denominator_units": FIXED_UNIT_COUNT,
        "fixed_records": FIXED_RECORD_COUNT,
        "lf_branch_share_population_std": lf_share_std,
        "hf_branch_share_population_std": hf_share_std,
        "fixed_roster_allocation_not_all_identical_supported": supports_nonidentical,
        "records": records,
        "unit_aggregate_metrics": unit_metrics,
        "failed_units": failures,
        "gate_evidence": gates,
        "limitations": list(protocol.config["limitations"]),
    }


def _fatal_result(
    records: list[dict[str, Any]],
    identity: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "rc": 2,
        "completeness": INCOMPLETE_EXECUTION,
        "run_id": identity["run_id"],
        "execution_scope_id": identity["execution_scope_id"],
        "exact": identity["exact"],
        "protocol_id": identity["protocol_id"],
        "protocol_digest": identity["protocol_digest"],
        "public_key_digest": identity["public_key_digest"],
        "fixed_unit_count": FIXED_UNIT_COUNT,
        "fixed_record_count": FIXED_RECORD_COUNT,
        "committed_unit_count": len(records) // 2,
        "record_count": len(records),
        "operational_error_class": _public_operational_error_class(error),
        "records": records,
    }


def _receipt(identity: dict[str, Any], committed: int) -> dict[str, Any]:
    return {
        "artifact_kind": "terminal",
        "run_id": identity["run_id"],
        "exact": identity["exact"],
        "execution_scope_id": identity["execution_scope_id"],
        "protocol_id": identity["protocol_id"],
        "protocol_digest": identity["protocol_digest"],
        "public_key_digest": identity["public_key_digest"],
        "fixed_unit_count": FIXED_UNIT_COUNT,
        "fixed_record_count": FIXED_RECORD_COUNT,
        "committed_unit_count": committed,
        "result_member": "result.json",
        "external_validation_required": True,
    }


def _publish_terminal(
    local_run_root: Path,
    sink_run_root: Path,
    identity: dict[str, Any],
    result: dict[str, Any],
    committed: int,
) -> None:
    _publish_pair(
        local_run_root=local_run_root,
        sink_run_root=sink_run_root,
        archive_name=f"{identity['run_id']}.zip",
        members=(
            ("receipt.json", _json_bytes(_receipt(identity, committed))),
            ("result.json", _json_bytes(result)),
        ),
    )


def _unit_transaction(
    *,
    unit: Any,
    pipeline: Any,
    assets: ContentEmbedAssets,
    key: bytes,
    wrong_keys: tuple[bytes, ...],
    identity: dict[str, Any],
    protocol: ContentChainProtocol,
) -> list[dict[str, Any]]:
    joint_generator = torch.Generator(device="cuda").manual_seed(unit.seed)
    null_generator = torch.Generator(device="cuda").manual_seed(unit.seed)
    output = run_sd35_content_adaptive(
        pipeline, unit.prompt, key, assets,
        height=unit.height, width=unit.width, generator=joint_generator,
    )
    primary_null = run_sd35_plain(
        pipeline, unit.prompt,
        height=unit.height, width=unit.width, generator=null_generator,
    )
    joint_scores = _blind_scores(
        output.image, key, wrong_keys,
        assets.hf_public_assets, assets.lf_public_assets,
    )
    null_scores = _blind_scores(
        primary_null, key, wrong_keys,
        assets.hf_public_assets, assets.lf_public_assets,
    )
    metrics = _candidate_aggregate_metrics(
        unit.unit_id,
        output.measurement,
        _psnr(output.image, primary_null),
        share_sum_absolute_tolerance=protocol.config["aggregate_measurement"][
            "branch_share_sum_absolute_tolerance"
        ],
    )
    return [
        _content_v2_record(
            run_id=identity["run_id"], unit_id=unit.unit_id,
            source_cluster_id=unit.source_id, arm=ARMS[0], condition="clean",
            code_revision=identity["exact"], config_digest=identity["protocol_digest"],
            key_public_digest=identity["public_key_digest"], status="success",
            scores=_flat_scores(joint_scores),
            metrics={name: float(value) for name, value in metrics.items() if name != "unit_id"},
        ),
        _content_v2_record(
            run_id=identity["run_id"], unit_id=unit.unit_id,
            source_cluster_id=unit.source_id, arm=ARMS[1], condition="clean",
            code_revision=identity["exact"], config_digest=identity["protocol_digest"],
            key_public_digest=identity["public_key_digest"], status="success",
            scores=_flat_scores(null_scores),
            metrics={"paired_rgb_psnr_db": metrics["paired_rgb_psnr_db"]},
        ),
    ]


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    local_work_root = Path(args.local_work_root).resolve()
    artifact_sink = Path(args.artifact_sink).resolve()
    key_text = os.environ.pop(KEY_ENV, "")
    token = os.environ.pop(TOKEN_ENV, "")
    if not key_text.strip() or not token.strip():
        raise RuntimeError("CEG_WM_ROOT_KEY_and_HF_TOKEN_are_required")
    key = normalize_detection_key(key_text)
    key_text = ""
    exact = _git_exact(repo_root, args.expected_exact)
    protocol = _load_protocol(repo_root)
    key_digest = public_key_digest(key)
    run_id = f"content-adaptive-v2-{protocol.protocol_digest[:12]}-{key_digest[:12]}"
    identity = _public_identity(protocol, exact=exact, key_digest=key_digest, run_id=run_id)
    local_run_root = local_work_root / run_id
    local_run_root.mkdir(parents=True, exist_ok=True)
    local_state_path = local_run_root / "state.json"
    sink_run_root = artifact_sink / run_id
    state = _resolve_state(
        local_state_path=local_state_path,
        sink_run_root=sink_run_root,
        identity=identity,
        protocol=protocol,
        now=_now(),
    )
    _progress(identity, state["committed_unit_count"], "identity_ready")
    _progress(identity, state["committed_unit_count"], "resume_ready")
    wrong_keys = _wrong_keys(key, protocol)
    result: dict[str, Any]
    try:
        try:
            pipeline, assets = _load_pipeline_and_assets(identity["model_id"], token)
        finally:
            token = ""
        for unit_index in range(state["committed_unit_count"], FIXED_UNIT_COUNT):
            unit = protocol.roster[unit_index]
            try:
                transaction = _unit_transaction(
                    unit=unit, pipeline=pipeline, assets=assets, key=key,
                    wrong_keys=wrong_keys, identity=identity, protocol=protocol,
                )
            except Exception as error:  # noqa: BLE001 - fixed denominator records the attempt
                error_class = _public_operational_error_class(error)
                transaction = [
                    _content_v2_record(
                        run_id=run_id, unit_id=unit.unit_id,
                        source_cluster_id=unit.source_id, arm=arm, condition="clean",
                        code_revision=exact, config_digest=protocol.protocol_digest,
                        key_public_digest=key_digest, status="operational_failure",
                        failure_reason=error_class,
                    )
                    for arm in ARMS
                ]
            prospective = dict(state)
            prospective["records"] = [*state["records"], *transaction]
            prospective["committed_unit_count"] = unit_index + 1
            _validate_state(prospective, identity, protocol)
            _write_local_state(local_state_path, prospective)
            state = prospective
            _progress(identity, state["committed_unit_count"], "unit_committed")
            now = _now()
            elapsed = now - state["checkpoint_time_anchor_unix_seconds"]
            if elapsed >= CHECKPOINT_INTERVAL_HOURS * 3600.0:
                checkpoint_sequence = state["checkpoint_sequence"]
                checkpoint_state = dict(state)
                checkpoint_state["checkpoint_sequence"] = checkpoint_sequence + 1
                checkpoint_state["checkpoint_time_anchor_unix_seconds"] = now
                archive_name = f"{run_id}.checkpoint-{checkpoint_sequence:04d}.zip"
                _publish_pair(
                    local_run_root=local_run_root,
                    sink_run_root=sink_run_root,
                    archive_name=archive_name,
                    members=(("state.json", _json_bytes(checkpoint_state)),),
                )
                _write_local_state(local_state_path, checkpoint_state)
                state = checkpoint_state
                _progress(identity, state["committed_unit_count"], "checkpoint_published")
        result = _derive_result(state["records"], protocol, identity)
    except Exception as error:  # noqa: BLE001 - sanitized operational terminal only
        result = _fatal_result(state["records"], identity, error)
    committed = state["committed_unit_count"]
    rc = int(result["rc"])
    _publish_terminal(local_run_root, sink_run_root, identity, result, committed)
    _summary(identity, committed, rc)
    return rc


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--local-work-root", required=True)
    parser.add_argument("--artifact-sink", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
