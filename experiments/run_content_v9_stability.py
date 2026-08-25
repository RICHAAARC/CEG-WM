"""Formal fixed-order multi-cohort evaluation for the accepted Content V9 asset."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from experiments import run_content_adaptive_dual_branch_v2_clean as engine
from experiments import run_content_v6_clean as v6_runner
from cegwm.method.content_adaptive_v2 import COUNTERFACTUAL_EFFECT_FIELDS
from cegwm.protocol.content_chain_v9_stability import (
    CONTENT_V9_STABILITY_ARTIFACT_CONTRACT_ID,
    CONTENT_V9_STABILITY_CALIBRATION_ASSET_SHA256,
    CONTENT_V9_STABILITY_CALIBRATION_ASSET_SIDECAR_FILE_SHA256,
    CONTENT_V9_STABILITY_EVALUATED_CANDIDATE_ID,
    CONTENT_V9_STABILITY_EXECUTION_SCOPE_ID,
    CONTENT_V9_STABILITY_PROTOCOL_DIGEST,
    CONTENT_V9_STABILITY_PROTOCOL_ID,
    CONTENT_V9_STABILITY_PUBLIC_KEY_DIGEST,
    CONTENT_V9_STABILITY_RECORD_CONTRACT_ID,
    CONTENT_V9_STABILITY_STATE_SCHEMA_ID,
    CONTENT_V9_STABILITY_TERMINAL_RECEIPT_ID,
    ContentV9StabilityContract,
    ContentV9StabilityUnit,
    deterministic_stability_run_id,
    load_content_v9_stability_contract,
    strict_weighted_gate,
)
from cegwm.runtime.content_weighted_joint_sd35_v9 import (
    ContentV9CalibrationAssets,
    ContentV9StabilityOutput,
    derive_stability_wrong_keys,
    run_content_v9_stability_unit,
)
from cegwm.shared.keys import normalize_detection_key, public_key_digest

KEY_ENV = engine.KEY_ENV
TOKEN_ENV = engine.TOKEN_ENV
CHECKPOINT_INTERVAL_HOURS = engine.CHECKPOINT_INTERVAL_HOURS
FIXED_UNIT_COUNT = 80
RECORDS_PER_UNIT = 2
FIXED_RECORD_COUNT = 160
SECTION_IDS = (
    "old_roster_reference",
    "current_v6_roster_reference",
    "novel_seed_stability_seed_01",
    "novel_seed_stability_seed_02",
)
LOGICAL_SECTION_IDS = (
    "old_roster_reference",
    "current_v6_roster_reference",
    "novel_seed_stability",
)
SECTION_COUNTS = (8, 8, 32, 32)
SECTION_GATE_MINIMUMS = (7, 7, 28, 28)
ARMS = (
    CONTENT_V9_STABILITY_EVALUATED_CANDIDATE_ID,
    f"primary_null__{CONTENT_V9_STABILITY_EVALUATED_CANDIDATE_ID}",
)
SCORE_BRANCHES = ("lf", "hf", "weighted_joint")
SCORE_LABELS = ("registered", *(f"wrong_{index:02d}" for index in range(16)))
SCORE_FIELDS = tuple(
    f"{branch}__{label}" for branch in SCORE_BRANCHES for label in SCORE_LABELS
)
RECORD_FIELDS = (
    "run_id", "unit_id", "source_cluster_id", "arm", "condition",
    "code_revision", "config_digest", "key_public_digest", "status",
    "failure_reason", "scores", "metrics", "record_contract_id",
)
CANDIDATE_METRIC_FIELDS = (
    "combined_relative_l2", "lf_effective_relative_l2", "hf_effective_relative_l2",
    "lf_branch_share", "hf_branch_share", *COUNTERFACTUAL_EFFECT_FIELDS,
    "minimum_counterfactual_effect", "probe_evaluation_count", "paired_rgb_psnr_db",
)
NULL_METRIC_FIELDS = ("paired_rgb_psnr_db",)
STATE_FIELDS = (
    "state_schema_id", "identity", "checkpoint_sequence",
    "checkpoint_time_anchor_unix_seconds", "committed_unit_count", "records",
)
IDENTITY_FIELDS = (
    "run_id", "exact", "execution_scope_id", "protocol_id", "protocol_digest",
    "public_key_digest", "calibration_asset_sha256",
    "calibration_asset_sidecar_file_sha256", "model_id", "ordered_sections",
    "ordered_arms", "record_contract_id", "fixed_unit_count", "records_per_unit",
    "fixed_record_count", "checkpoint_interval_hours", "artifact_contract_id",
)
COMPLETE_EXECUTION = "complete_for_content_v9_multi_cohort_stability_evaluation"
INCOMPLETE_EXECUTION = "incomplete_content_v9_multi_cohort_stability_execution"


def _ordered_units(contract: ContentV9StabilityContract) -> tuple[ContentV9StabilityUnit, ...]:
    return (
        *contract.old_roster_reference,
        *contract.current_v6_roster_reference,
        *contract.novel_seed_01,
        *contract.novel_seed_02,
    )


def _section_units(
    contract: ContentV9StabilityContract,
) -> tuple[tuple[ContentV9StabilityUnit, ...], ...]:
    return (
        contract.old_roster_reference,
        contract.current_v6_roster_reference,
        contract.novel_seed_01,
        contract.novel_seed_02,
    )


def _identity(
    contract: ContentV9StabilityContract, *, exact: str, key_digest: str
) -> dict[str, Any]:
    run_id = deterministic_stability_run_id(
        contract.protocol_digest,
        CONTENT_V9_STABILITY_CALIBRATION_ASSET_SHA256,
        key_digest,
    )
    return {
        "run_id": run_id,
        "exact": exact,
        "execution_scope_id": CONTENT_V9_STABILITY_EXECUTION_SCOPE_ID,
        "protocol_id": contract.config["protocol_id"],
        "protocol_digest": contract.protocol_digest,
        "public_key_digest": key_digest,
        "calibration_asset_sha256": CONTENT_V9_STABILITY_CALIBRATION_ASSET_SHA256,
        "calibration_asset_sidecar_file_sha256": (
            CONTENT_V9_STABILITY_CALIBRATION_ASSET_SIDECAR_FILE_SHA256
        ),
        "model_id": contract.v6_protocol.config["generation_runtime"]["model_id"],
        "ordered_sections": [
            {
                "section_id": section_id,
                "fixed_units": len(units),
                "fixed_records": len(units) * RECORDS_PER_UNIT,
                "weighted_gate_min_units": required,
                "ordered_roster": [[unit.unit_id, unit.source_id] for unit in units],
            }
            for section_id, units, required in zip(
                SECTION_IDS, _section_units(contract), SECTION_GATE_MINIMUMS, strict=True
            )
        ],
        "ordered_arms": list(ARMS),
        "record_contract_id": CONTENT_V9_STABILITY_RECORD_CONTRACT_ID,
        "fixed_unit_count": FIXED_UNIT_COUNT,
        "records_per_unit": RECORDS_PER_UNIT,
        "fixed_record_count": FIXED_RECORD_COUNT,
        "checkpoint_interval_hours": CHECKPOINT_INTERVAL_HOURS,
        "artifact_contract_id": CONTENT_V9_STABILITY_ARTIFACT_CONTRACT_ID,
    }


def _new_state(identity: dict[str, Any], now: float) -> dict[str, Any]:
    return {
        "state_schema_id": CONTENT_V9_STABILITY_STATE_SCHEMA_ID,
        "identity": identity,
        "checkpoint_sequence": 0,
        "checkpoint_time_anchor_unix_seconds": engine._finite_real(now, "checkpoint time anchor"),
        "committed_unit_count": 0,
        "records": [],
    }


def _finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validate_record(record: Any, *, unit: ContentV9StabilityUnit, arm_index: int,
                     identity: Mapping[str, Any]) -> None:
    if not isinstance(record, dict) or tuple(record) != RECORD_FIELDS:
        raise ValueError("Content V9 stability record fields or order differ")
    if (
        record["run_id"] != identity["run_id"]
        or record["unit_id"] != unit.unit_id
        or record["source_cluster_id"] != unit.source_id
        or record["arm"] != ARMS[arm_index]
        or record["condition"] != "clean"
        or record["code_revision"] != identity["exact"]
        or record["config_digest"] != identity["protocol_digest"]
        or record["key_public_digest"] != identity["public_key_digest"]
        or record["record_contract_id"] != CONTENT_V9_STABILITY_RECORD_CONTRACT_ID
    ):
        raise ValueError("Content V9 stability record identity differs")
    if record["status"] == "operational_failure":
        if record["failure_reason"] not in engine._PUBLIC_OPERATIONAL_ERROR_CLASSES:
            raise ValueError("Content V9 stability failure class differs")
        if record["scores"] or record["metrics"]:
            raise ValueError("Content V9 stability failure record must be sanitized")
        return
    if record["status"] != "success" or record["failure_reason"] is not None:
        raise ValueError("Content V9 stability record status differs")
    scores = record["scores"]
    if not isinstance(scores, dict) or tuple(scores) != SCORE_FIELDS:
        raise ValueError("Content V9 stability score fields or order differ")
    for name, value in scores.items():
        scalar = _finite(value, name)
        if not name.startswith("weighted_joint__") and not -1.0 <= scalar <= 1.0:
            raise ValueError("Content V9 stability branch score escaped [-1,1]")
    expected_metrics = CANDIDATE_METRIC_FIELDS if arm_index == 0 else NULL_METRIC_FIELDS
    metrics = record["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != set(expected_metrics):
        raise ValueError("Content V9 stability metric fields differ")
    for name in expected_metrics:
        _finite(metrics[name], name)
    if metrics["paired_rgb_psnr_db"] < 0.0:
        raise ValueError("Content V9 stability PSNR must be nonnegative")
    if arm_index == 0:
        if not 0.0 <= metrics["combined_relative_l2"] <= 0.012:
            raise ValueError("Content V9 stability combined budget differs")
        if not all(
            0.0 < metrics[name] <= 0.012
            for name in ("lf_effective_relative_l2", "hf_effective_relative_l2")
        ):
            raise ValueError("Content V9 stability effective branch budget differs")
        if not (
            0.0 < metrics["lf_branch_share"] < 1.0
            and 0.0 < metrics["hf_branch_share"] < 1.0
            and math.isclose(
                metrics["lf_branch_share"] + metrics["hf_branch_share"],
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Content V9 stability branch shares differ")
        effects = [metrics[name] for name in COUNTERFACTUAL_EFFECT_FIELDS]
        if any(value < 0.0 for value in effects):
            raise ValueError("Content V9 stability counterfactual effect differs")
        if metrics["minimum_counterfactual_effect"] != min(effects):
            raise ValueError("Content V9 stability minimum effect differs")
        if metrics["probe_evaluation_count"] != 64.0:
            raise ValueError("Content V9 stability probe count differs")


def _validate_state(
    state: Any, identity: dict[str, Any], contract: ContentV9StabilityContract
) -> dict[str, Any]:
    if not isinstance(state, dict) or tuple(state) != STATE_FIELDS:
        raise ValueError("Content V9 stability state fields or order differ")
    if state["state_schema_id"] != CONTENT_V9_STABILITY_STATE_SCHEMA_ID:
        raise ValueError("Content V9 stability state schema differs")
    if not isinstance(state["identity"], dict) or tuple(state["identity"]) != IDENTITY_FIELDS:
        raise ValueError("Content V9 stability identity fields or order differ")
    if not engine._same_json_bytes(state["identity"], identity):
        raise ValueError("Content V9 stability state identity differs")
    sequence = state["checkpoint_sequence"]
    committed = state["committed_unit_count"]
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise ValueError("Content V9 stability checkpoint sequence differs")
    if not isinstance(committed, int) or isinstance(committed, bool) or not 0 <= committed <= 80:
        raise ValueError("Content V9 stability committed count differs")
    anchor = _finite(state["checkpoint_time_anchor_unix_seconds"], "checkpoint time anchor")
    if anchor < 0.0:
        raise ValueError("Content V9 stability checkpoint anchor differs")
    records = state["records"]
    if not isinstance(records, list) or len(records) != committed * 2:
        raise ValueError("Content V9 stability state is not a whole-unit prefix")
    units = _ordered_units(contract)
    for unit_index in range(committed):
        transaction = records[unit_index * 2 : unit_index * 2 + 2]
        for arm_index, record in enumerate(transaction):
            _validate_record(record, unit=units[unit_index], arm_index=arm_index, identity=identity)
        statuses = tuple(record["status"] for record in transaction)
        if statuses not in (("success", "success"), ("operational_failure", "operational_failure")):
            raise ValueError("Content V9 stability transaction status is incomplete")
        if statuses[0] == "operational_failure" and (
            transaction[0]["failure_reason"] != transaction[1]["failure_reason"]
        ):
            raise ValueError("Content V9 stability transaction failure classes differ")
    return state


def _read_sink_state(archive: Path, sidecar: Path) -> dict[str, Any]:
    value = engine._read_checkpoint_pair(archive, sidecar)
    if not isinstance(value, dict):
        raise ValueError("Content V9 stability checkpoint state must be an object")
    return value


def _load_sink_checkpoint(
    sink_run_root: Path, identity: dict[str, Any], contract: ContentV9StabilityContract
) -> dict[str, Any] | None:
    if not sink_run_root.exists():
        return None
    pattern = re.compile(re.escape(identity["run_id"]) + r"\.checkpoint-([0-9]{4})\.zip")
    archives: dict[int, Path] = {}
    sidecars: dict[int, Path] = {}
    for path in sink_run_root.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            archives[int(match.group(1))] = path
        elif path.name.endswith(".zip.sha256"):
            match = pattern.fullmatch(path.name.removesuffix(".sha256"))
            if match:
                sidecars[int(match.group(1))] = path
    if set(archives) != set(sidecars):
        raise ValueError("Content V9 stability checkpoint pairs are incomplete")
    if not archives:
        return None
    sequences = sorted(archives)
    if sequences != list(range(sequences[-1] + 1)):
        raise ValueError("Content V9 stability checkpoint sequence is not contiguous")
    previous: dict[str, Any] | None = None
    for sequence in sequences:
        state = _validate_state(
            _read_sink_state(archives[sequence], sidecars[sequence]), identity, contract
        )
        if state["checkpoint_sequence"] != sequence + 1:
            raise ValueError("Content V9 stability checkpoint name and metadata differ")
        if previous is not None and (
            len(state["records"]) <= len(previous["records"])
            or not engine._same_json_bytes(
                {"records": state["records"][: len(previous["records"])]},
                {"records": previous["records"]},
            )
        ):
            raise ValueError("Content V9 stability checkpoint history diverges")
        previous = state
    return previous


def _resolve_state(
    *, local_state_path: Path, sink_run_root: Path, identity: dict[str, Any],
    contract: ContentV9StabilityContract, now: float,
) -> dict[str, Any]:
    engine._terminal_pair_presence(sink_run_root, identity["run_id"])
    local = None
    if local_state_path.exists():
        local = _validate_state(
            engine._read_json_bytes(local_state_path.read_bytes()), identity, contract
        )
    sink = _load_sink_checkpoint(sink_run_root, identity, contract)
    if local is None and sink is None:
        state = _new_state(identity, now)
        engine._write_local_state(local_state_path, state)
        return state
    if local is None:
        assert sink is not None
        engine._write_local_state(local_state_path, sink)
        return sink
    if sink is None:
        if local["checkpoint_sequence"] != 0:
            raise ValueError("Content V9 stability local checkpoint history is missing from sink")
        return local
    if sink["checkpoint_sequence"] > local["checkpoint_sequence"]:
        if not engine._same_json_bytes({"records": sink["records"]}, {"records": local["records"]}):
            raise ValueError("Content V9 stability checkpoint reconciliation differs")
        engine._write_local_state(local_state_path, sink)
        return sink
    if sink["checkpoint_sequence"] < local["checkpoint_sequence"]:
        raise ValueError("Content V9 stability sink checkpoint history rolls back local state")
    if len(sink["records"]) > len(local["records"]):
        raise ValueError("Content V9 stability sink contains uncommitted history")
    if not engine._same_json_bytes(
        {"records": local["records"][: len(sink["records"])]},
        {"records": sink["records"]},
    ) or local["checkpoint_time_anchor_unix_seconds"] != sink["checkpoint_time_anchor_unix_seconds"]:
        raise ValueError("Content V9 stability local and sink histories diverge")
    return local


def _flat_scores(scores: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if tuple(scores) != SCORE_BRANCHES or any(tuple(scores[name]) != SCORE_LABELS for name in SCORE_BRANCHES):
        raise ValueError("Content V9 stability blind score fields differ")
    return {
        f"{branch}__{label}": float(scores[branch][label])
        for branch in SCORE_BRANCHES for label in SCORE_LABELS
    }


def _record(
    *, identity: Mapping[str, Any], unit: ContentV9StabilityUnit, arm_index: int,
    status: str, failure_reason: str | None = None,
    scores: Mapping[str, float] | None = None, metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    record = {
        "run_id": identity["run_id"],
        "unit_id": unit.unit_id,
        "source_cluster_id": unit.source_id,
        "arm": ARMS[arm_index],
        "condition": "clean",
        "code_revision": identity["exact"],
        "config_digest": identity["protocol_digest"],
        "key_public_digest": identity["public_key_digest"],
        "status": status,
        "failure_reason": failure_reason,
        "scores": dict(scores or {}),
        "metrics": dict(metrics or {}),
        "record_contract_id": CONTENT_V9_STABILITY_RECORD_CONTRACT_ID,
    }
    _validate_record(record, unit=unit, arm_index=arm_index, identity=identity)
    return record


def _failure_transaction(
    unit: ContentV9StabilityUnit, identity: Mapping[str, Any], error: Exception
) -> list[dict[str, Any]]:
    error_class = engine._public_operational_error_class(error)
    return [
        _record(
            identity=identity, unit=unit, arm_index=arm_index,
            status="operational_failure", failure_reason=error_class,
        )
        for arm_index in range(2)
    ]


def _unit_transaction(
    *, pipeline: Any, unit: ContentV9StabilityUnit, key: bytes,
    wrong_keys: tuple[bytes, ...], assets: ContentV9CalibrationAssets,
    contract: ContentV9StabilityContract, identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = run_content_v9_stability_unit(
        pipeline, unit, key, wrong_keys, assets, contract.calibration_asset
    )
    if not isinstance(output, ContentV9StabilityOutput):
        raise TypeError("Content V9 stability unit requires its real runtime output")
    metrics = engine._candidate_aggregate_metrics(
        unit.unit_id,
        output.measurement,
        engine._psnr(output.image, output.primary_null),
        share_sum_absolute_tolerance=contract.v6_protocol.config["aggregate_measurement"][
            "branch_share_sum_absolute_tolerance"
        ],
    )
    candidate_scores = dict(output.candidate_scores)
    null_scores = dict(output.primary_null_scores)
    if tuple(candidate_scores) != ("lf", "hf", "weighted_joint") or tuple(null_scores) != (
        "lf", "hf", "weighted_joint"
    ):
        raise ValueError("Content V9 stability runtime score branches differ")
    return [
        _record(
            identity=identity, unit=unit, arm_index=0, status="success",
            scores=_flat_scores(candidate_scores),
            metrics={name: float(value) for name, value in metrics.items() if name != "unit_id"},
        ),
        _record(
            identity=identity, unit=unit, arm_index=1, status="success",
            scores=_flat_scores(null_scores),
            metrics={"paired_rgb_psnr_db": float(metrics["paired_rgb_psnr_db"])},
        ),
    ]


def _mechanical_evidence(unit_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(unit_metrics)
    budget = sum(metric["combined_relative_l2"] <= 0.012 for metric in unit_metrics)
    nonzero = sum(
        metric["lf_effective_relative_l2"] > 0.0
        and metric["hf_effective_relative_l2"] > 0.0
        for metric in unit_metrics
    )
    effects = sum(
        all(math.isfinite(float(metric[name])) and metric[name] >= 0.0 for name in COUNTERFACTUAL_EFFECT_FIELDS)
        for metric in unit_metrics
    )
    probes = sum(metric["probe_evaluation_count"] == 64 for metric in unit_metrics)
    shares = sum(
        0.0 < metric["lf_branch_share"] < 1.0
        and 0.0 < metric["hf_branch_share"] < 1.0
        and math.isclose(
            metric["lf_branch_share"] + metric["hf_branch_share"],
            1.0, rel_tol=0.0, abs_tol=1e-12,
        )
        for metric in unit_metrics
    )
    psnr = sum(metric["paired_rgb_psnr_db"] >= 30.0 for metric in unit_metrics)
    return {
        "combined_budget_pass_units": budget,
        "both_nonzero_branches_pass_units": nonzero,
        "baseline_differenced_probe_response_pass_units": effects,
        "probe_evaluation_count_64_pass_units": probes,
        "public_branch_share_valid_pass_units": shares,
        "paired_rgb_psnr_pass_units": psnr,
        "all_mechanical_requirements_pass": all(
            value == count for value in (budget, nonzero, effects, probes, shares, psnr)
        ),
    }


def _margin(record: Mapping[str, Any], null: Mapping[str, Any], branch: str, gate: str) -> float:
    scores = record["scores"]
    registered = float(scores[f"{branch}__registered"])
    if gate == "a":
        return registered - max(float(scores[f"{branch}__wrong_{index:02d}"]) for index in range(16))
    return registered - float(null["scores"][f"{branch}__registered"])


def _section_result(
    *, section_id: str, units: Sequence[ContentV9StabilityUnit], records: list[dict[str, Any]],
    required: int, identity: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    metrics: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        transaction = records[index * 2 : index * 2 + 2]
        if len(transaction) != 2:
            continue
        if transaction[0]["status"] == "operational_failure":
            failures.append({
                "unit_id": unit.unit_id,
                "status": "failed",
                "error_type": transaction[0]["failure_reason"],
            })
        else:
            metrics.append({"unit_id": unit.unit_id, **transaction[0]["metrics"]})
    complete = len(records) == len(units) * 2
    rc = 0 if complete and not failures and len(metrics) == len(units) else 2
    evidence = None
    if rc == 0:
        margins = {
            branch: {
                gate: [
                    _margin(records[index * 2], records[index * 2 + 1], branch, gate)
                    for index in range(len(units))
                ]
                for gate in ("a", "b")
            }
            for branch in SCORE_BRANCHES
        }
        weighted_a = strict_weighted_gate(margins["weighted_joint"]["a"], required=required)
        weighted_b = strict_weighted_gate(margins["weighted_joint"]["b"], required=required)
        diagnostics = {
            branch: {
                f"gate_{gate}_pass_units": sum(value > 0.0 for value in margins[branch][gate])
                for gate in ("a", "b")
            }
            for branch in ("lf", "hf")
        }
        evidence = {
            "weighted_joint": {
                "gate_a_pass_units": weighted_a[0],
                "gate_b_pass_units": weighted_b[0],
                "gate_a_pass": weighted_a[1],
                "gate_b_pass": weighted_b[1],
                "required_units": required,
                "strict_ties_fail": True,
            },
            "lf_hf_diagnostics_only_no_hard_veto": diagnostics,
            **_mechanical_evidence(metrics),
            "all_section_weighted_gates_pass": weighted_a[1] and weighted_b[1],
            "formal_fpr_claim": False,
        }
    return {
        "section_id": section_id,
        "rc": rc,
        "completeness": COMPLETE_EXECUTION if rc == 0 else INCOMPLETE_EXECUTION,
        "scientific_status": "not_adjudicated" if rc == 0 else "not_evaluable",
        "fixed_denominator_units": len(units),
        "fixed_records": len(units) * 2,
        "committed_unit_count": len(records) // 2,
        "records": records,
        "unit_aggregate_metrics": metrics,
        "failed_units": failures,
        "gate_evidence": evidence,
        "result_is_independent": True,
        "controls_other_section_execution": False,
        "exact": identity["exact"],
    }


def _novel_descriptives(section_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    left, right = section_results[2], section_results[3]
    if left["rc"] != 0 or right["rc"] != 0:
        return []
    rows: list[dict[str, Any]] = []
    for index in range(32):
        left_records = left["records"][index * 2 : index * 2 + 2]
        right_records = right["records"][index * 2 : index * 2 + 2]
        left_a = _margin(left_records[0], left_records[1], "weighted_joint", "a")
        left_b = _margin(left_records[0], left_records[1], "weighted_joint", "b")
        right_a = _margin(right_records[0], right_records[1], "weighted_joint", "a")
        right_b = _margin(right_records[0], right_records[1], "weighted_joint", "b")
        rows.append({
            "prompt_ordinal": index + 1,
            "seed_01_unit_id": left_records[0]["unit_id"],
            "seed_02_unit_id": right_records[0]["unit_id"],
            "weighted_gate_a_agreement": (left_a > 0.0) == (right_a > 0.0),
            "weighted_gate_b_agreement": (left_b > 0.0) == (right_b > 0.0),
            "weighted_gate_a_margin_delta_seed_02_minus_seed_01": right_a - left_a,
            "weighted_gate_b_margin_delta_seed_02_minus_seed_01": right_b - left_b,
        })
    return rows


def _result(
    state: Mapping[str, Any], identity: Mapping[str, Any], contract: ContentV9StabilityContract,
    *, fatal_error: Exception | None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    record_offset = 0
    for section_id, units, required in zip(
        SECTION_IDS, _section_units(contract), SECTION_GATE_MINIMUMS, strict=True
    ):
        count = min(len(units) * 2, max(0, len(state["records"]) - record_offset))
        records = list(state["records"][record_offset : record_offset + count])
        results.append(_section_result(
            section_id=section_id, units=units, records=records,
            required=required, identity=identity,
        ))
        record_offset += len(units) * 2
    all_attempted = state["committed_unit_count"] == FIXED_UNIT_COUNT
    rc = 0 if fatal_error is None and all_attempted and all(item["rc"] == 0 for item in results) else 2
    logical_results = [
        results[0],
        results[1],
        {
            "section_id": "novel_seed_stability",
            "seed_strata_in_order": list(SECTION_IDS[2:]),
            "seed_stratum_results": results[2:],
            "per_prompt_descriptives": _novel_descriptives(results),
            "result_is_independent": True,
            "controls_other_section_execution": False,
        },
    ]
    return {
        "rc": rc,
        "completeness": COMPLETE_EXECUTION if rc == 0 else INCOMPLETE_EXECUTION,
        "run_id": identity["run_id"],
        "execution_scope_id": identity["execution_scope_id"],
        "exact": identity["exact"],
        "protocol_id": identity["protocol_id"],
        "protocol_digest": identity["protocol_digest"],
        "public_key_digest": identity["public_key_digest"],
        "calibration_asset_sha256": identity["calibration_asset_sha256"],
        "calibration_asset_sidecar_file_sha256": identity[
            "calibration_asset_sidecar_file_sha256"
        ],
        "scientific_status": "not_adjudicated" if rc == 0 else "not_evaluable",
        "fixed_unit_count_metadata_only_not_a_denominator": FIXED_UNIT_COUNT,
        "fixed_record_count_metadata_only_not_a_denominator": FIXED_RECORD_COUNT,
        "committed_unit_count": state["committed_unit_count"],
        "sections_in_order": list(LOGICAL_SECTION_IDS),
        "section_results": logical_results,
        "pooled_denominator_absent": True,
        "cross_section_conjunction_absent": True,
        "combined_result_absent": True,
        "section_outcome_controls_later_execution": False,
        "operational_error_class": (
            None if fatal_error is None else engine._public_operational_error_class(fatal_error)
        ),
        "external_validation_required": True,
    }


def _receipt(identity: Mapping[str, Any], committed: int) -> dict[str, Any]:
    return {
        "artifact_kind": "terminal",
        "artifact_contract_id": CONTENT_V9_STABILITY_ARTIFACT_CONTRACT_ID,
        "receipt_contract_id": CONTENT_V9_STABILITY_TERMINAL_RECEIPT_ID,
        "run_id": identity["run_id"],
        "exact": identity["exact"],
        "execution_scope_id": identity["execution_scope_id"],
        "protocol_id": identity["protocol_id"],
        "protocol_digest": identity["protocol_digest"],
        "public_key_digest": identity["public_key_digest"],
        "calibration_asset_sha256": identity["calibration_asset_sha256"],
        "calibration_asset_sidecar_file_sha256": identity[
            "calibration_asset_sidecar_file_sha256"
        ],
        "fixed_unit_count": FIXED_UNIT_COUNT,
        "fixed_record_count": FIXED_RECORD_COUNT,
        "committed_unit_count": committed,
        "result_member": "result.json",
        "external_validation_required": True,
    }


def _progress(identity: Mapping[str, Any], committed: int, phase: str) -> None:
    print("CEGWM_PROGRESS " + json.dumps({
        "run_id": identity["run_id"], "committed": committed,
        "fixed_total": FIXED_UNIT_COUNT, "phase": phase,
    }, separators=(",", ":")), flush=True)


def _summary(identity: Mapping[str, Any], committed: int, rc: int) -> None:
    print("CEGWM_SUMMARY " + json.dumps({
        "run_id": identity["run_id"], "committed": committed,
        "fixed_total": FIXED_UNIT_COUNT, "rc": rc, "phase": "terminal",
    }, separators=(",", ":")), flush=True)


def _load_pipeline_and_assets(model_id: str, token: str) -> tuple[Any, ContentV9CalibrationAssets]:
    pipeline, assets = v6_runner._load_pipeline_and_assets(model_id, token)
    return pipeline, ContentV9CalibrationAssets(assets.evaluation_assets)


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    local_work_root = Path(args.local_work_root).resolve()
    artifact_sink = Path(args.artifact_sink).resolve()
    key_text = os.environ.pop(KEY_ENV, "")
    token = os.environ.pop(TOKEN_ENV, "")
    if not key_text.strip():
        token = ""
        raise RuntimeError("CEG_WM_ROOT_KEY_is_required")
    key = normalize_detection_key(key_text)
    key_text = ""
    exact = engine._git_exact(repo_root, args.expected_exact)
    contract = load_content_v9_stability_contract(repo_root)
    key_digest = public_key_digest(key)
    if key_digest != CONTENT_V9_STABILITY_PUBLIC_KEY_DIGEST:
        key = b""
        token = ""
        raise RuntimeError("Content V9 stability public key identity differs")
    identity = _identity(contract, exact=exact, key_digest=key_digest)
    if identity["run_id"] != (
        "content-v9-stability-9bc8a94c1d02-63c17e8200a9-805bc21e173a"
    ):
        key = b""
        token = ""
        raise RuntimeError("Content V9 stability deterministic run identity differs")
    local_run_root = local_work_root / identity["run_id"]
    local_run_root.mkdir(parents=True, exist_ok=True)
    local_state_path = local_run_root / "state.json"
    sink_run_root = artifact_sink / identity["run_id"]
    state = _resolve_state(
        local_state_path=local_state_path, sink_run_root=sink_run_root,
        identity=identity, contract=contract, now=engine._now(),
    )
    _progress(identity, state["committed_unit_count"], "identity_ready")
    _progress(identity, state["committed_unit_count"], "resume_ready")
    if state["committed_unit_count"] == FIXED_UNIT_COUNT:
        key = b""
        token = ""
        result = _result(state, identity, contract, fatal_error=None)
        engine._publish_pair(
            local_run_root=local_run_root, sink_run_root=sink_run_root,
            archive_name=f"{identity['run_id']}.zip",
            members=(("receipt.json", engine._json_bytes(_receipt(identity, 80))),
                     ("result.json", engine._json_bytes(result))),
        )
        _summary(identity, 80, int(result["rc"]))
        return int(result["rc"])
    if not token.strip():
        key = b""
        raise RuntimeError("HF_TOKEN_is_required_for_incomplete_execution")
    fatal_error: Exception | None = None
    try:
        try:
            pipeline, assets = _load_pipeline_and_assets(identity["model_id"], token)
        finally:
            token = ""
        wrong_keys = derive_stability_wrong_keys(key)
        units = _ordered_units(contract)
        for unit_index in range(state["committed_unit_count"], FIXED_UNIT_COUNT):
            unit = units[unit_index]
            try:
                transaction = _unit_transaction(
                    pipeline=pipeline, unit=unit, key=key, wrong_keys=wrong_keys,
                    assets=assets, contract=contract, identity=identity,
                )
            except Exception as error:  # noqa: BLE001 - fixed denominator retains failure
                transaction = _failure_transaction(unit, identity, error)
            prospective = dict(state)
            prospective["records"] = [*state["records"], *transaction]
            prospective["committed_unit_count"] = unit_index + 1
            _validate_state(prospective, identity, contract)
            engine._write_local_state(local_state_path, prospective)
            state = prospective
            _progress(identity, state["committed_unit_count"], "unit_committed")
            now = engine._now()
            if now - state["checkpoint_time_anchor_unix_seconds"] >= 7200.0:
                sequence = state["checkpoint_sequence"]
                checkpoint = dict(state)
                checkpoint["checkpoint_sequence"] = sequence + 1
                checkpoint["checkpoint_time_anchor_unix_seconds"] = now
                engine._publish_pair(
                    local_run_root=local_run_root,
                    sink_run_root=sink_run_root,
                    archive_name=f"{identity['run_id']}.checkpoint-{sequence:04d}.zip",
                    members=(("state.json", engine._json_bytes(checkpoint)),),
                )
                engine._write_local_state(local_state_path, checkpoint)
                state = checkpoint
                _progress(identity, state["committed_unit_count"], "checkpoint_published")
    except Exception as error:  # noqa: BLE001 - sanitized fatal terminal
        fatal_error = error
    finally:
        key = b""
        token = ""
    result = _result(state, identity, contract, fatal_error=fatal_error)
    rc = int(result["rc"])
    engine._publish_pair(
        local_run_root=local_run_root,
        sink_run_root=sink_run_root,
        archive_name=f"{identity['run_id']}.zip",
        members=(("receipt.json", engine._json_bytes(_receipt(
            identity, state["committed_unit_count"]
        ))), ("result.json", engine._json_bytes(result))),
    )
    _summary(identity, state["committed_unit_count"], rc)
    return rc


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--local-work-root", required=True)
    parser.add_argument("--artifact-sink", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
