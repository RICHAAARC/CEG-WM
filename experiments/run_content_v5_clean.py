"""One-shot control-then-primary Content V5 evaluation on the authentic V4 path."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
from typing import Any, Mapping

from experiments import run_content_adaptive_dual_branch_v2_clean as engine
from experiments import run_content_v4_clean as v4_runner
from cegwm.protocol.content_chain_v2 import ContentChainProtocol
from cegwm.protocol.content_chain_v5 import (
    CONTENT_V5_ARMS,
    CONTENT_V5_ARTIFACT_CONTRACT_ID,
    CONTENT_V5_CONTROL_RESULT_SCOPE_ID,
    CONTENT_V5_EXECUTION_SCOPE_ID,
    CONTENT_V5_PRIMARY_RESULT_SCOPE_ID,
    CONTENT_V5_RECORD_CONTRACT_ID,
    CONTENT_V5_RUN_PREFIX,
    CONTENT_V5_STATE_SCHEMA_ID,
    ContentV5PairedProtocol,
    evaluate_content_v5_decision,
    load_content_v5_clean_protocol,
)
from cegwm.shared.keys import normalize_detection_key, public_key_digest

COMPLETE_EXECUTION = (
    "complete_for_content_v5_whitened_lf_adaptive_hf_branchwise_or_evaluation"
)
UMBRELLA_COMPLETE_EXECUTION = "complete_for_content_v5_control_then_primary_umbrella"
UMBRELLA_INCOMPLETE_EXECUTION = "incomplete_content_v5_umbrella_interruption"
COHORT_IDS = ("control_1", "primary_1")
COHORT_ROLES = {
    "control_1": "reference_cohort",
    "primary_1": "primary_evaluation",
}
FIXED_UNITS_PER_COHORT = 8
RECORDS_PER_UNIT = 2
FIXED_RECORDS_PER_COHORT = 16
_IDENTITY_FIELDS = (
    "run_id", "exact", "execution_scope_id", "protocol_id", "protocol_digest",
    "public_key_digest", "model_id", "ordered_cohorts", "ordered_arms",
    "record_contract_id", "fixed_units_per_cohort", "records_per_unit",
    "fixed_records_per_cohort", "artifact_contract_id",
)
_STATE_FIELDS = (
    "state_schema_id", "identity", "committed_whole_unit_count", "cohorts",
)
_COHORT_STATE_FIELDS = (
    "cohort_id", "cohort_role", "committed_unit_count", "records",
)


def _load_protocol(repo_root: Path) -> ContentV5PairedProtocol:
    root = repo_root / "configs" / "content_chain"
    return load_content_v5_clean_protocol(
        root / "content_v5_lf_or_hf_clean_v1.json",
        root / "content_v5_primary_evaluation_v1.jsonl",
        root / "content_adaptive_dual_branch_v2_clean.jsonl",
    )


def _load_cohort_protocol(repo_root: Path, cohort_id: str) -> ContentChainProtocol:
    return _load_protocol(repo_root).cohort_protocol(cohort_id)


def _variant(
    *,
    name: str,
    result_scope_id: str,
    load_protocol: Callable[[Path], ContentChainProtocol],
) -> engine.ContentRunnerVariant:
    return engine.ContentRunnerVariant(
        name=name,
        execution_scope_id=result_scope_id,
        complete_execution=COMPLETE_EXECUTION,
        arms=CONTENT_V5_ARMS,
        record_contract_id=CONTENT_V5_RECORD_CONTRACT_ID,
        state_schema_id=CONTENT_V5_STATE_SCHEMA_ID,
        run_prefix=CONTENT_V5_RUN_PREFIX,
        load_protocol=load_protocol,
        load_pipeline_and_assets=v4_runner._load_pipeline_and_assets,
        run_joint=v4_runner._run_joint,
        lf_scorer=v4_runner.score_content_v4_lf_image,
        decision_evaluator=evaluate_content_v5_decision,
    )


CONTENT_V5_CONTROL_RUNNER_VARIANT = _variant(
    name="Content V5 control_1",
    result_scope_id=CONTENT_V5_CONTROL_RESULT_SCOPE_ID,
    load_protocol=lambda root: _load_cohort_protocol(root, "control_1"),
)
CONTENT_V5_PRIMARY_RUNNER_VARIANT = _variant(
    name="Content V5 primary_1",
    result_scope_id=CONTENT_V5_PRIMARY_RESULT_SCOPE_ID,
    load_protocol=lambda root: _load_cohort_protocol(root, "primary_1"),
)
CONTENT_V5_RUNNER_VARIANTS = {
    "control_1": CONTENT_V5_CONTROL_RUNNER_VARIANT,
    "primary_1": CONTENT_V5_PRIMARY_RUNNER_VARIANT,
}


def _umbrella_identity(
    paired: ContentV5PairedProtocol,
    *,
    exact: str,
    key_digest: str,
) -> dict[str, Any]:
    run_id = f"{CONTENT_V5_RUN_PREFIX}-{paired.protocol_digest[:12]}-{key_digest[:12]}"
    return {
        "run_id": run_id,
        "exact": exact,
        "execution_scope_id": CONTENT_V5_EXECUTION_SCOPE_ID,
        "protocol_id": paired.protocol_id,
        "protocol_digest": paired.protocol_digest,
        "public_key_digest": key_digest,
        "model_id": paired.config["generation_runtime"]["model_id"],
        "ordered_cohorts": [
            {
                "cohort_id": cohort_id,
                "cohort_role": COHORT_ROLES[cohort_id],
                "ordered_roster": [
                    [unit.unit_id, unit.source_id]
                    for unit in paired.cohorts[cohort_id]
                ],
            }
            for cohort_id in COHORT_IDS
        ],
        "ordered_arms": list(CONTENT_V5_ARMS),
        "record_contract_id": CONTENT_V5_RECORD_CONTRACT_ID,
        "fixed_units_per_cohort": FIXED_UNITS_PER_COHORT,
        "records_per_unit": RECORDS_PER_UNIT,
        "fixed_records_per_cohort": FIXED_RECORDS_PER_COHORT,
        "artifact_contract_id": CONTENT_V5_ARTIFACT_CONTRACT_ID,
    }


def _new_state(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_schema_id": CONTENT_V5_STATE_SCHEMA_ID,
        "identity": identity,
        "committed_whole_unit_count": 0,
        "cohorts": [
            {
                "cohort_id": cohort_id,
                "cohort_role": COHORT_ROLES[cohort_id],
                "committed_unit_count": 0,
                "records": [],
            }
            for cohort_id in COHORT_IDS
        ],
    }


def _validate_state(
    state: Any,
    identity: dict[str, Any],
    paired: ContentV5PairedProtocol,
) -> dict[str, Any]:
    if not isinstance(state, dict) or tuple(state) != _STATE_FIELDS:
        raise ValueError("Content V5 audit state fields or order differ")
    if state["state_schema_id"] != CONTENT_V5_STATE_SCHEMA_ID:
        raise ValueError("Content V5 audit state schema differs")
    received_identity = state["identity"]
    if not isinstance(received_identity, dict) or tuple(received_identity) != _IDENTITY_FIELDS:
        raise ValueError("Content V5 audit identity fields or order differ")
    if not engine._same_json_bytes(received_identity, identity):
        raise ValueError("Content V5 audit identity differs")
    cohorts = state["cohorts"]
    if not isinstance(cohorts, list) or len(cohorts) != len(COHORT_IDS):
        raise ValueError("Content V5 audit cohort set differs")
    committed_total = 0
    for cohort_index, cohort_id in enumerate(COHORT_IDS):
        cohort = cohorts[cohort_index]
        if not isinstance(cohort, dict) or tuple(cohort) != _COHORT_STATE_FIELDS:
            raise ValueError("Content V5 cohort audit fields or order differ")
        if (
            cohort["cohort_id"] != cohort_id
            or cohort["cohort_role"] != COHORT_ROLES[cohort_id]
        ):
            raise ValueError("Content V5 cohort audit identity or order differs")
        committed = cohort["committed_unit_count"]
        if (
            not isinstance(committed, int)
            or isinstance(committed, bool)
            or not 0 <= committed <= FIXED_UNITS_PER_COHORT
        ):
            raise ValueError("Content V5 committed cohort count differs")
        if cohort_id == "primary_1" and committed and cohorts[0]["committed_unit_count"] != 8:
            raise ValueError("Content V5 primary audit history precedes control completion")
        records = cohort["records"]
        if not isinstance(records, list) or len(records) != committed * RECORDS_PER_UNIT:
            raise ValueError("Content V5 cohort records are not whole-unit transactions")
        protocol = paired.cohort_protocol(cohort_id)
        variant = CONTENT_V5_RUNNER_VARIANTS[cohort_id]
        for unit_index in range(committed):
            unit = protocol.roster[unit_index]
            transaction = records[unit_index * 2 : unit_index * 2 + 2]
            for arm_index, record in enumerate(transaction):
                if not isinstance(record, dict):
                    raise TypeError("Content V5 audit record must be a JSON object")
                engine._validate_content_v2_record(record, variant=variant)
                if (
                    record["run_id"] != identity["run_id"]
                    or record["unit_id"] != unit.unit_id
                    or record["source_cluster_id"] != unit.source_id
                    or record["arm"] != CONTENT_V5_ARMS[arm_index]
                    or record["code_revision"] != identity["exact"]
                    or record["config_digest"] != identity["protocol_digest"]
                    or record["key_public_digest"] != identity["public_key_digest"]
                ):
                    raise ValueError("Content V5 audit record identity or order differs")
            statuses = tuple(record["status"] for record in transaction)
            if statuses not in (
                ("success", "success"),
                ("operational_failure", "operational_failure"),
            ):
                raise ValueError("Content V5 audit transaction status is incomplete")
            if statuses[0] == "operational_failure" and (
                transaction[0]["failure_reason"] != transaction[1]["failure_reason"]
            ):
                raise ValueError("Content V5 audit transaction failure classes differ")
        committed_total += committed
    if state["committed_whole_unit_count"] != committed_total:
        raise ValueError("Content V5 total whole-unit checkpoint count differs")
    return state


def _failure_transaction(
    *,
    unit: Any,
    identity: dict[str, Any],
    variant: engine.ContentRunnerVariant,
    error: Exception,
) -> list[dict[str, Any]]:
    error_class = engine._public_operational_error_class(error)
    return [
        engine._content_v2_record(
            run_id=identity["run_id"],
            unit_id=unit.unit_id,
            source_cluster_id=unit.source_id,
            arm=arm,
            condition="clean",
            code_revision=identity["exact"],
            config_digest=identity["protocol_digest"],
            key_public_digest=identity["public_key_digest"],
            status="operational_failure",
            failure_reason=error_class,
            variant=variant,
        )
        for arm in CONTENT_V5_ARMS
    ]


def _incomplete_cohort_result(
    *,
    cohort_id: str,
    records: list[dict[str, Any]],
    identity: dict[str, Any],
    paired: ContentV5PairedProtocol,
) -> dict[str, Any]:
    protocol = paired.cohort_protocol(cohort_id)
    failures = [
        {
            "unit_id": protocol.roster[index].unit_id,
            "status": "failed",
            "error_type": records[index * 2]["failure_reason"],
        }
        for index in range(len(records) // 2)
        if records[index * 2]["status"] == "operational_failure"
    ]
    return {
        "rc": 2,
        "completeness": UMBRELLA_INCOMPLETE_EXECUTION,
        "scientific_outcome_allowed": False,
        "scientific_status": "not_evaluable",
        "execution_scope_id": CONTENT_V5_RUNNER_VARIANTS[cohort_id].execution_scope_id,
        "exact": identity["exact"],
        "protocol_id": paired.protocol_id,
        "protocol_digest": paired.protocol_digest,
        "public_key_digest": identity["public_key_digest"],
        "fixed_denominator_units": FIXED_UNITS_PER_COHORT,
        "fixed_records": FIXED_RECORDS_PER_COHORT,
        "committed_unit_count": len(records) // 2,
        "records": records,
        "unit_aggregate_metrics": [],
        "failed_units": failures,
        "gate_evidence": None,
        "limitations": list(paired.config["limitations"]),
    }


def _cohort_reports(
    state: dict[str, Any],
    identity: dict[str, Any],
    paired: ContentV5PairedProtocol,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for cohort_index, cohort_id in enumerate(COHORT_IDS):
        cohort_state = state["cohorts"][cohort_index]
        records = cohort_state["records"]
        if cohort_state["committed_unit_count"] == FIXED_UNITS_PER_COHORT:
            result = engine._derive_result(
                records,
                paired.cohort_protocol(cohort_id),
                identity,
                variant=CONTENT_V5_RUNNER_VARIANTS[cohort_id],
            )
        else:
            result = _incomplete_cohort_result(
                cohort_id=cohort_id,
                records=records,
                identity=identity,
                paired=paired,
            )
        reports.append({
            "cohort_id": cohort_id,
            "cohort_role": COHORT_ROLES[cohort_id],
            "result_scope_id": CONTENT_V5_RUNNER_VARIANTS[cohort_id].execution_scope_id,
            "result": result,
        })
    return reports


def _umbrella_result(
    state: dict[str, Any],
    identity: dict[str, Any],
    paired: ContentV5PairedProtocol,
    *,
    fatal_error: Exception | None,
) -> dict[str, Any]:
    reports = _cohort_reports(state, identity, paired)
    both_attempted = all(
        cohort["committed_unit_count"] == FIXED_UNITS_PER_COHORT
        for cohort in state["cohorts"]
    )
    rc = 0 if fatal_error is None and both_attempted and all(
        report["result"]["rc"] == 0 for report in reports
    ) else 2
    return {
        "rc": rc,
        "completeness": (
            UMBRELLA_COMPLETE_EXECUTION if rc == 0 else UMBRELLA_INCOMPLETE_EXECUTION
        ),
        "run_id": identity["run_id"],
        "execution_scope_id": identity["execution_scope_id"],
        "exact": identity["exact"],
        "protocol_id": identity["protocol_id"],
        "protocol_digest": identity["protocol_digest"],
        "public_key_digest": identity["public_key_digest"],
        "cohorts_in_order": list(COHORT_IDS),
        "fixed_denominator_units_per_cohort": FIXED_UNITS_PER_COHORT,
        "fixed_records_per_cohort": FIXED_RECORDS_PER_COHORT,
        "committed_whole_unit_count": state["committed_whole_unit_count"],
        "both_cohorts_attempted": both_attempted,
        "operational_error_class": (
            None if fatal_error is None
            else engine._public_operational_error_class(fatal_error)
        ),
        "cohort_results": reports,
        "pooled_decision_absent": True,
        "cross_cohort_conjunction": False,
        "reference_result_controls_primary_execution": False,
        "umbrella_rc_operational_only": True,
        "scientific_decision_scope": "independent_cohort_results_only",
        "limitations": list(paired.config["limitations"]),
    }


def _receipt(identity: dict[str, Any], committed: int) -> dict[str, Any]:
    return {
        "artifact_kind": "terminal",
        "artifact_contract_id": CONTENT_V5_ARTIFACT_CONTRACT_ID,
        "run_id": identity["run_id"],
        "exact": identity["exact"],
        "execution_scope_id": identity["execution_scope_id"],
        "protocol_id": identity["protocol_id"],
        "protocol_digest": identity["protocol_digest"],
        "public_key_digest": identity["public_key_digest"],
        "cohorts_in_order": list(COHORT_IDS),
        "fixed_units_per_cohort": FIXED_UNITS_PER_COHORT,
        "fixed_records_per_cohort": FIXED_RECORDS_PER_COHORT,
        "committed_whole_unit_count": committed,
        "result_member": "result.json",
        "audit_state_member": "audit-state.json",
        "external_validation_required": True,
    }


def _publish_terminal(
    local_run_root: Path,
    sink_run_root: Path,
    identity: dict[str, Any],
    state: dict[str, Any],
    result: dict[str, Any],
) -> None:
    engine._publish_pair(
        local_run_root=local_run_root,
        sink_run_root=sink_run_root,
        archive_name=f"{identity['run_id']}.zip",
        members=(
            ("receipt.json", engine._json_bytes(
                _receipt(identity, state["committed_whole_unit_count"])
            )),
            ("result.json", engine._json_bytes(result)),
            ("audit-state.json", engine._json_bytes(state)),
        ),
    )


def _progress(identity: Mapping[str, Any], committed: int, phase: str) -> None:
    print("CEGWM_PROGRESS " + json.dumps({
        "run_id": identity["run_id"],
        "committed": committed,
        "fixed_total": FIXED_UNITS_PER_COHORT * len(COHORT_IDS),
        "phase": phase,
    }, separators=(",", ":")), flush=True)


def _summary(identity: Mapping[str, Any], committed: int, rc: int) -> None:
    print("CEGWM_SUMMARY " + json.dumps({
        "run_id": identity["run_id"],
        "committed": committed,
        "fixed_total": FIXED_UNITS_PER_COHORT * len(COHORT_IDS),
        "rc": rc,
        "phase": "terminal",
    }, separators=(",", ":")), flush=True)


def _execute_fresh(
    args: argparse.Namespace,
    *,
    key: bytes,
    token: str,
) -> int:
    repo_root = Path(args.repo_root).resolve()
    local_work_root = Path(args.local_work_root).resolve()
    artifact_sink = Path(args.artifact_sink).resolve()
    exact = engine._git_exact(repo_root, args.expected_exact)
    paired = _load_protocol(repo_root)
    key_digest = public_key_digest(key)
    identity = _umbrella_identity(paired, exact=exact, key_digest=key_digest)
    if not token.strip():
        key = b""
        token = ""
        raise RuntimeError("HF_TOKEN_is_required_for_fresh_execution")
    local_run_root = local_work_root / identity["run_id"]
    sink_run_root = artifact_sink / identity["run_id"]
    if local_run_root.exists() or sink_run_root.exists():
        key = b""
        token = ""
        raise FileExistsError("Content V5 run root already exists; resume and retry are forbidden")
    local_run_root.mkdir(parents=True, exist_ok=False)
    local_state_path = local_run_root / "audit-state.json"
    state = _new_state(identity)
    _validate_state(state, identity, paired)
    engine._write_local_state(local_state_path, state)
    _progress(identity, 0, "identity_ready")
    fatal_error: Exception | None = None
    try:
        try:
            pipeline, assets = v4_runner._load_pipeline_and_assets(identity["model_id"], token)
        finally:
            token = ""
        wrong_keys = engine._wrong_keys(key, paired.cohort_protocol("control_1"))
        for cohort_index, cohort_id in enumerate(COHORT_IDS):
            protocol = paired.cohort_protocol(cohort_id)
            variant = CONTENT_V5_RUNNER_VARIANTS[cohort_id]
            for unit_index, unit in enumerate(protocol.roster):
                try:
                    transaction = engine._unit_transaction(
                        unit=unit,
                        pipeline=pipeline,
                        assets=assets,
                        key=key,
                        wrong_keys=wrong_keys,
                        identity=identity,
                        protocol=protocol,
                        variant=variant,
                    )
                except Exception as error:  # noqa: BLE001 - retain the fixed denominator
                    transaction = _failure_transaction(
                        unit=unit,
                        identity=identity,
                        variant=variant,
                        error=error,
                    )
                prospective = {
                    "state_schema_id": state["state_schema_id"],
                    "identity": state["identity"],
                    "committed_whole_unit_count": state["committed_whole_unit_count"] + 1,
                    "cohorts": [
                        {
                            "cohort_id": item["cohort_id"],
                            "cohort_role": item["cohort_role"],
                            "committed_unit_count": item["committed_unit_count"],
                            "records": list(item["records"]),
                        }
                        for item in state["cohorts"]
                    ],
                }
                prospective_cohort = prospective["cohorts"][cohort_index]
                prospective_cohort["committed_unit_count"] = unit_index + 1
                prospective_cohort["records"].extend(transaction)
                _validate_state(prospective, identity, paired)
                engine._write_local_state(local_state_path, prospective)
                state = prospective
                _progress(
                    identity,
                    state["committed_whole_unit_count"],
                    f"{cohort_id}_unit_committed",
                )
    except Exception as error:  # noqa: BLE001 - sanitized terminal failure only
        fatal_error = error
    finally:
        key = b""
        token = ""
    result = _umbrella_result(
        state,
        identity,
        paired,
        fatal_error=fatal_error,
    )
    rc = int(result["rc"])
    _publish_terminal(local_run_root, sink_run_root, identity, state, result)
    _summary(identity, state["committed_whole_unit_count"], rc)
    return rc


def execute(args: argparse.Namespace) -> int:
    key_text = os.environ.pop(engine.KEY_ENV, "")
    token = os.environ.pop(engine.TOKEN_ENV, "")
    if not key_text.strip():
        token = ""
        raise RuntimeError("CEG_WM_ROOT_KEY_is_required")
    key = normalize_detection_key(key_text)
    key_text = ""
    try:
        return _execute_fresh(args, key=key, token=token)
    finally:
        key = b""
        token = ""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--local-work-root", required=True)
    parser.add_argument("--artifact-sink", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
