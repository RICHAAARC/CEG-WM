"""Fixed paired HF/LF clean-reference and attack-complementarity runner."""

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
import time
from typing import Any
import zipfile

import numpy as np
import torch

from cegwm.method.hf import (
    HF_CANDIDATE_ID,
    HF_INJECTION_STEP_INDEX,
    FrozenHFPublicAssets,
    score_hf_image,
)
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    LF_INJECTION_STEP_INDEX,
    FrozenLFPublicAssets,
    score_lf_image,
)
from cegwm.protocol.records import StageARecord
from cegwm.protocol.stage_a import (
    StageAProtocol,
    load_hf_lf_attack_complementarity_protocol,
)
from cegwm.runtime.diffusers_sd35 import (
    load_sd35_pipeline,
    run_sd35_hf,
    run_sd35_lf,
    run_sd35_plain,
)
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from cegwm.shared.prg import prg_bytes

from experiments.stage_a.attack_transforms import (
    ATTACK_IDS,
    CONDITION_ORDER,
    IDENTITY_REFERENCE,
    apply_attack,
    public_noise_domain,
)

KEY_ENV = "CEG_WM_ROOT_KEY"
TOKEN_ENV = "HF_TOKEN"
CHECKPOINT_INTERVAL_HOURS = 2.0
EXECUTION_SCOPE_ID = (
    "hf_lf_paired_clean_reference_and_non_geometric_attack_complementarity_v1"
)
COMPLETE_EXECUTION = (
    "complete_for_hf_lf_paired_clean_reference_and_"
    "attack_complementarity_execution"
)
INCOMPLETE_EXECUTION = "incomplete_operational_execution"
SCIENTIFIC_STATUS = "not_adjudicated"
LIMITATIONS = (
    "paired_clean_reference_plus_three_preregistered_non_geometric_attacks_only",
    "Pillow_version_and_negligible_numerical_platform_drift_recorded_not_gated",
    "lpips_not_evaluated",
    "no_calibrated_threshold_or_fixed_fpr_claim",
    "no_score_retention_ratio_or_cross_detector_raw_score_comparison",
    "no_hf_lf_fusion_routing_or_geometry_evaluation",
    "model_revision_and_weight_digest_not_recorded",
)
HF_EVALUATED_CANDIDATE_ID = "hf_tail_rademacher_v1_rankgate_v2"
HF_DETECTOR_STATISTIC_ID = "vae_reencode_hf_masked_normalized_correlation"
RECORD_ARMS = (
    HF_EVALUATED_CANDIDATE_ID,
    f"primary_null__{HF_EVALUATED_CANDIDATE_ID}",
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    f"primary_null__{LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID}",
)
_FATAL_ERROR_BY_PHASE = {
    "initialization": "initialization_failure",
    "resume_validation": "resume_validation_failure",
    "runtime_execution": "runtime_execution_failure",
    "checkpoint": "checkpoint_failure",
    "final_export": "final_export_failure",
}


def _completeness_for_rc(rc: int) -> str:
    return COMPLETE_EXECUTION if rc == 0 else INCOMPLETE_EXECUTION


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _json_write(temporary, payload)
    os.replace(temporary, path)


def _git_exact(repo_root: Path, expected_exact: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected_exact) is None:
        raise ValueError("expected exact must be a lowercase 40-character revision")
    resolved = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != expected_exact:
        raise RuntimeError("resolved revision differs from approved execution exact")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("execution checkout must be clean")
    return resolved


def _load_protocol(repo_root: Path) -> StageAProtocol:
    config_root = repo_root / "configs" / "stage_a"
    return load_hf_lf_attack_complementarity_protocol(
        config_root / "stage_a_hf_lf_attack_complementarity_v1.json",
        config_root / "hf_lf_attack_complementarity.jsonl",
    )


def _method_identities(protocol: StageAProtocol) -> dict[str, dict[str, str]]:
    return {
        method: dict(protocol.config["methods"][method])
        for method in ("hf", "lf")
    }


def _load_pipeline_and_assets(
    model_id: str,
    hf_token: str,
) -> tuple[Any, FrozenHFPublicAssets, FrozenLFPublicAssets]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_colab_execution")
    pipeline = load_sd35_pipeline(model_id, torch_dtype=torch.float16, token=hf_token)
    vae = getattr(pipeline, "vae", None)
    image_processor = getattr(pipeline, "image_processor", None)
    hf_assets = FrozenHFPublicAssets(
        vae=vae,
        image_processor=image_processor,
        image_processor_id=f"{model_id}:image_processor",
    )
    lf_assets = FrozenLFPublicAssets(
        vae=vae,
        image_processor=image_processor,
        image_processor_id=f"{model_id}:image_processor",
        candidate_id=LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        detector_statistic_id=LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        evaluated_candidate_id=LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    pipeline.to("cuda")
    return pipeline, hf_assets, lf_assets


def _wrong_keys(detection_key: bytes, protocol: StageAProtocol) -> tuple[bytes, ...]:
    keying = protocol.config["keying"]
    domain = keying["wrong_key_derivation_domain"]
    count = keying["wrong_key_count"]
    return tuple(
        prg_bytes(detection_key, f"{domain}/index={index}", 32)
        for index in range(count)
    )


def _scores(
    image: Any,
    detection_key: bytes,
    wrong_keys: tuple[bytes, ...],
    assets: FrozenHFPublicAssets | FrozenLFPublicAssets,
) -> dict[str, float]:
    scorer = score_hf_image if isinstance(assets, FrozenHFPublicAssets) else score_lf_image
    values = {"registered": float(scorer(image, detection_key, assets))}
    for index, wrong_key in enumerate(wrong_keys):
        values[f"wrong_{index:02d}"] = float(scorer(image, wrong_key, assets))
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("nonfinite_blind_score")
    return values


def _psnr(first: Any, second: Any) -> float:
    first_pixels = np.asarray(first, dtype=np.float64) / 255.0
    second_pixels = np.asarray(second, dtype=np.float64) / 255.0
    if first_pixels.shape != second_pixels.shape:
        raise ValueError("paired_image_shape_mismatch")
    mse = float(np.mean(np.square(first_pixels - second_pixels)))
    if mse <= 0.0 or not math.isfinite(mse):
        raise ValueError("paired_psnr_not_finite")
    value = -10.0 * math.log10(mse)
    if not math.isfinite(value):
        raise ValueError("paired_psnr_not_finite")
    return value


def _attacked_vs_pre_attack_psnr(attacked: Any, pre_attack: Any) -> float | None:
    first_pixels = np.asarray(attacked, dtype=np.float64) / 255.0
    second_pixels = np.asarray(pre_attack, dtype=np.float64) / 255.0
    if first_pixels.shape != second_pixels.shape:
        raise ValueError("attack_image_shape_mismatch")
    mse = float(np.mean(np.square(first_pixels - second_pixels)))
    if not math.isfinite(mse):
        raise ValueError("attack_psnr_not_finite")
    if mse == 0.0:
        return None
    value = -10.0 * math.log10(mse)
    if not math.isfinite(value):
        raise ValueError("attack_psnr_not_finite")
    return value


def _failure_transaction(
    unit: Any,
    protocol: StageAProtocol,
    run_id: str,
    revision: str,
    key_digest: str,
    reason: str,
) -> list[StageARecord]:
    return [
        StageARecord(
            run_id=run_id,
            unit_id=unit.unit_id,
            source_cluster_id=unit.source_id,
            arm=arm,
            condition=condition,
            code_revision=revision,
            config_digest=protocol.protocol_digest,
            key_public_digest=key_digest,
            status="operational_failure",
            failure_reason=reason,
        )
        for condition in CONDITION_ORDER
        for arm in RECORD_ARMS
    ]


def _new_state(
    *,
    run_id: str,
    resolved_exact: str,
    protocol: StageAProtocol,
    model_id: str,
    key_digest: str,
) -> dict[str, Any]:
    methods = _method_identities(protocol)
    rule = protocol.config["decision_rule"]
    return {
        "run_id": run_id,
        "execution_scope_id": EXECUTION_SCOPE_ID,
        "resolved_exact": resolved_exact,
        "protocol_digest": protocol.protocol_digest,
        "method_identities": methods,
        "condition_order": list(CONDITION_ORDER),
        "attack_ids": list(ATTACK_IDS),
        "record_arms_in_exact_condition_order": list(RECORD_ARMS),
        "ordered_roster_unit_ids": [
            unit.unit_id for unit in protocol.candidate_selection
        ],
        "ordered_roster_source_ids": [
            unit.source_id for unit in protocol.candidate_selection
        ],
        "model_id": model_id,
        "key_public_digest": key_digest,
        "rank_gate_a_min_units": rule[
            "gate_a_registered_top_rank_among_17_min_units"
        ],
        "rank_gate_b_min_units": rule[
            "gate_b_method_registered_gt_same_condition_primary_null_min_units"
        ],
        "checkpoint_interval_hours": CHECKPOINT_INTERVAL_HOURS,
        "checkpoint_sequence": 0,
        "committed_unit_count": 0,
        "committed_unit_ids": [],
        "records": [],
    }


def _resume_state(
    resume_zip: Path,
    resume_checksum: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    checksum_parts = resume_checksum.read_text(encoding="utf-8").strip().split()
    if len(checksum_parts) != 2 or checksum_parts[1] != resume_zip.name:
        raise ValueError("resume checksum file is malformed")
    if hashlib.sha256(resume_zip.read_bytes()).hexdigest() != checksum_parts[0]:
        raise ValueError("resume checkpoint checksum mismatch")
    with zipfile.ZipFile(resume_zip) as archive:
        if archive.namelist() != ["state.json"]:
            raise ValueError("resume checkpoint must contain only state.json")
        state = json.loads(archive.read("state.json"))
    identity_fields = (
        "run_id",
        "execution_scope_id",
        "resolved_exact",
        "protocol_digest",
        "method_identities",
        "condition_order",
        "attack_ids",
        "record_arms_in_exact_condition_order",
        "ordered_roster_unit_ids",
        "ordered_roster_source_ids",
        "model_id",
        "key_public_digest",
        "rank_gate_a_min_units",
        "rank_gate_b_min_units",
        "checkpoint_interval_hours",
    )
    if any(state.get(field) != expected.get(field) for field in identity_fields):
        raise ValueError("resume checkpoint identity mismatch")
    committed = state.get("committed_unit_ids")
    roster = expected["ordered_roster_unit_ids"]
    records = state.get("records")
    if not isinstance(committed, list) or committed != roster[: len(committed)]:
        raise ValueError("resume committed units must be an ordered roster prefix")
    if not committed:
        raise ValueError("resume checkpoint cannot be empty")
    if not isinstance(records, list) or len(records) != len(committed) * 16:
        raise ValueError("resume checkpoint record count mismatch")
    if state.get("committed_unit_count") != len(committed):
        raise ValueError("resume checkpoint committed count mismatch")
    for index, unit_id in enumerate(committed):
        transaction = records[index * 16 : index * 16 + 16]
        if [record.get("unit_id") for record in transaction] != [unit_id] * 16:
            raise ValueError("resume checkpoint record roster mismatch")
        validated = [StageARecord(**record) for record in transaction]
        expected_pairs = [
            (condition, arm)
            for condition in CONDITION_ORDER
            for arm in RECORD_ARMS
        ]
        if [(record.condition, record.arm) for record in validated] != expected_pairs:
            raise ValueError("resume checkpoint 16-record transaction mismatch")
        for record in validated:
            if (
                record.run_id != expected["run_id"]
                or record.code_revision != expected["resolved_exact"]
                or record.config_digest != expected["protocol_digest"]
                or record.key_public_digest != expected["key_public_digest"]
                or record.source_cluster_id
                != expected["ordered_roster_source_ids"][index]
            ):
                raise ValueError("resume checkpoint record identity mismatch")
    sequence = state.get("checkpoint_sequence")
    if not isinstance(sequence, int) or sequence < 1:
        raise ValueError("resume checkpoint sequence is invalid")
    return state


def _deterministic_run_id(
    resolved_exact: str,
    protocol: StageAProtocol,
    model_id: str,
    key_digest: str,
) -> str:
    identity = {
        "resolved_exact": resolved_exact,
        "execution_scope_id": EXECUTION_SCOPE_ID,
        "protocol_digest": protocol.protocol_digest,
        "method_identities": _method_identities(protocol),
        "condition_order": list(CONDITION_ORDER),
        "model_id": model_id,
        "key_public_digest": key_digest,
        "checkpoint_interval_hours": CHECKPOINT_INTERVAL_HOURS,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(
        b"CEG-WM/stage-a-hf-lf-attack-complementarity/run-id/v1\x00"
        + canonical.encode("utf-8")
    )
    return f"hlfac-{digest.hexdigest()[:24]}"


def _verify_checksum(zip_path: Path, checksum_path: Path) -> str:
    parts = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != zip_path.name:
        raise ValueError("artifact checksum file is malformed")
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if parts[0] != digest:
        raise ValueError("artifact checksum mismatch")
    return digest


def _attack_complementarity_evidence(
    records: list[StageARecord],
    expected: dict[str, Any],
    *,
    scientific_outcome_allowed: bool,
) -> dict[str, Any]:
    """Apply the paired-clean prerequisite before frozen attack comparison."""

    roster = expected["ordered_roster_unit_ids"]
    source_by_unit = dict(
        zip(roster, expected["ordered_roster_source_ids"], strict=True)
    )
    by_unit: dict[str, list[StageARecord]] = {unit_id: [] for unit_id in roster}
    for record in records:
        if record.unit_id not in by_unit:
            raise ValueError(
                "attack-complementarity evidence contains a unit outside the fixed roster"
            )
        if (
            record.run_id != expected["run_id"]
            or record.code_revision != expected["resolved_exact"]
            or record.config_digest != expected["protocol_digest"]
            or record.key_public_digest != expected["key_public_digest"]
            or record.source_cluster_id != source_by_unit[record.unit_id]
        ):
            raise ValueError("attack-complementarity evidence record identity mismatch")
        by_unit[record.unit_id].append(record)

    method_ids = {
        method: expected["method_identities"][method]["evaluated_candidate_id"]
        for method in ("hf", "lf")
    }
    accumulators = {
        condition: {
            method: {
                "gate_a_count": 0,
                "gate_b_count": 0,
                "margins": [],
                "paired_lifts": [],
                "paired_psnr": [],
                "attack_psnr": [],
            }
            for method in ("hf", "lf")
        }
        for condition in CONDITION_ORDER
    }
    successful_unit_count = 0
    unit_evidence: list[dict[str, Any]] = []
    wrong_fields = tuple(f"wrong_{index:02d}" for index in range(16))
    expected_score_fields = {"registered", *wrong_fields}
    for unit_id in roster:
        transaction = by_unit[unit_id]
        if not transaction:
            unit_evidence.append({
                "unit_id": unit_id,
                "status": "uncommitted",
                "candidates": {},
            })
            continue
        expected_pairs = [
            (condition, arm)
            for condition in CONDITION_ORDER
            for arm in RECORD_ARMS
        ]
        if (
            len(transaction) != 16
            or [(record.condition, record.arm) for record in transaction] != expected_pairs
        ):
            raise ValueError("evidence requires the exact ordered 16-record unit transaction")
        if any(record.status != "success" for record in transaction):
            unit_evidence.append({
                "unit_id": unit_id,
                "status": "operational_failure",
                "conditions": {},
            })
            continue
        if any(set(record.scores) != expected_score_fields for record in transaction):
            raise ValueError("attack-complementarity evidence score roster mismatch")
        values = [
            float(value)
            for record in transaction
            for value in record.scores.values()
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("attack-complementarity evidence contains a nonfinite score")
        conditions: dict[str, Any] = {}
        for condition_index, condition in enumerate(CONDITION_ORDER):
            condition_records = transaction[
                condition_index * 4 : condition_index * 4 + 4
            ]
            method_facts: dict[str, Any] = {}
            for method, offset in (("hf", 0), ("lf", 2)):
                candidate_record = condition_records[offset]
                null_record = condition_records[offset + 1]
                registered = float(candidate_record.scores["registered"])
                wrong_max = max(
                    float(candidate_record.scores[field]) for field in wrong_fields
                )
                null_registered = float(null_record.scores["registered"])
                margin = registered - wrong_max
                paired_lift = registered - null_registered
                gate_a = registered > wrong_max
                gate_b = registered > null_registered
                paired_psnr = float(
                    candidate_record.metrics.get("paired_rgb_psnr", float("nan"))
                )
                budget = float(
                    candidate_record.metrics.get(
                        "actual_dtype_relative_l2", float("nan")
                    )
                )
                if not (
                    math.isfinite(paired_psnr)
                    and math.isfinite(budget)
                    and 0.0 < budget <= 0.012
                ):
                    raise ValueError("evidence contains invalid quality or budget evidence")
                attack_psnr: float | None = None
                if condition in ATTACK_IDS:
                    raw_attack_psnr = candidate_record.metrics.get(
                        "attacked_vs_pre_attack_psnr"
                    )
                    if raw_attack_psnr is not None:
                        attack_psnr = float(raw_attack_psnr)
                        if not math.isfinite(attack_psnr):
                            raise ValueError("reported attack PSNR must be finite")
                elif "attacked_vs_pre_attack_psnr" in candidate_record.metrics:
                    raise ValueError("identity_reference attack PSNR must be not applicable")
                accumulator = accumulators[condition][method]
                accumulator["gate_a_count"] += int(gate_a)
                accumulator["gate_b_count"] += int(gate_b)
                accumulator["margins"].append(margin)
                accumulator["paired_lifts"].append(paired_lift)
                accumulator["paired_psnr"].append(paired_psnr)
                if attack_psnr is not None:
                    accumulator["attack_psnr"].append(attack_psnr)
                method_facts[method] = {
                    "registered_top_rank": gate_a,
                    "registered_gt_same_condition_primary_null": gate_b,
                    "correct_minus_wrong_key_max": margin,
                    "registered_minus_primary_null_registered": paired_lift,
                    "wrong_key_max": wrong_max,
                    "primary_null_registered": null_registered,
                    "paired_rgb_psnr": paired_psnr,
                    "attacked_vs_pre_attack_psnr": attack_psnr,
                    "actual_dtype_relative_l2": budget,
                }
            conditions[condition] = method_facts
        successful_unit_count += 1
        unit_evidence.append({
            "unit_id": unit_id,
            "status": "success",
            "conditions": conditions,
        })

    complete = successful_unit_count == len(roster) == 8 and len(records) == 128
    outcome_permitted = scientific_outcome_allowed and complete
    summaries_by_condition: dict[str, Any] = {}
    for condition in CONDITION_ORDER:
        summaries_by_condition[condition] = {}
        for method in ("hf", "lf"):
            accumulator = accumulators[condition][method]
            available = len(accumulator["margins"]) == 8
            summaries_by_condition[condition][method] = {
                "evaluated_candidate_id": method_ids[method],
                "gate_a_registered_top_rank_units": accumulator["gate_a_count"],
                "gate_a_required_units": expected["rank_gate_a_min_units"],
                "gate_a_pass": None,
                "gate_b_registered_gt_primary_null_units": accumulator["gate_b_count"],
                "gate_b_required_units": expected["rank_gate_b_min_units"],
                "gate_b_pass": None,
                "median_correct_minus_wrong_key_max_effect_size": (
                    float(np.median(accumulator["margins"])) if available else None
                ),
                "mean_correct_minus_wrong_key_max_effect_size": (
                    float(np.mean(accumulator["margins"])) if available else None
                ),
                "min_correct_minus_wrong_key_max_effect_size": (
                    float(min(accumulator["margins"])) if available else None
                ),
                "median_registered_minus_primary_null_effect_size": (
                    float(np.median(accumulator["paired_lifts"])) if available else None
                ),
                "median_paired_rgb_psnr": (
                    float(np.median(accumulator["paired_psnr"])) if available else None
                ),
                "median_attacked_vs_pre_attack_psnr": (
                    float(np.median(accumulator["attack_psnr"]))
                    if condition in ATTACK_IDS and accumulator["attack_psnr"]
                    else None
                ),
            }

    clean_both_methods_pass: bool | None = None
    attack_complementarity_pass: bool | None = None
    complementary_attack_ids: list[str] | None = None
    attack_complementarity_outcome: str | None = None
    evaluation_status = "not_evaluable_operational"
    if outcome_permitted:
        for method in ("hf", "lf"):
            facts = summaries_by_condition[IDENTITY_REFERENCE][method]
            facts["gate_a_pass"] = (
                facts["gate_a_registered_top_rank_units"]
                >= expected["rank_gate_a_min_units"]
            )
            facts["gate_b_pass"] = (
                facts["gate_b_registered_gt_primary_null_units"]
                >= expected["rank_gate_b_min_units"]
            )
        clean_both_methods_pass = all(
            summaries_by_condition[IDENTITY_REFERENCE][method]["gate_a_pass"]
            and summaries_by_condition[IDENTITY_REFERENCE][method]["gate_b_pass"]
            for method in ("hf", "lf")
        )
        if not clean_both_methods_pass:
            evaluation_status = "paired_clean_prerequisite_failed"
            attack_complementarity_outcome = (
                "SCIENTIFIC_NEGATIVE_FOR_PAIRED_CLEAN_PREREQUISITE_"
                "ATTACK_COMPLEMENTARITY_NOT_EVALUABLE_AND_STOP"
            )
        else:
            evaluation_status = "attack_complementarity_outcome"
            complementary_attack_ids = []
            for attack_id in ATTACK_IDS:
                for method in ("hf", "lf"):
                    facts = summaries_by_condition[attack_id][method]
                    facts["gate_a_pass"] = (
                        facts["gate_a_registered_top_rank_units"]
                        >= expected["rank_gate_a_min_units"]
                    )
                    facts["gate_b_pass"] = (
                        facts["gate_b_registered_gt_primary_null_units"]
                        >= expected["rank_gate_b_min_units"]
                    )
                    facts["method_survives_attack"] = (
                        facts["gate_a_pass"] and facts["gate_b_pass"]
                    )
                if (
                    summaries_by_condition[attack_id]["lf"]["method_survives_attack"]
                    and not summaries_by_condition[attack_id]["hf"]["method_survives_attack"]
                ):
                    complementary_attack_ids.append(attack_id)
            attack_complementarity_pass = bool(complementary_attack_ids)
            attack_complementarity_outcome = (
                "attack_complementarity_pass_candidate_for_agent5_adjudication"
                if attack_complementarity_pass
                else "SCIENTIFIC_NEGATIVE_FOR_COMPLEMENTARITY_AND_STOP"
            )

    clean_unit_evidence: list[dict[str, Any]] = []
    attack_unit_evidence: dict[str, list[dict[str, Any]]] = {
        attack_id: [] for attack_id in ATTACK_IDS
    }
    for unit in unit_evidence:
        clean_entry = {"unit_id": unit["unit_id"], "status": unit["status"]}
        if unit["status"] == "success":
            clean_entry["methods"] = unit["conditions"][IDENTITY_REFERENCE]
        else:
            clean_entry["methods"] = {}
        clean_unit_evidence.append(clean_entry)
        for attack_id in ATTACK_IDS:
            attack_entry = {"unit_id": unit["unit_id"], "status": unit["status"]}
            if unit["status"] == "success":
                attack_entry["methods"] = unit["conditions"][attack_id]
            else:
                attack_entry["methods"] = {}
            attack_unit_evidence[attack_id].append(attack_entry)

    paired_clean_prerequisite = {
        "condition_id": IDENTITY_REFERENCE,
        "identity_reference_is_attack": False,
        "hf": summaries_by_condition[IDENTITY_REFERENCE]["hf"],
        "lf": summaries_by_condition[IDENTITY_REFERENCE]["lf"],
        "both_methods_pass": clean_both_methods_pass,
        "unit_evidence": clean_unit_evidence,
    }
    attack_conditions: dict[str, Any] = {}
    for attack_id in ATTACK_IDS:
        for method in ("hf", "lf"):
            summaries_by_condition[attack_id][method].setdefault(
                "method_survives_attack", None
            )
        hf_survives = summaries_by_condition[attack_id]["hf"].get(
            "method_survives_attack"
        )
        lf_survives = summaries_by_condition[attack_id]["lf"].get(
            "method_survives_attack"
        )
        complementarity_condition = (
            bool(lf_survives and not hf_survives)
            if clean_both_methods_pass is True
            else None
        )
        attack_conditions[attack_id] = {
            "hf": summaries_by_condition[attack_id]["hf"],
            "lf": summaries_by_condition[attack_id]["lf"],
            "complementarity_condition": complementarity_condition,
            "unit_evidence": attack_unit_evidence[attack_id],
        }
    return {
        "scientific_outcome_allowed": outcome_permitted,
        "evaluation_status": evaluation_status,
        "fixed_unit_count": len(roster),
        "fixed_condition_count": 4,
        "fixed_attack_count": 3,
        "fixed_record_count": 128,
        "unit_transaction_record_count": 16,
        "paired_clean_prerequisite": paired_clean_prerequisite,
        "attack_conditions": attack_conditions,
        "attack_complementarity_pass": attack_complementarity_pass,
        "complementary_attack_ids": complementary_attack_ids,
        "attack_complementarity_outcome": attack_complementarity_outcome,
        "median_margin_is_gate": False,
        "primary_null_cutoff_is_gate": False,
        "score_retention_ratio_is_gate": False,
        "cross_detector_raw_score_comparison": False,
        "formal_fpr_claim": False,
    }


def _discover_checkpoint(run_store: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    checkpoint_names: set[str] = set()
    pattern = re.compile(r"checkpoint-(\d{4})-units-(\d{4})\.zip")
    for zip_path in sorted(run_store.glob("checkpoint-*.zip")):
        match = pattern.fullmatch(zip_path.name)
        if match is None:
            raise ValueError("checkpoint filename is malformed")
        checksum_path = run_store / f"{zip_path.name}.sha256"
        if not checksum_path.is_file():
            raise ValueError("checkpoint checksum is missing")
        state = _resume_state(zip_path, checksum_path, expected)
        sequence = int(match.group(1))
        committed_count = int(match.group(2))
        if state["checkpoint_sequence"] != sequence:
            raise ValueError("checkpoint sequence filename mismatch")
        if state["committed_unit_count"] != committed_count:
            raise ValueError("checkpoint count filename mismatch")
        candidates.append((sequence, committed_count, state))
        checkpoint_names.update({zip_path.name, checksum_path.name})
    orphan_checksums = {
        path.name for path in run_store.glob("checkpoint-*.zip.sha256")
    } - checkpoint_names
    if orphan_checksums:
        raise ValueError("checkpoint checksum has no matching ZIP")
    ranks = [(sequence, count) for sequence, count, _ in candidates]
    if len({sequence for sequence, _ in ranks}) != len(ranks):
        raise ValueError("checkpoint sequence is ambiguous")
    ordered = sorted(ranks)
    if any(later[1] <= earlier[1] for earlier, later in zip(ordered, ordered[1:])):
        raise ValueError("checkpoint committed counts are ambiguous")
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _checkpoint(state: dict[str, Any], output_dir: Path, checkpoint_sink: Path) -> None:
    sequence = int(state["checkpoint_sequence"]) + 1
    committed_count = len(state["committed_unit_ids"])
    checkpoint_state = dict(state)
    checkpoint_state["checkpoint_sequence"] = sequence
    state_path = output_dir / "state.json"
    _atomic_json_write(state_path, checkpoint_state)
    stem = f"checkpoint-{sequence:04d}-units-{committed_count:04d}"
    zip_path = output_dir / f"{stem}.zip"
    checksum_path = output_dir / f"{stem}.zip.sha256"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(state_path, arcname="state.json")
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    checksum_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path) as archive:
        if json.loads(archive.read("state.json")) != checkpoint_state:
            raise RuntimeError("local checkpoint verification failed")
    _publish_pair_create_only(
        zip_path,
        checksum_path,
        checkpoint_sink,
        artifact_kind="checkpoint",
    )
    state.clear()
    state.update(checkpoint_state)


def _export(output_dir: Path, receipt: dict[str, Any], records: list[StageARecord]) -> tuple[Path, str]:
    result = {
        "run_id": receipt["run_id"],
        "execution_scope_id": receipt["execution_scope_id"],
        "resolved_exact": receipt["resolved_exact"],
        "rc": receipt["rc"],
        "status": receipt["status"],
        "completeness": receipt["completeness"],
        "scientific_status": SCIENTIFIC_STATUS,
        "limitations": list(LIMITATIONS),
        "protocol_digest": receipt["protocol_digest"],
        "method_identities": receipt["method_identities"],
        "condition_order": receipt["condition_order"],
        "attack_ids": receipt["attack_ids"],
        "record_arms_in_exact_condition_order": receipt[
            "record_arms_in_exact_condition_order"
        ],
        "ordered_roster_unit_ids": receipt["ordered_roster_unit_ids"],
        "ordered_roster_source_ids": receipt["ordered_roster_source_ids"],
        "model_id": receipt["model_id"],
        "key_public_digest": receipt["key_public_digest"],
        "rank_gate_a_min_units": receipt["rank_gate_a_min_units"],
        "rank_gate_b_min_units": receipt["rank_gate_b_min_units"],
        "checkpoint_interval_hours": receipt["checkpoint_interval_hours"],
        "checkpoint_sequence": receipt["checkpoint_sequence"],
        "committed_unit_count": receipt["committed_unit_count"],
        "committed_unit_ids": receipt["committed_unit_ids"],
        "fixed_unit_count": 8,
        "fixed_condition_count": 4,
        "fixed_record_count": 128,
        "records": [record.to_dict() for record in records],
    }
    if "attack_complementarity_evidence" in receipt:
        result["attack_complementarity_evidence"] = receipt[
            "attack_complementarity_evidence"
        ]
    if "checkpoint_status" in receipt:
        result["checkpoint_status"] = receipt["checkpoint_status"]
    if "error_class" in receipt:
        result.update({
            "result_kind": receipt["result_kind"],
            "error_class": receipt["error_class"],
            "approved_execution_exact": receipt["approved_execution_exact"],
            "resume_status": receipt["resume_status"],
        })
    _json_write(output_dir / "receipt.json", receipt)
    _json_write(output_dir / "result.json", result)
    zip_path = output_dir / f"{receipt['run_id']}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("receipt.json", "result.json"):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, (output_dir / name).read_bytes())
    zip_digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    return zip_path, zip_digest


def _write_local_checksum(zip_path: Path, zip_digest: str) -> Path:
    checksum_path = zip_path.with_suffix(".zip.sha256")
    checksum_path.write_text(f"{zip_digest}  {zip_path.name}\n", encoding="utf-8")
    return checksum_path


def _publish_pair_create_only(
    zip_path: Path,
    checksum_path: Path,
    sink: Path,
    *,
    artifact_kind: str,
) -> None:
    """Publish one prevalidated ZIP/checksum pair without overwrite or orphans."""

    _verify_checksum(zip_path, checksum_path)
    destinations = (sink / zip_path.name, sink / checksum_path.name)
    if any(destination.exists() for destination in destinations):
        raise RuntimeError(f"{artifact_kind} sink refuses overwrite")
    created: list[Path] = []
    try:
        for source, destination in zip((zip_path, checksum_path), destinations):
            with destination.open("xb") as target:
                target.write(source.read_bytes())
            created.append(destination)
            if source.read_bytes() != destination.read_bytes():
                raise RuntimeError(f"{artifact_kind} sink copy verification failed")
    except Exception as error:
        cleanup_failed = False
        for destination in reversed(created):
            try:
                destination.unlink()
            except OSError:
                cleanup_failed = True
        if cleanup_failed:
            raise RuntimeError(f"{artifact_kind} sink pair cleanup failed") from error
        raise
    if not all(destination.is_file() for destination in destinations):
        raise RuntimeError(f"{artifact_kind} sink pair publication incomplete")


def _publish_final(zip_path: Path, checksum_path: Path, run_store: Path) -> None:
    _publish_terminal_pair_create_only(
        zip_path,
        checksum_path,
        run_store,
        artifact_kind="final",
    )


def _publish_terminal_pair_create_only(
    zip_path: Path,
    checksum_path: Path,
    sink: Path,
    *,
    artifact_kind: str,
) -> None:
    """Copy a completed terminal pair once; external adjudication validates it."""

    destinations = (sink / zip_path.name, sink / checksum_path.name)
    if any(destination.exists() for destination in destinations):
        raise RuntimeError(f"{artifact_kind} sink refuses overwrite")
    for source, destination in zip((zip_path, checksum_path), destinations):
        with source.open("rb") as source_stream, destination.open("xb") as target:
            shutil.copyfileobj(source_stream, target)


def _publish_failure(
    zip_path: Path,
    checksum_path: Path,
    run_store: Path,
) -> None:
    _publish_terminal_pair_create_only(
        zip_path,
        checksum_path,
        run_store,
        artifact_kind="failure",
    )


def _export_fatal(
    args: argparse.Namespace,
    context: dict[str, Any],
    error_class: str,
) -> tuple[Path, str, bool]:
    if error_class not in _FATAL_ERROR_BY_PHASE.values():
        raise ValueError("fatal error class is not predeclared")
    run_id = context.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]{7,63}", run_id) is None:
        raise ValueError("unsafe run id cannot name a fatal package")
    output_dir = context.get("output_dir")
    if output_dir is None:
        output_dir = Path(args.output_root).resolve() / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        context["output_dir"] = output_dir
        context["output_dir_owned"] = True
    elif not context.get("output_dir_owned"):
        raise RuntimeError("fatal package refuses an unowned output directory")
    state = context.get("state") or {}
    record_payloads = state.get("records", [])
    records = [StageARecord(**record) for record in record_payloads]
    committed_ids = list(state.get("committed_unit_ids", []))
    if len(records) != len(committed_ids) * 16:
        raise ValueError("fatal package refuses inconsistent committed records")
    approved_exact = (
        args.expected_exact
        if re.fullmatch(r"[0-9a-f]{40}", args.expected_exact) is not None
        else None
    )
    interval = context.get("checkpoint_interval_hours")
    receipt: dict[str, Any] = {
        "run_id": run_id,
        "execution_scope_id": context.get("execution_scope_id"),
        "approved_execution_exact": approved_exact,
        "resolved_exact": context.get("resolved_exact"),
        "rc": 2,
        "status": "operational_failure",
        "result_kind": "operational_failure_not_scientific",
        "error_class": error_class,
        "completeness": INCOMPLETE_EXECUTION,
        "scientific_status": SCIENTIFIC_STATUS,
        "protocol_digest": context.get("protocol_digest"),
        "method_identities": context.get("method_identities"),
        "condition_order": list(CONDITION_ORDER),
        "attack_ids": list(ATTACK_IDS),
        "record_arms_in_exact_condition_order": list(RECORD_ARMS),
        "ordered_roster_unit_ids": list(context.get("ordered_roster_unit_ids", [])),
        "ordered_roster_source_ids": list(
            context.get("ordered_roster_source_ids", [])
        ),
        "model_id": context.get("model_id"),
        "key_public_digest": context.get("key_public_digest"),
        "rank_gate_a_min_units": context.get("rank_gate_a_min_units"),
        "rank_gate_b_min_units": context.get("rank_gate_b_min_units"),
        "checkpoint_interval_hours": interval,
        "checkpoint_sequence": int(state.get("checkpoint_sequence", 0)),
        "committed_unit_count": len(committed_ids),
        "committed_unit_ids": committed_ids,
        "resume_status": context.get("resume_status", "not_requested"),
        "limitations": list(LIMITATIONS),
    }
    expected = context.get("expected_state")
    run_store = context.get("run_store")
    if not isinstance(expected, dict) or not isinstance(run_store, Path):
        raise RuntimeError("fatal package requires resolved run identity and sink")
    receipt["attack_complementarity_evidence"] = _attack_complementarity_evidence(
        records,
        expected,
        scientific_outcome_allowed=False,
    )
    base_zip, zip_digest = _export(output_dir, receipt, records)
    fatal_zip = output_dir / f"failure-{error_class}.zip"
    os.replace(base_zip, fatal_zip)
    fatal_checksum = output_dir / f"{fatal_zip.name}.sha256"
    fatal_checksum.write_text(f"{zip_digest}  {fatal_zip.name}\n", encoding="utf-8")
    published = False
    try:
        _publish_failure(fatal_zip, fatal_checksum, run_store)
        published = True
    except Exception:
        published = False
    return fatal_zip, zip_digest, published


def execute(args: argparse.Namespace, *, fatal_context: dict[str, Any] | None = None) -> int:
    context = fatal_context if fatal_context is not None else {}
    context["phase"] = "initialization"
    repo_root = Path(args.repo_root).resolve()
    resolved_exact = _git_exact(repo_root, args.expected_exact)
    context["resolved_exact"] = resolved_exact
    protocol = _load_protocol(repo_root)
    context["protocol_digest"] = protocol.protocol_digest
    execution_scope_id = protocol.config["execution_scope_id"]
    if execution_scope_id != EXECUTION_SCOPE_ID:
        raise RuntimeError("protocol_execution_scope_identity_mismatch")
    context["execution_scope_id"] = execution_scope_id
    method_identities = _method_identities(protocol)
    context["method_identities"] = method_identities
    context["ordered_roster_unit_ids"] = [
        unit.unit_id for unit in protocol.candidate_selection
    ]
    context["ordered_roster_source_ids"] = [
        unit.source_id for unit in protocol.candidate_selection
    ]
    decision_rule = protocol.config["decision_rule"]
    context["rank_gate_a_min_units"] = decision_rule[
        "gate_a_registered_top_rank_among_17_min_units"
    ]
    context["rank_gate_b_min_units"] = decision_rule[
        "gate_b_method_registered_gt_same_condition_primary_null_min_units"
    ]
    runtime_config = protocol.config["generation_runtime"]
    budget_config = protocol.config["budget"]
    if runtime_config["model_id"] != "stabilityai/stable-diffusion-3.5-medium":
        raise RuntimeError("protocol_model_identity_mismatch")
    if runtime_config["public_asset_rule"] != (
        "protocol_model_id_default_hub_resolution_without_revision_or_weight_digest"
    ):
        raise RuntimeError("protocol_public_asset_rule_mismatch")
    if (
        runtime_config["inference_steps"] != 20
        or budget_config["total_relative_l2_per_method"] != 0.012
        or runtime_config["injection_step_index_zero_based"]
        not in {HF_INJECTION_STEP_INDEX, LF_INJECTION_STEP_INDEX}
        or method_identities["hf"]["carrier_method_id"] != HF_CANDIDATE_ID
        or method_identities["hf"]["evaluated_candidate_id"]
        != HF_EVALUATED_CANDIDATE_ID
        or method_identities["hf"]["detector_statistic_id"]
        != HF_DETECTOR_STATISTIC_ID
        or method_identities["lf"]["carrier_method_id"]
        != LF_BALANCED_BLOCKS_CARRIER_METHOD_ID
        or method_identities["lf"]["evaluated_candidate_id"]
        != LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID
        or method_identities["lf"]["detector_statistic_id"]
        != LF_BLOCKNORM_DETECTOR_STATISTIC_ID
    ):
        raise RuntimeError("protocol_runtime_identity_mismatch")
    if len(protocol.candidate_selection) != 8 or protocol.untouched_confirmation:
        raise RuntimeError("attack_complementarity_roster_mismatch")
    model_id = runtime_config["model_id"]
    context["model_id"] = model_id
    context["checkpoint_interval_hours"] = CHECKPOINT_INTERVAL_HOURS
    run_store_root = Path(args.run_store_root).resolve()
    run_store_root.mkdir(parents=True, exist_ok=True)
    if not run_store_root.is_dir():
        raise ValueError("run store root must be a directory")

    raw_key = os.environ.pop(KEY_ENV, None)
    hf_token = os.environ.pop(TOKEN_ENV, None)
    if not isinstance(raw_key, str) or not raw_key.strip():
        hf_token = ""
        del hf_token
        raise RuntimeError("root_key_environment_input_required")
    detection_key = normalize_detection_key(raw_key)
    del raw_key
    key_digest = public_key_digest(detection_key)
    context["key_public_digest"] = key_digest
    run_id = _deterministic_run_id(resolved_exact, protocol, model_id, key_digest)
    context["run_id"] = run_id
    expected_state = _new_state(
        run_id=run_id,
        resolved_exact=resolved_exact,
        protocol=protocol,
        model_id=model_id,
        key_digest=key_digest,
    )
    context["expected_state"] = expected_state
    context["state"] = expected_state
    run_store = run_store_root / run_id
    if run_store.exists():
        if not run_store.is_dir():
            raise ValueError("deterministic run store path must be a directory")
    else:
        run_store.mkdir(parents=False, exist_ok=False)
    context["run_store"] = run_store
    if not isinstance(hf_token, str) or not hf_token.strip():
        hf_token = ""
        del hf_token, detection_key
        raise RuntimeError("hugging_face_token_environment_input_required")
    wrong_keys = _wrong_keys(detection_key, protocol)
    final_zip = run_store / f"{run_id}.zip"
    final_checksum = run_store / f"{run_id}.zip.sha256"
    final_names = {final_zip.name, final_checksum.name}
    checkpoint_name = re.compile(r"checkpoint-\d{4}-units-\d{4}\.zip(?:\.sha256)?")
    fatal_classes = "|".join(re.escape(value) for value in _FATAL_ERROR_BY_PHASE.values())
    failure_name = re.compile(rf"failure-(?:{fatal_classes})\.zip(?:\.sha256)?")
    context["phase"] = "resume_validation"
    for path in run_store.iterdir():
        if not path.is_file() or (
            path.name not in final_names
            and checkpoint_name.fullmatch(path.name) is None
            and failure_name.fullmatch(path.name) is None
        ):
            raise ValueError("run store contains an unexpected artifact")
    existing_failure_classes: list[str] = []
    for error_class in _FATAL_ERROR_BY_PHASE.values():
        failure_zip = run_store / f"failure-{error_class}.zip"
        failure_checksum = run_store / f"{failure_zip.name}.sha256"
        if failure_zip.exists() or failure_checksum.exists():
            if not (failure_zip.is_file() and failure_checksum.is_file()):
                raise ValueError("failure package pair is incomplete")
            existing_failure_classes.append(error_class)
    if existing_failure_classes:
        if len(existing_failure_classes) != 1:
            raise ValueError("multiple terminal failure package pairs exist")
        context["resume_status"] = "terminal_failure_pair_present"
        hf_token = ""
        del detection_key, wrong_keys, hf_token
        print(
            "CEGWM_FATAL " + json.dumps({
                "run_id": run_id,
                "error_class": existing_failure_classes[0],
                "export_status": "present_for_external_validation",
            }),
            flush=True,
        )
        return 2
    context["resume_status"] = "rejected"
    state = _discover_checkpoint(run_store, expected_state)
    context["state"] = state or expected_state
    if final_zip.exists() or final_checksum.exists():
        if not (final_zip.is_file() and final_checksum.is_file()):
            raise ValueError("final package pair is incomplete")
        context["resume_status"] = "terminal_final_pair_present"
        hf_token = ""
        del detection_key, wrong_keys, hf_token
        print(
            "CEGWM_PROGRESS " + json.dumps({
                "run_id": run_id,
                "committed": 8,
                "fixed_total": len(expected_state["ordered_roster_unit_ids"]),
                "phase": "terminal_pair_present",
            }),
            flush=True,
        )
        print(
            "CEGWM_SUMMARY " + json.dumps({
                "run_id": run_id,
                "resolved_exact": resolved_exact,
                "rc": 0,
                "status": "terminal_pair_present_for_external_validation",
                "zip_path": str(final_zip),
            }),
            flush=True,
        )
        return 0
    if state is None:
        state = expected_state
        context["resume_status"] = "fresh"
    else:
        context["resume_status"] = "accepted_checkpoint"
    context["state"] = state
    context["phase"] = "runtime_execution"
    output_dir = Path(args.output_root).resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    context["output_dir"] = output_dir
    context["output_dir_owned"] = True
    _atomic_json_write(output_dir / "state.json", state)
    receipt: dict[str, Any] = {
        "run_id": run_id,
        "execution_scope_id": execution_scope_id,
        "resolved_exact": resolved_exact,
        "rc": None,
        "status": "running",
        "completeness": INCOMPLETE_EXECUTION,
        "scientific_status": SCIENTIFIC_STATUS,
        "protocol_digest": protocol.protocol_digest,
        "method_identities": method_identities,
        "condition_order": list(CONDITION_ORDER),
        "attack_ids": list(ATTACK_IDS),
        "record_arms_in_exact_condition_order": list(RECORD_ARMS),
        "ordered_roster_unit_ids": expected_state["ordered_roster_unit_ids"],
        "ordered_roster_source_ids": expected_state["ordered_roster_source_ids"],
        "model_id": model_id,
        "key_public_digest": key_digest,
        "rank_gate_a_min_units": expected_state["rank_gate_a_min_units"],
        "rank_gate_b_min_units": expected_state["rank_gate_b_min_units"],
        "checkpoint_interval_hours": CHECKPOINT_INTERVAL_HOURS,
        "limitations": list(LIMITATIONS),
    }
    _json_write(output_dir / "receipt.json", receipt)
    records = [StageARecord(**record) for record in state["records"]]
    committed = set(state["committed_unit_ids"])
    pending_units = [
        unit
        for unit in protocol.candidate_selection
        if unit.unit_id not in committed
    ]
    any_failure = any(record.status != "success" for record in records)
    pipeline = None
    hf_assets = None
    lf_assets = None
    model_load_failed = False
    last_checkpoint_time = time.monotonic()
    if pending_units:
        try:
            pipeline, hf_assets, lf_assets = _load_pipeline_and_assets(
                model_id, hf_token
            )
        except Exception:
            model_load_failed = True
            any_failure = True
        finally:
            hf_token = ""
            del hf_token
    else:
        hf_token = ""
        del hf_token
    checkpoint_failure = False
    new_units_since_checkpoint = 0
    for unit in pending_units:
        if model_load_failed:
            transaction = _failure_transaction(
                unit,
                protocol,
                run_id,
                resolved_exact,
                key_digest,
                "model_load_failure",
            )
        else:
            try:
                hf_generator = torch.Generator(device="cuda").manual_seed(unit.seed)
                hf_output = run_sd35_hf(
                    pipeline,
                    unit.prompt,
                    detection_key,
                    hf_assets,
                    height=unit.height,
                    width=unit.width,
                    generator=hf_generator,
                )
                lf_generator = torch.Generator(device="cuda").manual_seed(unit.seed)
                lf_output = run_sd35_lf(
                    pipeline,
                    unit.prompt,
                    detection_key,
                    lf_assets,
                    height=unit.height,
                    width=unit.width,
                    generator=lf_generator,
                )
                null_generator = torch.Generator(device="cuda").manual_seed(unit.seed)
                null_image = run_sd35_plain(
                    pipeline,
                    unit.prompt,
                    height=unit.height,
                    width=unit.width,
                    generator=null_generator,
                )
                hf_budget = float(hf_output.injection_budget.relative_l2)
                lf_budget = float(lf_output.injection_budget.relative_l2)
                if not all(
                    math.isfinite(value) and 0.0 < value <= 0.012
                    for value in (hf_budget, lf_budget)
                ):
                    raise ValueError("actual_dtype_budget_invalid")
                hf_paired_psnr = _psnr(hf_output.image, null_image)
                lf_paired_psnr = _psnr(lf_output.image, null_image)
                transaction = []
                for condition in CONDITION_ORDER:
                    if condition == IDENTITY_REFERENCE:
                        hf_image = hf_output.image
                        lf_image = lf_output.image
                        current_null = null_image
                        noise_domain = None
                    else:
                        noise_domain = (
                            public_noise_domain(
                                protocol_id=protocol.protocol_id,
                                attack_id=condition,
                                unit_id=unit.unit_id,
                                source_id=unit.source_id,
                                generation_seed=unit.seed,
                                height=unit.height,
                                width=unit.width,
                            )
                            if condition == "gaussian_noise_std_0_01"
                            else None
                        )
                        hf_image = apply_attack(
                            hf_output.image, condition, noise_domain=noise_domain
                        )
                        lf_image = apply_attack(
                            lf_output.image, condition, noise_domain=noise_domain
                        )
                        current_null = apply_attack(
                            null_image, condition, noise_domain=noise_domain
                        )
                    hf_metrics = {
                        "actual_dtype_relative_l2": hf_budget,
                        "paired_rgb_psnr": hf_paired_psnr,
                    }
                    lf_metrics = {
                        "actual_dtype_relative_l2": lf_budget,
                        "paired_rgb_psnr": lf_paired_psnr,
                    }
                    hf_null_metrics = {"paired_rgb_psnr": hf_paired_psnr}
                    lf_null_metrics = {"paired_rgb_psnr": lf_paired_psnr}
                    if condition in ATTACK_IDS:
                        hf_attack_psnr = _attacked_vs_pre_attack_psnr(
                            hf_image, hf_output.image
                        )
                        lf_attack_psnr = _attacked_vs_pre_attack_psnr(
                            lf_image, lf_output.image
                        )
                        null_attack_psnr = _attacked_vs_pre_attack_psnr(
                            current_null, null_image
                        )
                        if hf_attack_psnr is not None:
                            hf_metrics["attacked_vs_pre_attack_psnr"] = hf_attack_psnr
                        if lf_attack_psnr is not None:
                            lf_metrics["attacked_vs_pre_attack_psnr"] = lf_attack_psnr
                        if null_attack_psnr is not None:
                            hf_null_metrics[
                                "attacked_vs_pre_attack_psnr"
                            ] = null_attack_psnr
                            lf_null_metrics[
                                "attacked_vs_pre_attack_psnr"
                            ] = null_attack_psnr
                    common = dict(
                        run_id=run_id,
                        unit_id=unit.unit_id,
                        source_cluster_id=unit.source_id,
                        condition=condition,
                        code_revision=resolved_exact,
                        config_digest=protocol.protocol_digest,
                        key_public_digest=key_digest,
                        status="success",
                    )
                    hf_null_scores = _scores(
                        current_null, detection_key, wrong_keys, hf_assets
                    )
                    lf_null_scores = _scores(
                        current_null, detection_key, wrong_keys, lf_assets
                    )
                    transaction.extend([
                        StageARecord(
                            arm=HF_EVALUATED_CANDIDATE_ID,
                            scores=_scores(
                                hf_image, detection_key, wrong_keys, hf_assets
                            ),
                            metrics=hf_metrics,
                            **common,
                        ),
                        StageARecord(
                            arm=f"primary_null__{HF_EVALUATED_CANDIDATE_ID}",
                            scores=hf_null_scores,
                            metrics=hf_null_metrics,
                            **common,
                        ),
                        StageARecord(
                            arm=LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
                            scores=_scores(
                                lf_image, detection_key, wrong_keys, lf_assets
                            ),
                            metrics=lf_metrics,
                            **common,
                        ),
                        StageARecord(
                            arm=(
                                "primary_null__"
                                + LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID
                            ),
                            scores=lf_null_scores,
                            metrics=lf_null_metrics,
                            **common,
                        ),
                    ])
                expected_pairs = [
                    (condition, arm)
                    for condition in CONDITION_ORDER
                    for arm in RECORD_ARMS
                ]
                if [(record.condition, record.arm) for record in transaction] != expected_pairs:
                    raise RuntimeError("unit_transaction_condition_arm_order_mismatch")
            except Exception:
                any_failure = True
                transaction = _failure_transaction(
                    unit,
                    protocol,
                    run_id,
                    resolved_exact,
                    key_digest,
                    "unit_execution_failure",
                )
        next_state = dict(state)
        next_state["committed_unit_ids"] = [*state["committed_unit_ids"], unit.unit_id]
        next_state["committed_unit_count"] = len(next_state["committed_unit_ids"])
        next_state["records"] = [
            *state["records"],
            *(record.to_dict() for record in transaction),
        ]
        _atomic_json_write(output_dir / "state.json", next_state)
        state.clear()
        state.update(next_state)
        records.extend(transaction)
        new_units_since_checkpoint += 1
        now = time.monotonic()
        if (
            new_units_since_checkpoint > 0
            and now - last_checkpoint_time >= CHECKPOINT_INTERVAL_HOURS * 3600.0
        ):
            try:
                context["phase"] = "checkpoint"
                _checkpoint(state, output_dir, run_store)
                last_checkpoint_time = now
                new_units_since_checkpoint = 0
            except Exception:
                checkpoint_failure = True
                any_failure = True
            finally:
                context["phase"] = "runtime_execution"
        print(
            "CEGWM_PROGRESS " + json.dumps({
                "run_id": run_id,
                "committed": len(state["committed_unit_ids"]),
                "fixed_total": len(expected_state["ordered_roster_unit_ids"]),
            }),
            flush=True,
        )

    if len(records) != 128:
        raise RuntimeError("fixed_record_roster_not_preserved")
    receipt["rc"] = 1 if any_failure else 0
    receipt["completeness"] = _completeness_for_rc(receipt["rc"])
    receipt["checkpoint_sequence"] = state["checkpoint_sequence"]
    receipt["committed_unit_count"] = state["committed_unit_count"]
    receipt["committed_unit_ids"] = list(state["committed_unit_ids"])
    if checkpoint_failure:
        receipt["checkpoint_status"] = "failure"
    else:
        receipt["checkpoint_status"] = "complete"
    receipt["attack_complementarity_evidence"] = _attack_complementarity_evidence(
        records,
        expected_state,
        scientific_outcome_allowed=receipt["rc"] == 0,
    )
    receipt["status"] = (
        "complete_with_operational_failures"
        if any_failure
        else "complete_for_adjudication"
    )
    context["phase"] = "final_export"
    zip_path, zip_digest = _export(output_dir, receipt, records)
    local_checksum = _write_local_checksum(zip_path, zip_digest)
    context["terminal_package_constructed"] = True
    _publish_final(zip_path, local_checksum, run_store)
    del detection_key, wrong_keys
    print(
        "CEGWM_SUMMARY " + json.dumps({
            "run_id": run_id,
            "resolved_exact": resolved_exact,
            "rc": receipt["rc"],
            "zip_path": str(final_zip),
            "zip_sha256": zip_digest,
        }),
        flush=True,
    )
    return int(receipt["rc"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--run-store-root", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    fatal_context: dict[str, Any] = {}
    try:
        return_code = execute(args, fatal_context=fatal_context)
    except (Exception, KeyboardInterrupt):
        phase = fatal_context.get("phase", "initialization")
        error_class = _FATAL_ERROR_BY_PHASE.get(phase, "initialization_failure")
        export_status = "unavailable"
        if fatal_context.get("terminal_package_constructed"):
            export_status = "local_only"
        else:
            try:
                _, _, published = _export_fatal(args, fatal_context, error_class)
                export_status = "published" if published else "local_only"
            except Exception:
                pass
        print(
            "CEGWM_FATAL " + json.dumps({
                "run_id": fatal_context.get("run_id"),
                "error_class": error_class,
                "export_status": export_status,
            }),
            flush=True,
        )
        return_code = 2
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
