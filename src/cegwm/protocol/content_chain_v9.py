"""Frozen Phase-1 contract for Content V9 calibrated weighted joint scoring."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

CONTENT_V9_BASE_EXACT = "49bb03ed697a47048f5730ecfc85a9f29cb0b58a"
CONTENT_V9_METHOD_ID = "content_v9_v6_calibrated_weighted_joint_v1"
CONTENT_V9_EVALUATED_CANDIDATE_ID = (
    "content_v9_v6_calibrated_weighted_joint_semantic_gate_v1"
)
CONTENT_V9_CALIBRATION_PROTOCOL_ID = (
    "cegwm-stage-a-content-v9-calibrated-weighted-joint-calibration-v1"
)
CONTENT_V9_CALIBRATION_PROTOCOL_DIGEST = (
    "68f37585eb6eab123bad7c1703767df08404718ce4771f73fbbec236491a1e01"
)
CONTENT_V9_FUTURE_EVALUATION_PROTOCOL_ID = (
    "cegwm-stage-a-content-v9-calibrated-weighted-joint-clean-v1"
)
CONTENT_V9_RECORD_CONTRACT_ID = "content_v9_calibrated_weighted_joint_record_v1"
CONTENT_V9_STATE_SCHEMA_ID = "content_v9_resumable_state_v1"
CONTENT_V9_RUN_PREFIX = "content-v9"
CONTENT_V9_CALIBRATION_RECEIPT_ID = "content_v9_calibration_receipt_v1"
CONTENT_V9_CALIBRATION_SPLIT = "content_v9_calibration_v1"
CONTENT_V9_EVALUATION_SPLIT = "content_v9_clean_evaluation_v1"
CONTENT_V9_CALIBRATION_MANIFEST = "content_v9_calibration_v1.jsonl"
CONTENT_V9_EVALUATION_MANIFEST = "content_v9_clean_evaluation_v1.jsonl"
CONTENT_V9_CALIBRATION_MANIFEST_SHA256 = (
    "1e7bafcce363177684c1edf5d858df5baa03933affb28e066603a616c6e1c785"
)
CONTENT_V9_CALIBRATION_PROMPT_LIST_SHA256 = (
    "81fb52e376022744e02a7be3f4d3ee2e36ff9398d37bcb9428e99fa40f64fc95"
)
CONTENT_V9_EVALUATION_MANIFEST_SHA256 = (
    "dd9d9c60974f07f3727b0c46b08c1678dfb9b57339a662735ad4178b9473849d"
)
CONTENT_V9_EVALUATION_PROMPT_LIST_SHA256 = (
    "ed8796c183200ec826023d44f9ebe06293968f5a387dbcf242addaf036e46617"
)
CONTENT_V9_CALIBRATION_COUNT = 32
CONTENT_V9_EVALUATION_COUNT = 8
CONTENT_V9_PAIRED_NULL_SCORE_COUNT = 1056
CONTENT_V9_CALIBRATION_KEY_DOMAIN = (
    "stage-a/content-v9-calibrated-weighted-joint-calibration-key/v1"
)
CONTENT_V9_WRONG_KEY_DOMAIN = "stage-a/content-adaptive-v2-external-wrong-key/v1"
CONTENT_V9_CALIBRATION_ASSET_SCHEMA_ID = (
    "cegwm_content_v9_calibrated_weighted_joint_asset_v1"
)
CONTENT_V9_CALIBRATION_ASSET_ROLE_ID = "content_v9_calibrated_weighted_joint_v1"

_CONFIG_NAME = "content_v9_calibrated_weighted_joint_phase1_v1.json"
_FIELDS = ("unit_id", "split", "source_id", "prompt", "seed", "height", "width")
_HEX40 = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class ContentV9Unit:
    unit_id: str
    split: str
    source_id: str
    prompt: str
    seed: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class ContentV9Phase1Contract:
    calibration: tuple[ContentV9Unit, ...]
    evaluation: tuple[ContentV9Unit, ...]
    config: Mapping[str, Any]
    protocol_digest: str


def _stable_line(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _load_manifest(
    path: Path,
    *,
    split: str,
    count: int,
    manifest_sha256: str,
    prompt_sha256: str,
    seed_start: int,
    unit_prefix: str,
    source_prefix: str,
) -> tuple[ContentV9Unit, ...]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256 or not raw.endswith(b"\n"):
        raise ValueError("Content V9 manifest bytes differ")
    lines = raw.splitlines()
    if len(lines) != count:
        raise ValueError("Content V9 manifest count differs")
    units: list[ContentV9Unit] = []
    for ordinal, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Content V9 manifest must be UTF-8 JSONL") from error
        if not isinstance(value, dict) or tuple(value) != _FIELDS or _stable_line(value) != line:
            raise ValueError("Content V9 manifest fields, order, or encoding differ")
        expected = {
            "unit_id": f"{unit_prefix}-{ordinal:04d}",
            "split": split,
            "source_id": f"{source_prefix}-{ordinal:04d}",
            "seed": seed_start + ordinal - 1,
            "height": 512,
            "width": 512,
        }
        if any(value.get(name) != expected_value for name, expected_value in expected.items()):
            raise ValueError("Content V9 ordered unit identity differs")
        if not isinstance(value["prompt"], str) or not value["prompt"].strip():
            raise ValueError("Content V9 prompt must be non-empty text")
        units.append(ContentV9Unit(**value))
    prompt_bytes = b"".join(unit.prompt.encode("utf-8") + b"\n" for unit in units)
    if hashlib.sha256(prompt_bytes).hexdigest() != prompt_sha256:
        raise ValueError("Content V9 ordered prompt identity differs")
    return tuple(units)


def _identity_sets(units: Iterable[ContentV9Unit]) -> tuple[set[Any], ...]:
    received = tuple(units)
    return (
        {unit.unit_id for unit in received},
        {unit.source_id for unit in received},
        {unit.prompt for unit in received},
        {unit.seed for unit in received},
        {(unit.prompt, unit.seed) for unit in received},
    )


def _require_unique_disjoint(
    calibration: tuple[ContentV9Unit, ...], evaluation: tuple[ContentV9Unit, ...]
) -> None:
    calibration_sets = _identity_sets(calibration)
    evaluation_sets = _identity_sets(evaluation)
    if any(len(values) != CONTENT_V9_CALIBRATION_COUNT for values in calibration_sets):
        raise ValueError("Content V9 calibration identities must be unique")
    if any(len(values) != CONTENT_V9_EVALUATION_COUNT for values in evaluation_sets):
        raise ValueError("Content V9 evaluation identities must be unique")
    if any(left & right for left, right in zip(calibration_sets, evaluation_sets, strict=True)):
        raise ValueError("Content V9 calibration and evaluation identities overlap")


def _expected_config() -> dict[str, Any]:
    return {
        "asset_contract": {
            "asset_role_id": CONTENT_V9_CALIBRATION_ASSET_ROLE_ID,
            "asset_schema_id": CONTENT_V9_CALIBRATION_ASSET_SCHEMA_ID,
            "create_only_json_and_sha256_sidecar": True,
            "fit_unit_count": CONTENT_V9_CALIBRATION_COUNT,
            "iss_asset_sha256": "d66ff88640a3d1a020646cfde3face7502282bf835c9d3fb746b518dfb02c231",
            "iss_asset_sidecar_file_sha256": "27094d56994bc6f5d93564bad79ddd9ce8218d2d193786f4816535ee1e7f6538",
            "numeric_payload": "five_binary64_big_endian_hex_values",
            "raw_calibration_scores_persisted": False,
            "whitening_asset_sha256": "a7021dd8b98bc4282b98ed5d1fe276236d99a3c9e80b9bdce015d28cf715633f",
            "whitening_asset_sidecar_file_sha256": "c900cce0980348eeadcf07d782b6169c4d46ac55d7154db0fc0a0a878cce0ced",
        },
        "calibration": {
            "candidate_registered_excluded": True,
            "key_domain": CONTENT_V9_CALIBRATION_KEY_DOMAIN,
            "manifest": CONTENT_V9_CALIBRATION_MANIFEST,
            "manifest_sha256": CONTENT_V9_CALIBRATION_MANIFEST_SHA256,
            "ordered_pair_count": CONTENT_V9_PAIRED_NULL_SCORE_COUNT,
            "pair_order": "per_unit_candidate_wrong_00_to_15_then_primary_null_registered_then_primary_null_wrong_00_to_15_each_lf_then_hf",
            "prompt_list_sha256": CONTENT_V9_CALIBRATION_PROMPT_LIST_SHA256,
            "units": CONTENT_V9_CALIBRATION_COUNT,
        },
        "future_evaluation": {
            "formal_fpr_claim": False,
            "manifest": CONTENT_V9_EVALUATION_MANIFEST,
            "manifest_sha256": CONTENT_V9_EVALUATION_MANIFEST_SHA256,
            "protocol_id": CONTENT_V9_FUTURE_EVALUATION_PROTOCOL_ID,
            "prompt_list_sha256": CONTENT_V9_EVALUATION_PROMPT_LIST_SHA256,
            "records": 16,
            "units": CONTENT_V9_EVALUATION_COUNT,
        },
        "future_read_only_comparison_provenance": {
            "required_route_evidence": [
                "Content-V2", "Content-V3", "Content-V4", "Content-V5",
                "Content-V6-current", "Content-V6-old-roster-reference",
                "Content-V7", "Content-V8",
            ],
            "v6_current_producer_exact": CONTENT_V9_BASE_EXACT,
            "v6_old_roster_reference_exact": "39720994cc3316af8c2cac586689d0811232b4c7",
            "consumed_by_v9_calibration_or_evaluation": False,
            "comparison_stage": "later_independent_read_only_artifact_comparison_only",
        },
        "identities": {
            "base_exact": CONTENT_V9_BASE_EXACT,
            "evaluated_candidate_id": CONTENT_V9_EVALUATED_CANDIDATE_ID,
            "method_id": CONTENT_V9_METHOD_ID,
            "record_contract_id": CONTENT_V9_RECORD_CONTRACT_ID,
            "state_schema_id": CONTENT_V9_STATE_SCHEMA_ID,
        },
        "inherited_v6": {
            "embedding_and_branch_scorers": "byte_semantic_unchanged_from_Content_V6_production",
            "joint_replacement_only": True,
            "wrong_key_domain": CONTENT_V9_WRONG_KEY_DOMAIN,
        },
        "joint_operator": {
            "denominator": "sqrt(w_lf^2+w_hf^2+2*w_lf*w_hf*rho)",
            "formula": "(w_lf*z_lf+w_hf*z_hf)/denominator",
            "hf_weight": 0.75,
            "lf_hf_gates": "diagnostic_only_no_hard_veto",
            "lf_weight": 0.25,
            "statistics": "binary64_fsum_mean_ddof1_std_and_pearson_rho",
            "weighted_gate_a": "registered_strictly_greater_than_max_16_wrong_ties_fail",
            "weighted_gate_b": "candidate_registered_strictly_greater_than_primary_null_registered",
            "weighted_gate_min_units": 7,
        },
        "protocol_id": CONTENT_V9_CALIBRATION_PROTOCOL_ID,
        "protocol_version": 1,
        "claim_ceiling": "calibrated_combined_attribution_only",
        "scientific_status": "calibration_not_run_no_final_v9_evaluation_identity",
    }


def load_content_v9_phase1_contract(repo_root: str | Path) -> ContentV9Phase1Contract:
    root = Path(repo_root) / "configs" / "content_chain"
    calibration = _load_manifest(
        root / CONTENT_V9_CALIBRATION_MANIFEST,
        split=CONTENT_V9_CALIBRATION_SPLIT,
        count=CONTENT_V9_CALIBRATION_COUNT,
        manifest_sha256=CONTENT_V9_CALIBRATION_MANIFEST_SHA256,
        prompt_sha256=CONTENT_V9_CALIBRATION_PROMPT_LIST_SHA256,
        seed_start=2026091000,
        unit_prefix="content-v9-calibration",
        source_prefix="content-v9-calibration-source",
    )
    evaluation = _load_manifest(
        root / CONTENT_V9_EVALUATION_MANIFEST,
        split=CONTENT_V9_EVALUATION_SPLIT,
        count=CONTENT_V9_EVALUATION_COUNT,
        manifest_sha256=CONTENT_V9_EVALUATION_MANIFEST_SHA256,
        prompt_sha256=CONTENT_V9_EVALUATION_PROMPT_LIST_SHA256,
        seed_start=2026092000,
        unit_prefix="content-v9-eval",
        source_prefix="content-v9-eval-source",
    )
    _require_unique_disjoint(calibration, evaluation)
    try:
        config = json.loads((root / _CONFIG_NAME).read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Content V9 phase1 config must be UTF-8 JSON") from error
    if config != _expected_config():
        raise ValueError("Content V9 phase1 config differs")
    canonical = json.dumps(
        {
            "calibration": [unit.__dict__ if hasattr(unit, "__dict__") else {
                field: getattr(unit, field) for field in _FIELDS
            } for unit in calibration],
            "config": config,
            "evaluation": [{field: getattr(unit, field) for field in _FIELDS} for unit in evaluation],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    protocol_digest = hashlib.sha256(canonical).hexdigest()
    if protocol_digest != CONTENT_V9_CALIBRATION_PROTOCOL_DIGEST:
        raise ValueError("Content V9 calibration canonical protocol digest differs")
    return ContentV9Phase1Contract(calibration, evaluation, config, protocol_digest)


def deterministic_calibration_run_id(protocol_digest: str, public_key_digest: str) -> str:
    if not isinstance(protocol_digest, str) or re.fullmatch(r"[0-9a-f]{64}", protocol_digest) is None:
        raise ValueError("protocol digest must be lowercase 64-hex")
    if not isinstance(public_key_digest, str) or re.fullmatch(r"[0-9a-f]{64}", public_key_digest) is None:
        raise ValueError("public key digest must be lowercase 64-hex")
    return f"{CONTENT_V9_RUN_PREFIX}-calibration-{protocol_digest[:12]}-{public_key_digest[:12]}"


__all__ = [name for name in globals() if name.startswith("CONTENT_V9_")] + [
    "ContentV9Phase1Contract",
    "ContentV9Unit",
    "deterministic_calibration_run_id",
    "load_content_v9_phase1_contract",
]
