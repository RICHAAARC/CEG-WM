"""
File purpose: Shared utilities for paper_workflow PW orchestration.
Module type: General module
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

from main.evaluation import attack_plan as eval_attack_plan
from main.evaluation import attack_runner as eval_attack_runner
from main.evaluation import protocol_loader as attack_protocol_loader
from main.watermarking.content_chain.ldpc_codec import build_ldpc_spec, encode_message_bits

from scripts.notebook_runtime_common import (
    REPO_ROOT,
    collect_attestation_env_summary,
    compute_file_sha256,
    ensure_directory,
    load_yaml_mapping,
    normalize_path_value,
    validate_path_within_base,
)

PAPER_WORKFLOW_ROOT_RELATIVE = "paper_workflow/families"
DEFAULT_CONFIG_RELATIVE_PATH = "configs/default.yaml"
DEFAULT_PW_BASE_CONFIG_RELATIVE_PATH = "paper_workflow/configs/pw_base.yaml"
DEFAULT_PW_MATRIX_CONFIG_RELATIVE_PATH = "paper_workflow/configs/pw_matrix.yaml"
ACTIVE_SAMPLE_ROLE = "positive_source"
CLEAN_NEGATIVE_SAMPLE_ROLE = "clean_negative"
PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE = "planner_conditioned_control_negative"
ATTACKED_POSITIVE_SAMPLE_ROLE = "attacked_positive"
ATTACKED_NEGATIVE_SAMPLE_ROLE = "attacked_negative"
ATTACK_SAMPLE_ROLES = [ATTACKED_POSITIVE_SAMPLE_ROLE, ATTACKED_NEGATIVE_SAMPLE_ROLE]
MIXED_ATTACK_SAMPLE_ROLE = "mixed_attack_roles"
ACTIVE_SOURCE_SAMPLE_ROLES = [
    ACTIVE_SAMPLE_ROLE,
    CLEAN_NEGATIVE_SAMPLE_ROLE,
    PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE,
]
RESERVED_SAMPLE_ROLES = ATTACK_SAMPLE_ROLES
SAMPLE_ROLE_DIRECTORY_NAMES = {
    ACTIVE_SAMPLE_ROLE: "positive",
    CLEAN_NEGATIVE_SAMPLE_ROLE: "negative",
    PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE: "control_negative",
}
SOURCE_TRUTH_STAGE = "PW01_Source_Event_Shards"
ATTACK_SEVERITY_RULE_VERSION = "pw_attack_severity_v1"
ATTACK_SEVERITY_AXIS_KIND = "family_local"
PAYLOAD_SIDECAR_SCHEMA_VERSION = "pw_payload_sidecar_v1"
PAYLOAD_SIDECAR_STATUS_SCHEMA_VERSION = "pw_payload_sidecar_status_v1"
PAYLOAD_CONSISTENCY_PROBE_MODE = "payload_consistency_probe_v1"
GEOMETRY_OPTIONAL_CLAIM_PLAN_FILE_NAME = "geometry_optional_claim_plan.json"
GEOMETRY_OPTIONAL_CLAIM_MODE = "optional_geometry_rescue_evidence_only"
GEOMETRY_OPTIONAL_CLAIM_SCOPE = "attacked_positive_content_failed_subset"
GEOMETRY_OPTIONAL_CLAIM_DIRECTIONALITY = "one_way_positive_only"
GEOMETRY_OPTIONAL_CLAIM_PROTOCOL_VERSION = "geometry_optional_claim_content_margin_boundary_v1"
GEOMETRY_OPTIONAL_CLAIM_BOUNDARY_METRIC = "abs_content_margin"
GEOMETRY_OPTIONAL_CLAIM_BOUNDARY_ABS_MIN = 0.02
GEOMETRY_OPTIONAL_CLAIM_BOUNDARY_ABS_MAX = 0.2


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    """
    Return a canonical JSON text representation.

    Args:
        payload: Mapping payload.

    Returns:
        Canonical JSON text.
    """
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_mapping_sha256(payload: Mapping[str, Any]) -> str:
    """
    Compute SHA256 for a canonical JSON mapping.

    Args:
        payload: Mapping payload.

    Returns:
        Lowercase SHA256 digest.
    """
    canonical_text = _canonical_json_text(payload)
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def _extract_mapping(node: Any) -> Dict[str, Any]:
    """
    Normalize an optional mapping node to dict.

    Args:
        node: Candidate mapping node.

    Returns:
        Normalized dict payload.
    """
    return dict(cast(Mapping[str, Any], node)) if isinstance(node, Mapping) else {}


def _extract_sign_list(node: Any) -> List[int]:
    """
    Normalize one sign list to integer values in {-1, +1}.

    Args:
        node: Candidate sign list.

    Returns:
        Normalized sign list.
    """
    if not isinstance(node, list):
        return []
    normalized: List[int] = []
    for raw_value in cast(List[object], node):
        if isinstance(raw_value, bool):
            continue
        if isinstance(raw_value, (int, float)):
            normalized.append(1 if float(raw_value) > 0.0 else -1)
    return normalized


def _extract_int_list(node: Any) -> List[int]:
    """
    Normalize one integer list.

    Args:
        node: Candidate integer list.

    Returns:
        Normalized integer list.
    """
    if not isinstance(node, list):
        return []
    normalized: List[int] = []
    for raw_value in cast(List[object], node):
        if isinstance(raw_value, int) and not isinstance(raw_value, bool):
            normalized.append(int(raw_value))
    return normalized


def _extract_int(value: Any) -> int | None:
    """
    Coerce one integer-like scalar.

    Args:
        value: Candidate scalar value.

    Returns:
        Parsed integer or None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return int(parsed) if float(parsed).is_integer() else None
    return None


def _coerce_finite_float(value: Any) -> float | None:
    """
    Coerce one scalar to finite float.

    Args:
        value: Candidate scalar value.

    Returns:
        Finite float or None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value_float = float(value)
        return value_float if math.isfinite(value_float) else None
    if isinstance(value, str) and value.strip():
        try:
            value_float = float(value.strip())
        except ValueError:
            return None
        return value_float if math.isfinite(value_float) else None
    return None


def _extract_str_list(node: Any) -> List[str]:
    """
    Normalize one string list.

    Args:
        node: Candidate string list.

    Returns:
        Normalized string list.
    """
    if not isinstance(node, list):
        return []

    normalized: List[str] = []
    for raw_value in cast(List[object], node):
        if not isinstance(raw_value, str) or not raw_value.strip():
            return []
        normalized.append(raw_value.strip())
    return normalized


def _normalize_matrix_concrete_conditions(node: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Normalize matrix-defined concrete attack conditions.

    Args:
        node: Candidate concrete-conditions node.

    Returns:
        Concrete condition rows keyed by attack_condition_base_key.
    """
    if node is None:
        return {}
    if not isinstance(node, Mapping):
        raise ValueError("pw_matrix.concrete_conditions must be mapping when provided")

    normalized_conditions: Dict[str, List[Dict[str, Any]]] = {}
    for raw_condition_key, raw_rows in cast(Mapping[str, Any], node).items():
        condition_key = str(raw_condition_key or "").strip()
        if not condition_key:
            raise ValueError("pw_matrix.concrete_conditions keys must be non-empty str")
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError(
                f"pw_matrix.concrete_conditions.{condition_key} must be non-empty list[mapping]"
            )

        seen_suffixes: set[str] = set()
        seen_level_indices: set[int] = set()
        normalized_rows: List[Dict[str, Any]] = []
        for row_index, raw_row in enumerate(cast(List[Any], raw_rows)):
            if not isinstance(raw_row, Mapping):
                raise ValueError(
                    f"pw_matrix.concrete_conditions.{condition_key}[{row_index}] must be mapping"
                )
            condition_suffix = str(raw_row.get("condition_suffix") or "").strip()
            severity_label = str(raw_row.get("severity_label") or "").strip()
            severity_level_index = _extract_int(raw_row.get("severity_level_index"))
            params_node = raw_row.get("params")

            if not condition_suffix:
                raise ValueError(
                    f"pw_matrix.concrete_conditions.{condition_key}[{row_index}].condition_suffix must be non-empty str"
                )
            if "::" in condition_suffix:
                raise ValueError(
                    f"pw_matrix.concrete_conditions.{condition_key}[{row_index}].condition_suffix must not contain '::'"
                )
            if not severity_label:
                raise ValueError(
                    f"pw_matrix.concrete_conditions.{condition_key}[{row_index}].severity_label must be non-empty str"
                )
            if severity_level_index is None or severity_level_index < 0:
                raise ValueError(
                    f"pw_matrix.concrete_conditions.{condition_key}[{row_index}].severity_level_index must be non-negative int"
                )
            if not isinstance(params_node, Mapping) or not params_node:
                raise ValueError(
                    f"pw_matrix.concrete_conditions.{condition_key}[{row_index}].params must be non-empty mapping"
                )
            if condition_suffix in seen_suffixes:
                raise ValueError(
                    f"duplicate pw_matrix concrete condition suffix under {condition_key}: {condition_suffix}"
                )
            if severity_level_index in seen_level_indices:
                raise ValueError(
                    f"duplicate pw_matrix concrete severity_level_index under {condition_key}: {severity_level_index}"
                )

            seen_suffixes.add(condition_suffix)
            seen_level_indices.add(severity_level_index)
            normalized_rows.append(
                {
                    "condition_suffix": condition_suffix,
                    "severity_label": severity_label,
                    "severity_level_index": severity_level_index,
                    "params": copy.deepcopy(dict(cast(Mapping[str, Any], params_node))),
                }
            )

        ordered_rows = sorted(
            normalized_rows,
            key=lambda row: (
                int(cast(int, row["severity_level_index"])),
                str(row["condition_suffix"]),
            ),
        )
        expected_level_indices = list(range(len(ordered_rows)))
        actual_level_indices = [
            int(cast(int, row["severity_level_index"]))
            for row in ordered_rows
        ]
        if actual_level_indices != expected_level_indices:
            raise ValueError(
                "pw_matrix concrete severity_level_index values must form a contiguous 0-based sequence: "
                f"condition_key={condition_key}, actual={actual_level_indices}"
            )
        normalized_conditions[condition_key] = ordered_rows

    return normalized_conditions


def _resolve_repo_config_path(
    repo_root: Path,
    path_value: Path | str | None,
    default_relative_path: str,
    label: str,
) -> Path:
    """
    Resolve one repository-bound config path.

    Args:
        repo_root: Repository root path.
        path_value: Optional absolute or repo-relative path override.
        default_relative_path: Default repo-relative path.
        label: Human-readable argument label.

    Returns:
        Resolved absolute config path.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if not isinstance(default_relative_path, str) or not default_relative_path:
        raise TypeError("default_relative_path must be non-empty str")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")

    if path_value is None:
        candidate_path = Path(default_relative_path).expanduser()
    elif isinstance(path_value, Path):
        candidate_path = path_value.expanduser()
    elif isinstance(path_value, str) and path_value.strip():
        candidate_path = Path(path_value.strip()).expanduser()
    else:
        raise TypeError(f"{label} must be Path, non-empty str, or None")

    if not candidate_path.is_absolute():
        candidate_path = repo_root / candidate_path
    return candidate_path.resolve()


def load_pw_matrix_config(
    repo_root: Path = REPO_ROOT,
    base_config_path: Path | str | None = None,
    matrix_config_path: Path | str | None = None,
) -> Dict[str, Any]:
    """
    Load the paper_workflow matrix config mapping.

    Args:
        repo_root: Repository root path.
        base_config_path: Optional pw_base config path. When provided and
            matrix_config_path is absent, matrix_config_path is resolved from
            this base config.
        matrix_config_path: Optional direct matrix config path. When provided,
            it takes precedence over base_config_path.

    Returns:
        Parsed matrix config mapping.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")

    if matrix_config_path is not None:
        resolved_matrix_config_path = _resolve_repo_config_path(
            repo_root,
            matrix_config_path,
            DEFAULT_PW_MATRIX_CONFIG_RELATIVE_PATH,
            "matrix_config_path",
        )
        return load_yaml_mapping(resolved_matrix_config_path)

    resolved_pw_base_path = _resolve_repo_config_path(
        repo_root,
        base_config_path,
        DEFAULT_PW_BASE_CONFIG_RELATIVE_PATH,
        "base_config_path",
    )
    pw_base_cfg = load_yaml_mapping(resolved_pw_base_path)
    matrix_config_relative_path = pw_base_cfg.get("matrix_config_path")
    resolved_matrix_config_path = _resolve_repo_config_path(
        repo_root,
        matrix_config_relative_path
        if isinstance(matrix_config_relative_path, str) and matrix_config_relative_path.strip()
        else None,
        DEFAULT_PW_MATRIX_CONFIG_RELATIVE_PATH,
        "matrix_config_path",
    )
    return load_yaml_mapping(resolved_matrix_config_path)


def resolve_pw_matrix_settings(matrix_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Resolve the normalized paper_workflow attack matrix settings.

    Args:
        matrix_cfg: Raw matrix config mapping.

    Returns:
        Normalized matrix settings.
    """
    if not isinstance(matrix_cfg, Mapping):
        raise TypeError("matrix_cfg must be Mapping")

    matrix_profile = str(matrix_cfg.get("matrix_profile") or "").strip()
    matrix_version = str(matrix_cfg.get("matrix_version") or "").strip()
    materialization_profile = str(matrix_cfg.get("materialization_profile") or "").strip()
    if not matrix_profile:
        raise ValueError("pw_matrix.matrix_profile must be non-empty str")
    if not matrix_version:
        raise ValueError("pw_matrix.matrix_version must be non-empty str")
    if not materialization_profile:
        raise ValueError("pw_matrix.materialization_profile must be non-empty str")

    attack_sets_node = matrix_cfg.get("attack_sets")
    if not isinstance(attack_sets_node, Mapping) or not attack_sets_node:
        raise ValueError("pw_matrix.attack_sets must be non-empty mapping")
    attack_sets: Dict[str, List[str]] = {}
    for raw_set_name, raw_set_payload in cast(Mapping[str, Any], attack_sets_node).items():
        set_name = str(raw_set_name or "").strip()
        if not set_name:
            raise ValueError("pw_matrix.attack_sets keys must be non-empty str")
        if not isinstance(raw_set_payload, Mapping):
            raise ValueError(f"pw_matrix.attack_sets.{set_name} must be mapping")
        families = _extract_str_list(cast(Mapping[str, Any], raw_set_payload).get("families"))
        if not families:
            raise ValueError(f"pw_matrix.attack_sets.{set_name}.families must be non-empty list[str]")
        attack_sets[set_name] = list(dict.fromkeys(families))

    concrete_conditions = _normalize_matrix_concrete_conditions(matrix_cfg.get("concrete_conditions"))
    if materialization_profile == "matrix_defined_concrete_conditions" and not concrete_conditions:
        raise ValueError(
            "pw_matrix.concrete_conditions must be non-empty when materialization_profile=matrix_defined_concrete_conditions"
        )

    geometry_optional_claim_node = matrix_cfg.get("geometry_optional_claim")
    if not isinstance(geometry_optional_claim_node, Mapping):
        raise ValueError("pw_matrix.geometry_optional_claim must be mapping")
    candidate_attack_set = str(geometry_optional_claim_node.get("candidate_attack_set") or "").strip()
    if candidate_attack_set not in attack_sets:
        raise ValueError("pw_matrix.geometry_optional_claim.candidate_attack_set must reference one attack_sets key")
    boundary_rule_version = str(geometry_optional_claim_node.get("boundary_rule_version") or "").strip()
    boundary_metric = str(geometry_optional_claim_node.get("boundary_metric") or "").strip()
    boundary_abs_margin_min = _coerce_finite_float(geometry_optional_claim_node.get("boundary_abs_margin_min"))
    boundary_abs_margin_max = _coerce_finite_float(geometry_optional_claim_node.get("boundary_abs_margin_max"))
    if not boundary_rule_version:
        raise ValueError("pw_matrix.geometry_optional_claim.boundary_rule_version must be non-empty str")
    if not boundary_metric:
        raise ValueError("pw_matrix.geometry_optional_claim.boundary_metric must be non-empty str")
    if boundary_abs_margin_min is None or boundary_abs_margin_min < 0.0:
        raise ValueError("pw_matrix.geometry_optional_claim.boundary_abs_margin_min must be finite and >= 0")
    if boundary_abs_margin_max is None or boundary_abs_margin_max <= 0.0:
        raise ValueError("pw_matrix.geometry_optional_claim.boundary_abs_margin_max must be finite and > 0")
    if boundary_abs_margin_max < boundary_abs_margin_min:
        raise ValueError("pw_matrix geometry boundary max must be >= min")

    system_event_count_sweep_node = matrix_cfg.get("system_event_count_sweep")
    if not isinstance(system_event_count_sweep_node, Mapping):
        raise ValueError("pw_matrix.system_event_count_sweep must be mapping")
    event_counts = sorted(
        {
            int(event_count)
            for event_count in cast(List[int], _extract_int_list(system_event_count_sweep_node.get("event_counts")))
            if int(event_count) > 0
        }
    )
    repeat_count = _extract_int(system_event_count_sweep_node.get("repeat_count"))
    random_seed = _extract_int(system_event_count_sweep_node.get("random_seed"))
    if not event_counts:
        raise ValueError("pw_matrix.system_event_count_sweep.event_counts must contain positive ints")
    if repeat_count is None or repeat_count <= 0:
        raise ValueError("pw_matrix.system_event_count_sweep.repeat_count must be positive int")
    if random_seed is None:
        raise ValueError("pw_matrix.system_event_count_sweep.random_seed must be int")

    return {
        "matrix_profile": matrix_profile,
        "matrix_version": matrix_version,
        "materialization_profile": materialization_profile,
        "attack_sets": attack_sets,
        "concrete_conditions": concrete_conditions,
        "geometry_optional_claim": {
            "candidate_attack_set": candidate_attack_set,
            "candidate_attack_families": list(attack_sets[candidate_attack_set]),
            "boundary_rule_version": boundary_rule_version,
            "boundary_metric": boundary_metric,
            "boundary_abs_margin_min": boundary_abs_margin_min,
            "boundary_abs_margin_max": boundary_abs_margin_max,
        },
        "system_event_count_sweep": {
            "event_counts": event_counts,
            "repeat_count": repeat_count,
            "random_seed": random_seed,
        },
    }


def _safe_mean(values: Sequence[int | float]) -> float | None:
    """
    Compute the finite mean for one numeric sequence.

    Args:
        values: Numeric values.

    Returns:
        Finite mean when available; otherwise None.
    """
    if not isinstance(values, Sequence):
        raise TypeError("values must be Sequence")

    normalized_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not normalized_values:
        return None
    return float(sum(normalized_values) / len(normalized_values))


def _payload_bytes_to_bipolar_bits(payload_bytes: bytes, message_length: int) -> List[int]:
    """
    Convert payload bytes to bipolar bits.

    Args:
        payload_bytes: Payload bytes.
        message_length: Required message length.

    Returns:
        Message bits encoded as {-1, +1}.
    """
    if not isinstance(payload_bytes, bytes) or not payload_bytes:
        raise ValueError("payload_bytes must be non-empty bytes")
    if not isinstance(message_length, int) or message_length <= 0:
        raise ValueError("message_length must be positive int")

    bits: List[int] = []
    for byte_value in payload_bytes:
        for shift in range(7, -1, -1):
            bits.append(1 if ((byte_value >> shift) & 1) == 1 else -1)
            if len(bits) >= message_length:
                return bits
    raise ValueError("payload_bytes does not contain enough bits for message_length")


def build_payload_reference_sidecar_payload(
    *,
    family_id: str,
    stage_name: str,
    event_id: str,
    event_index: int,
    sample_role: str,
    prompt_sha256: str | None,
    seed: int | None,
    embed_record: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build one event-level payload reference sidecar.

    Args:
        family_id: Family identifier.
        stage_name: Stage name.
        event_id: Event identifier.
        event_index: Event index.
        sample_role: Event sample role.
        prompt_sha256: Optional prompt digest.
        seed: Optional event seed.
        embed_record: Staged embed record payload.

    Returns:
        Payload reference sidecar payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(event_id, str) or not event_id:
        raise TypeError("event_id must be non-empty str")
    if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 0:
        raise TypeError("event_index must be non-negative int")
    if not isinstance(sample_role, str) or not sample_role:
        raise TypeError("sample_role must be non-empty str")
    if not isinstance(embed_record, Mapping):
        raise TypeError("embed_record must be Mapping")

    content_payload = _extract_mapping(embed_record.get("content_evidence") or embed_record.get("content_result"))
    score_parts = _extract_mapping(content_payload.get("score_parts"))
    lf_metrics = _extract_mapping(score_parts.get("lf_metrics"))
    if not lf_metrics:
        embed_trace = _extract_mapping(embed_record.get("embed_trace"))
        lf_metrics = _extract_mapping(embed_trace.get("lf_trace_summary"))
    if not lf_metrics:
        raise ValueError("embed_record missing lf_metrics required for payload reference sidecar")

    attestation_payload = _extract_mapping(embed_record.get("attestation"))
    message_length = _extract_int(lf_metrics.get("message_length"))
    if message_length is None or message_length <= 0:
        raise ValueError("embed_record lf_metrics missing valid message_length")
    ecc_sparsity = _extract_int(lf_metrics.get("ecc_sparsity")) or 3
    plan_digest = str(lf_metrics.get("plan_digest") or content_payload.get("plan_digest") or "")
    basis_digest = str(lf_metrics.get("basis_digest") or content_payload.get("basis_digest") or embed_record.get("basis_digest") or "")
    attestation_event_digest = str(
        attestation_payload.get("event_binding_digest")
        or lf_metrics.get("attestation_event_digest")
        or ""
    )
    event_binding_mode = attestation_payload.get("event_binding_mode")
    message_source = str(lf_metrics.get("message_source") or "plan_digest")
    parity_check_digest = lf_metrics.get("parity_check_digest")

    lf_payload_hex = attestation_payload.get("lf_payload_hex")
    if isinstance(lf_payload_hex, str) and lf_payload_hex:
        message_bits = _payload_bytes_to_bipolar_bits(bytes.fromhex(lf_payload_hex), message_length)
    else:
        seed_material = {
            "plan_digest": plan_digest,
            "tag": "lf_message",
            "basis_digest": basis_digest,
            "attestation_event_digest": attestation_event_digest,
        }
        seed_value = int(canonical_mapping_sha256(seed_material)[:16], 16)
        rng = random.Random(seed_value)
        message_bits = [1 if rng.random() < 0.5 else -1 for _ in range(message_length)]

    ldpc_seed_key = canonical_mapping_sha256(
        {
            "plan_digest": plan_digest,
            "basis_digest": basis_digest,
            "attestation_event_digest": attestation_event_digest,
            "message_length": message_length,
            "ecc_sparsity": ecc_sparsity,
            "message_source": message_source,
            "channel": "lf",
        }
    )
    ldpc_spec = build_ldpc_spec(
        message_length=message_length,
        ecc_sparsity=ecc_sparsity,
        seed_key=ldpc_seed_key,
    )
    code_bits = encode_message_bits(message_bits, ldpc_spec)

    reference_payload_digest = canonical_mapping_sha256(
        {
            "message_bits": message_bits,
            "code_bits": code_bits,
            "message_source": message_source,
            "plan_digest": plan_digest,
            "basis_digest": basis_digest,
            "attestation_event_digest": attestation_event_digest,
            "parity_check_digest": parity_check_digest,
        }
    )
    payload_binding_digest = canonical_mapping_sha256(
        {
            "reference_event_id": event_id,
            "sample_role": sample_role,
            "prompt_sha256": prompt_sha256,
            "seed": seed,
            "reference_payload_digest": reference_payload_digest,
        }
    )

    return {
        "artifact_type": "paper_workflow_payload_reference_sidecar",
        "schema_version": PAYLOAD_SIDECAR_SCHEMA_VERSION,
        "stage_name": stage_name,
        "family_id": family_id,
        "event_id": event_id,
        "event_index": event_index,
        "sample_role": sample_role,
        "reference_event_id": event_id,
        "prompt_sha256": prompt_sha256,
        "seed": seed,
        "message_source": message_source,
        "event_binding_mode": event_binding_mode if isinstance(event_binding_mode, str) and event_binding_mode else None,
        "plan_digest": plan_digest or None,
        "basis_digest": basis_digest or None,
        "attestation_event_digest": attestation_event_digest or None,
        "parity_check_digest": parity_check_digest if isinstance(parity_check_digest, str) and parity_check_digest else None,
        "message_length": message_length,
        "code_length": len(code_bits),
        "message_bits": message_bits,
        "code_bits": code_bits,
        "reference_payload_digest": reference_payload_digest,
        "payload_binding_digest": payload_binding_digest,
    }


def _derive_payload_probe_exact_fields(
    *,
    n_bits_compared: int | None,
    agreement_count: int | None,
    codeword_agreement: float | None,
    reference_bit_count: int | None,
    margin_threshold: float | None,
    support_rate: float | None = None,
) -> Dict[str, Any]:
    """
    Derive exact payload-probe fields using append-only compatibility rules.

    Args:
        n_bits_compared: Effective compared bit count.
        agreement_count: Agreement count when available.
        codeword_agreement: Bit-accuracy style agreement score.
        reference_bit_count: Total reference code length when available.
        margin_threshold: Margin threshold used by LF alignment repair logic.
        support_rate: Optional precomputed support rate.

    Returns:
        Exact payload-probe field mapping.
    """
    probe_effective_n_bits = (
        int(n_bits_compared)
        if isinstance(n_bits_compared, int) and not isinstance(n_bits_compared, bool) and n_bits_compared > 0
        else None
    )
    probe_agreement_count = (
        int(agreement_count)
        if isinstance(agreement_count, int)
        and not isinstance(agreement_count, bool)
        and agreement_count >= 0
        and (
            probe_effective_n_bits is None
            or agreement_count <= probe_effective_n_bits
        )
        else None
    )
    if (
        probe_agreement_count is None
        and probe_effective_n_bits is not None
        and codeword_agreement is not None
    ):
        derived_agreement_count = int(round(float(codeword_agreement) * float(probe_effective_n_bits)))
        probe_agreement_count = max(0, min(probe_effective_n_bits, derived_agreement_count))

    probe_bit_accuracy = _coerce_finite_float(codeword_agreement)
    if (
        probe_bit_accuracy is None
        and probe_effective_n_bits is not None
        and probe_agreement_count is not None
        and probe_effective_n_bits > 0
    ):
        probe_bit_accuracy = float(probe_agreement_count / probe_effective_n_bits)

    normalized_reference_bit_count = (
        int(reference_bit_count)
        if isinstance(reference_bit_count, int)
        and not isinstance(reference_bit_count, bool)
        and reference_bit_count > 0
        else None
    )
    normalized_support_rate = _coerce_finite_float(support_rate)
    if normalized_support_rate is None and probe_effective_n_bits is not None:
        if normalized_reference_bit_count is not None and normalized_reference_bit_count > 0:
            normalized_support_rate = float(probe_effective_n_bits / normalized_reference_bit_count)
        else:
            normalized_support_rate = 1.0
    if normalized_support_rate is not None:
        normalized_support_rate = float(max(0.0, min(1.0, normalized_support_rate)))

    return {
        "probe_margin_threshold": _coerce_finite_float(margin_threshold),
        "probe_reference_n_bits": normalized_reference_bit_count,
        "probe_effective_n_bits": probe_effective_n_bits,
        "probe_agreement_count": probe_agreement_count,
        "probe_bit_accuracy": probe_bit_accuracy,
        "probe_support_rate": normalized_support_rate,
    }


def build_payload_decode_sidecar_payload(
    *,
    family_id: str,
    stage_name: str,
    event_id: str,
    event_index: int,
    sample_role: str,
    reference_event_id: str,
    detect_payload: Mapping[str, Any],
    reference_sidecar: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build one event-level payload decode sidecar.

    Args:
        family_id: Family identifier.
        stage_name: Stage name.
        event_id: Event identifier.
        event_index: Event index.
        sample_role: Event sample role.
        reference_event_id: Reference positive-source event identifier.
        detect_payload: Staged detect record payload.
        reference_sidecar: Reference sidecar payload.

    Returns:
        Payload decode sidecar payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(event_id, str) or not event_id:
        raise TypeError("event_id must be non-empty str")
    if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 0:
        raise TypeError("event_index must be non-negative int")
    if not isinstance(sample_role, str) or not sample_role:
        raise TypeError("sample_role must be non-empty str")
    if not isinstance(reference_event_id, str) or not reference_event_id:
        raise TypeError("reference_event_id must be non-empty str")
    if not isinstance(detect_payload, Mapping):
        raise TypeError("detect_payload must be Mapping")
    if not isinstance(reference_sidecar, Mapping):
        raise TypeError("reference_sidecar must be Mapping")

    content_payload = _extract_mapping(detect_payload.get("content_evidence_payload"))
    score_parts = _extract_mapping(content_payload.get("score_parts"))
    lf_trace = _extract_mapping(score_parts.get("lf_trajectory_detect_trace"))
    if not lf_trace:
        lf_trace = _extract_mapping(score_parts.get("lf_detect_trace"))
    if not lf_trace:
        lf_trace = _extract_mapping(content_payload.get("lf_evidence_summary"))
    if not lf_trace:
        lf_trace = _extract_mapping(score_parts.get("lf_metrics"))

    attestation_payload = _extract_mapping(detect_payload.get("attestation"))
    trace_artifact = _extract_mapping(detect_payload.get("_lf_attestation_trace_artifact"))
    if not trace_artifact:
        trace_artifact = _extract_mapping(attestation_payload.get("_lf_attestation_trace_artifact"))
    alignment_artifact = _extract_mapping(detect_payload.get("_lf_alignment_table_artifact"))
    if not alignment_artifact:
        alignment_artifact = _extract_mapping(attestation_payload.get("_lf_alignment_table_artifact"))
    mismatch_indices = _extract_int_list(trace_artifact.get("mismatch_indices"))
    if not mismatch_indices:
        mismatch_indices = _extract_int_list(alignment_artifact.get("mismatch_indices"))

    decoded_bits = _extract_sign_list(detect_payload.get("decoded_bits"))
    decoded_bits_source = "detect_payload.decoded_bits"
    if not decoded_bits:
        decoded_bits = _extract_sign_list(trace_artifact.get("decoded_bits"))
        if decoded_bits:
            decoded_bits_source = "lf_attestation_trace_artifact.decoded_bits"
    if not decoded_bits:
        decoded_bits = _extract_sign_list(trace_artifact.get("posterior_signs"))
        if decoded_bits:
            decoded_bits_source = "lf_attestation_trace_artifact.posterior_signs"
    if not decoded_bits:
        decoded_bits = _extract_sign_list(trace_artifact.get("projected_lf_signs"))
        if decoded_bits:
            decoded_bits_source = "lf_attestation_trace_artifact.projected_lf_signs"
    if not decoded_bits:
        expected_bits = _extract_sign_list(trace_artifact.get("expected_bit_signs"))
        if not expected_bits:
            expected_bits = _extract_sign_list(alignment_artifact.get("expected_bit_signs"))
        if not expected_bits:
            expected_bits = _extract_sign_list(reference_sidecar.get("code_bits"))
        if expected_bits:
            decoded_bits = list(expected_bits)
            decoded_bits_source = "alignment_repair_from_expected_bits_and_mismatch_indices"
            for mismatch_index in mismatch_indices:
                if 0 <= mismatch_index < len(decoded_bits):
                    decoded_bits[mismatch_index] = -1 * int(decoded_bits[mismatch_index])
    if not decoded_bits:
        decoded_bits_source = "missing_upstream_decoded_bits"

    n_bits_compared = _extract_int(trace_artifact.get("n_bits_compared"))
    if n_bits_compared is None:
        n_bits_compared = _extract_int(alignment_artifact.get("n_bits_compared"))
    if n_bits_compared is None:
        n_bits_compared = _extract_int(lf_trace.get("n_bits_compared"))

    agreement_count = _extract_int(trace_artifact.get("agreement_count"))
    if agreement_count is None:
        agreement_indices = _extract_int_list(trace_artifact.get("agreement_indices"))
        if agreement_indices:
            agreement_count = len(agreement_indices)
    if agreement_count is None and isinstance(n_bits_compared, int) and mismatch_indices:
        agreement_count = max(0, n_bits_compared - len(mismatch_indices))

    codeword_agreement = _coerce_finite_float(lf_trace.get("codeword_agreement"))
    if codeword_agreement is None and isinstance(agreement_count, int) and isinstance(n_bits_compared, int) and n_bits_compared > 0:
        codeword_agreement = float(agreement_count / n_bits_compared)

    bit_error_count = None
    if isinstance(n_bits_compared, int) and mismatch_indices:
        bit_error_count = len(mismatch_indices)
    elif isinstance(n_bits_compared, int) and isinstance(agreement_count, int) and n_bits_compared >= agreement_count:
        bit_error_count = int(n_bits_compared - agreement_count)
    elif isinstance(n_bits_compared, int) and n_bits_compared > 0 and codeword_agreement is not None:
        bit_error_count = int(round((1.0 - codeword_agreement) * float(n_bits_compared)))

    if isinstance(n_bits_compared, int) and n_bits_compared > 0 and decoded_bits:
        decoded_bits = decoded_bits[:n_bits_compared]

    decode_failure_reason = None
    if not isinstance(n_bits_compared, int) or n_bits_compared <= 0:
        decode_failure_reason = str(
            lf_trace.get("lf_failure_reason")
            or lf_trace.get("content_failure_reason")
            or lf_trace.get("lf_absent_reason")
            or "missing_upstream_decoded_bits"
        )

    message_decode_success = None
    if isinstance(bit_error_count, int):
        message_decode_success = bit_error_count == 0

    payload_probe_consistency_score = codeword_agreement
    payload_probe_available = bool(
        isinstance(n_bits_compared, int)
        and n_bits_compared > 0
        and payload_probe_consistency_score is not None
    )
    payload_probe_reason = None if payload_probe_available else decode_failure_reason
    payload_probe_bp_converged = (
        trace_artifact.get("bp_converged") if isinstance(trace_artifact.get("bp_converged"), bool) else None
    )
    payload_probe_alignment_signal_available = bool(
        trace_artifact or alignment_artifact or mismatch_indices
    )
    payload_probe_reconstruction_applied = (
        decoded_bits_source == "alignment_repair_from_expected_bits_and_mismatch_indices"
    )
    reference_code_bits = _extract_sign_list(reference_sidecar.get("code_bits"))
    reference_bit_count = len(reference_code_bits) if reference_code_bits else None
    if reference_bit_count is None and decoded_bits:
        reference_bit_count = len(decoded_bits)
    probe_margin_threshold = _coerce_finite_float(alignment_artifact.get("confidence_threshold"))
    if probe_margin_threshold is None:
        probe_margin_threshold = _coerce_finite_float(alignment_artifact.get("alignment_margin_threshold"))
    if probe_margin_threshold is None:
        probe_margin_threshold = _coerce_finite_float(trace_artifact.get("confidence_threshold"))
    if probe_margin_threshold is None:
        probe_margin_threshold = _coerce_finite_float(trace_artifact.get("alignment_margin_threshold"))
    exact_probe_fields = _derive_payload_probe_exact_fields(
        n_bits_compared=n_bits_compared,
        agreement_count=agreement_count,
        codeword_agreement=codeword_agreement,
        reference_bit_count=reference_bit_count,
        margin_threshold=probe_margin_threshold,
    )
    payload_probe_consistency_score = (
        float(cast(float, exact_probe_fields["probe_bit_accuracy"]))
        if isinstance(exact_probe_fields.get("probe_bit_accuracy"), float)
        else codeword_agreement
    )

    reference_payload_digest = reference_sidecar.get("reference_payload_digest")
    payload_binding_digest = reference_sidecar.get("payload_binding_digest")
    if not isinstance(reference_payload_digest, str) or not reference_payload_digest:
        raise ValueError("reference_sidecar missing reference_payload_digest")
    if not isinstance(payload_binding_digest, str) or not payload_binding_digest:
        raise ValueError("reference_sidecar missing payload_binding_digest")

    return {
        "artifact_type": "paper_workflow_payload_decode_sidecar",
        "schema_version": PAYLOAD_SIDECAR_SCHEMA_VERSION,
        "stage_name": stage_name,
        "family_id": family_id,
        "event_id": event_id,
        "event_index": event_index,
        "sample_role": sample_role,
        "reference_event_id": reference_event_id,
        "reference_payload_digest": reference_payload_digest,
        "payload_binding_digest": payload_binding_digest,
        "message_source": (
            str(lf_trace.get("message_source"))
            if isinstance(lf_trace.get("message_source"), str) and str(lf_trace.get("message_source"))
            else reference_sidecar.get("message_source")
        ),
        "lf_detect_variant": (
            str(lf_trace.get("detect_variant"))
            if isinstance(lf_trace.get("detect_variant"), str) and str(lf_trace.get("detect_variant"))
            else detect_payload.get("lf_detect_variant")
        ),
        "decoded_bits": decoded_bits or None,
        "n_bits_compared": n_bits_compared,
        "agreement_count": agreement_count,
        "bit_error_count": bit_error_count,
        "codeword_agreement": codeword_agreement,
        **exact_probe_fields,
        "message_decode_success": message_decode_success,
        "decode_failure_reason": decode_failure_reason,
        "bp_converged": trace_artifact.get("bp_converged"),
        "bp_iteration_count": _extract_int(trace_artifact.get("bp_iteration_count")),
        "payload_probe_mode": PAYLOAD_CONSISTENCY_PROBE_MODE,
        "payload_probe_available": payload_probe_available,
        "payload_probe_status": "ready" if payload_probe_available else "not_available",
        "payload_probe_reason": payload_probe_reason,
        "payload_probe_source": decoded_bits_source,
        "payload_probe_reconstruction_applied": payload_probe_reconstruction_applied,
        "payload_probe_alignment_signal_available": payload_probe_alignment_signal_available,
        "payload_probe_consistency_score": payload_probe_consistency_score,
        "payload_probe_bp_converged": payload_probe_bp_converged,
    }


def build_payload_sidecar_status_payload(
    *,
    family_id: str,
    stage_name: str,
    event_id: str,
    event_index: int,
    sample_role: str,
    sidecar_name: str,
    required: bool,
    status: str,
    builder_name: str | None = None,
    reference_event_id: str | None = None,
    failure_reason: str | None = None,
    exception_type: str | None = None,
    exception_message: str | None = None,
    not_applicable_reason: str | None = None,
) -> Dict[str, Any]:
    """
    功能：构造 payload sidecar 生成状态工件。

    Build one payload sidecar generation-status artifact.

    Args:
        family_id: Family identifier.
        stage_name: Stage name.
        event_id: Event identifier.
        event_index: Event index.
        sample_role: Event sample role.
        sidecar_name: Stable payload sidecar name.
        required: Whether the sidecar is required on the current path.
        status: Sidecar generation status.
        builder_name: Optional authoritative builder name.
        reference_event_id: Optional reference event identifier.
        failure_reason: Optional failure reason token.
        exception_type: Optional exception type name.
        exception_message: Optional exception message.
        not_applicable_reason: Optional not-applicable reason token.

    Returns:
        Payload sidecar status artifact mapping.

    Raises:
        TypeError: If required fields are invalid.
        ValueError: If the status contract is inconsistent.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(event_id, str) or not event_id:
        raise TypeError("event_id must be non-empty str")
    if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 0:
        raise TypeError("event_index must be non-negative int")
    if not isinstance(sample_role, str) or not sample_role:
        raise TypeError("sample_role must be non-empty str")
    if not isinstance(sidecar_name, str) or not sidecar_name:
        raise TypeError("sidecar_name must be non-empty str")
    if not isinstance(required, bool):
        raise TypeError("required must be bool")
    if status not in {"ok", "failed", "not_applicable"}:
        raise ValueError("status must be one of {'ok', 'failed', 'not_applicable'}")
    if status == "failed" and (not isinstance(failure_reason, str) or not failure_reason):
        raise ValueError("failure_reason is required when status='failed'")
    if status == "not_applicable" and (not isinstance(not_applicable_reason, str) or not not_applicable_reason):
        raise ValueError("not_applicable_reason is required when status='not_applicable'")
    if required and status == "not_applicable":
        raise ValueError("required sidecar cannot use status='not_applicable'")

    return {
        "artifact_type": "paper_workflow_payload_sidecar_status",
        "schema_version": PAYLOAD_SIDECAR_STATUS_SCHEMA_VERSION,
        "family_id": family_id,
        "stage_name": stage_name,
        "event_id": event_id,
        "event_index": event_index,
        "sample_role": sample_role,
        "sidecar_name": sidecar_name,
        "required": required,
        "status": status,
        "payload_sidecar_present": status == "ok",
        "builder_name": builder_name if isinstance(builder_name, str) and builder_name else None,
        "reference_event_id": (
            reference_event_id if isinstance(reference_event_id, str) and reference_event_id else None
        ),
        "failure_reason": failure_reason if isinstance(failure_reason, str) and failure_reason else None,
        "exception_type": exception_type if isinstance(exception_type, str) and exception_type else None,
        "exception_message": (
            exception_message if isinstance(exception_message, str) and exception_message else None
        ),
        "not_applicable_reason": (
            not_applicable_reason if isinstance(not_applicable_reason, str) and not_applicable_reason else None
        ),
    }


def extract_payload_metrics_from_decode_sidecar(decode_sidecar: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract summary-friendly metrics from one payload decode sidecar.

    Args:
        decode_sidecar: Payload decode sidecar payload.

    Returns:
        Summary-friendly metric mapping.
    """
    if not isinstance(decode_sidecar, Mapping):
        raise TypeError("decode_sidecar must be Mapping")

    n_bits_compared = _extract_int(decode_sidecar.get("n_bits_compared"))
    codeword_agreement = _coerce_finite_float(decode_sidecar.get("codeword_agreement"))
    bit_error_count = _extract_int(decode_sidecar.get("bit_error_count"))
    if codeword_agreement is None and isinstance(bit_error_count, int) and isinstance(n_bits_compared, int) and n_bits_compared > 0:
        codeword_agreement = float(1.0 - (float(bit_error_count) / float(n_bits_compared)))
    probe_effective_n_bits = _extract_int(decode_sidecar.get("probe_effective_n_bits"))
    if probe_effective_n_bits is None:
        probe_effective_n_bits = n_bits_compared
    probe_agreement_count = _extract_int(decode_sidecar.get("probe_agreement_count"))
    if probe_agreement_count is None:
        probe_agreement_count = _extract_int(decode_sidecar.get("agreement_count"))
    probe_reference_n_bits = _extract_int(decode_sidecar.get("probe_reference_n_bits"))
    decoded_bits = _extract_sign_list(decode_sidecar.get("decoded_bits"))
    if probe_reference_n_bits is None and decoded_bits:
        probe_reference_n_bits = len(decoded_bits)
    probe_margin_threshold = _coerce_finite_float(decode_sidecar.get("probe_margin_threshold"))
    if probe_margin_threshold is None:
        probe_margin_threshold = _coerce_finite_float(decode_sidecar.get("alignment_margin_threshold"))
    if probe_margin_threshold is None:
        probe_margin_threshold = _coerce_finite_float(decode_sidecar.get("confidence_threshold"))
    exact_probe_fields = _derive_payload_probe_exact_fields(
        n_bits_compared=probe_effective_n_bits,
        agreement_count=probe_agreement_count,
        codeword_agreement=(
            _coerce_finite_float(decode_sidecar.get("probe_bit_accuracy"))
            or codeword_agreement
        ),
        reference_bit_count=probe_reference_n_bits,
        margin_threshold=probe_margin_threshold,
        support_rate=_coerce_finite_float(decode_sidecar.get("probe_support_rate")),
    )
    payload_probe_consistency_score = _coerce_finite_float(
        decode_sidecar.get("payload_probe_consistency_score")
    )
    if payload_probe_consistency_score is None:
        payload_probe_consistency_score = (
            float(cast(float, exact_probe_fields["probe_bit_accuracy"]))
            if isinstance(exact_probe_fields.get("probe_bit_accuracy"), float)
            else codeword_agreement
        )
    payload_probe_available = decode_sidecar.get("payload_probe_available")
    if not isinstance(payload_probe_available, bool):
        payload_probe_available = bool(
            isinstance(exact_probe_fields.get("probe_effective_n_bits"), int)
            and int(cast(int, exact_probe_fields["probe_effective_n_bits"])) > 0
            and isinstance(exact_probe_fields.get("probe_bit_accuracy"), float)
        )
    payload_probe_reason = decode_sidecar.get("payload_probe_reason")
    if not isinstance(payload_probe_reason, str) or not payload_probe_reason:
        payload_probe_reason = None
    if not payload_probe_available and payload_probe_reason is None:
        decode_failure_reason_value = decode_sidecar.get("decode_failure_reason")
        if isinstance(decode_failure_reason_value, str) and decode_failure_reason_value:
            payload_probe_reason = decode_failure_reason_value
    return {
        "codeword_agreement": codeword_agreement,
        "n_bits_compared": n_bits_compared,
        "bit_error_count": bit_error_count,
        "message_decode_success": (
            decode_sidecar.get("message_decode_success")
            if isinstance(decode_sidecar.get("message_decode_success"), bool)
            else None
        ),
        "decode_failure_reason": (
            str(decode_sidecar.get("decode_failure_reason"))
            if isinstance(decode_sidecar.get("decode_failure_reason"), str) and str(decode_sidecar.get("decode_failure_reason"))
            else None
        ),
        "message_source": (
            str(decode_sidecar.get("message_source"))
            if isinstance(decode_sidecar.get("message_source"), str) and str(decode_sidecar.get("message_source"))
            else None
        ),
        "lf_detect_variant": (
            str(decode_sidecar.get("lf_detect_variant"))
            if isinstance(decode_sidecar.get("lf_detect_variant"), str) and str(decode_sidecar.get("lf_detect_variant"))
            else None
        ),
        "reference_payload_digest": (
            str(decode_sidecar.get("reference_payload_digest"))
            if isinstance(decode_sidecar.get("reference_payload_digest"), str) and str(decode_sidecar.get("reference_payload_digest"))
            else None
        ),
        "payload_binding_digest": (
            str(decode_sidecar.get("payload_binding_digest"))
            if isinstance(decode_sidecar.get("payload_binding_digest"), str) and str(decode_sidecar.get("payload_binding_digest"))
            else None
        ),
        "payload_probe_mode": (
            str(decode_sidecar.get("payload_probe_mode"))
            if isinstance(decode_sidecar.get("payload_probe_mode"), str) and str(decode_sidecar.get("payload_probe_mode"))
            else PAYLOAD_CONSISTENCY_PROBE_MODE
        ),
        "payload_probe_available": payload_probe_available,
        "payload_probe_status": (
            str(decode_sidecar.get("payload_probe_status"))
            if isinstance(decode_sidecar.get("payload_probe_status"), str) and str(decode_sidecar.get("payload_probe_status"))
            else ("ready" if payload_probe_available else "not_available")
        ),
        "payload_probe_reason": payload_probe_reason,
        "payload_probe_source": (
            str(decode_sidecar.get("payload_probe_source"))
            if isinstance(decode_sidecar.get("payload_probe_source"), str) and str(decode_sidecar.get("payload_probe_source"))
            else "legacy_payload_metrics"
        ),
        "payload_probe_reconstruction_applied": (
            decode_sidecar.get("payload_probe_reconstruction_applied")
            if isinstance(decode_sidecar.get("payload_probe_reconstruction_applied"), bool)
            else False
        ),
        "payload_probe_alignment_signal_available": (
            decode_sidecar.get("payload_probe_alignment_signal_available")
            if isinstance(decode_sidecar.get("payload_probe_alignment_signal_available"), bool)
            else None
        ),
        "payload_probe_consistency_score": payload_probe_consistency_score,
        "payload_probe_bp_converged": (
            decode_sidecar.get("payload_probe_bp_converged")
            if isinstance(decode_sidecar.get("payload_probe_bp_converged"), bool)
            else (
                decode_sidecar.get("bp_converged")
                if isinstance(decode_sidecar.get("bp_converged"), bool)
                else None
            )
        ),
        **exact_probe_fields,
    }


def summarize_payload_probe_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    unavailable_reason: str,
) -> Dict[str, Any]:
    """
    功能：汇总 payload consistency probe 的辅助统计。 

    Summarize append-only payload consistency probe metrics.

    Args:
        rows: Row payloads carrying payload_probe_* fields.
        unavailable_reason: Stable reason when no probe is available.

    Returns:
        Probe summary payload.
    """
    if not isinstance(rows, Sequence):
        raise TypeError("rows must be Sequence")
    if not isinstance(unavailable_reason, str) or not unavailable_reason:
        raise TypeError("unavailable_reason must be non-empty str")

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        normalized_row = dict(cast(Mapping[str, Any], row))
        probe_consistency_score = (
            float(normalized_row["payload_probe_consistency_score"])
            if isinstance(normalized_row.get("payload_probe_consistency_score"), float)
            else _coerce_finite_float(normalized_row.get("codeword_agreement"))
        )
        probe_effective_n_bits = _extract_int(normalized_row.get("probe_effective_n_bits"))
        if probe_effective_n_bits is None:
            probe_effective_n_bits = _extract_int(normalized_row.get("n_bits_compared"))
        probe_agreement_count = _extract_int(normalized_row.get("probe_agreement_count"))
        if probe_agreement_count is None:
            probe_agreement_count = _extract_int(normalized_row.get("agreement_count"))
        probe_reference_n_bits = _extract_int(normalized_row.get("probe_reference_n_bits"))
        if probe_reference_n_bits is None and isinstance(normalized_row.get("decoded_bits"), list):
            decoded_bits = _extract_sign_list(normalized_row.get("decoded_bits"))
            if decoded_bits:
                probe_reference_n_bits = len(decoded_bits)
        probe_margin_threshold = _coerce_finite_float(normalized_row.get("probe_margin_threshold"))
        derived_probe_fields = _derive_payload_probe_exact_fields(
            n_bits_compared=probe_effective_n_bits,
            agreement_count=probe_agreement_count,
            codeword_agreement=(
                _coerce_finite_float(normalized_row.get("probe_bit_accuracy"))
                or probe_consistency_score
                or _coerce_finite_float(normalized_row.get("bit_accuracy"))
            ),
            reference_bit_count=probe_reference_n_bits,
            margin_threshold=probe_margin_threshold,
            support_rate=_coerce_finite_float(normalized_row.get("probe_support_rate")),
        )
        probe_available = normalized_row.get("payload_probe_available")
        if not isinstance(probe_available, bool):
            probe_available = bool(
                probe_consistency_score is not None
                and isinstance(derived_probe_fields.get("probe_effective_n_bits"), int)
                and int(cast(int, derived_probe_fields["probe_effective_n_bits"])) > 0
            )
        normalized_row["payload_probe_consistency_score"] = probe_consistency_score
        normalized_row["payload_probe_available"] = probe_available
        normalized_row.update(derived_probe_fields)
        if not isinstance(normalized_row.get("payload_probe_source"), str) and probe_available:
            normalized_row["payload_probe_source"] = "derived_from_row_metrics"
        normalized_rows.append(normalized_row)

    available_rows = [row for row in normalized_rows if row.get("payload_probe_available") is True]
    consistency_scores = [
        float(row["payload_probe_consistency_score"])
        for row in available_rows
        if isinstance(row.get("payload_probe_consistency_score"), float)
    ]
    message_success_values = [
        row["message_success"]
        for row in normalized_rows
        if isinstance(row.get("message_success"), bool)
    ]
    bp_converged_values = [
        row["payload_probe_bp_converged"]
        for row in available_rows
        if isinstance(row.get("payload_probe_bp_converged"), bool)
    ]
    alignment_signal_values = [
        row["payload_probe_alignment_signal_available"]
        for row in available_rows
        if isinstance(row.get("payload_probe_alignment_signal_available"), bool)
    ]
    reconstruction_values = [
        row["payload_probe_reconstruction_applied"]
        for row in available_rows
        if isinstance(row.get("payload_probe_reconstruction_applied"), bool)
    ]
    probe_effective_n_bits_total = sum(
        int(cast(int, row["probe_effective_n_bits"]))
        for row in available_rows
        if isinstance(row.get("probe_effective_n_bits"), int)
    )
    probe_agreement_count_total = sum(
        int(cast(int, row["probe_agreement_count"]))
        for row in available_rows
        if isinstance(row.get("probe_agreement_count"), int)
    )
    probe_reference_n_bits_total = sum(
        int(cast(int, row["probe_reference_n_bits"]))
        for row in available_rows
        if isinstance(row.get("probe_reference_n_bits"), int)
    )
    probe_bit_accuracy_values = [
        float(row["probe_bit_accuracy"])
        for row in available_rows
        if isinstance(row.get("probe_bit_accuracy"), float)
    ]
    probe_support_rate_values = [
        float(row["probe_support_rate"])
        for row in available_rows
        if isinstance(row.get("probe_support_rate"), float)
    ]
    probe_margin_threshold_values = [
        float(row["probe_margin_threshold"])
        for row in available_rows
        if isinstance(row.get("probe_margin_threshold"), float)
    ]

    if not available_rows:
        status_value = "not_available"
        reason_value = unavailable_reason
    elif len(available_rows) == len(normalized_rows):
        status_value = "ready"
        reason_value = None
    else:
        status_value = "partial"
        reason_value = f"payload probe available for {len(available_rows)}/{len(normalized_rows)} events"

    return {
        "probe_mode": PAYLOAD_CONSISTENCY_PROBE_MODE,
        "status": status_value,
        "reason": reason_value,
        "event_count": len(normalized_rows),
        "available_probe_event_count": len(available_rows),
        "missing_probe_event_count": len(normalized_rows) - len(available_rows),
        "probe_margin_threshold": (
            probe_margin_threshold_values[0]
            if len({round(value, 12) for value in probe_margin_threshold_values}) == 1 and probe_margin_threshold_values
            else _safe_mean(probe_margin_threshold_values)
        ),
        "probe_effective_n_bits": probe_effective_n_bits_total,
        "probe_agreement_count": probe_agreement_count_total,
        "probe_bit_accuracy": (
            float(probe_agreement_count_total / probe_effective_n_bits_total)
            if probe_effective_n_bits_total > 0
            else _safe_mean(probe_bit_accuracy_values)
        ),
        "probe_support_rate": (
            float(probe_effective_n_bits_total / probe_reference_n_bits_total)
            if probe_reference_n_bits_total > 0
            else _safe_mean(probe_support_rate_values)
        ),
        "mean_consistency_score": _safe_mean(consistency_scores),
        "min_consistency_score": min(consistency_scores) if consistency_scores else None,
        "max_consistency_score": max(consistency_scores) if consistency_scores else None,
        "message_success_count": sum(1 for value in message_success_values if value is True),
        "message_success_rate": (
            float(sum(1 for value in message_success_values if value is True) / len(message_success_values))
            if message_success_values
            else None
        ),
        "bp_converged_count": sum(1 for value in bp_converged_values if value is True),
        "bp_converged_rate": (
            float(sum(1 for value in bp_converged_values if value is True) / len(bp_converged_values))
            if bp_converged_values
            else None
        ),
        "alignment_signal_available_count": sum(1 for value in alignment_signal_values if value is True),
        "alignment_signal_available_rate": (
            float(sum(1 for value in alignment_signal_values if value is True) / len(alignment_signal_values))
            if alignment_signal_values
            else None
        ),
        "reconstruction_applied_count": sum(1 for value in reconstruction_values if value is True),
        "reconstruction_applied_rate": (
            float(sum(1 for value in reconstruction_values if value is True) / len(reconstruction_values))
            if reconstruction_values
            else None
        ),
        "probe_sources": sorted(
            {
                str(row.get("payload_probe_source"))
                for row in available_rows
                if isinstance(row.get("payload_probe_source"), str) and str(row.get("payload_probe_source"))
            }
        ),
    }


def build_family_root(drive_project_root: Path, family_id: str) -> Path:
    """
    Build the paper workflow family root path.

    Args:
        drive_project_root: Drive project root.
        family_id: Family identifier.

    Returns:
        Family root path.
    """
    if not family_id.strip():
        raise TypeError("family_id must be non-empty str")

    family_root = drive_project_root / "paper_workflow" / "families" / family_id.strip()
    validate_path_within_base(drive_project_root, family_root, "paper workflow family root")
    return family_root


def resolve_family_id_from_path(path_obj: Path) -> str:
    """
    Resolve one family_id by scanning ancestor layout segments.

    Args:
        path_obj: Candidate path inside one family root.

    Returns:
        Resolved family identifier.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")

    parts = list(path_obj.resolve().parts)
    for index, part in enumerate(parts[:-1]):
        if str(part).lower() == "families" and index + 1 < len(parts):
            family_id = str(parts[index + 1]).strip()
            if family_id:
                return family_id
    raise ValueError(f"unable to resolve family_id from path: {normalize_path_value(path_obj)}")


def resolve_family_layout_paths(family_root: Path) -> Dict[str, Path]:
    """
    Resolve canonical layout paths for one family root.

    Args:
        family_root: Family root path.

    Returns:
        Mapping of canonical directories and files.
    """
    manifests_root = family_root / "manifests"
    snapshots_root = family_root / "snapshots"
    source_shards_root = family_root / "source_shards"
    positive_shards_root = source_shards_root / "positive"
    negative_shards_root = source_shards_root / "negative"
    control_negative_shards_root = source_shards_root / "control_negative"
    return {
        "family_root": family_root,
        "manifests_root": manifests_root,
        "snapshots_root": snapshots_root,
        "source_shards_root": source_shards_root,
        "positive_shards_root": positive_shards_root,
        "negative_shards_root": negative_shards_root,
        "control_negative_shards_root": control_negative_shards_root,
        "runs_root": family_root / "runs",
        "logs_root": family_root / "logs",
        "runtime_state_root": family_root / "runtime_state",
        "exports_root": family_root / "exports",
        "family_manifest_path": manifests_root / "paper_eval_family_manifest.json",
        "source_event_grid_path": manifests_root / "source_event_grid.jsonl",
        "source_shard_plan_path": manifests_root / "source_shard_plan.json",
        "source_split_plan_path": manifests_root / "source_split_plan.json",
        "prompt_snapshot_path": snapshots_root / "prompt_snapshot.txt",
        "method_identity_snapshot_path": snapshots_root / "method_identity_snapshot.json",
        "config_snapshot_path": snapshots_root / "config_snapshot.yaml",
    }


def ensure_family_layout(family_root: Path) -> Dict[str, Path]:
    """
    Create the required family layout directories.

    Args:
        family_root: Family root path.

    Returns:
        Canonical layout path mapping.
    """
    layout = resolve_family_layout_paths(family_root)
    for key_name in [
        "family_root",
        "manifests_root",
        "snapshots_root",
        "positive_shards_root",
        "negative_shards_root",
        "control_negative_shards_root",
        "runs_root",
        "logs_root",
        "runtime_state_root",
        "exports_root",
    ]:
        ensure_directory(layout[key_name])
    return layout


def validate_source_sample_role(sample_role: str) -> str:
    """
    Validate one supported source sample role.

    Args:
        sample_role: Candidate source sample role.

    Returns:
        Normalized sample role.

    Raises:
        ValueError: If the role is not supported by PW00/PW01/PW02 source flow.
    """
    if not isinstance(sample_role, str) or not sample_role.strip():
        raise TypeError("sample_role must be non-empty str")

    normalized_role = sample_role.strip()
    if normalized_role not in ACTIVE_SOURCE_SAMPLE_ROLES:
        raise ValueError(f"unsupported sample_role: {normalized_role}")
    return normalized_role


def resolve_sample_role_directory_name(sample_role: str) -> str:
    """
    Resolve the directory token used by one source sample role.

    Args:
        sample_role: Supported source sample role.

    Returns:
        Directory token under source_shards/.
    """
    normalized_role = validate_source_sample_role(sample_role)
    return SAMPLE_ROLE_DIRECTORY_NAMES[normalized_role]


def build_source_shard_root(family_root: Path, sample_role: str, shard_index: int) -> Path:
    """
    Build one role-aware source shard root path.

    Args:
        family_root: Family root path.
        sample_role: Supported source sample role.
        shard_index: Zero-based shard index.

    Returns:
        Role-aware shard root path.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(shard_index, int) or isinstance(shard_index, bool) or shard_index < 0:
        raise TypeError("shard_index must be non-negative int")

    role_directory_name = resolve_sample_role_directory_name(sample_role)
    shard_root = family_root / "source_shards" / role_directory_name / f"shard_{shard_index:04d}"
    validate_path_within_base(family_root, shard_root, "paper workflow source shard root")
    return shard_root


def parse_seed_list(seed_list_value: object) -> List[int]:
    """
    Parse and normalize the seed list.

    Args:
        seed_list_value: Seed list value from CLI or config.

    Returns:
        Normalized integer seed list.

    Raises:
        ValueError: If the parsed seed list is empty.
    """
    raw_values: List[object]
    if isinstance(seed_list_value, str):
        seed_text = seed_list_value.strip()
        if not seed_text:
            raise ValueError("seed_list must be non-empty")
        if seed_text.startswith("["):
            parsed = json.loads(seed_text)
            if not isinstance(parsed, list):
                raise TypeError("seed_list JSON must be list")
            raw_values = list(cast(List[object], parsed))
        else:
            raw_values = [item.strip() for item in seed_text.split(",") if item.strip()]
    elif isinstance(seed_list_value, Sequence):
        raw_values = list(cast(Sequence[object], seed_list_value))
    else:
        raise TypeError("seed_list must be str or Sequence")

    normalized: List[int] = []
    for raw_item in raw_values:
        if isinstance(raw_item, bool):
            raise TypeError("seed values must be int, bool is not allowed")
        if isinstance(raw_item, int):
            normalized.append(int(raw_item))
            continue
        if isinstance(raw_item, str) and raw_item.strip():
            normalized.append(int(raw_item.strip()))
            continue
        raise TypeError(f"invalid seed value: {raw_item!r}")

    if not normalized:
        raise ValueError("seed_list must contain at least one seed")
    return normalized


def resolve_prompt_file_path(prompt_file: str, repo_root: Path = REPO_ROOT) -> Path:
    """
    Resolve prompt file path against repository root.

    Args:
        prompt_file: Prompt file path value.
        repo_root: Repository root path.

    Returns:
        Resolved prompt file path.
    """
    if not prompt_file.strip():
        raise TypeError("prompt_file must be non-empty str")

    candidate = Path(prompt_file.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def load_prompt_lines(prompt_file: str, repo_root: Path = REPO_ROOT) -> Tuple[Path, List[str]]:
    """
    Load non-empty prompt lines from prompt file.

    Args:
        prompt_file: Prompt file path value.
        repo_root: Repository root path.

    Returns:
        Tuple of resolved prompt path and prompt line list.

    Raises:
        FileNotFoundError: If prompt file does not exist.
        ValueError: If no non-empty prompt lines exist.
    """
    prompt_path = resolve_prompt_file_path(prompt_file, repo_root=repo_root)
    if not prompt_path.exists() or not prompt_path.is_file():
        raise FileNotFoundError(f"prompt file not found: {normalize_path_value(prompt_path)}")
    prompt_lines = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompt_lines:
        raise ValueError(f"prompt file has no non-empty lines: {normalize_path_value(prompt_path)}")
    return prompt_path, prompt_lines


def build_event_id(
    *,
    family_id: str,
    sample_role: str,
    prompt_index: int,
    prompt_sha256: str,
    seed: int,
) -> str:
    """
    Build deterministic event identifier.

    Args:
        family_id: Family identifier.
        sample_role: Sample role.
        prompt_index: Prompt index.
        prompt_sha256: Prompt SHA256 digest.
        seed: Generation seed.

    Returns:
        Stable event identifier.
    """
    if not family_id:
        raise TypeError("family_id must be non-empty str")
    if not sample_role:
        raise TypeError("sample_role must be non-empty str")
    if prompt_index < 0:
        raise TypeError("prompt_index must be non-negative int")
    if not prompt_sha256:
        raise TypeError("prompt_sha256 must be non-empty str")

    digest_payload: Dict[str, Any] = {
        "family_id": family_id,
        "sample_role": sample_role,
        "prompt_index": prompt_index,
        "prompt_sha256": prompt_sha256,
        "seed": seed,
    }
    return f"evt_{canonical_mapping_sha256(digest_payload)[:24]}"


def _build_prompt_sha256(prompt_text: str) -> str:
    """
    Compute prompt text SHA256.

    Args:
        prompt_text: Prompt text.

    Returns:
        Lowercase SHA256 digest.
    """
    if not prompt_text:
        raise TypeError("prompt_text must be non-empty str")
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def build_source_event_grid(
    *,
    family_id: str,
    prompt_lines: Sequence[str],
    seeds: Sequence[int],
    prompt_file: str,
    sample_roles: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Build deterministic multi-role source event grid.

    Args:
        family_id: Family identifier.
        prompt_lines: Prompt text sequence.
        seeds: Seed sequence.
        prompt_file: Prompt file path string.
        sample_roles: Ordered active source sample roles.

    Returns:
        Ordered event list across all requested source roles.
    """
    if not family_id:
        raise TypeError("family_id must be non-empty str")
    if not prompt_file:
        raise TypeError("prompt_file must be non-empty str")
    if not prompt_lines:
        raise ValueError("prompt_lines must be non-empty sequence")
    if not seeds:
        raise ValueError("seeds must be non-empty sequence")
    if not sample_roles:
        raise ValueError("sample_roles must be non-empty sequence")

    normalized_roles: List[str] = []
    for raw_role in sample_roles:
        normalized_role = validate_source_sample_role(str(raw_role))
        if normalized_role not in normalized_roles:
            normalized_roles.append(normalized_role)

    events: List[Dict[str, Any]] = []
    event_index = 0
    for sample_role in normalized_roles:
        for prompt_index, prompt_text_value in enumerate(prompt_lines):
            prompt_text = str(prompt_text_value).strip()
            if not prompt_text:
                raise ValueError(f"prompt_lines contains empty prompt at index {prompt_index}")
            prompt_sha256 = _build_prompt_sha256(prompt_text)
            for seed_value in seeds:
                event_id = build_event_id(
                    family_id=family_id,
                    sample_role=sample_role,
                    prompt_index=prompt_index,
                    prompt_sha256=prompt_sha256,
                    seed=int(seed_value),
                )
                events.append(
                    {
                        "event_id": event_id,
                        "event_index": event_index,
                        "sample_role": sample_role,
                        "prompt_index": prompt_index,
                        "source_prompt_index": prompt_index,
                        "prompt_text": prompt_text,
                        "prompt_sha256": prompt_sha256,
                        "seed": int(seed_value),
                        "prompt_file": prompt_file,
                    }
                )
                event_index += 1
    return events


def build_positive_source_event_grid(
    *,
    family_id: str,
    prompt_lines: Sequence[str],
    seeds: Sequence[int],
    prompt_file: str,
) -> List[Dict[str, Any]]:
    """
    Build deterministic positive_source event grid.

    Args:
        family_id: Family identifier.
        prompt_lines: Prompt text sequence.
        seeds: Seed sequence.
        prompt_file: Prompt file path string.

    Returns:
        Ordered event list.
    """
    return build_source_event_grid(
        family_id=family_id,
        prompt_lines=prompt_lines,
        seeds=seeds,
        prompt_file=prompt_file,
        sample_roles=[ACTIVE_SAMPLE_ROLE],
    )


def build_source_split_plan(
    *,
    family_id: str,
    events: Sequence[Mapping[str, Any]],
    calibration_fraction: float,
) -> Dict[str, Any]:
    """
    Build deterministic calibration/evaluate split plan for source roles.

    Args:
        family_id: Family identifier.
        events: Full source event grid.
        calibration_fraction: Fraction assigned to calibration within each role.

    Returns:
        Split-plan payload with role-specific and flattened event-id lists.
    """
    if not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(calibration_fraction, (int, float)):
        raise TypeError("calibration_fraction must be float")

    calibration_fraction_value = float(calibration_fraction)
    if not 0.0 < calibration_fraction_value < 1.0:
        raise ValueError("calibration_fraction must satisfy 0 < calibration_fraction < 1")

    events_by_role: Dict[str, List[Dict[str, Any]]] = {
        sample_role: []
        for sample_role in ACTIVE_SOURCE_SAMPLE_ROLES
    }
    for event_node in events:
        if not isinstance(event_node, Mapping):
            raise TypeError("events must contain mappings")
        event = dict(cast(Mapping[str, Any], event_node))
        sample_role = validate_source_sample_role(str(event.get("sample_role")))
        event_index = event.get("event_index")
        event_id = event.get("event_id")
        if not isinstance(event_index, int) or event_index < 0:
            raise ValueError("event_index must be non-negative int")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be non-empty str")
        events_by_role[sample_role].append(event)

    split_roles: Dict[str, Dict[str, Any]] = {}
    for sample_role in ACTIVE_SOURCE_SAMPLE_ROLES:
        ordered_events = sorted(events_by_role[sample_role], key=lambda item: int(item["event_index"]))
        if not ordered_events:
            raise ValueError(f"source split requires non-empty events for role: {sample_role}")

        calibration_count = int(len(ordered_events) * calibration_fraction_value)
        if calibration_count <= 0 or calibration_count >= len(ordered_events):
            raise ValueError(
                "source split requires non-empty calibration and evaluate partitions: "
                f"sample_role={sample_role}, event_count={len(ordered_events)}, calibration_fraction={calibration_fraction_value}"
            )

        calibration_events = ordered_events[:calibration_count]
        evaluate_events = ordered_events[calibration_count:]
        split_roles[sample_role] = {
            "event_count": len(ordered_events),
            "calibration_event_count": len(calibration_events),
            "evaluate_event_count": len(evaluate_events),
            "calibration_event_ids": [str(event["event_id"]) for event in calibration_events],
            "evaluate_event_ids": [str(event["event_id"]) for event in evaluate_events],
        }

    return {
        "artifact_type": "paper_workflow_source_split_plan",
        "schema_version": "pw_stage_02_v1",
        "family_id": family_id,
        "calibration_fraction": calibration_fraction_value,
        "sample_roles_active": list(ACTIVE_SOURCE_SAMPLE_ROLES),
        "sample_roles_reserved": list(RESERVED_SAMPLE_ROLES),
        "roles": split_roles,
        "calib_pos_event_ids": list(split_roles[ACTIVE_SAMPLE_ROLE]["calibration_event_ids"]),
        "calib_neg_event_ids": list(split_roles[CLEAN_NEGATIVE_SAMPLE_ROLE]["calibration_event_ids"]),
        "calib_control_event_ids": list(
            split_roles[PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE]["calibration_event_ids"]
        ),
        "eval_pos_event_ids": list(split_roles[ACTIVE_SAMPLE_ROLE]["evaluate_event_ids"]),
        "eval_neg_event_ids": list(split_roles[CLEAN_NEGATIVE_SAMPLE_ROLE]["evaluate_event_ids"]),
        "eval_control_event_ids": list(
            split_roles[PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE]["evaluate_event_ids"]
        ),
    }


def build_source_shard_plan(
    *,
    family_id: str,
    source_shard_count: int,
    events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build deterministic source shard plan from event grid.

    Args:
        family_id: Family identifier.
        source_shard_count: Total source shard count.
        events: Event grid entries.

    Returns:
        Source shard plan payload.
    """
    if not family_id:
        raise TypeError("family_id must be non-empty str")
    if source_shard_count <= 0:
        raise TypeError("source_shard_count must be positive int")

    role_events: Dict[str, List[Dict[str, Any]]] = {
        sample_role: []
        for sample_role in ACTIVE_SOURCE_SAMPLE_ROLES
    }
    for event_node in events:
        if not isinstance(event_node, Mapping):
            raise TypeError("events must contain mappings")
        event = dict(cast(Mapping[str, Any], event_node))
        sample_role = validate_source_sample_role(str(event.get("sample_role")))
        event_id = event.get("event_id")
        event_index = event.get("event_index")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be non-empty str")
        if not isinstance(event_index, int) or event_index < 0:
            raise ValueError("event_index must be non-negative int")
        role_events[sample_role].append(event)

    sample_role_plans: Dict[str, Dict[str, Any]] = {}
    for sample_role in ACTIVE_SOURCE_SAMPLE_ROLES:
        ordered_role_events = sorted(role_events[sample_role], key=lambda item: int(item["event_index"]))
        shard_rows: List[Dict[str, Any]] = [
            {
                "shard_index": shard_index,
                "sample_role": sample_role,
                "assigned_event_ids": [],
                "assigned_event_indices": [],
            }
            for shard_index in range(source_shard_count)
        ]
        for role_event_ordinal, event in enumerate(ordered_role_events):
            shard_index = role_event_ordinal % source_shard_count
            shard_rows[shard_index]["assigned_event_ids"].append(str(event["event_id"]))
            shard_rows[shard_index]["assigned_event_indices"].append(int(event["event_index"]))
        sample_role_plans[sample_role] = {
            "event_count": len(ordered_role_events),
            "shards": shard_rows,
        }

    return {
        "artifact_type": "paper_workflow_source_shard_plan",
        "schema_version": "pw01_v1",
        "family_id": family_id,
        "source_shard_count": source_shard_count,
        "sample_roles_active": list(ACTIVE_SOURCE_SAMPLE_ROLES),
        "sample_roles_reserved": list(RESERVED_SAMPLE_ROLES),
        "sample_role_plans": {
            **sample_role_plans,
            ATTACKED_POSITIVE_SAMPLE_ROLE: {
                "event_count": 0,
                "shards": [],
            },
        },
        "shards": list(sample_role_plans[ACTIVE_SAMPLE_ROLE]["shards"]),
    }


def _materialize_attack_param_variants_value(*, key_name: str | None, value: Any) -> List[Any]:
    """
    Materialize one protocol parameter value into concrete executable variants.

    Args:
        key_name: Optional parent key name.
        value: Raw protocol value.

    Returns:
        Concrete value variants used by PW03 attack execution.
    """
    if isinstance(value, Mapping):
        materialized_variants: List[Dict[str, Any]] = [{}]
        for nested_key, nested_value in cast(Mapping[str, Any], value).items():
            nested_variants = _materialize_attack_param_variants_value(
                key_name=str(nested_key),
                value=nested_value,
            )
            if not nested_variants:
                return []
            next_variants: List[Dict[str, Any]] = []
            for materialized_variant in materialized_variants:
                for nested_variant in nested_variants:
                    next_variant = copy.deepcopy(materialized_variant)
                    next_variant[str(nested_key)] = copy.deepcopy(nested_variant)
                    next_variants.append(next_variant)
            materialized_variants = next_variants
        return materialized_variants

    if isinstance(value, list):
        if key_name == "steps":
            return [[
                _materialize_attack_param_variants_value(key_name=None, value=item)[0]
                for item in cast(List[Any], value)
            ]]
        if not value:
            return []
        materialized_variants: List[Any] = []
        for item in cast(List[Any], value):
            materialized_variants.extend(
                copy.deepcopy(
                    _materialize_attack_param_variants_value(
                        key_name=key_name,
                        value=item,
                    )
                )
            )
        return materialized_variants

    return [copy.deepcopy(value)]


def materialize_attack_param_variants(params: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Materialize one protocol params object into concrete attack-config variants.

    Args:
        params: Protocol params mapping.

    Returns:
        Concrete attack params mappings.
    """
    if not isinstance(params, Mapping):
        raise TypeError("params must be Mapping")

    variants = _materialize_attack_param_variants_value(key_name=None, value=params)
    materialized_params: List[Dict[str, Any]] = []
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise TypeError("materialized attack param variant must be Mapping")
        materialized_params.append(dict(cast(Mapping[str, Any], variant)))
    if not materialized_params:
        raise ValueError("materialized attack params must contain at least one variant")
    return materialized_params


def materialize_attack_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Materialize one protocol params object into a concrete attack config.

    Args:
        params: Protocol params mapping.

    Returns:
        Concrete attack params mapping.
    """
    return materialize_attack_param_variants(params)[0]


def _resolve_attack_set_names(
    *,
    attack_family: str,
    attack_sets: Mapping[str, Sequence[str]],
) -> List[str]:
    """
    Resolve the stable attack-set names containing one attack family.

    Args:
        attack_family: Canonical attack family name.
        attack_sets: Mapping of attack-set names to allowed families.

    Returns:
        Sorted attack-set names.
    """
    if not isinstance(attack_family, str) or not attack_family:
        raise TypeError("attack_family must be non-empty str")
    if not isinstance(attack_sets, Mapping):
        raise TypeError("attack_sets must be Mapping")

    return sorted(
        set_name
        for set_name, families in attack_sets.items()
        if isinstance(set_name, str)
        and set_name
        and isinstance(families, Sequence)
        and attack_family in {str(family) for family in families}
    )


def _build_attack_condition_variant_suffix(
    *,
    severity_label: Any,
    attack_params_digest: str,
    variant_index: int,
) -> str:
    """
    Build one stable attack-condition suffix for a concrete variant.

    Args:
        severity_label: Optional severity label.
        attack_params_digest: Digest of the concrete params.
        variant_index: Zero-based variant index under the base condition.

    Returns:
        Stable attack-condition suffix.
    """
    if not isinstance(attack_params_digest, str) or not attack_params_digest:
        raise TypeError("attack_params_digest must be non-empty str")
    if not isinstance(variant_index, int) or isinstance(variant_index, bool) or variant_index < 0:
        raise TypeError("variant_index must be non-negative int")

    if isinstance(severity_label, str) and severity_label.strip():
        return severity_label.strip().replace(" ", "")
    return f"variant_{variant_index:02d}_{attack_params_digest[:8]}"


def _coerce_attack_numeric(value: Any) -> float | None:
    """
    Coerce one candidate attack parameter into a finite float.

    Args:
        value: Candidate scalar value.

    Returns:
        Finite float when available, otherwise None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value_float = float(value)
        return value_float if math.isfinite(value_float) else None
    if isinstance(value, str) and value.strip():
        try:
            value_float = float(value.strip())
        except ValueError:
            return None
        return value_float if math.isfinite(value_float) else None
    return None


def _resolve_attack_param_numeric(params: Mapping[str, Any], *key_names: str) -> float | None:
    """
    Resolve the first available numeric attack parameter value.

    Args:
        params: Attack parameter mapping.
        key_names: Candidate parameter names.

    Returns:
        Finite float when available, otherwise None.
    """
    for key_name in key_names:
        value_float = _coerce_attack_numeric(params.get(key_name))
        if value_float is not None:
            return value_float
    return None


def _format_attack_numeric_label(value: float) -> str:
    """
    Format one numeric severity value for stable labels.

    Args:
        value: Numeric value.

    Returns:
        Stable compact string.
    """
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.6g}"


def _build_attack_severity_metadata(
    *,
    attack_family: str,
    attack_params: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build frozen family-local severity metadata for one concrete attack condition.

    Args:
        attack_family: Canonical attack family name.
        attack_params: Concrete attack parameter mapping.

    Returns:
        Append-only severity metadata mapping.
    """
    if not isinstance(attack_family, str) or not attack_family:
        raise TypeError("attack_family must be non-empty str")
    if not isinstance(attack_params, Mapping):
        raise TypeError("attack_params must be Mapping")

    metadata: Dict[str, Any] = {
        "severity_rule_version": ATTACK_SEVERITY_RULE_VERSION,
        "severity_axis_kind": ATTACK_SEVERITY_AXIS_KIND,
        "severity_directionality": "larger_indicates_stronger_attack",
        "severity_status": "not_available",
        "severity_reason": "attack family has no frozen scalar severity rule",
        "severity_source_param": None,
        "severity_scalarization": None,
        "severity_value": None,
        "severity_sort_value": None,
        "severity_label": None,
        "severity_level_index": None,
    }

    if attack_family == "rotate":
        degrees = _resolve_attack_param_numeric(attack_params, "degrees")
        if degrees is not None:
            metadata.update(
                {
                    "severity_status": "ok",
                    "severity_reason": None,
                    "severity_source_param": "degrees",
                    "severity_scalarization": "abs_degrees",
                    "severity_value": abs(degrees),
                    "severity_sort_value": abs(degrees),
                    "severity_label": f"degrees={_format_attack_numeric_label(degrees)}",
                }
            )
        return metadata

    if attack_family == "resize":
        scale_factor = _resolve_attack_param_numeric(attack_params, "scale_factor", "scale_factors")
        if scale_factor is not None:
            severity_value = abs(1.0 - scale_factor)
            metadata.update(
                {
                    "severity_status": "ok",
                    "severity_reason": None,
                    "severity_source_param": "scale_factor",
                    "severity_scalarization": "abs_one_minus_scale_factor",
                    "severity_value": severity_value,
                    "severity_sort_value": severity_value,
                    "severity_label": f"scale_factor={_format_attack_numeric_label(scale_factor)}",
                }
            )
        return metadata

    if attack_family == "crop":
        crop_ratio = _resolve_attack_param_numeric(attack_params, "crop_ratio", "crop_ratios")
        if crop_ratio is not None:
            severity_value = max(0.0, 1.0 - crop_ratio)
            metadata.update(
                {
                    "severity_status": "ok",
                    "severity_reason": None,
                    "severity_source_param": "crop_ratio",
                    "severity_scalarization": "one_minus_crop_ratio",
                    "severity_value": severity_value,
                    "severity_sort_value": severity_value,
                    "severity_label": f"crop_ratio={_format_attack_numeric_label(crop_ratio)}",
                }
            )
        return metadata

    if attack_family == "translate":
        x_shift = _resolve_attack_param_numeric(attack_params, "x_shift")
        y_shift = _resolve_attack_param_numeric(attack_params, "y_shift")
        if x_shift is not None and y_shift is not None:
            severity_value = max(abs(x_shift), abs(y_shift))
            metadata.update(
                {
                    "severity_status": "ok",
                    "severity_reason": None,
                    "severity_source_param": "x_shift,y_shift",
                    "severity_scalarization": "max_abs_pixel_shift",
                    "severity_value": severity_value,
                    "severity_sort_value": severity_value,
                    "severity_label": (
                        f"shift=({_format_attack_numeric_label(x_shift)},{_format_attack_numeric_label(y_shift)})"
                    ),
                }
            )
        return metadata

    if attack_family == "jpeg":
        quality = _resolve_attack_param_numeric(attack_params, "quality")
        if quality is not None:
            severity_value = max(0.0, 100.0 - quality)
            metadata.update(
                {
                    "severity_status": "ok",
                    "severity_reason": None,
                    "severity_source_param": "quality",
                    "severity_scalarization": "one_hundred_minus_quality",
                    "severity_value": severity_value,
                    "severity_sort_value": severity_value,
                    "severity_label": f"quality={_format_attack_numeric_label(quality)}",
                }
            )
        return metadata

    if attack_family == "gaussian_noise":
        sigma = _resolve_attack_param_numeric(attack_params, "sigma")
        if sigma is not None:
            metadata.update(
                {
                    "severity_status": "ok",
                    "severity_reason": None,
                    "severity_source_param": "sigma",
                    "severity_scalarization": "sigma",
                    "severity_value": sigma,
                    "severity_sort_value": sigma,
                    "severity_label": f"sigma={_format_attack_numeric_label(sigma)}",
                }
            )
        return metadata

    if attack_family == "gaussian_blur":
        sigma = _resolve_attack_param_numeric(attack_params, "sigma")
        kernel_size = _resolve_attack_param_numeric(attack_params, "kernel_size")
        if sigma is not None:
            severity_value = sigma
            if kernel_size is not None:
                severity_value = float((sigma * 1000.0) + kernel_size)
            metadata.update(
                {
                    "severity_status": "ok",
                    "severity_reason": None,
                    "severity_source_param": "sigma,kernel_size",
                    "severity_scalarization": "sigma_primary_kernel_size_tiebreak",
                    "severity_value": severity_value,
                    "severity_sort_value": severity_value,
                    "severity_label": (
                        f"sigma={_format_attack_numeric_label(sigma)},kernel_size={_format_attack_numeric_label(kernel_size)}"
                        if kernel_size is not None
                        else f"sigma={_format_attack_numeric_label(sigma)}"
                    ),
                }
            )
        return metadata

    if attack_family == "composite":
        steps = attack_params.get("steps")
        step_families = [
            str(step.get("family"))
            for step in cast(List[Any], steps)
            if isinstance(step, Mapping) and isinstance(step.get("family"), str) and step.get("family")
        ] if isinstance(steps, list) else []
        metadata.update(
            {
                "severity_reason": "composite attack has no frozen scalar severity rule in current protocol",
                "severity_source_param": "steps",
                "severity_scalarization": "not_available",
                "severity_label": "+".join(step_families) if step_families else "composite",
            }
        )
        return metadata

    return metadata


def build_attack_condition_catalog(
    protocol_spec: Mapping[str, Any] | None = None,
    matrix_cfg: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Build the canonical PW03 attack-condition catalog.

    Args:
        protocol_spec: Optional standardized protocol spec. When absent the
            repository fact source is loaded.
        matrix_cfg: Optional matrix config. When absent the repository config
            is loaded.

    Returns:
        Ordered concrete attack-condition rows.
    """
    normalized_protocol_spec = (
        attack_protocol_loader.load_attack_protocol_spec()
        if protocol_spec is None
        else dict(cast(Mapping[str, Any], protocol_spec))
    )
    matrix_settings = resolve_pw_matrix_settings(
        load_pw_matrix_config() if matrix_cfg is None else matrix_cfg
    )
    materialization_profile = str(matrix_settings["materialization_profile"])
    attack_sets = cast(Mapping[str, Sequence[str]], matrix_settings["attack_sets"])
    concrete_conditions = cast(
        Mapping[str, Sequence[Mapping[str, Any]]],
        matrix_settings.get("concrete_conditions", {}),
    )
    geometry_optional_claim_settings = cast(Mapping[str, Any], matrix_settings["geometry_optional_claim"])
    geometry_candidate_families = {
        str(attack_family)
        for attack_family in cast(Sequence[str], geometry_optional_claim_settings["candidate_attack_families"])
    }
    attack_plan = eval_attack_plan.generate_attack_plan(dict(normalized_protocol_spec))
    if not attack_plan.conditions:
        raise ValueError("attack protocol must declare at least one condition")
    if materialization_profile == "matrix_defined_concrete_conditions":
        declared_condition_keys = set(concrete_conditions.keys())
        protocol_condition_keys = set(attack_plan.conditions)
        missing_condition_keys = sorted(protocol_condition_keys - declared_condition_keys)
        extra_condition_keys = sorted(declared_condition_keys - protocol_condition_keys)
        if missing_condition_keys or extra_condition_keys:
            raise ValueError(
                "pw_matrix concrete_conditions must match attack protocol conditions exactly: "
                f"missing={missing_condition_keys}, extra={extra_condition_keys}"
            )

    condition_rows: List[Dict[str, Any]] = []
    for condition_key in attack_plan.conditions:
        condition_spec = eval_attack_runner.resolve_condition_spec_from_protocol(
            dict(normalized_protocol_spec),
            condition_key,
        )
        attack_family = str(condition_spec["attack_family"])
        params_version = str(condition_spec["params_version"])
        attack_param_variants = materialize_attack_param_variants(cast(Mapping[str, Any], condition_spec["params"]))
        if materialization_profile == "matrix_defined_concrete_conditions":
            concrete_rows = concrete_conditions.get(condition_key)
            if not concrete_rows:
                raise ValueError(
                    f"pw_matrix.concrete_conditions missing rows for attack condition: {condition_key}"
                )
            allowed_variant_digests = {
                canonical_mapping_sha256(variant): variant
                for variant in attack_param_variants
            }
            seen_condition_keys: set[str] = set()
            for variant_index, concrete_row in enumerate(concrete_rows):
                attack_params = copy.deepcopy(dict(cast(Mapping[str, Any], concrete_row["params"])))
                attack_params_digest = canonical_mapping_sha256(attack_params)
                if attack_params_digest not in allowed_variant_digests:
                    raise ValueError(
                        "pw_matrix concrete condition params are not permitted by configs/attack_protocol.yaml: "
                        f"condition_key={condition_key}, condition_suffix={concrete_row.get('condition_suffix')}"
                    )
                severity_metadata = _build_attack_severity_metadata(
                    attack_family=attack_family,
                    attack_params=attack_params,
                )
                severity_metadata["severity_label"] = concrete_row.get("severity_label")
                severity_metadata["severity_level_index"] = concrete_row.get("severity_level_index")
                condition_suffix = str(concrete_row["condition_suffix"])
                attack_condition_key = f"{condition_key}::{condition_suffix}"
                if attack_condition_key in seen_condition_keys:
                    raise ValueError(
                        f"duplicate matrix-defined attack_condition_key generated: {attack_condition_key}"
                    )
                seen_condition_keys.add(attack_condition_key)
                condition_rows.append(
                    {
                        "attack_condition_base_key": condition_key,
                        "attack_family": attack_family,
                        "attack_config_name": condition_key,
                        "attack_params_version": params_version,
                        "attack_params": attack_params,
                        "attack_params_digest": attack_params_digest,
                        "attack_protocol_version": condition_spec.get("protocol_version", "<absent>"),
                        "attack_protocol_digest": condition_spec.get("protocol_digest", "<absent>"),
                        "attack_materialization_profile": materialization_profile,
                        "matrix_profile": matrix_settings["matrix_profile"],
                        "matrix_version": matrix_settings["matrix_version"],
                        "matrix_attack_set_names": _resolve_attack_set_names(
                            attack_family=attack_family,
                            attack_sets=attack_sets,
                        ),
                        "geometry_rescue_candidate": attack_family in geometry_candidate_families,
                        "attack_condition_suffix": condition_suffix,
                        "attack_condition_variant_index": variant_index,
                        "attack_condition_key": attack_condition_key,
                        **severity_metadata,
                    }
                )
            continue
        if materialization_profile == "first_value_per_condition":
            attack_param_variants = attack_param_variants[:1]
        elif materialization_profile != "protocol_list_cartesian_per_condition":
            raise ValueError(f"unsupported pw_matrix materialization_profile: {materialization_profile}")

        per_condition_rows: List[Dict[str, Any]] = []
        for attack_params in attack_param_variants:
            severity_metadata = _build_attack_severity_metadata(
                attack_family=attack_family,
                attack_params=attack_params,
            )
            attack_params_digest = canonical_mapping_sha256(attack_params)
            per_condition_rows.append(
                {
                    "attack_condition_base_key": condition_key,
                    "attack_family": attack_family,
                    "attack_config_name": condition_key,
                    "attack_params_version": params_version,
                    "attack_params": attack_params,
                    "attack_params_digest": attack_params_digest,
                    "attack_protocol_version": condition_spec.get("protocol_version", "<absent>"),
                    "attack_protocol_digest": condition_spec.get("protocol_digest", "<absent>"),
                    "attack_materialization_profile": materialization_profile,
                    "matrix_profile": matrix_settings["matrix_profile"],
                    "matrix_version": matrix_settings["matrix_version"],
                    "matrix_attack_set_names": _resolve_attack_set_names(
                        attack_family=attack_family,
                        attack_sets=attack_sets,
                    ),
                    "geometry_rescue_candidate": attack_family in geometry_candidate_families,
                    **severity_metadata,
                }
            )

        per_condition_rows = sorted(
            per_condition_rows,
            key=lambda row: (
                row.get("severity_status") != "ok",
                (
                    float(cast(float, row["severity_sort_value"]))
                    if isinstance(row.get("severity_sort_value"), (int, float))
                    and not isinstance(row.get("severity_sort_value"), bool)
                    else float("inf")
                ),
                str(row.get("severity_label") or ""),
                str(row["attack_params_digest"]),
            ),
        )
        seen_condition_keys: set[str] = set()
        for variant_index, row in enumerate(per_condition_rows):
            condition_key_value = condition_key
            if len(per_condition_rows) > 1:
                condition_suffix = _build_attack_condition_variant_suffix(
                    severity_label=row.get("severity_label"),
                    attack_params_digest=str(row["attack_params_digest"]),
                    variant_index=variant_index,
                )
                condition_key_value = f"{condition_key}::{condition_suffix}"
                if condition_key_value in seen_condition_keys:
                    condition_key_value = f"{condition_key_value}__{str(row['attack_params_digest'])[:8]}"
                row["attack_condition_suffix"] = condition_suffix
            else:
                row["attack_condition_suffix"] = None
            row["attack_condition_variant_index"] = variant_index
            row["attack_condition_key"] = condition_key_value
            seen_condition_keys.add(condition_key_value)
            condition_rows.append(row)

    for condition_index, row in enumerate(condition_rows):
        row["attack_condition_index"] = condition_index

    grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
    for condition_row in condition_rows:
        grouped_rows.setdefault(str(condition_row["attack_family"]), []).append(condition_row)
    if materialization_profile == "matrix_defined_concrete_conditions":
        return condition_rows
    for grouped_condition_rows in grouped_rows.values():
        sortable_rows = [
            row
            for row in grouped_condition_rows
            if row.get("severity_status") == "ok"
            and isinstance(row.get("severity_sort_value"), (int, float))
            and not isinstance(row.get("severity_sort_value"), bool)
        ]
        for severity_level_index, row in enumerate(
            sorted(
                sortable_rows,
                key=lambda item: (
                    float(cast(float, item["severity_sort_value"])),
                    str(item["attack_condition_key"]),
                ),
            )
        ):
            row["severity_level_index"] = severity_level_index
    return condition_rows


def build_attack_event_id(
    *,
    family_id: str,
    parent_event_id: str,
    attack_condition_key: str,
    attack_params_digest: str,
) -> str:
    """
    Build one deterministic PW03 attacked-event identifier.

    Args:
        family_id: Family identifier.
        parent_event_id: Parent positive source event identifier.
        attack_condition_key: Canonical attack condition key.
        attack_params_digest: Digest of the concrete attack params.

    Returns:
        Stable attacked-event identifier.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(parent_event_id, str) or not parent_event_id:
        raise TypeError("parent_event_id must be non-empty str")
    if not isinstance(attack_condition_key, str) or not attack_condition_key:
        raise TypeError("attack_condition_key must be non-empty str")
    if not isinstance(attack_params_digest, str) or not attack_params_digest:
        raise TypeError("attack_params_digest must be non-empty str")

    digest_payload = {
        "family_id": family_id,
        "parent_event_id": parent_event_id,
        "attack_condition_key": attack_condition_key,
        "attack_params_digest": attack_params_digest,
    }
    return f"atk_{canonical_mapping_sha256(digest_payload)[:24]}"


def build_attack_event_grid(
    *,
    family_id: str,
    parent_events: Sequence[Mapping[str, Any]],
    attack_conditions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build the deterministic attacked-event grid for PW03.

    Args:
        family_id: Family identifier.
        parent_events: Attack-eligible parent-event rows.
        attack_conditions: Concrete attack-condition rows.

    Returns:
        Ordered attacked-event rows.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not parent_events:
        raise ValueError("parent_events must be non-empty sequence")
    if not attack_conditions:
        raise ValueError("attack_conditions must be non-empty sequence")

    ordered_parent_events = sorted(
        [dict(cast(Mapping[str, Any], event)) for event in parent_events],
        key=lambda item: int(item.get("event_index", -1)),
    )
    attack_event_rows: List[Dict[str, Any]] = []
    attack_event_index = 0
    for parent_event in ordered_parent_events:
        parent_event_id = parent_event.get("event_id")
        parent_event_index = parent_event.get("event_index")
        parent_sample_role = parent_event.get("sample_role")
        if not isinstance(parent_event_id, str) or not parent_event_id:
            raise ValueError("parent event missing event_id")
        if not isinstance(parent_event_index, int) or parent_event_index < 0:
            raise ValueError("parent event missing event_index")
        if parent_sample_role not in {ACTIVE_SAMPLE_ROLE, CLEAN_NEGATIVE_SAMPLE_ROLE}:
            raise ValueError(
                "PW03 parent events must be positive_source or clean_negative, "
                f"got: {parent_sample_role}"
            )
        attack_sample_role = (
            ATTACKED_POSITIVE_SAMPLE_ROLE
            if parent_sample_role == ACTIVE_SAMPLE_ROLE
            else ATTACKED_NEGATIVE_SAMPLE_ROLE
        )

        for attack_condition_node in attack_conditions:
            attack_condition = dict(cast(Mapping[str, Any], attack_condition_node))
            attack_condition_key = attack_condition.get("attack_condition_key")
            attack_params_digest = attack_condition.get("attack_params_digest")
            if not isinstance(attack_condition_key, str) or not attack_condition_key:
                raise ValueError("attack condition missing attack_condition_key")
            if not isinstance(attack_params_digest, str) or not attack_params_digest:
                raise ValueError("attack condition missing attack_params_digest")

            attack_event_id = build_attack_event_id(
                family_id=family_id,
                parent_event_id=parent_event_id,
                attack_condition_key=attack_condition_key,
                attack_params_digest=attack_params_digest,
            )
            attack_event_rows.append(
                {
                    "event_id": attack_event_id,
                    "attack_event_id": attack_event_id,
                    "event_index": attack_event_index,
                    "attack_event_index": attack_event_index,
                    "sample_role": attack_sample_role,
                    "parent_event_id": parent_event_id,
                    "parent_event_index": parent_event_index,
                    "parent_sample_role": parent_sample_role,
                    "attack_family": attack_condition.get("attack_family"),
                    "attack_config_name": attack_condition.get("attack_config_name"),
                    "attack_condition_base_key": attack_condition.get("attack_condition_base_key"),
                    "attack_condition_key": attack_condition_key,
                    "attack_condition_variant_index": attack_condition.get("attack_condition_variant_index"),
                    "attack_condition_suffix": attack_condition.get("attack_condition_suffix"),
                    "attack_params_version": attack_condition.get("attack_params_version"),
                    "attack_params": copy.deepcopy(attack_condition.get("attack_params", {})),
                    "attack_params_digest": attack_params_digest,
                    "attack_protocol_version": attack_condition.get("attack_protocol_version"),
                    "attack_protocol_digest": attack_condition.get("attack_protocol_digest"),
                    "attack_materialization_profile": attack_condition.get("attack_materialization_profile"),
                    "matrix_profile": attack_condition.get("matrix_profile"),
                    "matrix_version": attack_condition.get("matrix_version"),
                    "matrix_attack_set_names": copy.deepcopy(attack_condition.get("matrix_attack_set_names", [])),
                    "geometry_rescue_candidate": attack_condition.get("geometry_rescue_candidate"),
                    "severity_rule_version": attack_condition.get("severity_rule_version"),
                    "severity_axis_kind": attack_condition.get("severity_axis_kind"),
                    "severity_directionality": attack_condition.get("severity_directionality"),
                    "severity_status": attack_condition.get("severity_status"),
                    "severity_reason": attack_condition.get("severity_reason"),
                    "severity_source_param": attack_condition.get("severity_source_param"),
                    "severity_scalarization": attack_condition.get("severity_scalarization"),
                    "severity_value": attack_condition.get("severity_value"),
                    "severity_sort_value": attack_condition.get("severity_sort_value"),
                    "severity_label": attack_condition.get("severity_label"),
                    "severity_level_index": attack_condition.get("severity_level_index"),
                }
            )
            attack_event_index += 1
    return attack_event_rows


def build_attack_shard_plan(
    *,
    family_id: str,
    attack_shard_count: int,
    events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build deterministic PW03 attack shard assignments.

    Args:
        family_id: Family identifier.
        attack_shard_count: Total attack shard count.
        events: Ordered attacked-event grid rows.

    Returns:
        Attack shard plan payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(attack_shard_count, int) or isinstance(attack_shard_count, bool) or attack_shard_count <= 0:
        raise TypeError("attack_shard_count must be positive int")

    shard_rows: List[Dict[str, Any]] = [
        {
            "attack_shard_index": shard_index,
            "assigned_attack_event_ids": [],
            "assigned_attack_event_indices": [],
            "assigned_parent_event_ids": [],
            "assigned_attack_families": [],
            "assigned_attack_config_names": [],
        }
        for shard_index in range(attack_shard_count)
    ]

    ordered_events = sorted(
        [dict(cast(Mapping[str, Any], event)) for event in events],
        key=lambda item: int(item.get("attack_event_index", item.get("event_index", -1))),
    )
    for attack_event_ordinal, event in enumerate(ordered_events):
        attack_event_id = event.get("attack_event_id", event.get("event_id"))
        attack_event_index = event.get("attack_event_index", event.get("event_index"))
        parent_event_id = event.get("parent_event_id")
        attack_family = event.get("attack_family")
        attack_config_name = event.get("attack_config_name")
        if not isinstance(attack_event_id, str) or not attack_event_id:
            raise ValueError("attack event missing attack_event_id")
        if not isinstance(attack_event_index, int) or attack_event_index < 0:
            raise ValueError("attack event missing attack_event_index")
        if not isinstance(parent_event_id, str) or not parent_event_id:
            raise ValueError("attack event missing parent_event_id")
        if not isinstance(attack_family, str) or not attack_family:
            raise ValueError("attack event missing attack_family")
        if not isinstance(attack_config_name, str) or not attack_config_name:
            raise ValueError("attack event missing attack_config_name")

        shard_index = attack_event_ordinal % attack_shard_count
        shard_row = shard_rows[shard_index]
        cast(List[str], shard_row["assigned_attack_event_ids"]).append(attack_event_id)
        cast(List[int], shard_row["assigned_attack_event_indices"]).append(attack_event_index)
        cast(List[str], shard_row["assigned_parent_event_ids"]).append(parent_event_id)
        cast(List[str], shard_row["assigned_attack_families"]).append(attack_family)
        cast(List[str], shard_row["assigned_attack_config_names"]).append(attack_config_name)

    attack_sample_roles = sorted(
        {
            str(event.get("sample_role"))
            for event in ordered_events
            if isinstance(event.get("sample_role"), str) and str(event.get("sample_role"))
        }
    )
    materialization_profiles = sorted(
        {
            str(event.get("attack_materialization_profile"))
            for event in ordered_events
            if isinstance(event.get("attack_materialization_profile"), str)
            and str(event.get("attack_materialization_profile"))
        }
    )
    matrix_profiles = sorted(
        {
            str(event.get("matrix_profile"))
            for event in ordered_events
            if isinstance(event.get("matrix_profile"), str) and str(event.get("matrix_profile"))
        }
    )
    matrix_versions = sorted(
        {
            str(event.get("matrix_version"))
            for event in ordered_events
            if isinstance(event.get("matrix_version"), str) and str(event.get("matrix_version"))
        }
    )
    return {
        "artifact_type": "paper_workflow_attack_shard_plan",
        "schema_version": "pw_stage_03_v1",
        "family_id": family_id,
        "sample_role": attack_sample_roles[0] if len(attack_sample_roles) == 1 else MIXED_ATTACK_SAMPLE_ROLE,
        "attack_sample_roles": attack_sample_roles,
        "attack_shard_count": attack_shard_count,
        "attack_event_count": len(ordered_events),
        "materialization_profile": materialization_profiles[0] if len(materialization_profiles) == 1 else materialization_profiles,
        "matrix_profile": matrix_profiles[0] if len(matrix_profiles) == 1 else matrix_profiles,
        "matrix_version": matrix_versions[0] if len(matrix_versions) == 1 else matrix_versions,
        "shards": shard_rows,
    }


def write_jsonl(path_obj: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """
    Write JSONL rows to file.

    Args:
        path_obj: Output JSONL path.
        rows: JSON-serializable row mappings.

    Returns:
        None.
    """
    ensure_directory(path_obj.parent)
    lines: List[str] = []
    for row in rows:
        lines.append(_canonical_json_text(row))
    path_obj.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_jsonl(path_obj: Path) -> List[Dict[str, Any]]:
    """
    Read JSONL rows from file.

    Args:
        path_obj: Input JSONL path.

    Returns:
        Parsed JSON object rows.
    """
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"jsonl file not found: {normalize_path_value(path_obj)}")

    rows: List[Dict[str, Any]] = []
    for line_index, line in enumerate(path_obj.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError(f"jsonl row must be object, line={line_index}")
        rows.append(cast(Dict[str, Any], parsed))
    return rows


def load_default_config_snapshot(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """
    Load the root default config snapshot.

    Args:
        repo_root: Repository root path.

    Returns:
        Parsed default config mapping.
    """
    return load_yaml_mapping((repo_root / DEFAULT_CONFIG_RELATIVE_PATH).resolve())


def build_method_identity_snapshot(
    *,
    default_cfg_obj: Mapping[str, Any],
    default_cfg_path: Path,
    source_alignment_reference_files: Sequence[str],
) -> Dict[str, Any]:
    """
    Build method identity snapshot for family manifest lineage.

    Args:
        default_cfg_obj: Parsed default config object.
        default_cfg_path: Root default config path.
        source_alignment_reference_files: Source truth reference files.

    Returns:
        Method identity snapshot payload.
    """
    paper_faithfulness_node = default_cfg_obj.get("paper_faithfulness")
    paper_faithfulness_cfg: Mapping[str, Any]
    if isinstance(paper_faithfulness_node, Mapping):
        paper_faithfulness_cfg = cast(Mapping[str, Any], paper_faithfulness_node)
    else:
        paper_faithfulness_cfg = cast(Mapping[str, Any], {})
    attestation_node = default_cfg_obj.get("attestation")
    attestation_cfg: Mapping[str, Any]
    if isinstance(attestation_node, Mapping):
        attestation_cfg = cast(Mapping[str, Any], attestation_node)
    else:
        attestation_cfg = cast(Mapping[str, Any], {})

    attestation_env_summary = collect_attestation_env_summary(default_cfg_obj)
    secret_fingerprint = canonical_mapping_sha256(attestation_env_summary)

    return {
        "source_truth_stage": SOURCE_TRUTH_STAGE,
        "default_config_path": normalize_path_value(default_cfg_path),
        "default_config_sha256": compute_file_sha256(default_cfg_path),
        "policy_path": default_cfg_obj.get("policy_path"),
        "paper_faithfulness": {
            "enabled": paper_faithfulness_cfg.get("enabled"),
        },
        "attestation": {
            "enabled": attestation_cfg.get("enabled"),
            "decision_mode": attestation_cfg.get("decision_mode"),
            "use_trajectory_mix": attestation_cfg.get("use_trajectory_mix"),
        },
        "pipeline_impl_id": default_cfg_obj.get("pipeline_impl_id"),
        "model_id": default_cfg_obj.get("model_id"),
        "hf_revision": default_cfg_obj.get("hf_revision"),
        "secret_fingerprint": secret_fingerprint,
        "secret_fingerprint_basis": {
            "required_env_vars": attestation_env_summary.get("required_env_vars", []),
            "present_env_vars": attestation_env_summary.get("present_env_vars", []),
            "missing_env_vars": attestation_env_summary.get("missing_env_vars", []),
        },
        "source_alignment_reference_files": [str(item) for item in source_alignment_reference_files],
    }
