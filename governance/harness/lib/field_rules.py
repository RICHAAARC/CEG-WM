"""提供 placeholder 与 random trace 字段规则。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from governance.harness.lib.naming_rules import (
    has_ordinal_identity_text,
    has_weak_semantic_identity_value,
    has_weak_semantic_path_name,
    has_weak_semantic_token,
    is_allowed_registered_numeric_field_role,
    is_scientific_l2_identifier,
)


ALLOWED_GOVERNANCE_LEVELS = {
    "internal_state",
    "cross_boundary",
    "persisted_protocol",
    "evidence_bearing",
}

# Detachable governance contract: these are the fields whose identity category
# is already registered by the project.  Keeping the contract here prevents a
# registry edit from silently disabling naming checks; it is not imported by
# project code and does not define any runtime identity value.
EXPECTED_METHOD_IDENTITY_FIELDS = frozenset(
    """
    candidate_id candidate_ids mode keyed_prg_version normal_quantile_table_sha256
    root_key_encoding stable_serialization uniform_protocol gaussian_protocol
    distribution shape spatial_shape latent_shape key_role carrier_config_digest
    mixing_coefficient lf_carrier_config_digest hf_carrier_config_digest
    embedder_config_digest content_relative_l2_nominal content_relative_l2_limit
    observation_protocol detector_identity detector_config_digest
    content_config_digest formal_mode diagnostic_identity route_config_digest
    route_identity branch calibration_identity function_id weight formula_identity
    combination_identity layer_name head_count head_width original_grid_side
    token_indices token_count relation_shape operator_identity geometry_config_digest
    geometry_ratio dihedral epsilon_inlier observation_geometry_config_digest
    search_config_digest estimation_identity_digest gamma_coverage gamma_uniqueness
    gamma_gap gamma_key gamma_inlier gamma_residual gamma_identity
    fitted_reliability_thresholds threshold_config_digest
    estimator_search_config_digest reliability_identity_digest
    rectification_config_digest preprocessing_identity detector_binding_digest
    hf_detector_identity hf_detector_config_digest tau tau_rescue threshold_identity
    decision_identity_digest expected_candidate_config_digest detector_mode
    candidate_binding_digest raw_detector_identity rectified_detector_identity
    raw_detector_config_digest rectified_detector_config_digest
    raw_preprocessing_identity rectified_preprocessing_identity routing_identity
    geometry_operation_identity geometry_estimation_identity
    geometry_reliability_identity threshold_trace raw_threshold_identity
    rectified_threshold_identity key_control_trace candidate_config_digest
    execution_expectation content_detector_binding_digest
    content_operation_config_digest geometry_operation_config_digest method_adapter
    adapter_id adapter_version component_bindings key_schedule_operations operation_id
    responsibility public_callable result_identity_field
    """.split()
)
EXPECTED_RUNTIME_IDENTITY_FIELDS = frozenset(
    """
    clean_callback_indices watermarked_callback_indices tensor_replay_identity
    materialization_replay_identity vae_scaling_factor_actual vae_shift_factor_actual
    runtime_schema_version runtime_config_digest pipeline_class
    scheduler_class inference_steps guidance_scale image_height image_width
    generation_seed_device latent_dtype template_dtype score_dtype callback_index
    callback_hold_scheduler_intervals vae_decode_protocol vae_encode_protocol
    vae_scaling_factor_source vae_shift_factor_source detection_schedule_index
    detection_conditioning_protocol qk_layer_names prompt prompt_2 prompt_3
    do_classifier_free_guidance detection_timestep qk_actual_dtype key_control
    qk_operator_identities key_controls cuda_available cpu_available cuda_device_count
    runtime_backend_name selected_device backend_type_identity
    qk_observation_callable_identity resource_identity_digest
    """.split()
)
EXPECTED_IDENTITY_FIELD_CATEGORIES = {
    **{name: "method_identity" for name in EXPECTED_METHOD_IDENTITY_FIELDS},
    **{name: "runtime_identity" for name in EXPECTED_RUNTIME_IDENTITY_FIELDS},
}


@dataclass(frozen=True)
class FieldRegistryRow:
    """表示字段登记表中的一行。"""

    field_name: str
    governance_level: str
    category: str
    required_suffix: str
    allowed_in_claims: str
    description: str


@dataclass(frozen=True)
class FieldRegistryInspection:
    """外层审计读取字段登记表的结果与不可静默忽略的问题。"""

    rows: dict[str, FieldRegistryRow]
    violations: tuple[dict[str, object], ...]


def inspect_field_registry(root: str | Path) -> FieldRegistryInspection:
    """Fail closed on a missing, malformed, duplicate, or degraded registry."""
    path = Path(root) / "docs" / "reference" / "field_registry.md"
    relative_path = "docs/reference/field_registry.md"
    rows: dict[str, FieldRegistryRow] = {}
    violations: list[dict[str, object]] = []
    if not path.exists():
        return FieldRegistryInspection(
            rows={},
            violations=(
                {"path": relative_path, "reason": "missing_field_registry"},
            ),
        )

    try:
        registry_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return FieldRegistryInspection(
            rows={},
            violations=(
                {"path": relative_path, "reason": "field_registry_unreadable"},
            ),
        )

    in_registry_table = False
    table_rows: set[str] = set()
    for line_number, line in enumerate(registry_lines, start=1):
        stripped = line.strip()
        if stripped.startswith("| field_name | governance_level | category |"):
            in_registry_table = True
            table_rows = set()
            continue
        if in_registry_table and not stripped.startswith("|"):
            in_registry_table = False
        if not in_registry_table:
            continue
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and set(cells[0]) == {"-"}:
            continue
        if len(cells) not in {7, 8}:
            violations.append(
                {
                    "path": relative_path,
                    "reason": "field_registry_row_unreadable",
                    "line": line_number,
                }
            )
            continue
        field_name = cells[0]
        if field_name in table_rows or (
            field_name in EXPECTED_IDENTITY_FIELD_CATEGORIES and field_name in rows
        ):
            violations.append(
                {
                    "path": relative_path,
                    "reason": "duplicate_field_registry_row",
                    "field_name": field_name,
                    "line": line_number,
                }
            )
            continue
        table_rows.add(field_name)
        # A few pre-existing rows omit replacement_required while retaining all
        # identity-category columns.  Their category remains unambiguous and is
        # read without mutating the project registry.
        description_index = 7 if len(cells) == 8 else 6
        parsed_row = FieldRegistryRow(
            field_name=field_name,
            governance_level=cells[1],
            category=cells[2],
            required_suffix=cells[3],
            allowed_in_claims=cells[5],
            description=cells[description_index],
        )
        rows.setdefault(field_name, parsed_row)

    if not rows:
        violations.append({"path": relative_path, "reason": "empty_field_registry"})
    for field_name, expected_category in EXPECTED_IDENTITY_FIELD_CATEGORIES.items():
        row = rows.get(field_name)
        if row is None:
            violations.append(
                {
                    "path": relative_path,
                    "reason": "registered_identity_field_missing",
                    "field_name": field_name,
                }
            )
        elif row.category != expected_category:
            violations.append(
                {
                    "path": relative_path,
                    "reason": "registered_identity_category_downgraded",
                    "field_name": field_name,
                    "expected_category": expected_category,
                    "actual_category": row.category,
                }
            )
    return FieldRegistryInspection(rows=rows, violations=tuple(violations))


def load_field_registry(root: str | Path) -> dict[str, FieldRegistryRow]:
    """读取 docs/reference/field_registry.md 中的字段登记表。"""
    return inspect_field_registry(root).rows


def validate_registry_rows(rows: dict[str, FieldRegistryRow]) -> list[dict[str, str]]:
    """校验字段登记表中的 placeholder、random、中间状态和 claim 规则。"""
    violations: list[dict[str, str]] = []
    for row in rows.values():
        if (
            not is_allowed_registered_numeric_field_role(row.field_name)
            and (
                has_weak_semantic_token(row.field_name)
                or has_weak_semantic_identity_value(row.field_name)
                or has_weak_semantic_path_name(row.field_name)
            )
        ):
            violations.append({"field_name": row.field_name, "reason": "weak_semantic_field_name"})
        if has_ordinal_identity_text(row.field_name) and not is_scientific_l2_identifier(
            row.field_name
        ):
            violations.append(
                {
                    "field_name": row.field_name,
                    "reason": "ordinal_identity_field_name",
                }
            )
        if row.governance_level not in ALLOWED_GOVERNANCE_LEVELS:
            violations.append({"field_name": row.field_name, "reason": "invalid_governance_level"})
        if not row.description or row.description.lower() in {"none", "n/a"}:
            violations.append({"field_name": row.field_name, "reason": "field_description_required"})
        if row.category == "placeholder" and not row.field_name.endswith("_placeholder"):
            violations.append({"field_name": row.field_name, "reason": "placeholder_suffix_required"})
        if row.category == "random" and not (row.field_name.endswith("_random") or row.field_name.endswith("_digest_random")):
            violations.append({"field_name": row.field_name, "reason": "random_suffix_required"})
        if row.category == "intermediate" and not row.field_name.endswith("_intermediate"):
            violations.append({"field_name": row.field_name, "reason": "intermediate_suffix_required"})
        if row.category == "temporary" and not row.field_name.endswith("_temporary"):
            violations.append({"field_name": row.field_name, "reason": "temporary_suffix_required"})
        if row.category == "cache" and not row.field_name.endswith("_cache"):
            violations.append({"field_name": row.field_name, "reason": "cache_suffix_required"})
        if row.category == "placeholder" and row.allowed_in_claims.lower() == "true":
            violations.append({"field_name": row.field_name, "reason": "placeholder_claim_support_forbidden"})
        if row.category in {"intermediate", "temporary", "cache"} and row.allowed_in_claims.lower() == "true":
            violations.append({"field_name": row.field_name, "reason": "non_final_state_claim_support_forbidden"})
    return violations
