"""验证弱语义名称和字段治理等级。"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.protocol.internal_record_registry import (
    INTERNAL_RECORD_FIELD_NAMES,
    INTERNAL_RECORD_SCHEMA_BINDINGS,
)
from governance.harness.lib.field_rules import FieldRegistryRow, validate_registry_rows
from governance.harness.lib.naming_rules import (
    ALLOWED_NARROW_SEMANTIC_LITERALS,
    has_malformed_semantic_numeric_suffix,
    has_ordinal_identity_text,
    has_ordinal_identity_polysemy,
    has_weak_semantic_token,
)


@pytest.mark.unit
def test_weak_semantic_identifiers_are_rejected() -> None:
    forbidden_names = (
        "method_v1",
        "method_v1v2",
        "stage_1_detector",
        "stage-1-detector",
        "p1_score",
        "proxy_metric",
        "new_detector",
        "final_result",
    )
    assert all(has_weak_semantic_token(name) for name in forbidden_names)


@pytest.mark.unit
def test_explicit_version_semantics_are_allowed() -> None:
    allowed_names = ("schema_version", "api_version", "model_revision", "upstream_commit")
    assert all(not has_weak_semantic_token(name) for name in allowed_names)


@pytest.mark.unit
def test_ordinal_work_package_identities_are_rejected() -> None:
    forbidden = (
        "A1",
        "A-2",
        "A3a",
        "A3b",
        "a3b_metric",
        "A3b-metric",
        "C0",
        "C1-P",
        "C1-M",
        "C1-E",
        "c1_specification_digest",
        "Runtime Batch 4",
        "Batch 3",
        "Batch_3",
        "Batch-3",
        "RuntimeBatch3",
        "stage_3",
        "A_1",
        "C_1",
        "R_1",
        "S_1",
        "P_1",
        "P-2",
        "p_1_metrics",
        "S1_metrics",
        "S-2-gate",
        "r1_candidate_semantics_revision",
        "c1_threshold_fit_metrics",
        "batch_12",
        "stage_8",
        "business_a2_candidate",
        "metrics_c1_threshold_fit",
        "pipeline_r1_revision",
        "gate_s2_candidate",
        "prefix_a3b_metric",
        "prefix_p_1_metric",
        "test_a2_candidate_behavior",
    )
    assert all(has_ordinal_identity_text(value) for value in forbidden)


@pytest.mark.unit
def test_batch_identity_boundary_does_not_capture_semantic_fixture_suffix() -> None:
    assert has_ordinal_identity_text("fixture_batch3")
    assert not has_ordinal_identity_text("BATCH3_ROOT")
    assert not has_ordinal_identity_text("BATCH3_SHAPE")


@pytest.mark.unit
def test_narrow_scientific_and_platform_literals_remain_allowed() -> None:
    assert ALLOWED_NARROW_SEMANTIC_LITERALS == {
        "relative_l2",
        "F32",
        "RGB8",
        "P95",
        "x86_64",
        "L4",
        "SHA-256",
        "SHA256",
    }
    assert all(
        not has_ordinal_identity_text(value)
        for value in ALLOWED_NARROW_SEMANTIC_LITERALS
    )


@pytest.mark.unit
def test_immediately_defined_local_math_notation_is_allowed() -> None:
    assert not has_ordinal_identity_text("C_0 = z_hf")
    assert not has_ordinal_identity_text("`C_1(w)` is local notation")
    assert not has_ordinal_identity_text("S_0 = f32(0)")
    assert has_ordinal_identity_text('function_id = "C_1"')


@pytest.mark.unit
def test_one_ordinal_token_cannot_name_two_formal_identities() -> None:
    assert has_ordinal_identity_polysemy(
        [("S1", "score_schema"), ("S1", "selection_protocol")]
    )
    assert not has_ordinal_identity_polysemy(
        [("R1", "candidate_semantics_revision"), ("R1", "candidate_semantics_revision")]
    )


@pytest.mark.unit
def test_field_registry_requires_semantic_level_and_description() -> None:
    rows = {
        "proxy_score": FieldRegistryRow(
            field_name="proxy_score",
            governance_level="level_1",
            category="protocol",
            required_suffix="none",
            allowed_in_claims="false",
            description="",
        )
    }
    reasons = {violation["reason"] for violation in validate_registry_rows(rows)}
    assert reasons == {
        "weak_semantic_field_name",
        "invalid_governance_level",
        "field_description_required",
    }


@pytest.mark.unit
def test_cross_boundary_ordinal_field_identity_is_rejected() -> None:
    rows = {
        "c1_specification_digest": FieldRegistryRow(
            field_name="c1_specification_digest",
            governance_level="cross_boundary",
            category="provenance",
            required_suffix="none",
            allowed_in_claims="false",
            description="旧序号字段身份测试。",
        )
    }
    reasons = {violation["reason"] for violation in validate_registry_rows(rows)}
    assert "ordinal_identity_field_name" in reasons


@pytest.mark.unit
def test_p_underscore_ordinal_field_identity_is_rejected() -> None:
    rows = {
        "p_1_metric": FieldRegistryRow(
            field_name="p_1_metric",
            governance_level="cross_boundary",
            category="provenance",
            required_suffix="none",
            allowed_in_claims="false",
            description="序号字段身份测试。",
        )
    }
    reasons = {violation["reason"] for violation in validate_registry_rows(rows)}
    assert "ordinal_identity_field_name" in reasons


@pytest.mark.unit
def test_prefixed_ordinal_field_identity_is_rejected() -> None:
    rows = {
        "gate_s2_candidate": FieldRegistryRow(
            field_name="gate_s2_candidate",
            governance_level="cross_boundary",
            category="provenance",
            required_suffix="none",
            allowed_in_claims="false",
            description="带前缀的序号字段身份测试。",
        )
    }
    reasons = {violation["reason"] for violation in validate_registry_rows(rows)}
    assert "ordinal_identity_field_name" in reasons


@pytest.mark.unit
def test_internal_executable_record_registry_is_mirrored_in_docs() -> None:
    registry_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "reference"
        / "field_registry.md"
    )
    document = registry_path.read_text(encoding="utf-8")
    documented_record_fields: set[str] = set()
    for line in document.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 8 and cells[4] == "true":
            documented_record_fields.add(cells[0].strip("`"))

    assert INTERNAL_RECORD_FIELD_NAMES <= documented_record_fields
    assert all(
        schema_identity in document
        for schema_identity in INTERNAL_RECORD_SCHEMA_BINDINGS.values()
    )


@pytest.mark.parametrize(
    ("field_name", "category", "expected_reason"),
    (
        ("backend", "placeholder", "placeholder_suffix_required"),
        ("seed_trace", "random", "random_suffix_required"),
        ("decoder_state", "intermediate", "intermediate_suffix_required"),
        ("render_artifact", "temporary", "temporary_suffix_required"),
        ("prepared_samples", "cache", "cache_suffix_required"),
    ),
)
@pytest.mark.unit
def test_cross_boundary_field_categories_require_semantic_suffixes(
    field_name: str,
    category: str,
    expected_reason: str,
) -> None:
    rows = {
        field_name: FieldRegistryRow(
            field_name=field_name,
            governance_level="cross_boundary",
            category=category,
            required_suffix="none",
            allowed_in_claims="false",
            description="合成字段规则测试。",
        )
    }
    reasons = {violation["reason"] for violation in validate_registry_rows(rows)}
    assert expected_reason in reasons


@pytest.mark.parametrize(
    ("field_name", "category"),
    (
        ("backend_placeholder", "placeholder"),
        ("seed_digest_random", "random"),
        ("decoder_state_intermediate", "intermediate"),
        ("render_artifact_temporary", "temporary"),
        ("prepared_samples_cache", "cache"),
    ),
)
@pytest.mark.unit
def test_cross_boundary_field_categories_accept_semantic_suffixes(
    field_name: str,
    category: str,
) -> None:
    rows = {
        field_name: FieldRegistryRow(
            field_name=field_name,
            governance_level="cross_boundary",
            category=category,
            required_suffix="none",
            allowed_in_claims="false",
            description="合成字段规则测试。",
        )
    }
    assert validate_registry_rows(rows) == []


@pytest.mark.unit
def test_malformed_semantic_numeric_suffix_is_narrowly_rejected() -> None:
    assert has_malformed_semantic_numeric_suffix("content_write_and_vae/3")
    assert not has_malformed_semantic_numeric_suffix("3/250")
    assert not has_malformed_semantic_numeric_suffix("0.70/0.30")
    assert not has_malformed_semantic_numeric_suffix(
        "content_relative_l2_nominal = 3/250"
    )
