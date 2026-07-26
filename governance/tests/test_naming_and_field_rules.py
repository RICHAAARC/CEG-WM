"""验证弱语义名称和字段治理等级。"""

from __future__ import annotations

import pytest

from governance.harness.lib.field_rules import FieldRegistryRow, validate_registry_rows
from governance.harness.lib.naming_rules import has_weak_semantic_token


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
