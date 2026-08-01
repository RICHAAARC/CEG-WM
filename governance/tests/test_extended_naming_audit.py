"""验证正式代码、注释和配置键的弱语义审计。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from governance.harness.audits.audit_naming_conventions import run_audit
from governance.harness.lib.naming_rules import is_allowed_file_name


def _write_minimal_audit_fixture(
    tmp_path: Path,
    *roots: str,
) -> None:
    policy_root = tmp_path / "governance" / "policies"
    policy_root.mkdir(parents=True)
    (policy_root / "project_roots.yaml").write_text(
        json.dumps(
            {
                "root_registry": {
                    root: {"audited": True} for root in roots
                },
                "governed_files": [],
            }
        ),
        encoding="utf-8",
    )
    for root in roots:
        (tmp_path / root).mkdir()
    registry_target = tmp_path / "docs" / "reference" / "field_registry.md"
    registry_target.parent.mkdir(parents=True, exist_ok=True)
    registry_source = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "reference"
        / "field_registry.md"
    )
    registry_target.write_text(
        registry_source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def _has_violation(
    report: dict,
    *,
    path: str,
    reason: str,
) -> bool:
    return any(
        violation["path"] == path and violation["reason"] == reason
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_snake_case_notebook_file_name_is_allowed() -> None:
    assert is_allowed_file_name("runtime_qualification.ipynb")


@pytest.mark.unit
@pytest.mark.parametrize(
    "file_name",
    (
        "RuntimeQualification.ipynb",
        "runtime-qualification.ipynb",
    ),
)
def test_non_snake_case_notebook_file_name_is_rejected(
    file_name: str,
) -> None:
    assert not is_allowed_file_name(file_name)


@pytest.mark.unit
def test_code_comments_identifiers_and_config_keys_are_audited(tmp_path: Path) -> None:
    policy_root = tmp_path / "governance" / "policies"
    policy_root.mkdir(parents=True)
    (policy_root / "project_roots.yaml").write_text(
        json.dumps(
            {
                "root_registry": {
                    "main": {"audited": True},
                    "configs": {"audited": True},
                },
                "governed_files": [],
            }
        ),
        encoding="utf-8",
    )
    main_root = tmp_path / "main"
    main_root.mkdir()
    (main_root / "method.py").write_text(
        "# proxy implementation\ndef method_v2(sample):\n    return sample\n",
        encoding="utf-8",
    )
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "method.yaml").write_text("stage_1: enabled\n", encoding="utf-8")

    report = run_audit(tmp_path)

    reasons = {violation["reason"] for violation in report["violations"]}
    assert "weak_semantic_identifier" in reasons
    assert "ordinal_identity_config_key" in reasons


@pytest.mark.unit
def test_single_letter_number_python_basename_is_rejected_by_ordinal_path_rule(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    (tmp_path / "main" / "a1.py").write_text("value = 1\n", encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/a1.py",
        reason="ordinal_identity_path_component",
    )


@pytest.mark.unit
def test_prefixed_letter_number_python_basename_is_rejected_by_ordinal_path_rule(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "c1_metrics.py"
    path.write_text("value = 1\n", encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/c1_metrics.py",
        reason="ordinal_identity_path_component",
    )


@pytest.mark.unit
def test_letter_number_python_docstring_is_rejected_by_ordinal_text_rule(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text('"""C1 docstring."""\n', encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/method.py",
        reason="ordinal_identity_docstring",
    )


@pytest.mark.unit
def test_letter_number_python_comment_is_rejected_by_ordinal_text_rule(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text("# R1 comment\nvalue = 1\n", encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/method.py",
        reason="ordinal_identity_comment",
    )


@pytest.mark.unit
def test_docstring_and_comment_report_distinct_weak_and_ordinal_reasons(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        '"""proxy C1 docstring."""\n# proxy R1 comment\nvalue = 1\n',
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
    }
    assert {"ordinal_identity_docstring", "ordinal_identity_comment"} <= reasons
    assert "weak_semantic_docstring" not in reasons
    assert "weak_semantic_comment" not in reasons


@pytest.mark.unit
def test_python_label_letter_number_is_rejected_as_formal_identity_value(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text('label = "A1"\n', encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/method.py",
        reason="ordinal_identity_python_string",
    )


@pytest.mark.unit
def test_python_dictionary_letter_number_is_rejected_as_formal_identity_value(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text('metadata = {"name": "C1"}\n', encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/method.py",
        reason="ordinal_identity_python_string",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "statement",
    (
        'self.label = "A1"',
        'self.name: str = "P-2"',
    ),
)
def test_python_attribute_assignment_is_a_formal_identity_context(
    tmp_path: Path,
    statement: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "class Holder:\n"
        "    def bind(self):\n"
        f"        {statement}\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/method.py",
        reason="ordinal_identity_python_string",
    )


@pytest.mark.unit
def test_local_dataclass_positional_control_tuple_is_a_formal_identity_context(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    legacy_identity = "hf_only_" + "c0"
    path.write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class ResponsibilityValidationSpec:\n"
        "    responsibility: str\n"
        "    scientific_question: str\n"
        "    splits: tuple[str, ...]\n"
        "    metrics: tuple[str, ...]\n"
        "    negative_controls: tuple[str, ...]\n"
        "    promotion_gates: tuple[str, ...]\n"
        "    record_fields: tuple[str, ...]\n"
        "\n"
        "SPECIFICATION = ResponsibilityValidationSpec(\n"
        "    'content_detector',\n"
        "    'Does the content detector preserve attribution?',\n"
        "    ('candidate_selection',),\n"
        "    ('combined_tpr',),\n"
        f"    ({legacy_identity!r},),\n"
        "    ('content_branch_promotion_gate_passed',),\n"
        "    ('detector_trace',),\n"
        ")\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/method.py",
        reason="ordinal_identity_python_string",
    )


@pytest.mark.unit
def test_local_dataclass_non_identity_positional_string_is_not_scanned(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    ordinary_value = "hf_only_" + "c0"
    path.write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class PayloadEnvelope:\n"
        "    payload: str\n"
        "    description: str\n"
        "\n"
        f"ENVELOPE = PayloadEnvelope({ordinary_value!r}, 'ordinary payload')\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert report["decision"] == "pass"
    assert not _has_violation(
        report,
        path="main/method.py",
        reason="ordinal_identity_python_string",
    )


@pytest.mark.unit
def test_local_dataclass_keyword_control_tuple_is_a_formal_identity_context(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    legacy_identity = "hf_only_" + "c0"
    path.write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class ValidationSpec:\n"
        "    responsibility: str\n"
        "    negative_controls: tuple[str, ...]\n"
        "\n"
        "SPECIFICATION = ValidationSpec(\n"
        "    responsibility='content_detector',\n"
        f"    negative_controls=({legacy_identity!r},),\n"
        ")\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/method.py",
        reason="ordinal_identity_python_string",
    )


@pytest.mark.unit
def test_p_underscore_path_and_identifier_are_rejected(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "p_1_metrics.py"
    path.write_text("p_1_metric = 1\n", encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/p_1_metrics.py",
        reason="ordinal_identity_path_component",
    )
    assert _has_violation(
        report,
        path="main/p_1_metrics.py",
        reason="ordinal_identity_identifier",
    )


@pytest.mark.unit
def test_letter_number_variant_suffix_is_rejected_across_python_identity_surfaces(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "a3b_metric.py"
    path.write_text(
        'a3b_metric = 1\nlabel = "a3b_metric"\n',
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/a3b_metric.py",
        reason="ordinal_identity_path_component",
    )
    assert _has_violation(
        report,
        path="main/a3b_metric.py",
        reason="ordinal_identity_identifier",
    )
    assert _has_violation(
        report,
        path="main/a3b_metric.py",
        reason="ordinal_identity_python_string",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "reason"),
    (
        ("business_a2_candidate = 1\n", "ordinal_identity_identifier"),
        ('label = "prefix_a3b_metric"\n', "ordinal_identity_python_string"),
        ('"""gate_s2_candidate"""\n', "ordinal_identity_docstring"),
        ("# metrics_c1_threshold_fit\nvalue = 1\n", "ordinal_identity_comment"),
    ),
)
def test_prefixed_ordinal_tokens_are_rejected_on_python_surfaces(
    tmp_path: Path,
    source: str,
    reason: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(report, path="main/method.py", reason=reason)


@pytest.mark.unit
def test_prefixed_ordinal_token_is_rejected_in_config_value(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    path = tmp_path / "configs" / "identity.json"
    path.write_text(
        json.dumps({"name": "metrics_c1_threshold_fit"}),
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="configs/identity.json",
        reason="ordinal_identity_config_value",
    )


@pytest.mark.unit
def test_prefixed_ordinal_token_is_rejected_in_path_basename(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "pipeline_r1_revision.py"
    path.write_text("value = 1\n", encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/pipeline_r1_revision.py",
        reason="ordinal_identity_path_component",
    )


@pytest.mark.unit
def test_project_test_function_ordinal_identity_is_rejected(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "tests")
    unit_root = tmp_path / "tests" / "unit"
    unit_root.mkdir()
    path = unit_root / "test_candidate.py"
    path.write_text(
        "def test_a2_candidate_behavior():\n"
        "    return None\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="tests/unit/test_candidate.py",
        reason="ordinal_identity_identifier",
    )


@pytest.mark.unit
def test_project_test_class_compact_ordinal_identity_is_rejected(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "tests")
    unit_root = tmp_path / "tests" / "unit"
    unit_root.mkdir()
    path = unit_root / "test_candidate.py"
    path.write_text("class CandidateA2:\n    pass\n", encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="tests/unit/test_candidate.py",
        reason="ordinal_identity_identifier",
    )


@pytest.mark.unit
def test_project_test_class_prefixed_ordinal_identity_is_rejected(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "tests")
    unit_root = tmp_path / "tests" / "unit"
    unit_root.mkdir()
    path = unit_root / "test_candidate.py"
    path.write_text("class A2Candidate:\n    pass\n", encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="tests/unit/test_candidate.py",
        reason="ordinal_identity_identifier",
    )


@pytest.mark.unit
@pytest.mark.parametrize("source", ("def router2():\n    pass\n", "class Artifact3:\n    pass\n"))
def test_python_callable_mechanical_numeric_identity_is_rejected(
    tmp_path: Path,
    source: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "tests")
    unit_root = tmp_path / "tests" / "unit"
    unit_root.mkdir()
    path = unit_root / "test_candidate.py"
    path.write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="tests/unit/test_candidate.py",
        reason="weak_semantic_identifier",
    )


@pytest.mark.unit
def test_test_function_ordinal_rule_does_not_suppress_weak_rule(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "tests")
    unit_root = tmp_path / "tests" / "unit"
    unit_root.mkdir()
    path = unit_root / "test_candidate.py"
    path.write_text(
        "def test_p1_behavior():\n"
        "    return None\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="tests/unit/test_candidate.py",
        reason="weak_semantic_identifier",
    )


@pytest.mark.unit
def test_project_test_path_is_checked_for_ordinal_identity(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "tests")
    unit_root = tmp_path / "tests" / "unit"
    unit_root.mkdir()
    path = unit_root / "test_a2_candidate.py"
    path.write_text(
        "def test_semantic_behavior():\n"
        "    return None\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="tests/unit/test_a2_candidate.py",
        reason="ordinal_identity_path_component",
    )


@pytest.mark.unit
def test_project_test_body_is_checked_for_ordinal_identity(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "tests")
    unit_root = tmp_path / "tests" / "unit"
    unit_root.mkdir()
    path = unit_root / "test_candidate.py"
    path.write_text(
        "def test_a2_candidate_behavior():\n"
        "    business_a2_candidate = 1\n"
        '    label = "prefix_a3b_metric"\n'
        "    return business_a2_candidate, label\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="tests/unit/test_candidate.py",
        reason="ordinal_identity_identifier",
    )
    assert _has_violation(
        report,
        path="tests/unit/test_candidate.py",
        reason="ordinal_identity_python_string",
    )


@pytest.mark.unit
def test_non_project_test_function_is_checked_for_ordinal_identity(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def test_a2_candidate_behavior():\n"
        "    return None\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/method.py",
        reason="ordinal_identity_identifier",
    )


@pytest.mark.unit
def test_config_label_letter_number_is_rejected_as_formal_identity_value(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    path = tmp_path / "configs" / "identity.json"
    path.write_text(json.dumps({"label": "A1"}), encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="configs/identity.json",
        reason="ordinal_identity_config_value",
    )


@pytest.mark.unit
def test_config_name_letter_number_is_rejected_as_formal_identity_value(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    path = tmp_path / "configs" / "identity.json"
    path.write_text(json.dumps({"name": "C1"}), encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="configs/identity.json",
        reason="ordinal_identity_config_value",
    )


@pytest.mark.unit
def test_registered_python_formal_mode_rejects_ordinal_identity_value(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text('formal_mode = "A1"\n', encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/method.py",
        reason="ordinal_identity_python_string",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field_name", "legacy_value"),
    (("formal_mode", "A1"), ("mode", "C1")),
)
def test_registered_config_identity_field_rejects_ordinal_value(
    tmp_path: Path,
    field_name: str,
    legacy_value: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    path = tmp_path / "configs" / "identity.json"
    path.write_text(
        json.dumps({field_name: legacy_value}),
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="configs/identity.json",
        reason="ordinal_identity_config_value",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "ordinal_identity",
    (
        "B1",
        "D-2",
        "E_3",
        "F1",
        "G2",
        "H3",
        "M1",
        "N2",
        "T3",
        "X1",
        "Y2",
        "Z3",
        "candidate_x1_gate",
    ),
)
def test_general_single_letter_number_identity_fails_closed(
    tmp_path: Path,
    ordinal_identity: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    path = tmp_path / "configs" / "identity.json"
    path.write_text(json.dumps({"label": ordinal_identity}), encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="configs/identity.json",
        reason="ordinal_identity_config_value",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "responsibility_word",
    (
        "phase",
        "step",
        "stage",
        "batch",
        "tier",
        "level",
        "group",
        "track",
        "route",
        "gate",
        "case",
        "option",
        "variant",
        "module",
        "component",
        "method",
        "model",
        "baseline",
        "run",
        "experiment",
        "trial",
    ),
)
def test_numbered_responsibility_identity_fails_closed(
    tmp_path: Path,
    responsibility_word: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    path = tmp_path / "configs" / "identity.json"
    path.write_text(
        json.dumps({"label": f"candidate_{responsibility_word}-1_output"}),
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="configs/identity.json",
        reason="ordinal_identity_config_value",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "unknown_identity",
    (
        "tmp",
        "temp",
        "misc",
        "other",
        "todo",
        "tbd",
        "dummy",
        "fake",
        "mock",
        "proxy",
        "new",
        "old",
        "latest",
        "best",
        "final",
        "backup",
        "copy",
        "foo",
        "bar",
    ),
)
def test_registered_formal_identity_rejects_unknown_or_temporary_value(
    tmp_path: Path,
    unknown_identity: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(f"formal_mode = {unknown_identity!r}\n", encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="main/method.py",
        reason="weak_semantic_python_string",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "mechanical_identity",
    (
        "detector2",
        "metric_3",
        "config_2",
        "result4",
        "method_v2",
        "router2",
        "artifact_3",
        "candidate4",
        "protocol_7",
    ),
)
def test_mechanical_numeric_identity_fails_closed(
    tmp_path: Path,
    mechanical_identity: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    path = tmp_path / "configs" / "identity.json"
    path.write_text(json.dumps({"label": mechanical_identity}), encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path="configs/identity.json",
        reason="weak_semantic_config_value",
    )


@pytest.mark.unit
def test_non_formal_synthetic_fixture_and_real_version_roles_are_allowed(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "tests", "configs")
    unit_root = tmp_path / "tests" / "unit"
    unit_root.mkdir()
    (unit_root / "test_synthetic_fixture.py").write_text(
        'fake_gpu = "fake_gpu"\nmock_backend = "mock_backend"\n',
        encoding="utf-8",
    )
    (tmp_path / "configs" / "versions.json").write_text(
        json.dumps(
            {
                "schema_version": "2026-08-01",
                "model_revision": "stabilityai/stable-diffusion-3.5-medium",
                "upstream_commit": "a" * 40,
                "platform_identity": "SD3.5",
            }
        ),
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize("surface", ("python", "config"))
def test_formal_letter_number_label_is_rejected_by_weak_value_rule(
    tmp_path: Path,
    surface: str,
) -> None:
    _write_minimal_audit_fixture(
        tmp_path,
        "main" if surface == "python" else "configs",
    )
    if surface == "python":
        path = tmp_path / "main" / "method.py"
        path.write_text('label = "P1"\n', encoding="utf-8")
        relative = "main/method.py"
        reason = "weak_semantic_python_string"
    else:
        path = tmp_path / "configs" / "identity.json"
        path.write_text(json.dumps({"label": "P1"}), encoding="utf-8")
        relative = "configs/identity.json"
        reason = "weak_semantic_config_value"

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(report, path=relative, reason=reason)


@pytest.mark.unit
@pytest.mark.parametrize("value", ("P_1", "P-2"))
@pytest.mark.parametrize("surface", ("python", "config"))
def test_formal_p_variants_are_rejected_by_ordinal_value_rule(
    tmp_path: Path,
    value: str,
    surface: str,
) -> None:
    _write_minimal_audit_fixture(
        tmp_path,
        "main" if surface == "python" else "configs",
    )
    if surface == "python":
        path = tmp_path / "main" / "method.py"
        path.write_text(f"label = {value!r}\n", encoding="utf-8")
        relative = "main/method.py"
        reason = "ordinal_identity_python_string"
    else:
        path = tmp_path / "configs" / "identity.json"
        path.write_text(json.dumps({"label": value}), encoding="utf-8")
        relative = "configs/identity.json"
        reason = "ordinal_identity_config_value"

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(report, path=relative, reason=reason)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("file_name", "content"),
    (
        ("method.md", "P1\n"),
        ("method.svg", '<svg xmlns="http://www.w3.org/2000/svg"><text>P2</text></svg>'),
        ("method.drawio", '<mxfile><diagram><mxCell value="P1"/></diagram></mxfile>'),
    ),
)
def test_text_surfaces_reject_standalone_weak_ordinal_identity(
    tmp_path: Path,
    file_name: str,
    content: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "docs")
    path = tmp_path / "docs" / file_name
    path.write_text(content, encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path=f"docs/{file_name}",
        reason="weak_semantic_markdown",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("file_name", "content"),
    (
        ("method.md", "A1\n"),
        ("method.svg", '<svg xmlns="http://www.w3.org/2000/svg"><text>S2</text></svg>'),
        ("method.drawio", '<mxfile><diagram><mxCell value="C1-P"/></diagram></mxfile>'),
    ),
)
def test_text_surfaces_reject_ordinal_identity(
    tmp_path: Path,
    file_name: str,
    content: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "docs")
    path = tmp_path / "docs" / file_name
    path.write_text(content, encoding="utf-8")

    report = run_audit(tmp_path)

    assert report["decision"] == "fail"
    assert _has_violation(
        report,
        path=f"docs/{file_name}",
        reason="ordinal_identity_markdown",
    )


@pytest.mark.unit
def test_cross_surface_ordinal_identities_are_audited(tmp_path: Path) -> None:
    policy_root = tmp_path / "governance" / "policies"
    policy_root.mkdir(parents=True)
    (policy_root / "project_roots.yaml").write_text(
        json.dumps(
            {
                "root_registry": {
                    "main": {"audited": True},
                    "configs": {"audited": True},
                    "docs": {"audited": True},
                    "notebooks": {"audited": True},
                },
                "governed_files": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "main").mkdir()
    (tmp_path / "main" / "method.py").write_text(
        'c1_specification_digest = "C1-P"\n',
        encoding="utf-8",
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "method.json").write_text(
        json.dumps(
            {
                "function_id": "C0",
                "c1_specification_digest": "x",
                "artifact_path": "results/stage2/output.json",
                "protocol_id": "S1",
                "artifact_id": "S1",
                "design_paths": ["stages/stage3/design.md"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "method.md").write_text(
        "Runtime Batch 2\n",
        encoding="utf-8",
    )
    notebook_root = tmp_path / "notebooks"
    notebook_root.mkdir()
    (notebook_root / "entrypoint.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["A3b"]},
                    {"cell_type": "code", "source": ["phase = 'C1-E'"]},
                    {"cell_type": "markdown", "source": ["P2"]},
                    {"cell_type": "code", "source": ["label = 'P1'"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_audit(tmp_path)
    reasons = {violation["reason"] for violation in report["violations"]}
    assert {
        "ordinal_identity_identifier",
        "ordinal_identity_python_string",
        "ordinal_identity_config_key",
        "ordinal_identity_config_value",
        "ordinal_identity_markdown",
        "ordinal_identity_notebook_markdown",
        "ordinal_identity_notebook_code",
        "weak_semantic_notebook_markdown",
        "ordinal_identity_polysemy",
    } <= reasons


@pytest.mark.unit
def test_narrow_literals_pass_full_cross_surface_audit(tmp_path: Path) -> None:
    policy_root = tmp_path / "governance" / "policies"
    policy_root.mkdir(parents=True)
    (policy_root / "project_roots.yaml").write_text(
        json.dumps(
            {
                "root_registry": {
                    "main": {"audited": True},
                    "configs": {"audited": True},
                    "docs": {"audited": True},
                    "notebooks": {"audited": True},
                },
                "governed_files": [],
            }
        ),
        encoding="utf-8",
    )
    literals = (
        "relative_l2 F32 RGB8 P95 x86_64 L4 SHA-256 SHA256 "
        "SD3.5 3/250 0.70/0.30 content_relative_l2_nominal = 3/250"
    )
    (tmp_path / "main").mkdir()
    (tmp_path / "main" / "method.py").write_text(
        f'ARTIFACT_IDENTITY = "{literals}"\n', encoding="utf-8"
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "method.json").write_text(
        json.dumps({"artifact_identity": literals}), encoding="utf-8"
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "method.md").write_text(literals, encoding="utf-8")
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "entrypoint.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": [literals]},
                    {"cell_type": "code", "source": [f'identity = "{literals}"']},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_malformed_semantic_numeric_suffix_is_audited(tmp_path: Path) -> None:
    policy_root = tmp_path / "governance" / "policies"
    policy_root.mkdir(parents=True)
    (policy_root / "project_roots.yaml").write_text(
        json.dumps(
            {
                "root_registry": {"main": {"audited": True}},
                "governed_files": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "main").mkdir()
    (tmp_path / "main" / "runtime_backend.py").write_text(
        '"""Backend connected to content_write_and_vae/3 protocols."""\n',
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert "malformed_semantic_numeric_suffix" in {
        violation["reason"] for violation in report["violations"]
    }
