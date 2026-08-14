"""验证正式代码、注释和配置键的弱语义审计。"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from governance.harness.audits.audit_naming_conventions import (
    _attested_upstream_source_paths,
    _upstream_source_directory_preflight,
    run_audit,
)
from governance.harness.audits.audit_placeholder_random_fields import (
    run_audit as run_field_audit,
)
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


def _copy_attested_upstream_source(tmp_path: Path) -> Path:
    _write_minimal_audit_fixture(tmp_path, "runtime")
    source = (
        Path(__file__).resolve().parents[2]
        / "runtime"
        / "_vendor"
        / "transparent_background"
    )
    destination = tmp_path / "runtime" / "_vendor" / "transparent_background"
    destination.parent.mkdir(parents=True)
    shutil.copytree(source, destination)
    return destination


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
                "pipeline_class": "StableDiffusion3Pipeline",
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
    _write_minimal_audit_fixture(tmp_path, "main", "configs", "docs", "notebooks")
    literals = (
        "relative_l2 F32 RGB8 P95 x86_64 L4 SHA-256 SHA256 "
        "SD3.5 3/250 0.70/0.30 content_relative_l2_nominal = 3/250"
    )
    (tmp_path / "main" / "method.py").write_text(
        f'ARTIFACT_IDENTITY = "{literals}"\n'
        "decoded = b64decode(payload)\n"
        "encoded = b64encode(payload)\n"
        "encoding_error = Base64Error\n"
        "image_error = Rgb8ImageError\n"
        "quality_image = HfOnlyReferenceRgb8Image\n"
        "runtime_configuration = Sd35RuntimeConfiguration\n"
        "runtime_backend = Sd35PipelineBackend\n"
        "external_pipeline = StableDiffusion3Pipeline\n"
        "runtime_sd35_flowmatch = object()\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "method.json").write_text(
        json.dumps({"artifact_identity": literals}), encoding="utf-8"
    )
    (tmp_path / "docs" / "method.md").write_text(literals, encoding="utf-8")
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier",
    (
        "router2",
        "candidate4",
        "router2_gate",
        "candidate4Gate",
        "router2gate",
        "candidate4selector",
        "routerbase64Gate",
        "b64candidate",
        "base64router",
        "rgb8candidate",
        "sd35candidate",
        "sha256router",
        "f32candidate",
        "p95router",
    ),
)
def test_business_variable_mechanical_suffix_is_rejected_by_real_audit(
    tmp_path: Path,
    identifier: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(f"{identifier} = object()\n", encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="main/method.py",
        reason="weak_semantic_identifier",
    )
    assert any(
        violation.get("identifier") == identifier
        for violation in report["violations"]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "identity",
    (
        "candidate_v2_router_protocol",
        "candidate_v7_execution_manifest",
        "candidate-v9-operation-record",
        "hf_only_v3_reference_protocol",
    ),
)
def test_version_token_role_stacking_in_label_is_rejected_by_real_audit(
    tmp_path: Path,
    identity: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    path = tmp_path / "configs" / "identity.json"
    path.write_text(json.dumps({"label": identity}), encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="configs/identity.json",
        reason="weak_semantic_config_value",
    )
    assert any(
        violation.get("value") == identity
        and violation.get("context") == "label"
        for violation in report["violations"]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "context",
    (
        "candidate_schema",
        "revision_gate",
        "compatibility_name",
        "version_label",
        "protocol_schema_identity",
        "model_revision_label",
        "schema_identity_label",
        "candidate_protocol_id",
        "candidate_schema_version",
        "candidate_model_revision",
        "xProtocolId",
        "ProtocolIdSuffix",
        "xRunPhaseId",
        "SchemaVersionSuffix",
        "x_protocol_id",
        "protocol_id_suffix",
        "xprotocolid",
        "protocolidsuffix",
        "XPROTOCOLID",
        "PROTOCOLIDSUFFIX",
        "preprotocolidpost",
        "xxrunphaseidyy",
        "schemaversionsuffix",
        "XSCHEMAVERSIONY",
        "outerprotocolidinner",
        "alphaprotocolidomega",
        "customrunphaseidbinding",
        "guardedschemaversionalias",
        "registeredmodelrevisioncopy",
        "wrappedmanifestidshadow",
        "promptidentitysuffix",
        "accessidentitysuffix",
        "operationidentitysuffix",
        "registeredkeyfamilyidsuffix",
        "internalvalidationprotocolidsuffix",
        "x-protocol-id",
        "protocol-id-suffix",
    ),
)
def test_forged_version_context_config_key_is_rejected_by_real_audit(
    tmp_path: Path,
    context: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    path = tmp_path / "configs" / "identity.json"
    identity = "candidate_v8_router_protocol"
    path.write_text(json.dumps({context: identity}), encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="configs/identity.json",
        reason="weak_semantic_config_value",
    )
    assert any(
        violation.get("value") == identity
        and violation.get("context") == context
        for violation in report["violations"]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "context",
    (
        "candidate_schema",
        "revision_gate",
        "compatibility_name",
        "version_label",
        "protocol_schema_identity",
        "model_revision_label",
        "schema_identity_label",
        "candidate_protocol_id",
        "candidate_schema_version",
        "candidate_model_revision",
        "xProtocolId",
        "ProtocolIdSuffix",
        "xRunPhaseId",
        "SchemaVersionSuffix",
        "x_protocol_id",
        "protocol_id_suffix",
        "xprotocolid",
        "protocolidsuffix",
        "XPROTOCOLID",
        "PROTOCOLIDSUFFIX",
        "preprotocolidpost",
        "xxrunphaseidyy",
        "schemaversionsuffix",
        "XSCHEMAVERSIONY",
        "outerprotocolidinner",
        "alphaprotocolidomega",
        "customrunphaseidbinding",
        "guardedschemaversionalias",
        "registeredmodelrevisioncopy",
        "wrappedmanifestidshadow",
        "promptidentitysuffix",
        "accessidentitysuffix",
        "operationidentitysuffix",
        "registeredkeyfamilyidsuffix",
        "internalvalidationprotocolidsuffix",
    ),
)
def test_forged_version_context_python_name_is_rejected_by_real_audit(
    tmp_path: Path,
    context: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        f'{context} = "candidate_v8_router_protocol"\n',
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="main/method.py",
        reason="weak_semantic_python_string",
    )
    assert any(
        violation.get("context") == context
        for violation in report["violations"]
    )
    if context in {
        "xProtocolId",
        "ProtocolIdSuffix",
        "xRunPhaseId",
        "SchemaVersionSuffix",
        "x_protocol_id",
        "protocol_id_suffix",
        "xprotocolid",
        "protocolidsuffix",
        "XPROTOCOLID",
        "PROTOCOLIDSUFFIX",
        "preprotocolidpost",
        "xxrunphaseidyy",
        "schemaversionsuffix",
        "XSCHEMAVERSIONY",
        "outerprotocolidinner",
        "alphaprotocolidomega",
        "customrunphaseidbinding",
        "guardedschemaversionalias",
        "registeredmodelrevisioncopy",
        "wrappedmanifestidshadow",
        "promptidentitysuffix",
        "accessidentitysuffix",
        "operationidentitysuffix",
        "registeredkeyfamilyidsuffix",
        "internalvalidationprotocolidsuffix",
    }:
        assert _has_violation(
            report,
            path="main/method.py",
            reason="version_context_identifier_not_canonical",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "context",
    ("schema_version", "protocol_id", "run_phase_id", "model_revision"),
)
@pytest.mark.parametrize(
    ("identity", "reason"),
    (
        ("A1", "ordinal_identity_config_value"),
        ("fake", "weak_semantic_config_value"),
        ("candidate_v2_router_protocol", "weak_semantic_config_value"),
        ("v2_candidate_router_protocol", "weak_semantic_config_value"),
    ),
)
def test_registered_version_field_values_are_audited_by_real_audit(
    tmp_path: Path,
    context: str,
    identity: str,
    reason: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    path = tmp_path / "configs" / "identity.json"
    path.write_text(json.dumps({context: identity}), encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="configs/identity.json",
        reason=reason,
    )
    assert any(
        violation.get("value") == identity
        and violation.get("context") == context
        for violation in report["violations"]
        if violation.get("reason") == reason
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "context",
    (
        "SchemaVersion",
        "RunPhaseId",
        "ProtocolId",
        "ModelRevision",
        "schema-version",
        "run-phase-id",
        "protocol-id",
        "xProtocolId",
        "ProtocolIdSuffix",
        "xRunPhaseId",
        "SchemaVersionSuffix",
        "x-protocol-id",
        "protocol-id-suffix",
        "xprotocolid",
        "protocolidsuffix",
        "XPROTOCOLID",
        "PROTOCOLIDSUFFIX",
        "preprotocolidpost",
        "xxrunphaseidyy",
        "schemaversionsuffix",
        "XSCHEMAVERSIONY",
        "outerprotocolidinner",
        "alphaprotocolidomega",
        "customrunphaseidbinding",
        "guardedschemaversionalias",
        "registeredmodelrevisioncopy",
        "wrappedmanifestidshadow",
        "promptidentitysuffix",
        "accessidentitysuffix",
        "operationidentitysuffix",
        "registeredkeyfamilyidsuffix",
        "internalvalidationprotocolidsuffix",
        "SCHEMA_VERSION",
        "PROTOCOL_ID",
        "RUN_PHASE_ID",
        "MODEL_REVISION",
    ),
)
def test_noncanonical_version_context_key_fails_closed_in_real_audit(
    tmp_path: Path,
    context: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    path = tmp_path / "configs" / "identity.json"
    path.write_text(
        json.dumps({context: "candidate_v2_router_protocol"}),
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="configs/identity.json",
        reason="version_context_key_not_canonical",
    )
    assert _has_violation(
        report,
        path="configs/identity.json",
        reason="weak_semantic_config_value",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "identity",
    ("candidate_2", "candidate-2", "release_7"),
)
def test_model_revision_does_not_allow_arbitrary_bare_numeric_terminal(
    tmp_path: Path,
    identity: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    path = tmp_path / "configs" / "identity.json"
    path.write_text(
        json.dumps({"model_revision": identity}),
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="configs/identity.json",
        reason="weak_semantic_config_value",
    )


@pytest.mark.unit
def test_explicit_version_contexts_preserve_revisioned_values_in_real_audit(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs", "main")
    path = tmp_path / "configs" / "versions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ceg_wm_record_schema_v4",
                "model_revision": "registered_model_v2",
                "upstream_commit": "upstream_release_v3",
                "run_phase_id": "hf_only_threshold_fit_v1",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "main" / "method.py").write_text(
        'HF_ONLY_REFERENCE_SCHEMA_VERSION = "ceg_wm_reference_schema_v1"\n'
        'EXPECTED_RUN_PHASE_ID = "hf_only_threshold_fit_v1"\n'
        'INTERNAL_VALIDATION_PROTOCOL_ID = "ceg_wm_internal_validation_v2"\n'
        'CURRENT_EXECUTION_ACCESS_IDENTITY = "internal_execution_access_v2"\n'
        'PROMPT_IDENTITY = "runtime_qualification_prompt_v1"\n'
        'SCHEMA_VERSION = "record_schema_v5"\n'
        'PROTOCOL_ID = "registered_protocol_v6"\n'
        'RUN_PHASE_ID = "threshold_fit_v7"\n'
        'MODEL_REVISION = "registered_model_v8"\n'
        'model_revision = "model_revision_1"\n',
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_unrelated_compact_identifier_substrings_are_not_version_contexts(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        'protocolidentifier = "ordinary_identifier_role"\n'
        'artifactprotocolidentifier = "ordinary_identifier_role"\n'
        'transportprotocolidentity = "ordinary_transport_identity"\n'
        'extendedtransportprotocolidentity = "ordinary_transport_identity"\n'
        'schemaversioningpolicy = "registered_serialization_policy"\n'
        'modelschemaversioningpolicy = "registered_serialization_policy"\n',
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    "context",
    (
        "protocolidentifieralias",
        "artifactprotocolidentifieralias",
        "transportprotocolidentityalias",
        "extendedtransportprotocolidentityalias",
        "schemaversioningpolicyalias",
        "modelschemaversioningpolicyalias",
        "protocolidentifiercopy",
        "transportprotocolidentityshadow",
        "schemaversioningpolicycopy",
        "modelschemaversioningpolicyshadow",
    ),
)
@pytest.mark.parametrize(
    ("root_name", "file_name", "context_reason", "value_reason"),
    (
        (
            "configs",
            "identity.json",
            "version_context_key_not_canonical",
            "weak_semantic_config_value",
        ),
        (
            "main",
            "method.py",
            "version_context_identifier_not_canonical",
            "weak_semantic_python_string",
        ),
    ),
)
def test_morphological_version_context_with_added_wrapper_fails_real_audit(
    tmp_path: Path,
    context: str,
    root_name: str,
    file_name: str,
    context_reason: str,
    value_reason: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, root_name)
    path = tmp_path / root_name / file_name
    if root_name == "configs":
        path.write_text(
            json.dumps({context: "candidate_v8_router_protocol"}),
            encoding="utf-8",
        )
    else:
        path.write_text(
            f'{context} = "candidate_v8_router_protocol"\n',
            encoding="utf-8",
        )

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path=f"{root_name}/{file_name}",
        reason=context_reason,
    )
    assert _has_violation(
        report,
        path=f"{root_name}/{file_name}",
        reason=value_reason,
    )


@pytest.mark.unit
@pytest.mark.parametrize("attribute", ("planner6Route", "planner6route"))
def test_business_attribute_compact_mechanical_suffix_is_rejected_by_real_audit(
    tmp_path: Path,
    attribute: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "class Router:\n"
        "    def configure(self):\n"
        f"        self.{attribute} = object()\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="main/method.py",
        reason="weak_semantic_identifier",
    )
    assert any(
        violation.get("identifier") == attribute
        for violation in report["violations"]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("root", "source"),
    (
        ("main", "def fake_detector():\n    pass\n"),
        ("main", "class MockDetector:\n    pass\n"),
        ("tests", "def test_fake_detector():\n    pass\n"),
        ("tests", "def test_fake_detector_behavior():\n    pass\n"),
        ("tests", "def test_mock_router_case():\n    pass\n"),
        ("tests", "def test_dummy_backend_scenario():\n    pass\n"),
        ("tests", "def test_fake_detector_uses_case():\n    pass\n"),
    ),
)
def test_weak_callable_or_test_node_identity_is_rejected_by_real_audit(
    tmp_path: Path,
    root: str,
    source: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, root)
    if root == "tests":
        destination = tmp_path / root / "unit"
        destination.mkdir()
        relative = "tests/unit/test_candidate.py"
    else:
        destination = tmp_path / root
        relative = "main/method.py"
    (tmp_path / relative).write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path=relative,
        reason="weak_semantic_identifier",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "reason"),
    (
        ("# router2 selects the route\nvalue = 1\n", "weak_semantic_comment"),
        ('def select_route():\n    """Use router2."""\n    pass\n', "weak_semantic_docstring"),
        ("# router2_gate selects the route\nvalue = 1\n", "weak_semantic_comment"),
        ('def select_route():\n    """Use router2_gate."""\n    pass\n', "weak_semantic_docstring"),
        ("# router2Gate selects the route\nvalue = 1\n", "weak_semantic_comment"),
        ('def select_route():\n    """Use router2Gate."""\n    pass\n', "weak_semantic_docstring"),
    ),
)
def test_business_code_prose_mechanical_identity_is_rejected_by_real_audit(
    tmp_path: Path,
    source: str,
    reason: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(report, path="main/method.py", reason=reason)


@pytest.mark.unit
def test_immediately_bound_local_mathematical_names_pass_real_audit(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "C_0 = 1\n"
        "module_score = C_0 + 1\n"
        "\n"
        "def combine_scores():\n"
        "    C_0 = 1\n"
        "    C_1: float = 2\n"
        "    S_0 = 3\n"
        "    return C_0 + C_1 + S_0\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_same_line_completed_mathematical_binding_precedes_read_in_real_audit(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def score_value():\n"
        "    C_0 = 1; return C_0\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_mathematical_and_tensor_expressions_bind_local_notation_in_real_audit(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "import torch\n"
        "\n"
        "def combine_scores(input_tensor, weight):\n"
        "    C_0 = 1.0 + weight * 2.0\n"
        "    C_1: float = (C_0 ** 2.0) / weight\n"
        "    S_0 = input_tensor.mean(dim=0) + torch.sqrt(C_1)\n"
        "    tensor_value = torch.tensor([C_0, C_1], dtype=torch.float32)\n"
        "    return S_0 + tensor_value.sum()\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_parameter_tensor_and_local_math_function_sources_pass_real_audit(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "import math\n"
        "import numpy as np\n"
        "import torch\n"
        "\n"
        "def square_value(value):\n"
        "    return value * value\n"
        "\n"
        "def combine_scores(input_tensor, scale):\n"
        "    C_0 = input_tensor[0].sum() * scale\n"
        "    C_1 = square_value(C_0) + torch.sqrt(scale)\n"
        "    S_0 = np.mean(input_tensor) + math.sqrt(C_1)\n"
        "    return C_0 + C_1 + S_0\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    (
        (
            "def derive(value):\n"
            "    C_0 = square_value(value)\n"
            "    return C_0\n"
            "\n"
            "def square_value(value):\n"
            "    return value * value\n"
            "\n"
            "result = derive(2.0)\n"
        ),
        (
            "def outer(value):\n"
            "    def derive():\n"
            "        S_0 = square_value(value)\n"
            "        return S_0\n"
            "\n"
            "    def square_value(value):\n"
            "        return value * value\n"
            "\n"
            "    return derive()\n"
        ),
    ),
)
def test_unique_later_mathematical_helper_binding_passes_real_audit(
    tmp_path: Path,
    source: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    (
        (
            "def outer():\n"
            "    def derive():\n"
            "        C_0 = scale * 2.0\n"
            "        return C_0\n"
            "    scale = 3.0\n"
            "    return derive()\n"
        ),
        (
            "def square_value(number):\n"
            "    return number * number\n"
            "def derive(value):\n"
            "    S_0 = square_value(value)\n"
            "    return S_0\n"
            "result = derive(2.0)\n"
        ),
        (
            "def derive():\n"
            "    C_1 = square_value(2.0)\n"
            "    return C_1\n"
            "def square_value(number):\n"
            "    return number * number\n"
            "alias = derive\n"
            "result = alias()\n"
        ),
        (
            "def square(number=2.0):\n"
            "    return number * number\n"
            "S_0 = square()\n"
            "value = S_0\n"
        ),
        (
            "def square(*, number=2.0):\n"
            "    return number * number\n"
            "C_0 = square()\n"
            "value = C_0\n"
        ),
    ),
)
def test_runtime_ordered_closure_and_argument_provenance_pass_real_audit(
    tmp_path: Path,
    source: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "notation"),
    (
        (
            "def derive():\n"
            "    C_0 = build_value()\n"
            "    return C_0\n"
            "\n"
            "def build_value():\n"
            '    return "R2"\n'
            "\n"
            "result = derive()\n",
            "C_0",
        ),
        (
            "def square_value(value):\n"
            "    return value * value\n"
            "\n"
            "def derive(value):\n"
            "    C_1 = square_value(value)\n"
            "    return C_1\n"
            "\n"
            "square_value = str\n",
            "C_1",
        ),
        (
            "def square_value(value):\n"
            "    return value * value\n"
            "\n"
            "def derive(value):\n"
            "    S_0 = square_value(value)\n"
            "    return S_0\n"
            "\n"
            "class square_value:\n"
            "    pass\n",
            "S_0",
        ),
        (
            "def square_value(value):\n"
            "    return value * value\n"
            "\n"
            "def derive(value):\n"
            "    C_0 = square_value(value)\n"
            "    return C_0\n"
            "\n"
            "def square_value(value):\n"
            "    return value + value\n",
            "C_0",
        ),
        (
            "def derive(value):\n"
            "    C_0 = square_value(value)\n"
            "    return C_0\n"
            "result = derive(2.0)\n"
            "def square_value(number):\n"
            "    return number * number\n",
            "C_0",
        ),
        (
            "def outer(value):\n"
            "    def derive():\n"
            "        C_1 = square_value(value)\n"
            "        return C_1\n"
            "    result = derive()\n"
            "    def square_value(number):\n"
            "        return number * number\n"
            "    return result\n",
            "C_1",
        ),
        (
            "async def square_value(number):\n"
            "    return number * number\n"
            "def derive(value):\n"
            "    S_0 = square_value(value)\n"
            "    return S_0\n",
            "S_0",
        ),
        (
            "def outer():\n"
            "    def derive():\n"
            "        C_0 = scale * 2.0\n"
            "        return C_0\n"
            "    result = derive()\n"
            "    scale = 3.0\n"
            "    return result\n",
            "C_0",
        ),
        (
            "def derive():\n"
            "    C_0 = square_value(2.0)\n"
            "    return C_0\n"
            "alias = derive\n"
            "result = alias()\n"
            "def square_value(number):\n"
            "    return number * number\n",
            "C_0",
        ),
        (
            "def outer():\n"
            "    def derive():\n"
            "        C_1 = square_value(2.0)\n"
            "        return C_1\n"
            "    alias = derive\n"
            "    result = alias()\n"
            "    def square_value(number):\n"
            "        return number * number\n"
            "    return result\n",
            "C_1",
        ),
        (
            "def square(number=\"R2\"):\n"
            "    return number * number\n"
            "S_0 = square()\n"
            "value = S_0\n",
            "S_0",
        ),
        (
            "def derive():\n"
            "    C_0 = square_value(2.0)\n"
            "    return C_0\n"
            "alias = derive\n"
            "alias = other_callable\n"
            "result = alias()\n"
            "def square_value(number):\n"
            "    return number * number\n",
            "C_0",
        ),
    ),
)
def test_later_nonmath_or_competing_helper_binding_fails_real_audit(
    tmp_path: Path,
    source: str,
    notation: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == notation
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    "reference_statement",
    (
        "reference = derive if flag else fallback",
        "reference = (derive, fallback)",
        "reference = [derive]",
        'reference = {"callback": derive}',
        "register_callback(derive)",
        "register_callback(callback=derive)",
        "def invoke_callback(callback=derive):\n    return callback",
    ),
)
def test_indirect_caller_reference_before_helper_definition_fails_real_audit(
    tmp_path: Path,
    reference_statement: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    C_0 = square_value(2.0)\n"
        "    return C_0\n"
        f"{reference_statement}\n"
        "def square_value(number):\n"
        "    return number * number\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
def test_nested_indirect_caller_reference_before_helper_fails_real_audit(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def outer():\n"
        "    def derive():\n"
        "        C_0 = square_value(2.0)\n"
        "        return C_0\n"
        "    callbacks = [derive]\n"
        "    def square_value(number):\n"
        "        return number * number\n"
        "    return callbacks\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
def test_indirect_caller_reference_after_helper_definition_passes_real_audit(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    S_0 = square_value(2.0)\n"
        "    return S_0\n"
        "def square_value(number):\n"
        "    return number * number\n"
        "callbacks = [derive]\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    "reference_statement",
    (
        "reference = alias if flag else fallback",
        "reference = (alias, fallback)",
        "reference = [alias]",
        'reference = {"callbacks": [[alias]]}',
        "register_callback(alias)",
        "register_callback(callback=alias)",
        "def invoke_callback(callback=alias):\n    return callback",
        "@register_callback(alias)\ndef registered():\n    return 1.0",
        "def registered(callback: alias):\n    return 1.0",
    ),
)
def test_aliased_indirect_reference_before_helper_definition_fails_real_audit(
    tmp_path: Path,
    reference_statement: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    C_0 = square_value(2.0)\n"
        "    return C_0\n"
        "alias = derive\n"
        f"{reference_statement}\n"
        "def square_value(number):\n"
        "    return number * number\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
def test_nested_aliased_reference_before_helper_definition_fails_real_audit(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def outer():\n"
        "    def derive():\n"
        "        C_1 = square_value(2.0)\n"
        "        return C_1\n"
        "    alias = derive\n"
        "    callbacks = [[alias]]\n"
        "    def square_value(number):\n"
        "        return number * number\n"
        "    return callbacks\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_1"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
def test_aliased_indirect_reference_after_helper_definition_passes_real_audit(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    S_0 = square_value(2.0)\n"
        "    return S_0\n"
        "alias = derive\n"
        "def square_value(number):\n"
        "    return number * number\n"
        "callbacks = [alias]\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    "alias_statement",
    ("return derive()", "return alias()"),
)
def test_nested_caller_executed_before_helper_definition_fails_real_audit(
    tmp_path: Path,
    alias_statement: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    C_0 = square_value(2.0)\n"
        "    return C_0\n"
        "alias = derive\n"
        "def expose():\n"
        f"    {alias_statement}\n"
        "result = expose()\n"
        "def square_value(number):\n"
        "    return number * number\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    "alias_statement",
    ("return derive()", "return alias()"),
)
def test_nested_caller_executed_after_helper_definition_passes_real_audit(
    tmp_path: Path,
    alias_statement: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    S_0 = square_value(2.0)\n"
        "    return S_0\n"
        "alias = derive\n"
        "def expose():\n"
        f"    {alias_statement}\n"
        "def square_value(number):\n"
        "    return number * number\n"
        "result = expose()\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("caller_setup", "caller_invocation"),
    (
        (
            "def expose():\n"
            "    return alias\n",
            "result = expose()()\n",
        ),
        ("callback = lambda: alias()\n", "result = callback()\n"),
        (
            "def expose(callback=lambda: alias()):\n"
            "    return callback()\n",
            "result = expose()\n",
        ),
    ),
)
def test_executed_callable_exposure_before_helper_definition_fails_real_audit(
    tmp_path: Path,
    caller_setup: str,
    caller_invocation: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    C_0 = square_value(2.0)\n"
        "    return C_0\n"
        "alias = derive\n"
        f"{caller_setup}"
        f"{caller_invocation}"
        "def square_value(number):\n"
        "    return number * number\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    ("caller_setup", "caller_invocation"),
    (
        (
            "def expose():\n"
            "    return alias\n",
            "result = expose()()\n",
        ),
        ("callback = lambda: alias()\n", "result = callback()\n"),
        (
            "def expose(callback=lambda: alias()):\n"
            "    return callback()\n",
            "result = expose()\n",
        ),
    ),
)
def test_callable_definition_without_pre_helper_execution_passes_real_audit(
    tmp_path: Path,
    caller_setup: str,
    caller_invocation: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    S_0 = square_value(2.0)\n"
        "    return S_0\n"
        "alias = derive\n"
        f"{caller_setup}"
        "def square_value(number):\n"
        "    return number * number\n"
        f"{caller_invocation}",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("caller_setup", "caller_invocation"),
    (
        (
            "def expose():\n"
            "    local_callback = alias\n"
            "    return local_callback\n",
            "result = expose()()\n",
        ),
        (
            "def fallback():\n"
            "    return 0.0\n"
            "def expose(flag):\n"
            "    if flag:\n"
            "        local_callback = alias\n"
            "    else:\n"
            "        local_callback = fallback\n"
            "    return local_callback\n",
            "result = expose(flag)()\n",
        ),
        (
            "def expose():\n"
            "    return alias\n"
            "def outer():\n"
            "    return expose\n",
            "result = outer()()()\n",
        ),
        (
            "callback = lambda: alias()\n"
            "callback_alias = callback\n",
            "result = callback_alias()\n",
        ),
        ("", "result = (lambda: (lambda: alias()))()()\n"),
    ),
)
def test_chained_callable_execution_before_helper_definition_fails_real_audit(
    tmp_path: Path,
    caller_setup: str,
    caller_invocation: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    C_0 = square_value(2.0)\n"
        "    return C_0\n"
        "alias = derive\n"
        f"{caller_setup}"
        f"{caller_invocation}"
        "def square_value(number):\n"
        "    return number * number\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    ("caller_setup", "caller_invocation"),
    (
        (
            "def expose():\n"
            "    local_callback = alias\n"
            "    return local_callback\n",
            "result = expose()()\n",
        ),
        (
            "def expose():\n"
            "    return alias\n"
            "def outer():\n"
            "    return expose\n",
            "result = outer()()()\n",
        ),
        (
            "callback = lambda: alias()\n"
            "callback_alias = callback\n",
            "result = callback_alias()\n",
        ),
        ("", "result = (lambda: (lambda: alias()))()()\n"),
    ),
)
def test_chained_callable_execution_after_helper_definition_passes_real_audit(
    tmp_path: Path,
    caller_setup: str,
    caller_invocation: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    S_0 = square_value(2.0)\n"
        "    return S_0\n"
        "alias = derive\n"
        f"{caller_setup}"
        "def square_value(number):\n"
        "    return number * number\n"
        f"{caller_invocation}",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_exhaustive_safe_conditional_callable_does_not_expose_derive_real_audit(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    C_1 = square_value(2.0)\n"
        "    return C_1\n"
        "def fallback():\n"
        "    return 0.0\n"
        "alias = derive\n"
        "def expose(flag):\n"
        "    if flag:\n"
        "        local_callback = fallback\n"
        "    else:\n"
        "        local_callback = fallback\n"
        "    return local_callback\n"
        "result = expose(flag)()\n"
        "def square_value(number):\n"
        "    return number * number\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_rebound_alias_indirect_reference_resolves_current_binding_real_audit(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    C_1 = square_value(2.0)\n"
        "    return C_1\n"
        "def fallback():\n"
        "    return 0.0\n"
        "alias = derive\n"
        "alias = fallback\n"
        "callbacks = [alias]\n"
        "def square_value(number):\n"
        "    return number * number\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_exhaustive_fallback_alias_rebinding_does_not_expose_old_target(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    C_1 = square_value(2.0)\n"
        "    return C_1\n"
        "def fallback():\n"
        "    return 0.0\n"
        "alias = derive\n"
        "if flag:\n"
        "    alias = fallback\n"
        "else:\n"
        "    alias = fallback\n"
        "callbacks = [alias]\n"
        "def square_value(number):\n"
        "    return number * number\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    "safe_control",
    (
        (
            "match flag:\n"
            "    case 0:\n"
            "        alias = fallback\n"
            "    case _:\n"
            "        alias = fallback\n"
        ),
        (
            "if False:\n"
            "    alias = derive\n"
            "else:\n"
            "    alias = fallback\n"
        ),
    ),
)
def test_exhaustive_or_unreachable_alias_branch_keeps_safe_target_real_audit(
    tmp_path: Path,
    safe_control: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    C_1 = square_value(2.0)\n"
        "    return C_1\n"
        "def fallback():\n"
        "    return 0.0\n"
        "alias = derive\n"
        f"{safe_control}"
        "callbacks = [alias]\n"
        "def square_value(number):\n"
        "    return number * number\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    "conditional_rebinding",
    (
        "if flag:\n    alias = fallback",
        (
            "alias = fallback\n"
            "if flag:\n"
            "    alias = derive\n"
            "else:\n"
            "    alias = fallback"
        ),
    ),
)
def test_nonexhaustive_or_derive_alias_branch_remains_exposed_real_audit(
    tmp_path: Path,
    conditional_rebinding: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    C_0 = square_value(2.0)\n"
        "    return C_0\n"
        "def fallback():\n"
        "    return 0.0\n"
        "alias = derive\n"
        f"{conditional_rebinding}\n"
        "callbacks = [alias]\n"
        "def square_value(number):\n"
        "    return number * number\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    "alias_setup",
    (
        "alias = derive\nforwarded_alias = alias\ncallbacks = [forwarded_alias]",
        (
            "def fallback():\n"
            "    return 0.0\n"
            "alias = derive\n"
            "forwarded_alias = alias\n"
            "alias = fallback\n"
            "callbacks = [forwarded_alias]"
        ),
        (
            "def fallback():\n"
            "    return 0.0\n"
            "alias = derive\n"
            "if flag:\n"
            "    alias = fallback\n"
            "callbacks = [alias]"
        ),
    ),
)
def test_multihop_or_conditional_alias_exposure_before_helper_fails_real_audit(
    tmp_path: Path,
    alias_setup: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    C_0 = square_value(2.0)\n"
        "    return C_0\n"
        f"{alias_setup}\n"
        "def square_value(number):\n"
        "    return number * number\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize("future_import", (True, False))
@pytest.mark.parametrize("annotation_expression", ("alias", "alias()"))
def test_annotation_exposure_respects_runtime_annotation_evaluation_real_audit(
    tmp_path: Path,
    future_import: bool,
    annotation_expression: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        ("from __future__ import annotations\n" if future_import else "")
        + "def derive():\n"
        "    S_0 = square_value(2.0)\n"
        "    return S_0\n"
        "alias = derive\n"
        f"def expose(callback: {annotation_expression}):\n"
        "    return callback\n"
        "def square_value(number):\n"
        "    return number * number\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)
    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "S_0"
    }
    if future_import:
        assert report["decision"] == "pass"
    else:
        assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
def test_numeric_keyword_argument_keeps_local_helper_mathematical(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def square(*, number):\n"
        "    return number * number\n"
        "C_1 = square(number=2.0)\n"
        "value = C_1\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_keyword_container_is_not_a_mathematical_parameter_value(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def collect_values(**values):\n"
        "    return values\n"
        "C_1 = collect_values(score=2.0)\n"
        "value = C_1\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_1"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    (
        (
            "def total_values(**values):\n"
            "    return sum(values.values())\n"
            "C_0 = total_values(left=1.0, right=2.0)\n"
            "value = C_0\n"
        ),
        (
            "def count_values(**values):\n"
            "    return len(values)\n"
            "S_0 = count_values(left=1.0)\n"
            "value = S_0\n"
        ),
    ),
)
def test_narrow_keyword_container_math_consumption_passes_real_audit(
    tmp_path: Path,
    source: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("return_expression", "notation"),
    (
        ("sum(values.keys())", "C_0"),
        ("sum(values.items())", "C_1"),
        ("values.values()", "S_0"),
        ("len(values, 1)", "C_0"),
    ),
)
def test_keyword_container_nonmathematical_consumption_fails_real_audit(
    tmp_path: Path,
    return_expression: str,
    notation: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def collect_values(**values):\n"
        f"    return {return_expression}\n"
        f"{notation} = collect_values(score=2.0)\n"
        f"value = {notation}\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == notation
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    "rebind_statement",
    (
        'values = {"identity": "R1"}',
        'if flag:\n        values = {"identity": "R1"}',
    ),
)
def test_rebound_keyword_container_cannot_supply_local_mathematics_real_audit(
    tmp_path: Path,
    rebind_statement: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def total_values(**values):\n"
        f"    {rebind_statement}\n"
        "    return sum(values.values())\n"
        "C_1 = total_values(score=2.0)\n"
        "value = C_1\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_1"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation_statement", "consumer_expression"),
    (
        ('(values := {"identity": "R1"})', "sum(values.values())"),
        ('values.update({"identity": "R1"})', "sum(values.values())"),
        ("values.clear()", "sum(values.values())"),
        ('values.update({"identity": "R1"})', "len(values)"),
    ),
)
def test_nonassign_keyword_container_mutation_invalidates_math_real_audit(
    tmp_path: Path,
    mutation_statement: str,
    consumer_expression: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def total_values(**values):\n"
        f"    {mutation_statement}\n"
        f"    return {consumer_expression}\n"
        "C_0 = total_values(score=2.0)\n"
        "value = C_0\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    "escape_statement",
    (
        (
            "mapping = values\n"
            '    mapping.update({"identity": "R1"})'
        ),
        'dict.update(values, {"identity": "R1"})',
        "mutate_mapping(values)",
        "mapping = values\n    mutate_mapping(mapping)",
        (
            "def mutate_mapping(mapping):\n"
            '        mapping.update({"identity": "R1"})\n'
            "    mutate_mapping(values)"
        ),
    ),
)
def test_escaped_keyword_container_cannot_supply_math_real_audit(
    tmp_path: Path,
    escape_statement: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def total_values(**values):\n"
        f"    {escape_statement}\n"
        "    return sum(values.values())\n"
        "C_0 = total_values(score=2.0)\n"
        "value = C_0\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
def test_annotated_tensor_parameter_supports_narrow_math_method(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive(loader: Tensor, scale):\n"
        "    C_0 = loader.sum() * scale\n"
        "    return C_0\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("scope_prefix", "scope_suffix"),
    (
        ("", ""),
        ("def derive(flag):\n", "    return C_0\n"),
    ),
)
def test_identity_branch_cannot_bind_local_mathematical_notation_real_audit(
    tmp_path: Path,
    scope_prefix: str,
    scope_suffix: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    indent = "    " if scope_prefix else ""
    path = tmp_path / "main" / "method.py"
    path.write_text(
        scope_prefix
        + f"{indent}if flag:\n"
        f'{indent}    candidate_value = "R1"\n'
        f"{indent}else:\n"
        f"{indent}    candidate_value = 2.0\n"
        f"{indent}C_0 = candidate_value\n"
        + scope_suffix,
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    ("scope_prefix", "scope_suffix", "notation"),
    (
        ("", "", "S_0"),
        ("def derive(flag):\n", "    return S_0\n", "S_0"),
    ),
)
def test_two_mathematical_branches_bind_local_notation_real_audit(
    tmp_path: Path,
    scope_prefix: str,
    scope_suffix: str,
    notation: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    indent = "    " if scope_prefix else ""
    path = tmp_path / "main" / "method.py"
    path.write_text(
        scope_prefix
        + f"{indent}if flag:\n"
        f"{indent}    candidate_value = 1.0\n"
        f"{indent}else:\n"
        f"{indent}    candidate_value = 2.0\n"
        f"{indent}{notation} = candidate_value\n"
        + scope_suffix,
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    (
        (
            "def derive(flag, items):\n"
            "    try:\n"
            "        candidate_value = 1.0\n"
            "    except Exception:\n"
            '        candidate_value = "R1"\n'
            "    C_0 = candidate_value\n"
            "    return C_0\n"
        ),
        (
            "def derive(flag, items):\n"
            "    candidate_value = 1.0\n"
            "    try:\n"
            "        pass\n"
            "    finally:\n"
            '        candidate_value = "R1"\n'
            "    C_0 = candidate_value\n"
            "    return C_0\n"
        ),
        (
            "def derive(flag, items):\n"
            "    match flag:\n"
            "        case 0:\n"
            "            candidate_value = 1.0\n"
            "        case _:\n"
            '            candidate_value = "R1"\n'
            "    C_0 = candidate_value\n"
            "    return C_0\n"
        ),
        (
            "def derive(flag, items):\n"
            "    match flag:\n"
            "        case 0:\n"
            "            candidate_value = 1.0\n"
            "    C_0 = candidate_value\n"
            "    return C_0\n"
        ),
        (
            "def derive(flag, items):\n"
            "    for item in items:\n"
            "        candidate_value = 1.0\n"
            "    C_0 = candidate_value\n"
            "    return C_0\n"
        ),
        (
            "def derive(flag, items):\n"
            "    candidate_value = 1.0\n"
            "    for item in items:\n"
            '        candidate_value = "R1"\n'
            "    C_0 = candidate_value\n"
            "    return C_0\n"
        ),
        (
            "def derive(flag, items):\n"
            "    while False:\n"
            "        candidate_value = 1.0\n"
            "    C_0 = candidate_value\n"
            "    return C_0\n"
        ),
        (
            "def derive(flag, items):\n"
            "    candidate_value = 1.0\n"
            "    while flag:\n"
            '        candidate_value = "R1"\n'
            "    C_0 = candidate_value\n"
            "    return C_0\n"
        ),
    ),
)
def test_optional_control_flow_identity_source_fails_real_audit(
    tmp_path: Path,
    source: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    (
        (
            "def derive(flag, items):\n"
            "    try:\n"
            "        candidate_value = 1.0\n"
            "    except Exception:\n"
            "        candidate_value = 2.0\n"
            "    S_0 = candidate_value\n"
            "    return S_0\n"
        ),
        (
            "def derive(flag, items):\n"
            "    try:\n"
            '        candidate_value = "R1"\n'
            "    finally:\n"
            "        candidate_value = 2.0\n"
            "    S_0 = candidate_value\n"
            "    return S_0\n"
        ),
        (
            "def derive(flag, items):\n"
            "    match flag:\n"
            "        case 0:\n"
            "            candidate_value = 1.0\n"
            "        case _:\n"
            "            candidate_value = 2.0\n"
            "    S_0 = candidate_value\n"
            "    return S_0\n"
        ),
        (
            "def derive(flag, items):\n"
            "    candidate_value = 1.0\n"
            "    for item in items:\n"
            "        candidate_value = 2.0\n"
            "    S_0 = candidate_value\n"
            "    return S_0\n"
        ),
        (
            "def derive(flag, items):\n"
            "    candidate_value = 1.0\n"
            "    while False:\n"
            '        candidate_value = "R1"\n'
            "    S_0 = candidate_value\n"
            "    return S_0\n"
        ),
        (
            "def derive(flag, items):\n"
            "    candidate_value = 1.0\n"
            "    while flag:\n"
            "        candidate_value = 2.0\n"
            "    S_0 = candidate_value\n"
            "    return S_0\n"
        ),
    ),
)
def test_inevitable_mathematical_control_flow_source_passes_real_audit(
    tmp_path: Path,
    source: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    (
        (
            "def derive(flag):\n"
            "    if flag:\n"
            "        raise RuntimeError\n"
            '        candidate_value = "R1"\n'
            "    else:\n"
            "        candidate_value = 1.0\n"
            "    S_0 = candidate_value\n"
            "    return S_0\n"
        ),
        (
            "def derive(flag):\n"
            "    if flag:\n"
            "        return 0.0\n"
            '        candidate_value = "R1"\n'
            "    else:\n"
            "        candidate_value = 1.0\n"
            "    S_0 = candidate_value\n"
            "    return S_0\n"
        ),
        (
            "def derive(flag):\n"
            "    candidate_value = 1.0\n"
            "    for item in (1,):\n"
            "        continue\n"
            '        candidate_value = "R1"\n'
            "    S_0 = candidate_value\n"
            "    return S_0\n"
        ),
        (
            "def derive(flag):\n"
            "    candidate_value = 1.0\n"
            "    for item in (1,):\n"
            "        break\n"
            '        candidate_value = "R1"\n'
            "    else:\n"
            '        candidate_value = "R1"\n'
            "    S_0 = candidate_value\n"
            "    return S_0\n"
        ),
        (
            "def derive(flag):\n"
            "    candidate_value = 1.0\n"
            "    if False:\n"
            '        candidate_value = "R1"\n'
            "    S_0 = candidate_value\n"
            "    return S_0\n"
        ),
    ),
)
def test_terminated_or_unreachable_identity_path_does_not_pollute_math(
    tmp_path: Path,
    source: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    (
        (
            "def derive():\n"
            "    raise RuntimeError\n"
            "    candidate_value = 1.0\n"
            "    C_0 = candidate_value\n"
        ),
        (
            "def derive():\n"
            "    return 0.0\n"
            "    candidate_value = 1.0\n"
            "    C_0 = candidate_value\n"
        ),
        (
            "def derive(items):\n"
            "    for item in items:\n"
            "        continue\n"
            "        candidate_value = 1.0\n"
            "        C_0 = candidate_value\n"
        ),
        (
            "def derive(items):\n"
            "    for item in items:\n"
            "        break\n"
            "        candidate_value = 1.0\n"
            "        C_0 = candidate_value\n"
        ),
    ),
)
def test_dead_mathematical_assignment_after_terminator_fails_real_audit(
    tmp_path: Path,
    source: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    (
        (
            "def choose_value(flag):\n"
            "    if flag:\n"
            "        return 1.0\n"
            "    return 2.0\n"
            "C_0 = choose_value(1)\n"
            "value = C_0\n"
        ),
        (
            "def choose_value(flag):\n"
            "    if flag:\n"
            "        return 1.0\n"
            "    else:\n"
            "        return 2.0\n"
            "S_0 = choose_value(0)\n"
            "value = S_0\n"
        ),
    ),
)
def test_complete_local_function_return_paths_bind_local_notation(
    tmp_path: Path,
    source: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "notation"),
    (
        (
            "def maybe_value(flag):\n"
            "    if flag:\n"
            "        return 1.0\n"
            "C_0 = maybe_value(1)\n"
            "value = C_0\n",
            "C_0",
        ),
        (
            "def maybe_value(flag):\n"
            "    if flag:\n"
            "        return 1.0\n"
            "S_0 = maybe_value(0)\n"
            "value = S_0\n",
            "S_0",
        ),
        (
            "def maybe_value(values):\n"
            "    for value in values:\n"
            "        return value\n"
            "    return 2.0\n"
            "C_0 = maybe_value([1.0])\n"
            "value = C_0\n",
            "C_0",
        ),
        (
            'values = ["hello"]\n'
            "torch = values\n"
            "C_0 = torch.pop()\n"
            "value = C_0\n",
            "C_0",
        ),
        (
            "def derive(torch):\n"
            "    C_1 = torch.pop()\n"
            "    return C_1\n",
            "C_1",
        ),
        ('math = "hello"\nS_0 = math.sqrt()\nvalue = S_0\n', "S_0"),
        ("sum = str\nC_0 = sum()\nvalue = C_0\n", "C_0"),
        ("float = str\nC_1 = float()\nvalue = C_1\n", "C_1"),
    ),
)
def test_incomplete_returns_or_shadowed_math_sources_do_not_bind_notation(
    tmp_path: Path,
    source: str,
    notation: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == notation
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "notation"),
    (
        (
            "import json as torch\n"
            "C_0 = torch.sqrt(1.0)\n"
            "value = C_0\n",
            "C_0",
        ),
        (
            "import json as np\n"
            "C_1 = np.mean([1.0])\n"
            "value = C_1\n",
            "C_1",
        ),
        (
            "from json import loads as sum\n"
            "S_0 = sum(1.0)\n"
            "value = S_0\n",
            "S_0",
        ),
        (
            "import torch\n"
            "def derive(scale):\n"
            "    C_0 = torch.sqrt(scale)\n"
            "    return C_0\n"
            'values = ["hello"]\n'
            "torch = values\n"
            "value = derive(1.0)\n",
            "C_0",
        ),
        (
            "import torch\n"
            "def outer(scale):\n"
            "    def derive():\n"
            "        C_1 = torch.sqrt(scale)\n"
            "        return C_1\n"
            '    torch = "hello"\n'
            "    return derive()\n",
            "C_1",
        ),
        (
            "def derive(values):\n"
            "    S_0 = sum(values)\n"
            "    return S_0\n"
            "sum = str\n",
            "S_0",
        ),
        (
            "import torch\n"
            "def derive(scale):\n"
            "    C_0 = torch.sqrt(scale)\n"
            "    return C_0\n"
            "class torch:\n"
            "    pass\n",
            "C_0",
        ),
        (
            "import torch\n"
            "def derive(scale):\n"
            "    C_1 = torch.sqrt(scale)\n"
            "    return C_1\n"
            "def torch():\n"
            "    return 1.0\n",
            "C_1",
        ),
        (
            "def derive(loader):\n"
            "    C_0 = loader.sum()\n"
            "    return C_0\n",
            "C_0",
        ),
    ),
)
def test_unproven_import_shadow_or_parameter_sources_do_not_bind_notation(
    tmp_path: Path,
    source: str,
    notation: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == notation
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "notation"),
    (
        ('alias = "B7"\nC_0 = alias\nvalue = C_0\n', "C_0"),
        (
            "class Source:\n"
            "    pass\n"
            "source = Source()\n"
            'source.alias = "route9"\n'
            "S_0 = source.alias\n"
            "value = S_0\n",
            "S_0",
        ),
        ('values = ["mock"]\nC_1 = values[0]\nvalue = C_1\n', "C_1"),
        (
            "def build_value():\n"
            '    return "R2"\n'
            "C_0 = build_value()\n"
            "value = C_0\n",
            "C_0",
        ),
    ),
)
def test_indirect_nonmathematical_sources_do_not_bind_local_notation(
    tmp_path: Path,
    source: str,
    notation: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == notation
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "notation"),
    (
        ('C_0 = "A1"; result = C_0\n', "C_0"),
        ('C_1: str = "A1"; result = C_1\n', "C_1"),
        ('S_0 = "candidate1"; result = S_0\n', "S_0"),
        ('C_0 = 1.0; C_0 = "A1"; result = C_0\n', "C_0"),
        ("C_0 = identity_label; result = C_0\n", "C_0"),
        ("S_0 = mock_backend; result = S_0\n", "S_0"),
        ("C_1: str = 1.0; result = C_1\n", "C_1"),
    ),
)
def test_nonmathematical_rhs_does_not_bind_local_notation_in_real_audit(
    tmp_path: Path,
    source: str,
    notation: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == notation
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    (
        "C_0: float\nvalue = C_0\n",
        "def score_value():\n    C_0: float\n    return C_0\n",
    ),
)
def test_annotation_without_value_does_not_bind_local_mathematical_name(
    tmp_path: Path,
    source: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    (
        "value = C_0; C_0 = 1\n",
        "C_0 = C_0 + 1\n",
        "value = (C_0 := 1)\n",
        "class Scores:\n    C_0 = 1\n    value = C_0\n",
    ),
)
def test_nonpreceding_or_nonassignment_mathematical_binding_fails_real_audit(
    tmp_path: Path,
    source: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    (
        "# C_0 = zero state\nvalue = 1\n",
        "# S_0 = synchronization origin\nvalue = 1\n",
        "# C_1(w) = weighted content score\nvalue = 1\n",
        'def describe_score():\n    """C_0 = zero state."""\n    return 1\n',
        'def describe_sync():\n    """S_0 = synchronization origin."""\n    return 1\n',
        'def describe_weighting():\n    """C_1(w) = weighted content score."""\n    return 1\n',
        '# The expression `C_1(w)` is local notation.\nvalue = 1\n',
        'def describe_expression():\n    """Use `C_1(w)` in this derivation."""\n    return 1\n',
    ),
)
def test_structurally_local_mathematical_prose_passes_real_audit(
    tmp_path: Path,
    source: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    (tmp_path / "main" / "method.py").write_text(source, encoding="utf-8")

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_existing_narrow_scientific_literals_pass_with_local_mathematics(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    (tmp_path / "main" / "method.py").write_text(
        "C_0 = 1\n"
        "relative_l2 = C_0\n"
        "F32 = relative_l2\n"
        "RGB8 = F32\n"
        "P95 = RGB8\n"
        "L4 = P95\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "reason", "identifier"),
    (
        ("class C_0:\n    pass\n", "ordinal_identity_identifier", "C_0"),
        ("def C_0():\n    pass\n", "ordinal_identity_identifier", "C_0"),
        (
            "def score_value(C_0):\n    return C_0\n",
            "ordinal_identity_identifier",
            "C_0",
        ),
        ("value = object().C_0\n", "ordinal_identity_identifier", "C_0"),
        ("value = select_score(C_0=1)\n", "ordinal_identity_identifier", "C_0"),
        ("c_0 = 1\n", "ordinal_identity_identifier", "c_0"),
        ("C_0_candidate = 1\n", "ordinal_identity_identifier", "C_0_candidate"),
        ("C_2 = 1\n", "ordinal_identity_identifier", "C_2"),
        (
            "def outer():\n"
            "    C_0 = 1\n"
            "    def inner():\n"
            "        return C_0\n"
            "    return inner()\n",
            "ordinal_identity_identifier",
            "C_0",
        ),
    ),
)
def test_nonlocal_or_identity_mathematical_names_fail_real_audit(
    tmp_path: Path,
    source: str,
    reason: str,
    identifier: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(report, path="main/method.py", reason=reason)
    assert any(
        violation.get("identifier") == identifier
        for violation in report["violations"]
        if violation["path"] == "main/method.py" and violation["reason"] == reason
    )


@pytest.mark.unit
@pytest.mark.parametrize("formal_binding", ("function_id", "label"))
def test_formal_string_mathematical_name_fails_real_audit(
    tmp_path: Path,
    formal_binding: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(f'{formal_binding} = "C_1"\n', encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="main/method.py",
        reason="ordinal_identity_python_string",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("formal_binding", "notation"),
    (("function_id", "C_1"), ("label", "C_0")),
)
def test_config_formal_identity_mathematical_name_fails_real_audit(
    tmp_path: Path,
    formal_binding: str,
    notation: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    path = tmp_path / "configs" / "identity.json"
    path.write_text(
        json.dumps({formal_binding: notation}),
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="configs/identity.json",
        reason="ordinal_identity_config_value",
    )


@pytest.mark.unit
def test_mathematical_name_path_fails_real_audit(tmp_path: Path) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "c_0.py"
    path.write_text("value = 1\n", encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="main/c_0.py",
        reason="ordinal_identity_path_component",
    )


@pytest.mark.unit
@pytest.mark.parametrize("directory_name", ("C_0", "c_0"))
def test_mathematical_name_directory_path_fails_real_audit(
    tmp_path: Path,
    directory_name: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    directory = tmp_path / "main" / directory_name
    directory.mkdir()
    (directory / "method.py").write_text("value = 1\n", encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path=f"main/{directory_name}",
        reason="ordinal_identity_path_component",
    )


@pytest.mark.unit
def test_mathematical_name_test_node_fails_real_audit(tmp_path: Path) -> None:
    _write_minimal_audit_fixture(tmp_path, "tests")
    unit_root = tmp_path / "tests" / "unit"
    unit_root.mkdir()
    path = unit_root / "test_candidate.py"
    path.write_text("def test_c_0_candidate():\n    pass\n", encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="tests/unit/test_candidate.py",
        reason="ordinal_identity_identifier",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "reason"),
    (
        ("# use C_0 candidate\nvalue = 1\n", "ordinal_identity_comment"),
        (
            'def describe_score():\n    """Use C_0 candidate."""\n    return 1\n',
            "ordinal_identity_docstring",
        ),
        ("# use c_0 candidate\nvalue = 1\n", "ordinal_identity_comment"),
        ("# C_0_candidate = label\nvalue = 1\n", "ordinal_identity_comment"),
    ),
)
def test_nondefinition_mathematical_prose_fails_real_audit(
    tmp_path: Path,
    source: str,
    reason: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(source, encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(report, path="main/method.py", reason=reason)


@pytest.mark.unit
def test_semantic_negative_test_nodes_and_local_fake_fixtures_remain_allowed(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "tests")
    unit_root = tmp_path / "tests" / "unit"
    unit_root.mkdir()
    (unit_root / "test_semantic_negative_controls.py").write_text(
        "def test_private_core_rejects_real_evidence_with_fake_factory():\n"
        "    fake_gpu = object()\n"
        "\n"
        "def test_verified_package_capability_rejects_fake_alias_and_class():\n"
        "    mock_backend = object()\n"
        "\n"
        "def test_mock_backend_initialization_preserves_frozen_identity():\n"
        "    pass\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


def _registry_path(tmp_path: Path) -> Path:
    return tmp_path / "docs" / "reference" / "field_registry.md"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "reason", "field_name"),
    (
        ("downgrade", "registered_identity_category_downgraded", "mode"),
        ("missing", "registered_identity_field_missing", "formal_mode"),
        ("duplicate", "duplicate_field_registry_row", "mode"),
    ),
)
def test_registry_identity_contract_mutation_fails_both_outer_audits(
    tmp_path: Path,
    mutation: str,
    reason: str,
    field_name: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    registry_path = _registry_path(tmp_path)
    text = registry_path.read_text(encoding="utf-8")
    target = next(
        line for line in text.splitlines() if line.startswith(f"| {field_name} |")
    )
    if mutation == "downgrade":
        replacement = target.replace("| method_identity |", "| scalar |", 1)
        text = text.replace(target, replacement, 1)
    elif mutation == "missing":
        text = text.replace(target + "\n", "", 1)
    else:
        text = text.replace(target, target + "\n" + target, 1)
    registry_path.write_text(text, encoding="utf-8")

    for report in (run_audit(tmp_path), run_field_audit(tmp_path)):
        assert report["decision"] == "fail"
        assert any(
            violation.get("path") == "docs/reference/field_registry.md"
            and violation.get("reason") == reason
            and violation.get("field_name") == field_name
            for violation in report["violations"]
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("registry_state", "reason"),
    (
        ("missing", "missing_field_registry"),
        ("empty", "empty_field_registry"),
    ),
)
def test_naming_audit_fails_closed_without_readable_registry(
    tmp_path: Path,
    registry_state: str,
    reason: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    registry_path = _registry_path(tmp_path)
    if registry_state == "missing":
        registry_path.unlink()
    else:
        registry_path.write_text("# empty registry\n", encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="docs/reference/field_registry.md",
        reason=reason,
    )


@pytest.mark.unit
def test_unreadable_registered_identity_row_fails_both_outer_audits(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    registry_path = _registry_path(tmp_path)
    text = registry_path.read_text(encoding="utf-8")
    target = next(
        line for line in text.splitlines() if line.startswith("| formal_mode |")
    )
    registry_path.write_text(
        text.replace(target, "| formal_mode | cross_boundary | method_identity |", 1),
        encoding="utf-8",
    )

    for report in (run_audit(tmp_path), run_field_audit(tmp_path)):
        assert report["decision"] == "fail"
        assert _has_violation(
            report,
            path="docs/reference/field_registry.md",
            reason="field_registry_row_unreadable",
        )


@pytest.mark.unit
def test_non_utf8_field_registry_fails_both_outer_audits(tmp_path: Path) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    _registry_path(tmp_path).write_bytes(b"\xff\xfe\x80")

    for report in (run_audit(tmp_path), run_field_audit(tmp_path)):
        assert report["decision"] == "fail"
        assert _has_violation(
            report,
            path="docs/reference/field_registry.md",
            reason="field_registry_unreadable",
        )


@pytest.mark.unit
def test_field_registry_directory_fails_both_outer_audits(tmp_path: Path) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    registry_path = _registry_path(tmp_path)
    registry_path.unlink()
    registry_path.mkdir()

    for report in (run_audit(tmp_path), run_field_audit(tmp_path)):
        assert report["decision"] == "fail"
        assert _has_violation(
            report,
            path="docs/reference/field_registry.md",
            reason="field_registry_unreadable",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("suffix", "content"),
    (
        ("json", '{"formal_mode": "A1"'),
        ("yaml", 'formal_mode: ["A1"'),
        ("toml", 'formal_mode = "A1'),
    ),
)
def test_unreadable_config_fails_closed_with_explicit_reason(
    tmp_path: Path,
    suffix: str,
    content: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    relative = f"configs/method.{suffix}"
    (tmp_path / relative).write_text(content, encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(report, path=relative, reason="config_unreadable")


@pytest.mark.unit
@pytest.mark.parametrize("suffix", ("json", "yaml", "toml"))
def test_non_utf8_config_fails_closed_with_explicit_reason(
    tmp_path: Path,
    suffix: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "configs")
    relative = f"configs/method.{suffix}"
    (tmp_path / relative).write_bytes(b"\xff\xfe\x80")

    report = run_audit(tmp_path)

    assert _has_violation(report, path=relative, reason="config_unreadable")


@pytest.mark.unit
@pytest.mark.parametrize("helper_before_call", (False, True))
def test_immediately_executed_nested_lambda_respects_helper_definition_order(
    tmp_path: Path,
    helper_before_call: bool,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    helper = "def square_value(number):\n    return number * number\n"
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    C_0 = square_value(2.0)\n"
        "    return C_0\n"
        "callback = lambda: (lambda: derive())()\n"
        + (helper if helper_before_call else "")
        + "result = callback()\n"
        + ("" if helper_before_call else helper),
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    if helper_before_call:
        assert report["decision"] == "pass"
    else:
        reasons = {
            violation["reason"]
            for violation in report["violations"]
            if violation["path"] == "main/method.py"
            and violation.get("identifier") == "C_0"
        }
        assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    ("returned_expression", "helper_before_call", "should_pass"),
    (
        ("(lambda: derive()) if flag else (lambda: fallback())", False, False),
        ("(lambda: fallback()) if flag else (lambda: fallback())", False, True),
        ("(lambda: derive()) if flag else (lambda: fallback())", True, True),
        ("(lambda: derive()) if False else (lambda: fallback())", False, True),
        ("(lambda: derive()) if True else (lambda: fallback())", False, False),
    ),
)
def test_immediately_invoked_returned_lambda_respects_possible_call_targets(
    tmp_path: Path,
    returned_expression: str,
    helper_before_call: bool,
    should_pass: bool,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    helper = "def square_value(number):\n    return number * number\n"
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    S_0 = square_value(2.0)\n"
        "    return S_0\n"
        "def fallback():\n"
        "    return 0.0\n"
        "def expose(flag):\n"
        f"    return {returned_expression}\n"
        + (helper if helper_before_call else "")
        + "result = expose(flag)()\n"
        + ("" if helper_before_call else helper),
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    if should_pass:
        assert report["decision"] == "pass"
    else:
        reasons = {
            violation["reason"]
            for violation in report["violations"]
            if violation["path"] == "main/method.py"
            and violation.get("identifier") == "S_0"
        }
        assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    ("guard", "should_pass"),
    (("False", True), ("True", False), ("dynamic_guard", False)),
)
def test_match_guard_reachability_controls_local_mathematical_binding(
    tmp_path: Path,
    guard: str,
    should_pass: bool,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive(flag, dynamic_guard):\n"
        "    match flag:\n"
        f"        case _ if {guard}:\n"
        '            candidate_value = "R1"\n'
        "        case _:\n"
        "            candidate_value = 1.0\n"
        "    C_0 = candidate_value\n"
        "    return C_0\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    if should_pass:
        assert report["decision"] == "pass"
    else:
        reasons = {
            violation["reason"]
            for violation in report["violations"]
            if violation["path"] == "main/method.py"
            and violation.get("identifier") == "C_0"
        }
        assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    ("final_assignment", "should_pass"),
    (
        ("alias = fallback", True),
        ("alias = derive", False),
        (
            "if flag:\n"
            "        alias = fallback\n"
            "    else:\n"
            "        alias = derive",
            False,
        ),
    ),
)
def test_finally_alias_overwrite_controls_pre_helper_indirect_exposure(
    tmp_path: Path,
    final_assignment: str,
    should_pass: bool,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def derive():\n"
        "    C_1 = square_value(2.0)\n"
        "    return C_1\n"
        "def fallback():\n"
        "    return 0.0\n"
        "alias = derive\n"
        "try:\n"
        "    alias = derive\n"
        "finally:\n"
        f"    {final_assignment}\n"
        "callbacks = [alias]\n"
        "def square_value(number):\n"
        "    return number * number\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    if should_pass:
        assert report["decision"] == "pass"
    else:
        reasons = {
            violation["reason"]
            for violation in report["violations"]
            if violation["path"] == "main/method.py"
            and violation.get("identifier") == "C_1"
        }
        assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
def test_exact_readonly_len_helper_can_consume_keyword_container(tmp_path: Path) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def count(mapping):\n"
        "    return len(mapping)\n"
        "def count_values(**values):\n"
        "    return count(values)\n"
        "S_0 = count_values(left=1.0)\n"
        "value = S_0\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_exact_readonly_len_helper_accepts_exact_keyword_binding(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def count(mapping):\n"
        "    return len(mapping)\n"
        "def count_values(**values):\n"
        "    return count(mapping=values)\n"
        "C_0 = count_values(left=1.0)\n"
        "value = C_0\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("helper", "call_expression"),
    (
        ("def count(mapping):\n    return len(mapping)\n", "count(container=values)"),
        (
            "def count(mapping):\n    return len(mapping)\n",
            "count(values, mapping=values)",
        ),
        (
            "def count(mapping):\n    return len(mapping)\n",
            "count(mapping=values, extra=1.0)",
        ),
        (
            "def count(mapping):\n    return len(mapping)\n",
            'count(**{"mapping": values})',
        ),
        (
            "def inspect(mapping):\n    return mapping\n",
            "inspect(mapping=values)",
        ),
    ),
)
def test_keyword_container_helper_binding_remains_exact_and_fail_closed(
    tmp_path: Path,
    helper: str,
    call_expression: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        helper
        + "def count_values(**values):\n"
        + f"    return {call_expression}\n"
        + "C_0 = count_values(left=1.0)\n"
        + "value = C_0\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)
    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    "helper_body",
    (
        "inspect_mapping(mapping)\n    return len(mapping)",
        'mapping.update({"identity": "R1"})\n    return len(mapping)',
        "return mapping",
    ),
)
def test_non_readonly_helper_cannot_consume_keyword_container(
    tmp_path: Path,
    helper_body: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def inspect(mapping):\n"
        f"    {helper_body}\n"
        "def count_values(**values):\n"
        "    return inspect(values)\n"
        "S_0 = count_values(left=1.0)\n"
        "value = S_0\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)
    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "S_0"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    "expression",
    (
        '1.0 if True else "A1"',
        '"A1" if False else 1.0',
    ),
)
def test_constant_conditional_math_ignores_unreachable_identity_branch(
    tmp_path: Path,
    expression: str,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        f"C_0 = {expression}\n"
        "value = C_0\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_dynamic_conditional_with_two_math_branches_passes_real_audit(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def choose(flag):\n"
        "    S_0 = 1.0 if flag else 2.0\n"
        "    return S_0\n",
        encoding="utf-8",
    )

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_dynamic_conditional_identity_branch_blocks_local_math_notation(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "main")
    path = tmp_path / "main" / "method.py"
    path.write_text(
        "def choose(flag):\n"
        '    C_1 = 1.0 if flag else "A1"\n'
        "    return C_1\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)
    reasons = {
        violation["reason"]
        for violation in report["violations"]
        if violation["path"] == "main/method.py"
        and violation.get("identifier") == "C_1"
    }
    assert {"weak_semantic_identifier", "ordinal_identity_identifier"} <= reasons


@pytest.mark.unit
def test_exact_attested_upstream_source_keeps_paths_checked_without_project_renames(
    tmp_path: Path,
) -> None:
    _copy_attested_upstream_source(tmp_path)

    preflight = _upstream_source_directory_preflight(tmp_path)
    assert not preflight.source_tree_absent
    assert preflight.nonreal_directory_path is None
    report = run_audit(tmp_path)

    assert report["decision"] == "pass"
    assert "runtime/_vendor" in report["checked_paths"]
    assert (
        "runtime/_vendor/transparent_background/InSPyReNet.py"
        in report["checked_paths"]
    )
    assert "runtime/_vendor/transparent_background/SOURCE.json" in report["checked_paths"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("source_repository", "https://example.invalid/source"),
        ("upstream_commit", "0" * 40),
        ("vendored_namespace", "runtime.transparent_background"),
    ),
)
def test_upstream_source_manifest_authority_drift_restores_project_naming_checks(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    vendor_root = _copy_attested_upstream_source(tmp_path)
    manifest_path = vendor_root / "SOURCE.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest[field] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="runtime/_vendor/transparent_background/InSPyReNet.py",
        reason="file_name_not_snake_case",
    )
    assert _has_violation(
        report,
        path="runtime/_vendor/transparent_background/InSPyReNet.py",
        reason="weak_semantic_identifier",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_local_path",
    (
        "transparent_background/InSPyReNet.py",
        "modules/*.py",
        "../InSPyReNet.py",
    ),
)
def test_upstream_source_manifest_rejects_prefix_glob_and_parent_paths(
    tmp_path: Path,
    invalid_local_path: str,
) -> None:
    vendor_root = _copy_attested_upstream_source(tmp_path)
    manifest_path = vendor_root / "SOURCE.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["files"][2]["local_path"] = invalid_local_path
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="runtime/_vendor/transparent_background/InSPyReNet.py",
        reason="ordinal_identity_identifier",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("local_sha256", "0" * 64),
        ("upstream_sha256", "0" * 64),
        ("transformations", []),
    ),
)
def test_upstream_source_file_attestation_drift_restores_project_naming_checks(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    vendor_root = _copy_attested_upstream_source(tmp_path)
    manifest_path = vendor_root / "SOURCE.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["files"][2][field] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="runtime/_vendor/transparent_background/InSPyReNet.py",
        reason="weak_semantic_identifier",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "drift_kind",
    ("payload", "symlink", "directory"),
)
def test_upstream_source_nonregular_or_byte_drift_restores_semantic_checks(
    tmp_path: Path,
    drift_kind: str,
) -> None:
    vendor_root = _copy_attested_upstream_source(tmp_path)
    target = vendor_root / "modules" / "context_module.py"
    if drift_kind == "payload":
        target.write_bytes(target.read_bytes() + b"\n# proxy route_1\n")
    elif drift_kind == "symlink":
        target.unlink()
        target.symlink_to("layers.py")
    else:
        target.unlink()
        target.mkdir()

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="runtime/_vendor/transparent_background/InSPyReNet.py",
        reason="weak_semantic_identifier",
    )
    assert _has_violation(
        report,
        path="runtime/_vendor/transparent_background/InSPyReNet.py",
        reason="ordinal_identity_identifier",
    )


@pytest.mark.unit
def test_adjacent_vendor_and_project_paths_remain_normally_audited(
    tmp_path: Path,
) -> None:
    vendor_root = _copy_attested_upstream_source(tmp_path)
    adjacent = vendor_root / "phase1.py"
    adjacent.write_text("# proxy route_1\nvalue = 1\n", encoding="utf-8")

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="runtime/_vendor/transparent_background/phase1.py",
        reason="ordinal_identity_path_component",
    )
    assert _has_violation(
        report,
        path="runtime/_vendor/transparent_background/phase1.py",
        reason="weak_semantic_comment",
    )


@pytest.mark.unit
def test_source_manifest_and_namespace_initializer_are_not_semantically_exempt(
    tmp_path: Path,
) -> None:
    vendor_root = _copy_attested_upstream_source(tmp_path)
    manifest_path = vendor_root / "SOURCE.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["stage_1"] = "enabled"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (vendor_root / "__init__.py").write_text(
        "def proxy_backend():\n    return None\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    assert _has_violation(
        report,
        path="runtime/_vendor/transparent_background/SOURCE.json",
        reason="ordinal_identity_config_key",
    )
    assert _has_violation(
        report,
        path="runtime/_vendor/transparent_background/__init__.py",
        reason="weak_semantic_identifier",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "symlinked_directory",
    (
        "runtime",
        "runtime/_vendor",
        "runtime/_vendor/transparent_background",
        "runtime/_vendor/transparent_background/modules",
        "runtime/_vendor/transparent_background/backbones",
    ),
)
def test_upstream_source_attestation_rejects_every_symlinked_directory_without_following(
    tmp_path: Path,
    symlinked_directory: str,
) -> None:
    _copy_attested_upstream_source(tmp_path)
    original = tmp_path / symlinked_directory
    external_root = tmp_path / "external_source_sentinel"
    external_root.mkdir()
    external_directory = external_root / "attested_tree"
    shutil.move(original.as_posix(), external_directory.as_posix())
    sentinel = external_directory / "external_sentinel.py"
    sentinel.write_text("# proxy route_1\n", encoding="utf-8")
    original.symlink_to(external_directory, target_is_directory=True)

    preflight = _upstream_source_directory_preflight(tmp_path)
    assert not preflight.source_tree_absent
    assert preflight.nonreal_directory_path == Path(symlinked_directory)
    assert _attested_upstream_source_paths(tmp_path) == frozenset()
    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    assert report["checked_paths"].count(symlinked_directory) == 1
    assert len(report["checked_paths"]) == len(set(report["checked_paths"]))
    assert sum(
        violation["path"] == symlinked_directory
        and violation["reason"] == "attested_upstream_source_directory_not_real"
        for violation in report["violations"]
    ) == 1
    serialized_report = json.dumps(report, sort_keys=True)
    assert "external_source_sentinel" not in serialized_report
    assert "external_sentinel.py" not in serialized_report
    assert "proxy route_1" not in serialized_report
    assert all(
        not checked_path.startswith(f"{symlinked_directory}/")
        for checked_path in report["checked_paths"]
    )
    if symlinked_directory in {
        "runtime/_vendor/transparent_background/modules",
        "runtime/_vendor/transparent_background/backbones",
    }:
        assert _has_violation(
            report,
            path="runtime/_vendor/transparent_background/InSPyReNet.py",
            reason="file_name_not_snake_case",
        )


@pytest.mark.unit
def test_invalid_upstream_directory_is_appended_when_scanner_does_not_yield_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_attested_upstream_source(tmp_path)
    modules = tmp_path / "runtime" / "_vendor" / "transparent_background" / "modules"
    external_modules = tmp_path / "outside_modules"
    shutil.move(modules.as_posix(), external_modules.as_posix())
    modules.symlink_to(external_modules, target_is_directory=True)
    visible = tmp_path / "runtime" / "generic_visibility.py"
    visible.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "governance.harness.audits.audit_naming_conventions.iter_governed_paths",
        lambda _root: iter((tmp_path / "runtime", visible)),
    )

    report = run_audit(tmp_path)

    failed_path = "runtime/_vendor/transparent_background/modules"
    assert report["checked_paths"].count(failed_path) == 1
    assert "runtime/generic_visibility.py" in report["checked_paths"]
    assert _has_violation(
        report,
        path=failed_path,
        reason="attested_upstream_source_directory_not_real",
    )


@pytest.mark.unit
def test_generic_governed_symlink_candidate_remains_visible_without_global_policy(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "runtime")
    target = tmp_path / "visible_reference.py"
    target.write_text("value = 1\n", encoding="utf-8")
    candidate = tmp_path / "runtime" / "visible_reference.py"
    candidate.symlink_to(target)

    report = run_audit(tmp_path)

    assert "runtime/visible_reference.py" in report["checked_paths"]


@pytest.mark.unit
def test_upstream_source_attestation_is_empty_when_vendor_tree_is_absent(
    tmp_path: Path,
) -> None:
    _write_minimal_audit_fixture(tmp_path, "runtime")

    preflight = _upstream_source_directory_preflight(tmp_path)
    assert preflight.source_tree_absent
    assert preflight.nonreal_directory_path is None
    assert _attested_upstream_source_paths(tmp_path) == frozenset()
    report = run_audit(tmp_path)
    assert report["decision"] == "pass"
    assert "runtime" in report["checked_paths"]
