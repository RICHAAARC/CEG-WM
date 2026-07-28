"""验证正式代码、注释和配置键的弱语义审计。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from governance.harness.audits.audit_naming_conventions import run_audit
from governance.harness.lib.naming_rules import is_allowed_file_name


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
    assert "weak_semantic_comment" in reasons
    assert "weak_semantic_config_key" in reasons
