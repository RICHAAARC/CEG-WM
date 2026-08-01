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
    literals = "relative_l2 F32 RGB8 P95 x86_64 L4 SHA-256 SHA256"
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
