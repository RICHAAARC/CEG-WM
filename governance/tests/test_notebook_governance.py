"""验证 Notebook 位置和提交状态治理。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from governance.harness.audits.audit_notebook_boundaries import run_audit


@pytest.mark.constraint
def test_repository_notebooks_satisfy_boundary_policy() -> None:
    report = run_audit(Path.cwd())
    assert report["decision"] == "pass"


@pytest.mark.unit
def test_committed_notebook_outputs_are_rejected(tmp_path: Path) -> None:
    policy_root = tmp_path / "governance" / "policies"
    policy_root.mkdir(parents=True)
    (policy_root / "project_roots.yaml").write_text(
        json.dumps({"root_registry": {"notebooks": {"audited": True}}}),
        encoding="utf-8",
    )
    (policy_root / "notebook_rules.yaml").write_text(
        json.dumps(
            {
                "notebook_root": "notebooks",
                "allowed_notebook_roots": ["notebooks"],
                "committed_outputs": "fail",
                "committed_execution_counts": "fail",
                "max_notebook_bytes": 5000000,
            }
        ),
        encoding="utf-8",
    )
    notebook_root = tmp_path / "notebooks"
    notebook_root.mkdir()
    (notebook_root / "executed.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 1,
                        "outputs": [{"output_type": "stream", "name": "stdout", "text": ["result\n"]}],
                        "source": ["print('result')"],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )

    report = run_audit(tmp_path)

    reasons = {violation["reason"] for violation in report["violations"]}
    assert report["decision"] == "fail"
    assert reasons == {"committed_notebook_output", "committed_notebook_execution_count"}


@pytest.mark.unit
def test_notebook_code_cannot_import_outer_control_plane(tmp_path: Path) -> None:
    policy_root = tmp_path / "governance" / "policies"
    policy_root.mkdir(parents=True)
    (policy_root / "project_roots.yaml").write_text(
        json.dumps({"root_registry": {"notebooks": {"audited": True}}}), encoding="utf-8"
    )
    (policy_root / "notebook_rules.yaml").write_text(
        json.dumps(
            {
                "notebook_root": "notebooks",
                "allowed_notebook_roots": ["notebooks"],
                "committed_outputs": "fail",
                "committed_execution_counts": "fail",
                "max_notebook_bytes": 5000000,
            }
        ),
        encoding="utf-8",
    )
    notebook_root = tmp_path / "notebooks"
    notebook_root.mkdir()
    (notebook_root / "entrypoint.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "outputs": [],
                        "source": ["import pathlib, governance.harness\n"],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    assert any(item["reason"] == "control_plane_import_forbidden" for item in report["violations"])
