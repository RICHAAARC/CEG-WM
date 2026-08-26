from __future__ import annotations

from pathlib import Path

import pytest

from governance.harness.audits.dependencies import run_audit as run_dependency_audit
from governance.harness.audits.notebooks import run_audit as run_notebook_audit
from governance.harness.run_all import run_all


@pytest.mark.constraint
def test_dependency_boundaries_pass() -> None:
    assert run_dependency_audit(Path.cwd())["decision"] == "pass"


@pytest.mark.constraint
def test_notebook_boundaries_pass() -> None:
    assert run_notebook_audit(Path.cwd())["decision"] == "pass"


@pytest.mark.constraint
def test_minimal_harness_contains_only_registered_checks() -> None:
    report = run_all(Path.cwd())
    assert report["overall_decision"] == "pass"
    assert [item["audit_name"] for item in report["audits"]] == ["dependencies", "notebooks"]
