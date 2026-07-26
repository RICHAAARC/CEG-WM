"""验证外层 harness 的基础契约。"""

from pathlib import Path

import pytest

from governance.harness.run_all_audits import run_all_audits


@pytest.mark.constraint
def test_project_contract_exists() -> None:
    assert Path(".codex/project_contract.md").exists()


@pytest.mark.constraint
def test_harness_audits_pass_for_template() -> None:
    assert run_all_audits(Path.cwd())["overall_decision"] == "pass"


@pytest.mark.constraint
def test_main_core_package_exists() -> None:
    assert Path("main/__init__.py").exists()
    assert not Path("src").exists()
