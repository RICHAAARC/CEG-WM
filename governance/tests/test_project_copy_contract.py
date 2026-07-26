"""验证复制 CEG-WM 治理项目时外层护栏仍可独立工作。"""

from pathlib import Path
import shutil

import pytest

from governance.harness.lib.project_policy import load_root_policy
from governance.harness.run_all_audits import run_all_audits


@pytest.mark.constraint
def test_governed_project_copy_preserves_project_authorities(tmp_path: Path) -> None:
    source_root = Path.cwd()
    copied_root = tmp_path / "copied_project"
    copied_root.mkdir()
    policy = load_root_policy(source_root)
    excluded_kinds = {
        "agent_state",
        "local_environment",
        "version_control",
        "generated_cache",
        "editor_state",
        "generated_asset",
        "external_dependency",
    }
    for root_name, metadata in policy["root_registry"].items():
        source = source_root / root_name
        if not source.exists() or metadata["kind"] in excluded_kinds:
            continue
        shutil.copytree(source, copied_root / root_name)
    for file_name in policy["governed_files"]:
        shutil.copy2(source_root / file_name, copied_root / file_name)
    assert (copied_root / "README.md").exists()
    assert (copied_root / ".codex" / "research_state" / "research_definition.yaml").exists()
    assert run_all_audits(copied_root)["overall_decision"] == "pass"
