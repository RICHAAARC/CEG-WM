"""验证复制 CEG-WM 治理项目时外层护栏仍可独立工作。"""

from pathlib import Path
import shutil

import pytest

from governance.harness.lib.project_policy import load_root_policy
from governance.harness.run_all_audits import run_all_audits


@pytest.mark.constraint
def test_governed_project_copy_without_revision_fails_revision_bound_provenance(
    tmp_path: Path,
) -> None:
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
    report = run_all_audits(copied_root)
    assert report["overall_decision"] == "fail"
    failures = [
        audit_result
        for audit_result in report["audit_results"]
        if audit_result["decision"] != "pass"
    ]
    assert [
        {
            "audit_name": failure["audit_name"],
            "violations": failure["violations"],
        }
        for failure in failures
    ] == [
        {
            "audit_name": "audit_research_definition",
            "violations": [
                {
                    "path": ".git",
                    "reason": "construction_repository_revision_unavailable",
                }
            ],
        },
        {
            "audit_name": "audit_method_readiness",
            "violations": [
                {
                    "path": ".codex/research_state/method_readiness.yaml",
                    "reason": "method_independent_review_revision_unverifiable",
                },
                {
                    "path": ".codex/research_state/salient_local_lf_candidate_readiness.yaml",
                    "reason": "salient_local_lf_review_revision_unverifiable",
                }
            ],
        },
    ]
