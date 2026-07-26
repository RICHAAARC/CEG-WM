"""审计基础目录边界是否存在。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.json_report import build_report, exit_with_report

REQUIRED_PATHS = [
    "README.md",
    "requirements_cpu.txt",
    ".codex/project_contract.md",
    ".codex/research_state/README.md",
    ".codex/research_state/research_definition.yaml",
    ".agents/skills",
    "governance/README.md",
    "governance/policies/README.md",
    "governance/harness/README.md",
    "governance/tools/extract_release_package.py",
    "governance/docs/extraction_manifest_contract.md",
    "docs/README.md",
    "docs/design/research_definition.md",
    "docs/design/method_architecture.md",
    "docs/design/content_chain.md",
    "docs/design/geometry_chain.md",
    "docs/design/joint_decision.md",
    "docs/design/evaluation_design.md",
    "docs/reference/test_inventory.md",
    "governance/docs/file_organization.md",
    "governance/harness/run_all_audits.py",
    "governance/harness/audits/audit_research_definition.py",
    "governance/contracts/architecture.md",
    "governance/policies/project_roots.yaml",
    "governance/policies/dependency_rules.yaml",
    "governance/policies/project_skills.yaml",
    "governance/policies/notebook_rules.yaml",
    "governance/policies/method_readiness_rules.yaml",
    "governance/templates/research_definition.yaml",
    "governance/templates/method_readiness.yaml",
    "governance/templates/README.md",
    "templates/comparison_protocol.yaml",
    "notebooks/colab",
    "experiments/methods/baselines",
    "configs/baselines",
    "configs/experiments",
    "tests/README.md",
    "governance/pytest.ini",
]
FORBIDDEN_CHECKED_IN_DIRS = ["outputs"]


def run_audit(root: str | Path) -> dict:
    root_path = Path(root)
    violations = []
    checked_paths = []
    for relative in REQUIRED_PATHS:
        checked_paths.append(relative)
        if not (root_path / relative).exists():
            violations.append({"path": relative, "reason": "required_path_missing"})
    for relative in FORBIDDEN_CHECKED_IN_DIRS:
        checked_paths.append(relative)
        if (root_path / relative).exists():
            violations.append({"path": relative, "reason": "checked_in_runtime_output_root_forbidden"})
    return build_report("audit_file_organization_contract", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
