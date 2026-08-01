"""审计 CEG-WM 研究定义、方法不变量和未实现阶段边界。"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.json_report import build_report, exit_with_report
from governance.harness.lib.project_policy import load_json_compatible_yaml


PROJECT_STAGE_PATTERN = re.compile(r"`project_stage`\s*:\s*`(?P<stage>[a-z][a-z0-9_]*)`")
REQUIRED_MANIFEST_FIELDS = (
    "project_name",
    "design_paths",
    "method_invariants",
    "implementation_status",
)


def _is_within(relative: Path, root: Path) -> bool:
    return relative == root or root in relative.parents


def _has_substantive_design(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    headings = [line for line in text.splitlines() if line.startswith("#")]
    return len(text.strip()) >= 200 and len(headings) >= 2 and "[TODO" not in text


def _has_substantive_python(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return True
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return True
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            return True
    return False


def _git(
    root_path: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root_path), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _contract_stage_at_revision(
    root_path: Path,
    revision: str,
) -> str | None:
    result = _git(
        root_path,
        "show",
        f"{revision}:.codex/project_contract.md",
    )
    if result.returncode != 0:
        return None
    match = PROJECT_STAGE_PATTERN.search(result.stdout)
    return match.group("stage") if match else None


def _append_construction_admission_violations(
    root_path: Path,
    policy: dict,
    project_stage: str,
    checked_paths: list[str],
    violations: list[dict],
) -> None:
    if project_stage not in set(policy["implementation_authorized_stages"]):
        return

    manifest_path = root_path / policy["construction_admission_manifest_path"]
    manifest_relative = str(manifest_path.relative_to(root_path))
    checked_paths.append(manifest_relative)
    try:
        manifest = load_json_compatible_yaml(manifest_path)
    except (OSError, ValueError, UnicodeError) as error:
        violations.append(
            {
                "path": manifest_relative,
                "reason": "construction_admission_manifest_unreadable",
                "detail": str(error),
            }
        )
        return

    expected_values = {
        "candidate_specification_status": "closed",
        "independent_review_decision": "approve",
    }
    for field, expected in expected_values.items():
        if manifest.get(field) != expected:
            violations.append(
                {
                    "path": manifest_relative,
                    "reason": "construction_admission_decision_invalid",
                    "field": field,
                    "expected": expected,
                }
            )
    authorization_reference = manifest.get("user_authorization_reference")
    if (
        not isinstance(authorization_reference, str)
        or len(authorization_reference.strip()) < 8
        or "TODO" in authorization_reference.upper()
    ):
        violations.append(
            {
                "path": manifest_relative,
                "reason": "construction_user_authorization_reference_invalid",
            }
        )
    base_revision = manifest.get("authorization_base_revision")
    if (
        not isinstance(base_revision, str)
        or re.fullmatch(r"[0-9a-f]{40,64}", base_revision) is None
    ):
        violations.append(
            {
                "path": manifest_relative,
                "reason": "construction_authorization_base_revision_invalid",
            }
        )
        return

    repository_root = _git(root_path, "rev-parse", "--show-toplevel")
    if (
        repository_root.returncode != 0
        or Path(repository_root.stdout.strip()).resolve() != root_path.resolve()
    ):
        violations.append(
            {
                "path": ".git",
                "reason": "construction_repository_revision_unavailable",
            }
        )
        return
    head = _git(root_path, "rev-parse", "--verify", "HEAD^{commit}")
    base = _git(
        root_path,
        "rev-parse",
        "--verify",
        f"{base_revision}^{{commit}}",
    )
    if head.returncode != 0 or base.returncode != 0:
        violations.append(
            {
                "path": manifest_relative,
                "reason": "construction_repository_revision_unresolvable",
            }
        )
        return
    canonical_base = base.stdout.strip()
    if canonical_base != base_revision:
        violations.append(
            {
                "path": manifest_relative,
                "reason": "construction_authorization_base_revision_not_canonical",
            }
        )
    ancestry = _git(
        root_path,
        "merge-base",
        "--is-ancestor",
        canonical_base,
        head.stdout.strip(),
    )
    if ancestry.returncode != 0:
        violations.append(
            {
                "path": manifest_relative,
                "reason": "construction_authorization_base_not_ancestor",
            }
        )
        return

    history = _git(
        root_path,
        "rev-list",
        "--reverse",
        f"{canonical_base}..{head.stdout.strip()}",
    )
    revisions = history.stdout.splitlines() if history.returncode == 0 else []
    transition_revision = next(
        (
            revision
            for revision in revisions
            if _contract_stage_at_revision(root_path, revision)
            == policy["construction_authorization_stage"]
        ),
        None,
    )
    if transition_revision is None:
        violations.append(
            {
                "path": manifest_relative,
                "reason": "construction_authorization_stage_transition_missing",
            }
        )
        return

    parent = _git(root_path, "rev-parse", f"{transition_revision}^")
    parent_stage = (
        _contract_stage_at_revision(root_path, parent.stdout.strip())
        if parent.returncode == 0
        else None
    )
    if parent_stage != "research_defined":
        violations.append(
            {
                "path": manifest_relative,
                "reason": "construction_authorization_stage_transition_not_direct",
                "parent_stage": parent_stage,
            }
        )
    changed = _git(
        root_path,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        transition_revision,
    )
    changed_main_paths = sorted(
        path
        for path in changed.stdout.splitlines()
        if path == policy["implementation_root"]
        or path.startswith(f"{policy['implementation_root']}/")
    )
    if changed.returncode != 0 or changed_main_paths:
        violations.append(
            {
                "path": manifest_relative,
                "reason": "construction_stage_transition_contains_method_change",
                "paths": changed_main_paths,
            }
        )


def _append_forbidden_implementation_violations(
    root_path: Path,
    policy: dict,
    project_stage: str,
    checked_paths: list[str],
    violations: list[dict],
) -> None:
    if project_stage not in set(policy["implementation_forbidden_stages"]):
        return
    implementation_root = root_path / policy["implementation_root"]
    for path in implementation_root.rglob("*.py"):
        relative = path.relative_to(root_path)
        checked_paths.append(relative.as_posix())
        if path.name != "__init__.py" and _has_substantive_python(path):
            violations.append(
                {
                    "path": relative.as_posix(),
                    "reason": "research_stage_method_implementation_forbidden",
                }
            )


CURRENT_STATUS_HEADING_PATTERN = re.compile(
    r"(?mi)^##\s+(?:Current(?:\s+[A-Za-z]+)*\s+Status|当前检查点)\s*$"
)
CURRENT_CLAIM_PATTERN = re.compile(
    r"实际\s*(?:stage|阶段)/status|当前项目登记为|(?m:^当前检查点：)"
)
STATUS_PAIR_PATTERN = re.compile(
    r"`(?P<stage>[a-z][a-z0-9_]*)\s*/\s*"
    r"(?P<status>[a-z][a-z0-9_]*)`"
)


def _explicit_current_status_segments(text: str) -> tuple[str, ...]:
    starts = [match.end() for match in CURRENT_STATUS_HEADING_PATTERN.finditer(text)]
    starts.extend(match.start() for match in CURRENT_CLAIM_PATTERN.finditer(text))
    segments: list[str] = []
    for start in sorted(set(starts)):
        following_heading = re.search(r"(?m)^##\s+", text[start:])
        end = start + following_heading.start() if following_heading else len(text)
        segments.append(text[start:end])
    return tuple(segments)


def _append_design_current_status_violations(
    path: Path,
    relative: Path,
    project_stage: str,
    implementation_status: str,
    violations: list[dict],
) -> None:
    text = path.read_text(encoding="utf-8")
    for segment in _explicit_current_status_segments(text):
        pairs = tuple(STATUS_PAIR_PATTERN.finditer(segment))
        if not pairs or any(
            match.group("stage") != project_stage
            or match.group("status") != implementation_status
            for match in pairs
        ):
            violations.append(
                {
                    "path": relative.as_posix(),
                    "reason": "registered_design_current_status_mismatch",
                    "expected_project_stage": project_stage,
                    "expected_implementation_status": implementation_status,
                }
            )
            return


def run_audit(root: str | Path) -> dict:
    root_path = Path(root)
    policy_path = root_path / "governance" / "policies" / "method_readiness_rules.yaml"
    contract_path = root_path / ".codex" / "project_contract.md"
    checked_paths = [
        str(policy_path.relative_to(root_path)),
        str(contract_path.relative_to(root_path)),
    ]
    violations = []

    try:
        policy = load_json_compatible_yaml(policy_path)
        contract_text = contract_path.read_text(encoding="utf-8")
    except (OSError, ValueError, UnicodeError) as error:
        violations.append(
            {
                "path": str(policy_path.relative_to(root_path)),
                "reason": "research_definition_authority_unreadable",
                "detail": str(error),
            }
        )
        return build_report("audit_research_definition", "fail", violations, checked_paths)

    stage_match = PROJECT_STAGE_PATTERN.search(contract_text)
    if not stage_match:
        violations.append(
            {
                "path": str(contract_path.relative_to(root_path)),
                "reason": "project_stage_missing",
            }
        )
        return build_report("audit_research_definition", "fail", violations, checked_paths)

    project_stage = stage_match.group("stage")
    if project_stage not in set(policy["stage_order"]):
        violations.append(
            {
                "path": str(contract_path.relative_to(root_path)),
                "reason": "project_stage_not_registered",
                "stage": project_stage,
            }
        )
        return build_report("audit_research_definition", "fail", violations, checked_paths)
    _append_forbidden_implementation_violations(
        root_path,
        policy,
        project_stage,
        checked_paths,
        violations,
    )
    _append_construction_admission_violations(
        root_path,
        policy,
        project_stage,
        checked_paths,
        violations,
    )
    if project_stage not in set(policy["research_definition_stages"]):
        return build_report(
            "audit_research_definition",
            "fail" if violations else "pass",
            violations,
            checked_paths,
        )

    manifest_path = root_path / policy["research_definition_manifest_path"]
    checked_paths.append(str(manifest_path.relative_to(root_path)))
    try:
        manifest = load_json_compatible_yaml(manifest_path)
    except (OSError, ValueError, UnicodeError) as error:
        violations.append(
            {
                "path": str(manifest_path.relative_to(root_path)),
                "reason": "research_definition_manifest_unreadable",
                "detail": str(error),
            }
        )
        return build_report("audit_research_definition", "fail", violations, checked_paths)

    missing_fields = [field for field in REQUIRED_MANIFEST_FIELDS if field not in manifest]
    if missing_fields:
        violations.append(
            {
                "path": str(manifest_path.relative_to(root_path)),
                "reason": "research_definition_field_missing",
                "fields": missing_fields,
            }
        )
        return build_report("audit_research_definition", "fail", violations, checked_paths)

    if manifest["project_name"] != policy["project_name"]:
        violations.append(
            {
                "path": str(manifest_path.relative_to(root_path)),
                "reason": "research_definition_project_name_mismatch",
            }
        )

    design_paths = manifest["design_paths"] if isinstance(manifest["design_paths"], dict) else {}
    required_roles = set(policy["required_research_design_roles"])
    missing_roles = sorted(required_roles - set(design_paths))
    if missing_roles:
        violations.append(
            {
                "path": str(manifest_path.relative_to(root_path)),
                "reason": "research_design_role_missing",
                "roles": missing_roles,
            }
        )
    design_root = Path(policy["design_root"])
    for role in sorted(required_roles & set(design_paths)):
        relative = Path(str(design_paths[role]))
        path = root_path / relative
        checked_paths.append(relative.as_posix())
        if (
            not _is_within(relative, design_root)
            or path.name == "README.md"
            or not path.is_file()
        ):
            violations.append(
                {
                    "path": relative.as_posix(),
                    "reason": "research_design_path_invalid",
                    "role": role,
                }
            )
        elif not _has_substantive_design(path):
            violations.append(
                {
                    "path": relative.as_posix(),
                    "reason": "research_design_not_substantive",
                    "role": role,
                }
            )
        else:
            _append_design_current_status_violations(
                path,
                relative,
                project_stage,
                str(manifest["implementation_status"]),
                violations,
            )

    invariants = (
        set(str(value) for value in manifest["method_invariants"])
        if isinstance(manifest["method_invariants"], list)
        else set()
    )
    missing_invariants = sorted(set(policy["required_method_invariants"]) - invariants)
    if missing_invariants:
        violations.append(
            {
                "path": str(manifest_path.relative_to(root_path)),
                "reason": "research_method_invariant_missing",
                "invariants": missing_invariants,
            }
        )

    if (
        project_stage in set(policy["implementation_forbidden_stages"])
        and manifest["implementation_status"] != "not_implemented"
    ):
        violations.append(
            {
                "path": str(manifest_path.relative_to(root_path)),
                "reason": "research_stage_implementation_status_invalid",
            }
        )

    return build_report(
        "audit_research_definition",
        "fail" if violations else "pass",
        violations,
        checked_paths,
    )


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
