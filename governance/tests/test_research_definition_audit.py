"""验证 CEG-WM 研究定义阶段门禁。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from governance.harness.audits.audit_research_definition import run_audit


REQUIRED_ROLES = (
    "research_definition",
    "method_architecture",
    "content_chain",
    "geometry_chain",
    "joint_decision",
    "evaluation_design",
    "candidate_specification",
    "algorithm_primitives",
    "method_mechanism",
    "research_construction_roadmap",
)
REQUIRED_INVARIANTS = (
    "content_evidence_primary",
    "geometry_no_direct_positive",
)


def _write_authority(root: Path, stage: str) -> None:
    policy_root = root / "governance" / "policies"
    policy_root.mkdir(parents=True)
    (policy_root / "method_readiness_rules.yaml").write_text(
        json.dumps(
            {
                "project_name": "ceg_wm",
                "stage_order": [
                    "project_constraint_framework",
                    "research_defined",
                    "method_construction_authorized",
                    "method_implemented",
                ],
                "research_definition_stages": [
                    "research_defined",
                    "method_construction_authorized",
                    "method_implemented",
                ],
                "research_definition_manifest_path": (
                    ".codex/research_state/research_definition.yaml"
                ),
                "implementation_authorized_stages": [
                    "method_construction_authorized",
                    "method_implemented",
                ],
                "construction_authorization_stage": (
                    "method_construction_authorized"
                ),
                "construction_admission_manifest_path": (
                    ".codex/research_state/"
                    "method_construction_admission.yaml"
                ),
                "required_research_design_roles": list(REQUIRED_ROLES),
                "required_method_invariants": list(REQUIRED_INVARIANTS),
                "implementation_forbidden_stages": [
                    "project_constraint_framework",
                    "research_defined",
                ],
                "design_root": "docs/design",
                "implementation_root": "main",
            }
        ),
        encoding="utf-8",
    )
    contract_root = root / ".codex"
    contract_root.mkdir()
    (contract_root / "project_contract.md").write_text(
        f"- `project_stage`: `{stage}`\n",
        encoding="utf-8",
    )


def _write_valid_definition(root: Path) -> None:
    design_root = root / "docs" / "design"
    design_root.mkdir(parents=True)
    design_paths = {}
    for role in REQUIRED_ROLES:
        relative = f"docs/design/{role}.md"
        design_paths[role] = relative
        (root / relative).write_text(
            f"# {role}\n\n## Scope\n\n" + ("substantive research boundary " * 12),
            encoding="utf-8",
        )

    manifest_path = root / ".codex" / "research_state" / "research_definition.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "project_name": "ceg_wm",
                "design_paths": design_paths,
                "method_invariants": list(REQUIRED_INVARIANTS),
                "implementation_status": "not_implemented",
            }
        ),
        encoding="utf-8",
    )


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _initialize_clean_construction_transition(root: Path) -> None:
    _write_authority(root, "research_defined")
    _write_valid_definition(root)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Governance Test")
    _git(root, "config", "user.email", "governance@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "candidate specification closed")
    base_revision = _git(root, "rev-parse", "HEAD")

    contract_path = root / ".codex" / "project_contract.md"
    contract_path.write_text(
        "- `project_stage`: `method_construction_authorized`\n",
        encoding="utf-8",
    )
    admission_path = (
        root
        / ".codex"
        / "research_state"
        / "method_construction_admission.yaml"
    )
    admission_path.write_text(
        json.dumps(
            {
                "candidate_specification_status": "closed",
                "independent_review_decision": "approve",
                "user_authorization_reference": "user-authorization-001",
                "authorization_base_revision": base_revision,
            }
        ),
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "authorize method construction")


@pytest.mark.unit
def test_framework_stage_does_not_require_research_definition(tmp_path: Path) -> None:
    _write_authority(tmp_path, "project_constraint_framework")
    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_research_stage_rejects_missing_manifest(tmp_path: Path) -> None:
    _write_authority(tmp_path, "research_defined")
    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    assert report["violations"][0]["reason"] == "research_definition_manifest_unreadable"


@pytest.mark.unit
def test_research_stage_accepts_substantive_design_and_invariants(tmp_path: Path) -> None:
    _write_authority(tmp_path, "research_defined")
    _write_valid_definition(tmp_path)
    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_registered_design_rejects_explicit_stale_current_status(tmp_path: Path) -> None:
    _write_authority(tmp_path, "research_defined")
    _write_valid_definition(tmp_path)
    design_path = tmp_path / "docs" / "design" / "method_architecture.md"
    design_path.write_text(
        design_path.read_text(encoding="utf-8")
        + "\n\n## Current Status\n\n`runtime_verified / implemented`\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    assert any(
        violation["reason"] == "registered_design_current_status_mismatch"
        for violation in report["violations"]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "current_claim",
    (
        "## Current Implementation Status\n\n`runtime_verified / implemented`\n",
        "实际 stage/status 已同步为 `runtime_verified / implemented`。\n",
        "实际阶段/status 已同步为 `runtime_verified / implemented`。\n",
        "当前项目登记为\n`runtime_verified / implemented`。\n",
        "当前检查点：实际 stage/status 为 `runtime_verified / implemented`。\n",
    ),
)
def test_explicit_current_claim_formats_fail_closed(
    tmp_path: Path,
    current_claim: str,
) -> None:
    _write_authority(tmp_path, "research_defined")
    _write_valid_definition(tmp_path)
    design_path = tmp_path / "docs" / "design" / "method_architecture.md"
    design_path.write_text(
        design_path.read_text(encoding="utf-8") + "\n\n" + current_claim,
        encoding="utf-8",
    )
    report = run_audit(tmp_path)
    assert any(
        violation["reason"] == "registered_design_current_status_mismatch"
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_historical_runtime_stage_text_is_not_current_status(tmp_path: Path) -> None:
    _write_authority(tmp_path, "research_defined")
    _write_valid_definition(tmp_path)
    design_path = tmp_path / "docs" / "design" / "method_architecture.md"
    design_path.write_text(
        design_path.read_text(encoding="utf-8")
        + "\n\n## Historical Route\n\nThe earlier route mentioned runtime_verified.\n",
        encoding="utf-8",
    )
    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_all_ten_registered_designs_enter_checked_paths(tmp_path: Path) -> None:
    _write_authority(tmp_path, "research_defined")
    _write_valid_definition(tmp_path)
    report = run_audit(tmp_path)
    expected = {f"docs/design/{role}.md" for role in REQUIRED_ROLES}
    assert expected <= set(report["checked_paths"])


@pytest.mark.unit
def test_research_stage_rejects_missing_invariant(tmp_path: Path) -> None:
    _write_authority(tmp_path, "research_defined")
    _write_valid_definition(tmp_path)
    manifest_path = tmp_path / ".codex" / "research_state" / "research_definition.yaml"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["method_invariants"].remove("geometry_no_direct_positive")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    assert any(
        violation["reason"] == "research_method_invariant_missing"
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_research_stage_rejects_substantive_method_implementation(tmp_path: Path) -> None:
    _write_authority(tmp_path, "research_defined")
    _write_valid_definition(tmp_path)
    implementation_path = tmp_path / "main" / "content_detector.py"
    implementation_path.parent.mkdir()
    implementation_path.write_text(
        "def detect(score, threshold):\n    return score >= threshold\n",
        encoding="utf-8",
    )

    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    assert any(
        violation["reason"] == "research_stage_method_implementation_forbidden"
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_construction_authorized_stage_permits_substantive_method_work(
    tmp_path: Path,
) -> None:
    _initialize_clean_construction_transition(tmp_path)
    implementation_path = tmp_path / "main" / "content_detector.py"
    implementation_path.parent.mkdir()
    implementation_path.write_text(
        "def detect(score, threshold):\n    return score >= threshold\n",
        encoding="utf-8",
    )
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "implement content detector")

    assert run_audit(tmp_path)["decision"] == "pass"


@pytest.mark.unit
def test_construction_stage_rejects_missing_auditable_revision(
    tmp_path: Path,
) -> None:
    _write_authority(tmp_path, "method_construction_authorized")
    _write_valid_definition(tmp_path)

    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    assert any(
        violation["reason"] == "construction_admission_manifest_unreadable"
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_stage_transition_cannot_include_method_implementation(
    tmp_path: Path,
) -> None:
    _write_authority(tmp_path, "research_defined")
    _write_valid_definition(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Governance Test")
    _git(tmp_path, "config", "user.email", "governance@example.invalid")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "candidate specification closed")
    base_revision = _git(tmp_path, "rev-parse", "HEAD")

    contract_path = tmp_path / ".codex" / "project_contract.md"
    contract_path.write_text(
        "- `project_stage`: `method_construction_authorized`\n",
        encoding="utf-8",
    )
    admission_path = (
        tmp_path
        / ".codex"
        / "research_state"
        / "method_construction_admission.yaml"
    )
    admission_path.write_text(
        json.dumps(
            {
                "candidate_specification_status": "closed",
                "independent_review_decision": "approve",
                "user_authorization_reference": "user-authorization-001",
                "authorization_base_revision": base_revision,
            }
        ),
        encoding="utf-8",
    )
    implementation_path = tmp_path / "main" / "content_detector.py"
    implementation_path.parent.mkdir()
    implementation_path.write_text(
        "def detect(score, threshold):\n    return score >= threshold\n",
        encoding="utf-8",
    )
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "invalid combined transition")

    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    assert any(
        violation["reason"]
        == "construction_stage_transition_contains_method_change"
        for violation in report["violations"]
    )


@pytest.mark.unit
def test_method_implemented_cannot_skip_construction_stage(
    tmp_path: Path,
) -> None:
    _write_authority(tmp_path, "research_defined")
    _write_valid_definition(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Governance Test")
    _git(tmp_path, "config", "user.email", "governance@example.invalid")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "candidate specification closed")
    base_revision = _git(tmp_path, "rev-parse", "HEAD")

    contract_path = tmp_path / ".codex" / "project_contract.md"
    contract_path.write_text(
        "- `project_stage`: `method_implemented`\n",
        encoding="utf-8",
    )
    admission_path = (
        tmp_path
        / ".codex"
        / "research_state"
        / "method_construction_admission.yaml"
    )
    admission_path.write_text(
        json.dumps(
            {
                "candidate_specification_status": "closed",
                "independent_review_decision": "approve",
                "user_authorization_reference": "user-authorization-001",
                "authorization_base_revision": base_revision,
            }
        ),
        encoding="utf-8",
    )
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "invalid direct completion")

    report = run_audit(tmp_path)
    assert report["decision"] == "fail"
    assert any(
        violation["reason"]
        == "construction_authorization_stage_transition_missing"
        for violation in report["violations"]
    )
