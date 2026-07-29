"""验证治理 policy 与受治理根目录行为。"""

import json
from pathlib import Path

import pytest

from governance.harness.audits.audit_dependency_boundaries import run_audit as run_dependency_audit
from governance.harness.audits.audit_project_root_registry import run_audit
from governance.harness.audits.audit_skill_file_presence import run_audit as run_skill_audit
from governance.harness.lib.dependency_rules import dependency_violation_reason, load_dependency_policy
from governance.harness.lib.project_policy import governed_roots, load_root_policy, load_skill_policy


@pytest.mark.constraint
def test_project_roots_are_explicitly_registered() -> None:
    report = run_audit(Path.cwd())
    assert report["decision"] == "pass"


@pytest.mark.unit
def test_local_environment_is_not_a_governed_scan_root() -> None:
    policy = load_root_policy(Path.cwd())
    assert policy["root_registry"][".conda"]["audited"] is False
    assert policy["root_registry"][".venv"]["audited"] is False
    assert ".conda" not in governed_roots(Path.cwd())
    assert ".venv" not in governed_roots(Path.cwd())


@pytest.mark.unit
def test_method_and_attack_layers_are_orthogonal() -> None:
    policy = load_dependency_policy(Path.cwd())
    reason = dependency_violation_reason("experiments.methods", "experiments.attacks.crop", policy)
    assert reason == "project_layer_dependency_forbidden"


@pytest.mark.unit
def test_content_and_geometry_chains_are_orthogonal() -> None:
    policy = load_dependency_policy(Path.cwd())
    reason = dependency_violation_reason(
        "main.content_chain",
        "main.geometry_chain.registration",
        policy,
    )
    assert reason == "project_layer_dependency_forbidden"


@pytest.mark.unit
def test_joint_decision_can_consume_both_chains() -> None:
    policy = load_dependency_policy(Path.cwd())
    assert (
        dependency_violation_reason(
            "main.joint_decision",
            "main.content_chain.detector",
            policy,
        )
        is None
    )
    assert (
        dependency_violation_reason(
            "main.joint_decision",
            "main.geometry_chain.recovery",
            policy,
        )
        is None
    )


@pytest.mark.unit
def test_runtime_cannot_bypass_main_public_surface() -> None:
    policy = load_dependency_policy(Path.cwd())
    reason = dependency_violation_reason(
        "runtime",
        "main.geometry_chain.qk_sync",
        policy,
    )
    assert reason == "project_layer_dependency_forbidden"


@pytest.mark.unit
def test_project_layers_cannot_import_governance() -> None:
    policy = load_dependency_policy(Path.cwd())
    reason = dependency_violation_reason("runtime", "governance.harness", policy)
    assert reason == "control_plane_import_forbidden"


@pytest.mark.constraint
def test_delivery_code_cannot_import_governance(tmp_path: Path) -> None:
    policy_root = tmp_path / "governance" / "policies"
    policy_root.mkdir(parents=True)
    (policy_root / "dependency_rules.yaml").write_text(
        json.dumps(
            {
                "layers": {},
                "forbidden_dependency": "governance",
                "delivery_code_roots": ["scripts"],
                "record_writer_layers": [],
            }
        ),
        encoding="utf-8",
    )
    scripts_root = tmp_path / "scripts"
    scripts_root.mkdir()
    (scripts_root / "entrypoint.py").write_text("from governance.harness import run_all_audits\n", encoding="utf-8")
    report = run_dependency_audit(tmp_path)
    assert report["decision"] == "fail"
    assert report["violations"][0]["reason"] == "control_plane_import_forbidden"


@pytest.mark.constraint
def test_only_registered_runner_layer_can_write_experiment_records(
    tmp_path: Path,
) -> None:
    policy_root = tmp_path / "governance" / "policies"
    policy_root.mkdir(parents=True)
    (policy_root / "dependency_rules.yaml").write_text(
        json.dumps(
            {
                "layers": {
                    "experiments.methods": {
                        "allowed_project_dependencies": [],
                    },
                    "experiments.runners": {
                        "allowed_project_dependencies": [
                            "experiments.methods",
                        ],
                    },
                },
                "forbidden_dependency": "governance",
                "delivery_code_roots": [],
                "record_writer_layers": ["experiments.runners"],
            }
        ),
        encoding="utf-8",
    )
    methods_root = tmp_path / "experiments" / "methods"
    methods_root.mkdir(parents=True)
    (methods_root / "bad_writer.py").write_text(
        "from pathlib import Path\n"
        "def persist(path):\n"
        "    Path(path).write_text('record', encoding='utf-8')\n",
        encoding="utf-8",
    )
    runners_root = tmp_path / "experiments" / "runners"
    runners_root.mkdir(parents=True)
    (runners_root / "writer.py").write_text(
        "from pathlib import Path\n"
        "def persist(path):\n"
        "    Path(path).write_text('record', encoding='utf-8')\n",
        encoding="utf-8",
    )

    report = run_dependency_audit(tmp_path)

    assert report["decision"] == "fail"
    assert [
        violation["path"]
        for violation in report["violations"]
        if violation["reason"] == "record_write_outside_authorized_layer"
    ] == ["experiments/methods/bad_writer.py"]


@pytest.mark.unit
def test_project_skill_registry_matches_skill_directories() -> None:
    root = Path.cwd()
    policy = load_skill_policy(root)
    skill_root = root / policy["skill_root"]
    actual = sorted(path.name for path in skill_root.iterdir() if path.is_dir() and not path.name.startswith("."))
    assert actual == sorted(policy["required_skills"])


@pytest.mark.constraint
def test_project_skills_satisfy_registered_contract() -> None:
    report = run_skill_audit(Path.cwd())
    assert report["decision"] == "pass"


@pytest.mark.unit
def test_unregistered_project_skill_is_rejected(tmp_path: Path) -> None:
    policy_root = tmp_path / "governance" / "policies"
    policy_root.mkdir(parents=True)
    (policy_root / "project_skills.yaml").write_text(
        json.dumps(
            {
                "skill_root": ".agents/skills",
                "required_skills": [],
                "required_files": ["SKILL.md", "agents/openai.yaml"],
                "required_sections": ["## Workflow", "## Blocking Rules", "## Required Validation"],
                "unknown_skill": "fail",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / ".agents" / "skills" / "unregistered-skill").mkdir(parents=True)

    report = run_skill_audit(tmp_path)

    assert report["decision"] == "fail"
    assert any(violation["reason"] == "unregistered_project_skill" for violation in report["violations"])
