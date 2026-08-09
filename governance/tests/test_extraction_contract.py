"""验证交付候选包与外层治理保持隔离。"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from governance.harness.audits.audit_dependency_boundaries import run_audit as run_dependency_boundary_audit
from governance.harness.audits.audit_release_extraction_contract import run_audit as run_release_extraction_audit
from governance.tools.extract_release_package import extract_profile


@pytest.mark.constraint
def test_dependency_boundaries_pass_for_template() -> None:
    assert run_dependency_boundary_audit(Path.cwd())["decision"] == "pass"


@pytest.mark.constraint
def test_release_extraction_contract_pass_for_template() -> None:
    assert run_release_extraction_audit(Path.cwd())["decision"] == "pass"


@pytest.mark.constraint
def test_minimal_method_package_excludes_outer_layers(tmp_path: Path) -> None:
    manifest = extract_profile(Path.cwd(), tmp_path / "minimal", "minimal_method_package", dry_run=True)
    copied_files = manifest["copied_files"]
    assert copied_files
    assert all(not path.startswith((".agents/", ".codex/", "governance/", "experiments/", "notebooks/")) for path in copied_files)
    assert all(not path.startswith(("third_party/", "configs/baselines/", "configs/experiments/")) for path in copied_files)
    assert manifest["release_candidate_ready"] is False


@pytest.mark.constraint
def test_extracted_package_uses_package_local_readme(tmp_path: Path) -> None:
    output = tmp_path / "minimal"
    manifest = extract_profile(Path.cwd(), output, "minimal_method_package")
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert manifest["safety_violations"] == []
    assert readme.startswith("# Minimal Method Package Candidate")
    assert "](docs/" not in readme and "](.codex/" not in readme


@pytest.mark.constraint
def test_documented_temporary_path_does_not_block_artifact_package(tmp_path: Path) -> None:
    manifest = extract_profile(Path.cwd(), tmp_path / "paper", "paper_artifact_rebuild_package", dry_run=True)
    assert manifest["safety_violations"] == []


@pytest.mark.constraint
def test_extraction_rejects_sensitive_config_and_absolute_local_path(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "main").mkdir(parents=True)
    (root / "configs" / "methods").mkdir(parents=True)
    (root / "templates" / "release_readmes").mkdir(parents=True)
    (root / "main" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (root / "templates" / "release_readmes" / "minimal_method_package.md").write_text(
        "# Minimal Method Package Candidate\n", encoding="utf-8"
    )
    (root / "configs" / "methods" / "unsafe.yaml").write_text(
        "api_key: example\ncache_dir: /home/example/cache\n", encoding="utf-8"
    )
    manifest = extract_profile(root, tmp_path / "dry", "minimal_method_package", dry_run=True)
    assert {item["reason"] for item in manifest["safety_violations"]} == {
        "sensitive_config_key",
        "absolute_local_path",
    }
    with pytest.raises(ValueError, match="候选包安全检查失败"):
        extract_profile(root, tmp_path / "actual", "minimal_method_package")


@pytest.mark.constraint
def test_third_party_requires_explicit_flag_and_registered_baseline(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "third_party").mkdir(parents=True)
    (root / "third_party" / "upstream.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "templates" / "release_readmes").mkdir(parents=True)
    (root / "templates" / "release_readmes" / "experiment_execution_package.md").write_text(
        "# Experiment Execution Package Candidate\n", encoding="utf-8"
    )
    default = extract_profile(root, tmp_path / "default", "experiment_execution_package", dry_run=True)
    assert all(not path.startswith("third_party/") for path in default["copied_files"])
    explicit = extract_profile(
        root,
        tmp_path / "explicit",
        "experiment_execution_package",
        dry_run=True,
        include_third_party=True,
    )
    assert "third_party/upstream.py" in explicit["copied_files"]
    assert {item["reason"] for item in explicit["safety_violations"]} == {"baseline_provenance_missing"}


@pytest.mark.constraint
def test_artifact_package_keeps_research_evidence_docs_only(tmp_path: Path) -> None:
    manifest = extract_profile(Path.cwd(), tmp_path / "paper", "paper_artifact_rebuild_package", dry_run=True)
    copied_files = manifest["copied_files"]
    assert "docs/reference/artifact_evidence.md" in copied_files
    assert "docs/reference/extraction_profiles.md" not in copied_files
    assert "governance/tools/extract_release_package.py" not in copied_files
    assert "scripts/extract_release_package.py" not in copied_files
    assert all(not path.startswith("scripts/") or path.startswith("scripts/artifact_rebuild/") for path in copied_files)
    assert all(not path.startswith(("governance/", "docs/governance/")) for path in copied_files)
    assert sorted(
        path
        for path in copied_files
        if path.startswith("tests/")
        and Path(path).name.startswith("test_")
        and path.endswith(".py")
    ) == ["tests/functional/test_governed_artifact_structures.py"]
    assert "tests/functional/test_lf_null_whitened_detector.py" not in copied_files


@pytest.mark.constraint
def test_experiment_package_contains_execution_layers_not_outer_workflow(tmp_path: Path) -> None:
    manifest = extract_profile(Path.cwd(), tmp_path / "execution", "experiment_execution_package", dry_run=True)
    copied_files = manifest["copied_files"]
    assert not manifest["missing_paths"]
    for prefix in ("main/", "runtime/", "experiments/", "infrastructure/"):
        assert any(path.startswith(prefix) for path in copied_files)
    assert all(not path.startswith(("governance/", ".codex/", ".agents/", "notebooks/", "paper_artifacts/")) for path in copied_files)
    assert "governance/tools/extract_release_package.py" not in copied_files
    assert "scripts/extract_release_package.py" not in copied_files
    assert all(not path.startswith("scripts/") or path.startswith("scripts/experiment_execution/") for path in copied_files)
    assert "root_path" not in manifest and "output_path" not in manifest


@pytest.mark.constraint
def test_actual_packages_exclude_outer_tool_and_validate_applicable_contents(tmp_path: Path) -> None:
    root = Path.cwd()
    package_paths = {}
    for profile_name in (
        "minimal_method_package",
        "experiment_execution_package",
        "paper_artifact_rebuild_package",
    ):
        output = tmp_path / profile_name
        manifest = extract_profile(root, output, profile_name)
        package_paths[profile_name] = output
        assert "governance/tools/extract_release_package.py" not in manifest["copied_files"]
        assert "scripts/extract_release_package.py" not in manifest["copied_files"]
        assert not (output / "governance").exists()
        assert not (output / "scripts" / "extract_release_package.py").exists()

    commands = {
        "minimal_method_package": "import main",
        "experiment_execution_package": "import main, runtime, experiments.protocol",
        "paper_artifact_rebuild_package": "import experiments.protocol, paper_artifacts",
    }
    for profile_name, command in commands.items():
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=package_paths[profile_name],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    paper_test = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "-p",
            "no:cacheprovider",
        ],
        cwd=package_paths["paper_artifact_rebuild_package"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert paper_test.returncode == 0, paper_test.stdout + paper_test.stderr
