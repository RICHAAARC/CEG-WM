from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from governance.tools.extract_package import extract_profile


@pytest.mark.constraint
@pytest.mark.parametrize("profile_name", ["method_core", "stage_a_execution"])
def test_profiles_are_structurally_valid_but_honestly_not_ready(profile_name: str, tmp_path: Path) -> None:
    manifest = extract_profile(Path.cwd(), tmp_path / profile_name, profile_name, dry_run=True)
    assert manifest["structurally_valid"] is True
    assert manifest["release_candidate_ready"] is False
    assert manifest["readiness_violations"]
    assert all(
        not path.startswith((".agents/", ".codex/", "governance/", "notebooks/", "outputs/"))
        for path in manifest["copied_files"]
    )


@pytest.mark.constraint
def test_method_core_extracts_and_runs_without_outer_guard(tmp_path: Path) -> None:
    output = tmp_path / "method_core"
    manifest = extract_profile(Path.cwd(), output, "method_core")
    assert manifest["structurally_valid"] is True
    assert not (output / "governance").exists()
    assert not (output / ".codex").exists()

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(output / "src")
    imported = subprocess.run(
        [sys.executable, "-c", "import cegwm; assert cegwm.__version__ == '0.1.0'"],
        cwd=output,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr

    tested = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=output,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert tested.returncode == 0, tested.stdout + tested.stderr


@pytest.mark.constraint
def test_stage_a_execution_extracts_without_outer_layers(tmp_path: Path) -> None:
    output = tmp_path / "stage_a_execution"
    manifest = extract_profile(Path.cwd(), output, "stage_a_execution")

    assert manifest["structurally_valid"] is True
    assert manifest["release_candidate_ready"] is False
    assert (output / "experiments" / "stage_a" / "README.md").is_file()
    assert (output / "configs" / "stage_a" / "README.md").is_file()
    assert not (output / "governance").exists()
    assert not (output / "notebooks").exists()


@pytest.mark.constraint
def test_sensitive_configuration_is_reported(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "src" / "cegwm").mkdir(parents=True)
    (source / "tests" / "unit").mkdir(parents=True)
    (source / "configs" / "method").mkdir(parents=True)
    (source / "templates" / "packages").mkdir(parents=True)
    (source / "src" / "cegwm" / "__init__.py").write_text("", encoding="utf-8")
    (source / "tests" / "unit" / "README.md").write_text("tests\n", encoding="utf-8")
    (source / "configs" / "method" / "unsafe.yaml").write_text("api_key: example\n", encoding="utf-8")
    (source / "templates" / "packages" / "method_core.md").write_text("# package\n", encoding="utf-8")
    (source / "pyproject.toml").write_text(
        "[build-system]\nrequires=['setuptools>=68']\nbuild-backend='setuptools.build_meta'\n"
        "[project]\nname='example'\nversion='0.0.1'\n",
        encoding="utf-8",
    )

    manifest = extract_profile(source, tmp_path / "output", "method_core", dry_run=True)

    assert {item["reason"] for item in manifest["safety_violations"]} == {"sensitive_config_key"}
    assert manifest["structurally_valid"] is False
