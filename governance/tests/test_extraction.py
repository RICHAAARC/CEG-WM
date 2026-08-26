from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from governance.tools.extract_package import extract_profile


@pytest.mark.constraint
@pytest.mark.parametrize("profile_name", ["content_chain_execution"])
def test_profiles_are_structurally_valid_and_ready(profile_name: str, tmp_path: Path) -> None:
    manifest = extract_profile(Path.cwd(), tmp_path / profile_name, profile_name, dry_run=True)
    assert manifest["structurally_valid"] is True
    assert manifest["release_candidate_ready"] is True
    assert manifest["readiness_violations"] == []
    assert all(
        not path.startswith((".agents/", ".codex/", "governance/", "notebooks/", "outputs/"))
        for path in manifest["copied_files"]
    )
    assert not any(".egg-info/" in path for path in manifest["copied_files"])


@pytest.mark.constraint
def test_content_chain_execution_extracts_without_outer_layers(tmp_path: Path) -> None:
    output = tmp_path / "content_chain_execution"
    manifest = extract_profile(Path.cwd(), output, "content_chain_execution")

    assert manifest["structurally_valid"] is True
    assert manifest["release_candidate_ready"] is True
    assert (output / "experiments" / "run_content_chain.py").is_file()
    assert (output / "configs" / "content_chain" / "content_chain_stability.json").is_file()
    assert not (output / "governance").exists()
    assert not (output / "notebooks").exists()

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(output / "src")
    tested = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
            "tests/unit", "-o", "addopts=",
        ],
        cwd=output,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert tested.returncode == 0, tested.stdout + tested.stderr


@pytest.mark.constraint
def test_sensitive_configuration_is_reported(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "src" / "cegwm").mkdir(parents=True)
    (source / "tests" / "unit").mkdir(parents=True)
    (source / "tests" / "integration").mkdir(parents=True)
    (source / "experiments").mkdir(parents=True)
    (source / "configs" / "content_chain").mkdir(parents=True)
    (source / "templates" / "packages").mkdir(parents=True)
    (source / "src" / "cegwm" / "__init__.py").write_text("", encoding="utf-8")
    (source / "tests" / "unit" / "README.md").write_text("tests\n", encoding="utf-8")
    (source / "tests" / "integration" / "README.md").write_text("tests\n", encoding="utf-8")
    (source / "experiments" / "README.md").write_text("runners\n", encoding="utf-8")
    (source / "configs" / "content_chain" / "unsafe.yaml").write_text(
        "api_key: example\n", encoding="utf-8"
    )
    (source / "templates" / "packages" / "content_chain_execution.md").write_text(
        "# package\n", encoding="utf-8"
    )
    (source / "pyproject.toml").write_text(
        "[build-system]\nrequires=['setuptools>=68']\nbuild-backend='setuptools.build_meta'\n"
        "[project]\nname='example'\nversion='0.0.1'\n",
        encoding="utf-8",
    )

    manifest = extract_profile(
        source, tmp_path / "output", "content_chain_execution", dry_run=True
    )

    assert {item["reason"] for item in manifest["safety_violations"]} == {"sensitive_config_key"}
    assert manifest["structurally_valid"] is False
