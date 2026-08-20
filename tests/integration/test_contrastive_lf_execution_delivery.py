from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/experiment_execution/build_contrastive_lf_branch_attribution_package.py"


@pytest.mark.integration
def test_exact_package_is_deterministic_and_gitless_authenticatable(tmp_path: Path) -> None:
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert not subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout, "exact package integration requires the final clean committed checkout"
    archives = (
        tmp_path / "first" / "candidate.zip",
        tmp_path / "second" / "candidate.zip",
    )
    manifests = []
    for archive in archives:
        archive.parent.mkdir()
        subprocess.run(
            (sys.executable, str(BUILDER), "--repository-root", str(ROOT), "--source-revision", revision, "--output", str(archive)),
            check=True,
        )
        manifests.append(json.loads(archive.with_suffix(".zip.manifest.json").read_text()))
    assert archives[0].read_bytes() == archives[1].read_bytes()
    assert manifests[0] == manifests[1]
    with ZipFile(archives[0]) as source:
        names = source.namelist()
        assert not any(part in name.split("/") for name in names for part in (".agents", ".codex", "governance", "tests", "notebooks", "__pycache__"))
        source.extractall(tmp_path / "extracted")
    embedded = json.loads((tmp_path / "extracted/contrastive_lf_branch_attribution_package_manifest.json").read_text())
    assert embedded["package_ready"] is True
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    bootstrap = tmp_path / "extracted/scripts/experiment_execution/contrastive_lf_branch_attribution_bootstrap.py"
    completed = subprocess.run(
        (
            sys.executable,
            str(bootstrap),
            "--expected-revision", revision,
            "--expected-package-identity", embedded["package_identity"],
            "--expected-embedded-manifest-sha256", sha256((tmp_path / "extracted/contrastive_lf_branch_attribution_package_manifest.json").read_bytes()).hexdigest(),
            "--authenticate-only",
        ),
        cwd=unrelated,
        env={"PATH": "/nonexistent"},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    imported = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "from scripts.experiment_execution."
                "contrastive_lf_branch_attribution_entrypoint import main; "
                "assert callable(main)"
            ),
        ),
        cwd=unrelated,
        env={"PATH": "/nonexistent", "PYTHONPATH": str(tmp_path / "extracted")},
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
