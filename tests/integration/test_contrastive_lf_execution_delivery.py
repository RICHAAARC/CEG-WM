from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest

from scripts.experiment_execution.build_contrastive_lf_branch_attribution_package import (
    ContrastiveLfPackageError,
    _parse_roster_exclusion_bindings,
)


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
    roster_path = tmp_path / (
        "extracted/configs/experiments/"
        "contrastive_lf_branch_attribution_prompt_roster.json"
    )
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    binding_paths = [
        binding["relative_path"]
        for binding in roster["exclusion_source_bindings"]
    ]
    assert len(binding_paths) == len(set(binding_paths)) == 30
    copied_paths = {entry["path"] for entry in embedded["copied_files"]}
    assert set(binding_paths) <= copied_paths
    assert set(binding_paths) <= set(names)

    invalid_roster = dict(roster)
    invalid_roster["exclusion_source_bindings"] = list(
        roster["exclusion_source_bindings"]
    )
    invalid_roster["exclusion_source_bindings"][1] = dict(
        invalid_roster["exclusion_source_bindings"][0]
    )
    with pytest.raises(
        ContrastiveLfPackageError,
        match="prompt roster exclusion path is duplicated",
    ):
        _parse_roster_exclusion_bindings(
            json.dumps(invalid_roster).encode("utf-8")
        )
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
    loaded = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                "from experiments.protocol.contrastive_lf_branch_attribution "
                "import load_manifest; "
                "root=Path(sys.argv[1]); "
                "null_fit=load_manifest(root/'configs/experiments/"
                "contrastive_lf_null_fit_manifest.json', "
                "expected_role='contrastive_lf_null_fit'); "
                "selection=load_manifest(root/'configs/experiments/"
                "contrastive_lf_candidate_selection_manifest.json', "
                "expected_role='contrastive_lf_candidate_selection'); "
                "assert null_fit.role_id == 'contrastive_lf_null_fit'; "
                "assert selection.role_id == 'contrastive_lf_candidate_selection'"
            ),
            str(tmp_path / "extracted"),
        ),
        cwd=unrelated,
        env={
            "PATH": "/nonexistent",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(tmp_path / "extracted"),
        },
        capture_output=True,
        text=True,
    )
    assert loaded.returncode == 0, loaded.stderr
    runs_root = tmp_path / "runs"
    smoke_run_id = "contrastive-lf-branch-attribution-" + "a" * 32
    smoke_session_id = "stage-a-session-" + "b" * 32
    launched = subprocess.run(
        (
            sys.executable,
            str(bootstrap),
            "--expected-revision",
            revision,
            "--expected-package-identity",
            embedded["package_identity"],
            "--expected-embedded-manifest-sha256",
            sha256(
                (
                    tmp_path
                    / "extracted/contrastive_lf_branch_attribution_package_manifest.json"
                ).read_bytes()
            ).hexdigest(),
            "--new-run-id",
            smoke_run_id,
            "--session-id",
            smoke_session_id,
            "--runs-root",
            str(runs_root),
            "--package-sha256",
            sha256(archives[0].read_bytes()).hexdigest(),
        ),
        cwd=unrelated,
        env={"PATH": "/nonexistent", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )
    assert launched.returncode == 2, launched.stderr
    final_root = (
        runs_root
        / smoke_run_id
        / "final"
    )
    receipt = json.loads(
        (final_root / "contrastive_lf_execution_receipt.json").read_text()
    )
    result = json.loads((final_root / receipt["result_filename"]).read_text())
    assert result["result_classification"] == "operational_failure"
    assert result["science_started"] is False
    assert result["scientific_unit_count"] == 0
    assert (
        runs_root
        / smoke_run_id
        / "sessions"
        / f"{smoke_session_id}.json"
    ).is_file()
