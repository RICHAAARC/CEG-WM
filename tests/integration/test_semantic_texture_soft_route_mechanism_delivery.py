"""Package and create-only delivery checks for soft-route validation."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

import pytest

from experiments.protocol.semantic_texture_soft_route_mechanism_validation import SELECTION_ROLE, load_manifest
from experiments.runners.semantic_texture_soft_route_mechanism_validation import (
    SoftRouteMechanismBranchScores,
    SoftRouteMechanismGeneration,
    SoftRouteMechanismStandardizedScores,
    execute_soft_route_mechanism_split,
)
from scripts.experiment_execution import build_semantic_texture_soft_route_mechanism_validation_package as builder
from scripts.experiment_execution.semantic_texture_soft_route_candidate_selection_bootstrap import _package_revision
from scripts.experiment_execution.semantic_texture_soft_route_candidate_selection_server import (
    SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError,
    finalize_soft_route_mechanism_candidate_selection_delivery,
    finalize_soft_route_mechanism_failure_delivery,
)


pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    for relative in sorted(builder.SOFT_ROUTE_MECHANISM_EXACT_SOURCE_FILES):
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    cache = repository / "scripts" / "experiment_execution" / "__pycache__"
    cache.mkdir()
    (cache / "excluded.pyc").write_bytes(b"interpreter cache")
    (repository / "scripts" / "experiment_execution" / "excluded.pyo").write_bytes(b"optimized cache")
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "SoftRouteMechanism Test")
    _git(repository, "config", "user.email", "soft_route_mechanism@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "soft_route_mechanism fixture")
    return repository, _git(repository, "rev-parse", "HEAD")


def test_soft_route_mechanism_package_is_deterministic_exact_and_gitless_replayable(tmp_path: Path) -> None:
    repository, revision = _repository(tmp_path)
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    one = builder.build_semantic_texture_soft_route_mechanism_validation_package(repository_root=repository, source_revision=revision, output=first, split="candidate_selection")
    two = builder.build_semantic_texture_soft_route_mechanism_validation_package(repository_root=repository, source_revision=revision, output=second, split="candidate_selection")
    assert first.read_bytes() == second.read_bytes()
    assert one["archive_sha256"] == two["archive_sha256"]
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert names == sorted(builder.SOFT_ROUTE_MECHANISM_EXACT_SOURCE_FILES) + [builder.EMBEDDED_MANIFEST_PATH]
        assert not any("__pycache__" in PurePosixPath(name).parts or PurePosixPath(name).suffix in {".pyc", ".pyo"} for name in names)
        archive.extractall(extracted)
    assert _package_revision(extracted) == revision
    assert not any(
        forbidden in PurePosixPath(name).parts
        for name in names
        for forbidden in (".agents", ".codex", "governance", "notebooks", "tests")
    )
    completed = subprocess.run(
        [sys.executable, "-c", "from experiments.protocol.semantic_texture_soft_route_mechanism_validation import PROTOCOL_ID; print(PROTOCOL_ID)"],
        cwd=extracted,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "semantic_texture_soft_route_mechanism_validation"


def _run_gitless_bootstrap(
    *,
    extracted: Path,
    unrelated_cwd: Path,
    execution_root: Path,
    output_root: Path,
) -> subprocess.CompletedProcess[str]:
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment["PATH"] = ""
    return subprocess.run(
        [
            sys.executable,
            str(
                extracted
                / "scripts/experiment_execution/semantic_texture_soft_route_candidate_selection_bootstrap.py"
            ),
            "--repository-root",
            str(extracted),
            "--checkpoint",
            str(unrelated_cwd / "missing-checkpoint.pth"),
            "--execution-root",
            str(execution_root),
            "--entrypoint-args",
            "--execute",
            "--run-id",
            "semantic-texture-soft-route-gitless-replay",
            "--output-root",
            str(output_root),
            "--detector-asset-bundle",
            str(unrelated_cwd / "missing-asset.json"),
        ],
        cwd=unrelated_cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_gitless_absolute_bootstrap_authenticates_before_package_local_imports(tmp_path: Path) -> None:
    repository, revision = _repository(tmp_path)
    archive_path = tmp_path / "candidate.zip"
    package = builder.build_semantic_texture_soft_route_mechanism_validation_package(
        repository_root=repository,
        source_revision=revision,
        output=archive_path,
        split="candidate_selection",
    )
    extracted, unrelated = tmp_path / "extracted", tmp_path / "unrelated"
    unrelated.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)
    embedded = (extracted / builder.EMBEDDED_MANIFEST_PATH).read_bytes()
    assert sha256(embedded).hexdigest() == package["embedded_manifest_sha256"]

    completed = _run_gitless_bootstrap(
        extracted=extracted,
        unrelated_cwd=unrelated,
        execution_root=tmp_path / "execution",
        output_root=tmp_path / "output",
    )
    assert completed.returncode == 2
    receipt = json.loads(completed.stdout)
    assert receipt["observed_repository_revision"] == revision
    assert receipt["blocked_class"] == "environment_blocked"
    assert receipt["status"] == "blocked"
    assert not any(
        path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
        for path in extracted.rglob("*")
    )


def test_gitless_bootstrap_rejects_arbitrary_extra_persistent_member(tmp_path: Path) -> None:
    repository, revision = _repository(tmp_path)
    archive_path = tmp_path / "candidate.zip"
    builder.build_semantic_texture_soft_route_mechanism_validation_package(
        repository_root=repository,
        source_revision=revision,
        output=archive_path,
        split="candidate_selection",
    )
    extracted, unrelated = tmp_path / "extracted", tmp_path / "unrelated"
    unrelated.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)
    (extracted / "unexpected-persistent-member.txt").write_text("must fail closed", encoding="utf-8")

    completed = _run_gitless_bootstrap(
        extracted=extracted,
        unrelated_cwd=unrelated,
        execution_root=tmp_path / "execution",
        output_root=tmp_path / "output",
    )
    assert completed.returncode == 2
    assert json.loads(completed.stdout) == {
        "blocked_class": "integrity_blocked",
        "failure_delivery_status": "not_created",
        "stage": "bootstrap",
        "status": "blocked",
    }
    assert not any(path.name == "__pycache__" for path in extracted.rglob("*"))


class _Operations:
    def __init__(self) -> None:
        self.count = 0

    def clean(self, entry):
        return SoftRouteMechanismGeneration(entry.source_cluster_id, "clean_unwatermarked", None, None, 0.0)

    def write(self, entry, arm):
        return SoftRouteMechanismGeneration(f"{entry.source_cluster_id}:{arm}", arm, "m" * 64, "combined_relative_l2_3_250", 0.0)

    def attack(self, entry, generation, attack):
        return generation.image

    def observe(self, image, *, wrong_key_index):
        self.count += 1
        value = float(self.count) if wrong_key_index is None else -float(self.count)
        return SoftRouteMechanismBranchScores(value, value, value, "d" * 64, "h" * 64, "l" * 64, wrong_key_index is None)

    def build_calibration(self, primary, *, partition_identity):
        return "h" * 64, "l" * 64, "a" * 64, "b" * 64

    def install_calibration(self, calibration):
        raise AssertionError("selection must fit once")

    def standardize(self, scores):
        return SoftRouteMechanismStandardizedScores(scores.hf_score, scores.lf_score, scores.max_score, scores.detector_identity)

    def close(self):
        return None


def test_soft_route_mechanism_delivery_is_create_only_complete_and_sha_last(tmp_path: Path) -> None:
    manifest = load_manifest(ROOT / "configs/experiments/semantic_texture_soft_route_candidate_selection_manifest.json", expected_role=SELECTION_ROLE)
    result = execute_soft_route_mechanism_split(manifest, _Operations())
    output = tmp_path / "delivery"
    code, receipt = finalize_soft_route_mechanism_candidate_selection_delivery(result, observed_repository_revision="1" * 40, run_id="soft_route_mechanism-delivery", output_root=output)
    assert code in {0, 2}
    assert (output / "SHA256SUMS").stat().st_mtime_ns >= max(path.stat().st_mtime_ns for path in output.iterdir() if path.name != "SHA256SUMS")
    sums = (output / "SHA256SUMS").read_text("ascii").splitlines()
    for line in sums:
        digest, name = line.split("  ", 1)
        assert sha256((output / name).read_bytes()).hexdigest() == digest
    with zipfile.ZipFile(output / receipt["archive_filename"]) as archive:
        assert archive.namelist() == [
            "semantic_texture_soft_route_selection_artifact.json",
            "semantic_texture_soft_route_candidate_selection_result.json",
        ]
    with pytest.raises(SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError):
        finalize_soft_route_mechanism_candidate_selection_delivery(result, observed_repository_revision="1" * 40, run_id="soft_route_mechanism-delivery", output_root=output)


def test_soft_route_mechanism_bounded_failure_is_exported_before_nonzero(tmp_path: Path) -> None:
    output = tmp_path / "failure"
    code, receipt = finalize_soft_route_mechanism_failure_delivery(observed_repository_revision="2" * 40, run_id="soft_route_mechanism-failure", output_root=output, stage="entrypoint", failure_reason="RuntimeError")
    assert code == 2
    persisted = "\n".join(path.read_text("utf-8", errors="ignore") for path in output.iterdir() if path.suffix != ".zip")
    assert "Traceback" not in persisted
    assert str(tmp_path) not in persisted
    assert (output / "SHA256SUMS").exists()
    assert receipt["status"] == "blocked"
