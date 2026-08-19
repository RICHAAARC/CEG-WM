"""Package and create-only delivery checks for soft-route validation."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

import pytest

from experiments.protocol.semantic_texture_soft_route_mechanism_validation import (
    CONFIRMATION_MANIFEST_DIGEST,
    CONFIRMATION_ROLE,
    PROTOCOL_ID,
    SELECTION_ROLE,
    load_manifest,
)
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
from scripts.experiment_execution.semantic_texture_soft_route_untouched_confirmation_server import (
    CONFIRMATION_ARTIFACT_FILENAME,
    RECEIPT_FILENAME as CONFIRMATION_RECEIPT_FILENAME,
    RESULT_FILENAME as CONFIRMATION_RESULT_FILENAME,
    SemanticTextureSoftRouteMechanismConfirmationDeliveryError,
    finalize_soft_route_mechanism_untouched_confirmation_delivery,
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


def test_gitless_absolute_confirmation_bootstrap_propagates_bounded_failure(tmp_path: Path) -> None:
    repository, revision = _repository(tmp_path)
    archive_path = tmp_path / "confirmation.zip"
    builder.build_semantic_texture_soft_route_mechanism_validation_package(
        repository_root=repository,
        source_revision=revision,
        output=archive_path,
        split="untouched_confirmation",
    )
    extracted, unrelated = tmp_path / "extracted", tmp_path / "unrelated"
    unrelated.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    environment["PATH"] = ""
    output = tmp_path / "confirmation-output"
    completed = subprocess.run(
        [
            sys.executable,
            str(
                extracted
                / "scripts/experiment_execution/semantic_texture_soft_route_untouched_confirmation_bootstrap.py"
            ),
            "--repository-root",
            str(extracted),
            "--checkpoint",
            str(unrelated / "missing-checkpoint.pth"),
            "--execution-root",
            str(tmp_path / "confirmation-execution"),
            "--entrypoint-args",
            "--execute",
            "--run-id",
            "semantic-texture-soft-route-confirmation-replay",
            "--output-root",
            str(output),
            "--selection-artifact-root",
            str(unrelated),
            "--selection-artifact-sha256",
            "a" * 64,
            "--detector-asset-bundle",
            str(unrelated / "missing-asset.json"),
        ],
        cwd=unrelated,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert completed.stderr == ""
    receipt = json.loads(completed.stdout)
    assert receipt["blocked_class"] == "environment_blocked"
    assert receipt["untouched_confirmation_passed"] is False
    assert receipt["provisional_authority_retired"] is False
    assert receipt["observed_repository_revision"] == revision
    assert (output / CONFIRMATION_RESULT_FILENAME).exists()
    assert (output / CONFIRMATION_RECEIPT_FILENAME).exists()
    assert (output / "SHA256SUMS").exists()
    persisted = "\n".join(
        path.read_text("utf-8", errors="ignore")
        for path in output.iterdir()
        if path.suffix != ".zip"
    )
    assert "candidate_selection_passed" not in persisted
    assert "selection_artifact_filename" not in persisted
    assert not any(
        path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
        for path in extracted.rglob("*")
    )


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


class _ConfirmationOperations(_Operations):
    def __init__(self) -> None:
        super().__init__()
        self.installed = None

    def build_calibration(self, primary, *, partition_identity):
        raise AssertionError("confirmation must not refit")

    def install_calibration(self, calibration):
        self.installed = calibration


class _FirstGenerationFailureOperations(_Operations):
    def clean(self, entry):
        raise RuntimeError("private details must remain bounded")


def _selection_manifest():
    return load_manifest(
        ROOT
        / "configs/experiments/semantic_texture_soft_route_candidate_selection_manifest.json",
        expected_role=SELECTION_ROLE,
    )


def _first_generation_failure_result():
    return execute_soft_route_mechanism_split(
        _selection_manifest(), _FirstGenerationFailureOperations()
    )


def test_soft_route_mechanism_delivery_is_create_only_complete_and_sha_last(tmp_path: Path) -> None:
    result = execute_soft_route_mechanism_split(_selection_manifest(), _Operations())
    result = replace(result, passed=True)
    output = tmp_path / "delivery"
    code, receipt = finalize_soft_route_mechanism_candidate_selection_delivery(result, observed_repository_revision="1" * 40, run_id="soft_route_mechanism-delivery", output_root=output)
    assert code == 0
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


def test_runner_failure_delivery_preserves_fixed_denominator_without_selection_authority(
    tmp_path: Path,
) -> None:
    result = _first_generation_failure_result()
    assert result.passed is False
    assert result.provisional_calibration is None
    assert len(result.generations) == 160
    assert len(result.records) == 384
    assert result.generations[0].execution_status == "failed"
    assert all(
        record.execution_status == "unstarted" for record in result.generations[1:]
    )
    assert all(record.execution_status == "unstarted" for record in result.records)

    output = tmp_path / "runner-failure"
    code, receipt = finalize_soft_route_mechanism_candidate_selection_delivery(
        result,
        observed_repository_revision="2" * 40,
        run_id="soft-route-mechanism-runner-failure",
        output_root=output,
    )
    assert code == 2
    assert receipt["candidate_selection_passed"] is False
    assert receipt["provisional_calibration_digest"] is None
    assert receipt["selection_artifact_filename"] is None
    assert receipt["selection_artifact_sha256"] is None
    assert receipt["blocked_class"] == "implementation_blocked"
    assert receipt["failure_reason"] == "RuntimeError"
    assert not (output / "semantic_texture_soft_route_selection_artifact.json").exists()

    persisted = json.loads(
        (output / "semantic_texture_soft_route_candidate_selection_result.json").read_text(
            "utf-8"
        )
    )
    assert len(persisted["generations"]) == 160
    assert len(persisted["records"]) == 384
    assert persisted["generations"][0]["execution_status"] == "failed"
    assert persisted["generations"][0]["failure_reason"] == "RuntimeError"
    assert {
        record["execution_status"] for record in persisted["generations"][1:]
    } == {"unstarted"}
    assert {record["execution_status"] for record in persisted["records"]} == {
        "unstarted"
    }
    assert persisted["provisional_calibration"] is None
    assert persisted["selection_artifact_filename"] is None
    assert persisted["selection_artifact_sha256"] is None
    serialized = json.dumps(persisted, sort_keys=True)
    assert "private details must remain bounded" not in serialized
    assert "Traceback" not in serialized
    assert str(tmp_path) not in serialized

    with zipfile.ZipFile(output / receipt["archive_filename"]) as archive:
        assert archive.namelist() == [
            "semantic_texture_soft_route_candidate_selection_result.json"
        ]
    sums = (output / "SHA256SUMS").read_text("ascii").splitlines()
    assert len(sums) == 3
    for line in sums:
        digest, name = line.split("  ", 1)
        assert sha256((output / name).read_bytes()).hexdigest() == digest
    assert (output / "SHA256SUMS").stat().st_mtime_ns >= max(
        path.stat().st_mtime_ns
        for path in output.iterdir()
        if path.name != "SHA256SUMS"
    )
    assert {path.name for path in output.iterdir()} == {
        "semantic_texture_soft_route_candidate_selection_result.json",
        "semantic_texture_soft_route_candidate_selection_receipt.json",
        receipt["archive_filename"],
        "SHA256SUMS",
    }
    with pytest.raises(SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError):
        finalize_soft_route_mechanism_candidate_selection_delivery(
            result,
            observed_repository_revision="2" * 40,
            run_id="soft-route-mechanism-runner-failure",
            output_root=output,
        )


@pytest.mark.parametrize(
    "malformation",
    (
        "generation_denominator",
        "detector_role",
        "frozen_identity",
        "missing_failed_slot",
        "completed_without_calibration",
        "non_finite_value",
    ),
)
def test_runner_failure_delivery_rejects_malformed_or_forged_results(
    tmp_path: Path, malformation: str
) -> None:
    result = _first_generation_failure_result()
    if malformation == "generation_denominator":
        result = replace(result, generations=result.generations[:-1])
    elif malformation == "detector_role":
        records = list(result.records)
        records[0] = replace(records[0], key_role="wrong", wrong_key_index=0)
        result = replace(result, records=tuple(records))
    elif malformation == "frozen_identity":
        original = result.generations[0].source_cluster_id
        generations = tuple(
            replace(
                record,
                source_cluster_id="e" * 64,
                image_lineage_digest="f" * 64,
            )
            if record.source_cluster_id == original
            else record
            for record in result.generations
        )
        records = tuple(
            replace(
                record,
                source_cluster_id="e" * 64,
                image_lineage_digest="f" * 64,
            )
            if record.source_cluster_id == original
            else record
            for record in result.records
        )
        result = replace(result, generations=generations, records=records)
    elif malformation == "missing_failed_slot":
        generations = list(result.generations)
        generations[0] = replace(
            generations[0], execution_status="unstarted", failure_reason=None
        )
        result = replace(result, generations=tuple(generations))
    elif malformation == "completed_without_calibration":
        completed = execute_soft_route_mechanism_split(
            _selection_manifest(), _Operations()
        )
        result = replace(
            completed,
            passed=False,
            blocked_class="implementation_blocked",
            provisional_calibration=None,
        )
    elif malformation == "non_finite_value":
        generations = list(result.generations)
        generations[0] = replace(generations[0], paired_rgb8_mse=float("nan"))
        result = replace(result, generations=tuple(generations))
    output = tmp_path / malformation
    with pytest.raises(SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError):
        finalize_soft_route_mechanism_candidate_selection_delivery(
            result,
            observed_repository_revision="3" * 40,
            run_id=f"soft-route-mechanism-{malformation}",
            output_root=output,
        )
    assert not output.exists()


def test_soft_route_mechanism_bounded_failure_is_exported_before_nonzero(tmp_path: Path) -> None:
    output = tmp_path / "failure"
    code, receipt = finalize_soft_route_mechanism_failure_delivery(observed_repository_revision="2" * 40, run_id="soft_route_mechanism-failure", output_root=output, stage="entrypoint", failure_reason="RuntimeError")
    assert code == 2
    persisted = "\n".join(path.read_text("utf-8", errors="ignore") for path in output.iterdir() if path.suffix != ".zip")
    assert "Traceback" not in persisted
    assert str(tmp_path) not in persisted
    assert (output / "SHA256SUMS").exists()
    assert receipt["status"] == "blocked"


def test_successful_confirmation_delivery_separates_and_retires_authority(tmp_path: Path) -> None:
    selection_manifest = load_manifest(
        ROOT / "configs/experiments/semantic_texture_soft_route_candidate_selection_manifest.json",
        expected_role=SELECTION_ROLE,
    )
    confirmation_manifest = load_manifest(
        ROOT / "configs/experiments/semantic_texture_soft_route_untouched_confirmation_manifest.json",
        expected_role=CONFIRMATION_ROLE,
    )
    selection_result = execute_soft_route_mechanism_split(
        selection_manifest, _Operations()
    )
    operations = _ConfirmationOperations()
    confirmation_result = execute_soft_route_mechanism_split(
        confirmation_manifest,
        operations,
        provisional_calibration=selection_result.provisional_calibration,
    )
    assert all(
        record.execution_status == "completed"
        for record in (*confirmation_result.generations, *confirmation_result.records)
    )
    confirmation_result = replace(confirmation_result, passed=True)
    source_artifact_sha256 = "c" * 64
    output = tmp_path / "confirmation-delivery"
    code, receipt = finalize_soft_route_mechanism_untouched_confirmation_delivery(
        confirmation_result,
        observed_repository_revision="3" * 40,
        run_id="soft-route-mechanism-confirmation-delivery",
        output_root=output,
        source_selection_artifact_sha256=source_artifact_sha256,
        source_selection_manifest_digest=selection_manifest.digest(),
    )
    assert code == 0
    expected_authority = {
        "protocol_id": PROTOCOL_ID,
        "confirmation_manifest_digest": CONFIRMATION_MANIFEST_DIGEST,
        "untouched_confirmation_passed": True,
        "source_selection_manifest_digest": selection_manifest.digest(),
        "source_selection_artifact_sha256": source_artifact_sha256,
        "provisional_calibration_digest": selection_result.provisional_calibration.digest(),
        "provisional_authority_retired": True,
        "diagnostic_only": True,
        "science_started": False,
        "scientific_unit_count": 0,
        "candidate_promoted": False,
        "formal_tau_created": False,
        "formal_fpr_created": False,
    }
    artifact = json.loads((output / CONFIRMATION_ARTIFACT_FILENAME).read_text("utf-8"))
    assert artifact == expected_authority
    result = json.loads((output / CONFIRMATION_RESULT_FILENAME).read_text("utf-8"))
    assert {field: result[field] for field in expected_authority} == expected_authority
    persisted_receipt = json.loads((output / CONFIRMATION_RECEIPT_FILENAME).read_text("utf-8"))
    assert {field: persisted_receipt[field] for field in expected_authority} == expected_authority
    assert receipt["confirmation_artifact_filename"] == CONFIRMATION_ARTIFACT_FILENAME
    assert len(result["generations"]) == 160
    assert len(result["records"]) == 384
    assert "provisional_calibration" not in artifact
    persisted = "\n".join(
        path.read_text("utf-8", errors="ignore")
        for path in output.iterdir()
        if path.suffix != ".zip"
    )
    assert "candidate_selection_passed" not in persisted
    assert "selection_artifact_filename" not in persisted
    with zipfile.ZipFile(output / receipt["archive_filename"]) as archive:
        assert archive.namelist() == [
            CONFIRMATION_ARTIFACT_FILENAME,
            CONFIRMATION_RESULT_FILENAME,
        ]
    assert (output / "SHA256SUMS").stat().st_mtime_ns >= max(
        path.stat().st_mtime_ns
        for path in output.iterdir()
        if path.name != "SHA256SUMS"
    )
    with pytest.raises(SemanticTextureSoftRouteMechanismConfirmationDeliveryError):
        finalize_soft_route_mechanism_untouched_confirmation_delivery(
            confirmation_result,
            observed_repository_revision="3" * 40,
            run_id="soft-route-mechanism-confirmation-delivery",
            output_root=output,
            source_selection_artifact_sha256=source_artifact_sha256,
            source_selection_manifest_digest=selection_manifest.digest(),
        )


def test_failed_confirmation_delivery_does_not_retire_authority(tmp_path: Path) -> None:
    selection_manifest = load_manifest(
        ROOT / "configs/experiments/semantic_texture_soft_route_candidate_selection_manifest.json",
        expected_role=SELECTION_ROLE,
    )
    confirmation_manifest = load_manifest(
        ROOT / "configs/experiments/semantic_texture_soft_route_untouched_confirmation_manifest.json",
        expected_role=CONFIRMATION_ROLE,
    )
    selection_result = execute_soft_route_mechanism_split(
        selection_manifest, _Operations()
    )
    confirmation_result = execute_soft_route_mechanism_split(
        confirmation_manifest,
        _ConfirmationOperations(),
        provisional_calibration=selection_result.provisional_calibration,
    )
    confirmation_result = replace(
        confirmation_result,
        passed=False,
        blocked_class="implementation_blocked",
    )
    output = tmp_path / "failed-confirmation"
    code, receipt = finalize_soft_route_mechanism_untouched_confirmation_delivery(
        confirmation_result,
        observed_repository_revision="4" * 40,
        run_id="soft-route-mechanism-failed-confirmation",
        output_root=output,
        source_selection_artifact_sha256="d" * 64,
        source_selection_manifest_digest=selection_manifest.digest(),
    )
    assert code == 2
    assert receipt["status"] == "blocked"
    assert receipt["untouched_confirmation_passed"] is False
    assert receipt["provisional_authority_retired"] is False
    artifact = json.loads((output / CONFIRMATION_ARTIFACT_FILENAME).read_text("utf-8"))
    assert artifact["untouched_confirmation_passed"] is False
    assert artifact["provisional_authority_retired"] is False
