"""Create-only bounded delivery for soft-route candidate selection."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from experiments.runners.semantic_texture_soft_route_mechanism_validation import SoftRouteMechanismSplitResult


RESULT_FILENAME = "semantic_texture_soft_route_candidate_selection_result.json"
RECEIPT_FILENAME = "semantic_texture_soft_route_candidate_selection_receipt.json"
SELECTION_FILENAME = "semantic_texture_soft_route_selection_artifact.json"
CHECKSUMS_FILENAME = "SHA256SUMS"
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")


class SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError(RuntimeError):
    """Candidate-selection delivery did not retain its create-only boundary."""


def _blob(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_create_only_blob(path: Path, blob: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(blob)


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def finalize_soft_route_mechanism_candidate_selection_delivery(
    result: SoftRouteMechanismSplitResult, *, observed_repository_revision: str, run_id: str, output_root: str | Path,
    expected_role: str = "semantic_texture_soft_route_candidate_selection",
    result_filename: str = RESULT_FILENAME,
    receipt_filename: str = RECEIPT_FILENAME,
    artifact_filename: str = SELECTION_FILENAME,
    archive_prefix: str = "semantic_texture_soft_route_candidate_selection",
) -> tuple[int, dict[str, object]]:
    """Persist the exact selection matrix and provisional-only artifact once."""

    if (
        type(result) is not SoftRouteMechanismSplitResult
        or result.role_id != expected_role
        or RUN_ID.fullmatch(run_id) is None
        or REVISION.fullmatch(observed_repository_revision) is None
    ):
        raise SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError("selection result identity is invalid")
    if (
        len(result.generations) != 160
        or len(result.records) != 384
        or result.provisional_calibration is None
        or result.science_started
        or result.candidate_promoted
        or result.formal_tau_created
    ):
        raise SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError("selection result boundary is invalid")
    root = Path(output_root).resolve()
    if root.exists():
        raise SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError("selection output already exists")
    root.mkdir(parents=True)
    artifact = {
        "protocol_id": result.protocol_id,
        "selection_manifest_digest": result.manifest_digest,
        "provisional_calibration": asdict(result.provisional_calibration),
        "candidate_selection_passed": result.passed,
        "diagnostic_only": True,
        "science_started": False,
        "scientific_unit_count": 0,
        "candidate_promoted": False,
        "formal_tau_created": False,
        "formal_fpr_created": False,
    }
    artifact_blob = _blob(artifact)
    result_value = {
        **artifact,
        "observed_repository_revision": observed_repository_revision,
        "run_id": run_id,
        "records": [asdict(record) for record in result.records],
        "generations": [asdict(record) for record in result.generations],
        "status": "passed" if result.passed else "blocked",
        "blocked_class": None if result.passed else "integrity_blocked",
    }
    result_blob = _blob(result_value)
    artifact_path, result_path = root / artifact_filename, root / result_filename
    _write_create_only_blob(artifact_path, artifact_blob)
    _write_create_only_blob(result_path, result_blob)
    archive_path = root / f"{archive_prefix}_{run_id}.zip"
    with ZipFile(archive_path, "x", compression=ZIP_STORED) as archive:
        archive.writestr(_zip_info(artifact_filename), artifact_blob)
        archive.writestr(_zip_info(result_filename), result_blob)
    receipt = {
        "archive_filename": archive_path.name,
        "archive_sha256": sha256(archive_path.read_bytes()).hexdigest(),
        "candidate_selection_passed": result.passed,
        "diagnostic_only": True,
        "observed_repository_revision": observed_repository_revision,
        "protocol_id": result.protocol_id,
        "result_filename": result_filename,
        "result_sha256": sha256(result_blob).hexdigest(),
        "run_id": run_id,
        "selection_artifact_filename": artifact_filename,
        "selection_artifact_sha256": sha256(artifact_blob).hexdigest(),
        "selection_manifest_digest": result.manifest_digest,
        "status": result_value["status"],
    }
    receipt_blob = _blob(receipt)
    _write_create_only_blob(root / receipt_filename, receipt_blob)
    sums = (
        f"{sha256(artifact_blob).hexdigest()}  {artifact_filename}\n"
        f"{sha256(result_blob).hexdigest()}  {result_filename}\n"
        f"{receipt['archive_sha256']}  {archive_path.name}\n"
        f"{sha256(receipt_blob).hexdigest()}  {receipt_filename}\n"
    ).encode("ascii")
    _write_create_only_blob(root / CHECKSUMS_FILENAME, sums)
    return (0 if result.passed else 2), {**receipt, "receipt_filename": receipt_filename, "sha256sums_filename": CHECKSUMS_FILENAME, "sha256sums_sha256": sha256(sums).hexdigest()}


def finalize_soft_route_mechanism_failure_delivery(
    *,
    observed_repository_revision: str,
    run_id: str,
    output_root: str | Path,
    stage: str,
    failure_reason: str,
    result_filename: str = RESULT_FILENAME,
    receipt_filename: str = RECEIPT_FILENAME,
    archive_prefix: str = "semantic_texture_soft_route_candidate_selection",
) -> tuple[int, dict[str, object]]:
    """Export bounded pre-execution evidence without paths or traceback text."""

    if (
        not stage
        or not failure_reason
        or len(stage) > 80
        or len(failure_reason) > 80
        or RUN_ID.fullmatch(run_id) is None
        or (
            observed_repository_revision != "unavailable"
            and REVISION.fullmatch(observed_repository_revision) is None
        )
    ):
        raise SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError(
            "bounded failure identity is invalid"
        )
    root = Path(output_root).resolve()
    if root.exists():
        raise SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError(
            "selection output already exists"
        )
    root.mkdir(parents=True)
    result_value = {
        "blocked_class": "implementation_blocked",
        "candidate_promoted": False,
        "diagnostic_only": True,
        "failure_reason": failure_reason,
        "formal_fpr_created": False,
        "formal_tau_created": False,
        "observed_repository_revision": observed_repository_revision,
        "run_id": run_id,
        "science_started": False,
        "scientific_unit_count": 0,
        "stage": stage,
        "status": "blocked",
    }
    result_blob = _blob(result_value)
    _write_create_only_blob(root / result_filename, result_blob)
    archive_path = root / f"{archive_prefix}_{run_id}.zip"
    with ZipFile(archive_path, "x", compression=ZIP_STORED) as archive:
        archive.writestr(_zip_info(result_filename), result_blob)
    receipt = {
        "archive_filename": archive_path.name,
        "archive_sha256": sha256(archive_path.read_bytes()).hexdigest(),
        "diagnostic_only": True,
        "observed_repository_revision": observed_repository_revision,
        "result_filename": result_filename,
        "result_sha256": sha256(result_blob).hexdigest(),
        "run_id": run_id,
        "status": "blocked",
    }
    receipt_blob = _blob(receipt)
    _write_create_only_blob(root / receipt_filename, receipt_blob)
    sums = (
        f"{sha256(result_blob).hexdigest()}  {result_filename}\n"
        f"{receipt['archive_sha256']}  {archive_path.name}\n"
        f"{sha256(receipt_blob).hexdigest()}  {receipt_filename}\n"
    ).encode("ascii")
    _write_create_only_blob(root / CHECKSUMS_FILENAME, sums)
    return 2, {
        **receipt,
        "receipt_filename": receipt_filename,
        "sha256sums_filename": CHECKSUMS_FILENAME,
        "sha256sums_sha256": sha256(sums).hexdigest(),
    }
