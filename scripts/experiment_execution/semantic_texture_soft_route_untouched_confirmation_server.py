"""Create-only bounded delivery for untouched soft-route confirmation."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from experiments.protocol.semantic_texture_soft_route_mechanism_validation import (
    CONFIRMATION_MANIFEST_DIGEST,
    CONFIRMATION_ROLE,
    PROTOCOL_ID,
    SELECTION_MANIFEST_DIGEST,
)
from experiments.runners.semantic_texture_soft_route_mechanism_validation import (
    SoftRouteMechanismSplitResult,
)


RESULT_FILENAME = "semantic_texture_soft_route_untouched_confirmation_result.json"
RECEIPT_FILENAME = "semantic_texture_soft_route_untouched_confirmation_receipt.json"
CONFIRMATION_ARTIFACT_FILENAME = "semantic_texture_soft_route_untouched_confirmation_artifact.json"
CHECKSUMS_FILENAME = "SHA256SUMS"
ARCHIVE_PREFIX = "semantic_texture_soft_route_untouched_confirmation"
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
_BOUNDED_BLOCKED_CLASSES = frozenset(
    {
        "environment_blocked",
        "identity_blocked",
        "implementation_blocked",
        "integrity_blocked",
        "resource_blocked",
    }
)


class SemanticTextureSoftRouteMechanismConfirmationDeliveryError(RuntimeError):
    """Untouched-confirmation delivery did not retain its authority boundary."""


def _blob(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_create_only_blob(path: Path, blob: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(blob)


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _confirmation_authority_fields(
    *,
    source_selection_artifact_sha256: str | None,
    source_selection_manifest_digest: str | None,
    provisional_calibration_digest: str | None,
    untouched_confirmation_passed: bool,
) -> dict[str, object]:
    return {
        "protocol_id": PROTOCOL_ID,
        "confirmation_manifest_digest": CONFIRMATION_MANIFEST_DIGEST,
        "untouched_confirmation_passed": untouched_confirmation_passed,
        "source_selection_manifest_digest": source_selection_manifest_digest,
        "source_selection_artifact_sha256": source_selection_artifact_sha256,
        "provisional_calibration_digest": provisional_calibration_digest,
        "provisional_authority_retired": untouched_confirmation_passed,
        "diagnostic_only": True,
        "science_started": False,
        "scientific_unit_count": 0,
        "candidate_promoted": False,
        "formal_tau_created": False,
        "formal_fpr_created": False,
    }


def finalize_soft_route_mechanism_untouched_confirmation_delivery(
    result: SoftRouteMechanismSplitResult,
    *,
    observed_repository_revision: str,
    run_id: str,
    output_root: str | Path,
    source_selection_artifact_sha256: str,
    source_selection_manifest_digest: str,
) -> tuple[int, dict[str, object]]:
    """Persist confirmation-only identity without re-emitting selection authority."""

    calibration = (
        result.provisional_calibration
        if type(result) is SoftRouteMechanismSplitResult
        else None
    )
    if (
        type(result) is not SoftRouteMechanismSplitResult
        or result.protocol_id != PROTOCOL_ID
        or result.role_id != CONFIRMATION_ROLE
        or result.manifest_digest != CONFIRMATION_MANIFEST_DIGEST
        or RUN_ID.fullmatch(run_id) is None
        or REVISION.fullmatch(observed_repository_revision) is None
        or DIGEST.fullmatch(source_selection_artifact_sha256) is None
        or source_selection_manifest_digest != SELECTION_MANIFEST_DIGEST
        or calibration is None
        or calibration.selection_manifest_digest != source_selection_manifest_digest
        or calibration.retired
        or len(result.generations) != 160
        or len(result.records) != 384
        or result.science_started
        or result.scientific_unit_count != 0
        or result.candidate_promoted
        or result.formal_tau_created
    ):
        raise SemanticTextureSoftRouteMechanismConfirmationDeliveryError(
            "confirmation result authority is invalid"
        )
    completed = all(
        record.execution_status == "completed"
        for record in (*result.generations, *result.records)
    )
    if result.passed and not completed:
        raise SemanticTextureSoftRouteMechanismConfirmationDeliveryError(
            "confirmation pass identity is incomplete"
        )
    authority = _confirmation_authority_fields(
        source_selection_artifact_sha256=source_selection_artifact_sha256,
        source_selection_manifest_digest=source_selection_manifest_digest,
        provisional_calibration_digest=calibration.digest(),
        untouched_confirmation_passed=result.passed,
    )
    root = Path(output_root).resolve()
    if root.exists():
        raise SemanticTextureSoftRouteMechanismConfirmationDeliveryError(
            "confirmation output already exists"
        )
    root.mkdir(parents=True)
    artifact_blob = _blob(authority)
    result_value = {
        **authority,
        "observed_repository_revision": observed_repository_revision,
        "run_id": run_id,
        "records": [asdict(record) for record in result.records],
        "generations": [asdict(record) for record in result.generations],
        "status": "passed" if result.passed else "blocked",
        "blocked_class": (
            None
            if result.passed
            else (result.blocked_class or "implementation_blocked")
        ),
    }
    result_blob = _blob(result_value)
    artifact_path = root / CONFIRMATION_ARTIFACT_FILENAME
    result_path = root / RESULT_FILENAME
    _write_create_only_blob(artifact_path, artifact_blob)
    _write_create_only_blob(result_path, result_blob)
    archive_path = root / f"{ARCHIVE_PREFIX}_{run_id}.zip"
    with ZipFile(archive_path, "x", compression=ZIP_STORED) as archive:
        archive.writestr(_zip_info(CONFIRMATION_ARTIFACT_FILENAME), artifact_blob)
        archive.writestr(_zip_info(RESULT_FILENAME), result_blob)
    receipt = {
        **authority,
        "archive_filename": archive_path.name,
        "archive_sha256": sha256(archive_path.read_bytes()).hexdigest(),
        "confirmation_artifact_filename": CONFIRMATION_ARTIFACT_FILENAME,
        "confirmation_artifact_sha256": sha256(artifact_blob).hexdigest(),
        "observed_repository_revision": observed_repository_revision,
        "result_filename": RESULT_FILENAME,
        "result_sha256": sha256(result_blob).hexdigest(),
        "run_id": run_id,
        "status": result_value["status"],
    }
    receipt_blob = _blob(receipt)
    _write_create_only_blob(root / RECEIPT_FILENAME, receipt_blob)
    sums = (
        f"{sha256(artifact_blob).hexdigest()}  {CONFIRMATION_ARTIFACT_FILENAME}\n"
        f"{sha256(result_blob).hexdigest()}  {RESULT_FILENAME}\n"
        f"{receipt['archive_sha256']}  {archive_path.name}\n"
        f"{sha256(receipt_blob).hexdigest()}  {RECEIPT_FILENAME}\n"
    ).encode("ascii")
    _write_create_only_blob(root / CHECKSUMS_FILENAME, sums)
    return (0 if result.passed else 2), {
        **receipt,
        "receipt_filename": RECEIPT_FILENAME,
        "sha256sums_filename": CHECKSUMS_FILENAME,
        "sha256sums_sha256": sha256(sums).hexdigest(),
    }


def finalize_soft_route_mechanism_untouched_confirmation_failure_delivery(
    *,
    observed_repository_revision: str,
    run_id: str,
    output_root: str | Path,
    stage: str,
    failure_reason: str,
    source_selection_artifact_sha256: str | None = None,
    source_selection_manifest_digest: str | None = None,
    provisional_calibration_digest: str | None = None,
) -> tuple[int, dict[str, object]]:
    """Persist bounded confirmation-named failure without authority retirement."""

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
        raise SemanticTextureSoftRouteMechanismConfirmationDeliveryError(
            "bounded confirmation failure identity is invalid"
        )
    safe_source_artifact_sha256 = (
        source_selection_artifact_sha256
        if source_selection_artifact_sha256 is not None
        and DIGEST.fullmatch(source_selection_artifact_sha256) is not None
        else None
    )
    safe_source_manifest_digest = (
        source_selection_manifest_digest
        if source_selection_manifest_digest == SELECTION_MANIFEST_DIGEST
        else None
    )
    safe_calibration_digest = (
        provisional_calibration_digest
        if provisional_calibration_digest is not None
        and DIGEST.fullmatch(provisional_calibration_digest) is not None
        else None
    )
    blocked_class = (
        failure_reason
        if failure_reason in _BOUNDED_BLOCKED_CLASSES
        else "implementation_blocked"
    )
    authority = _confirmation_authority_fields(
        source_selection_artifact_sha256=safe_source_artifact_sha256,
        source_selection_manifest_digest=safe_source_manifest_digest,
        provisional_calibration_digest=safe_calibration_digest,
        untouched_confirmation_passed=False,
    )
    result_value = {
        **authority,
        "blocked_class": blocked_class,
        "failure_reason": failure_reason,
        "observed_repository_revision": observed_repository_revision,
        "run_id": run_id,
        "stage": stage,
        "status": "blocked",
    }
    root = Path(output_root).resolve()
    if root.exists():
        raise SemanticTextureSoftRouteMechanismConfirmationDeliveryError(
            "confirmation output already exists"
        )
    root.mkdir(parents=True)
    result_blob = _blob(result_value)
    _write_create_only_blob(root / RESULT_FILENAME, result_blob)
    archive_path = root / f"{ARCHIVE_PREFIX}_{run_id}.zip"
    with ZipFile(archive_path, "x", compression=ZIP_STORED) as archive:
        archive.writestr(_zip_info(RESULT_FILENAME), result_blob)
    receipt = {
        **authority,
        "archive_filename": archive_path.name,
        "archive_sha256": sha256(archive_path.read_bytes()).hexdigest(),
        "blocked_class": blocked_class,
        "observed_repository_revision": observed_repository_revision,
        "result_filename": RESULT_FILENAME,
        "result_sha256": sha256(result_blob).hexdigest(),
        "run_id": run_id,
        "status": "blocked",
    }
    receipt_blob = _blob(receipt)
    _write_create_only_blob(root / RECEIPT_FILENAME, receipt_blob)
    sums = (
        f"{sha256(result_blob).hexdigest()}  {RESULT_FILENAME}\n"
        f"{receipt['archive_sha256']}  {archive_path.name}\n"
        f"{sha256(receipt_blob).hexdigest()}  {RECEIPT_FILENAME}\n"
    ).encode("ascii")
    _write_create_only_blob(root / CHECKSUMS_FILENAME, sums)
    return 2, {
        **receipt,
        "receipt_filename": RECEIPT_FILENAME,
        "sha256sums_filename": CHECKSUMS_FILENAME,
        "sha256sums_sha256": sha256(sums).hexdigest(),
    }
