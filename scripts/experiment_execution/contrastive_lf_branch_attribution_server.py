"""Create-only bounded Stage-A delivery for success and authenticated negatives."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from re import compile as regex
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from experiments.protocol.contrastive_lf_branch_attribution import PROTOCOL_ID
from experiments.runners.contrastive_lf_branch_attribution import StageAExecutionResult


REVISION = regex(r"^[0-9a-f]{40}$")
RUN_ID = regex(r"^contrastive-lf-branch-attribution-[0-9a-f]{32}$")
RESULT_FILENAME = "contrastive_lf_stage_a_result.json"
NULL_ARTIFACT_FILENAME = "contrastive_lf_null_fit_artifact.json"
SELECTION_ARTIFACT_FILENAME = "contrastive_lf_candidate_selection_artifact.json"
MANIFEST_FILENAME = "contrastive_lf_execution_manifest.json"
LOG_FILENAME = "contrastive_lf_execution_log_summary.json"
RECEIPT_FILENAME = "contrastive_lf_execution_receipt.json"
CHECKSUMS_FILENAME = "SHA256SUMS"


class ContrastiveLfDeliveryError(RuntimeError):
    pass


def _blob(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write(path: Path, blob: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(blob)


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def finalize_contrastive_lf_delivery(
    result: StageAExecutionResult,
    *,
    observed_repository_revision: str,
    run_id: str,
    output_root: str | Path,
    session_provenance: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    if type(result) is not StageAExecutionResult or REVISION.fullmatch(observed_repository_revision) is None or RUN_ID.fullmatch(run_id) is None:
        raise ContrastiveLfDeliveryError("Stage-A delivery identity is invalid")
    try:
        result.validate_for_delivery()
    except Exception as exc:
        raise ContrastiveLfDeliveryError("Stage-A runner result is not deliverable") from exc
    if result.science_started or result.scientific_unit_count != 0 or result.candidate_promoted or result.formal_tau_created or result.formal_fpr_created:
        raise ContrastiveLfDeliveryError("Stage-A zero-science boundary drifted")
    root = Path(output_root).resolve()
    if root.exists():
        raise ContrastiveLfDeliveryError("Stage-A output root already exists")
    root.mkdir(parents=True)
    selection = result.selection_result
    selection_passed = bool(selection and selection.candidate_selection_passed)
    provenance = {} if session_provenance is None else dict(session_provenance)
    if set(provenance) - {
        "cache_diagnostics",
        "heterogeneous_revisions",
        "producer_revisions",
        "session_id",
    }:
        raise ContrastiveLfDeliveryError("Stage-A session provenance is invalid")
    result_value = {
        "candidate_promoted": False,
        "candidate_selection_passed": selection_passed,
        "diagnostic_only": True,
        "failure_reason": result.failure_reason,
        "formal_fpr_created": False,
        "formal_tau_created": False,
        "full_ceg_wm_eligible": False,
        "null_fit_artifact_digest": None if result.null_fit_artifact is None else result.null_fit_artifact.artifact_digest,
        "null_fit_record_count": len(result.null_fit_records),
        "null_fit_records": [record.canonical_payload() for record in result.null_fit_records],
        "observed_repository_revision": observed_repository_revision,
        "protocol_id": PROTOCOL_ID,
        "result_classification": result.result_classification,
        "run_id": run_id,
        **provenance,
        "science_started": False,
        "scientific_unit_count": 0,
        "selected_candidate_id": None if selection is None else selection.selected_candidate_id,
        "selection_record_count": len(result.selection_records),
        "selection_records": [record.canonical_payload() for record in result.selection_records],
        "selection_result": None if selection is None else asdict(selection),
        "status": "completed" if result.result_classification in {"success", "scientific_failure"} else "blocked",
    }
    payloads: dict[str, bytes] = {RESULT_FILENAME: _blob(result_value)}
    if result.null_fit_artifact is not None:
        payloads[NULL_ARTIFACT_FILENAME] = _blob(result.null_fit_artifact.canonical_payload())
    if selection_passed:
        assert selection is not None and result.null_fit_artifact is not None
        selection_artifact = {
            "candidate_promoted": False,
            "candidate_selection_passed": True,
            "diagnostic_only": True,
            "formal_fpr_created": False,
            "formal_tau_created": False,
            "full_ceg_wm_eligible": False,
            "implementation_revision": observed_repository_revision,
            "null_fit_artifact_digest": result.null_fit_artifact.artifact_digest,
            "protocol_id": PROTOCOL_ID,
            "record_collection_digest": selection.record_collection_digest,
            "denominator_reports": [asdict(report) for report in selection.denominator_reports],
            "gate_reports": [asdict(report) for report in selection.gate_reports],
            "first_failed_gate": selection.first_failed_gate,
            "result_classification": selection.result_classification,
            "selected_candidate_id": selection.selected_candidate_id,
            "selection_record_count": len(result.selection_records),
            "selection_records": [record.canonical_payload() for record in result.selection_records],
            "selection_manifest_digest": selection.sample_manifest_digest,
        }
        payloads[SELECTION_ARTIFACT_FILENAME] = _blob(selection_artifact)
    manifest = {
        "artifact_filenames": sorted(name for name in payloads if name != RESULT_FILENAME),
        "observed_repository_revision": observed_repository_revision,
        "protocol_id": PROTOCOL_ID,
        "result_classification": result.result_classification,
        "result_filename": RESULT_FILENAME,
        "run_id": run_id,
        "schema_version": 1,
        **provenance,
    }
    log_summary = {
        "failure_reason": result.failure_reason,
        "first_failed_record_id": next((record.template.record_id for record in (*result.null_fit_records, *result.selection_records) if record.execution_status == "failed"), None),
        "result_classification": result.result_classification,
        "traceback_persisted": False,
    }
    payloads[MANIFEST_FILENAME] = _blob(manifest)
    payloads[LOG_FILENAME] = _blob(log_summary)
    for name, blob in payloads.items():
        _write(root / name, blob)
    archive_name = f"contrastive_lf_stage_a_{run_id}.zip"
    archive_path = root / archive_name
    with ZipFile(archive_path, "x", compression=ZIP_STORED) as archive:
        for name in sorted(payloads):
            archive.writestr(_zip_info(name), payloads[name])
    receipt = {
        "archive_filename": archive_name,
        "archive_sha256": sha256(archive_path.read_bytes()).hexdigest(),
        "candidate_selection_passed": selection_passed,
        "observed_repository_revision": observed_repository_revision,
        "protocol_id": PROTOCOL_ID,
        "result_classification": result.result_classification,
        "result_filename": RESULT_FILENAME,
        "result_sha256": sha256(payloads[RESULT_FILENAME]).hexdigest(),
        "run_id": run_id,
        **provenance,
        "selection_artifact_filename": SELECTION_ARTIFACT_FILENAME if selection_passed else None,
        "selection_artifact_sha256": sha256(payloads[SELECTION_ARTIFACT_FILENAME]).hexdigest() if selection_passed else None,
        "status": result_value["status"],
        **provenance,
    }
    receipt_blob = _blob(receipt)
    _write(root / RECEIPT_FILENAME, receipt_blob)
    rows = [f"{sha256(blob).hexdigest()}  {name}\n" for name, blob in sorted(payloads.items())]
    rows.extend((f"{receipt['archive_sha256']}  {archive_name}\n", f"{sha256(receipt_blob).hexdigest()}  {RECEIPT_FILENAME}\n"))
    sums = "".join(rows).encode("ascii")
    _write(root / CHECKSUMS_FILENAME, sums)
    code = 0 if result.result_classification == "success" else 2
    return code, {
        **receipt,
        "receipt_filename": RECEIPT_FILENAME,
        "checksum_manifest_filename": CHECKSUMS_FILENAME,
        "checksum_manifest_sha256": sha256(sums).hexdigest(),
    }


def finalize_contrastive_lf_preexecution_failure(
    *, observed_repository_revision: str, run_id: str, output_root: str | Path, failure_reason: str
) -> tuple[int, dict[str, object]]:
    if not failure_reason or len(failure_reason) > 120:
        raise ContrastiveLfDeliveryError("bounded pre-execution failure is invalid")
    empty = StageAExecutionResult((), None, (), None, "operational_failure", failure_reason)
    return finalize_contrastive_lf_delivery(empty, observed_repository_revision=observed_repository_revision, run_id=run_id, output_root=output_root)


__all__ = [
    "CHECKSUMS_FILENAME",
    "ContrastiveLfDeliveryError",
    "finalize_contrastive_lf_delivery",
    "finalize_contrastive_lf_preexecution_failure",
]
