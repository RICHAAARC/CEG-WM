"""Create-only delivery for the Phase-B diagnostic soft-detector asset bundle."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Sequence
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from experiments.protocol.semantic_texture_soft_detector_assets import (
    SemanticTextureSoftDetectorAssetBundle,
)


RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
RESULT_FILENAME = "semantic_texture_soft_detector_asset_result.json"
BUNDLE_FILENAME = "semantic_texture_soft_detector_asset_bundle.json"
RECEIPT_FILENAME = "semantic_texture_soft_detector_asset_receipt.json"
CHECKSUMS_FILENAME = "SHA256SUMS"


class SemanticTextureSoftDetectorAssetServerError(RuntimeError):
    """Asset delivery cannot preserve its create-only boundary."""


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _write_exclusive(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)


def finalize_semantic_texture_soft_detector_asset_delivery(
    bundle: SemanticTextureSoftDetectorAssetBundle,
    *,
    observed_repository_revision: str,
    run_id: str,
    output_root: str | Path,
) -> tuple[int, dict[str, object]]:
    """Persist a bounded zero-science asset bundle and result-only archive."""

    if type(bundle) is not SemanticTextureSoftDetectorAssetBundle:
        raise SemanticTextureSoftDetectorAssetServerError("asset bundle type is invalid")
    bundle.validate()
    if REVISION.fullmatch(observed_repository_revision) is None or RUN_ID.fullmatch(run_id) is None:
        raise SemanticTextureSoftDetectorAssetServerError("asset delivery identity is invalid")
    parent = Path(output_root).resolve()
    if not parent.is_dir():
        raise SemanticTextureSoftDetectorAssetServerError("asset delivery parent is invalid")
    root = parent / bundle.bundle_digest
    if root.exists():
        raise SemanticTextureSoftDetectorAssetServerError("asset delivery root already exists")
    root.mkdir()
    bundle_blob = _canonical_bytes({**bundle.canonical_payload(), "bundle_digest": bundle.bundle_digest})
    result = {
        "asset_bundle_digest": bundle.bundle_digest,
        "candidate_promoted": False,
        "diagnostic_only": True,
        "formal_tau_created": False,
        "observed_repository_revision": observed_repository_revision,
        "profile_id": "semantic_texture_soft_detector_asset_preparation",
        "run_id": run_id,
        "science_started": False,
        "scientific_claims_supported": False,
        "scientific_unit_count": 0,
        "status": "passed",
    }
    result_blob = _canonical_bytes(result)
    bundle_path, result_path = root / BUNDLE_FILENAME, root / RESULT_FILENAME
    _write_exclusive(bundle_path, bundle_blob)
    _write_exclusive(result_path, result_blob)
    archive_path = root / f"semantic_texture_soft_detector_assets_{bundle.bundle_digest}.zip"
    with ZipFile(archive_path, "x", compression=ZIP_STORED) as archive:
        archive.writestr(_zip_info(RESULT_FILENAME), result_blob)
    digest = lambda blob: sha256(blob).hexdigest()
    receipt = {
        "asset_bundle_digest": bundle.bundle_digest,
        "bundle_filename": BUNDLE_FILENAME,
        "bundle_sha256": digest(bundle_blob),
        "candidate_promoted": False,
        "diagnostic_only": True,
        "formal_tau_created": False,
        "observed_repository_revision": observed_repository_revision,
        "result_filename": RESULT_FILENAME,
        "result_sha256": digest(result_blob),
        "run_id": run_id,
        "science_started": False,
        "scientific_claims_supported": False,
        "scientific_unit_count": 0,
        "status": "passed",
        "zip_filename": archive_path.name,
        "zip_sha256": sha256(archive_path.read_bytes()).hexdigest(),
    }
    receipt_blob = _canonical_bytes(receipt)
    receipt_path = root / RECEIPT_FILENAME
    _write_exclusive(receipt_path, receipt_blob)
    checksums_blob = (
        f"{digest(bundle_blob)}  {BUNDLE_FILENAME}\n"
        f"{digest(result_blob)}  {RESULT_FILENAME}\n"
        f"{receipt['zip_sha256']}  {archive_path.name}\n"
        f"{digest(receipt_blob)}  {RECEIPT_FILENAME}\n"
    ).encode("ascii")
    _write_exclusive(root / CHECKSUMS_FILENAME, checksums_blob)
    return 0, {
        **receipt,
        "receipt_filename": RECEIPT_FILENAME,
        "receipt_sha256": digest(receipt_blob),
        "sha256sums_filename": CHECKSUMS_FILENAME,
        "sha256sums_sha256": digest(checksums_blob),
    }


def finalize_semantic_texture_soft_detector_asset_blocked_delivery(
    *,
    observed_repository_revision: str,
    run_id: str,
    blocked_class: str,
    output_root: str | Path,
) -> tuple[int, dict[str, object]]:
    """Persist only bounded failed preparation state, never an asset bundle."""

    if REVISION.fullmatch(observed_repository_revision) is None or RUN_ID.fullmatch(run_id) is None:
        raise SemanticTextureSoftDetectorAssetServerError("asset delivery identity is invalid")
    if blocked_class not in {"environment_blocked", "resource_blocked", "implementation_blocked", "integrity_blocked"}:
        raise SemanticTextureSoftDetectorAssetServerError("asset delivery blocked class is invalid")
    parent = Path(output_root).resolve()
    root = parent / run_id
    if not parent.is_dir() or root.exists():
        raise SemanticTextureSoftDetectorAssetServerError("asset blocked delivery root is invalid")
    root.mkdir()
    result = {
        "candidate_promoted": False, "diagnostic_only": True,
        "formal_tau_created": False, "observed_repository_revision": observed_repository_revision,
        "profile_id": "semantic_texture_soft_detector_asset_preparation", "run_id": run_id,
        "science_started": False, "scientific_claims_supported": False,
        "scientific_unit_count": 0, "status": "blocked", "blocked_class": blocked_class,
    }
    result_blob = _canonical_bytes(result)
    result_path = root / RESULT_FILENAME
    _write_exclusive(result_path, result_blob)
    archive_path = root / f"semantic_texture_soft_detector_assets_{run_id}.zip"
    with ZipFile(archive_path, "x", compression=ZIP_STORED) as archive:
        archive.writestr(_zip_info(RESULT_FILENAME), result_blob)
    receipt = {**result, "result_filename": RESULT_FILENAME, "result_sha256": sha256(result_blob).hexdigest(), "zip_filename": archive_path.name, "zip_sha256": sha256(archive_path.read_bytes()).hexdigest()}
    receipt_blob = _canonical_bytes(receipt)
    receipt_path = root / RECEIPT_FILENAME
    _write_exclusive(receipt_path, receipt_blob)
    checksums_blob = (f"{sha256(result_blob).hexdigest()}  {RESULT_FILENAME}\n{receipt['zip_sha256']}  {archive_path.name}\n{sha256(receipt_blob).hexdigest()}  {RECEIPT_FILENAME}\n").encode("ascii")
    _write_exclusive(root / CHECKSUMS_FILENAME, checksums_blob)
    return 2, {**receipt, "receipt_filename": RECEIPT_FILENAME, "receipt_sha256": sha256(receipt_blob).hexdigest(), "sha256sums_filename": CHECKSUMS_FILENAME, "sha256sums_sha256": sha256(checksums_blob).hexdigest()}


__all__ = [
    "BUNDLE_FILENAME", "CHECKSUMS_FILENAME", "RECEIPT_FILENAME", "RESULT_FILENAME",
    "SemanticTextureSoftDetectorAssetServerError", "finalize_semantic_texture_soft_detector_asset_blocked_delivery", "finalize_semantic_texture_soft_detector_asset_delivery",
]
