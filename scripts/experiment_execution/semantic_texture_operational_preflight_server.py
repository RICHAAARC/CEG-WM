"""Persist the semantic-texture operational result, ZIP, then receipt."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence
from zipfile import ZIP_STORED, ZipFile, ZipInfo


RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
MAX_RESULT_BYTES = 256 * 1024
MAX_DIAGNOSTIC_FILES = 8
MAX_DIAGNOSTIC_BYTES = 64 * 1024
MAX_ARCHIVE_BYTES = 1024 * 1024
RESULT_FILENAME = "semantic_texture_operational_result.json"
RECEIPT_FILENAME = "semantic_texture_operational_receipt.json"


class SemanticTextureOperationalServerError(RuntimeError):
    """The final operational delivery could not be persisted safely."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zip_info(path_text: str) -> ZipInfo:
    info = ZipInfo(path_text, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _safe_diagnostic_name(path_text: str) -> str:
    path = PurePosixPath(path_text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] != "diagnostics"
        or not path.parts[1].endswith(".json")
        or "\\" in path_text
    ):
        raise SemanticTextureOperationalServerError(
            "diagnostic archive path is invalid"
        )
    return path.as_posix()


def _validated_result(result: object) -> dict[str, object]:
    if not hasattr(result, "as_dict"):
        raise SemanticTextureOperationalServerError(
            "operational result object is invalid"
        )
    value = result.as_dict()
    if (
        type(value) is not dict
        or value.get("profile_id") != "semantic_texture_operational_preflight"
        or value.get("status") != "blocked"
        or value.get("aggregate") is not None
        or value.get("scientific_unit_count") != 0
        or value.get("science_started") is not False
        or value.get("formal_tau_created") is not False
        or value.get("candidate_promoted") is not False
        or value.get("scientific_claims_supported") is not False
        or value.get("blocked_class")
        not in {
            "environment_blocked",
            "resource_blocked",
            "implementation_blocked",
            "identity_blocked",
            "integrity_blocked",
        }
        or type(value.get("unit_outcomes")) is not list
        or len(value["unit_outcomes"]) != 2
    ):
        raise SemanticTextureOperationalServerError(
            "operational result boundary drifted"
        )
    return value


def finalize_semantic_texture_operational_preflight_delivery(
    result: object,
    *,
    output_root: str | Path,
    diagnostics: Mapping[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    """Create result JSON, deterministic ZIP, and one final external receipt."""

    result_value = _validated_result(result)
    run_id = result_value.get("run_id")
    result_identity = result_value.get("result_identity")
    if (
        type(run_id) is not str
        or RUN_ID.fullmatch(run_id) is None
        or type(result_identity) is not str
        or DIGEST.fullmatch(result_identity) is None
    ):
        raise SemanticTextureOperationalServerError(
            "result delivery identity is invalid"
        )
    root = Path(output_root).resolve()
    if root.exists():
        raise SemanticTextureOperationalServerError(
            "output root must be absent"
        )
    root.mkdir(parents=True)
    result_path = root / RESULT_FILENAME
    archive_path = root / f"semantic_texture_operational_{run_id}.zip"
    receipt_path = root / RECEIPT_FILENAME
    result_blob = _canonical_bytes(result_value)
    if len(result_blob) > MAX_RESULT_BYTES:
        raise SemanticTextureOperationalServerError(
            "operational result exceeds its bound"
        )
    with result_path.open("xb") as handle:
        handle.write(result_blob)

    diagnostic_entries: list[tuple[str, bytes]] = []
    for path_text, value in sorted((diagnostics or {}).items()):
        path_name = _safe_diagnostic_name(path_text)
        blob = _canonical_bytes(value)
        if len(blob) > MAX_DIAGNOSTIC_BYTES:
            raise SemanticTextureOperationalServerError(
                "diagnostic exceeds its bound"
            )
        diagnostic_entries.append((path_name, blob))
    if len(diagnostic_entries) > MAX_DIAGNOSTIC_FILES:
        raise SemanticTextureOperationalServerError(
            "diagnostic count exceeds its bound"
        )
    with ZipFile(archive_path, mode="x", compression=ZIP_STORED) as archive:
        archive.writestr(_zip_info(RESULT_FILENAME), result_blob)
        for path_text, blob in diagnostic_entries:
            archive.writestr(_zip_info(path_text), blob)
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise SemanticTextureOperationalServerError(
            "operational archive exceeds its bound"
        )
    archive_sha256 = _sha256_file(archive_path)
    receipt = {
        "aggregate": None,
        "archive_filename": archive_path.name,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_path.stat().st_size,
        "blocked_class": result_value["blocked_class"],
        "candidate_promoted": False,
        "configuration_digest": result_value["configuration_digest"],
        "formal_tau_created": False,
        "package_identity": result_value["package_identity"],
        "profile_id": result_value["profile_id"],
        "result_filename": result_path.name,
        "result_identity": result_identity,
        "result_sha256": sha256(result_blob).hexdigest(),
        "run_id": run_id,
        "science_started": False,
        "scientific_claims_supported": False,
        "scientific_unit_count": 0,
        "source_revision": result_value["source_revision"],
        "status": "blocked",
    }
    with receipt_path.open("xb") as handle:
        handle.write(_canonical_bytes(receipt))
    return 2, {
        **receipt,
        "receipt_filename": receipt_path.name,
        "receipt_sha256": _sha256_file(receipt_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--describe-boundary",
        action="store_true",
        help="describe the Phase A persistence order without executing a model",
    )
    arguments = parser.parse_args(argv)
    if arguments.describe_boundary:
        print(
            json.dumps(
                {
                    "archive_contains_receipt": False,
                    "persistence_order": [
                        "result_json",
                        "bounded_deterministic_zip",
                        "zip_sha256",
                        "immutable_external_receipt",
                    ],
                    "science_started": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
