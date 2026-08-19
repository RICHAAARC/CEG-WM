"""Persist the semantic-texture operational result, result-only archive, external receipt, and final completion checksums."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Sequence
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from experiments.runners.semantic_texture_operational_preflight import (
    ALLOWED_PRE_EXECUTION_STAGES,
    ALLOWED_SEMANTIC_RUNTIME_INITIALIZATION_STEPS,
)


RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
MAX_RESULT_BYTES = 256 * 1024
MAX_ARCHIVE_BYTES = 1024 * 1024
RESULT_FILENAME = "semantic_texture_operational_result.json"
RECEIPT_FILENAME = "semantic_texture_operational_receipt.json"
DELIVERY_COMPLETION_CHECKSUMS_FILENAME = "SHA256SUMS"
WRITE_UNIT_ID = "semantic_texture_write_operational"
BLIND_DETECTION_UNIT_ID = "semantic_texture_blind_detection_operational"
BLOCKED_CLASSES = frozenset(
    {
        "environment_blocked",
        "resource_blocked",
        "implementation_blocked",
        "identity_blocked",
        "integrity_blocked",
    }
)
RESULT_FIELDS = frozenset(
    {
        "aggregate",
        "asset_authority_status",
        "asset_bundle_digest",
        "blocked_class",
        "candidate_promoted",
        "configuration_digest",
        "diagnostic_only",
        "formal_tau_created",
        "model_id",
        "model_revision",
        "observed_repository_revision",
        "pre_execution_stage",
        "semantic_runtime_initialization_step",
        "profile_id",
        "result_identity",
        "run_id",
        "schema_version",
        "science_started",
        "scientific_claims_supported",
        "scientific_unit_count",
        "status",
        "unit_outcomes",
    }
)
UNIT_OUTCOME_FIELDS = frozenset(
    {
        "blocked_class",
        "elapsed_seconds",
        "public_result_identity",
        "sanitized_error_category",
        "sanitized_error_message",
        "sanitized_trace_tail",
        "started",
        "status",
        "unit_id",
        "witness_identity",
    }
)


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


def _validated_result(result: object) -> dict[str, object]:
    as_dict = getattr(result, "as_dict", None)
    if not callable(as_dict):
        raise SemanticTextureOperationalServerError(
            "operational result object is invalid"
        )
    value = as_dict()
    if (
        type(value) is not dict
        or set(value) != RESULT_FIELDS
        or value.get("profile_id") != "semantic_texture_operational_preflight"
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 4
        or value.get("status") not in {"blocked", "passed"}
        or value.get("aggregate") is not None
        or value.get("asset_authority_status") != "diagnostic_bundle_authenticated"
        or type(value.get("asset_bundle_digest")) is not str
        or DIGEST.fullmatch(value["asset_bundle_digest"]) is None
        or value.get("diagnostic_only") is not True
        or type(value.get("scientific_unit_count")) is not int
        or value.get("scientific_unit_count") != 0
        or value.get("science_started") is not False
        or value.get("formal_tau_created") is not False
        or value.get("candidate_promoted") is not False
        or value.get("scientific_claims_supported") is not False
        or (
            value.get("blocked_class") is not None
            and value.get("blocked_class") not in BLOCKED_CLASSES
        )
        or any(
            type(value.get(field)) is not str or not value[field]
            for field in ("model_id", "model_revision")
        )
        or type(value.get("run_id")) is not str
        or RUN_ID.fullmatch(value["run_id"]) is None
        or type(value.get("observed_repository_revision")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", value["observed_repository_revision"]) is None
        or any(
            type(value.get(field)) is not str
            or DIGEST.fullmatch(value[field]) is None
            for field in (
                "configuration_digest",
                "result_identity",
            )
        )
        or type(value.get("unit_outcomes")) is not list
        or len(value["unit_outcomes"]) != 2
    ):
        raise SemanticTextureOperationalServerError(
            "operational result boundary drifted"
        )
    write_outcome, detector_outcome = value["unit_outcomes"]
    if any(
        type(outcome) is not dict
        or set(outcome) != UNIT_OUTCOME_FIELDS
        or type(outcome.get("started")) is not bool
        or type(outcome.get("elapsed_seconds")) not in (int, float)
        or isinstance(outcome.get("elapsed_seconds"), bool)
        or not math.isfinite(outcome["elapsed_seconds"])
        or outcome["elapsed_seconds"] < 0
        or outcome.get("sanitized_error_message") is not None
        or outcome.get("sanitized_trace_tail") != []
        for outcome in value["unit_outcomes"]
    ):
        raise SemanticTextureOperationalServerError(
            "operational unit outcome boundary drifted"
        )
    if (
        write_outcome["unit_id"] != WRITE_UNIT_ID
        or detector_outcome["unit_id"] != BLIND_DETECTION_UNIT_ID
    ):
        raise SemanticTextureOperationalServerError(
            "operational unit roster drifted"
        )
    passed = value["status"] == "passed"
    pre_execution_failure = (
        write_outcome["started"] is False
        and detector_outcome["started"] is False
    )
    if pre_execution_failure:
        if (
            value.get("pre_execution_stage") not in ALLOWED_PRE_EXECUTION_STAGES
            or type(value["pre_execution_stage"]) is not str
            or write_outcome["status"] != "blocked"
            or write_outcome["blocked_class"] not in BLOCKED_CLASSES
            or write_outcome["sanitized_error_category"]
            != write_outcome["blocked_class"]
            or write_outcome["public_result_identity"] is not None
            or write_outcome["witness_identity"] is not None
            or detector_outcome["blocked_class"]
            != write_outcome["blocked_class"]
        ):
            raise SemanticTextureOperationalServerError(
                "pre-execution unit boundary drifted"
            )
        if value["pre_execution_stage"] == "semantic_runtime_initialization":
            if (
                type(value.get("semantic_runtime_initialization_step")) is not str
                or value["semantic_runtime_initialization_step"]
                not in ALLOWED_SEMANTIC_RUNTIME_INITIALIZATION_STEPS
            ):
                raise SemanticTextureOperationalServerError(
                    "semantic runtime initialization detail drifted"
                )
        elif value.get("semantic_runtime_initialization_step") is not None:
            raise SemanticTextureOperationalServerError(
                "unrelated pre-execution stage retains semantic runtime detail"
            )
    elif (
        value.get("pre_execution_stage") is not None
        or value.get("semantic_runtime_initialization_step") is not None
    ):
        raise SemanticTextureOperationalServerError(
            "started unit result retains a pre-execution stage"
        )
    elif write_outcome["started"] is not True:
        raise SemanticTextureOperationalServerError(
            "write unit start identity drifted"
        )
    elif write_outcome["status"] == "passed":
        if (
            write_outcome["blocked_class"] is not None
            or write_outcome["sanitized_error_category"] is not None
            or type(write_outcome["public_result_identity"]) is not str
            or DIGEST.fullmatch(write_outcome["public_result_identity"]) is None
            or type(write_outcome["witness_identity"]) is not str
            or DIGEST.fullmatch(write_outcome["witness_identity"]) is None
            or detector_outcome["started"] is not True
        ):
            raise SemanticTextureOperationalServerError(
                "passed write unit boundary drifted"
            )
    elif write_outcome["status"] == "blocked":
        if (
            write_outcome["blocked_class"] not in BLOCKED_CLASSES
            or write_outcome["sanitized_error_category"]
            != write_outcome["blocked_class"]
            or write_outcome["public_result_identity"] is not None
            or write_outcome["witness_identity"] is not None
            or detector_outcome["started"] is not False
            or detector_outcome["blocked_class"]
            != write_outcome["blocked_class"]
        ):
            raise SemanticTextureOperationalServerError(
                "blocked write unit boundary drifted"
            )
    else:
        raise SemanticTextureOperationalServerError(
            "write unit status drifted"
        )
    if passed:
        if (
            write_outcome["status"] != "passed"
            or detector_outcome["status"] != "passed"
            or write_outcome["blocked_class"] is not None
            or detector_outcome["blocked_class"] is not None
            or detector_outcome["sanitized_error_category"] is not None
            or type(detector_outcome["public_result_identity"]) is not str
            or DIGEST.fullmatch(detector_outcome["public_result_identity"]) is None
            or detector_outcome["witness_identity"] is not None
            or value["blocked_class"] is not None
        ):
            raise SemanticTextureOperationalServerError(
                "passed detector result boundary drifted"
            )
        return value
    if detector_outcome["status"] != "blocked" or detector_outcome["blocked_class"] not in BLOCKED_CLASSES or detector_outcome["sanitized_error_category"] != detector_outcome["blocked_class"] or detector_outcome["public_result_identity"] is not None or detector_outcome["witness_identity"] is not None:
        raise SemanticTextureOperationalServerError("blocked detector result boundary drifted")
    expected_blocked_class = (
        write_outcome["blocked_class"]
        if write_outcome["status"] == "blocked"
        else detector_outcome["blocked_class"]
    )
    if value["blocked_class"] != expected_blocked_class:
        raise SemanticTextureOperationalServerError(
            "result blocked classification drifted"
        )
    return value


def finalize_semantic_texture_operational_preflight_delivery(
    result: object,
    *,
    output_root: str | Path,
) -> tuple[int, dict[str, object]]:
    """Create result, ZIP, external receipt, and the final completion marker."""

    result_value = _validated_result(result)
    run_id = result_value["run_id"]
    result_identity = result_value["result_identity"]
    root = Path(output_root).resolve()
    if root.exists():
        raise SemanticTextureOperationalServerError(
            "output root must be absent"
        )
    root.mkdir(parents=True)
    result_path = root / RESULT_FILENAME
    archive_path = root / f"semantic_texture_operational_{run_id}.zip"
    receipt_path = root / RECEIPT_FILENAME
    delivery_completion_checksums_path = (
        root / DELIVERY_COMPLETION_CHECKSUMS_FILENAME
    )
    result_blob = _canonical_bytes(result_value)
    if len(result_blob) > MAX_RESULT_BYTES:
        raise SemanticTextureOperationalServerError(
            "operational result exceeds its bound"
        )
    with result_path.open("xb") as handle:
        handle.write(result_blob)

    with ZipFile(archive_path, mode="x", compression=ZIP_STORED) as archive:
        archive.writestr(_zip_info(RESULT_FILENAME), result_blob)
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
        "asset_authority_status": result_value["asset_authority_status"],
        "asset_bundle_digest": result_value["asset_bundle_digest"],
        "blocked_class": result_value["blocked_class"],
        "candidate_promoted": False,
        "configuration_digest": result_value["configuration_digest"],
        "diagnostic_only": True,
        "formal_tau_created": False,
        "model_id": result_value["model_id"],
        "model_revision": result_value["model_revision"],
        "observed_repository_revision": result_value[
            "observed_repository_revision"
        ],
        "pre_execution_stage": result_value["pre_execution_stage"],
        "semantic_runtime_initialization_step": result_value[
            "semantic_runtime_initialization_step"
        ],
        "profile_id": result_value["profile_id"],
        "result_filename": result_path.name,
        "result_identity": result_identity,
        "result_sha256": sha256(result_blob).hexdigest(),
        "run_id": run_id,
        "science_started": False,
        "scientific_claims_supported": False,
        "scientific_unit_count": 0,
        "status": result_value["status"],
    }
    with receipt_path.open("xb") as handle:
        receipt_blob = _canonical_bytes(receipt)
        handle.write(receipt_blob)
    delivery_completion_checksums_blob = (
        f"{sha256(result_blob).hexdigest()}  {result_path.name}\n"
        f"{archive_sha256}  {archive_path.name}\n"
        f"{sha256(receipt_blob).hexdigest()}  {receipt_path.name}\n"
    ).encode("ascii")
    with delivery_completion_checksums_path.open("xb") as handle:
        handle.write(delivery_completion_checksums_blob)
    write_outcome, detector_outcome = result_value["unit_outcomes"]
    expected_operational_success = (
        write_outcome["started"] is True
        and write_outcome["status"] == "passed"
        and detector_outcome["started"] is True
        and detector_outcome["status"] == "passed"
        and result_value["status"] == "passed"
    )
    return (0 if expected_operational_success else 2), {
        **receipt,
        "receipt_filename": receipt_path.name,
        "receipt_sha256": _sha256_file(receipt_path),
        "sha256sums_filename": delivery_completion_checksums_path.name,
        "sha256sums_sha256": sha256(
            delivery_completion_checksums_blob
        ).hexdigest(),
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
                        "sha256sums_completion_marker",
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
