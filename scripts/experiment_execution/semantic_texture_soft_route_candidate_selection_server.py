"""Create-only bounded delivery for soft-route candidate selection."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from experiments.protocol.semantic_texture_soft_route_mechanism_validation import (
    ARMS,
    ATTACKS,
    CLUSTER_COUNT,
    PROTOCOL_ID,
    SELECTION_MANIFEST_DIGEST,
    SELECTION_ROLE,
)
from experiments.runners.semantic_texture_soft_route_mechanism_validation import SoftRouteMechanismSplitResult


RESULT_FILENAME = "semantic_texture_soft_route_candidate_selection_result.json"
RECEIPT_FILENAME = "semantic_texture_soft_route_candidate_selection_receipt.json"
SELECTION_FILENAME = "semantic_texture_soft_route_selection_artifact.json"
CHECKSUMS_FILENAME = "SHA256SUMS"
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")


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


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError(reason)


def _fixed_failed_result_layout(
    result: SoftRouteMechanismSplitResult,
) -> tuple[int, ...]:
    """Authenticate the runner's frozen selection denominator and role layout."""

    _require(
        len(result.generations) == CLUSTER_COUNT * len(ARMS),
        "selection generation denominator is invalid",
    )
    _require(
        len(result.records) == CLUSTER_COUNT * 12,
        "selection detector denominator is invalid",
    )
    identities: list[tuple[str, str]] = []
    for ordinal in range(CLUSTER_COUNT):
        chunk = result.generations[
            ordinal * len(ARMS) : (ordinal + 1) * len(ARMS)
        ]
        identity = (chunk[0].source_cluster_id, chunk[0].image_lineage_digest)
        _require(
            all(type(value) is str and DIGEST.fullmatch(value) for value in identity),
            "selection generation identity is invalid",
        )
        identities.append(identity)
        for arm_id, record in zip(ARMS, chunk, strict=True):
            _require(
                type(record.record_attempt_index) is int
                and record.record_attempt_index == 1
                and (record.source_cluster_id, record.image_lineage_digest) == identity
                and record.arm_id == arm_id,
                "selection generation layout is invalid",
            )
            _require(
                record.paired_rgb8_mse is None
                or (
                    type(record.paired_rgb8_mse) is float
                    and isfinite(record.paired_rgb8_mse)
                ),
                "selection generation value is invalid",
            )
    _require(
        len(set(identities)) == CLUSTER_COUNT
        and len({identity[0] for identity in identities}) == CLUSTER_COUNT
        and len({identity[1] for identity in identities}) == CLUSTER_COUNT,
        "selection generation identity is duplicated",
    )

    primary_indexes: list[int] = []
    numeric_fields = (
        "hf_score", "lf_score", "hf_standardized_score",
        "lf_standardized_score", "max_standardized_score", "paired_rgb8_mse",
    )
    index = 0
    for ordinal, identity in enumerate(identities):
        for arm_id in ARMS:
            for attack_id in ATTACKS:
                roles = [("registered", None)]
                if arm_id == "semantic_texture_soft_routed":
                    roles.append(("wrong", ordinal))
                for key_role, wrong_key_index in roles:
                    record = result.records[index]
                    _require(
                        type(record.record_attempt_index) is int
                        and record.record_attempt_index == 1
                        and (record.source_cluster_id, record.image_lineage_digest)
                        == identity
                        and (record.arm_id, record.attack_id) == (arm_id, attack_id)
                        and (record.key_role, record.wrong_key_index)
                        == (key_role, wrong_key_index)
                        and (
                            wrong_key_index is None
                            or type(record.wrong_key_index) is int
                        ),
                        "selection detector layout is invalid",
                    )
                    _require(
                        all(
                            value is None
                            or (type(value) is float and isfinite(value))
                            for value in (getattr(record, field) for field in numeric_fields)
                        ),
                        "selection detector value is invalid",
                    )
                    if (arm_id, attack_id, key_role) == (
                        ARMS[0], ATTACKS[0], "registered"
                    ):
                        primary_indexes.append(index)
                    index += 1
    return tuple(primary_indexes)


def _bounded_failed_result_reason(
    result: SoftRouteMechanismSplitResult, primary_indexes: tuple[int, ...]
) -> str:
    """Accept one failed slot, completed prefix, and bounded unstarted tail."""

    generation_statuses = tuple(
        record.execution_status for record in result.generations
    )
    case_statuses = tuple(record.execution_status for record in result.records)
    failed_generations = tuple(
        index for index, status in enumerate(generation_statuses) if status == "failed"
    )
    failed_cases = tuple(
        index for index, status in enumerate(case_statuses) if status == "failed"
    )
    _require(
        len(failed_generations) + len(failed_cases) == 1,
        "selection failed slot is invalid",
    )
    if failed_generations:
        failed_index = failed_generations[0]
        _require(
            all(status == "completed" for status in generation_statuses[:failed_index])
            and all(status == "unstarted" for status in generation_statuses[failed_index + 1 :])
            and all(status == "unstarted" for status in case_statuses),
            "selection generation failure tail is invalid",
        )
        failed_record = result.generations[failed_index]
    else:
        _require(
            all(status == "completed" for status in generation_statuses),
            "selection detector failure generation prefix is invalid",
        )
        failed_index = failed_cases[0]
        primary_set = set(primary_indexes)
        _require(failed_index in primary_set, "selection detector failure slot is invalid")
        primary_statuses = tuple(case_statuses[index] for index in primary_indexes)
        primary_failed_ordinal = primary_indexes.index(failed_index)
        observation_failure = (
            all(status == "completed" for status in primary_statuses[:primary_failed_ordinal])
            and all(status == "unstarted" for status in primary_statuses[primary_failed_ordinal + 1 :])
        )
        calibration_failure = (
            primary_failed_ordinal == 0
            and all(status == "completed" for status in primary_statuses[1:])
        )
        _require(
            all(
                status == "unstarted"
                for index, status in enumerate(case_statuses)
                if index not in primary_set
            )
            and (observation_failure or calibration_failure),
            "selection detector failure tail is invalid",
        )
        failed_record = result.records[failed_index]
    failure_reason = failed_record.failure_reason
    _require(
        type(failure_reason) is str
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", failure_reason) is not None,
        "selection failure reason is invalid",
    )
    for record in (*result.generations, *result.records):
        _require(
            record.execution_status in {"completed", "failed", "unstarted"}
            and (
                record.execution_status == "failed" or record.failure_reason is None
            ),
            "selection record status is invalid",
        )
        if record.execution_status in {"failed", "unstarted"}:
            if hasattr(record, "hf_score"):
                evidence = (
                    record.hf_score, record.lf_score,
                    record.hf_standardized_score, record.lf_standardized_score,
                    record.max_standardized_score, record.paired_rgb8_mse,
                    record.materialization_replay_identity, record.budget_identity,
                )
                retained_primary = (
                    record.execution_status == "failed"
                    and all(value is not None for value in evidence[:2])
                    and all(value is None for value in evidence[2:5])
                    and evidence[5] is not None
                    and evidence[6:] == (None, None)
                )
                _require(
                    all(value is None for value in evidence) or retained_primary,
                    "selection detector failure evidence is inconsistent",
                )
            else:
                _require(
                    (
                        record.materialization_replay_identity,
                        record.budget_identity,
                        record.paired_rgb8_mse,
                    ) == (None, None, None),
                    "selection generation failure evidence is inconsistent",
                )
        elif hasattr(record, "hf_score"):
            _require(
                record.hf_score is not None
                and record.lf_score is not None
                and record.paired_rgb8_mse is not None
                and record.hf_standardized_score is None
                and record.lf_standardized_score is None
                and record.max_standardized_score is None
                and record.materialization_replay_identity is None
                and record.budget_identity is None,
                "selection detector prefix evidence is inconsistent",
            )
        else:
            replay_identity = record.materialization_replay_identity
            budget_identity = record.budget_identity
            _require(
                record.paired_rgb8_mse is not None
                and (
                    (record.arm_id == ARMS[0] and (replay_identity, budget_identity) == (None, None))
                    or (
                        record.arm_id != ARMS[0]
                        and all(
                            type(value) is str and value
                            for value in (replay_identity, budget_identity)
                        )
                    )
                ),
                "selection generation prefix evidence is inconsistent",
            )
    return failure_reason


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
        or expected_role != SELECTION_ROLE
        or result.protocol_id != PROTOCOL_ID
        or result.manifest_digest != SELECTION_MANIFEST_DIGEST
        or result.role_id != SELECTION_ROLE
        or RUN_ID.fullmatch(run_id) is None
        or REVISION.fullmatch(observed_repository_revision) is None
    ):
        raise SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError("selection result identity is invalid")
    if (
        len(result.generations) != CLUSTER_COUNT * len(ARMS)
        or len(result.records) != CLUSTER_COUNT * 12
        or result.diagnostic_only is not True
        or result.science_started is not False
        or type(result.scientific_unit_count) is not int
        or result.scientific_unit_count != 0
        or type(result.passed) is not bool
        or result.candidate_promoted is not False
        or result.formal_tau_created is not False
    ):
        raise SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError("selection result boundary is invalid")
    completed = all(
        record.execution_status == "completed"
        for record in (*result.generations, *result.records)
    )
    runner_failed = result.passed is False and result.provisional_calibration is None
    if completed:
        if result.provisional_calibration is None or result.blocked_class is not None:
            raise SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError(
                "selection completed result boundary is invalid"
            )
        failure_reason = None
    elif runner_failed:
        if result.blocked_class != "implementation_blocked":
            raise SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError(
                "selection failed result classification is invalid"
            )
        primary_indexes = _fixed_failed_result_layout(result)
        failure_reason = _bounded_failed_result_reason(result, primary_indexes)
    else:
        raise SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError(
            "selection result execution boundary is invalid"
        )
    root = Path(output_root).resolve()
    if root.exists():
        raise SemanticTextureSoftRouteSoftRouteMechanismSelectionDeliveryError("selection output already exists")
    root.mkdir(parents=True)
    authority = {
        "protocol_id": result.protocol_id,
        "selection_manifest_digest": result.manifest_digest,
        "candidate_selection_passed": result.passed,
        "diagnostic_only": True,
        "science_started": False,
        "scientific_unit_count": 0,
        "candidate_promoted": False,
        "formal_tau_created": False,
        "formal_fpr_created": False,
    }
    if runner_failed:
        artifact_filename_value = None
        artifact_sha256 = None
        result_authority = {
            **authority,
            "provisional_calibration": None,
            "provisional_calibration_digest": None,
            "selection_artifact_filename": None,
            "selection_artifact_sha256": None,
        }
    else:
        artifact = {
            **authority,
            "provisional_calibration": asdict(result.provisional_calibration),
        }
        artifact_blob = _blob(artifact)
        artifact_filename_value = artifact_filename
        artifact_sha256 = sha256(artifact_blob).hexdigest()
        result_authority = artifact
    result_value = {
        **result_authority,
        "observed_repository_revision": observed_repository_revision,
        "run_id": run_id,
        "records": [asdict(record) for record in result.records],
        "generations": [asdict(record) for record in result.generations],
        "status": "passed" if result.passed else "blocked",
        "blocked_class": (
            result.blocked_class
            if runner_failed
            else None if result.passed else "integrity_blocked"
        ),
    }
    if runner_failed:
        result_value["failure_reason"] = failure_reason
    result_blob = _blob(result_value)
    result_path = root / result_filename
    if not runner_failed:
        _write_create_only_blob(root / artifact_filename, artifact_blob)
    _write_create_only_blob(result_path, result_blob)
    archive_path = root / f"{archive_prefix}_{run_id}.zip"
    with ZipFile(archive_path, "x", compression=ZIP_STORED) as archive:
        if not runner_failed:
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
        "selection_artifact_filename": artifact_filename_value,
        "selection_artifact_sha256": artifact_sha256,
        "selection_manifest_digest": result.manifest_digest,
        "status": result_value["status"],
    }
    if runner_failed:
        receipt.update(
            {
                "blocked_class": result_value["blocked_class"],
                "failure_reason": failure_reason,
                "provisional_calibration_digest": None,
            }
        )
    receipt_blob = _blob(receipt)
    _write_create_only_blob(root / receipt_filename, receipt_blob)
    sum_rows = []
    if not runner_failed:
        sum_rows.append(f"{artifact_sha256}  {artifact_filename}\n")
    sum_rows.extend(
        (
            f"{sha256(result_blob).hexdigest()}  {result_filename}\n",
            f"{receipt['archive_sha256']}  {archive_path.name}\n",
            f"{sha256(receipt_blob).hexdigest()}  {receipt_filename}\n",
        )
    )
    sums = "".join(sum_rows).encode("ascii")
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
