"""Create-only persistence and recovery for development exploration workers.

The persistent root is authoritative only through independently verifiable
``COMMITTED`` markers.  A Drive/FUSE rename or mutable ledger is never treated
as proof that a scientific unit completed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from typing import Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from experiments.protocol.development_exploration import DevelopmentStudyUnit
from experiments.protocol.development_records import (
    ATTEMPT_DISPOSITIONS,
    DEVELOPMENT_RECORD_MEMBER_PATH,
    OPERATIONAL_RECORD_KIND,
    OPERATIONAL_RECORD_MEMBER_PATH,
    OPERATIONAL_RECORD_PHASES,
    ROUTING_REFERENCE_RECORD_KIND,
    ROUTING_REFERENCE_RECORD_MEMBER_PATH,
    DevelopmentRecordError,
    DevelopmentOperationalRecord,
    DevelopmentRoutingReferenceRecord,
    DevelopmentScientificRecord,
    validate_record_against_intent,
)
from experiments.protocol.internal_splits import AnalysisUnitIdentity


SCHEMA_VERSION = "ceg_wm_development_worker_persistence_v1"
MANIFEST_SCHEMA_VERSION = "ceg_wm_development_artifact_manifest_v1"
RESULT_SCHEMA_VERSION = "ceg_wm_development_committed_marker_v2"
DIAGNOSTIC_SCHEMA_VERSION = "ceg_wm_development_session_receipt_v1"
IDENTITY_SCHEMA_VERSION = "ceg_wm_development_single_writer_lease_v1"
MAXIMUM_ATTEMPTS = 3
SOFT_STOP_SECONDS = 21 * 60 * 60
HARD_SESSION_CAP_SECONDS = 24 * 60 * 60
GPU_MIX_POLICY = (
    "gpu_models_may_resume_identical_units_but_latency_and_cost_are_"
    "summarized_within_gpu_model_only"
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY = re.compile(r"^[a-z][a-z0-9_]*$")
_EXECUTABLE_DESERIALIZATION_SUFFIXES = frozenset(
    {
        ".dill",
        ".exe",
        ".joblib",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".py",
        ".pyc",
        ".sh",
        ".so",
    }
)


class DevelopmentPersistenceError(RuntimeError):
    """Persistent identity, create-only, or recovery verification failed."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DevelopmentPersistenceError("payload is not canonical JSON data") from exc


def canonical_digest(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def development_unit_roster_digest(
    units: Sequence[DevelopmentStudyUnit],
) -> str:
    if not units or any(type(item) is not DevelopmentStudyUnit for item in units):
        raise DevelopmentPersistenceError("development unit roster is invalid")
    return _protocol_payload_digest(tuple(asdict(item) for item in units))


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_from_epoch(epoch_seconds: int) -> str:
    if type(epoch_seconds) is not int or epoch_seconds < 0:
        raise DevelopmentPersistenceError("epoch time is invalid")
    return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_strict_utc(value: object, role: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise DevelopmentPersistenceError(f"{role} is not strict UTC")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise DevelopmentPersistenceError(f"{role} is not strict UTC") from exc
    if parsed.tzinfo != timezone.utc or parsed.isoformat().replace("+00:00", "Z") != value:
        raise DevelopmentPersistenceError(f"{role} is not canonical strict UTC")
    return parsed


def _protocol_payload_digest(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DevelopmentPersistenceError("protocol payload is not canonical JSON data") from exc
    return sha256(payload).hexdigest()


def _identity(value: object, role: str) -> str:
    if type(value) is not str or _IDENTITY.fullmatch(value) is None:
        raise DevelopmentPersistenceError(f"{role} is not a stable identity")
    return value


def _digest(value: object, role: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise DevelopmentPersistenceError(f"{role} is not a SHA-256 digest")
    return value


def _regular_directory(path: Path, role: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_absolute():
        raise DevelopmentPersistenceError(f"{role} must be absolute")
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir() or resolved.is_symlink():
        raise DevelopmentPersistenceError(f"{role} must be a regular directory")
    return resolved


def _safe_member_path(value: object) -> str:
    if type(value) is not str or not value:
        raise DevelopmentPersistenceError("bundle member path is empty")
    if "\\" in value:
        raise DevelopmentPersistenceError("bundle member path uses backslashes")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DevelopmentPersistenceError("bundle member path is unsafe")
    if path.parts[0].startswith("~"):
        raise DevelopmentPersistenceError("bundle member path is unsafe")
    if path.suffix.casefold() in _EXECUTABLE_DESERIALIZATION_SUFFIXES:
        raise DevelopmentPersistenceError(
            "bundle member uses executable deserialization format"
        )
    return path.as_posix()


def _create_only(path: Path, payload: bytes) -> None:
    if path.parent.is_symlink():
        raise DevelopmentPersistenceError("create-only parent cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise DevelopmentPersistenceError(f"create-only conflict: {path.name}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def _read_canonical_json(path: Path, role: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise DevelopmentPersistenceError(f"{role} must be a regular file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevelopmentPersistenceError(f"{role} is unreadable") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise DevelopmentPersistenceError(f"{role} is not canonical JSON")
    return value


def _reject_secrets(payloads: Sequence[bytes], raw_secret_values: Sequence[str]) -> None:
    for secret in raw_secret_values:
        if type(secret) is not str or not secret:
            raise DevelopmentPersistenceError("raw secret scan value is invalid")
        encoded = secret.encode("utf-8")
        if any(encoded in payload for payload in payloads):
            raise DevelopmentPersistenceError("raw secret material reached persisted bytes")


def _registered_unit_id(unit_index: int) -> str:
    return f"development_unit_{unit_index:04d}"


def _study_unit_payload(unit: DevelopmentStudyUnit) -> dict[str, object]:
    return {
        "unit_id": _registered_unit_id(unit.unit_index),
        "unit_index": unit.unit_index,
        "shard_id": unit.phase,
        "phase": unit.phase,
        "responsibility_id": unit.responsibility_id,
        "source_cluster_ordinal": unit.source_cluster_ordinal,
        "content_branch_id": unit.content_branch_id,
        "geometry_case_id": unit.geometry_case_id,
        "maximum_record_attempts": unit.maximum_record_attempts,
        "maximum_duration_seconds": unit.maximum_duration_seconds,
    }


def _unit_intent_binding_payload(intent: "UnitIntent") -> dict[str, object]:
    return {
        "unit_id": intent.unit_id,
        "unit_index": intent.unit_index,
        "shard_id": intent.shard_id,
        "phase": intent.phase,
        "responsibility_id": intent.responsibility_id,
        "source_cluster_ordinal": intent.source_cluster_ordinal,
        "content_branch_id": intent.content_branch_id,
        "geometry_case_id": intent.geometry_case_id,
        "maximum_record_attempts": intent.maximum_record_attempts,
        "maximum_duration_seconds": intent.maximum_duration_seconds,
        "analysis_unit_identity": intent.analysis_unit_identity,
        "analysis_unit_identity_digest": intent.analysis_unit_identity_digest,
        "scientific_question_id": intent.scientific_question_id,
        "development_case_id": intent.development_case_id,
        "candidate_identity": intent.candidate_identity,
        "candidate_config_digest": intent.candidate_config_digest,
    }


@dataclass(frozen=True, slots=True)
class FrozenWorkerIdentity:
    revision: str
    protocol_digest: str
    execution_intent_authority_digest: str
    input_manifest_digest: str
    candidate_config_digest: str
    unit_roster_digest: str

    def validate(self) -> None:
        if type(self.revision) is not str or _REVISION.fullmatch(self.revision) is None:
            raise DevelopmentPersistenceError("worker revision must be a full Git SHA")
        for role in (
            "protocol_digest",
            "execution_intent_authority_digest",
            "input_manifest_digest",
            "candidate_config_digest",
            "unit_roster_digest",
        ):
            _digest(getattr(self, role), role)

    def digest(self) -> str:
        self.validate()
        return canonical_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentUnitBinding:
    unit_id: str
    unit_index: int
    phase: str
    responsibility_id: str
    source_cluster_ordinal: int
    content_branch_id: str
    geometry_case_id: str
    maximum_record_attempts: int
    maximum_duration_seconds: int
    analysis_unit_identity: AnalysisUnitIdentity
    analysis_unit_identity_digest: str
    scientific_question_id: str
    development_case_id: str
    candidate_identity: str
    candidate_config_digest: str
    unit_descriptor_digest: str

    def descriptor_payload(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "unit_index": self.unit_index,
            "shard_id": self.phase,
            "phase": self.phase,
            "responsibility_id": self.responsibility_id,
            "source_cluster_ordinal": self.source_cluster_ordinal,
            "content_branch_id": self.content_branch_id,
            "geometry_case_id": self.geometry_case_id,
            "maximum_record_attempts": self.maximum_record_attempts,
            "maximum_duration_seconds": self.maximum_duration_seconds,
            "analysis_unit_identity": asdict(self.analysis_unit_identity),
            "analysis_unit_identity_digest": self.analysis_unit_identity_digest,
            "scientific_question_id": self.scientific_question_id,
            "development_case_id": self.development_case_id,
            "candidate_identity": self.candidate_identity,
            "candidate_config_digest": self.candidate_config_digest,
        }

    def study_unit(self) -> DevelopmentStudyUnit:
        return DevelopmentStudyUnit(
            unit_index=self.unit_index,
            phase=self.phase,
            responsibility_id=self.responsibility_id,
            source_cluster_ordinal=self.source_cluster_ordinal,
            content_branch_id=self.content_branch_id,
            geometry_case_id=self.geometry_case_id,
            maximum_record_attempts=self.maximum_record_attempts,
            maximum_duration_seconds=self.maximum_duration_seconds,
        )

    def validate(self) -> None:
        for role, value in (
            ("registered unit_id", self.unit_id),
            ("registered phase", self.phase),
            ("registered responsibility_id", self.responsibility_id),
            ("registered content_branch_id", self.content_branch_id),
            ("registered geometry_case_id", self.geometry_case_id),
            ("registered scientific_question_id", self.scientific_question_id),
            ("registered development_case_id", self.development_case_id),
            ("registered candidate_identity", self.candidate_identity),
        ):
            _identity(value, role)
        if type(self.unit_index) is not int or self.unit_index < 0:
            raise DevelopmentPersistenceError("registered unit index is invalid")
        if self.unit_id != _registered_unit_id(self.unit_index):
            raise DevelopmentPersistenceError("registered unit identity drifted")
        if type(self.source_cluster_ordinal) is not int or self.source_cluster_ordinal < 0:
            raise DevelopmentPersistenceError("registered source cluster ordinal is invalid")
        if type(self.maximum_record_attempts) is not int or not 1 <= self.maximum_record_attempts <= MAXIMUM_ATTEMPTS:
            raise DevelopmentPersistenceError("registered attempt limit is invalid")
        if type(self.maximum_duration_seconds) is not int or self.maximum_duration_seconds < 1:
            raise DevelopmentPersistenceError("registered duration limit is invalid")
        if type(self.analysis_unit_identity) is not AnalysisUnitIdentity:
            raise DevelopmentPersistenceError("analysis unit identity exact type is required")
        if self.analysis_unit_identity.validate():
            raise DevelopmentPersistenceError("analysis unit identity is invalid")
        _digest(self.analysis_unit_identity_digest, "analysis unit identity")
        if self.analysis_unit_identity_digest != canonical_digest(
            asdict(self.analysis_unit_identity)
        ):
            raise DevelopmentPersistenceError("analysis unit identity digest drifted")
        _digest(self.candidate_config_digest, "registered candidate config")
        _digest(self.unit_descriptor_digest, "registered unit descriptor")
        if self.unit_descriptor_digest != canonical_digest(self.descriptor_payload()):
            raise DevelopmentPersistenceError("registered unit descriptor digest drifted")


def create_frozen_development_unit_binding(
    unit: DevelopmentStudyUnit,
    *,
    analysis_unit_identity: AnalysisUnitIdentity,
    scientific_question_id: str,
    development_case_id: str,
    candidate_identity: str,
    candidate_config_digest: str,
) -> FrozenDevelopmentUnitBinding:
    if type(unit) is not DevelopmentStudyUnit:
        raise DevelopmentPersistenceError("development study unit exact type is required")
    if type(analysis_unit_identity) is not AnalysisUnitIdentity:
        raise DevelopmentPersistenceError("analysis unit identity exact type is required")
    payload = {
        **_study_unit_payload(unit),
        "analysis_unit_identity": asdict(analysis_unit_identity),
        "analysis_unit_identity_digest": canonical_digest(
            asdict(analysis_unit_identity)
        ),
        "scientific_question_id": scientific_question_id,
        "development_case_id": development_case_id,
        "candidate_identity": candidate_identity,
        "candidate_config_digest": candidate_config_digest,
    }
    binding = FrozenDevelopmentUnitBinding(
        unit_id=payload["unit_id"],
        unit_index=payload["unit_index"],
        phase=payload["shard_id"],
        responsibility_id=payload["responsibility_id"],
        source_cluster_ordinal=payload["source_cluster_ordinal"],
        content_branch_id=payload["content_branch_id"],
        geometry_case_id=payload["geometry_case_id"],
        maximum_record_attempts=payload["maximum_record_attempts"],
        maximum_duration_seconds=payload["maximum_duration_seconds"],
        analysis_unit_identity=analysis_unit_identity,
        analysis_unit_identity_digest=payload["analysis_unit_identity_digest"],
        scientific_question_id=scientific_question_id,
        development_case_id=development_case_id,
        candidate_identity=candidate_identity,
        candidate_config_digest=candidate_config_digest,
        unit_descriptor_digest="0" * 64,
    )
    binding = FrozenDevelopmentUnitBinding(
        **{
            **asdict(binding),
            "analysis_unit_identity": analysis_unit_identity,
            "unit_descriptor_digest": canonical_digest(binding.descriptor_payload()),
        }
    )
    binding.validate()
    return binding


@dataclass(frozen=True, slots=True)
class PersistentLease:
    schema_version: str
    run_id: str
    session_id: str
    fencing_token: int
    acquired_at_utc: str
    expires_at_epoch_seconds: int
    worker_identity_digest: str

    def validate(self) -> None:
        if self.schema_version != IDENTITY_SCHEMA_VERSION:
            raise DevelopmentPersistenceError("lease schema drifted")
        _identity(self.run_id, "lease run_id")
        _identity(self.session_id, "lease session_id")
        if type(self.fencing_token) is not int or self.fencing_token < 1:
            raise DevelopmentPersistenceError("fencing token is invalid")
        started_at_epoch_seconds = int(
            _parse_strict_utc(self.acquired_at_utc, "lease acquired_at_utc").timestamp()
        )
        if self.acquired_at_utc != _utc_from_epoch(started_at_epoch_seconds):
            raise DevelopmentPersistenceError("lease session start UTC drifted")
        if type(self.expires_at_epoch_seconds) is not int:
            raise DevelopmentPersistenceError("lease expiry is invalid")
        if not (
            started_at_epoch_seconds < self.expires_at_epoch_seconds
            < started_at_epoch_seconds + HARD_SESSION_CAP_SECONDS
        ):
            raise DevelopmentPersistenceError("lease duration violates session cap")
        _digest(self.worker_identity_digest, "lease worker identity")


@dataclass(frozen=True, slots=True)
class UnitIntent:
    schema_version: str
    protocol_digest: str
    revision: str
    run_id: str
    shard_id: str
    unit_id: str
    unit_index: int
    phase: str
    responsibility_id: str
    source_cluster_ordinal: int
    content_branch_id: str
    geometry_case_id: str
    maximum_record_attempts: int
    maximum_duration_seconds: int
    unit_roster_digest: str
    analysis_unit_identity: dict[str, object]
    analysis_unit_identity_digest: str
    scientific_question_id: str
    development_case_id: str
    candidate_identity: str
    candidate_config_digest: str
    unit_descriptor_digest: str
    attempt_index: int
    session_id: str
    fencing_token: int
    worker_identity_digest: str
    parent_attempt_intent_digest: str | None
    created_at_utc: str

    def payload(self) -> dict[str, object]:
        return asdict(self)

    def digest(self) -> str:
        return canonical_digest(self.payload())

    def validate(self, worker_identity: FrozenWorkerIdentity) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise DevelopmentPersistenceError("unit intent schema drifted")
        if self.protocol_digest != worker_identity.protocol_digest:
            raise DevelopmentPersistenceError("unit intent protocol drifted")
        if self.revision != worker_identity.revision:
            raise DevelopmentPersistenceError("unit intent revision drifted")
        for role, value in (
            ("intent run_id", self.run_id),
            ("intent shard_id", self.shard_id),
            ("intent unit_id", self.unit_id),
            ("intent session_id", self.session_id),
            ("intent phase", self.phase),
            ("intent responsibility_id", self.responsibility_id),
            ("intent content_branch_id", self.content_branch_id),
            ("intent geometry_case_id", self.geometry_case_id),
            ("intent scientific_question_id", self.scientific_question_id),
            ("intent development_case_id", self.development_case_id),
            ("intent candidate_identity", self.candidate_identity),
        ):
            _identity(value, role)
        if type(self.unit_index) is not int or self.unit_index < 0:
            raise DevelopmentPersistenceError("unit intent index is invalid")
        if type(self.source_cluster_ordinal) is not int or self.source_cluster_ordinal < 0:
            raise DevelopmentPersistenceError("unit intent source cluster is invalid")
        if type(self.maximum_record_attempts) is not int or self.maximum_record_attempts < 1:
            raise DevelopmentPersistenceError("unit intent record attempt limit is invalid")
        if type(self.maximum_duration_seconds) is not int or self.maximum_duration_seconds < 1:
            raise DevelopmentPersistenceError("unit intent duration limit is invalid")
        if type(self.attempt_index) is not int or not 0 <= self.attempt_index < MAXIMUM_ATTEMPTS:
            raise DevelopmentPersistenceError("unit intent attempt is invalid")
        if self.attempt_index >= self.maximum_record_attempts:
            raise DevelopmentPersistenceError("unit intent exceeds registered attempt limit")
        if type(self.fencing_token) is not int or self.fencing_token < 1:
            raise DevelopmentPersistenceError("unit intent fencing token is invalid")
        _digest(self.worker_identity_digest, "unit intent worker identity")
        _digest(self.unit_roster_digest, "unit intent roster")
        _digest(self.analysis_unit_identity_digest, "unit intent analysis identity")
        _digest(self.candidate_config_digest, "unit intent candidate config")
        _digest(self.unit_descriptor_digest, "unit intent descriptor")
        if self.unit_roster_digest != worker_identity.unit_roster_digest:
            raise DevelopmentPersistenceError("unit intent roster drifted")
        if type(self.analysis_unit_identity) is not dict:
            raise DevelopmentPersistenceError("unit intent analysis identity is invalid")
        try:
            analysis_identity = AnalysisUnitIdentity(**self.analysis_unit_identity)
        except TypeError as exc:
            raise DevelopmentPersistenceError("unit intent analysis identity schema is invalid") from exc
        if analysis_identity.validate():
            raise DevelopmentPersistenceError("unit intent analysis identity is invalid")
        if self.analysis_unit_identity_digest != canonical_digest(
            asdict(analysis_identity)
        ):
            raise DevelopmentPersistenceError("unit intent analysis identity digest drifted")
        if self.unit_descriptor_digest != canonical_digest(
            _unit_intent_binding_payload(self)
        ):
            raise DevelopmentPersistenceError("unit intent descriptor digest drifted")
        if self.worker_identity_digest != worker_identity.digest():
            raise DevelopmentPersistenceError("unit intent worker identity drifted")
        if self.attempt_index == 0 and self.parent_attempt_intent_digest is not None:
            raise DevelopmentPersistenceError("initial attempt cannot have a parent")
        if self.attempt_index > 0:
            _digest(self.parent_attempt_intent_digest, "parent attempt intent")
        _parse_strict_utc(self.created_at_utc, "unit intent created_at_utc")


def _validate_routing_reference_record_against_intent(
    record: DevelopmentRoutingReferenceRecord,
    intent: UnitIntent,
) -> None:
    try:
        record.validate()
    except DevelopmentRecordError as exc:
        raise DevelopmentPersistenceError(str(exc)) from exc
    if (
        intent.phase != ROUTING_REFERENCE_RECORD_KIND
        or intent.responsibility_id != "content_router"
        or record.run_id != intent.run_id
        or record.protocol_digest != intent.protocol_digest
        or record.method_code_revision != intent.revision
        or record.unit_index != intent.unit_index
        or record.phase != intent.phase
        or record.source_cluster_ordinal != intent.source_cluster_ordinal
        or record.candidate_config_digest != intent.candidate_config_digest
        or record.attempt_index != intent.attempt_index
        or record.retry_parent_intent_digest
        != intent.parent_attempt_intent_digest
        or record.maximum_duration_seconds != intent.maximum_duration_seconds
    ):
        raise DevelopmentPersistenceError(
            "routing reference record differs from its frozen intent"
        )


def _validate_operational_record_against_intent(
    record: DevelopmentOperationalRecord,
    intent: UnitIntent,
) -> None:
    try:
        record.validate()
    except DevelopmentRecordError as exc:
        raise DevelopmentPersistenceError(str(exc)) from exc
    if (
        intent.phase not in OPERATIONAL_RECORD_PHASES
        or record.run_id != intent.run_id
        or record.protocol_digest != intent.protocol_digest
        or record.method_code_revision != intent.revision
        or record.unit_index != intent.unit_index
        or record.phase != intent.phase
        or record.source_cluster_ordinal != intent.source_cluster_ordinal
        or record.candidate_config_digest != intent.candidate_config_digest
        or record.attempt_index != intent.attempt_index
        or record.retry_parent_intent_digest != intent.parent_attempt_intent_digest
        or record.maximum_duration_seconds != intent.maximum_duration_seconds
    ):
        raise DevelopmentPersistenceError(
            "operational record differs from its frozen intent"
        )


@dataclass(frozen=True, slots=True)
class ArtifactMember:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    schema_version: str
    members: tuple[ArtifactMember, ...]

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "members": [asdict(member) for member in self.members],
        }

    def digest(self) -> str:
        return canonical_digest(self.payload())


@dataclass(frozen=True, slots=True)
class CommittedUnit:
    schema_version: str
    protocol_digest: str
    revision: str
    run_id: str
    shard_id: str
    unit_id: str
    unit_index: int
    attempt_index: int
    session_id: str
    fencing_token: int
    intent_digest: str
    attempt_disposition: str
    record_kind: str
    record_id: str
    record_digest: str
    record_bytes: int
    actual_elapsed_seconds: float
    maximum_duration_seconds: int
    bundle_sha256: str
    bundle_bytes: int
    artifact_manifest_digest: str
    worker_identity_digest: str
    parent_attempt_intent_digest: str | None
    committed_at_utc: str

    def payload(self) -> dict[str, object]:
        return asdict(self)

    def digest(self) -> str:
        return canonical_digest(self.payload())


@dataclass(frozen=True, slots=True)
class InterruptedAttempt:
    unit_id: str
    attempt_index: int
    failure_class: str
    failure_reason: str
    retry_parent_intent_digest: str


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    committed_units: tuple[CommittedUnit, ...]
    interrupted_attempts: tuple[InterruptedAttempt, ...]
    next_attempt_by_unit: tuple[tuple[str, int], ...]
    ledger_digest: str


class DevelopmentSessionCursor:
    """Ephemeral cursor derived from one fully verified session-start recovery."""

    def __init__(
        self,
        *,
        store: "DevelopmentPersistentStore",
        lease: PersistentLease,
        recovery: RecoveryReport,
        verified_records: Sequence[
            tuple[
                CommittedUnit,
                DevelopmentScientificRecord
                | DevelopmentOperationalRecord
                | DevelopmentRoutingReferenceRecord,
            ]
        ],
    ) -> None:
        self._store = store
        self._lease = lease
        self._committed_units = list(recovery.committed_units)
        self._records_by_attempt: dict[
            tuple[str, int],
            DevelopmentScientificRecord
            | DevelopmentOperationalRecord
            | DevelopmentRoutingReferenceRecord,
        ] = {
            (marker.unit_id, marker.attempt_index): record
            for marker, record in verified_records
        }
        self._routing_reference_records = {
            record.unit_index: record
            for marker, record in verified_records
            if type(record) is DevelopmentRoutingReferenceRecord
            and marker.attempt_disposition == "success"
        }
        self._operational_records = {
            record.unit_index: record
            for marker, record in verified_records
            if type(record) is DevelopmentOperationalRecord
            and marker.attempt_disposition == "success"
        }
        self._initial_committed_count = len(recovery.committed_units)
        self._open_intent: UnitIntent | None = None
        self._closed = False
        (
            self._next_unit_index,
            self._next_attempt_index,
            self._parent_attempt_intent_digest,
        ) = store._next_claim_from_recovery(recovery)

    @property
    def initial_committed_count(self) -> int:
        return self._initial_committed_count

    @property
    def next_unit_index(self) -> int:
        return self._next_unit_index

    @property
    def committed_units(self) -> tuple[CommittedUnit, ...]:
        return tuple(self._committed_units)

    @property
    def routing_reference_records(
        self,
    ) -> tuple[DevelopmentRoutingReferenceRecord, ...]:
        return tuple(
            self._routing_reference_records[index]
            for index in sorted(self._routing_reference_records)
        )

    @property
    def terminal_routing_reference_records(
        self,
    ) -> tuple[DevelopmentRoutingReferenceRecord, ...]:
        """Return each verified routing unit's latest non-retry terminal record."""

        latest_by_unit: dict[str, CommittedUnit] = {}
        for marker in self._committed_units:
            if marker.record_kind == ROUTING_REFERENCE_RECORD_KIND:
                latest_by_unit[marker.unit_id] = marker
        records: list[DevelopmentRoutingReferenceRecord] = []
        for marker in sorted(
            latest_by_unit.values(), key=lambda item: item.unit_index
        ):
            if marker.attempt_disposition == "retryable_resource_failure":
                continue
            record = self._records_by_attempt.get(
                (marker.unit_id, marker.attempt_index)
            )
            if type(record) is not DevelopmentRoutingReferenceRecord:
                raise DevelopmentPersistenceError(
                    "session cursor lacks its verified routing reference record"
                )
            records.append(record)
        return tuple(records)

    @property
    def operational_records(self) -> tuple[DevelopmentOperationalRecord, ...]:
        return tuple(
            self._operational_records[index]
            for index in sorted(self._operational_records)
        )

    @property
    def terminal_scientific_evidence(
        self,
    ) -> tuple[tuple[DevelopmentScientificRecord, CommittedUnit], ...]:
        latest_by_unit: dict[str, CommittedUnit] = {}
        for marker in self._committed_units:
            if marker.record_kind == "development_scientific_record":
                latest_by_unit[marker.unit_id] = marker
        evidence: list[tuple[DevelopmentScientificRecord, CommittedUnit]] = []
        for marker in sorted(
            latest_by_unit.values(), key=lambda item: item.unit_index
        ):
            if marker.attempt_disposition == "retryable_resource_failure":
                continue
            record = self._records_by_attempt.get(
                (marker.unit_id, marker.attempt_index)
            )
            if type(record) is not DevelopmentScientificRecord:
                raise DevelopmentPersistenceError(
                    "session cursor lacks its verified scientific record"
                )
            evidence.append((record, marker))
        return tuple(evidence)


@dataclass(frozen=True, slots=True)
class SessionReceipt:
    schema_version: str
    session_id: str
    run_id: str
    started_at_utc: str
    ended_at_utc: str
    gpu_model: str
    cuda_identity: str
    environment_digest: str
    revision: str
    package_sha256: str
    walltime_seconds: float
    peak_vram_bytes: int
    termination_reason: str
    soft_stop_seconds: int
    hard_session_cap_seconds: int
    gpu_mix_policy: str
    committed_unit_ids: tuple[str, ...]
    public_secret_identity_digests: tuple[str, ...]


class DevelopmentPersistentStore:
    """Single-run content-addressed store for an interruptible worker."""

    def __init__(
        self,
        persistent_root: Path,
        *,
        run_id: str,
        worker_identity: FrozenWorkerIdentity,
        registered_unit_bindings: Sequence[FrozenDevelopmentUnitBinding],
    ) -> None:
        self.run_id = _identity(run_id, "run_id")
        worker_identity.validate()
        self.worker_identity = worker_identity
        self._registered_unit_bindings = self._validate_registered_unit_bindings(
            registered_unit_bindings
        )
        root = _regular_directory(Path(persistent_root), "persistent_root")
        self.run_root = _regular_directory(root / self.run_id, "run_root")
        for name in (
            "leases",
            "intents",
            "bundles",
            "markers",
            "receipts",
            "module_outcomes",
        ):
            _regular_directory(self.run_root / name, name)
        identity_path = self.run_root / "frozen_worker_identity.json"
        identity_bytes = canonical_json_bytes(asdict(worker_identity))
        if identity_path.exists():
            if identity_path.is_symlink() or identity_path.read_bytes() != identity_bytes:
                raise DevelopmentPersistenceError("frozen worker identity drifted")
        else:
            _create_only(identity_path, identity_bytes)

    def persist_verified_module_outcome(
        self,
        *,
        responsibility_id: str,
        outcome_record_id: str,
        payload: Mapping[str, object],
    ) -> Path:
        """Create or replay one outcome derived from verified COMMITTED records."""

        _identity(responsibility_id, "responsibility_id")
        _digest(outcome_record_id, "outcome_record_id")
        if type(payload) is not dict:
            raise DevelopmentPersistenceError("module outcome payload is invalid")
        if (
            payload.get("responsibility_id") != responsibility_id
            or payload.get("outcome_record_id") != outcome_record_id
        ):
            raise DevelopmentPersistenceError("module outcome identity drifted")
        destination = self.run_root / "module_outcomes" / (
            responsibility_id + ".json"
        )
        encoded = canonical_json_bytes(payload)
        if destination.exists():
            if destination.is_symlink() or destination.read_bytes() != encoded:
                raise DevelopmentPersistenceError(
                    "persisted verified module outcome drifted"
                )
            return destination
        _create_only(destination, encoded)
        if destination.read_bytes() != encoded:
            raise DevelopmentPersistenceError(
                "persisted verified module outcome replay failed"
            )
        return destination

    def _validate_registered_unit_bindings(
        self,
        registered_unit_bindings: Sequence[FrozenDevelopmentUnitBinding],
    ) -> dict[int, FrozenDevelopmentUnitBinding]:
        if isinstance(registered_unit_bindings, (str, bytes)) or not isinstance(
            registered_unit_bindings, Sequence
        ):
            raise DevelopmentPersistenceError("registered unit roster is invalid")
        bindings = tuple(registered_unit_bindings)
        if not bindings:
            raise DevelopmentPersistenceError("registered unit roster cannot be empty")
        if any(type(item) is not FrozenDevelopmentUnitBinding for item in bindings):
            raise DevelopmentPersistenceError("registered unit binding exact type is required")
        for item in bindings:
            item.validate()
        if tuple(item.unit_index for item in bindings) != tuple(range(len(bindings))):
            raise DevelopmentPersistenceError("registered unit indexes are not contiguous")
        units = tuple(item.study_unit() for item in bindings)
        payload = tuple(asdict(item) for item in units)
        if _protocol_payload_digest(payload) != self.worker_identity.unit_roster_digest:
            raise DevelopmentPersistenceError("registered unit roster digest drifted")
        if len({item.unit_id for item in bindings}) != len(bindings):
            raise DevelopmentPersistenceError("registered unit identities collide")
        if len({item.unit_descriptor_digest for item in bindings}) != len(bindings):
            raise DevelopmentPersistenceError("registered unit descriptors collide")
        return {item.unit_index: item for item in bindings}

    @property
    def registered_unit_bindings(self) -> tuple[FrozenDevelopmentUnitBinding, ...]:
        return tuple(
            self._registered_unit_bindings[index]
            for index in range(len(self._registered_unit_bindings))
        )

    def _verify_intent_registered_binding(self, intent: UnitIntent) -> None:
        if intent.run_id != self.run_id:
            raise DevelopmentPersistenceError("unit intent run identity drifted")
        registered = self._registered_unit_bindings.get(intent.unit_index)
        if registered is None:
            raise DevelopmentPersistenceError("unit intent is outside frozen roster")
        expected = registered.descriptor_payload()
        if _unit_intent_binding_payload(intent) != expected:
            raise DevelopmentPersistenceError("unit intent registered binding drifted")
        if intent.unit_descriptor_digest != registered.unit_descriptor_digest:
            raise DevelopmentPersistenceError("unit intent descriptor digest drifted")

    def acquire_lease(
        self,
        *,
        session_id: str,
        now_epoch_seconds: int,
        lease_duration_seconds: int,
    ) -> PersistentLease:
        _identity(session_id, "session_id")
        if type(now_epoch_seconds) is not int or type(lease_duration_seconds) is not int:
            raise DevelopmentPersistenceError("lease time values must be integers")
        if lease_duration_seconds <= 0 or lease_duration_seconds >= HARD_SESSION_CAP_SECONDS:
            raise DevelopmentPersistenceError("lease duration violates session cap")
        leases = self._load_leases()
        if leases:
            active = leases[-1]
            if active.expires_at_epoch_seconds > now_epoch_seconds:
                raise DevelopmentPersistenceError("run already has an active writer lease")
            token = active.fencing_token + 1
        else:
            token = 1
        lease = PersistentLease(
            schema_version=IDENTITY_SCHEMA_VERSION,
            run_id=self.run_id,
            session_id=session_id,
            fencing_token=token,
            acquired_at_utc=_utc_from_epoch(now_epoch_seconds),
            expires_at_epoch_seconds=now_epoch_seconds + lease_duration_seconds,
            worker_identity_digest=self.worker_identity.digest(),
        )
        lease.validate()
        _create_only(
            self.run_root / "leases" / f"fence_{token:08d}.json",
            canonical_json_bytes(asdict(lease)),
        )
        if self._load_leases()[-1] != lease:
            raise DevelopmentPersistenceError("lease lost fencing race")
        return lease

    def _next_claim_from_recovery(
        self,
        recovery: RecoveryReport,
    ) -> tuple[int, int, str | None]:
        latest_by_unit: dict[str, CommittedUnit] = {}
        for marker in recovery.committed_units:
            latest_by_unit[marker.unit_id] = marker
        terminal_indexes = tuple(
            sorted(
                marker.unit_index
                for marker in latest_by_unit.values()
                if marker.attempt_disposition != "retryable_resource_failure"
            )
        )
        if terminal_indexes != tuple(range(len(terminal_indexes))):
            raise DevelopmentPersistenceError(
                "terminal commits are not a frozen roster prefix"
            )
        next_attempts = dict(recovery.next_attempt_by_unit)
        if len(next_attempts) > 1:
            raise DevelopmentPersistenceError(
                "multiple frozen units are concurrently retryable"
            )
        if not next_attempts:
            return len(terminal_indexes), 0, None
        unit_id, attempt_index = next(iter(next_attempts.items()))
        registered = next(
            (
                binding
                for binding in self._registered_unit_bindings.values()
                if binding.unit_id == unit_id
            ),
            None,
        )
        if registered is None:
            raise DevelopmentPersistenceError(
                "retryable unit is outside frozen roster"
            )
        interrupted_parent = next(
            (
                item.retry_parent_intent_digest
                for item in recovery.interrupted_attempts
                if item.unit_id == unit_id
                and item.attempt_index == attempt_index - 1
            ),
            None,
        )
        committed_parent = latest_by_unit.get(unit_id)
        parent_digest = (
            interrupted_parent
            if interrupted_parent is not None
            else committed_parent.intent_digest
            if committed_parent is not None
            else None
        )
        return registered.unit_index, attempt_index, parent_digest

    def open_session_cursor(
        self,
        lease: PersistentLease,
        *,
        now_epoch_seconds: int,
    ) -> DevelopmentSessionCursor:
        """Perform the session's sole full recovery and create an in-memory cursor."""

        self._require_active_lease(lease, now_epoch_seconds)
        recovery = self.recover(now_epoch_seconds=now_epoch_seconds)
        verified_records: list[
            tuple[
                CommittedUnit,
                DevelopmentScientificRecord
                | DevelopmentOperationalRecord
                | DevelopmentRoutingReferenceRecord,
            ]
        ] = []
        for marker in recovery.committed_units:
            record = self._verify_committed(marker)
            verified_records.append((marker, record))
        routing_records = [
            record
            for marker, record in verified_records
            if type(record) is DevelopmentRoutingReferenceRecord
            and marker.attempt_disposition == "success"
        ]
        operational_records = [
            record
            for marker, record in verified_records
            if type(record) is DevelopmentOperationalRecord
            and marker.attempt_disposition == "success"
        ]
        expected_operational_indexes = tuple(
            binding.unit_index
            for binding in self.registered_unit_bindings
            if binding.phase in OPERATIONAL_RECORD_PHASES
        )
        observed_operational_indexes = tuple(
            record.unit_index
            for record in sorted(
                operational_records, key=lambda item: item.unit_index
            )
        )
        if observed_operational_indexes != expected_operational_indexes[
            : len(observed_operational_indexes)
        ]:
            raise DevelopmentPersistenceError(
                "operational records are not a frozen roster prefix"
            )
        terminal_reference_markers, successful_reference_markers = (
            self._validated_routing_reference_recovery(recovery)
        )
        successful_reference_indexes = tuple(
            marker.unit_index for marker in successful_reference_markers
        )
        if tuple(
            record.unit_index
            for record in sorted(
                routing_records, key=lambda item: item.unit_index
            )
        ) != successful_reference_indexes:
            raise DevelopmentPersistenceError(
                "routing reference success records differ from verified terminal recovery"
            )
        if len(terminal_reference_markers) < len(successful_reference_markers):
            raise DevelopmentPersistenceError(
                "routing reference success records exceed terminal coverage"
            )
        return DevelopmentSessionCursor(
            store=self,
            lease=lease,
            recovery=recovery,
            verified_records=verified_records,
        )

    def _validated_routing_reference_recovery(
        self,
        recovery: RecoveryReport,
    ) -> tuple[tuple[CommittedUnit, ...], tuple[CommittedUnit, ...]]:
        """Validate terminal routing coverage separately from success evidence."""

        latest_by_unit: dict[str, CommittedUnit] = {}
        for marker in recovery.committed_units:
            if marker.record_kind != ROUTING_REFERENCE_RECORD_KIND:
                continue
            previous = latest_by_unit.get(marker.unit_id)
            if previous is None or marker.attempt_index > previous.attempt_index:
                latest_by_unit[marker.unit_id] = marker
        terminal = tuple(
            sorted(
                (
                    marker
                    for marker in latest_by_unit.values()
                    if marker.attempt_disposition
                    != "retryable_resource_failure"
                ),
                key=lambda item: item.unit_index,
            )
        )
        expected_indexes = tuple(
            binding.unit_index
            for binding in self.registered_unit_bindings
            if binding.phase == ROUTING_REFERENCE_RECORD_KIND
        )
        terminal_indexes = tuple(marker.unit_index for marker in terminal)
        if terminal_indexes != expected_indexes[: len(terminal_indexes)]:
            raise DevelopmentPersistenceError(
                "terminal routing reference records are not a frozen roster prefix"
            )
        successful = tuple(
            marker
            for marker in terminal
            if marker.attempt_disposition == "success"
        )
        successful_indexes = tuple(marker.unit_index for marker in successful)
        if (
            len(set(successful_indexes)) != len(successful_indexes)
            or successful_indexes != tuple(sorted(successful_indexes))
            or any(index not in terminal_indexes for index in successful_indexes)
        ):
            raise DevelopmentPersistenceError(
                "routing reference success records are not a valid terminal subset"
            )
        return terminal, successful

    def _validate_session_cursor(
        self,
        cursor: DevelopmentSessionCursor,
        lease: PersistentLease,
    ) -> None:
        if (
            type(cursor) is not DevelopmentSessionCursor
            or cursor._store is not self
            or cursor._lease != lease
            or cursor._closed
        ):
            raise DevelopmentPersistenceError(
                "development session cursor is not active for this store and lease"
            )

    def create_session_intent(
        self,
        cursor: DevelopmentSessionCursor,
        lease: PersistentLease,
        *,
        now_epoch_seconds: int,
    ) -> UnitIntent:
        """Claim the cursor's exact next unit without rescanning prior bundles."""

        self._validate_session_cursor(cursor, lease)
        self._require_active_lease(lease, now_epoch_seconds)
        if cursor._open_intent is not None:
            raise DevelopmentPersistenceError(
                "session cursor already has an uncommitted intent"
            )
        lease_start_epoch = int(
            _parse_strict_utc(
                lease.acquired_at_utc, "lease acquired_at_utc"
            ).timestamp()
        )
        if now_epoch_seconds - lease_start_epoch >= SOFT_STOP_SECONDS:
            raise DevelopmentPersistenceError(
                "session soft stop forbids claiming a new unit"
            )
        unit_index = cursor._next_unit_index
        attempt_index = cursor._next_attempt_index
        parent_digest = cursor._parent_attempt_intent_digest
        registered = self._registered_unit_bindings.get(unit_index)
        if registered is None:
            raise DevelopmentPersistenceError(
                "all frozen development units are committed"
            )
        if not 0 <= attempt_index < registered.maximum_record_attempts:
            raise DevelopmentPersistenceError(
                "session cursor attempt exceeds registered unit limit"
            )
        if attempt_index == 0 and parent_digest is not None:
            raise DevelopmentPersistenceError(
                "initial attempt cannot have a parent"
            )
        if attempt_index > 0:
            prior = UnitIntent(
                **_read_canonical_json(
                    self._intent_path(registered.unit_id, attempt_index - 1),
                    "parent unit intent",
                )
            )
            prior.validate(self.worker_identity)
            self._verify_intent_registered_binding(prior)
            if parent_digest != prior.digest():
                raise DevelopmentPersistenceError("retry parent intent drifted")
        intent = UnitIntent(
            schema_version=SCHEMA_VERSION,
            protocol_digest=self.worker_identity.protocol_digest,
            revision=self.worker_identity.revision,
            run_id=self.run_id,
            shard_id=registered.phase,
            unit_id=registered.unit_id,
            unit_index=registered.unit_index,
            phase=registered.phase,
            responsibility_id=registered.responsibility_id,
            source_cluster_ordinal=registered.source_cluster_ordinal,
            content_branch_id=registered.content_branch_id,
            geometry_case_id=registered.geometry_case_id,
            maximum_record_attempts=registered.maximum_record_attempts,
            maximum_duration_seconds=registered.maximum_duration_seconds,
            unit_roster_digest=self.worker_identity.unit_roster_digest,
            analysis_unit_identity=asdict(registered.analysis_unit_identity),
            analysis_unit_identity_digest=registered.analysis_unit_identity_digest,
            scientific_question_id=registered.scientific_question_id,
            development_case_id=registered.development_case_id,
            candidate_identity=registered.candidate_identity,
            candidate_config_digest=registered.candidate_config_digest,
            unit_descriptor_digest=registered.unit_descriptor_digest,
            attempt_index=attempt_index,
            session_id=lease.session_id,
            fencing_token=lease.fencing_token,
            worker_identity_digest=self.worker_identity.digest(),
            parent_attempt_intent_digest=parent_digest,
            created_at_utc=_utc_from_epoch(now_epoch_seconds),
        )
        intent.validate(self.worker_identity)
        self._verify_intent_registered_binding(intent)
        _create_only(
            self._intent_path(intent.unit_id, intent.attempt_index),
            canonical_json_bytes(intent.payload()),
        )
        cursor._open_intent = intent
        return intent

    def create_intent(
        self,
        lease: PersistentLease,
        *,
        unit_id: str,
        unit_index: int,
        attempt_index: int,
        parent_attempt_intent_digest: str | None,
        now_epoch_seconds: int,
    ) -> UnitIntent:
        self._require_active_lease(lease, now_epoch_seconds)
        lease_start_epoch = int(
            _parse_strict_utc(lease.acquired_at_utc, "lease acquired_at_utc").timestamp()
        )
        if now_epoch_seconds - lease_start_epoch >= SOFT_STOP_SECONDS:
            raise DevelopmentPersistenceError("session soft stop forbids claiming a new unit")
        _identity(unit_id, "unit_id")
        if type(unit_index) is not int or unit_index < 0:
            raise DevelopmentPersistenceError("unit index is invalid")
        if type(attempt_index) is not int or not 0 <= attempt_index < MAXIMUM_ATTEMPTS:
            raise DevelopmentPersistenceError("attempt index exceeds frozen limit")
        if parent_attempt_intent_digest is not None:
            _digest(parent_attempt_intent_digest, "parent attempt intent")
        registered = self._registered_unit_bindings.get(unit_index)
        if registered is None:
            raise DevelopmentPersistenceError("unit index is outside frozen roster")
        if attempt_index >= registered.maximum_record_attempts:
            raise DevelopmentPersistenceError("attempt index exceeds registered unit limit")
        if unit_id != registered.unit_id:
            raise DevelopmentPersistenceError("unit identity is outside frozen roster")
        unit_commits = sorted(
            (
                item
                for item in self._verified_commits()
                if item.unit_id == unit_id
            ),
            key=lambda item: item.attempt_index,
        )
        if unit_commits and unit_commits[-1].attempt_disposition != (
            "retryable_resource_failure"
        ):
            raise DevelopmentPersistenceError("terminal committed unit cannot be rerun")
        expected = self.next_attempt_index(unit_id)
        if attempt_index != expected:
            raise DevelopmentPersistenceError("attempt sequence is not contiguous")
        if attempt_index == 0 and parent_attempt_intent_digest is not None:
            raise DevelopmentPersistenceError("initial attempt cannot have a parent")
        if attempt_index > 0:
            prior = UnitIntent(
                **_read_canonical_json(
                    self._intent_path(unit_id, attempt_index - 1),
                    "parent unit intent",
                )
            )
            prior.validate(self.worker_identity)
            self._verify_intent_registered_binding(prior)
            if parent_attempt_intent_digest != prior.digest():
                raise DevelopmentPersistenceError("retry parent intent drifted")
        intent = UnitIntent(
            schema_version=SCHEMA_VERSION,
            protocol_digest=self.worker_identity.protocol_digest,
            revision=self.worker_identity.revision,
            run_id=self.run_id,
            shard_id=registered.phase,
            unit_id=unit_id,
            unit_index=unit_index,
            phase=registered.phase,
            responsibility_id=registered.responsibility_id,
            source_cluster_ordinal=registered.source_cluster_ordinal,
            content_branch_id=registered.content_branch_id,
            geometry_case_id=registered.geometry_case_id,
            maximum_record_attempts=registered.maximum_record_attempts,
            maximum_duration_seconds=registered.maximum_duration_seconds,
            unit_roster_digest=self.worker_identity.unit_roster_digest,
            analysis_unit_identity=asdict(registered.analysis_unit_identity),
            analysis_unit_identity_digest=registered.analysis_unit_identity_digest,
            scientific_question_id=registered.scientific_question_id,
            development_case_id=registered.development_case_id,
            candidate_identity=registered.candidate_identity,
            candidate_config_digest=registered.candidate_config_digest,
            unit_descriptor_digest=registered.unit_descriptor_digest,
            attempt_index=attempt_index,
            session_id=lease.session_id,
            fencing_token=lease.fencing_token,
            worker_identity_digest=self.worker_identity.digest(),
            parent_attempt_intent_digest=parent_attempt_intent_digest,
            created_at_utc=_utc_from_epoch(now_epoch_seconds),
        )
        intent.validate(self.worker_identity)
        self._verify_intent_registered_binding(intent)
        _create_only(self._intent_path(unit_id, attempt_index), canonical_json_bytes(intent.payload()))
        return intent

    def commit_unit(
        self,
        lease: PersistentLease,
        intent: UnitIntent,
        *,
        record: DevelopmentScientificRecord
        | DevelopmentOperationalRecord
        | DevelopmentRoutingReferenceRecord,
        diagnostic_members: Mapping[str, bytes] | None = None,
        raw_secret_values: Sequence[str] = (),
        now_epoch_seconds: int,
    ) -> CommittedUnit:
        self._require_active_lease(lease, now_epoch_seconds)
        if intent.fencing_token != lease.fencing_token or intent.session_id != lease.session_id:
            raise DevelopmentPersistenceError("intent does not belong to active lease")
        intent_path = self._intent_path(intent.unit_id, intent.attempt_index)
        if _read_canonical_json(intent_path, "unit intent") != intent.payload():
            raise DevelopmentPersistenceError("unit intent bytes drifted")
        self._verify_intent_registered_binding(intent)
        if type(record) is DevelopmentScientificRecord:
            if intent.phase in {*OPERATIONAL_RECORD_PHASES, ROUTING_REFERENCE_RECORD_KIND}:
                raise DevelopmentPersistenceError(
                    "operational unit requires its registered record kind"
                )
            try:
                validate_record_against_intent(record, intent)
            except DevelopmentRecordError as exc:
                raise DevelopmentPersistenceError(str(exc)) from exc
            record_kind = "development_scientific_record"
            record_member_path = DEVELOPMENT_RECORD_MEMBER_PATH
        elif type(record) is DevelopmentOperationalRecord:
            _validate_operational_record_against_intent(record, intent)
            record_kind = OPERATIONAL_RECORD_KIND
            record_member_path = OPERATIONAL_RECORD_MEMBER_PATH
        elif type(record) is DevelopmentRoutingReferenceRecord:
            _validate_routing_reference_record_against_intent(record, intent)
            record_kind = ROUTING_REFERENCE_RECORD_KIND
            record_member_path = ROUTING_REFERENCE_RECORD_MEMBER_PATH
        else:
            raise DevelopmentPersistenceError(
                "formal development record exact type is required"
            )
        disposition = record.attempt_disposition()
        if (
            disposition == "retryable_resource_failure"
            and intent.attempt_index + 1 >= intent.maximum_record_attempts
        ):
            raise DevelopmentPersistenceError(
                "last registered attempt cannot remain retryable"
            )
        record_bytes = canonical_json_bytes(record.payload())
        if diagnostic_members is None:
            diagnostic_members = {}
        if not isinstance(diagnostic_members, Mapping):
            raise DevelopmentPersistenceError("diagnostic members mapping is invalid")
        normalized: dict[str, bytes] = {record_member_path: record_bytes}
        normalized_casefold: set[str] = set()
        for name, payload in diagnostic_members.items():
            safe_name = _safe_member_path(name)
            folded_name = safe_name.casefold()
            if folded_name in {
                "artifact_manifest.json",
                DEVELOPMENT_RECORD_MEMBER_PATH.casefold(),
                OPERATIONAL_RECORD_MEMBER_PATH.casefold(),
                ROUTING_REFERENCE_RECORD_MEMBER_PATH.casefold(),
            }:
                raise DevelopmentPersistenceError("bundle member name is reserved")
            if folded_name in normalized_casefold:
                raise DevelopmentPersistenceError("bundle member casefold identity collision")
            if type(payload) is not bytes:
                raise DevelopmentPersistenceError("bundle members must be bytes")
            normalized[safe_name] = payload
            normalized_casefold.add(folded_name)
        normalized_casefold.add(record_member_path.casefold())
        if len(normalized) != len(diagnostic_members) + 1:
            raise DevelopmentPersistenceError("bundle member identity collision")
        _reject_secrets(tuple(normalized.values()), raw_secret_values)
        artifact_manifest = ArtifactManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            members=tuple(
                ArtifactMember(path=name, size_bytes=len(payload), sha256=sha256(payload).hexdigest())
                for name, payload in sorted(normalized.items())
            ),
        )
        manifest_bytes = canonical_json_bytes(artifact_manifest.payload())
        _reject_secrets((manifest_bytes,), raw_secret_values)
        archive_bytes = self._build_bundle(normalized, manifest_bytes)
        bundle_digest = sha256(archive_bytes).hexdigest()
        bundle_path = self.run_root / "bundles" / f"sha256_{bundle_digest}.zip"
        _create_only(bundle_path, archive_bytes)
        marker = CommittedUnit(
            schema_version=RESULT_SCHEMA_VERSION,
            protocol_digest=self.worker_identity.protocol_digest,
            revision=self.worker_identity.revision,
            run_id=self.run_id,
            shard_id=intent.shard_id,
            unit_id=intent.unit_id,
            unit_index=intent.unit_index,
            attempt_index=intent.attempt_index,
            session_id=intent.session_id,
            fencing_token=intent.fencing_token,
            intent_digest=intent.digest(),
            attempt_disposition=disposition,
            record_kind=record_kind,
            record_id=record.record_id,
            record_digest=sha256(record_bytes).hexdigest(),
            record_bytes=len(record_bytes),
            actual_elapsed_seconds=float(record.actual_elapsed_seconds),
            maximum_duration_seconds=record.maximum_duration_seconds,
            bundle_sha256=bundle_digest,
            bundle_bytes=len(archive_bytes),
            artifact_manifest_digest=artifact_manifest.digest(),
            worker_identity_digest=self.worker_identity.digest(),
            parent_attempt_intent_digest=intent.parent_attempt_intent_digest,
            committed_at_utc=_utc_from_epoch(now_epoch_seconds),
        )
        marker_bytes = canonical_json_bytes(marker.payload())
        _reject_secrets((marker_bytes,), raw_secret_values)
        _create_only(self._marker_path(intent.unit_id, intent.attempt_index), marker_bytes)
        self._verify_committed(marker)
        return marker

    def commit_session_unit(
        self,
        cursor: DevelopmentSessionCursor,
        lease: PersistentLease,
        intent: UnitIntent,
        *,
        record: DevelopmentScientificRecord
        | DevelopmentOperationalRecord
        | DevelopmentRoutingReferenceRecord,
        diagnostic_members: Mapping[str, bytes] | None = None,
        raw_secret_values: Sequence[str] = (),
        now_epoch_seconds: int,
    ) -> CommittedUnit:
        """Commit one cursor claim and advance only the verified in-memory state."""

        self._validate_session_cursor(cursor, lease)
        if cursor._open_intent != intent:
            raise DevelopmentPersistenceError(
                "session commit does not match its open intent"
            )
        marker = self.commit_unit(
            lease,
            intent,
            record=record,
            diagnostic_members=diagnostic_members,
            raw_secret_values=raw_secret_values,
            now_epoch_seconds=now_epoch_seconds,
        )
        cursor._committed_units.append(marker)
        cursor._records_by_attempt[(marker.unit_id, marker.attempt_index)] = record
        if (
            type(record) is DevelopmentRoutingReferenceRecord
            and marker.attempt_disposition == "success"
        ):
            cursor._routing_reference_records[record.unit_index] = record
        elif type(record) is DevelopmentOperationalRecord:
            cursor._operational_records[record.unit_index] = record
        if marker.attempt_disposition == "retryable_resource_failure":
            next_attempt = intent.attempt_index + 1
            if next_attempt >= intent.maximum_record_attempts:
                raise DevelopmentPersistenceError(
                    "retryable cursor commit exhausted frozen attempts"
                )
            cursor._next_unit_index = intent.unit_index
            cursor._next_attempt_index = next_attempt
            cursor._parent_attempt_intent_digest = intent.digest()
        else:
            cursor._next_unit_index = intent.unit_index + 1
            cursor._next_attempt_index = 0
            cursor._parent_attempt_intent_digest = None
        cursor._open_intent = None
        return marker

    def recover(self, *, now_epoch_seconds: int | None = None) -> RecoveryReport:
        if now_epoch_seconds is None:
            now_epoch_seconds = int(time.time())
        if type(now_epoch_seconds) is not int or now_epoch_seconds < 0:
            raise DevelopmentPersistenceError("recovery time is invalid")
        leases = self._load_leases()
        commits = tuple(self._verified_commits(leases=leases))
        receipts = self._load_receipts(commits=commits, leases=leases)
        receipts_by_session = {item.session_id: item for item in receipts}
        commits_by_unit: dict[str, list[CommittedUnit]] = {}
        for marker in commits:
            commits_by_unit.setdefault(marker.unit_id, []).append(marker)
        retry_parents = {
            item.parent_attempt_intent_digest
            for item in commits
            if item.parent_attempt_intent_digest is not None
        }
        intents: list[UnitIntent] = []
        for path in sorted((self.run_root / "intents").glob("*.json")):
            payload = _read_canonical_json(path, "unit intent")
            try:
                intent = UnitIntent(**payload)
            except TypeError as exc:
                raise DevelopmentPersistenceError("unit intent schema is invalid") from exc
            intent.validate(self.worker_identity)
            self._verify_intent_registered_binding(intent)
            if path != self._intent_path(intent.unit_id, intent.attempt_index):
                raise DevelopmentPersistenceError("unit intent path identity drifted")
            self._lease_for_lineage(
                leases,
                session_id=intent.session_id,
                fencing_token=intent.fencing_token,
                role="unit intent",
            )
            intents.append(intent)
            if intent.parent_attempt_intent_digest is not None:
                retry_parents.add(intent.parent_attempt_intent_digest)
        self._validate_intent_attempt_lineage(intents)
        interrupted: list[InterruptedAttempt] = []
        next_attempts: dict[str, int] = {}
        for intent in intents:
            marker_path = self._marker_path(intent.unit_id, intent.attempt_index)
            if marker_path.exists():
                continue
            if intent.digest() in retry_parents:
                continue
            prior_commits = sorted(
                commits_by_unit.get(intent.unit_id, ()),
                key=lambda item: item.attempt_index,
            )
            if prior_commits:
                latest = prior_commits[-1]
                if (
                    latest.attempt_disposition != "retryable_resource_failure"
                    or intent.attempt_index != latest.attempt_index + 1
                    or intent.parent_attempt_intent_digest != latest.intent_digest
                ):
                    raise DevelopmentPersistenceError(
                        "dangling intent is not a retryable committed successor"
                    )
            lease = self._lease_for_lineage(
                leases,
                session_id=intent.session_id,
                fencing_token=intent.fencing_token,
                role="dangling intent",
            )
            session_closed = intent.session_id in receipts_by_session
            if not session_closed and lease.expires_at_epoch_seconds > now_epoch_seconds:
                continue
            interrupted.append(
                InterruptedAttempt(
                    unit_id=intent.unit_id,
                    attempt_index=intent.attempt_index,
                    failure_class="resource_failure",
                    failure_reason="colab_session_interrupted",
                    retry_parent_intent_digest=intent.digest(),
                )
            )
            next_attempts[intent.unit_id] = intent.attempt_index + 1
        for unit_id, unit_commits in commits_by_unit.items():
            latest = max(unit_commits, key=lambda item: item.attempt_index)
            successor_claimed = any(
                item.unit_id == unit_id
                and item.attempt_index == latest.attempt_index + 1
                for item in intents
            )
            if (
                latest.attempt_disposition == "retryable_resource_failure"
                and not successor_claimed
            ):
                next_attempts[unit_id] = latest.attempt_index + 1
        for unit_id, value in next_attempts.items():
            registered = next(
                item
                for item in self._registered_unit_bindings.values()
                if item.unit_id == unit_id
            )
            if value >= registered.maximum_record_attempts:
                raise DevelopmentPersistenceError(
                    "interrupted unit exhausted frozen attempts"
                )
        ledger_payload = [
            item.payload()
            for item in sorted(
                commits, key=lambda item: (item.unit_index, item.attempt_index)
            )
        ]
        return RecoveryReport(
            committed_units=tuple(
                sorted(commits, key=lambda item: (item.unit_index, item.attempt_index))
            ),
            interrupted_attempts=tuple(interrupted),
            next_attempt_by_unit=tuple(sorted(next_attempts.items())),
            ledger_digest=canonical_digest(ledger_payload),
        )

    def _validate_intent_attempt_lineage(
        self,
        intents: Sequence[UnitIntent],
    ) -> None:
        by_unit: dict[str, list[UnitIntent]] = {}
        for intent in intents:
            by_unit.setdefault(intent.unit_id, []).append(intent)
        for unit_intents in by_unit.values():
            ordered = sorted(unit_intents, key=lambda item: item.attempt_index)
            if [item.attempt_index for item in ordered] != list(range(len(ordered))):
                raise DevelopmentPersistenceError("attempt history is not contiguous")
            for prior, current in zip(ordered, ordered[1:]):
                if current.parent_attempt_intent_digest != prior.digest():
                    raise DevelopmentPersistenceError("retry parent intent drifted")

    def next_attempt_index(self, unit_id: str) -> int:
        _identity(unit_id, "unit_id")
        registered = next(
            (
                item
                for item in self._registered_unit_bindings.values()
                if item.unit_id == unit_id
            ),
            None,
        )
        if registered is None:
            raise DevelopmentPersistenceError("unit identity is outside frozen roster")
        unit_commits = sorted(
            (
                item
                for item in self._verified_commits()
                if item.unit_id == unit_id
            ),
            key=lambda item: item.attempt_index,
        )
        if unit_commits and unit_commits[-1].attempt_disposition != (
            "retryable_resource_failure"
        ):
            raise DevelopmentPersistenceError("terminal committed unit has no next attempt")
        attempts = []
        for path in (self.run_root / "intents").glob(f"{unit_id}__attempt_*.json"):
            payload = _read_canonical_json(path, "unit intent")
            attempts.append(int(payload["attempt_index"]))
        if not attempts:
            return 0
        ordered = sorted(attempts)
        if ordered != list(range(ordered[-1] + 1)):
            raise DevelopmentPersistenceError("attempt history is not contiguous")
        next_index = ordered[-1] + 1
        if next_index >= registered.maximum_record_attempts:
            raise DevelopmentPersistenceError("frozen attempt budget exhausted")
        return next_index

    def verified_terminal_scientific_records(
        self,
        *,
        now_epoch_seconds: int,
    ) -> tuple[DevelopmentScientificRecord, ...]:
        """Return only exact terminal records after full store recovery checks."""

        return tuple(
            record
            for record, _marker in self.verified_terminal_scientific_evidence(
                now_epoch_seconds=now_epoch_seconds
            )
        )

    def verified_terminal_routing_reference_records(
        self,
        *,
        now_epoch_seconds: int,
    ) -> tuple[DevelopmentRoutingReferenceRecord, ...]:
        """Read committed operational fit inputs through the common recovery path."""

        recovery = self.recover(now_epoch_seconds=now_epoch_seconds)
        _terminal_markers, markers = (
            self._validated_routing_reference_recovery(recovery)
        )
        records = tuple(
            self._verify_committed(marker)
            for marker in sorted(markers, key=lambda item: item.unit_index)
        )
        if any(
            type(record) is not DevelopmentRoutingReferenceRecord
            for record in records
        ):
            raise DevelopmentPersistenceError(
                "routing reference evidence contains another record kind"
            )
        if tuple(record.unit_index for record in records) != tuple(
            marker.unit_index for marker in markers
        ):
            raise DevelopmentPersistenceError(
                "routing reference records differ from verified success evidence"
            )
        return records

    def verified_terminal_scientific_evidence(
        self,
        *,
        now_epoch_seconds: int,
    ) -> tuple[tuple[DevelopmentScientificRecord, CommittedUnit], ...]:
        """Return records together with their fully verified COMMITTED markers."""

        recovery = self.recover(now_epoch_seconds=now_epoch_seconds)
        latest_by_unit: dict[str, CommittedUnit] = {}
        for marker in recovery.committed_units:
            if marker.record_kind == "development_scientific_record":
                latest_by_unit[marker.unit_id] = marker
        if any(
            marker.attempt_disposition == "retryable_resource_failure"
            for marker in latest_by_unit.values()
        ):
            raise DevelopmentPersistenceError(
                "retryable scientific unit has no terminal record"
            )
        evidence = tuple(
            (self._verify_committed(marker), marker)
            for marker in sorted(
                latest_by_unit.values(), key=lambda item: item.unit_index
            )
        )
        if any(type(record) is not DevelopmentScientificRecord for record, _ in evidence):
            raise DevelopmentPersistenceError(
                "scientific evidence contains another record kind"
            )
        scientific_indexes = tuple(
            binding.unit_index
            for binding in self.registered_unit_bindings
            if binding.phase not in {
                *OPERATIONAL_RECORD_PHASES,
                ROUTING_REFERENCE_RECORD_KIND,
            }
        )
        if tuple(record.unit_index for record, _marker in evidence) != (
            scientific_indexes[: len(evidence)]
        ):
            raise DevelopmentPersistenceError(
                "terminal scientific records are not a frozen roster prefix"
            )
        return evidence

    def verified_terminal_scientific_evidence_for_unit_indexes(
        self,
        unit_indexes: Sequence[int],
        *,
        now_epoch_seconds: int,
    ) -> tuple[tuple[DevelopmentScientificRecord, CommittedUnit], ...]:
        """Verify an exact frozen unit set without treating a mutable ledger as authority."""

        if isinstance(unit_indexes, (str, bytes)) or not isinstance(
            unit_indexes, Sequence
        ):
            raise DevelopmentPersistenceError("requested unit indexes are invalid")
        requested = tuple(unit_indexes)
        if (
            not requested
            or any(type(index) is not int for index in requested)
            or len(requested) != len(set(requested))
            or tuple(sorted(requested)) != requested
        ):
            raise DevelopmentPersistenceError(
                "requested unit indexes must be unique and ordered"
            )
        if any(index not in self._registered_unit_bindings for index in requested):
            raise DevelopmentPersistenceError(
                "requested unit index is outside frozen roster"
            )
        if any(
            self._registered_unit_bindings[index].phase
            in {*OPERATIONAL_RECORD_PHASES, ROUTING_REFERENCE_RECORD_KIND}
            for index in requested
        ):
            raise DevelopmentPersistenceError(
                "requested scientific evidence includes an operational unit"
            )
        recovery = self.recover(now_epoch_seconds=now_epoch_seconds)
        latest_by_index: dict[int, CommittedUnit] = {}
        for marker in recovery.committed_units:
            latest_by_index[marker.unit_index] = marker
        missing = tuple(index for index in requested if index not in latest_by_index)
        if missing:
            raise DevelopmentPersistenceError(
                "requested frozen units lack terminal COMMITTED evidence"
            )
        markers = tuple(latest_by_index[index] for index in requested)
        if any(
            marker.attempt_disposition == "retryable_resource_failure"
            for marker in markers
        ):
            raise DevelopmentPersistenceError(
                "requested scientific unit has no terminal record"
            )
        evidence = tuple((self._verify_committed(marker), marker) for marker in markers)
        if any(type(record) is not DevelopmentScientificRecord for record, _ in evidence):
            raise DevelopmentPersistenceError(
                "requested scientific evidence contains another record kind"
            )
        if tuple(record.unit_index for record, _marker in evidence) != requested:
            raise DevelopmentPersistenceError(
                "requested scientific evidence order drifted"
            )
        return evidence

    def write_session_receipt(
        self,
        receipt: SessionReceipt,
        *,
        raw_secret_values: Sequence[str] = (),
        session_cursor: DevelopmentSessionCursor | None = None,
    ) -> Path:
        leases = self._load_leases()
        if session_cursor is None:
            commits = tuple(self._verified_commits(leases=leases))
        else:
            self._validate_session_cursor(session_cursor, session_cursor._lease)
            commits = session_cursor.committed_units
        self._validate_session_receipt(receipt, commits=commits, leases=leases)
        payload = canonical_json_bytes(asdict(receipt))
        _reject_secrets((payload,), raw_secret_values)
        path = self.run_root / "receipts" / f"{_identity(receipt.session_id, 'session_id')}.json"
        _create_only(path, payload)
        if session_cursor is not None:
            session_cursor._closed = True
        return path

    def _validate_session_receipt(
        self,
        receipt: SessionReceipt,
        *,
        commits: Sequence[CommittedUnit],
        leases: Sequence[PersistentLease],
    ) -> None:
        if type(receipt) is not SessionReceipt:
            raise DevelopmentPersistenceError("session receipt exact type is required")
        if receipt.schema_version != DIAGNOSTIC_SCHEMA_VERSION:
            raise DevelopmentPersistenceError("session receipt schema drifted")
        if receipt.run_id != self.run_id or receipt.revision != self.worker_identity.revision:
            raise DevelopmentPersistenceError("session receipt frozen identity drifted")
        _digest(receipt.package_sha256, "session receipt package")
        lease = self._lease_for_session(leases, receipt.session_id)
        started = _parse_strict_utc(receipt.started_at_utc, "session started_at_utc")
        ended = _parse_strict_utc(receipt.ended_at_utc, "session ended_at_utc")
        if receipt.started_at_utc != lease.acquired_at_utc:
            raise DevelopmentPersistenceError("session receipt start does not match lease")
        elapsed = (ended - started).total_seconds()
        if elapsed < 0.0:
            raise DevelopmentPersistenceError("session UTC order is invalid")
        if ended > datetime.fromtimestamp(lease.expires_at_epoch_seconds, timezone.utc):
            raise DevelopmentPersistenceError("session receipt exceeds lease expiry")
        if (
            isinstance(receipt.walltime_seconds, bool)
            or not isinstance(receipt.walltime_seconds, (int, float))
            or not 0.0 <= float(receipt.walltime_seconds) < HARD_SESSION_CAP_SECONDS
        ):
            raise DevelopmentPersistenceError("session walltime violates hard cap")
        if abs(float(receipt.walltime_seconds) - elapsed) > 1e-6:
            raise DevelopmentPersistenceError("session walltime and UTC interval differ")
        if receipt.soft_stop_seconds != SOFT_STOP_SECONDS or receipt.hard_session_cap_seconds != HARD_SESSION_CAP_SECONDS:
            raise DevelopmentPersistenceError("session stop policy drifted")
        if receipt.gpu_mix_policy != GPU_MIX_POLICY:
            raise DevelopmentPersistenceError("GPU mix policy drifted")
        _identity(receipt.session_id, "session receipt session_id")
        _identity(receipt.gpu_model, "session GPU model")
        _identity(receipt.cuda_identity, "session CUDA identity")
        _identity(receipt.termination_reason, "session termination reason")
        _digest(receipt.environment_digest, "session environment")
        if type(receipt.peak_vram_bytes) is not int or receipt.peak_vram_bytes < 1:
            raise DevelopmentPersistenceError("session peak VRAM is invalid")
        if type(receipt.committed_unit_ids) is not tuple or any(
            type(value) is not str for value in receipt.committed_unit_ids
        ):
            raise DevelopmentPersistenceError("session committed unit identities are invalid")
        if len(receipt.committed_unit_ids) != len(set(receipt.committed_unit_ids)):
            raise DevelopmentPersistenceError("session committed unit identities collide")
        expected_units = tuple(
            item.unit_id
            for item in sorted(commits, key=lambda item: item.unit_index)
            if item.session_id == receipt.session_id
        )
        if receipt.committed_unit_ids != expected_units:
            raise DevelopmentPersistenceError("session committed unit identities differ from markers")
        if any(
            not started
            <= _parse_strict_utc(item.committed_at_utc, "marker committed_at_utc")
            <= ended
            for item in commits
            if item.session_id == receipt.session_id
        ):
            raise DevelopmentPersistenceError(
                "session receipt ends before a committed marker"
            )
        if type(receipt.public_secret_identity_digests) is not tuple:
            raise DevelopmentPersistenceError("public secret identity digest roster is invalid")
        for value in receipt.public_secret_identity_digests:
            _digest(value, "public secret identity")

    def _load_receipts(
        self,
        *,
        commits: Sequence[CommittedUnit],
        leases: Sequence[PersistentLease],
    ) -> tuple[SessionReceipt, ...]:
        receipts: list[SessionReceipt] = []
        seen_sessions: set[str] = set()
        for path in sorted((self.run_root / "receipts").glob("*.json")):
            payload = _read_canonical_json(path, "session receipt")
            for field_name in (
                "committed_unit_ids",
                "public_secret_identity_digests",
            ):
                if type(payload.get(field_name)) is not list:
                    raise DevelopmentPersistenceError(
                        "session receipt tuple field is invalid"
                    )
                payload[field_name] = tuple(payload[field_name])
            try:
                receipt = SessionReceipt(**payload)
            except TypeError as exc:
                raise DevelopmentPersistenceError("session receipt schema is invalid") from exc
            self._validate_session_receipt(receipt, commits=commits, leases=leases)
            if receipt.session_id in seen_sessions:
                raise DevelopmentPersistenceError("multiple receipts exist for one session")
            if path.name != f"{receipt.session_id}.json":
                raise DevelopmentPersistenceError("session receipt path identity drifted")
            seen_sessions.add(receipt.session_id)
            receipts.append(receipt)
        return tuple(receipts)

    def _build_bundle(self, members: Mapping[str, bytes], manifest_bytes: bytes) -> bytes:
        buffer = BytesIO()
        with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, payload in (*sorted(members.items()), ("artifact_manifest.json", manifest_bytes)):
                info = ZipInfo(name)
                info.date_time = (1980, 1, 1, 0, 0, 0)
                info.create_system = 3
                info.external_attr = stat.S_IFREG << 16 | 0o600 << 16
                archive.writestr(info, payload, compress_type=ZIP_DEFLATED, compresslevel=6)
        return buffer.getvalue()

    def _verify_committed(
        self, marker: CommittedUnit
    ) -> (
        DevelopmentScientificRecord
        | DevelopmentOperationalRecord
        | DevelopmentRoutingReferenceRecord
    ):
        if marker.schema_version != RESULT_SCHEMA_VERSION:
            raise DevelopmentPersistenceError("COMMITTED marker schema drifted")
        if (
            marker.protocol_digest != self.worker_identity.protocol_digest
            or marker.revision != self.worker_identity.revision
            or marker.run_id != self.run_id
        ):
            raise DevelopmentPersistenceError("COMMITTED marker frozen identity drifted")
        for role, value in (
            ("marker shard_id", marker.shard_id),
            ("marker unit_id", marker.unit_id),
            ("marker session_id", marker.session_id),
        ):
            _identity(value, role)
        if type(marker.unit_index) is not int or marker.unit_index < 0:
            raise DevelopmentPersistenceError("marker unit index is invalid")
        if type(marker.attempt_index) is not int or not 0 <= marker.attempt_index < MAXIMUM_ATTEMPTS:
            raise DevelopmentPersistenceError("marker attempt index is invalid")
        if type(marker.fencing_token) is not int or marker.fencing_token < 1:
            raise DevelopmentPersistenceError("marker fencing token is invalid")
        if marker.attempt_index == 0 and marker.parent_attempt_intent_digest is not None:
            raise DevelopmentPersistenceError("initial marker cannot have a parent")
        if marker.attempt_index > 0:
            _digest(marker.parent_attempt_intent_digest, "marker parent attempt intent")
        if marker.worker_identity_digest != self.worker_identity.digest():
            raise DevelopmentPersistenceError("marker worker identity drifted")
        if marker.attempt_disposition not in ATTEMPT_DISPOSITIONS:
            raise DevelopmentPersistenceError("marker attempt disposition is invalid")
        if marker.record_kind not in {
            "development_scientific_record",
            OPERATIONAL_RECORD_KIND,
            ROUTING_REFERENCE_RECORD_KIND,
        }:
            raise DevelopmentPersistenceError("marker record kind is invalid")
        _digest(marker.record_id, "marker record identity")
        _digest(marker.record_digest, "marker record")
        if type(marker.record_bytes) is not int or marker.record_bytes < 1:
            raise DevelopmentPersistenceError("marker record size is invalid")
        if (
            isinstance(marker.actual_elapsed_seconds, bool)
            or not isinstance(marker.actual_elapsed_seconds, (int, float))
            or float(marker.actual_elapsed_seconds) < 0.0
        ):
            raise DevelopmentPersistenceError("marker actual elapsed time is invalid")
        if type(marker.maximum_duration_seconds) is not int or marker.maximum_duration_seconds < 1:
            raise DevelopmentPersistenceError("marker duration limit is invalid")
        _parse_strict_utc(marker.committed_at_utc, "marker committed_at_utc")
        for role in ("protocol_digest", "intent_digest", "bundle_sha256", "artifact_manifest_digest", "worker_identity_digest"):
            _digest(getattr(marker, role), f"marker {role}")
        bundle_path = self.run_root / "bundles" / f"sha256_{marker.bundle_sha256}.zip"
        if (
            not bundle_path.is_file()
            or bundle_path.is_symlink()
            or file_sha256(bundle_path) != marker.bundle_sha256
            or bundle_path.stat().st_size != marker.bundle_bytes
        ):
            raise DevelopmentPersistenceError("committed bundle digest or size drifted")
        with ZipFile(bundle_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            folded_names = [name.casefold() for name in names]
            if (
                len(folded_names) != len(set(folded_names))
                or folded_names.count("artifact_manifest.json") != 1
                or "artifact_manifest.json" not in names
            ):
                raise DevelopmentPersistenceError("bundle member identities are invalid")
            record_member_path = (
                OPERATIONAL_RECORD_MEMBER_PATH
                if marker.record_kind == OPERATIONAL_RECORD_KIND
                else ROUTING_REFERENCE_RECORD_MEMBER_PATH
                if marker.record_kind == ROUTING_REFERENCE_RECORD_KIND
                else DEVELOPMENT_RECORD_MEMBER_PATH
            )
            if record_member_path not in names:
                raise DevelopmentPersistenceError(
                    "bundle lacks its formal development record"
                )
            for info in infos:
                _safe_member_path(info.filename)
                mode = info.external_attr >> 16
                if (
                    not stat.S_ISREG(mode)
                    or stat.S_ISLNK(mode)
                    or info.is_dir()
                    or mode & 0o111
                ):
                    raise DevelopmentPersistenceError(
                        "bundle members must be non-executable regular files"
                    )
            manifest_raw = archive.read("artifact_manifest.json")
            try:
                manifest_value = json.loads(manifest_raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DevelopmentPersistenceError("artifact manifest is unreadable") from exc
            if canonical_json_bytes(manifest_value) != manifest_raw:
                raise DevelopmentPersistenceError("artifact manifest is not canonical")
            if canonical_digest(manifest_value) != marker.artifact_manifest_digest:
                raise DevelopmentPersistenceError("artifact manifest digest drifted")
            if set(manifest_value) != {"schema_version", "members"} or manifest_value["schema_version"] != MANIFEST_SCHEMA_VERSION:
                raise DevelopmentPersistenceError("artifact manifest schema drifted")
            if type(manifest_value["members"]) is not list or any(
                type(item) is not dict
                or set(item) != {"path", "size_bytes", "sha256"}
                or type(item["path"]) is not str
                or type(item["size_bytes"]) is not int
                or item["size_bytes"] < 0
                or type(item["sha256"]) is not str
                or _DIGEST.fullmatch(item["sha256"]) is None
                for item in manifest_value["members"]
            ):
                raise DevelopmentPersistenceError("artifact manifest members are invalid")
            manifest_members = manifest_value.get("members", [])
            manifest_paths = [item["path"] for item in manifest_members]
            if len({path.casefold() for path in manifest_paths}) != len(manifest_paths):
                raise DevelopmentPersistenceError("artifact manifest paths collide by casefold")
            for path in manifest_paths:
                _safe_member_path(path)
                if path.casefold() == "artifact_manifest.json":
                    raise DevelopmentPersistenceError("artifact manifest recursively lists itself")
            expected = {item["path"]: item for item in manifest_members}
            if set(expected) != set(names) - {"artifact_manifest.json"}:
                raise DevelopmentPersistenceError("artifact manifest coverage drifted")
            for name, item in expected.items():
                payload = archive.read(name)
                if len(payload) != item["size_bytes"] or sha256(payload).hexdigest() != item["sha256"]:
                    raise DevelopmentPersistenceError("bundle member digest drifted")
            record_raw = archive.read(record_member_path)
            if (
                len(record_raw) != marker.record_bytes
                or sha256(record_raw).hexdigest() != marker.record_digest
            ):
                raise DevelopmentPersistenceError(
                    "marker record digest or size drifted"
                )
            try:
                record_payload = json.loads(record_raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DevelopmentPersistenceError(
                    "formal development record is unreadable"
                ) from exc
            if type(record_payload) is not dict or canonical_json_bytes(record_payload) != record_raw:
                raise DevelopmentPersistenceError(
                    "formal development record is not canonical JSON"
                )
            try:
                record = (
                    DevelopmentOperationalRecord.from_payload(record_payload)
                    if marker.record_kind == OPERATIONAL_RECORD_KIND
                    else DevelopmentRoutingReferenceRecord.from_payload(record_payload)
                    if marker.record_kind == ROUTING_REFERENCE_RECORD_KIND
                    else DevelopmentScientificRecord.from_payload(record_payload)
                )
            except DevelopmentRecordError as exc:
                raise DevelopmentPersistenceError(str(exc)) from exc
            if (
                record.record_id != marker.record_id
                or
                record.attempt_disposition() != marker.attempt_disposition
                or float(record.actual_elapsed_seconds)
                != float(marker.actual_elapsed_seconds)
                or record.maximum_duration_seconds
                != marker.maximum_duration_seconds
            ):
                raise DevelopmentPersistenceError(
                    "marker disposition or duration differs from scientific record"
                )
            return record

    def _verified_commits(
        self,
        *,
        leases: Sequence[PersistentLease] | None = None,
    ) -> list[CommittedUnit]:
        if leases is None:
            leases = self._load_leases()
        commits: list[CommittedUnit] = []
        seen_attempts: set[tuple[str, int]] = set()
        for path in sorted((self.run_root / "markers").glob("*.COMMITTED.json")):
            payload = _read_canonical_json(path, "COMMITTED marker")
            try:
                marker = CommittedUnit(**payload)
            except TypeError as exc:
                raise DevelopmentPersistenceError("COMMITTED marker schema is invalid") from exc
            attempt_identity = (marker.unit_id, marker.attempt_index)
            if attempt_identity in seen_attempts:
                raise DevelopmentPersistenceError("duplicate committed attempt exists")
            if path != self._marker_path(marker.unit_id, marker.attempt_index):
                raise DevelopmentPersistenceError("COMMITTED marker path identity drifted")
            record = self._verify_committed(marker)
            intent = _read_canonical_json(
                self._intent_path(marker.unit_id, marker.attempt_index), "unit intent"
            )
            intent_object = UnitIntent(**intent)
            intent_object.validate(self.worker_identity)
            self._verify_intent_registered_binding(intent_object)
            if type(record) is DevelopmentRoutingReferenceRecord:
                _validate_routing_reference_record_against_intent(
                    record, intent_object
                )
            elif type(record) is DevelopmentOperationalRecord:
                _validate_operational_record_against_intent(record, intent_object)
            else:
                try:
                    validate_record_against_intent(record, intent_object)
                except DevelopmentRecordError as exc:
                    raise DevelopmentPersistenceError(str(exc)) from exc
            lease = self._lease_for_lineage(
                leases,
                session_id=marker.session_id,
                fencing_token=marker.fencing_token,
                role="COMMITTED marker",
            )
            lease_start = _parse_strict_utc(
                lease.acquired_at_utc, "lease acquired_at_utc"
            )
            intent_time = _parse_strict_utc(
                intent_object.created_at_utc, "unit intent created_at_utc"
            )
            marker_time = _parse_strict_utc(
                marker.committed_at_utc, "marker committed_at_utc"
            )
            lease_expiry = datetime.fromtimestamp(
                lease.expires_at_epoch_seconds, timezone.utc
            )
            if not lease_start <= intent_time <= marker_time < lease_expiry:
                raise DevelopmentPersistenceError(
                    "marker intent time is outside session lease"
                )
            if canonical_digest(intent) != marker.intent_digest:
                raise DevelopmentPersistenceError("marker intent digest drifted")
            if (
                marker.unit_index != intent_object.unit_index
                or marker.shard_id != intent_object.shard_id
                or marker.session_id != intent_object.session_id
                or marker.fencing_token != intent_object.fencing_token
                or marker.parent_attempt_intent_digest
                != intent_object.parent_attempt_intent_digest
            ):
                raise DevelopmentPersistenceError("marker and intent identities differ")
            seen_attempts.add(attempt_identity)
            commits.append(marker)
        self._validate_committed_attempt_lineage(commits)
        self._verify_bundle_reference_completeness(commits)
        return commits

    def _validate_committed_attempt_lineage(
        self,
        commits: Sequence[CommittedUnit],
    ) -> None:
        by_unit: dict[str, list[CommittedUnit]] = {}
        for marker in commits:
            by_unit.setdefault(marker.unit_id, []).append(marker)
        for unit_id, unit_commits in by_unit.items():
            ordered = sorted(unit_commits, key=lambda item: item.attempt_index)
            registered = next(
                item
                for item in self._registered_unit_bindings.values()
                if item.unit_id == unit_id
            )
            if len(ordered) > registered.maximum_record_attempts:
                raise DevelopmentPersistenceError(
                    "committed attempt history exceeds frozen limit"
                )
            by_attempt = {item.attempt_index: item for item in ordered}
            for current in ordered:
                if current.attempt_index == 0:
                    continue
                previous = by_attempt.get(current.attempt_index - 1)
                if previous is not None and previous.attempt_disposition != (
                    "retryable_resource_failure"
                ):
                    raise DevelopmentPersistenceError(
                        "terminal committed attempt cannot have a successor"
                    )
                expected_parent = (
                    previous.intent_digest
                    if previous is not None
                    else canonical_digest(
                        _read_canonical_json(
                            self._intent_path(
                                current.unit_id, current.attempt_index - 1
                            ),
                            "interrupted parent unit intent",
                        )
                    )
                )
                if current.parent_attempt_intent_digest != expected_parent:
                    raise DevelopmentPersistenceError(
                        "committed retry parent lineage drifted"
                    )
            for previous, current in zip(ordered, ordered[1:]):
                if (
                    previous.attempt_disposition != "retryable_resource_failure"
                    and current.attempt_index > previous.attempt_index
                ):
                    raise DevelopmentPersistenceError(
                        "terminal committed attempt cannot have a successor"
                    )
            latest = ordered[-1]
            if (
                latest.attempt_disposition == "retryable_resource_failure"
                and latest.attempt_index + 1 >= registered.maximum_record_attempts
            ):
                raise DevelopmentPersistenceError(
                    "last registered attempt cannot remain retryable"
                )

    def _verify_bundle_reference_completeness(
        self,
        commits: Sequence[CommittedUnit],
    ) -> None:
        expected = [
            self.run_root / "bundles" / f"sha256_{item.bundle_sha256}.zip"
            for item in commits
        ]
        if len(expected) != len(set(expected)):
            raise DevelopmentPersistenceError(
                "content-addressed bundle is referenced by multiple markers"
            )
        observed = set((self.run_root / "bundles").glob("*"))
        if observed != set(expected):
            raise DevelopmentPersistenceError(
                "orphan or unreferenced bundle exists"
            )

    def _load_leases(self) -> list[PersistentLease]:
        leases: list[PersistentLease] = []
        for path in sorted((self.run_root / "leases").glob("fence_*.json")):
            try:
                lease = PersistentLease(**_read_canonical_json(path, "lease"))
            except TypeError as exc:
                raise DevelopmentPersistenceError("lease schema is invalid") from exc
            lease.validate()
            if lease.run_id != self.run_id or lease.worker_identity_digest != self.worker_identity.digest():
                raise DevelopmentPersistenceError("lease frozen identity drifted")
            if path.name != f"fence_{lease.fencing_token:08d}.json":
                raise DevelopmentPersistenceError("lease path identity drifted")
            leases.append(lease)
        tokens = [item.fencing_token for item in leases]
        if tokens != list(range(1, len(tokens) + 1)):
            raise DevelopmentPersistenceError("lease fencing history is not contiguous")
        if len({item.session_id for item in leases}) != len(leases):
            raise DevelopmentPersistenceError("session identity is reused across leases")
        return leases

    def _lease_for_lineage(
        self,
        leases: Sequence[PersistentLease],
        *,
        session_id: str,
        fencing_token: int,
        role: str,
    ) -> PersistentLease:
        matches = [
            item
            for item in leases
            if item.session_id == session_id and item.fencing_token == fencing_token
        ]
        if len(matches) != 1:
            raise DevelopmentPersistenceError(f"{role} lease/session/fence lineage is missing")
        return matches[0]

    def _lease_for_session(
        self,
        leases: Sequence[PersistentLease],
        session_id: str,
    ) -> PersistentLease:
        _identity(session_id, "session_id")
        matches = [item for item in leases if item.session_id == session_id]
        if len(matches) != 1:
            raise DevelopmentPersistenceError("session lease lineage is missing or ambiguous")
        return matches[0]

    def _require_active_lease(self, lease: PersistentLease, now_epoch_seconds: int) -> None:
        lease.validate()
        lease_start_epoch = int(
            _parse_strict_utc(lease.acquired_at_utc, "lease acquired_at_utc").timestamp()
        )
        if type(now_epoch_seconds) is not int or now_epoch_seconds < lease_start_epoch:
            raise DevelopmentPersistenceError("session operation time is invalid")
        leases = self._load_leases()
        if not leases or leases[-1] != lease:
            raise DevelopmentPersistenceError("stale fencing token")
        if lease.expires_at_epoch_seconds <= now_epoch_seconds:
            raise DevelopmentPersistenceError("writer lease expired")

    def _intent_path(self, unit_id: str, attempt_index: int) -> Path:
        return self.run_root / "intents" / f"{unit_id}__attempt_{attempt_index}.json"

    def _marker_path(self, unit_id: str, attempt_index: int) -> Path:
        return self.run_root / "markers" / f"{unit_id}__attempt_{attempt_index}.COMMITTED.json"

    def _committed_marker_paths(self, unit_id: str) -> tuple[Path, ...]:
        return tuple((self.run_root / "markers").glob(f"{unit_id}__attempt_*.COMMITTED.json"))
