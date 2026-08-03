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
from typing import Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


SCHEMA_VERSION = "ceg_wm_development_worker_persistence_v1"
MANIFEST_SCHEMA_VERSION = "ceg_wm_development_artifact_manifest_v1"
RESULT_SCHEMA_VERSION = "ceg_wm_development_committed_marker_v1"
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
    {".dill", ".joblib", ".pickle", ".pkl", ".pt", ".pth"}
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


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    if path.suffix.lower() in _EXECUTABLE_DESERIALIZATION_SUFFIXES:
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


@dataclass(frozen=True, slots=True)
class FrozenWorkerIdentity:
    revision: str
    protocol_digest: str
    execution_intent_authority_digest: str
    input_manifest_digest: str
    candidate_config_digest: str
    package_sha256: str
    bootstrap_sha256: str

    def validate(self) -> None:
        if type(self.revision) is not str or _REVISION.fullmatch(self.revision) is None:
            raise DevelopmentPersistenceError("worker revision must be a full Git SHA")
        for role in (
            "protocol_digest",
            "execution_intent_authority_digest",
            "input_manifest_digest",
            "candidate_config_digest",
            "package_sha256",
            "bootstrap_sha256",
        ):
            _digest(getattr(self, role), role)

    def digest(self) -> str:
        self.validate()
        return canonical_digest(asdict(self))


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
        if type(self.expires_at_epoch_seconds) is not int:
            raise DevelopmentPersistenceError("lease expiry is invalid")
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
        ):
            _identity(value, role)
        if type(self.unit_index) is not int or self.unit_index < 0:
            raise DevelopmentPersistenceError("unit intent index is invalid")
        if type(self.attempt_index) is not int or not 0 <= self.attempt_index < MAXIMUM_ATTEMPTS:
            raise DevelopmentPersistenceError("unit intent attempt is invalid")
        if type(self.fencing_token) is not int or self.fencing_token < 1:
            raise DevelopmentPersistenceError("unit intent fencing token is invalid")
        _digest(self.worker_identity_digest, "unit intent worker identity")
        if self.worker_identity_digest != worker_identity.digest():
            raise DevelopmentPersistenceError("unit intent worker identity drifted")
        if self.attempt_index == 0 and self.parent_attempt_intent_digest is not None:
            raise DevelopmentPersistenceError("initial attempt cannot have a parent")
        if self.attempt_index > 0:
            _digest(self.parent_attempt_intent_digest, "parent attempt intent")


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
    ) -> None:
        self.run_id = _identity(run_id, "run_id")
        worker_identity.validate()
        self.worker_identity = worker_identity
        root = _regular_directory(Path(persistent_root), "persistent_root")
        self.run_root = _regular_directory(root / self.run_id, "run_root")
        for name in ("leases", "intents", "bundles", "markers", "receipts"):
            _regular_directory(self.run_root / name, name)
        identity_path = self.run_root / "frozen_worker_identity.json"
        identity_bytes = canonical_json_bytes(asdict(worker_identity))
        if identity_path.exists():
            if identity_path.is_symlink() or identity_path.read_bytes() != identity_bytes:
                raise DevelopmentPersistenceError("frozen worker identity drifted")
        else:
            _create_only(identity_path, identity_bytes)

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
            acquired_at_utc=_utc_now(),
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

    def create_intent(
        self,
        lease: PersistentLease,
        *,
        shard_id: str,
        unit_id: str,
        unit_index: int,
        attempt_index: int,
        parent_attempt_intent_digest: str | None,
        now_epoch_seconds: int,
    ) -> UnitIntent:
        self._require_active_lease(lease, now_epoch_seconds)
        _identity(shard_id, "shard_id")
        _identity(unit_id, "unit_id")
        if type(unit_index) is not int or unit_index < 0:
            raise DevelopmentPersistenceError("unit index is invalid")
        if type(attempt_index) is not int or not 0 <= attempt_index < MAXIMUM_ATTEMPTS:
            raise DevelopmentPersistenceError("attempt index exceeds frozen limit")
        if parent_attempt_intent_digest is not None:
            _digest(parent_attempt_intent_digest, "parent attempt intent")
        if self._committed_marker_paths(unit_id):
            raise DevelopmentPersistenceError("committed unit cannot be rerun")
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
            if parent_attempt_intent_digest != prior.digest():
                raise DevelopmentPersistenceError("retry parent intent drifted")
        intent = UnitIntent(
            schema_version=SCHEMA_VERSION,
            protocol_digest=self.worker_identity.protocol_digest,
            revision=self.worker_identity.revision,
            run_id=self.run_id,
            shard_id=shard_id,
            unit_id=unit_id,
            unit_index=unit_index,
            attempt_index=attempt_index,
            session_id=lease.session_id,
            fencing_token=lease.fencing_token,
            worker_identity_digest=self.worker_identity.digest(),
            parent_attempt_intent_digest=parent_attempt_intent_digest,
            created_at_utc=_utc_now(),
        )
        intent.validate(self.worker_identity)
        _create_only(self._intent_path(unit_id, attempt_index), canonical_json_bytes(intent.payload()))
        return intent

    def commit_unit(
        self,
        lease: PersistentLease,
        intent: UnitIntent,
        *,
        members: Mapping[str, bytes],
        raw_secret_values: Sequence[str] = (),
        now_epoch_seconds: int,
    ) -> CommittedUnit:
        self._require_active_lease(lease, now_epoch_seconds)
        if intent.fencing_token != lease.fencing_token or intent.session_id != lease.session_id:
            raise DevelopmentPersistenceError("intent does not belong to active lease")
        intent_path = self._intent_path(intent.unit_id, intent.attempt_index)
        if _read_canonical_json(intent_path, "unit intent") != intent.payload():
            raise DevelopmentPersistenceError("unit intent bytes drifted")
        if not members:
            raise DevelopmentPersistenceError("unit bundle cannot be empty")
        normalized: dict[str, bytes] = {}
        for name, payload in members.items():
            safe_name = _safe_member_path(name)
            if safe_name == "artifact_manifest.json":
                raise DevelopmentPersistenceError("artifact manifest name is reserved")
            if type(payload) is not bytes:
                raise DevelopmentPersistenceError("bundle members must be bytes")
            normalized[safe_name] = payload
        if len(normalized) != len(members):
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
            bundle_sha256=bundle_digest,
            bundle_bytes=len(archive_bytes),
            artifact_manifest_digest=artifact_manifest.digest(),
            worker_identity_digest=self.worker_identity.digest(),
            parent_attempt_intent_digest=intent.parent_attempt_intent_digest,
            committed_at_utc=_utc_now(),
        )
        marker_bytes = canonical_json_bytes(marker.payload())
        _reject_secrets((marker_bytes,), raw_secret_values)
        _create_only(self._marker_path(intent.unit_id, intent.attempt_index), marker_bytes)
        self._verify_committed(marker)
        return marker

    def recover(self) -> RecoveryReport:
        commits = tuple(self._verified_commits())
        committed_ids = {item.unit_id for item in commits}
        committed_retry_parents = {
            item.parent_attempt_intent_digest
            for item in commits
            if item.parent_attempt_intent_digest is not None
        }
        interrupted: list[InterruptedAttempt] = []
        next_attempts: dict[str, int] = {}
        for path in sorted((self.run_root / "intents").glob("*.json")):
            payload = _read_canonical_json(path, "unit intent")
            intent = UnitIntent(**payload)
            intent.validate(self.worker_identity)
            marker_path = self._marker_path(intent.unit_id, intent.attempt_index)
            if marker_path.exists():
                continue
            if intent.unit_id in committed_ids:
                if intent.digest() in committed_retry_parents:
                    continue
                raise DevelopmentPersistenceError("unbound dangling intent exists after a committed attempt")
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
        for marker in commits:
            next_attempts.pop(marker.unit_id, None)
        if any(value >= MAXIMUM_ATTEMPTS for value in next_attempts.values()):
            raise DevelopmentPersistenceError("interrupted unit exhausted frozen attempts")
        ledger_payload = [item.payload() for item in sorted(commits, key=lambda item: item.unit_index)]
        return RecoveryReport(
            committed_units=tuple(sorted(commits, key=lambda item: item.unit_index)),
            interrupted_attempts=tuple(interrupted),
            next_attempt_by_unit=tuple(sorted(next_attempts.items())),
            ledger_digest=canonical_digest(ledger_payload),
        )

    def next_attempt_index(self, unit_id: str) -> int:
        _identity(unit_id, "unit_id")
        if self._committed_marker_paths(unit_id):
            raise DevelopmentPersistenceError("committed unit has no next attempt")
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
        if next_index >= MAXIMUM_ATTEMPTS:
            raise DevelopmentPersistenceError("frozen attempt budget exhausted")
        return next_index

    def write_session_receipt(
        self,
        receipt: SessionReceipt,
        *,
        raw_secret_values: Sequence[str] = (),
    ) -> Path:
        if receipt.schema_version != DIAGNOSTIC_SCHEMA_VERSION:
            raise DevelopmentPersistenceError("session receipt schema drifted")
        if receipt.run_id != self.run_id or receipt.revision != self.worker_identity.revision:
            raise DevelopmentPersistenceError("session receipt frozen identity drifted")
        if receipt.package_sha256 != self.worker_identity.package_sha256:
            raise DevelopmentPersistenceError("session receipt package identity drifted")
        if (
            isinstance(receipt.walltime_seconds, bool)
            or not isinstance(receipt.walltime_seconds, (int, float))
            or not 0.0 <= float(receipt.walltime_seconds) < HARD_SESSION_CAP_SECONDS
        ):
            raise DevelopmentPersistenceError("session walltime violates hard cap")
        if receipt.soft_stop_seconds != SOFT_STOP_SECONDS or receipt.hard_session_cap_seconds != HARD_SESSION_CAP_SECONDS:
            raise DevelopmentPersistenceError("session stop policy drifted")
        if receipt.gpu_mix_policy != GPU_MIX_POLICY:
            raise DevelopmentPersistenceError("GPU mix policy drifted")
        _digest(receipt.environment_digest, "session environment")
        for value in receipt.public_secret_identity_digests:
            _digest(value, "public secret identity")
        payload = canonical_json_bytes(asdict(receipt))
        _reject_secrets((payload,), raw_secret_values)
        path = self.run_root / "receipts" / f"{_identity(receipt.session_id, 'session_id')}.json"
        _create_only(path, payload)
        return path

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

    def _verify_committed(self, marker: CommittedUnit) -> None:
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
        for role in ("protocol_digest", "intent_digest", "bundle_sha256", "artifact_manifest_digest", "worker_identity_digest"):
            _digest(getattr(marker, role), f"marker {role}")
        bundle_path = self.run_root / "bundles" / f"sha256_{marker.bundle_sha256}.zip"
        if file_sha256(bundle_path) != marker.bundle_sha256 or bundle_path.stat().st_size != marker.bundle_bytes:
            raise DevelopmentPersistenceError("committed bundle digest or size drifted")
        with ZipFile(bundle_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or "artifact_manifest.json" not in names:
                raise DevelopmentPersistenceError("bundle member identities are invalid")
            for info in infos:
                _safe_member_path(info.filename)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode) or info.is_dir():
                    raise DevelopmentPersistenceError("bundle links and directories are forbidden")
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
                or type(item["size_bytes"]) is not int
                or item["size_bytes"] < 0
                or _DIGEST.fullmatch(item["sha256"]) is None
                for item in manifest_value["members"]
            ):
                raise DevelopmentPersistenceError("artifact manifest members are invalid")
            expected = {item["path"]: item for item in manifest_value.get("members", [])}
            if set(expected) != set(names) - {"artifact_manifest.json"}:
                raise DevelopmentPersistenceError("artifact manifest coverage drifted")
            for name, item in expected.items():
                payload = archive.read(name)
                if len(payload) != item["size_bytes"] or sha256(payload).hexdigest() != item["sha256"]:
                    raise DevelopmentPersistenceError("bundle member digest drifted")

    def _verified_commits(self) -> list[CommittedUnit]:
        commits: list[CommittedUnit] = []
        seen_units: set[str] = set()
        for path in sorted((self.run_root / "markers").glob("*.COMMITTED.json")):
            payload = _read_canonical_json(path, "COMMITTED marker")
            marker = CommittedUnit(**payload)
            if marker.unit_id in seen_units:
                raise DevelopmentPersistenceError("multiple committed attempts exist for one unit")
            self._verify_committed(marker)
            intent = _read_canonical_json(
                self._intent_path(marker.unit_id, marker.attempt_index), "unit intent"
            )
            intent_object = UnitIntent(**intent)
            intent_object.validate(self.worker_identity)
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
            seen_units.add(marker.unit_id)
            commits.append(marker)
        return commits

    def _load_leases(self) -> list[PersistentLease]:
        leases: list[PersistentLease] = []
        for path in sorted((self.run_root / "leases").glob("fence_*.json")):
            lease = PersistentLease(**_read_canonical_json(path, "lease"))
            lease.validate()
            if lease.run_id != self.run_id or lease.worker_identity_digest != self.worker_identity.digest():
                raise DevelopmentPersistenceError("lease frozen identity drifted")
            leases.append(lease)
        tokens = [item.fencing_token for item in leases]
        if tokens != list(range(1, len(tokens) + 1)):
            raise DevelopmentPersistenceError("lease fencing history is not contiguous")
        return leases

    def _require_active_lease(self, lease: PersistentLease, now_epoch_seconds: int) -> None:
        lease.validate()
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
