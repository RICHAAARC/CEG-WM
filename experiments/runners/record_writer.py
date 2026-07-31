"""Atomic, schema-checked writer for internal governed record collections."""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator, Mapping

from experiments.protocol.internal_record_registry import (
    INTERNAL_RECORD_FIELD_NAMES,
    INTERNAL_RECORD_SCHEMA_BINDINGS,
)
from experiments.protocol.internal_case import (
    FrozenCaseInputManifest,
    InternalCaseManifestEntry,
    derive_internal_record_id,
)
from experiments.protocol.internal_records import (
    BranchScoreTrace,
    DecisionTrace,
    DetectorTrace,
    GeometryTrace,
    InternalValidationRecord,
    KeyControlTrace,
    PromotionGateAssessment,
    ProvenanceTrace,
    RoutingTrace,
    RunCaseRecordCollection,
    ThresholdTrace,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    FrozenSplitManifest,
)
from experiments.protocol.internal_validation import (
    FrozenInternalValidationProtocol,
    validate_run_case_record_collection,
)
from experiments.protocol.c1_hf_threshold_fit_records import (
    C1_HF_THRESHOLD_FIT_MAXIMUM_ATTEMPTS,
    C1_HF_THRESHOLD_FIT_RECORD_SCHEMA_VERSION,
    C1_HF_THRESHOLD_FIT_SPLIT,
    C1HfThresholdFitAttemptRecord,
    C1HfThresholdFitRecordIdentity,
    C1HfThresholdFitUnitRecordCollection,
    canonical_c1_hf_threshold_fit_record_bytes,
    derive_c1_hf_threshold_fit_attempt_id,
    load_c1_hf_threshold_fit_record_collection,
    validate_c1_hf_threshold_fit_record_collection,
)


_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class GovernedRecordWriterError(ValueError):
    """A record write, resume, or replay boundary failed closed."""


class C1HfThresholdFitRecordWriter:
    """Typed, incremental, atomic writer for exactly one pre-tau C1 unit."""

    def __init__(
        self,
        *,
        records_root: str | Path,
        identity: C1HfThresholdFitRecordIdentity,
    ) -> None:
        if type(identity) is not C1HfThresholdFitRecordIdentity:
            raise GovernedRecordWriterError("C1 record identity exact type is required")
        identity.validate()
        root = Path(records_root)
        if not root.is_absolute():
            raise GovernedRecordWriterError("records_root must be absolute")
        self._identity = deepcopy(identity)
        self._path = (
            root
            / identity.run_id
            / "threshold_fit"
            / f"shard_{identity.shard_index:02d}"
            / f"unit_{identity.unit_index:04d}.json"
        )
        self._lock_path = self._path.parent / f".{self._path.name}.lock"

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> C1HfThresholdFitUnitRecordCollection | None:
        with self._locked():
            if not self._path.exists():
                return None
            return self._load_unlocked()

    def append_attempt(
        self,
        attempt: C1HfThresholdFitAttemptRecord,
    ) -> C1HfThresholdFitUnitRecordCollection:
        if type(attempt) is not C1HfThresholdFitAttemptRecord:
            raise GovernedRecordWriterError("C1 attempt exact type is required")
        attempt.validate()
        with self._locked():
            existing = self._load_unlocked() if self._path.exists() else None
            prior = existing.attempts if existing is not None else ()
            for persisted in prior:
                if persisted.attempt_id != attempt.attempt_id:
                    continue
                if persisted == attempt:
                    return existing  # type: ignore[return-value]
                raise GovernedRecordWriterError("C1 attempt identity conflict")
            if prior:
                terminal = prior[-1].status in {"success", "excluded"} or (
                    prior[-1].failure_class
                    in {"execution_failure", "scientific_failure"}
                )
                if len(prior) >= C1_HF_THRESHOLD_FIT_MAXIMUM_ATTEMPTS or terminal:
                    raise GovernedRecordWriterError(
                        "C1 attempt continues after terminal outcome"
                    )
            expected_index = len(prior)
            if (
                attempt.attempt_index != expected_index
                or attempt.attempt_id
                != derive_c1_hf_threshold_fit_attempt_id(
                    self._identity,
                    expected_index,
                )
            ):
                raise GovernedRecordWriterError("C1 attempt sequence identity drifted")
            collection = C1HfThresholdFitUnitRecordCollection(
                schema_version=C1_HF_THRESHOLD_FIT_RECORD_SCHEMA_VERSION,
                split=C1_HF_THRESHOLD_FIT_SPLIT,
                identity=deepcopy(self._identity),
                attempts=(*prior, attempt),
            )
            validate_c1_hf_threshold_fit_record_collection(
                collection,
                expected_identity=self._identity,
            )
            self._write_atomic(collection)
            return collection

    def _load_unlocked(self) -> C1HfThresholdFitUnitRecordCollection:
        if not self._path.is_file() or self._path.is_symlink():
            raise GovernedRecordWriterError("C1 record path must be a regular file")
        try:
            return load_c1_hf_threshold_fit_record_collection(
                self._path,
                expected_identity=self._identity,
            )
        except ValueError as exc:
            raise GovernedRecordWriterError("C1 record replay failed closed") from exc

    def _write_atomic(
        self,
        collection: C1HfThresholdFitUnitRecordCollection,
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_c1_hf_threshold_fit_record_bytes(collection)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                dir=self._path.parent,
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._path)
            directory_fd = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class FrozenRecordBindings:
    """Run-wide identities that every record must bind exactly."""

    run_id: str
    case_id: str
    input_manifest_digest: str
    method_code_revision: str
    candidate_config_digest: str
    method_config_digest: str
    execution_config_digest: str
    model_revision: str
    environment_digest: str
    resource_identity_digest: str

    def __post_init__(self) -> None:
        for role in ("run_id", "case_id"):
            value = getattr(self, role)
            if type(value) is not str or not _SAFE_ID_PATTERN.fullmatch(value):
                raise GovernedRecordWriterError(f"{role} is not a safe stable identity")
        for role in (
            "input_manifest_digest",
            "candidate_config_digest",
            "method_config_digest",
            "execution_config_digest",
            "environment_digest",
            "resource_identity_digest",
        ):
            if not _DIGEST_PATTERN.fullmatch(getattr(self, role)):
                raise GovernedRecordWriterError(f"{role} must be a SHA-256 digest")
        for role in ("method_code_revision", "model_revision"):
            if not _REVISION_PATTERN.fullmatch(getattr(self, role)):
                raise GovernedRecordWriterError(
                    f"{role} must be an exact 40-character revision"
                )


def canonical_record_digest(record: InternalValidationRecord) -> str:
    """Digest one exact serialized record without treating it as a claim."""

    return sha256(_canonical_json_bytes(record.to_dict())).hexdigest()


class GovernedRecordWriter:
    """The only project layer authorized to materialize internal formal records."""

    def __init__(
        self,
        *,
        records_root: str | Path,
        frozen_protocol: FrozenInternalValidationProtocol,
        split_manifest: FrozenSplitManifest,
        input_manifest: FrozenCaseInputManifest,
        bindings: FrozenRecordBindings,
    ) -> None:
        if type(frozen_protocol) is not FrozenInternalValidationProtocol:
            raise GovernedRecordWriterError("frozen protocol exact type is required")
        if type(split_manifest) is not FrozenSplitManifest:
            raise GovernedRecordWriterError("split manifest exact type is required")
        if type(input_manifest) is not FrozenCaseInputManifest:
            raise GovernedRecordWriterError("input manifest exact type is required")
        if type(bindings) is not FrozenRecordBindings:
            raise GovernedRecordWriterError("frozen record bindings exact type is required")
        protocol_violations = frozen_protocol.validate()
        manifest_violations = split_manifest.validate()
        if protocol_violations:
            raise GovernedRecordWriterError(
                f"frozen protocol invalid: {','.join(protocol_violations)}"
            )
        if manifest_violations:
            raise GovernedRecordWriterError(
                f"split manifest invalid: {','.join(manifest_violations)}"
            )
        input_manifest_violations = input_manifest.validate(
            protocol=frozen_protocol,
            split_manifest=split_manifest,
        )
        if input_manifest_violations:
            raise GovernedRecordWriterError(
                "input manifest invalid: "
                f"{','.join(input_manifest_violations)}"
            )
        if input_manifest.digest() != bindings.input_manifest_digest:
            raise GovernedRecordWriterError(
                "input manifest digest differs from frozen record bindings"
            )
        case_entries = tuple(
            entry
            for entry in input_manifest.entries
            if entry.analysis_unit_identity.case_id == bindings.case_id
        )
        if not case_entries:
            raise GovernedRecordWriterError(
                "input manifest has no entry for the bound case"
            )
        root = Path(records_root)
        if not root.is_absolute():
            raise GovernedRecordWriterError("records_root must be absolute")
        self._records_root = root
        self._protocol_snapshot = _canonical_json_bytes(asdict(frozen_protocol))
        self._split_manifest_snapshot = _canonical_json_bytes(
            asdict(split_manifest)
        )
        self._input_manifest_snapshot = _canonical_json_bytes(
            asdict(input_manifest)
        )
        self._bindings_snapshot = _canonical_json_bytes(asdict(bindings))
        self._construction_anchor_digest = sha256(
            b"\0".join(
                (
                    self._protocol_snapshot,
                    self._split_manifest_snapshot,
                    self._input_manifest_snapshot,
                    self._bindings_snapshot,
                )
            )
        ).hexdigest()
        self._protocol = deepcopy(frozen_protocol)
        self._split_manifest = deepcopy(split_manifest)
        self._input_manifest = deepcopy(input_manifest)
        self._bindings = deepcopy(bindings)
        self._registered_fields = INTERNAL_RECORD_FIELD_NAMES
        self._path = root / self._bindings.run_id / f"{self._bindings.case_id}.json"
        self._lock_path = (
            root / self._bindings.run_id / f".{self._bindings.case_id}.lock"
        )
        self._assert_internal_anchors()

    def assert_context_anchors(
        self,
        *,
        frozen_protocol: FrozenInternalValidationProtocol,
        split_manifest: FrozenSplitManifest,
        input_manifest: FrozenCaseInputManifest,
        bindings: FrozenRecordBindings,
    ) -> None:
        """Reject context objects that differ from construction-time anchors."""

        if type(frozen_protocol) is not FrozenInternalValidationProtocol:
            raise GovernedRecordWriterError("context protocol exact type is required")
        if type(split_manifest) is not FrozenSplitManifest:
            raise GovernedRecordWriterError(
                "context split manifest exact type is required"
            )
        if type(input_manifest) is not FrozenCaseInputManifest:
            raise GovernedRecordWriterError(
                "context input manifest exact type is required"
            )
        if type(bindings) is not FrozenRecordBindings:
            raise GovernedRecordWriterError("context bindings exact type is required")
        observed = (
            _canonical_json_bytes(asdict(frozen_protocol)),
            _canonical_json_bytes(asdict(split_manifest)),
            _canonical_json_bytes(asdict(input_manifest)),
            _canonical_json_bytes(asdict(bindings)),
        )
        expected = (
            self._protocol_snapshot,
            self._split_manifest_snapshot,
            self._input_manifest_snapshot,
            self._bindings_snapshot,
        )
        if observed != expected:
            raise GovernedRecordWriterError(
                "runner context drifted from writer construction anchors"
            )

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> RunCaseRecordCollection | None:
        """Load and fully replay-validate the current real record file."""

        self._assert_internal_anchors()
        with self._locked():
            return self._load_unlocked()

    def append_record(
        self,
        record: InternalValidationRecord,
        *,
        promotion_gate_assessments: tuple[PromotionGateAssessment, ...] = (),
        promotion_stop_gate_id: str | None = None,
    ) -> RunCaseRecordCollection:
        """Atomically append one pending outcome or return an identical write."""

        self._assert_internal_anchors()
        if type(record) is not InternalValidationRecord:
            raise GovernedRecordWriterError("record exact type is required")
        with self._locked():
            existing = self._load_unlocked()
            if existing is not None:
                for persisted in existing.records:
                    if persisted.record_id != record.record_id:
                        continue
                    if canonical_record_digest(persisted) == canonical_record_digest(record):
                        return existing
                    raise GovernedRecordWriterError("record identity conflict")
                _reject_completed_duplicate(existing, record)
                records = (*existing.records, record)
                if (
                    promotion_gate_assessments
                    and promotion_gate_assessments[
                        : len(existing.promotion_gate_assessments)
                    ]
                    != existing.promotion_gate_assessments
                ):
                    raise GovernedRecordWriterError(
                        "promotion assessment history conflict"
                    )
                assessments = promotion_gate_assessments or (
                    existing.promotion_gate_assessments
                )
                if (
                    existing.promotion_stop_gate_id is not None
                    and promotion_stop_gate_id
                    != existing.promotion_stop_gate_id
                ):
                    raise GovernedRecordWriterError(
                        "promotion stop identity conflict"
                    )
                stop_gate = promotion_stop_gate_id or (
                    existing.promotion_stop_gate_id
                )
            else:
                records = (record,)
                assessments = promotion_gate_assessments
                stop_gate = promotion_stop_gate_id
            collection = RunCaseRecordCollection(
                record_collection_schema_version=(
                    self._protocol.record_collection_schema_version
                ),
                run_id=self._bindings.run_id,
                case_id=self._bindings.case_id,
                protocol_id=self._protocol.protocol_id,
                protocol_version=self._protocol.protocol_version,
                protocol_digest=self._protocol.digest(),
                split_manifest_digest=self._split_manifest.digest(),
                record_schema_version=self._protocol.record_schema_version,
                maximum_record_attempts=self._protocol.maximum_record_attempts,
                records=records,
                promotion_gate_assessments=assessments,
                promotion_stop_gate_id=stop_gate,
            )
            self._validate_collection(collection)
            self._write_atomic(collection)
            return collection

    def _load_unlocked(self) -> RunCaseRecordCollection | None:
        self._assert_internal_anchors()
        if not self._path.exists():
            return None
        if not self._path.is_file() or self._path.is_symlink():
            raise GovernedRecordWriterError("record path must be a regular file")
        raw_bytes = self._path.read_bytes()
        try:
            document = json.loads(
                raw_bytes.decode("utf-8"),
                parse_constant=lambda value: (_raise_non_finite(value)),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GovernedRecordWriterError("record file is not valid UTF-8 JSON") from exc
        collection = _collection_from_mapping(document)
        canonical = _canonical_json_bytes(collection.to_dict()) + b"\n"
        if raw_bytes != canonical:
            raise GovernedRecordWriterError("record bytes drifted from canonical form")
        self._validate_collection(collection)
        return collection

    def _validate_collection(self, collection: RunCaseRecordCollection) -> None:
        self._assert_internal_anchors()
        violations = validate_run_case_record_collection(
            collection,
            self._protocol,
            self._split_manifest,
        )
        if violations:
            raise GovernedRecordWriterError(
                f"record collection schema invalid: {','.join(violations)}"
            )
        if collection.run_id != self._bindings.run_id:
            raise GovernedRecordWriterError("record collection run identity drifted")
        if collection.case_id != self._bindings.case_id:
            raise GovernedRecordWriterError("record collection case identity drifted")
        prior_record_by_unit: dict[str, InternalValidationRecord] = {}
        terminal_units: set[str] = set()
        trusted_case_entries = tuple(
            entry
            for entry in self._input_manifest.entries
            if entry.analysis_unit_identity.case_id == self._bindings.case_id
        )
        for sequence_index, record in enumerate(collection.records):
            _validate_record_bindings(record, self._bindings)
            entry = _resolve_trusted_entry(record, trusted_case_entries)
            unit_id = record.analysis_unit_identity.unit_id
            if unit_id in terminal_units:
                raise GovernedRecordWriterError(
                    "record continues after a terminal analysis-unit outcome"
                )
            prior_record = prior_record_by_unit.get(unit_id)
            _validate_record_against_expectation(
                record,
                entry=entry,
                bindings=self._bindings,
                sequence_index=sequence_index,
                prior_record=prior_record,
            )
            prior_record_by_unit[unit_id] = record
            if _record_is_terminal(record):
                terminal_units.add(unit_id)
        payload = collection.to_dict()
        unregistered = sorted(_mapping_keys(payload) - self._registered_fields)
        if unregistered:
            raise GovernedRecordWriterError(
                f"record fields are absent from field registry: {','.join(unregistered)}"
            )

    def _write_atomic(self, collection: RunCaseRecordCollection) -> None:
        self._assert_internal_anchors()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical_json_bytes(collection.to_dict()) + b"\n"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                dir=self._path.parent,
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._path)
            directory_fd = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _assert_internal_anchors(self) -> None:
        if (
            _canonical_json_bytes(asdict(self._protocol))
            != self._protocol_snapshot
            or _canonical_json_bytes(asdict(self._split_manifest))
            != self._split_manifest_snapshot
            or _canonical_json_bytes(asdict(self._input_manifest))
            != self._input_manifest_snapshot
            or _canonical_json_bytes(asdict(self._bindings))
            != self._bindings_snapshot
            or sha256(
                b"\0".join(
                    (
                        self._protocol_snapshot,
                        self._split_manifest_snapshot,
                        self._input_manifest_snapshot,
                        self._bindings_snapshot,
                    )
                )
            ).hexdigest()
            != self._construction_anchor_digest
        ):
            raise GovernedRecordWriterError(
                "writer construction anchor drift detected"
            )
        if INTERNAL_RECORD_SCHEMA_BINDINGS != {
            "record_collection_schema_version": (
                self._protocol.record_collection_schema_version
            ),
            "record_schema_version": self._protocol.record_schema_version,
        }:
            raise GovernedRecordWriterError(
                "executable record registry schema binding drifted"
            )


def _validate_record_bindings(
    record: InternalValidationRecord,
    bindings: FrozenRecordBindings,
) -> None:
    provenance = record.provenance_trace
    expected = {
        "input_manifest_digest": bindings.input_manifest_digest,
        "method_code_revision": bindings.method_code_revision,
        "candidate_config_digest": bindings.candidate_config_digest,
        "method_config_digest": bindings.method_config_digest,
        "execution_config_digest": bindings.execution_config_digest,
        "model_revision": bindings.model_revision,
        "environment_digest": bindings.environment_digest,
        "resource_identity_digest": bindings.resource_identity_digest,
    }
    for role, value in expected.items():
        if getattr(provenance, role) != value:
            raise GovernedRecordWriterError(f"record {role} drifted")


def _resolve_trusted_entry(
    record: InternalValidationRecord,
    entries: tuple[InternalCaseManifestEntry, ...],
) -> InternalCaseManifestEntry:
    matches = tuple(
        entry
        for entry in entries
        if entry.analysis_unit_identity == record.analysis_unit_identity
        and entry.split == record.split
    )
    if len(matches) != 1:
        raise GovernedRecordWriterError(
            "record must resolve to exactly one frozen input-manifest entry"
        )
    return matches[0]


def _validate_record_against_expectation(
    record: InternalValidationRecord,
    *,
    entry: InternalCaseManifestEntry,
    bindings: FrozenRecordBindings,
    sequence_index: int,
    prior_record: InternalValidationRecord | None,
) -> None:
    if record.analysis_unit_identity != entry.analysis_unit_identity:
        raise GovernedRecordWriterError(
            "record analysis-unit identity drifted"
        )
    if record.split != entry.split:
        raise GovernedRecordWriterError("record split drifted")
    if record.record_sequence_index != sequence_index:
        raise GovernedRecordWriterError("record sequence identity drifted")

    expected_attempt_index = (
        0 if prior_record is None else prior_record.record_attempt_index + 1
    )
    if record.record_attempt_index != expected_attempt_index:
        raise GovernedRecordWriterError("record attempt identity drifted")
    expected_record_id = derive_internal_record_id(
        run_id=bindings.run_id,
        case_id=bindings.case_id,
        input_manifest_digest=bindings.input_manifest_digest,
        analysis_unit_identity=entry.analysis_unit_identity,
        attempt_index=expected_attempt_index,
    )
    if record.record_id != expected_record_id:
        raise GovernedRecordWriterError("record deterministic identity drifted")
    expected_parent = None if prior_record is None else prior_record.record_id
    if record.retry_of_record_id != expected_parent:
        raise GovernedRecordWriterError("record retry lineage drifted")

    provenance = record.provenance_trace
    per_unit_provenance = {
        "input_artifact_digest": entry.input_artifact_digest,
        "attack_config_digest": entry.attack_config_digest,
        "metric_set_digest": entry.metric_set_digest,
    }
    for role, expected in per_unit_provenance.items():
        if getattr(provenance, role) != expected:
            raise GovernedRecordWriterError(f"record {role} drifted")

    if record.routing_trace != entry.routing_trace:
        raise GovernedRecordWriterError("record routing trace drifted")
    if record.key_control_trace != entry.key_control_trace:
        raise GovernedRecordWriterError("record key-control trace drifted")

    expectation = entry.execution_expectation
    detector = record.detector_trace
    expected_detector = {
        "raw_detector_identity": expectation.raw_detector_identity,
        "rectified_detector_identity": expectation.rectified_detector_identity,
        "raw_detector_config_digest": expectation.raw_detector_config_digest,
        "rectified_detector_config_digest": (
            expectation.rectified_detector_config_digest
        ),
        "raw_preprocessing_identity": expectation.raw_preprocessing_identity,
        "rectified_preprocessing_identity": (
            expectation.rectified_preprocessing_identity
        ),
    }
    for role, expected in expected_detector.items():
        if getattr(detector, role) != expected:
            raise GovernedRecordWriterError(f"record {role} drifted")

    threshold = record.threshold_trace
    expected_threshold = {
        "raw_threshold_identity": expectation.raw_threshold_identity,
        "rectified_threshold_identity": (
            expectation.rectified_threshold_identity
        ),
        "tau": expectation.tau,
        "tau_rescue": expectation.tau_rescue,
    }
    for role, expected in expected_threshold.items():
        if getattr(threshold, role) != expected:
            raise GovernedRecordWriterError(f"record {role} drifted")

    geometry = record.geometry_trace
    if (
        geometry.geometry_operation_identity
        != expectation.geometry_operation_identity
    ):
        raise GovernedRecordWriterError(
            "record geometry operation identity drifted"
        )
    if (
        geometry.geometry_reliability_config_digest
        != expectation.geometry_reliability_config_digest
    ):
        raise GovernedRecordWriterError(
            "record geometry reliability configuration digest drifted"
        )


def _record_is_terminal(record: InternalValidationRecord) -> bool:
    return (
        record.execution_status in {"success", "excluded"}
        or (
            record.execution_status == "failed"
            and record.failure_class
            in {"execution_failure", "scientific_failure"}
        )
    )


def _reject_completed_duplicate(
    collection: RunCaseRecordCollection,
    record: InternalValidationRecord,
) -> None:
    identity = record.analysis_unit_identity
    prior = [
        item
        for item in collection.records
        if item.analysis_unit_identity == identity
    ]
    if not prior:
        return
    last = max(prior, key=lambda item: item.record_attempt_index)
    completed = (
        last.execution_status in {"success", "excluded"}
        or (
            last.execution_status == "failed"
            and last.failure_class in {"execution_failure", "scientific_failure"}
        )
    )
    if completed:
        raise GovernedRecordWriterError("completed analysis unit cannot be duplicated")


def _mapping_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        result = set(value)
        for nested in value.values():
            result.update(_mapping_keys(nested))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for nested in value:
            result.update(_mapping_keys(nested))
        return result
    return set()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GovernedRecordWriterError("record is not canonical JSON data") from exc


def _raise_non_finite(value: str) -> None:
    raise GovernedRecordWriterError(f"non-finite JSON value is forbidden: {value}")


def _exact_mapping(
    value: object,
    expected: set[str],
    role: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise GovernedRecordWriterError(f"{role} fields drifted")
    return value


def _collection_from_mapping(value: object) -> RunCaseRecordCollection:
    raw = _exact_mapping(
        value,
        {
            "record_collection_schema_version",
            "run_id",
            "case_id",
            "protocol_id",
            "protocol_version",
            "protocol_digest",
            "split_manifest_digest",
            "record_schema_version",
            "maximum_record_attempts",
            "records",
            "promotion_gate_assessments",
            "promotion_stop_gate_id",
        },
        "record collection",
    )
    if type(raw["records"]) is not list or type(raw["promotion_gate_assessments"]) is not list:
        raise GovernedRecordWriterError("record collection arrays are invalid")
    return RunCaseRecordCollection(
        record_collection_schema_version=raw["record_collection_schema_version"],
        run_id=raw["run_id"],
        case_id=raw["case_id"],
        protocol_id=raw["protocol_id"],
        protocol_version=raw["protocol_version"],
        protocol_digest=raw["protocol_digest"],
        split_manifest_digest=raw["split_manifest_digest"],
        record_schema_version=raw["record_schema_version"],
        maximum_record_attempts=raw["maximum_record_attempts"],
        records=tuple(_record_from_mapping(item) for item in raw["records"]),
        promotion_gate_assessments=tuple(
            _assessment_from_mapping(item)
            for item in raw["promotion_gate_assessments"]
        ),
        promotion_stop_gate_id=raw["promotion_stop_gate_id"],
    )


def _record_from_mapping(value: object) -> InternalValidationRecord:
    raw = _exact_mapping(
        value,
        {
            "record_id",
            "run_id",
            "protocol_id",
            "protocol_version",
            "record_schema_version",
            "analysis_unit_identity",
            "split",
            "record_sequence_index",
            "record_attempt_index",
            "execution_status",
            "failure_class",
            "failure_reason",
            "exclusion_reason",
            "exclusion_rule_id",
            "retry_of_record_id",
            "detector_trace",
            "branch_score_trace",
            "routing_trace",
            "geometry_trace",
            "threshold_trace",
            "key_control_trace",
            "decision_trace",
            "provenance_trace",
        },
        "record",
    )
    return InternalValidationRecord(
        **{
            **raw,
            "analysis_unit_identity": AnalysisUnitIdentity(
                **_exact_mapping(
                    raw["analysis_unit_identity"],
                    {
                        "unit_id",
                        "case_id",
                        "source_cluster_id",
                        "prompt_digest",
                        "generation_seed",
                        "image_lineage_digest",
                        "registered_key_family_digest",
                    },
                    "analysis unit identity",
                )
            ),
            "detector_trace": DetectorTrace(**_trace_mapping(raw, "detector_trace", DetectorTrace)),
            "branch_score_trace": BranchScoreTrace(
                **_trace_mapping(raw, "branch_score_trace", BranchScoreTrace)
            ),
            "routing_trace": RoutingTrace(**_trace_mapping(raw, "routing_trace", RoutingTrace)),
            "geometry_trace": GeometryTrace(**_trace_mapping(raw, "geometry_trace", GeometryTrace)),
            "threshold_trace": ThresholdTrace(
                **_trace_mapping(raw, "threshold_trace", ThresholdTrace)
            ),
            "key_control_trace": KeyControlTrace(
                **_trace_mapping(raw, "key_control_trace", KeyControlTrace)
            ),
            "decision_trace": DecisionTrace(
                **_trace_mapping(raw, "decision_trace", DecisionTrace)
            ),
            "provenance_trace": ProvenanceTrace(
                **_trace_mapping(raw, "provenance_trace", ProvenanceTrace)
            ),
        }
    )


def _assessment_from_mapping(value: object) -> PromotionGateAssessment:
    raw = _exact_mapping(
        value,
        {"gate_id", "gate_status", "evidence_record_ids", "stop_outcome"},
        "promotion gate assessment",
    )
    if type(raw["evidence_record_ids"]) is not list:
        raise GovernedRecordWriterError(
            "promotion gate evidence_record_ids must be an array"
        )
    return PromotionGateAssessment(
        gate_id=raw["gate_id"],
        gate_status=raw["gate_status"],
        evidence_record_ids=tuple(raw["evidence_record_ids"]),
        stop_outcome=raw["stop_outcome"],
    )


def _trace_mapping(
    record: Mapping[str, object],
    role: str,
    dataclass_type: type,
) -> dict[str, Any]:
    fields = set(dataclass_type.__dataclass_fields__)
    return _exact_mapping(record[role], fields, role)
