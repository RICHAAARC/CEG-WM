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


_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class GovernedRecordWriterError(ValueError):
    """A record write, resume, or replay boundary failed closed."""


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
        bindings: FrozenRecordBindings,
    ) -> None:
        if type(frozen_protocol) is not FrozenInternalValidationProtocol:
            raise GovernedRecordWriterError("frozen protocol exact type is required")
        if type(split_manifest) is not FrozenSplitManifest:
            raise GovernedRecordWriterError("split manifest exact type is required")
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
        root = Path(records_root)
        if not root.is_absolute():
            raise GovernedRecordWriterError("records_root must be absolute")
        self._records_root = root
        self._protocol_snapshot = _canonical_json_bytes(asdict(frozen_protocol))
        self._split_manifest_snapshot = _canonical_json_bytes(
            asdict(split_manifest)
        )
        self._bindings_snapshot = _canonical_json_bytes(asdict(bindings))
        self._construction_anchor_digest = sha256(
            b"\0".join(
                (
                    self._protocol_snapshot,
                    self._split_manifest_snapshot,
                    self._bindings_snapshot,
                )
            )
        ).hexdigest()
        self._protocol = deepcopy(frozen_protocol)
        self._split_manifest = deepcopy(split_manifest)
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
        bindings: FrozenRecordBindings,
    ) -> None:
        """Reject context objects that differ from construction-time anchors."""

        if type(frozen_protocol) is not FrozenInternalValidationProtocol:
            raise GovernedRecordWriterError("context protocol exact type is required")
        if type(split_manifest) is not FrozenSplitManifest:
            raise GovernedRecordWriterError(
                "context split manifest exact type is required"
            )
        if type(bindings) is not FrozenRecordBindings:
            raise GovernedRecordWriterError("context bindings exact type is required")
        observed = (
            _canonical_json_bytes(asdict(frozen_protocol)),
            _canonical_json_bytes(asdict(split_manifest)),
            _canonical_json_bytes(asdict(bindings)),
        )
        expected = (
            self._protocol_snapshot,
            self._split_manifest_snapshot,
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
        for record in collection.records:
            _validate_record_bindings(record, self._bindings)
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
            or _canonical_json_bytes(asdict(self._bindings))
            != self._bindings_snapshot
            or sha256(
                b"\0".join(
                    (
                        self._protocol_snapshot,
                        self._split_manifest_snapshot,
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
