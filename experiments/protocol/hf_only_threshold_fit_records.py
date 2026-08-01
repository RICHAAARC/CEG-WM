"""Typed incremental records for the pre-tau HF-only threshold-fit GPU execution phase."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Mapping

from .internal_splits import AnalysisUnitIdentity


HF_ONLY_THRESHOLD_FIT_RECORD_SCHEMA_VERSION = (
    "ceg_wm_hf_only_threshold_fit_unit_record_v2"
)
HF_ONLY_THRESHOLD_FIT_SPLIT = "content_threshold_fit"
HF_ONLY_THRESHOLD_FIT_SHARD_COUNT = 16
HF_ONLY_THRESHOLD_FIT_UNITS_PER_SHARD = 256
HF_ONLY_THRESHOLD_FIT_UNIT_COUNT = 4096
HF_ONLY_THRESHOLD_FIT_MAXIMUM_ATTEMPTS = 3
HF_ONLY_THRESHOLD_FIT_FAILURE_CLASSES = frozenset(
    {"resource_failure", "execution_failure", "scientific_failure"}
)
HF_ONLY_THRESHOLD_FIT_REAL_EXECUTION_EVIDENCE = "real_sd35_gpu"
HF_ONLY_THRESHOLD_FIT_SYNTHETIC_EXECUTION_EVIDENCE = "synthetic_cpu_fixture"
HF_ONLY_THRESHOLD_FIT_EXECUTION_EVIDENCE_KINDS = frozenset(
    {
        HF_ONLY_THRESHOLD_FIT_REAL_EXECUTION_EVIDENCE,
        HF_ONLY_THRESHOLD_FIT_SYNTHETIC_EXECUTION_EVIDENCE,
    }
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class HfOnlyThresholdFitRecordError(ValueError):
    """A typed threshold-fit record failed closed."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _exact_keys(raw: Mapping[str, object], expected: set[str], role: str) -> None:
    if set(raw) != expected:
        raise HfOnlyThresholdFitRecordError(f"{role} fields drifted")


def _require_digest(value: object, role: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise HfOnlyThresholdFitRecordError(f"{role} must be SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class HfOnlyThresholdFitFactRecord:
    score_float64_hex: str
    image_digest: str
    input_artifact_digest: str
    detector_identity: str
    detector_config_digest: str
    detection_key_public_digest: str
    selected_device: str

    def score(self) -> float:
        try:
            value = float.fromhex(self.score_float64_hex)
        except (TypeError, ValueError) as exc:
            raise HfOnlyThresholdFitRecordError(
                "score_float64_hex is invalid"
            ) from exc
        if not math.isfinite(value) or value.hex() != self.score_float64_hex:
            raise HfOnlyThresholdFitRecordError(
                "score_float64_hex is not canonical finite binary64"
            )
        return value

    def validate(self) -> None:
        self.score()
        for role in (
            "image_digest",
            "input_artifact_digest",
            "detector_config_digest",
            "detection_key_public_digest",
        ):
            _require_digest(getattr(self, role), role)
        if self.input_artifact_digest != self.image_digest:
            raise HfOnlyThresholdFitRecordError(
                "threshold-fit input artifact must be the scored public image"
            )
        if type(self.detector_identity) is not str or not self.detector_identity:
            raise HfOnlyThresholdFitRecordError("detector identity is invalid")
        if self.selected_device != "cuda:0":
            raise HfOnlyThresholdFitRecordError("selected device must be cuda:0")


@dataclass(frozen=True, slots=True)
class HfOnlyThresholdFitAttemptRecord:
    attempt_id: str
    attempt_index: int
    resource_identity_digest: str
    status: str
    failure_class: str | None
    failure_type: str | None
    exclusion_rule_id: str | None
    retry_of_attempt_id: str | None
    fact: HfOnlyThresholdFitFactRecord | None

    def validate(self) -> None:
        _require_digest(self.attempt_id, "attempt_id")
        _require_digest(
            self.resource_identity_digest,
            "resource_identity_digest",
        )
        if (
            type(self.attempt_index) is not int
            or not 0 <= self.attempt_index < HF_ONLY_THRESHOLD_FIT_MAXIMUM_ATTEMPTS
        ):
            raise HfOnlyThresholdFitRecordError("attempt index is invalid")
        if self.attempt_index == 0 and self.retry_of_attempt_id is not None:
            raise HfOnlyThresholdFitRecordError("initial attempt retry parent is forbidden")
        if self.attempt_index > 0:
            _require_digest(self.retry_of_attempt_id, "retry_of_attempt_id")
        if self.status == "success":
            if (
                self.failure_class is not None
                or self.failure_type is not None
                or self.exclusion_rule_id is not None
                or type(self.fact) is not HfOnlyThresholdFitFactRecord
            ):
                raise HfOnlyThresholdFitRecordError("success attempt fields drifted")
            self.fact.validate()
            return
        if self.status == "excluded":
            if (
                self.failure_class is not None
                or self.failure_type is not None
                or type(self.exclusion_rule_id) is not str
                or _SAFE_ID.fullmatch(self.exclusion_rule_id) is None
                or self.fact is not None
            ):
                raise HfOnlyThresholdFitRecordError("excluded attempt fields drifted")
            return
        if self.status not in {"failed", "retry"}:
            raise HfOnlyThresholdFitRecordError("attempt status is invalid")
        if (
            self.failure_class not in HF_ONLY_THRESHOLD_FIT_FAILURE_CLASSES
            or type(self.failure_type) is not str
            or not self.failure_type
            or self.exclusion_rule_id is not None
            or self.fact is not None
        ):
            raise HfOnlyThresholdFitRecordError("failed attempt fields drifted")
        if self.status == "retry" and self.failure_class != "resource_failure":
            raise HfOnlyThresholdFitRecordError("only resource failure may retry")
        if (
            self.status == "retry"
            and self.attempt_index == HF_ONLY_THRESHOLD_FIT_MAXIMUM_ATTEMPTS - 1
        ):
            raise HfOnlyThresholdFitRecordError("final resource attempt cannot retry")
        if (
            self.status == "failed"
            and self.failure_class == "resource_failure"
            and self.attempt_index < HF_ONLY_THRESHOLD_FIT_MAXIMUM_ATTEMPTS - 1
        ):
            raise HfOnlyThresholdFitRecordError(
                "pre-final resource failure must retain retry status"
            )


@dataclass(frozen=True, slots=True)
class HfOnlyThresholdFitRecordIdentity:
    run_id: str
    committed_revision: str
    execution_evidence_kind: str
    hf_only_reference_specification_digest: str
    protocol_id: str
    protocol_version: str
    protocol_digest: str
    shard_index: int
    unit_index: int
    execution_config_digest: str
    fit_manifest_digest: str
    metric_binding_digest: str
    metric_registry_digest: str
    candidate_config_digest: str
    method_config_digest: str
    runtime_config_digest: str
    model_revision: str
    detector_identity: str
    detector_config_digest: str
    preprocessing_identity: str
    registered_key_family_digest: str
    registered_key_public_digest: str
    environment_digest: str
    analysis_unit_identity: AnalysisUnitIdentity

    def validate(self) -> None:
        if type(self.run_id) is not str or _SAFE_ID.fullmatch(self.run_id) is None:
            raise HfOnlyThresholdFitRecordError("run_id is not a safe identity")
        if _REVISION.fullmatch(self.committed_revision) is None:
            raise HfOnlyThresholdFitRecordError("committed revision is invalid")
        if self.execution_evidence_kind not in HF_ONLY_THRESHOLD_FIT_EXECUTION_EVIDENCE_KINDS:
            raise HfOnlyThresholdFitRecordError("execution evidence kind is invalid")
        if (
            type(self.shard_index) is not int
            or not 0 <= self.shard_index < HF_ONLY_THRESHOLD_FIT_SHARD_COUNT
            or type(self.unit_index) is not int
            or not 0 <= self.unit_index < HF_ONLY_THRESHOLD_FIT_UNIT_COUNT
            or self.unit_index // HF_ONLY_THRESHOLD_FIT_UNITS_PER_SHARD
            != self.shard_index
        ):
            raise HfOnlyThresholdFitRecordError("shard or unit index is invalid")
        for role in (
            "hf_only_reference_specification_digest",
            "protocol_digest",
            "execution_config_digest",
            "fit_manifest_digest",
            "metric_binding_digest",
            "metric_registry_digest",
            "candidate_config_digest",
            "method_config_digest",
            "runtime_config_digest",
            "detector_config_digest",
            "registered_key_family_digest",
            "registered_key_public_digest",
            "environment_digest",
        ):
            _require_digest(getattr(self, role), role)
        if _REVISION.fullmatch(self.model_revision) is None:
            raise HfOnlyThresholdFitRecordError("model revision is invalid")
        for role in (
            "protocol_id",
            "protocol_version",
            "detector_identity",
            "preprocessing_identity",
        ):
            if type(getattr(self, role)) is not str or not getattr(self, role):
                raise HfOnlyThresholdFitRecordError(f"{role} is invalid")
        violations = self.analysis_unit_identity.validate()
        if violations:
            raise HfOnlyThresholdFitRecordError(
                f"analysis unit identity is invalid: {','.join(violations)}"
            )
        if (
            self.analysis_unit_identity.registered_key_family_digest
            != self.registered_key_family_digest
        ):
            raise HfOnlyThresholdFitRecordError("registered key family drifted")


@dataclass(frozen=True, slots=True)
class HfOnlyThresholdFitUnitRecordCollection:
    schema_version: str
    split: str
    identity: HfOnlyThresholdFitRecordIdentity
    attempts: tuple[HfOnlyThresholdFitAttemptRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def derive_hf_only_threshold_fit_attempt_id(
    identity: HfOnlyThresholdFitRecordIdentity,
    attempt_index: int,
) -> str:
    identity.validate()
    if (
        type(attempt_index) is not int
        or not 0 <= attempt_index < HF_ONLY_THRESHOLD_FIT_MAXIMUM_ATTEMPTS
    ):
        raise HfOnlyThresholdFitRecordError("attempt index is invalid")
    return sha256(
        _canonical_bytes(
            {
                "identity": asdict(identity),
                "attempt_index": attempt_index,
                "schema_version": HF_ONLY_THRESHOLD_FIT_RECORD_SCHEMA_VERSION,
            }
        )
    ).hexdigest()


def validate_hf_only_threshold_fit_record_collection(
    collection: HfOnlyThresholdFitUnitRecordCollection,
    *,
    expected_identity: HfOnlyThresholdFitRecordIdentity | None = None,
) -> None:
    if type(collection) is not HfOnlyThresholdFitUnitRecordCollection:
        raise HfOnlyThresholdFitRecordError("record collection exact type is required")
    if (
        collection.schema_version != HF_ONLY_THRESHOLD_FIT_RECORD_SCHEMA_VERSION
        or collection.split != HF_ONLY_THRESHOLD_FIT_SPLIT
    ):
        raise HfOnlyThresholdFitRecordError("record collection identity drifted")
    collection.identity.validate()
    if expected_identity is not None and collection.identity != expected_identity:
        raise HfOnlyThresholdFitRecordError("record binding identity drifted")
    if not collection.attempts:
        raise HfOnlyThresholdFitRecordError("empty attempt collection is forbidden")
    if len(collection.attempts) > HF_ONLY_THRESHOLD_FIT_MAXIMUM_ATTEMPTS:
        raise HfOnlyThresholdFitRecordError("maximum attempt count exceeded")
    terminal_seen = False
    for expected_index, attempt in enumerate(collection.attempts):
        if terminal_seen:
            raise HfOnlyThresholdFitRecordError("attempt continues after terminal outcome")
        attempt.validate()
        if attempt.attempt_index != expected_index:
            raise HfOnlyThresholdFitRecordError("attempt sequence is not contiguous")
        if attempt.attempt_id != derive_hf_only_threshold_fit_attempt_id(
            collection.identity, expected_index
        ):
            raise HfOnlyThresholdFitRecordError("attempt identity drifted")
        if attempt.status == "success":
            assert attempt.fact is not None
            if (
                attempt.fact.detector_identity
                != collection.identity.detector_identity
                or attempt.fact.detector_config_digest
                != collection.identity.detector_config_digest
                or attempt.fact.detection_key_public_digest
                != collection.identity.registered_key_public_digest
                or attempt.fact.selected_device != "cuda:0"
            ):
                raise HfOnlyThresholdFitRecordError(
                    "success fact differs from record binding identity"
                )
        if expected_index > 0 and attempt.retry_of_attempt_id != collection.attempts[
            expected_index - 1
        ].attempt_id:
            raise HfOnlyThresholdFitRecordError("retry parent lineage drifted")
        if expected_index > 0 and collection.attempts[expected_index - 1].status not in {
            "retry",
            "failed",
        }:
            raise HfOnlyThresholdFitRecordError("retry parent status is invalid")
        terminal_seen = attempt.status in {"success", "excluded"} or (
            attempt.failure_class in {"execution_failure", "scientific_failure"}
        )
    last = collection.attempts[-1]
    if (
        len(collection.attempts) == HF_ONLY_THRESHOLD_FIT_MAXIMUM_ATTEMPTS
        and last.status in {"failed", "retry"}
        and last.failure_class == "resource_failure"
    ):
        terminal_seen = True


def parse_hf_only_threshold_fit_record_collection(
    raw: object,
) -> HfOnlyThresholdFitUnitRecordCollection:
    if type(raw) is not dict:
        raise HfOnlyThresholdFitRecordError("record collection must be an object")
    _exact_keys(raw, {"schema_version", "split", "identity", "attempts"}, "collection")
    identity_raw = raw["identity"]
    attempts_raw = raw["attempts"]
    if type(identity_raw) is not dict or type(attempts_raw) is not list:
        raise HfOnlyThresholdFitRecordError("record collection members are invalid")
    _exact_keys(
        identity_raw,
        {
            "run_id",
            "committed_revision",
            "execution_evidence_kind",
            "hf_only_reference_specification_digest",
            "protocol_id",
            "protocol_version",
            "protocol_digest",
            "shard_index",
            "unit_index",
            "execution_config_digest",
            "fit_manifest_digest",
            "metric_binding_digest",
            "metric_registry_digest",
            "candidate_config_digest",
            "method_config_digest",
            "runtime_config_digest",
            "model_revision",
            "detector_identity",
            "detector_config_digest",
            "preprocessing_identity",
            "registered_key_family_digest",
            "registered_key_public_digest",
            "environment_digest",
            "analysis_unit_identity",
        },
        "identity",
    )
    unit_raw = identity_raw["analysis_unit_identity"]
    if type(unit_raw) is not dict:
        raise HfOnlyThresholdFitRecordError("analysis unit identity must be an object")
    _exact_keys(
        unit_raw,
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
    identity = HfOnlyThresholdFitRecordIdentity(
        **{
            **{key: value for key, value in identity_raw.items() if key != "analysis_unit_identity"},
            "analysis_unit_identity": AnalysisUnitIdentity(**unit_raw),
        }
    )
    attempts: list[HfOnlyThresholdFitAttemptRecord] = []
    for attempt_raw in attempts_raw:
        if type(attempt_raw) is not dict:
            raise HfOnlyThresholdFitRecordError("attempt must be an object")
        _exact_keys(
            attempt_raw,
            {
                "attempt_id",
                "attempt_index",
                "resource_identity_digest",
                "status",
                "failure_class",
                "failure_type",
                "exclusion_rule_id",
                "retry_of_attempt_id",
                "fact",
            },
            "attempt",
        )
        fact_raw = attempt_raw["fact"]
        fact: HfOnlyThresholdFitFactRecord | None
        if fact_raw is None:
            fact = None
        elif type(fact_raw) is dict:
            _exact_keys(
                fact_raw,
                {
                    "score_float64_hex",
                    "image_digest",
                    "input_artifact_digest",
                    "detector_identity",
                    "detector_config_digest",
                    "detection_key_public_digest",
                    "selected_device",
                },
                "fact",
            )
            fact = HfOnlyThresholdFitFactRecord(**fact_raw)
        else:
            raise HfOnlyThresholdFitRecordError("attempt fact is invalid")
        attempts.append(
            HfOnlyThresholdFitAttemptRecord(
                **{
                    **{key: value for key, value in attempt_raw.items() if key != "fact"},
                    "fact": fact,
                }
            )
        )
    collection = HfOnlyThresholdFitUnitRecordCollection(
        schema_version=raw["schema_version"],
        split=raw["split"],
        identity=identity,
        attempts=tuple(attempts),
    )
    validate_hf_only_threshold_fit_record_collection(collection)
    return collection


def load_hf_only_threshold_fit_record_collection(
    path: str | Path,
    *,
    expected_identity: HfOnlyThresholdFitRecordIdentity | None = None,
) -> HfOnlyThresholdFitUnitRecordCollection:
    record_path = Path(path)
    try:
        raw_bytes = record_path.read_bytes()
        raw = json.loads(
            raw_bytes.decode("utf-8"),
            parse_constant=lambda value: (_raise_non_finite(value)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HfOnlyThresholdFitRecordError("record file is not valid UTF-8 JSON") from exc
    collection = parse_hf_only_threshold_fit_record_collection(raw)
    if raw_bytes != _canonical_bytes(collection.to_dict()) + b"\n":
        raise HfOnlyThresholdFitRecordError("record bytes are not canonical")
    validate_hf_only_threshold_fit_record_collection(
        collection,
        expected_identity=expected_identity,
    )
    return collection


def replay_hf_only_threshold_fit_record_collection(
    collection: HfOnlyThresholdFitUnitRecordCollection,
    *,
    expected_identity: HfOnlyThresholdFitRecordIdentity,
) -> HfOnlyThresholdFitAttemptRecord:
    validate_hf_only_threshold_fit_record_collection(
        collection,
        expected_identity=expected_identity,
    )
    return collection.attempts[-1]


def canonical_hf_only_threshold_fit_record_bytes(
    collection: HfOnlyThresholdFitUnitRecordCollection,
) -> bytes:
    validate_hf_only_threshold_fit_record_collection(collection)
    return _canonical_bytes(collection.to_dict()) + b"\n"


def _raise_non_finite(value: str) -> object:
    raise ValueError(f"non-finite JSON constant forbidden: {value}")


__all__ = [
    "HF_ONLY_THRESHOLD_FIT_EXECUTION_EVIDENCE_KINDS",
    "HF_ONLY_THRESHOLD_FIT_FAILURE_CLASSES",
    "HF_ONLY_THRESHOLD_FIT_MAXIMUM_ATTEMPTS",
    "HF_ONLY_THRESHOLD_FIT_RECORD_SCHEMA_VERSION",
    "HF_ONLY_THRESHOLD_FIT_REAL_EXECUTION_EVIDENCE",
    "HF_ONLY_THRESHOLD_FIT_SHARD_COUNT",
    "HF_ONLY_THRESHOLD_FIT_SPLIT",
    "HF_ONLY_THRESHOLD_FIT_SYNTHETIC_EXECUTION_EVIDENCE",
    "HF_ONLY_THRESHOLD_FIT_UNIT_COUNT",
    "HF_ONLY_THRESHOLD_FIT_UNITS_PER_SHARD",
    "HfOnlyThresholdFitAttemptRecord",
    "HfOnlyThresholdFitFactRecord",
    "HfOnlyThresholdFitRecordError",
    "HfOnlyThresholdFitRecordIdentity",
    "HfOnlyThresholdFitUnitRecordCollection",
    "canonical_hf_only_threshold_fit_record_bytes",
    "derive_hf_only_threshold_fit_attempt_id",
    "load_hf_only_threshold_fit_record_collection",
    "parse_hf_only_threshold_fit_record_collection",
    "replay_hf_only_threshold_fit_record_collection",
    "validate_hf_only_threshold_fit_record_collection",
]
