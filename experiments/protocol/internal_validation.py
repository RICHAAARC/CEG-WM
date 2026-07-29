"""冻结内部科学验证协议配置的加载、摘要与完整性检查。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.protocol.internal_matrix import (
    REQUIRED_METHOD_RESPONSIBILITIES,
    SPLIT_PREREQUISITE_GATES,
)
from experiments.protocol.internal_records import (
    INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION,
    INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION,
    MAXIMUM_RECORD_ATTEMPTS,
    RunCaseRecordCollection,
    _validate_run_case_record_collection_structure,
)
from experiments.protocol.internal_splits import (
    FrozenSplitManifest,
    INTERNAL_VALIDATION_PROTOCOL_ID,
    INTERNAL_VALIDATION_PROTOCOL_VERSION,
    INTERNAL_VALIDATION_SPLITS,
)


INTERNAL_VALIDATION_PROTOCOL_KIND = "internal_scientific_validation"
RECORD_COLLECTION_BINDING_FIELDS = (
    "protocol_digest",
    "split_manifest_digest",
)
RETRYABLE_PARENT_STATUS_ORDER = ("failed", "retry")
RETRY_PARENT_REQUIRED_AFTER_ATTEMPT_ZERO = True
SPLIT_ASSIGNMENT_MODE = "explicit_source_cluster_manifest"
SOURCE_CLUSTER_IDENTITY_FIELDS = (
    "prompt_digest",
    "generation_seed",
    "image_lineage_digest",
    "registered_key_family_digest",
)
CURRENT_EXECUTION_ALLOWED_SPLIT_ORDER = INTERNAL_VALIDATION_SPLITS[:-1]
HELD_OUT_EVALUATION_ACCESS = "fail_closed_current_execution"
EXECUTION_STATUS_ORDER = ("success", "failed", "excluded", "retry")
PROMOTION_FAILURE_SEMANTICS = (
    "stop_and_record_failed_or_closed_negative_without_advancing"
)
SCIENTIFIC_CLAIM_BOUNDARY = (
    "schema_and_cpu_constraints_only_no_scientific_validity_claim"
)


@dataclass(frozen=True)
class FrozenInternalValidationProtocol:
    protocol_id: str
    protocol_version: str
    protocol_kind: str
    record_schema_version: str
    record_collection_schema_version: str
    record_collection_binding_fields: tuple[str, ...]
    maximum_record_attempts: int
    retryable_parent_statuses: tuple[str, ...]
    retry_parent_required_after_attempt_zero: bool
    split_assignment_mode: str
    source_cluster_identity_fields: tuple[str, ...]
    splits: tuple[str, ...]
    current_execution_allowed_splits: tuple[str, ...]
    held_out_evaluation_access: str
    execution_statuses: tuple[str, ...]
    method_responsibilities: tuple[str, ...]
    split_prerequisite_gates: dict[str, tuple[str, ...]]
    promotion_failure_semantics: str
    scientific_claim_boundary: str

    def digest(self) -> str:
        canonical = json.dumps(
            asdict(self),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if self.protocol_kind != INTERNAL_VALIDATION_PROTOCOL_KIND:
            violations.append("protocol_kind_invalid")
        if self.protocol_id != INTERNAL_VALIDATION_PROTOCOL_ID:
            violations.append("protocol_id_frozen_identity_mismatch")
        if self.protocol_version != INTERNAL_VALIDATION_PROTOCOL_VERSION:
            violations.append("protocol_version_frozen_identity_mismatch")
        if self.record_schema_version != INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION:
            violations.append("record_schema_version_frozen_identity_mismatch")
        if (
            self.record_collection_schema_version
            != INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION
        ):
            violations.append("record_collection_schema_version_frozen_identity_mismatch")
        if self.maximum_record_attempts != MAXIMUM_RECORD_ATTEMPTS:
            violations.append("maximum_record_attempts_frozen_value_mismatch")
        if self.record_collection_binding_fields != RECORD_COLLECTION_BINDING_FIELDS:
            violations.append("record_collection_binding_fields_invalid")
        if self.retryable_parent_statuses != RETRYABLE_PARENT_STATUS_ORDER:
            violations.append("retryable_parent_statuses_invalid")
        if (
            self.retry_parent_required_after_attempt_zero
            is not RETRY_PARENT_REQUIRED_AFTER_ATTEMPT_ZERO
        ):
            violations.append("retry_parent_required_after_attempt_zero_invalid")
        if self.split_assignment_mode != SPLIT_ASSIGNMENT_MODE:
            violations.append("split_assignment_mode_invalid")
        if self.source_cluster_identity_fields != SOURCE_CLUSTER_IDENTITY_FIELDS:
            violations.append("source_cluster_identity_fields_invalid")
        if self.splits != INTERNAL_VALIDATION_SPLITS:
            violations.append("split_identity_or_order_invalid")
        if (
            self.current_execution_allowed_splits
            != CURRENT_EXECUTION_ALLOWED_SPLIT_ORDER
        ):
            violations.append("current_execution_allowed_splits_invalid")
        if self.held_out_evaluation_access != HELD_OUT_EVALUATION_ACCESS:
            violations.append("held_out_evaluation_access_invalid")
        if self.execution_statuses != EXECUTION_STATUS_ORDER:
            violations.append("execution_statuses_invalid")
        if self.method_responsibilities != REQUIRED_METHOD_RESPONSIBILITIES:
            violations.append("method_responsibilities_invalid")
        if self.split_prerequisite_gates != SPLIT_PREREQUISITE_GATES:
            violations.append("split_prerequisite_gates_invalid")
        if self.promotion_failure_semantics != PROMOTION_FAILURE_SEMANTICS:
            violations.append("promotion_failure_semantics_invalid")
        if self.scientific_claim_boundary != SCIENTIFIC_CLAIM_BOUNDARY:
            violations.append("scientific_claim_boundary_invalid")
        return tuple(dict.fromkeys(violations))


def load_frozen_internal_validation_protocol(
    path: str | Path,
) -> FrozenInternalValidationProtocol:
    raw: dict[str, Any]
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    raw["source_cluster_identity_fields"] = tuple(raw["source_cluster_identity_fields"])
    raw["splits"] = tuple(raw["splits"])
    raw["current_execution_allowed_splits"] = tuple(raw["current_execution_allowed_splits"])
    raw["execution_statuses"] = tuple(raw["execution_statuses"])
    raw["record_collection_binding_fields"] = tuple(
        raw["record_collection_binding_fields"]
    )
    raw["retryable_parent_statuses"] = tuple(raw["retryable_parent_statuses"])
    raw["method_responsibilities"] = tuple(raw["method_responsibilities"])
    raw["split_prerequisite_gates"] = {
        split_name: tuple(gates)
        for split_name, gates in raw["split_prerequisite_gates"].items()
    }
    protocol = FrozenInternalValidationProtocol(**raw)
    violations = protocol.validate()
    if violations:
        raise ValueError(", ".join(violations))
    return protocol


def validate_run_case_record_collection(
    collection: RunCaseRecordCollection,
    frozen_protocol: FrozenInternalValidationProtocol,
    split_manifest: FrozenSplitManifest,
) -> tuple[str, ...]:
    """以精确冻结 dataclass 为 trust anchor 校验 run/case records。"""
    violations: list[str] = []
    if type(frozen_protocol) is not FrozenInternalValidationProtocol:
        violations.append("frozen_protocol_exact_type_required")
    if type(split_manifest) is not FrozenSplitManifest:
        violations.append("split_manifest_exact_type_required")
    if violations:
        return tuple(violations)

    frozen_protocol_violations = frozen_protocol.validate()
    if frozen_protocol_violations:
        violations.append("frozen_protocol_invalid")
        violations.extend(frozen_protocol_violations)
    split_manifest_violations = split_manifest.validate()
    if split_manifest_violations:
        violations.append("split_manifest_invalid")
        violations.extend(split_manifest_violations)

    violations.extend(
        _validate_run_case_record_collection_structure(
            collection,
            frozen_protocol_id=frozen_protocol.protocol_id,
            frozen_protocol_version=frozen_protocol.protocol_version,
            frozen_record_schema_version=frozen_protocol.record_schema_version,
            frozen_record_collection_schema_version=(
                frozen_protocol.record_collection_schema_version
            ),
            frozen_maximum_record_attempts=frozen_protocol.maximum_record_attempts,
            actual_protocol_digest=frozen_protocol.digest(),
            actual_split_manifest_digest=split_manifest.digest(),
            split_manifest_protocol_id=split_manifest.protocol_id,
            split_manifest_protocol_version=split_manifest.protocol_version,
            manifest_assignments=split_manifest.assignments,
        )
    )
    return tuple(dict.fromkeys(violations))
