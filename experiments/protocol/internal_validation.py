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
from experiments.protocol.internal_records import EXECUTION_STATUSES
from experiments.protocol.internal_splits import (
    CURRENT_EXECUTION_ALLOWED_SPLITS,
    INTERNAL_VALIDATION_SPLITS,
)


@dataclass(frozen=True)
class FrozenInternalValidationProtocol:
    protocol_id: str
    protocol_version: str
    protocol_kind: str
    record_schema_version: str
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
        for name in (
            "protocol_id",
            "protocol_version",
            "record_schema_version",
            "promotion_failure_semantics",
            "scientific_claim_boundary",
        ):
            if not getattr(self, name).strip():
                violations.append(f"{name}_missing")
        if self.protocol_kind != "internal_scientific_validation":
            violations.append("protocol_kind_invalid")
        if self.split_assignment_mode != "explicit_source_cluster_manifest":
            violations.append("split_assignment_mode_invalid")
        if self.source_cluster_identity_fields != (
            "prompt_digest",
            "generation_seed",
            "image_lineage_digest",
            "registered_key_family_digest",
        ):
            violations.append("source_cluster_identity_fields_invalid")
        if self.splits != INTERNAL_VALIDATION_SPLITS:
            violations.append("split_identity_or_order_invalid")
        if frozenset(self.current_execution_allowed_splits) != CURRENT_EXECUTION_ALLOWED_SPLITS:
            violations.append("current_execution_allowed_splits_invalid")
        if "held_out_evaluation" in self.current_execution_allowed_splits:
            violations.append("current_execution_held_out_access_forbidden")
        if self.held_out_evaluation_access != "fail_closed_current_execution":
            violations.append("held_out_evaluation_access_invalid")
        if frozenset(self.execution_statuses) != EXECUTION_STATUSES:
            violations.append("execution_statuses_invalid")
        if self.method_responsibilities != REQUIRED_METHOD_RESPONSIBILITIES:
            violations.append("method_responsibilities_invalid")
        if self.split_prerequisite_gates != SPLIT_PREREQUISITE_GATES:
            violations.append("split_prerequisite_gates_invalid")
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
