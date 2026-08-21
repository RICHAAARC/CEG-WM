"""Minimal fixed-denominator records for Stage-A method feasibility."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping

UnitStatus = Literal[
    "success",
    "scientific_failure",
    "operational_failure",
    "excluded",
]

_ALLOWED_STATUSES = {
    "success",
    "scientific_failure",
    "operational_failure",
    "excluded",
}


@dataclass(frozen=True, slots=True)
class StageARecord:
    """One predeclared Stage-A unit, including explicit failure outcomes.

    The schema intentionally stores only public key identity. Raw or derived key
    material is never a record field.
    """

    run_id: str
    unit_id: str
    source_cluster_id: str
    arm: str
    condition: str
    code_revision: str
    config_digest: str
    key_public_digest: str
    status: UnitStatus
    failure_reason: str | None = None
    scores: Mapping[str, float] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        required_text = {
            "run_id": self.run_id,
            "unit_id": self.unit_id,
            "source_cluster_id": self.source_cluster_id,
            "arm": self.arm,
            "condition": self.condition,
            "code_revision": self.code_revision,
            "config_digest": self.config_digest,
            "key_public_digest": self.key_public_digest,
        }
        empty_fields = [name for name, value in required_text.items() if not value.strip()]
        if empty_fields:
            raise ValueError(f"empty required fields: {', '.join(sorted(empty_fields))}")
        if self.status not in _ALLOWED_STATUSES:
            raise ValueError(f"unsupported status: {self.status}")
        if self.status == "success" and self.failure_reason is not None:
            raise ValueError("successful records cannot carry failure_reason")
        if self.status != "success" and not (self.failure_reason or "").strip():
            raise ValueError("non-success records require failure_reason")
        if self.schema_version != 1:
            raise ValueError("unsupported schema_version")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible copy without changing immutable state."""
        payload = asdict(self)
        payload["scores"] = dict(self.scores)
        payload["metrics"] = dict(self.metrics)
        return payload
