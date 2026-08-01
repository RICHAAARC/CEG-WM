"""提供 placeholder 与 random trace 字段规则。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from governance.harness.lib.naming_rules import (
    has_ordinal_identity_text,
    has_weak_semantic_token,
)


ALLOWED_GOVERNANCE_LEVELS = {
    "internal_state",
    "cross_boundary",
    "persisted_protocol",
    "evidence_bearing",
}


@dataclass(frozen=True)
class FieldRegistryRow:
    """表示字段登记表中的一行。"""

    field_name: str
    governance_level: str
    category: str
    required_suffix: str
    allowed_in_claims: str
    description: str


def load_field_registry(root: str | Path) -> dict[str, FieldRegistryRow]:
    """读取 docs/reference/field_registry.md 中的字段登记表。"""
    path = Path(root) / "docs" / "reference" / "field_registry.md"
    rows: dict[str, FieldRegistryRow] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 8 or cells[0] in {"field_name", "---"} or set(cells[0]) == {"-"}:
            continue
        rows[cells[0]] = FieldRegistryRow(
            field_name=cells[0],
            governance_level=cells[1],
            category=cells[2],
            required_suffix=cells[3],
            allowed_in_claims=cells[5],
            description=cells[7],
        )
    return rows


def validate_registry_rows(rows: dict[str, FieldRegistryRow]) -> list[dict[str, str]]:
    """校验字段登记表中的 placeholder、random、中间状态和 claim 规则。"""
    violations: list[dict[str, str]] = []
    for row in rows.values():
        if has_weak_semantic_token(row.field_name):
            violations.append({"field_name": row.field_name, "reason": "weak_semantic_field_name"})
        if has_ordinal_identity_text(row.field_name):
            violations.append(
                {
                    "field_name": row.field_name,
                    "reason": "ordinal_identity_field_name",
                }
            )
        if row.governance_level not in ALLOWED_GOVERNANCE_LEVELS:
            violations.append({"field_name": row.field_name, "reason": "invalid_governance_level"})
        if not row.description or row.description.lower() in {"none", "n/a"}:
            violations.append({"field_name": row.field_name, "reason": "field_description_required"})
        if row.category == "placeholder" and not row.field_name.endswith("_placeholder"):
            violations.append({"field_name": row.field_name, "reason": "placeholder_suffix_required"})
        if row.category == "random" and not (row.field_name.endswith("_random") or row.field_name.endswith("_digest_random")):
            violations.append({"field_name": row.field_name, "reason": "random_suffix_required"})
        if row.category == "intermediate" and not row.field_name.endswith("_intermediate"):
            violations.append({"field_name": row.field_name, "reason": "intermediate_suffix_required"})
        if row.category == "temporary" and not row.field_name.endswith("_temporary"):
            violations.append({"field_name": row.field_name, "reason": "temporary_suffix_required"})
        if row.category == "cache" and not row.field_name.endswith("_cache"):
            violations.append({"field_name": row.field_name, "reason": "cache_suffix_required"})
        if row.category == "placeholder" and row.allowed_in_claims.lower() == "true":
            violations.append({"field_name": row.field_name, "reason": "placeholder_claim_support_forbidden"})
        if row.category in {"intermediate", "temporary", "cache"} and row.allowed_in_claims.lower() == "true":
            violations.append({"field_name": row.field_name, "reason": "non_final_state_claim_support_forbidden"})
    return violations
