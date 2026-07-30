"""冻结内部科学验证的分析单位、source cluster 与 split 访问边界。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Iterable


_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")

LEGACY_INTERNAL_VALIDATION_PROTOCOL_ID = "ceg_wm_internal_scientific_validation_v1"
LEGACY_INTERNAL_VALIDATION_PROTOCOL_VERSION = "1.0.0"
INTERNAL_VALIDATION_PROTOCOL_ID = "ceg_wm_internal_scientific_validation_v2"
INTERNAL_VALIDATION_PROTOCOL_VERSION = "2.0.0"
CURRENT_EXECUTION_ACCESS_IDENTITY = "internal_scientific_validation_current_execution_v2"

INTERNAL_VALIDATION_SPLITS = (
    "development",
    "candidate_selection",
    "untouched_confirmation",
    "content_threshold_fit",
    "rescue_threshold_fit",
    "reliability_fit",
    "end_to_end_check",
    "held_out_evaluation",
)

CURRENT_EXECUTION_ALLOWED_SPLITS = frozenset(INTERNAL_VALIDATION_SPLITS[:-1])


def _canonical_digest(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def derive_source_cluster_id(
    *,
    prompt_digest: str,
    generation_seed: int,
    image_lineage_digest: str,
    registered_key_family_digest: str,
) -> str:
    """从禁止跨 split 的四项 lineage 身份导出 source-cluster identity。"""
    for name, value in (
        ("prompt_digest", prompt_digest),
        ("image_lineage_digest", image_lineage_digest),
        ("registered_key_family_digest", registered_key_family_digest),
    ):
        if not _DIGEST_PATTERN.fullmatch(value):
            raise ValueError(f"{name}_invalid")
    if not isinstance(generation_seed, int) or isinstance(generation_seed, bool):
        raise ValueError("generation_seed_invalid")
    return _canonical_digest(
        {
            "generation_seed": generation_seed,
            "image_lineage_digest": image_lineage_digest,
            "prompt_digest": prompt_digest,
            "registered_key_family_digest": registered_key_family_digest,
        }
    )


@dataclass(frozen=True)
class AnalysisUnitIdentity:
    """一个 case 中可记录的 unit，显式绑定其不可拆分 source cluster。"""

    unit_id: str
    case_id: str
    source_cluster_id: str
    prompt_digest: str
    generation_seed: int
    image_lineage_digest: str
    registered_key_family_digest: str

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        for name in ("unit_id", "case_id"):
            if not getattr(self, name).strip():
                violations.append(f"{name}_missing")
        try:
            expected_cluster_id = derive_source_cluster_id(
                prompt_digest=self.prompt_digest,
                generation_seed=self.generation_seed,
                image_lineage_digest=self.image_lineage_digest,
                registered_key_family_digest=self.registered_key_family_digest,
            )
        except ValueError as error:
            violations.append(str(error))
        else:
            if self.source_cluster_id != expected_cluster_id:
                violations.append("source_cluster_id_identity_mismatch")
        return tuple(violations)


@dataclass(frozen=True)
class SplitAssignment:
    """把一个 unit/case/source-cluster identity 显式分配到一个职责 split。"""

    identity: AnalysisUnitIdentity
    split: str


@dataclass(frozen=True)
class FrozenSplitManifest:
    """显式、可摘要且禁止 source cluster 泄漏的内部 split manifest。"""

    protocol_id: str
    protocol_version: str
    manifest_id: str
    manifest_revision: str
    assignments: tuple[SplitAssignment, ...]

    def digest(self) -> str:
        return _canonical_digest(asdict(self))

    def validate(self, *, require_all_splits: bool = True) -> tuple[str, ...]:
        violations: list[str] = []
        for name in ("protocol_id", "protocol_version", "manifest_id", "manifest_revision"):
            if not getattr(self, name).strip():
                violations.append(f"{name}_missing")
        if self.protocol_id != INTERNAL_VALIDATION_PROTOCOL_ID:
            violations.append("protocol_id_frozen_identity_mismatch")
        if self.protocol_version != INTERNAL_VALIDATION_PROTOCOL_VERSION:
            violations.append("protocol_version_frozen_identity_mismatch")

        seen_units: set[str] = set()
        cluster_splits: dict[str, str] = {}
        observed_splits: set[str] = set()
        for assignment in self.assignments:
            violations.extend(assignment.identity.validate())
            if assignment.split not in INTERNAL_VALIDATION_SPLITS:
                violations.append("split_invalid")
            else:
                observed_splits.add(assignment.split)
            unit_id = assignment.identity.unit_id
            if unit_id in seen_units:
                violations.append("unit_id_duplicate")
            seen_units.add(unit_id)
            cluster_id = assignment.identity.source_cluster_id
            prior_split = cluster_splits.setdefault(cluster_id, assignment.split)
            if prior_split != assignment.split:
                violations.append("source_cluster_split_leakage")

        if require_all_splits:
            missing_splits = set(INTERNAL_VALIDATION_SPLITS) - observed_splits
            violations.extend(f"split_missing:{split_name}" for split_name in sorted(missing_splits))
        return tuple(dict.fromkeys(violations))


@dataclass(frozen=True)
class SplitAccessGrant:
    """一次执行可读取的 split 集合；当前配置不包含最终 held-out。"""

    access_identity: str
    allowed_splits: frozenset[str]

    @classmethod
    def current_execution(cls) -> SplitAccessGrant:
        return cls(
            access_identity=CURRENT_EXECUTION_ACCESS_IDENTITY,
            allowed_splits=CURRENT_EXECUTION_ALLOWED_SPLITS,
        )


def authorize_split_access(
    manifest: FrozenSplitManifest,
    requested_splits: Iterable[str],
    grant: SplitAccessGrant,
) -> tuple[SplitAssignment, ...]:
    """校验 manifest 后返回授权 rows；任何 held-out evaluation 访问默认失败。"""
    violations = manifest.validate()
    if violations:
        raise ValueError(", ".join(violations))
    if (
        grant.access_identity != CURRENT_EXECUTION_ACCESS_IDENTITY
        or grant.allowed_splits != CURRENT_EXECUTION_ALLOWED_SPLITS
    ):
        raise PermissionError("split_access_grant_not_current_authority")
    requested = frozenset(requested_splits)
    unknown = requested - set(INTERNAL_VALIDATION_SPLITS)
    if unknown:
        raise PermissionError(f"unknown_split_access:{','.join(sorted(unknown))}")
    forbidden = requested - grant.allowed_splits
    if forbidden:
        raise PermissionError(f"split_access_forbidden:{','.join(sorted(forbidden))}")
    return tuple(assignment for assignment in manifest.assignments if assignment.split in requested)
