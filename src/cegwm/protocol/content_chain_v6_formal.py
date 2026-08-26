"""One-shot Content V6 ISS fit and two-roster formal contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from cegwm.protocol.content_chain_v2 import ContentChainProtocol, ContentChainUnit, _freeze
from cegwm.protocol.content_chain_v6 import (
    CONTENT_V6_ARMS,
    CONTENT_V6_RECORD_CONTRACT_ID,
    V6_DEVELOPMENT_MANIFEST,
    V6_DEVELOPMENT_MANIFEST_SHA256,
    V6_EVALUATION_MANIFEST,
    V6_EVALUATION_MANIFEST_SHA256,
    load_content_v6_clean_protocol,
    load_content_v6_data_contract,
)
from cegwm.protocol.content_chain_v6_reference_oldroster import (
    CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_MANIFEST,
    CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_SHA256,
    load_content_v6_reference_oldroster_protocol,
)

CONTENT_V6_FORMAL_PROTOCOL_ID = (
    "cegwm-stage-a-content-v6-detector-domain-iss-formal-initial-v1"
)
CONTENT_V6_FORMAL_EXECUTION_SCOPE_ID = (
    "content_v6_iss_fit_then_old_and_current_independent_evaluations_v1"
)
CONTENT_V6_FORMAL_RUN_PREFIX = "content-v6-formal-initial"
CONTENT_V6_FORMAL_ARMS = CONTENT_V6_ARMS
CONTENT_V6_FORMAL_RECORD_CONTRACT_ID = CONTENT_V6_RECORD_CONTRACT_ID

UNIT_SET_32V1 = {
    "unit_set_id": "content_units_32_v1",
    "display_label": "[32V1]",
    "manifest": V6_DEVELOPMENT_MANIFEST,
    "manifest_sha256": V6_DEVELOPMENT_MANIFEST_SHA256,
    "fixed_units": 32,
    "role": "development",
}
UNIT_SET_8V1 = {
    "unit_set_id": "content_units_8_v1",
    "display_label": "[8V1]",
    "manifest": CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_MANIFEST,
    "manifest_sha256": CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_SHA256,
    "fixed_units": 8,
    "role": "evaluation",
}
UNIT_SET_8V3 = {
    "unit_set_id": "content_units_8_v3",
    "display_label": "[8V3]",
    "manifest": V6_EVALUATION_MANIFEST,
    "manifest_sha256": V6_EVALUATION_MANIFEST_SHA256,
    "fixed_units": 8,
    "role": "evaluation",
}


@dataclass(frozen=True, slots=True)
class ContentV6FormalProtocol:
    protocol_id: str
    config: Mapping[str, Any]
    development: tuple[Any, ...]
    evaluations: tuple[ContentChainProtocol, ContentChainProtocol]
    protocol_digest: str


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _identity_sets(units: Iterable[Any]) -> tuple[set[Any], ...]:
    received = tuple(units)
    return (
        {unit.unit_id for unit in received},
        {unit.source_id for unit in received},
        {unit.prompt for unit in received},
        {unit.seed for unit in received},
        {(unit.prompt, unit.seed) for unit in received},
    )


def _require_unique_disjoint(*groups: tuple[Any, ...]) -> None:
    identities = tuple(_identity_sets(group) for group in groups)
    for group, fields in zip(groups, identities, strict=True):
        if any(len(field) != len(group) for field in fields):
            raise ValueError("Content V6 formal unit identities must be unique")
    for left_index, left in enumerate(identities):
        for right in identities[left_index + 1 :]:
            if any(a & b for a, b in zip(left, right, strict=True)):
                raise ValueError("Content V6 formal unit sets must be disjoint")


def _formal_config(base: ContentChainProtocol) -> dict[str, Any]:
    config = _thaw(base.config)
    config["protocol_id"] = CONTENT_V6_FORMAL_PROTOCOL_ID
    config["execution_scope_id"] = CONTENT_V6_FORMAL_EXECUTION_SCOPE_ID
    config["scientific_status"] = (
        "not_evaluated_until_complete_real_gpu_two_roster_result"
    )
    controller = config["iss_controller"]
    for field in ("asset_repo_path", "asset_sha256", "asset_sidecar_sha256"):
        controller.pop(field, None)
    controller["runtime_asset_binding"] = (
        "fit_32V1_then_atomic_create_only_pair_before_both_evaluations"
    )
    config["unit_sets"] = [UNIT_SET_32V1, UNIT_SET_8V1, UNIT_SET_8V3]
    config["execution_flow"] = {
        "phase_order": ["development_32V1", "evaluation_8V1", "evaluation_8V3", "terminal"],
        "development": UNIT_SET_32V1,
        "evaluations": [UNIT_SET_8V1, UNIT_SET_8V3],
        "records_per_evaluation_unit": 2,
        "independent_failures_denominators_and_gates": True,
        "pooling_allowed": False,
        "cross_cohort_conjunction_allowed": False,
        "combined_result_allowed": False,
        "retry_or_resume_allowed": False,
    }
    config["limitations"] = list(config["limitations"]) + [
        "no_combined_result_across_8V1_and_8V3",
    ]
    return config


def load_content_v6_formal_protocol(repo_root: str | Path) -> ContentV6FormalProtocol:
    root = Path(repo_root)
    base = load_content_v6_clean_protocol(root)
    old = load_content_v6_reference_oldroster_protocol(root)
    data = load_content_v6_data_contract(root)
    current = tuple(ContentChainUnit(**asdict(unit)) for unit in data.evaluation)
    development = tuple(data.development)
    _require_unique_disjoint(development, old.roster, current)
    config = _formal_config(base)
    canonical = json.dumps(
        {
            "config": config,
            "development": [asdict(unit) for unit in development],
            "evaluations": [
                [asdict(unit) for unit in old.roster],
                [asdict(unit) for unit in current],
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    rosters = (old.roster, current)
    labels = ("evaluation_01_8V1", "evaluation_02_8V3")
    evaluations = tuple(
        ContentChainProtocol(
            protocol_id=f"{CONTENT_V6_FORMAL_PROTOCOL_ID}/{label}",
            config=_freeze(config),
            roster=roster,
            protocol_digest=hashlib.sha256(
                digest.encode("ascii") + b"\0" + label.encode("ascii")
            ).hexdigest(),
        )
        for label, roster in zip(labels, rosters, strict=True)
    )
    return ContentV6FormalProtocol(
        CONTENT_V6_FORMAL_PROTOCOL_ID,
        _freeze(config),
        development,
        (evaluations[0], evaluations[1]),
        digest,
    )


__all__ = [
    "CONTENT_V6_FORMAL_ARMS",
    "CONTENT_V6_FORMAL_EXECUTION_SCOPE_ID",
    "CONTENT_V6_FORMAL_PROTOCOL_ID",
    "CONTENT_V6_FORMAL_RECORD_CONTRACT_ID",
    "CONTENT_V6_FORMAL_RUN_PREFIX",
    "ContentV6FormalProtocol",
    "UNIT_SET_8V1",
    "UNIT_SET_8V3",
    "UNIT_SET_32V1",
    "load_content_v6_formal_protocol",
]
