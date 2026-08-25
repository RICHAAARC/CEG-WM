"""Reference-only Content V6 protocol over the frozen Content V2 roster."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from cegwm.protocol.content_chain_v2 import ContentChainProtocol, ContentChainUnit, _freeze
from cegwm.protocol.content_chain_v6 import (
    CONTENT_V6_PROTOCOL_DIGEST,
    CONTENT_V6_PROTOCOL_ID,
    load_content_v6_clean_protocol,
)

CONTENT_V6_REFERENCE_OLDROSTER_PROTOCOL_ID = (
    "cegwm-stage-a-content-v6-detector-domain-iss-reference-oldroster-v1"
)
CONTENT_V6_REFERENCE_OLDROSTER_EXECUTION_SCOPE_ID = (
    "content_v6_detector_domain_iss_reference_oldroster_"
    "engineering_and_stage_a_evaluation_v1"
)
CONTENT_V6_REFERENCE_OLDROSTER_METHOD_ID = (
    "content_v6_detector_domain_iss_lf_adaptive_hf_reference_oldroster_v1"
)
CONTENT_V6_REFERENCE_OLDROSTER_CANDIDATE_ID = (
    "content_v6_detector_domain_iss_lf_adaptive_hf_"
    "semantic_gate_reference_oldroster_v1"
)
CONTENT_V6_REFERENCE_OLDROSTER_RECORD_CONTRACT_ID = (
    "content_v6_detector_domain_iss_reference_oldroster_record_v1"
)
CONTENT_V6_REFERENCE_OLDROSTER_STATE_SCHEMA_ID = (
    "content_v6_reference_oldroster_resumable_state_v1"
)
CONTENT_V6_REFERENCE_OLDROSTER_RUN_PREFIX = "content-v6-reference-oldroster"
CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_MANIFEST = (
    "content_adaptive_dual_branch_v2_clean.jsonl"
)
CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_SHA256 = (
    "dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88"
)
CONTENT_V6_REFERENCE_OLDROSTER_SPLIT = "content_adaptive_dual_branch_v2_clean_v1"
CONTENT_V6_REFERENCE_OLDROSTER_PROTOCOL_DIGEST = (
    "c98175252406cce147b329b016fe3f6acb62b2ed1ba5bba66ca9fea5ae37fa80"
)
CONTENT_V6_REFERENCE_OLDROSTER_ARMS = (
    CONTENT_V6_REFERENCE_OLDROSTER_CANDIDATE_ID,
    f"primary_null__{CONTENT_V6_REFERENCE_OLDROSTER_CANDIDATE_ID}",
)

_CONFIG_NAME = "content_v6_iss_reference_oldroster_v1.json"
_FIELDS = ("unit_id", "split", "source_id", "prompt", "seed", "height", "width")
_EXPECTED_DEFINITION = {
    "schema_version": 1,
    "protocol_id": CONTENT_V6_REFERENCE_OLDROSTER_PROTOCOL_ID,
    "execution_scope_id": CONTENT_V6_REFERENCE_OLDROSTER_EXECUTION_SCOPE_ID,
    "content_method_id": CONTENT_V6_REFERENCE_OLDROSTER_METHOD_ID,
    "evaluated_candidate_id": CONTENT_V6_REFERENCE_OLDROSTER_CANDIDATE_ID,
    "record_contract_id": CONTENT_V6_REFERENCE_OLDROSTER_RECORD_CONTRACT_ID,
    "state_schema_id": CONTENT_V6_REFERENCE_OLDROSTER_STATE_SCHEMA_ID,
    "run_prefix": CONTENT_V6_REFERENCE_OLDROSTER_RUN_PREFIX,
    "base_v6_protocol": {
        "protocol_id": CONTENT_V6_PROTOCOL_ID,
        "protocol_digest": CONTENT_V6_PROTOCOL_DIGEST,
        "config_repo_path": "configs/content_chain/content_v6_iss_clean_v1.json",
    },
    "cohort": {
        "roster_manifest": CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_MANIFEST,
        "roster_sha256": CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_SHA256,
        "split": CONTENT_V6_REFERENCE_OLDROSTER_SPLIT,
        "fixed_units": 8,
        "fixed_records": 16,
    },
    "execution_separation": {
        "fresh_generation_required": True,
        "independent_artifact_required": True,
        "pooling_allowed": False,
        "current_v6_evaluation_roster_allowed": False,
        "imported_result_or_artifact_allowed": False,
    },
}


def _stable_line(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _load_definition(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Content V6 reference definition must be UTF-8 JSON") from error
    if not isinstance(value, dict) or value != _EXPECTED_DEFINITION:
        raise ValueError("Content V6 reference definition differs")
    return value


def _load_reference_roster(path: Path) -> tuple[ContentChainUnit, ...]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_SHA256:
        raise ValueError("Content V6 reference roster bytes differ")
    if not raw.endswith(b"\n"):
        raise ValueError("Content V6 reference roster must end with a newline")
    units: list[ContentChainUnit] = []
    for line in raw.splitlines():
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Content V6 reference roster must be UTF-8 JSONL") from error
        if not isinstance(value, dict) or tuple(value) != _FIELDS or _stable_line(value) != line:
            raise ValueError("Content V6 reference roster line differs")
        if value["split"] != CONTENT_V6_REFERENCE_OLDROSTER_SPLIT:
            raise ValueError("Content V6 reference roster split differs")
        units.append(ContentChainUnit(**value))
    if len(units) != 8:
        raise ValueError("Content V6 reference roster unit count differs")
    identity_fields = (
        {unit.unit_id for unit in units},
        {unit.source_id for unit in units},
        {unit.prompt for unit in units},
        {unit.seed for unit in units},
        {(unit.prompt, unit.seed) for unit in units},
    )
    if any(len(values) != 8 for values in identity_fields):
        raise ValueError("Content V6 reference roster identities must be unique")
    return tuple(units)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _reference_config(base: ContentChainProtocol) -> dict[str, Any]:
    config = _thaw(base.config)
    config["protocol_id"] = CONTENT_V6_REFERENCE_OLDROSTER_PROTOCOL_ID
    config["execution_scope_id"] = CONTENT_V6_REFERENCE_OLDROSTER_EXECUTION_SCOPE_ID
    identities = config["method_identities"]
    identities["content_method_id"] = CONTENT_V6_REFERENCE_OLDROSTER_METHOD_ID
    identities["evaluated_candidate_id"] = CONTENT_V6_REFERENCE_OLDROSTER_CANDIDATE_ID
    flow = config["execution_flow"]
    flow.update({
        "roster_manifest": CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_MANIFEST,
        "formal_roster_sha256": CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_SHA256,
        "split": CONTENT_V6_REFERENCE_OLDROSTER_SPLIT,
        "record_arms_in_order": list(CONTENT_V6_REFERENCE_OLDROSTER_ARMS),
        "flat_score_field_rule": (
            "prefix_double_underscore_label_within_"
            f"{CONTENT_V6_REFERENCE_OLDROSTER_RECORD_CONTRACT_ID}"
        ),
        "record_contract_id": CONTENT_V6_REFERENCE_OLDROSTER_RECORD_CONTRACT_ID,
    })
    return config


def load_content_v6_reference_oldroster_protocol(
    repo_root: str | Path,
) -> ContentChainProtocol:
    """Load the independent reference-cohort protocol through the frozen V6 base."""

    root = Path(repo_root)
    definition = _load_definition(
        root / "configs" / "content_chain" / _CONFIG_NAME
    )
    base = load_content_v6_clean_protocol(root)
    if (
        base.protocol_id != CONTENT_V6_PROTOCOL_ID
        or base.protocol_digest != CONTENT_V6_PROTOCOL_DIGEST
    ):
        raise ValueError("Content V6 reference base protocol differs")
    roster = _load_reference_roster(
        root / "configs" / "content_chain" / CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_MANIFEST
    )
    config = _reference_config(base)
    canonical = json.dumps(
        {
            "definition": definition,
            "config": config,
            "roster": [asdict(unit) for unit in roster],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != CONTENT_V6_REFERENCE_OLDROSTER_PROTOCOL_DIGEST:
        raise ValueError("Content V6 reference canonical protocol digest differs")
    return ContentChainProtocol(
        protocol_id=CONTENT_V6_REFERENCE_OLDROSTER_PROTOCOL_ID,
        config=_freeze(config),
        roster=roster,
        protocol_digest=digest,
    )


__all__ = [
    "CONTENT_V6_REFERENCE_OLDROSTER_ARMS",
    "CONTENT_V6_REFERENCE_OLDROSTER_CANDIDATE_ID",
    "CONTENT_V6_REFERENCE_OLDROSTER_EXECUTION_SCOPE_ID",
    "CONTENT_V6_REFERENCE_OLDROSTER_METHOD_ID",
    "CONTENT_V6_REFERENCE_OLDROSTER_PROTOCOL_DIGEST",
    "CONTENT_V6_REFERENCE_OLDROSTER_PROTOCOL_ID",
    "CONTENT_V6_REFERENCE_OLDROSTER_RECORD_CONTRACT_ID",
    "CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_MANIFEST",
    "CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_SHA256",
    "CONTENT_V6_REFERENCE_OLDROSTER_RUN_PREFIX",
    "CONTENT_V6_REFERENCE_OLDROSTER_SPLIT",
    "CONTENT_V6_REFERENCE_OLDROSTER_STATE_SCHEMA_ID",
    "load_content_v6_reference_oldroster_protocol",
]
