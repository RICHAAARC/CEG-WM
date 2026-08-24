"""Frozen Phase-1 data contract for Content V6 detector-domain ISS."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from cegwm.protocol.content_chain_v2 import (
    _CONTENT_ANALYSIS,
    _freeze,
    ContentChainProtocol,
    ContentChainUnit,
)
from cegwm.protocol.content_chain_v3 import (
    _AGGREGATE_MEASUREMENT as _V3_AGGREGATE_MEASUREMENT,
)
from cegwm.protocol.content_chain_v4 import (
    _BUDGET as _V4_BUDGET,
    _DECISION_RULE as _V4_DECISION_RULE,
    _DETECTION_ACCESS as _V4_DETECTION_ACCESS,
    _LF_DETECTION_OPERATOR as _V4_LF_DETECTION_OPERATOR,
    _LIMITATIONS as _V4_LIMITATIONS,
    _METHOD_IDENTITIES as _V4_METHOD_IDENTITIES,
)

V6_PERSONAL_SPEC_ID = "CEGWM_V6_PERSONAL_SPEC_V1"
V6_PERSONAL_SPEC_SHA256 = (
    "770a0d79cfdb9d98156f6b8d585ae0c0554313f5dfd745ceb5e228d7f3fc02ce"
)
V6_DEVELOPMENT_SPLIT = "content_v6_iss_development_v1"
V6_EVALUATION_SPLIT = "content_v6_iss_clean_v1"
V6_DEVELOPMENT_MANIFEST = "content_v6_iss_development_v1.jsonl"
V6_EVALUATION_MANIFEST = "content_v6_iss_clean.jsonl"
V6_DEVELOPMENT_MANIFEST_SHA256 = (
    "4ff3efa6b98efb62d542b210ebf00f3fc624513342475ce417e9099e334066ea"
)
V6_EVALUATION_MANIFEST_SHA256 = (
    "20058788bfe7d75878e7263efda2b8de94c6fdcd3a963f64368f2ba4d594868f"
)
V6_DEVELOPMENT_PROMPT_LIST_SHA256 = (
    "fd2120c0ed9be832687a30de85d38dac5fb2abb23b7bd372c7d327d004cbc9ba"
)
V6_EVALUATION_PROMPT_LIST_SHA256 = (
    "ec1b29c673fa109c6078b3dc070d3dd42aa93f834aaaf387d282aa475bd2b219"
)
CONTENT_V6_METHOD_ID = "content_v6_detector_domain_iss_lf_adaptive_hf_v1"
CONTENT_V6_EVALUATED_CANDIDATE_ID = (
    "content_v6_detector_domain_iss_lf_adaptive_hf_semantic_gate_v1"
)
CONTENT_V6_PROTOCOL_ID = "cegwm-stage-a-content-v6-detector-domain-iss-clean-v1"
CONTENT_V6_EXECUTION_SCOPE_ID = (
    "content_v6_detector_domain_iss_engineering_and_stage_a_evaluation_v1"
)
CONTENT_V6_RECORD_CONTRACT_ID = "content_v6_detector_domain_iss_record_v1"
CONTENT_V6_STATE_SCHEMA_ID = "content_v6_resumable_state_v1"
CONTENT_V6_RUN_PREFIX = "content-v6"
CONTENT_V6_PROTOCOL_DIGEST = (
    "855fb511afa23548c30a5fcad17525589b340aac7067ae3491941fc8fc99427d"
)
CONTENT_V6_ARMS = (
    CONTENT_V6_EVALUATED_CANDIDATE_ID,
    f"primary_null__{CONTENT_V6_EVALUATED_CANDIDATE_ID}",
)
V6_ISS_ASSET_SHA256 = "d66ff88640a3d1a020646cfde3face7502282bf835c9d3fb746b518dfb02c231"
V6_ISS_ASSET_SIDECAR_SHA256 = (
    "27094d56994bc6f5d93564bad79ddd9ce8218d2d193786f4816535ee1e7f6538"
)

_GENERATION_RUNTIME = {
    "model_id": "stabilityai/stable-diffusion-3.5-medium",
    "inference_steps": 20,
    "injection_step_index_zero_based": 18,
    "generation_rule": "callback_free_pass1_primary_null_then_same_seed_pass2_iss_joint",
}
_METHOD_IDENTITIES = {
    **_V4_METHOD_IDENTITIES,
    "content_method_id": CONTENT_V6_METHOD_ID,
    "lf_embedding_transform_id": (
        "content_v6_detector_domain_iss_lf_preprojection_multiplier_v1"
    ),
    "branch_amplitude_rule": (
        "V4_LF_preprojection_delta_times_beta_ISS_HF_preprojection_delta_unchanged_"
        "then_common_actual_dtype_relative_l2_projector"
    ),
    "evaluated_candidate_id": CONTENT_V6_EVALUATED_CANDIDATE_ID,
}
_ISS_CONTROLLER = {
    "controller_id": "content_v6_detector_domain_iss_lf_preprojection_multiplier_v1",
    "asset_role_id": "content_v6_iss_gain_target_v1",
    "asset_schema_id": "cegwm_content_v6_iss_gain_target_asset_v1",
    "asset_repo_path": "configs/content_chain/assets/content_v6_iss_gain_target_v1.json",
    "asset_sha256": V6_ISS_ASSET_SHA256,
    "asset_sidecar_sha256": V6_ISS_ASSET_SIDECAR_SHA256,
    "host_observation": "sole_callback_free_pass1_primary_null_ordinary_RGB",
    "host_scorer": "content_v4_whitened_lf_dct_matched_cosine_v1",
    "formula": "beta_equals_clamp_total_multiplier_of_(m-h)/g_inclusive_1_to_2",
    "application": "LF_preprojection_delta_only",
    "hf_preprojection_delta": "unchanged_from_V4",
    "joint_projector": "unchanged_V4_common_actual_dtype_relative_l2_projector",
    "pass1_reuse": "sole_primary_null_record_no_third_generation",
    "blind_detector_consumes_host_observation_or_beta": False,
}
_EXECUTION_FLOW = {
    "roster_manifest": V6_EVALUATION_MANIFEST,
    "formal_roster_sha256": V6_EVALUATION_MANIFEST_SHA256,
    "split": V6_EVALUATION_SPLIT,
    "fixed_units": 8,
    "record_arms_in_order": list(CONTENT_V6_ARMS),
    "unit_transaction_record_count": 2,
    "fixed_records": 16,
    "record_score_prefixes_in_order": ["lf", "hf", "joint"],
    "score_labels_per_prefix": "registered_then_wrong_00_through_wrong_15",
    "flat_score_field_rule": (
        "prefix_double_underscore_label_within_content_v6_detector_domain_iss_record_v1"
    ),
    "record_contract_id": CONTENT_V6_RECORD_CONTRACT_ID,
    "record_fields_in_order": [
        "run_id", "unit_id", "source_cluster_id", "arm", "condition",
        "code_revision", "config_digest", "key_public_digest", "status",
        "failure_reason", "scores", "metrics", "record_contract_id",
    ],
    "failure_units_remain_in_denominator": True,
    "replacement_units_allowed": False,
    "retry_units_allowed": False,
    "outcome_requires_complete_rc0": True,
}
_KEYING = {
    "task": "zero_bit_keyed_attribution",
    "normalization": "NFC_UTF8_for_text_exact_bytes_for_binary",
    "prg": "HMAC_SHA256_counter_v1",
    "wrong_key_count": 16,
    "wrong_key_derivation_domain": "stage-a/content-adaptive-v2-external-wrong-key/v1",
    "primary_null": True,
    "payload_bits": 0,
}

_FIELDS = ("unit_id", "split", "source_id", "prompt", "seed", "height", "width")
_V4_FORMAL_MANIFEST = "content_adaptive_dual_branch_v2_clean.jsonl"
_V4_FIT_MANIFEST = "content_v4_clean_null_whitening_fit_v1.json"


@dataclass(frozen=True, slots=True)
class ContentV6Unit:
    unit_id: str
    split: str
    source_id: str
    prompt: str
    seed: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class ContentV6DataContract:
    development: tuple[ContentV6Unit, ...]
    evaluation: tuple[ContentV6Unit, ...]
    development_manifest_sha256: str
    evaluation_manifest_sha256: str


def _stable_line(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _parse_unit(value: Any, *, split: str, ordinal: int, count: int) -> ContentV6Unit:
    if not isinstance(value, dict) or tuple(value) != _FIELDS:
        raise ValueError("Content V6 manifest fields or order differ")
    expected_prefix = "content-v6-iss-dev" if split == V6_DEVELOPMENT_SPLIT else "content-v6-iss-eval"
    expected_source = f"{expected_prefix}-source-{ordinal:04d}"
    expected_unit = f"{expected_prefix}-{ordinal:04d}"
    expected_seed = (2026082400 if count == 32 else 2026082500) + ordinal - 1
    for field in ("unit_id", "split", "source_id", "prompt"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"Content V6 {field} must be non-empty text")
    if (
        value["unit_id"] != expected_unit
        or value["split"] != split
        or value["source_id"] != expected_source
        or isinstance(value["seed"], bool)
        or value["seed"] != expected_seed
        or value["height"] != 512
        or value["width"] != 512
    ):
        raise ValueError("Content V6 ordered unit identity differs")
    return ContentV6Unit(**value)


def _load_jsonl(
    path: Path,
    *,
    split: str,
    count: int,
    manifest_sha256: str,
    prompt_sha256: str,
) -> tuple[ContentV6Unit, ...]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256 or not raw.endswith(b"\n"):
        raise ValueError("Content V6 manifest bytes differ")
    lines = raw.splitlines()
    if len(lines) != count:
        raise ValueError("Content V6 manifest unit count differs")
    units: list[ContentV6Unit] = []
    for ordinal, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Content V6 manifest must be UTF-8 JSONL") from error
        if not isinstance(value, dict) or _stable_line(value) != line:
            raise ValueError("Content V6 manifest line must use stable JSON")
        units.append(_parse_unit(value, split=split, ordinal=ordinal, count=count))
    prompt_bytes = b"".join(unit.prompt.encode("utf-8") + b"\n" for unit in units)
    if hashlib.sha256(prompt_bytes).hexdigest() != prompt_sha256:
        raise ValueError("Content V6 ordered prompt identity differs")
    return tuple(units)


def _identity_sets(units: Iterable[ContentV6Unit]) -> tuple[set[Any], ...]:
    received = tuple(units)
    return (
        {unit.unit_id for unit in received},
        {unit.source_id for unit in received},
        {unit.prompt for unit in received},
        {unit.seed for unit in received},
        {(unit.prompt, unit.seed) for unit in received},
    )


def _historical_units(root: Path) -> tuple[ContentV6Unit, ...]:
    historical: list[ContentV6Unit] = []
    formal_path = root / _V4_FORMAL_MANIFEST
    for line in formal_path.read_bytes().splitlines():
        value = json.loads(line)
        historical.append(ContentV6Unit(
            value["unit_id"], value["split"], value["source_id"], value["prompt"],
            value["seed"], value["height"], value["width"],
        ))
    fit_value = json.loads((root / _V4_FIT_MANIFEST).read_bytes())
    for value in fit_value["entries"]:
        historical.append(ContentV6Unit(
            value["unit_id"], "content_v4_fit", value["unit_id"], value["prompt"],
            value["generation_seed"], 512, 512,
        ))
    return tuple(historical)


def _require_unique_and_disjoint(
    development: tuple[ContentV6Unit, ...],
    evaluation: tuple[ContentV6Unit, ...],
    historical: tuple[ContentV6Unit, ...],
) -> None:
    dev_sets = _identity_sets(development)
    eval_sets = _identity_sets(evaluation)
    historical_sets = _identity_sets(historical)
    for values, expected in ((dev_sets, 32), (eval_sets, 8)):
        if any(len(field) != expected for field in values):
            raise ValueError("Content V6 manifest identities must be unique")
    for dev, evaluation_field, old in zip(dev_sets, eval_sets, historical_sets, strict=True):
        if dev & evaluation_field or dev & old or evaluation_field & old:
            raise ValueError("Content V6 development/evaluation data identities overlap")


def load_content_v6_data_contract(repo_root: str | Path) -> ContentV6DataContract:
    """Load the fixed dev/eval manifests without constructing a final run identity."""

    root = Path(repo_root) / "configs" / "content_chain"
    development = _load_jsonl(
        root / V6_DEVELOPMENT_MANIFEST,
        split=V6_DEVELOPMENT_SPLIT,
        count=32,
        manifest_sha256=V6_DEVELOPMENT_MANIFEST_SHA256,
        prompt_sha256=V6_DEVELOPMENT_PROMPT_LIST_SHA256,
    )
    evaluation = _load_jsonl(
        root / V6_EVALUATION_MANIFEST,
        split=V6_EVALUATION_SPLIT,
        count=8,
        manifest_sha256=V6_EVALUATION_MANIFEST_SHA256,
        prompt_sha256=V6_EVALUATION_PROMPT_LIST_SHA256,
    )
    _require_unique_and_disjoint(development, evaluation, _historical_units(root))
    return ContentV6DataContract(
        development,
        evaluation,
        V6_DEVELOPMENT_MANIFEST_SHA256,
        V6_EVALUATION_MANIFEST_SHA256,
    )


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _validate_final_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != 1 or config.get("protocol_id") != CONTENT_V6_PROTOCOL_ID:
        raise ValueError("unexpected Content V6 protocol identity")
    if config.get("execution_scope_id") != CONTENT_V6_EXECUTION_SCOPE_ID:
        raise ValueError("unexpected Content V6 execution scope")
    if config.get("scientific_status") != "not_evaluated_until_complete_real_gpu_rc0":
        raise ValueError("Content V6 protocol cannot preclaim scientific evidence")
    expected_sections = {
        "generation_runtime": _GENERATION_RUNTIME,
        "content_analysis": _CONTENT_ANALYSIS,
        "method_identities": _METHOD_IDENTITIES,
        "lf_detection_operator": _V4_LF_DETECTION_OPERATOR,
        "iss_controller": _ISS_CONTROLLER,
        "budget": _V4_BUDGET,
        "aggregate_measurement": _V3_AGGREGATE_MEASUREMENT,
        "detection_access": _V4_DETECTION_ACCESS,
        "keying": _KEYING,
        "execution_flow": _EXECUTION_FLOW,
        "decision_rule": _V4_DECISION_RULE,
    }
    for name, expected in expected_sections.items():
        if _mapping(config, name) != expected:
            raise ValueError(f"Content V6 {name.replace('_', ' ')} differs")
    if config.get("limitations") != _V4_LIMITATIONS:
        raise ValueError("Content V6 limitations differ")
    if set(config) != {
        "protocol_version", "protocol_id", "execution_scope_id", "scientific_status",
        *expected_sections, "limitations",
    }:
        raise ValueError("Content V6 config fields differ")


def load_content_v6_clean_protocol(repo_root: str | Path) -> ContentChainProtocol:
    """Load the final fixed-eight V6 route and its accepted public ISS asset."""

    root = Path(repo_root)
    config_path = root / "configs" / "content_chain" / "content_v6_iss_clean_v1.json"
    try:
        config = json.loads(config_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Content V6 config must be UTF-8 JSON") from error
    if not isinstance(config, dict):
        raise ValueError("Content V6 config must be an object")
    _validate_final_config(config)
    contract = load_content_v6_data_contract(root)
    from cegwm.method.content_iss_v6 import load_frozen_content_v6_iss_asset

    load_frozen_content_v6_iss_asset(root)
    roster = tuple(ContentChainUnit(**asdict(unit)) for unit in contract.evaluation)
    canonical = json.dumps(
        {"config": config, "roster": [asdict(unit) for unit in roster]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    protocol_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if protocol_digest != CONTENT_V6_PROTOCOL_DIGEST:
        raise ValueError("Content V6 canonical protocol digest differs")
    return ContentChainProtocol(
        protocol_id=config["protocol_id"],
        config=_freeze(config),
        roster=roster,
        protocol_digest=protocol_digest,
    )


__all__ = [
    "CONTENT_V6_ARMS",
    "CONTENT_V6_EVALUATED_CANDIDATE_ID",
    "CONTENT_V6_EXECUTION_SCOPE_ID",
    "CONTENT_V6_METHOD_ID",
    "CONTENT_V6_PROTOCOL_DIGEST",
    "CONTENT_V6_PROTOCOL_ID",
    "CONTENT_V6_RECORD_CONTRACT_ID",
    "CONTENT_V6_RUN_PREFIX",
    "CONTENT_V6_STATE_SCHEMA_ID",
    "ContentV6DataContract",
    "ContentV6Unit",
    "V6_DEVELOPMENT_MANIFEST",
    "V6_DEVELOPMENT_MANIFEST_SHA256",
    "V6_DEVELOPMENT_PROMPT_LIST_SHA256",
    "V6_EVALUATION_MANIFEST",
    "V6_EVALUATION_MANIFEST_SHA256",
    "V6_EVALUATION_PROMPT_LIST_SHA256",
    "V6_PERSONAL_SPEC_ID",
    "V6_PERSONAL_SPEC_SHA256",
    "load_content_v6_clean_protocol",
    "load_content_v6_data_contract",
]
