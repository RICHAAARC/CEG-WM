"""Frozen one-shot formal contract for Content V7 ordinary-score ISS."""

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
    _AGGREGATE_MEASUREMENT,
    load_content_v3_clean_protocol,
)

CONTENT_V7_PROTOCOL_ID = "cegwm-stage-a-content-v7-ordinary-iss-formal-initial-v1"
CONTENT_V7_EXECUTION_SCOPE_ID = (
    "content_v7_ordinary_iss_fit_then_two_independent_stage_a_evaluations_v1"
)
CONTENT_V7_METHOD_ID = "content_v7_ordinary_score_iss_lf_adaptive_hf_v1"
CONTENT_V7_EVALUATED_CANDIDATE_ID = (
    "content_v7_ordinary_score_iss_lf_adaptive_hf_semantic_gate_v1"
)
CONTENT_V7_RECORD_CONTRACT_ID = "content_v7_ordinary_iss_record_v1"
CONTENT_V7_ARMS = (
    CONTENT_V7_EVALUATED_CANDIDATE_ID,
    f"primary_null__{CONTENT_V7_EVALUATED_CANDIDATE_ID}",
)
CONTENT_V7_RUN_PREFIX = "content-v7-formal-initial"
V7_FORMAL_CONFIG = "content_v7_ordinary_iss_formal_initial_v1.json"
V7_DEVELOPMENT_MANIFEST = "content_v7_ordinary_iss_development_v1.jsonl"
V7_DEVELOPMENT_SPLIT = "content_v6_iss_development_v1"
V7_DEVELOPMENT_MANIFEST_SHA256 = (
    "4ff3efa6b98efb62d542b210ebf00f3fc624513342475ce417e9099e334066ea"
)
V7_DEVELOPMENT_PROMPT_LIST_SHA256 = (
    "fd2120c0ed9be832687a30de85d38dac5fb2abb23b7bd372c7d327d004cbc9ba"
)
V7_EVALUATION_1_MANIFEST = "content_adaptive_dual_branch_v2_clean.jsonl"
V7_EVALUATION_1_MANIFEST_SHA256 = (
    "dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88"
)
V7_EVALUATION_2_MANIFEST = "content_v6_iss_clean.jsonl"
V7_EVALUATION_2_SPLIT = "content_v6_iss_clean_v1"
V7_EVALUATION_2_MANIFEST_SHA256 = (
    "20058788bfe7d75878e7263efda2b8de94c6fdcd3a963f64368f2ba4d594868f"
)
V7_EVALUATION_2_PROMPT_LIST_SHA256 = (
    "ec1b29c673fa109c6078b3dc070d3dd42aa93f834aaaf387d282aa475bd2b219"
)

_FIELDS = ("unit_id", "split", "source_id", "prompt", "seed", "height", "width")
_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
_WRONG_KEY_DOMAIN = "stage-a/content-adaptive-v2-external-wrong-key/v1"


@dataclass(frozen=True, slots=True)
class ContentV7Unit:
    unit_id: str
    split: str
    source_id: str
    prompt: str
    seed: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class ContentV7DataContract:
    development: tuple[ContentV7Unit, ...]
    evaluations: tuple[tuple[ContentChainUnit, ...], tuple[ContentChainUnit, ...]]
    development_manifest_sha256: str
    evaluation_manifest_sha256s: tuple[str, str]


@dataclass(frozen=True, slots=True)
class ContentV7FormalProtocol:
    protocol_id: str
    config: Mapping[str, Any]
    data: ContentV7DataContract
    evaluations: tuple[ContentChainProtocol, ContentChainProtocol]
    protocol_digest: str


def _stable_line(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _parse_v6_identity(
    value: Any,
    *,
    ordinal: int,
    split: str,
    role: str,
) -> ContentV7Unit:
    if not isinstance(value, dict) or tuple(value) != _FIELDS:
        raise ValueError(f"Content V7 {role} manifest fields or order differ")
    prefix = "content-v6-iss-dev" if role == "development" else "content-v6-iss-eval"
    seed_base = 2026082400 if role == "development" else 2026082500
    if (
        value["unit_id"] != f"{prefix}-{ordinal:04d}"
        or value["split"] != split
        or value["source_id"] != f"{prefix}-source-{ordinal:04d}"
        or not isinstance(value["prompt"], str)
        or not value["prompt"].strip()
        or isinstance(value["seed"], bool)
        or value["seed"] != seed_base + ordinal - 1
        or value["height"] != 512
        or value["width"] != 512
    ):
        raise ValueError(f"Content V7 ordered {role} identity differs")
    return ContentV7Unit(**value)


def _load_exact_jsonl(
    path: Path,
    *,
    count: int,
    split: str,
    role: str,
    manifest_sha256: str,
    prompt_sha256: str | None,
) -> tuple[ContentV7Unit, ...]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256 or not raw.endswith(b"\n"):
        raise ValueError(f"Content V7 {role} manifest bytes differ")
    lines = raw.splitlines()
    if len(lines) != count:
        raise ValueError(f"Content V7 {role} manifest unit count differs")
    units: list[ContentV7Unit] = []
    for ordinal, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Content V7 {role} manifest must be UTF-8 JSONL") from error
        if not isinstance(value, dict) or _stable_line(value) != line:
            raise ValueError(f"Content V7 {role} manifest line must use stable JSON")
        units.append(
            _parse_v6_identity(value, ordinal=ordinal, split=split, role=role)
        )
    if prompt_sha256 is not None:
        prompt_bytes = b"".join(unit.prompt.encode("utf-8") + b"\n" for unit in units)
        if hashlib.sha256(prompt_bytes).hexdigest() != prompt_sha256:
            raise ValueError(f"Content V7 ordered {role} prompt identity differs")
    return tuple(units)


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
    sets = tuple(_identity_sets(group) for group in groups)
    for group, identities in zip(groups, sets, strict=True):
        if any(len(field) != len(group) for field in identities):
            raise ValueError("Content V7 data identities must be unique within each role")
    for left_index, left in enumerate(sets):
        for right in sets[left_index + 1 :]:
            if any(a & b for a, b in zip(left, right, strict=True)):
                raise ValueError("Content V7 development/evaluation data roles overlap")


def load_content_v7_data_contract(repo_root: str | Path) -> ContentV7DataContract:
    root = Path(repo_root)
    config_root = root / "configs" / "content_chain"
    development = _load_exact_jsonl(
        config_root / V7_DEVELOPMENT_MANIFEST,
        count=32,
        split=V7_DEVELOPMENT_SPLIT,
        role="development",
        manifest_sha256=V7_DEVELOPMENT_MANIFEST_SHA256,
        prompt_sha256=V7_DEVELOPMENT_PROMPT_LIST_SHA256,
    )
    v3 = load_content_v3_clean_protocol(
        config_root / "content_v3_clean_v1.json",
        config_root / V7_EVALUATION_1_MANIFEST,
    )
    if hashlib.sha256(
        (config_root / V7_EVALUATION_1_MANIFEST).read_bytes()
    ).hexdigest() != V7_EVALUATION_1_MANIFEST_SHA256:
        raise ValueError("Content V7 first evaluation manifest bytes differ")
    second_units = _load_exact_jsonl(
        config_root / V7_EVALUATION_2_MANIFEST,
        count=8,
        split=V7_EVALUATION_2_SPLIT,
        role="evaluation",
        manifest_sha256=V7_EVALUATION_2_MANIFEST_SHA256,
        prompt_sha256=V7_EVALUATION_2_PROMPT_LIST_SHA256,
    )
    second = tuple(ContentChainUnit(**asdict(unit)) for unit in second_units)
    _require_unique_disjoint(development, v3.roster, second)
    return ContentV7DataContract(
        development,
        (v3.roster, second),
        V7_DEVELOPMENT_MANIFEST_SHA256,
        (V7_EVALUATION_1_MANIFEST_SHA256, V7_EVALUATION_2_MANIFEST_SHA256),
    )


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Content V7 {key} must be an object")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("protocol_version") != 1
        or config.get("protocol_id") != CONTENT_V7_PROTOCOL_ID
        or config.get("execution_scope_id") != CONTENT_V7_EXECUTION_SCOPE_ID
        or config.get("scientific_status")
        != "not_evaluated_until_complete_real_gpu_two_roster_result"
    ):
        raise ValueError("Content V7 formal identity or status differs")
    if _mapping(config, "generation_runtime") != {
        "model_id": _MODEL_ID,
        "inference_steps": 20,
        "injection_step_index_zero_based": 18,
        "generation_rule": (
            "same_seed_callback_free_pass1_sole_primary_null_then_pass2_writer"
        ),
    }:
        raise ValueError("Content V7 generation runtime differs")
    if _mapping(config, "content_analysis") != _CONTENT_ANALYSIS:
        raise ValueError("Content V7 content analysis differs")
    if _mapping(config, "method_identities") != {
        "content_method_id": CONTENT_V7_METHOD_ID,
        "source_writer_id": "content_v3_unweighted_lf_adaptive_hf_v1",
        "hf_base_carrier_method_id": "hf_tail_rademacher_v1",
        "hf_base_evaluated_candidate_id": "hf_tail_rademacher_v1_rankgate_v2",
        "lf_base_carrier_method_id": "lf_shell_balanced_blocks_v2",
        "lf_base_evaluated_candidate_id": (
            "lf_shell_balanced_blocks_v2_blocknorm_median_v1"
        ),
        "lf_embedding_transform_id": (
            "content_v7_ordinary_iss_lf_preprojection_multiplier_v1"
        ),
        "hf_embedding_transform_id": (
            "hf_content_tiles_semantic_gate_texture_two_scale_response_consistency_sensitivity_v1"
        ),
        "combined_budget_projector_id": "dual_branch_actual_dtype_relative_l2_v1",
        "evaluated_candidate_id": CONTENT_V7_EVALUATED_CANDIDATE_ID,
    }:
        raise ValueError("Content V7 method identities differ")
    if _mapping(config, "lf_detection_operator") != {
        "callable_id": "cegwm.method.lf.score_lf_image",
        "observation": "final_ordinary_RGB",
        "carrier_method_id": "lf_shell_balanced_blocks_v2",
        "detector_statistic_id": "lf_block_centered_normalized_median_corr_v2",
        "evaluated_candidate_id": (
            "lf_shell_balanced_blocks_v2_blocknorm_median_v1"
        ),
    }:
        raise ValueError("Content V7 LF detection operator differs")
    if _mapping(config, "iss_fit") != {
        "development_manifest": V7_DEVELOPMENT_MANIFEST,
        "development_manifest_sha256": V7_DEVELOPMENT_MANIFEST_SHA256,
        "fixed_units": 32,
        "development_key_domain": (
            "stage-a/content-v7-ordinary-iss-development-key/v1"
        ),
        "beta_development": 1,
        "gain": "median(beta1_registered-host_registered)",
        "competition": "max(host_registered,16_beta1_wrong)",
        "target": "rank28(competition)+2^-12",
        "fit_failure_prevents_evaluation": True,
        "asset_publication": "atomic_create_only_pair_before_evaluation",
        "refit_or_fallback_allowed": False,
    }:
        raise ValueError("Content V7 ISS fit contract differs")
    if _mapping(config, "budget") != {
        "combined_total_relative_l2": 0.012,
        "measurement": "actual_dtype_final_minus_actual_dtype_base",
        "single_shared_budget_not_per_branch": True,
        "both_effective_branches_nonzero": True,
        "beta_application": "LF_preprojection_delta_only",
        "hf_preprojection_delta": "unchanged_from_content_v3",
        "joint_projector": "unchanged_content_v3_common_actual_dtype_relative_l2",
    }:
        raise ValueError("Content V7 budget or beta application differs")
    if _mapping(config, "aggregate_measurement") != _AGGREGATE_MEASUREMENT:
        raise ValueError("Content V7 aggregate measurement differs")
    detection = _mapping(config, "detection_access")
    if detection.get("allowed_inputs") != [
        "image", "detection_key", "frozen_public_assets"
    ] or detection.get("joint_score") != "min(s_LF,s_HF)":
        raise ValueError("Content V7 blind detection access differs")
    forbidden = detection.get("forbidden_inputs")
    if not isinstance(forbidden, list) or any(
        item not in forbidden
        for item in (
            "original_image", "embed_record", "private_latent", "embedding_latent",
            "embed_side_route", "host_observation", "beta",
        )
    ):
        raise ValueError("Content V7 detector forbidden inputs differ")
    if _mapping(config, "keying") != {
        "task": "zero_bit_keyed_attribution",
        "normalization": "NFC_UTF8_for_text_exact_bytes_for_binary",
        "prg": "HMAC_SHA256_counter_v1",
        "wrong_key_count": 16,
        "wrong_key_derivation_domain": _WRONG_KEY_DOMAIN,
        "primary_null": True,
        "payload_bits": 0,
    }:
        raise ValueError("Content V7 keying differs")
    flow = _mapping(config, "execution_flow")
    if flow != {
        "phase_order": ["fit_and_publish_asset", "evaluation_01", "evaluation_02", "terminal"],
        "evaluation_invocations": [
            {
                "invocation_id": "evaluation_01_content_v3_roster",
                "roster_manifest": V7_EVALUATION_1_MANIFEST,
                "roster_sha256": V7_EVALUATION_1_MANIFEST_SHA256,
                "fixed_units": 8,
                "records_per_unit": 2,
            },
            {
                "invocation_id": "evaluation_02_content_v6_roster",
                "roster_manifest": V7_EVALUATION_2_MANIFEST,
                "roster_sha256": V7_EVALUATION_2_MANIFEST_SHA256,
                "fixed_units": 8,
                "records_per_unit": 2,
            },
        ],
        "independent_failures_denominators_and_gates": True,
        "pooling_allowed": False,
        "outcome_conditioned_control_allowed": False,
        "cross_invocation_resume_allowed": False,
        "implicit_resume_allowed": False,
        "interruption_action": "stop",
        "terminal_result_count": 2,
        "terminal_reporting": {
            "evaluation_01": {"fixed_units": 8, "fixed_records": 16},
            "evaluation_02": {"fixed_units": 8, "fixed_records": 16},
        },
        "cross_cohort_conjunction_allowed": False,
        "combined_result_allowed": False,
    }:
        raise ValueError("Content V7 formal execution flow differs")
    if _mapping(config, "decision_rule") != {
        "fixed_units_per_invocation": 8,
        "fixed_records_per_invocation": 16,
        "registered_top_rank_among_17_min_units_per_branch": 7,
        "joint_registered_gt_primary_null_registered_min_units_per_branch": 7,
        "strict_comparison_ties_fail": True,
        "quality_and_runtime_gates_apply_separately": True,
        "paired_rgb_psnr_min_db": 30.0,
        "formal_fpr_claim": False,
    }:
        raise ValueError("Content V7 decision rule differs")
    if config.get("limitations") != [
        "cpu_and_fake_tests_are_engineering_only",
        "no_scientific_completion_without_complete_real_gpu_two_roster_result",
        "no_calibrated_threshold_or_fixed_fpr_claim",
        "clean_only_no_attack_or_geometry_claim",
    ]:
        raise ValueError("Content V7 limitations differ")
    expected_fields = {
        "protocol_version", "protocol_id", "execution_scope_id", "scientific_status",
        "generation_runtime", "content_analysis", "method_identities",
        "lf_detection_operator", "iss_fit", "budget", "aggregate_measurement",
        "detection_access", "keying", "execution_flow", "decision_rule", "limitations",
    }
    if set(config) != expected_fields:
        raise ValueError("Content V7 config fields differ")


def load_content_v7_formal_protocol(
    repo_root: str | Path,
) -> ContentV7FormalProtocol:
    root = Path(repo_root)
    config_path = root / "configs" / "content_chain" / V7_FORMAL_CONFIG
    try:
        config = json.loads(config_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Content V7 config must be UTF-8 JSON") from error
    if not isinstance(config, dict):
        raise ValueError("Content V7 config must be an object")
    _validate_config(config)
    data = load_content_v7_data_contract(root)
    canonical = json.dumps(
        {
            "config": config,
            "development": [asdict(unit) for unit in data.development],
            "evaluations": [
                [asdict(unit) for unit in roster] for roster in data.evaluations
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    protocol_digest = hashlib.sha256(canonical).hexdigest()
    evaluation_protocols: list[ContentChainProtocol] = []
    for index, roster in enumerate(data.evaluations, 1):
        invocation_id = config["execution_flow"]["evaluation_invocations"][index - 1][
            "invocation_id"
        ]
        digest = hashlib.sha256(
            protocol_digest.encode("ascii")
            + b"\0"
            + invocation_id.encode("ascii")
            + b"\0"
            + json.dumps(
                [asdict(unit) for unit in roster],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        evaluation_protocols.append(ContentChainProtocol(
            protocol_id=f"{CONTENT_V7_PROTOCOL_ID}/{invocation_id}",
            config=_freeze(config),
            roster=roster,
            protocol_digest=digest,
        ))
    return ContentV7FormalProtocol(
        CONTENT_V7_PROTOCOL_ID,
        _freeze(config),
        data,
        (evaluation_protocols[0], evaluation_protocols[1]),
        protocol_digest,
    )


__all__ = [
    "CONTENT_V7_ARMS",
    "CONTENT_V7_EVALUATED_CANDIDATE_ID",
    "CONTENT_V7_EXECUTION_SCOPE_ID",
    "CONTENT_V7_METHOD_ID",
    "CONTENT_V7_PROTOCOL_ID",
    "CONTENT_V7_RECORD_CONTRACT_ID",
    "CONTENT_V7_RUN_PREFIX",
    "ContentV7DataContract",
    "ContentV7FormalProtocol",
    "ContentV7Unit",
    "V7_DEVELOPMENT_MANIFEST",
    "V7_DEVELOPMENT_MANIFEST_SHA256",
    "V7_DEVELOPMENT_PROMPT_LIST_SHA256",
    "V7_DEVELOPMENT_SPLIT",
    "V7_EVALUATION_1_MANIFEST_SHA256",
    "V7_EVALUATION_2_MANIFEST_SHA256",
    "load_content_v7_data_contract",
    "load_content_v7_formal_protocol",
]
