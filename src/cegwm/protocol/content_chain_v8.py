"""Frozen loader for the one-invocation Content V8 formal-initial protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

V8_CONFIG = "content_v8_v2_spatial_lf_iss_formal_initial_v1.json"
V8_DEVELOPMENT_MANIFEST = "content_v6_iss_development_v1.jsonl"
V8_DEVELOPMENT_MANIFEST_SHA256 = "4ff3efa6b98efb62d542b210ebf00f3fc624513342475ce417e9099e334066ea"
V8_DEVELOPMENT_PROMPT_LIST_SHA256 = "fd2120c0ed9be832687a30de85d38dac5fb2abb23b7bd372c7d327d004cbc9ba"
V8_EVALUATION_MANIFESTS_IN_ORDER = (
    "content_adaptive_dual_branch_v2_clean.jsonl",
    "content_v6_iss_clean.jsonl",
)
V8_EVALUATION_MANIFEST_SHA256_IN_ORDER = (
    "dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88",
    "20058788bfe7d75878e7263efda2b8de94c6fdcd3a963f64368f2ba4d594868f",
)
V8_EVALUATION_PROMPT_LIST_SHA256_IN_ORDER = (
    "e887479fbfda23d2eddd3d8a6a354e0baac18446e4ebce09bc64a768fa1b43f6",
    "ec1b29c673fa109c6078b3dc070d3dd42aa93f834aaaf387d282aa475bd2b219",
)
V8_DEVELOPMENT_KEY_DOMAIN = "stage-a/content-v8-v2-spatial-lf-iss-development-key/v1"
V8_WRONG_KEY_DOMAIN = "stage-a/content-adaptive-v2-external-wrong-key/v1"

_FIELDS = ("unit_id", "split", "source_id", "prompt", "seed", "height", "width")
_V2_EVALUATION_SEEDS = (
    1213061, 1238321, 1263581, 1288843,
    1314103, 1339367, 1364627, 1389887,
)


@dataclass(frozen=True, slots=True)
class ContentV8Unit:
    unit_id: str
    split: str
    source_id: str
    prompt: str
    seed: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class ContentV8Roster:
    role: str
    manifest: str
    manifest_sha256: str
    units: tuple[ContentV8Unit, ...]


@dataclass(frozen=True, slots=True)
class ContentV8Protocol:
    protocol_id: str
    execution_scope_id: str
    config: Mapping[str, Any]
    development: tuple[ContentV8Unit, ...]
    evaluation_rosters: tuple[ContentV8Roster, ...]
    protocol_digest: str


def _stable_line(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _expected_identity(role: str, ordinal: int) -> tuple[str, str, str, int]:
    if role == "development":
        prefix = "content-v6-iss-dev"
        return (
            f"{prefix}-{ordinal:04d}",
            "content_v6_iss_development_v1",
            f"{prefix}-source-{ordinal:04d}",
            2026082400 + ordinal - 1,
        )
    if role == "content_v2_reference":
        return (
            f"content-adaptive-v2-{ordinal:04d}",
            "content_adaptive_dual_branch_v2_clean_v1",
            f"content-v2-prompt-{8100 + ordinal}",
            _V2_EVALUATION_SEEDS[ordinal - 1],
        )
    if role == "content_v6_current":
        prefix = "content-v6-iss-eval"
        return (
            f"{prefix}-{ordinal:04d}",
            "content_v6_iss_clean_v1",
            f"{prefix}-source-{ordinal:04d}",
            2026082500 + ordinal - 1,
        )
    raise ValueError("unknown Content V8 roster role")


def _load_jsonl(
    path: Path,
    *,
    role: str,
    expected_count: int,
    expected_sha256: str,
    prompt_sha256: str,
) -> tuple[ContentV8Unit, ...]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256 or not raw.endswith(b"\n"):
        raise ValueError(f"Content V8 {role} manifest bytes differ")
    lines = raw.splitlines()
    if len(lines) != expected_count:
        raise ValueError(f"Content V8 {role} denominator differs")
    units: list[ContentV8Unit] = []
    for ordinal, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Content V8 manifests must be UTF-8 JSONL") from error
        if not isinstance(value, dict) or tuple(value) != _FIELDS or _stable_line(value) != line:
            raise ValueError("Content V8 manifest fields, order, or encoding differ")
        if any(not isinstance(value[name], str) or not value[name].strip() for name in _FIELDS[:4]):
            raise ValueError("Content V8 manifest text identity is empty")
        unit_id, split, source_id, seed = _expected_identity(role, ordinal)
        if (
            (value["unit_id"], value["split"], value["source_id"], value["seed"])
            != (unit_id, split, source_id, seed)
            or value["height"] != 512
            or value["width"] != 512
            or isinstance(value["seed"], bool)
        ):
            raise ValueError(f"Content V8 {role} ordered identity differs")
        units.append(ContentV8Unit(**value))
    prompt_bytes = b"".join(unit.prompt.encode("utf-8") + b"\n" for unit in units)
    if hashlib.sha256(prompt_bytes).hexdigest() != prompt_sha256:
        raise ValueError(f"Content V8 {role} ordered prompt identity differs")
    return tuple(units)


def _identity_sets(units: Iterable[ContentV8Unit]) -> tuple[set[Any], ...]:
    received = tuple(units)
    return (
        {item.unit_id for item in received},
        {item.source_id for item in received},
        {item.prompt for item in received},
        {item.seed for item in received},
        {(item.prompt, item.seed) for item in received},
    )


def _require_unique_and_disjoint(groups: tuple[tuple[ContentV8Unit, ...], ...]) -> None:
    expected = (32, 8, 8)
    sets = tuple(_identity_sets(group) for group in groups)
    for group_sets, count in zip(sets, expected, strict=True):
        if any(len(values) != count for values in group_sets):
            raise ValueError("Content V8 identities must be unique within each roster")
    for left_index, left in enumerate(sets):
        for right in sets[left_index + 1:]:
            if any(first & second for first, second in zip(left, right, strict=True)):
                raise ValueError("Content V8 development and evaluation identities overlap")


def _validate_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise ValueError("Content V8 config must be an object")
    if (
        config.get("protocol_version") != 1
        or config.get("protocol_id") != "cegwm-stage-a-content-v8-v2-spatial-lf-detector-domain-iss-formal-initial-v1"
        or config.get("execution_scope_id") != "content_v8_v2_spatial_lf_detector_domain_iss_formal_initial_v1"
        or config.get("scientific_status") != "not_evaluated_until_complete_real_gpu_execution"
    ):
        raise ValueError("Content V8 top-level protocol identity differs")
    runtime = config.get("generation_runtime")
    if runtime != {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "inference_steps": 20,
        "injection_step_index_zero_based": 18,
        "probe_evaluations_per_write": 64,
        "pass_rule": "callback_free_pass1_is_sole_primary_null_then_independent_same_seed_pass2_is_sole_write",
    }:
        raise ValueError("Content V8 runtime identity differs")
    fit = config.get("fit")
    if fit != {
        "development_manifest": V8_DEVELOPMENT_MANIFEST,
        "development_manifest_sha256": V8_DEVELOPMENT_MANIFEST_SHA256,
        "development_count": 32,
        "development_key_domain": V8_DEVELOPMENT_KEY_DOMAIN,
        "beta_development": 1.0,
        "gain_formula": "median_of_32_beta1_ordinary_registered_minus_ordinary_registered_host",
        "target_formula": "rank28_of_max_ordinary_registered_host_and_16_beta1_ordinary_wrong_scores_plus_2^-12",
        "failure_rule": "exactly_32_finite_measurements_positive_gain_and_bounded_target_or_stop_before_evaluation",
    }:
        raise ValueError("Content V8 fit identity differs")
    if config.get("method") != {
        "evaluated_candidate_id": "content_v8_v2_spatial_lf_detector_domain_iss_semantic_gate_v1",
        "lf_write_rule": "beta_times_A_LF_content_times_normalize_lf_tile_weights_content_hadamard_c_LF",
        "hf_write_rule": "unchanged_content_v2_spatial_hf_preprojection_delta",
        "projector": "shared_actual_dtype_relative_l2_at_most_0.012",
        "controller": "beta_equals_clamp_(m-h)/g_inclusive_1_to_2",
        "host_score": "cegwm.method.lf.score_lf_image_on_pass1_ordinary_final_RGB_registered_key",
        "joint_score": "min(lf,hf)",
        "forbidden_detector_transforms": ["whitening", "detrending", "DCT"],
    }:
        raise ValueError("Content V8 method identity differs")
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict) or (
        evaluation.get("units_per_roster"),
        evaluation.get("records_per_unit"),
        evaluation.get("wrong_key_count"),
        evaluation.get("wrong_key_domain"),
        evaluation.get("separation_rule"),
    ) != (
        8, 2, 16, V8_WRONG_KEY_DOMAIN,
        "independent_records_failures_denominators_and_gates_no_pooling_cross_resume_or_outcome_control",
    ):
        raise ValueError("Content V8 evaluation separation contract differs")
    roster_specs = evaluation.get("rosters_in_order")
    if (
        not isinstance(roster_specs, list)
        or len(roster_specs) != 2
        or any(not isinstance(item, dict) for item in roster_specs)
        or tuple(
        (item.get("role"), item.get("manifest"), item.get("sha256"))
        for item in roster_specs
    ) != (
        ("content_v2_reference", V8_EVALUATION_MANIFESTS_IN_ORDER[0], V8_EVALUATION_MANIFEST_SHA256_IN_ORDER[0]),
        ("content_v6_current", V8_EVALUATION_MANIFESTS_IN_ORDER[1], V8_EVALUATION_MANIFEST_SHA256_IN_ORDER[1]),
    )):
        raise ValueError("Content V8 evaluation roster order differs")
    if evaluation.get("gates") != {
        "registered_top_rank_among_17_min_units": 7,
        "registered_write_gt_primary_null_registered_min_units": 7,
        "strict_ties_fail": True,
        "combined_budget_pass_units": 8,
        "both_nonzero_branches_pass_units": 8,
        "probe_evaluation_count_64_pass_units": 8,
        "paired_rgb_psnr_min_db": 30.0,
        "paired_rgb_psnr_pass_units": 8,
    }:
        raise ValueError("Content V8 strict gate contract differs")
    publication = config.get("publication")
    if publication != {
        "runtime_asset": "canonical_JSON_plus_SHA256_sidecar_create_only_before_evaluation",
        "terminal": "one_create_only_ZIP_plus_SHA256_binding_runtime_asset_and_both_independent_results",
        "resume": False,
    }:
        raise ValueError("Content V8 execution must not imply resume")
    if config.get("limitations") != [
        "user_selected_real_GPU_execution_required",
        "no_fixed_FPR_or_calibrated_threshold_claim",
        "no_attack_geometry_or_rectification_claim",
        "complete_results_are_evaluable_not_scientifically_adjudicated",
    ]:
        raise ValueError("Content V8 limitations differ")


def load_content_v8_protocol(repo_root: str | Path) -> ContentV8Protocol:
    root = Path(repo_root) / "configs" / "content_chain"
    config = json.loads((root / V8_CONFIG).read_bytes())
    _validate_config(config)
    development = _load_jsonl(
        root / V8_DEVELOPMENT_MANIFEST,
        role="development",
        expected_count=32,
        expected_sha256=V8_DEVELOPMENT_MANIFEST_SHA256,
        prompt_sha256=V8_DEVELOPMENT_PROMPT_LIST_SHA256,
    )
    roles = ("content_v2_reference", "content_v6_current")
    rosters = tuple(
        ContentV8Roster(
            role,
            manifest,
            manifest_sha,
            _load_jsonl(
                root / manifest,
                role=role,
                expected_count=8,
                expected_sha256=manifest_sha,
                prompt_sha256=prompt_sha,
            ),
        )
        for role, manifest, manifest_sha, prompt_sha in zip(
            roles,
            V8_EVALUATION_MANIFESTS_IN_ORDER,
            V8_EVALUATION_MANIFEST_SHA256_IN_ORDER,
            V8_EVALUATION_PROMPT_LIST_SHA256_IN_ORDER,
            strict=True,
        )
    )
    _require_unique_and_disjoint((development, *(roster.units for roster in rosters)))
    canonical = json.dumps(
        {
            "config": config,
            "development": [asdict(unit) for unit in development],
            "evaluation_rosters": [
                {"role": roster.role, "manifest": roster.manifest, "units": [asdict(unit) for unit in roster.units]}
                for roster in rosters
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ContentV8Protocol(
        config["protocol_id"],
        config["execution_scope_id"],
        _freeze(config),
        development,
        rosters,
        hashlib.sha256(canonical).hexdigest(),
    )


__all__ = [
    "ContentV8Protocol", "ContentV8Roster", "ContentV8Unit",
    "V8_CONFIG", "V8_DEVELOPMENT_KEY_DOMAIN", "V8_DEVELOPMENT_MANIFEST_SHA256",
    "V8_EVALUATION_MANIFEST_SHA256_IN_ORDER", "V8_WRONG_KEY_DOMAIN",
    "load_content_v8_protocol",
]
