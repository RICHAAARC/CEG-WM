"""Loader and fail-closed validation for the finite Stage-A protocol."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

_ALLOWED_DETECTION_INPUTS = (
    "image",
    "detection_key",
    "frozen_public_assets",
)
_FORBIDDEN_DETECTION_INPUTS = {
    "original_image",
    "prompt",
    "embed_record",
    "private_latent",
    "embedding_latent",
    "embed_side_route",
    "route",
    "mask",
    "cached_qk",
    "qk",
}
_UNIT_FIELDS = {"unit_id", "split", "source_id", "prompt", "seed", "height", "width"}


@dataclass(frozen=True, slots=True)
class StageAUnit:
    unit_id: str
    split: str
    source_id: str
    prompt: str
    seed: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class StageAProtocol:
    """The frozen choices plus disjoint selection and confirmation units."""

    protocol_id: str
    config: Mapping[str, Any]
    candidate_selection: tuple[StageAUnit, ...]
    untouched_confirmation: tuple[StageAUnit, ...]
    protocol_digest: str


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _require_nonempty_text(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text")
    return value


def _load_units(path: Path, expected_split: str) -> tuple[StageAUnit, ...]:
    units: list[StageAUnit] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number} cannot be blank")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number} is not valid JSON") from error
            if not isinstance(payload, dict) or set(payload) != _UNIT_FIELDS:
                raise ValueError(f"{path.name}:{line_number} has unexpected unit fields")
            for field in ("unit_id", "split", "source_id", "prompt"):
                _require_nonempty_text(payload, field)
            if payload["split"] != expected_split:
                raise ValueError(f"{path.name}:{line_number} has the wrong split")
            for field in ("seed", "height", "width"):
                value = payload[field]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f"{path.name}:{line_number} {field} must be an integer")
            if payload["seed"] < 0 or payload["height"] < 256 or payload["width"] < 256:
                raise ValueError(f"{path.name}:{line_number} has an invalid seed or image size")
            units.append(StageAUnit(**payload))
    if not units:
        raise ValueError(f"{path.name} must contain at least one fixed unit")
    identifiers = [unit.unit_id for unit in units]
    sources = [unit.source_id for unit in units]
    if len(identifiers) != len(set(identifiers)) or len(sources) != len(set(sources)):
        raise ValueError(f"{path.name} repeats a unit_id or source_id")
    return tuple(units)


def _validate_detection_access(config: Mapping[str, Any]) -> None:
    access = _require_mapping(config, "detection_access")
    allowed = access.get("allowed_inputs")
    forbidden = access.get("forbidden_inputs")
    if not isinstance(allowed, list) or tuple(allowed) != _ALLOWED_DETECTION_INPUTS:
        raise ValueError(
            "detection allowed_inputs must be exactly image, detection_key, frozen_public_assets"
        )
    if not isinstance(forbidden, list) or set(forbidden) != _FORBIDDEN_DETECTION_INPUTS:
        raise ValueError("detection forbidden_inputs must enumerate every private-state input")
    if set(allowed) & _FORBIDDEN_DETECTION_INPUTS:
        raise ValueError("detection allowed_inputs contains private embedding state")
    if access.get("threshold_status") != "deferred_calibration_not_stage_a":
        raise ValueError("Stage A cannot define or claim a calibrated formal threshold")


def _validate_finite_method_choices(config: Mapping[str, Any]) -> None:
    keying = _require_mapping(config, "keying")
    if keying.get("task") != "zero_bit_keyed_attribution":
        raise ValueError("Stage A is zero-bit keyed attribution, not payload recovery")
    if keying.get("wrong_key_count") != 16 or keying.get("primary_null") is not True:
        raise ValueError("wrong-key count and primary-null must remain predeclared and separate")

    bands = _require_mapping(config, "bands")
    lf = bands.get("lf_radius")
    hf = bands.get("hf_radius")
    if not (
        isinstance(lf, list)
        and isinstance(hf, list)
        and len(lf) == len(hf) == 2
        and all(isinstance(value, (int, float)) for value in (*lf, *hf))
        and 0.0 <= lf[0] < lf[1] < hf[0] < hf[1] <= 1.0
    ):
        raise ValueError("LF and HF radial bands must be finite and mutually exclusive")

    budget = _require_mapping(config, "budget")
    relative_l2 = budget.get("total_relative_l2")
    if not isinstance(relative_l2, (int, float)) or not 0.0 < relative_l2 <= 0.02:
        raise ValueError("Stage-A total relative L2 budget must be in (0, 0.02]")
    if budget.get("measurement") != "actual_dtype_final_minus_actual_dtype_base":
        raise ValueError("budget must be measured on the actual-dtype perturbation")
    if budget.get("shared_across_active_carriers") is not True:
        raise ValueError("all active carriers must share one total budget")

    hf_anchor = _require_mapping(config, "hf_anchor")
    _require_nonempty_text(hf_anchor, "candidate_id")
    if hf_anchor.get("uses_lf") is not False:
        raise ValueError("the HF anchor must remain independent of LF")
    lf_candidates = config.get("lf_candidates")
    if not isinstance(lf_candidates, list) or not 1 <= len(lf_candidates) <= 3:
        raise ValueError("Stage A must contain between one and three finite LF candidates")
    candidate_ids = []
    for candidate in lf_candidates:
        if not isinstance(candidate, dict):
            raise ValueError("every LF candidate must be an object")
        candidate_ids.append(_require_nonempty_text(candidate, "candidate_id"))
        if candidate.get("uses_hf") is not False:
            raise ValueError("LF candidates must be evaluated independently before any fusion")
        subband = candidate.get("radial_subband")
        if not (
            isinstance(subband, list)
            and len(subband) == 2
            and all(isinstance(value, (int, float)) for value in subband)
            and lf[0] <= subband[0] < subband[1] <= lf[1]
        ):
            raise ValueError("every LF candidate subband must lie inside the frozen LF band")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("LF candidate IDs must be unique")

    attacks = config.get("attacks")
    if not isinstance(attacks, list) or not attacks:
        raise ValueError("attacks must be a finite non-empty list")
    attack_ids: list[str] = []
    for attack in attacks:
        if not isinstance(attack, dict):
            raise ValueError("every attack must be an object")
        attack_ids.append(_require_nonempty_text(attack, "attack_id"))
        if attack.get("kind") not in {"clean", "non_geometric"}:
            raise ValueError("Stage A permits only clean and preregistered non-geometric attacks")
    if len(attack_ids) != len(set(attack_ids)) or attack_ids.count("identity") != 1:
        raise ValueError("attack IDs must be unique and include identity exactly once")

    flow = _require_mapping(config, "execution_flow")
    if flow.get("attack_evaluation_requires_clean_lf_pass") is not True:
        raise ValueError("attack evaluation must stop unless clean LF attribution passes")
    if flow.get("confirmation_requires_frozen_candidate") is not True:
        raise ValueError("untouched confirmation requires a frozen selected candidate")
    if flow.get("failure_units_remain_in_denominator") is not True:
        raise ValueError("failed units must remain in the fixed denominator")
    if flow.get("replacement_units_allowed") is not False:
        raise ValueError("replacement or success-subset units are forbidden")

    selection_rule = _require_mapping(config, "selection_rule")
    clean_gate = _require_mapping(selection_rule, "clean_gate")
    attack_gate = _require_mapping(selection_rule, "attack_complementarity_gate")
    confirmation_rule = _require_mapping(config, "confirmation_rule")
    selection_count = flow.get("candidate_selection_units")
    confirmation_count = flow.get("untouched_confirmation_units")
    if clean_gate.get("fixed_units") != selection_count:
        raise ValueError("clean selection gate must use the fixed selection denominator")
    if attack_gate.get("fixed_units_per_attack") != selection_count:
        raise ValueError("every attack must use the fixed selection denominator")
    if confirmation_rule.get("fixed_units") != confirmation_count:
        raise ValueError("confirmation gate must use the fixed confirmation denominator")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def load_stage_a_protocol(
    config_path: str | Path,
    candidate_selection_path: str | Path,
    untouched_confirmation_path: str | Path,
) -> StageAProtocol:
    """Load and validate the exact Stage-A choices and fixed unit manifests."""

    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Stage-A config must be an object")
    if config.get("protocol_version") != 1:
        raise ValueError("unsupported Stage-A protocol_version")
    protocol_id = _require_nonempty_text(config, "protocol_id")
    _validate_detection_access(config)
    _validate_finite_method_choices(config)

    selection = _load_units(Path(candidate_selection_path), "candidate_selection")
    confirmation = _load_units(Path(untouched_confirmation_path), "untouched_confirmation")
    execution = _require_mapping(config, "execution_flow")
    if execution.get("candidate_selection_units") != len(selection):
        raise ValueError("candidate-selection manifest count differs from the frozen denominator")
    if execution.get("untouched_confirmation_units") != len(confirmation):
        raise ValueError("confirmation manifest count differs from the frozen denominator")
    selection_ids = {unit.unit_id for unit in selection}
    confirmation_ids = {unit.unit_id for unit in confirmation}
    selection_sources = {unit.source_id for unit in selection}
    confirmation_sources = {unit.source_id for unit in confirmation}
    if selection_ids & confirmation_ids or selection_sources & confirmation_sources:
        raise ValueError("candidate selection and untouched confirmation must be disjoint")

    digest_payload = {
        "config": config,
        "candidate_selection": [asdict(unit) for unit in selection],
        "untouched_confirmation": [asdict(unit) for unit in confirmation],
    }
    canonical = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return StageAProtocol(
        protocol_id=protocol_id,
        config=_freeze(config),
        candidate_selection=selection,
        untouched_confirmation=confirmation,
        protocol_digest=digest,
    )


def _validate_hf_v2_confirmation_choices(config: Mapping[str, Any]) -> None:
    """Validate only the frozen HF-v2 untouched-confirmation choices."""

    _validate_detection_access(config)

    development = _require_mapping(config, "development_evidence")
    if (
        development.get("candidate_selection_role")
        != "protocol_v1_pilot_only_not_v2_confirmation_evidence"
        or development.get("protocol_v1_outcome") != "preserved_scientific_negative"
    ):
        raise ValueError("HF-v2 must keep protocol-v1 observations as pilot-only provenance")

    runtime = _require_mapping(config, "generation_runtime")
    if runtime.get("model_id") != "stabilityai/stable-diffusion-3.5-medium":
        raise ValueError("HF-v2 must preserve the frozen SD3.5 model name")
    if runtime.get("inference_steps") != 20:
        raise ValueError("HF-v2 must preserve the 20-step runtime")
    if runtime.get("public_asset_rule") != (
        "protocol_model_id_default_hub_resolution_without_revision_or_weight_digest"
    ):
        raise ValueError("HF-v2 public asset identity differs from the frozen runtime")

    keying = _require_mapping(config, "keying")
    if (
        keying.get("task") != "zero_bit_keyed_attribution"
        or keying.get("normalization") != "NFC_UTF8_for_text_exact_bytes_for_binary"
        or keying.get("prg") != "HMAC_SHA256_counter_v1"
        or keying.get("wrong_key_count") != 16
        or keying.get("wrong_key_derivation_domain")
        != "stage-a/external-wrong-key/v1"
        or keying.get("primary_null") is not True
        or keying.get("payload_bits") != 0
    ):
        raise ValueError("HF-v2 must preserve the frozen key and control semantics")

    bands = _require_mapping(config, "bands")
    if bands.get("lf_radius") != [0.04, 0.24] or bands.get("hf_radius") != [0.58, 1.0]:
        raise ValueError("HF-v2 must preserve the frozen frequency bands")

    budget = _require_mapping(config, "budget")
    if (
        budget.get("total_relative_l2") != 0.012
        or budget.get("measurement") != "actual_dtype_final_minus_actual_dtype_base"
        or budget.get("shared_across_active_carriers") is not True
    ):
        raise ValueError("HF-v2 must preserve the actual-dtype 0.012 budget")
    quality = _require_mapping(budget, "quality_evidence")
    if quality != {
        "actual_dtype_relative_l2": "enforced",
        "rgb_psnr": "reported",
        "lpips_alex": "not_evaluated",
    }:
        raise ValueError("HF-v2 quality evidence status differs from the frozen plan")

    candidate = _require_mapping(config, "hf_confirmation_candidate")
    if (
        candidate.get("evaluated_candidate_id") != "hf_tail_rademacher_v1_rankgate_v2"
        or candidate.get("carrier_method_id") != "hf_tail_rademacher_v1"
        or candidate.get("injection_step_index_zero_based") != 18
        or candidate.get("carrier") != "keyed_rademacher_on_hf_rfft_band"
        or candidate.get("blind_score")
        != "vae_reencode_hf_masked_normalized_correlation"
        or candidate.get("uses_lf") is not False
    ):
        raise ValueError("HF-v2 candidate must change only its evaluated decision identity")

    controls = _require_mapping(config, "controls")
    if (
        controls.get("correct_key") != "registered_detection_key"
        or controls.get("wrong_key") != "16_external_domain_separated_keys"
        or controls.get("primary_null") != "same_generation_unit_without_embedding"
        or controls.get("report_wrong_key_and_primary_null_separately") is not True
    ):
        raise ValueError("HF-v2 controls must remain paired and separately reported")

    rule = _require_mapping(config, "confirmation_rule")
    expected_rule_fields = {
        "condition",
        "fixed_units",
        "registered_top_rank_among_17_min_units",
        "exchangeable_key_null_tail_expression",
        "exchangeable_key_null_tail_probability",
        "exchangeable_key_null_interpretation",
        "paired_hf_registered_gt_primary_null_registered_min_units",
        "paired_sign_test_null_tail_expression",
        "paired_sign_test_null_tail_probability",
        "paired_sign_test_interpretation",
        "median_correct_minus_wrong_key_max_role",
        "primary_null_role",
        "wrong_key_role",
        "formal_threshold_status",
        "failure_outcome",
    }
    if set(rule) != expected_rule_fields:
        raise ValueError("HF-v2 confirmation rule contains an unregistered gate field")
    if (
        rule.get("condition") != "identity"
        or rule.get("fixed_units") != 8
        or rule.get("registered_top_rank_among_17_min_units") != 7
        or rule.get("exchangeable_key_null_tail_expression") != "129/17^8"
        or not math.isclose(
            rule.get("exchangeable_key_null_tail_probability", math.nan),
            1.849261547452937e-08,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or rule.get("exchangeable_key_null_interpretation")
        != "rationale_only_not_formal_fpr_claim"
        or rule.get("paired_hf_registered_gt_primary_null_registered_min_units") != 7
        or rule.get("paired_sign_test_null_tail_expression") != "9/256"
        or not math.isclose(
            rule.get("paired_sign_test_null_tail_probability", math.nan),
            0.03515625,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or rule.get("paired_sign_test_interpretation")
        != "supportive_paired_evidence_not_independent_multiplied_p_value"
        or rule.get("median_correct_minus_wrong_key_max_role")
        != "reported_effect_size_only_no_pass_threshold"
        or rule.get("primary_null_role") != "reported_separately_no_pass_cutoff"
        or rule.get("wrong_key_role") != "reported_separately"
        or rule.get("formal_threshold_status") != "not_introduced_at_stage_a"
        or rule.get("failure_outcome")
        != "SCIENTIFIC_NEGATIVE_AND_STOP_IF_EITHER_GATE_FAILS"
    ):
        raise ValueError("HF-v2 rank-gate semantics differ from the preregistration")

    flow = _require_mapping(config, "execution_flow")
    if (
        flow.get("untouched_confirmation_manifest") != "untouched_confirmation.jsonl"
        or flow.get("untouched_confirmation_units") != 8
        or flow.get("identity_only") is not True
        or flow.get("attacks_authorized") is not False
        or flow.get("failure_units_remain_in_denominator") is not True
        or flow.get("replacement_units_allowed") is not False
        or flow.get("operational_failure_counts_as_scientific_failure") is not False
        or flow.get("candidate_outcome_requires_complete_rc0") is not True
        or flow.get("stop_if_confirmation_fails") is not True
    ):
        raise ValueError("HF-v2 execution flow must preserve the fixed confirmation denominator")


def load_hf_v2_confirmation_protocol(
    config_path: str | Path,
    untouched_confirmation_path: str | Path,
) -> StageAProtocol:
    """Load the rank-gate protocol and its previously untouched fixed roster."""

    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("HF-v2 config must be an object")
    if config.get("protocol_version") != 2:
        raise ValueError("unsupported HF-v2 protocol_version")
    protocol_id = _require_nonempty_text(config, "protocol_id")
    if protocol_id != "cegwm-stage-a-hf-v2-rankgate":
        raise ValueError("unexpected HF-v2 protocol identity")
    _validate_hf_v2_confirmation_choices(config)

    confirmation = _load_units(
        Path(untouched_confirmation_path),
        "untouched_confirmation",
    )
    execution = _require_mapping(config, "execution_flow")
    if execution.get("untouched_confirmation_units") != len(confirmation):
        raise ValueError("HF-v2 confirmation manifest differs from the fixed denominator")

    digest_payload = {
        "config": config,
        "untouched_confirmation": [asdict(unit) for unit in confirmation],
    }
    canonical = json.dumps(
        digest_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return StageAProtocol(
        protocol_id=protocol_id,
        config=_freeze(config),
        candidate_selection=(),
        untouched_confirmation=confirmation,
        protocol_digest=digest,
    )


def _validate_lf_a3_selection_choices(config: Mapping[str, Any]) -> None:
    """Validate the finite LF selection protocol without adding a schema layer."""

    _validate_detection_access(config)
    runtime = _require_mapping(config, "generation_runtime")
    if runtime != {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "inference_steps": 20,
        "injection_step_index_zero_based": 18,
        "public_asset_rule": "protocol_model_id_default_hub_resolution_without_revision_or_weight_digest",
        "candidate_generator_rule": "separately_initialized_same_seed_generators_no_latent_or_callback_state_reuse",
        "primary_null_rule": "one_new_same_prompt_seed_no_callback_image_shared_only_as_image_observation",
    }:
        raise ValueError("LF-A3 runtime identity differs from the frozen production path")

    keying = _require_mapping(config, "keying")
    if keying != {
        "task": "zero_bit_keyed_attribution",
        "normalization": "NFC_UTF8_for_text_exact_bytes_for_binary",
        "prg": "HMAC_SHA256_counter_v1",
        "wrong_key_count": 16,
        "wrong_key_derivation_domain": "stage-a/external-wrong-key/v1",
        "carrier_domain_public_inputs": "carrier_method_id_shape_channel_only",
        "primary_null": True,
        "payload_bits": 0,
    }:
        raise ValueError("LF-A3 key or control identity differs from the frozen plan")

    bands = _require_mapping(config, "bands")
    if bands != {
        "radius_definition": "rfft2_radial_frequency_divided_by_2d_nyquist_corner",
        "lf_radius": [0.04, 0.24],
        "core_radius": [0.04, 0.14],
        "core_upper_bound": "exclusive",
        "shell_radius": [0.14, 0.24],
        "shell_upper_bound": "inclusive",
        "partition_rule": "r_equals_0.14_belongs_only_to_shell_disjoint_union_equals_lf_band",
    }:
        raise ValueError("LF-A3 core and shell must be an exact disjoint LF partition")

    budget = _require_mapping(config, "budget")
    if (
        budget.get("total_relative_l2") != 0.012
        or budget.get("measurement") != "actual_dtype_final_minus_actual_dtype_base"
        or budget.get("allocation")
        != "full_single_carrier_budget_per_independent_candidate_never_coinjected"
        or budget.get("quality_evidence") != {
            "actual_dtype_relative_l2": "enforced",
            "rgb_psnr": "reported_candidate_image_vs_shared_primary_null",
            "lpips_alex": "not_evaluated",
        }
    ):
        raise ValueError("LF-A3 must preserve the single-carrier actual-dtype 0.012 budget")

    candidates = config.get("lf_candidates")
    if candidates != [
        {
            "carrier_method_id": "lf_core_rademacher_v1",
            "radial_subband": [0.04, 0.14],
            "upper_bound": "exclusive",
        },
        {
            "carrier_method_id": "lf_shell_rademacher_v1",
            "radial_subband": [0.14, 0.24],
            "upper_bound": "inclusive",
        },
    ]:
        raise ValueError("LF-A3 requires exactly the frozen core and shell candidates")
    arms = config.get("record_arms_in_exact_unit_order")
    if arms != [
        "lf_core_rademacher_v1",
        "primary_null__lf_core_rademacher_v1",
        "lf_shell_rademacher_v1",
        "primary_null__lf_shell_rademacher_v1",
    ]:
        raise ValueError("LF-A3 record arms must preserve the exact four-record unit order")

    controls = _require_mapping(config, "controls")
    if controls != {
        "correct_key": "registered_detection_key",
        "wrong_key": "16_external_domain_separated_keys",
        "primary_null": "one_shared_same_generation_unit_image_scored_independently_by_each_candidate_detector",
        "report_wrong_key_and_primary_null_separately": True,
    }:
        raise ValueError("LF-A3 controls must keep one shared image and candidate-specific scores")

    rule = _require_mapping(config, "selection_rule")
    if rule != {
        "fixed_units_per_candidate": 8,
        "registered_top_rank_among_17_min_units": 7,
        "paired_lf_registered_gt_candidate_scored_primary_null_registered_min_units": 7,
        "strict_comparison_ties_fail": True,
        "candidate_eligibility": "both_rank_and_paired_gates_required",
        "neither_eligible_outcome": "SCIENTIFIC_NEGATIVE_AND_STOP",
        "one_eligible_outcome": "freeze_the_single_eligible_candidate",
        "both_eligible_ranking": [
            "gate_a_count_desc",
            "gate_b_count_desc",
            "median_correct_minus_wrong_key_max_desc",
            "median_lf_minus_primary_null_registered_desc",
            "median_paired_rgb_psnr_desc",
            "candidate_id_asc",
        ],
        "absolute_margin_role": "report_and_ranking_only_no_pass_threshold",
        "primary_null_role": "reported_separately_no_pass_cutoff",
        "formal_threshold_status": "not_introduced_at_stage_a",
        "formal_fpr_claim": False,
    }:
        raise ValueError("LF-A3 selection rule differs from the frozen scale-free gates")

    flow = _require_mapping(config, "execution_flow")
    if flow != {
        "candidate_selection_manifest": "candidate_selection.jsonl",
        "candidate_selection_units": 8,
        "fixed_records": 32,
        "unit_transaction_record_count": 4,
        "untouched_confirmation_manifest_reserved": "untouched_confirmation.jsonl",
        "failure_units_remain_in_denominator": True,
        "replacement_units_allowed": False,
        "outcome_requires_complete_rc0": True,
        "operational_failure_counts_as_scientific_failure": False,
        "checkpoint_committed_units_immutable": True,
        "interrupted_uncommitted_unit_reruns_whole_transaction": True,
    }:
        raise ValueError("LF-A3 execution flow must preserve the 8-unit/32-record denominator")


def load_lf_a3_selection_protocol(
    config_path: str | Path,
    candidate_selection_path: str | Path,
) -> StageAProtocol:
    """Load only the LF candidate-selection roster; confirmation stays untouched."""

    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or config.get("protocol_version") != 1:
        raise ValueError("unsupported LF-A3 protocol version")
    protocol_id = _require_nonempty_text(config, "protocol_id")
    if protocol_id != "cegwm-stage-a-lf-a3-clean-selection-v1":
        raise ValueError("unexpected LF-A3 protocol identity")
    _validate_lf_a3_selection_choices(config)
    selection = _load_units(Path(candidate_selection_path), "candidate_selection")
    if len(selection) != 8 or [unit.unit_id for unit in selection] != [
        f"selection-{index:04d}" for index in range(1, 9)
    ]:
        raise ValueError("LF-A3 selection manifest differs from the fixed roster")
    digest_payload = {
        "config": config,
        "candidate_selection": [asdict(unit) for unit in selection],
    }
    canonical = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return StageAProtocol(
        protocol_id=protocol_id,
        config=_freeze(config),
        candidate_selection=selection,
        untouched_confirmation=(),
        protocol_digest=digest,
    )
