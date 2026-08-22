"""Finite, descriptive-only plan for the standalone LF/HF frequency response."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL_ID = "standalone-lf-hf-frequency-response-v1"
EVIDENCE_CONTRACT = "STANDALONE_LF_HF_FREQUENCY_RESPONSE_EVIDENCE"
CONDITIONS = (
    "identity", "jpeg_q90", "jpeg_q75", "jpeg_q50", "gaussian_blur_sigma_0_5",
    "gaussian_blur_sigma_1", "gaussian_blur_sigma_2", "gaussian_noise_std_0_005",
    "gaussian_noise_std_0_01", "gaussian_noise_std_0_02",
)
HF_ARM = "hf_tail_rademacher_v1_rankgate_v2"
LF_ARM = "lf_shell_balanced_blocks_v2_blocknorm_median_v1"
RECORD_ARMS = (HF_ARM, f"primary_null__{HF_ARM}", LF_ARM, f"primary_null__{LF_ARM}")
_DETECTION_ACCESS = {
    "allowed_inputs": ["image", "detection_key", "frozen_public_assets"],
    "forbidden_inputs": [
        "original_image", "prompt", "embed_record", "private_latent", "embedding_latent",
        "embed_side_route", "route", "mask", "cached_qk", "qk",
    ],
}
_GENERATION_RUNTIME = {
    "model_id": "stabilityai/stable-diffusion-3.5-medium",
    "inference_steps": 20,
    "same_seed_independent_generators": ["hf", "lf", "plain"],
}
_METHODS = {
    "hf": {
        "carrier_method_id": "hf_tail_rademacher_v1",
        "evaluated_candidate_id": HF_ARM,
        "detector_statistic_id": "vae_reencode_hf_masked_normalized_correlation",
    },
    "lf": {
        "carrier_method_id": "lf_shell_balanced_blocks_v2",
        "evaluated_candidate_id": LF_ARM,
        "detector_statistic_id": "lf_block_centered_normalized_median_corr_v2",
    },
}
_BUDGET = {
    "actual_callback_dtype_relative_l2_per_method_max": 0.012,
    "independent_full_budget": True,
    "never_coinject_or_fuse": True,
}
_KEYING = {
    "wrong_key_count": 16,
    "wrong_key_derivation_domain": "stage-a/frequency-response/wrong-key/v1",
}
_TRANSFORM_CONTRACT = {
    "ordinary_image_boundary": {
        "input": "ordinary_rgb8",
        "output": "ordinary_rgb8_same_shape",
    },
    "identity": {"pixel_operation": "pass_through"},
    "jpeg": {
        "implementation": "pillow_in_memory_encode_decode",
        "quality_by_condition": {"jpeg_q90": 90, "jpeg_q75": 75, "jpeg_q50": 50},
        "encode": {
            "format": "JPEG", "subsampling": 2, "optimize": False, "progressive": False,
            "exif": "empty", "icc_profile": None,
        },
        "decode_mode": "RGB",
    },
    "gaussian_blur": {
        "sigma_by_condition": {
            "gaussian_blur_sigma_0_5": 0.5,
            "gaussian_blur_sigma_1": 1.0,
            "gaussian_blur_sigma_2": 2.0,
        },
        "minimum_height_and_width": 2,
        "radius_rule": "ceil(3*sigma)",
        "kernel": "normalized_exp(-0.5*(coordinate/sigma)^2)",
        "computation_dtype": "float64",
        "pass_order": ["horizontal", "vertical"],
        "padding": "numpy_reflect",
        "quantization": "clip_0_1_then_multiply_255_rint_uint8",
    },
    "gaussian_noise": {
        "std_by_condition": {
            "gaussian_noise_std_0_005": 0.005,
            "gaussian_noise_std_0_01": 0.01,
            "gaussian_noise_std_0_02": 0.02,
        },
        "public_noise_root": {
            "derivation": "sha256",
            "input_utf8": "CEG-WM/frequency-response/public-noise/v1",
        },
        "public_noise_domain": {
            "prefix": "frequency-response/public-noise/v1/",
            "payload_fields": [
                "protocol_id", "condition", "unit_id", "source_id", "generation_seed", "height", "width",
            ],
            "canonicalization": "json_sort_keys_compact_utf8_sha256_hex",
            "independent_of": ["key", "method", "pixels", "outcome"],
        },
        "generator": {
            "function": "prg_normal",
            "key": "fixed_public_noise_root",
            "domain": "public_noise_domain",
            "shape": "ordinary_rgb_pixels_shape",
            "dtype": "float64",
        },
        "application": "float64_rgb8_div_255_plus_std_times_noise",
        "quantization": "clip_0_1_then_multiply_255_rint_uint8",
    },
}
_EXECUTION = {
    "fixed_units": 8,
    "records_per_unit": 40,
    "fixed_records": 320,
    "failures_remain_in_denominator": True,
    "replacement_units_allowed": False,
    "complete_rc": 0,
    "automatic_fresh_or_resume": True,
    "checkpoint_interval_hours": 2.0,
    "checkpoint_only_after_new_complete_unit": True,
    "short_run_final_only": True,
    "checkpoint_schema": "standalone-lf-hf-frequency-response-checkpoint-v1",
    "committed_unit_transactions_immutable": True,
    "active_state_location": "local_only",
    "artifact_sink_pairs": [
        "complete_checkpoint_zip_and_sha256",
        "complete_final_zip_and_sha256",
        "complete_failure_zip_and_sha256",
    ],
    "artifact_publication": "create_only",
    "terminal_pair_prevents_rerun": True,
}
_LIMITATIONS = [
    "descriptive_per_method_response_only",
    "no_calibrated_threshold_or_fixed_fpr_claim",
    "no_winner_complementarity_joint_content_gate_or_robustness_promotion",
    "ordinary_rgb_attacks_only",
]
_CONFIG_KEYS = {
    "protocol_id", "evidence_contract", "detection_access", "generation_runtime", "methods", "budget",
    "keying", "conditions", "record_arms_in_exact_condition_order", "transform_contract", "execution",
    "limitations",
}


@dataclass(frozen=True, slots=True)
class FrequencyResponseUnit:
    unit_id: str
    source_id: str
    prompt: str
    seed: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class FrequencyResponsePlan:
    protocol_id: str
    protocol_digest: str
    roster_digest: str
    model_id: str
    method_identities: dict[str, dict[str, str]]
    units: tuple[FrequencyResponseUnit, ...]

    @property
    def config_digest(self) -> str:
        """Compatibility name used by StageARecord for the full protocol digest."""

        return self.protocol_digest


def _canonical_digest(payload: Any) -> str:
    payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _matches_exact_json_literal(actual: Any, expected: Any) -> bool:
    return json.dumps(actual, sort_keys=True, separators=(",", ":")) == json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    )


def load_plan(config_path: Path, roster_path: Path) -> FrequencyResponsePlan:
    """Load the fixed plan before runtime or ordinary-image observation begins."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    roster = [json.loads(line) for line in roster_path.read_text(encoding="utf-8").splitlines() if line]
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise ValueError("frequency-response config fields differ")
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("frequency-response protocol identity differs")
    if config.get("evidence_contract") != EVIDENCE_CONTRACT:
        raise ValueError("frequency-response evidence contract differs")
    if not _matches_exact_json_literal(config.get("detection_access"), _DETECTION_ACCESS):
        raise ValueError("blind detection access contract differs")
    if not _matches_exact_json_literal(config.get("generation_runtime"), _GENERATION_RUNTIME):
        raise ValueError("generation runtime contract differs")
    if not _matches_exact_json_literal(config.get("methods"), _METHODS):
        raise ValueError("carrier, candidate, or detector identity differs")
    if not _matches_exact_json_literal(config.get("budget"), _BUDGET):
        raise ValueError("method budget contract differs")
    if not _matches_exact_json_literal(config.get("keying"), _KEYING):
        raise ValueError("wrong-key contract differs")
    if tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError("frequency-response condition order differs")
    if tuple(config.get("record_arms_in_exact_condition_order", ())) != RECORD_ARMS:
        raise ValueError("frequency-response detector arm order differs")
    if not _matches_exact_json_literal(config.get("transform_contract"), _TRANSFORM_CONTRACT):
        raise ValueError("ordinary RGB transform contract differs")
    if not _matches_exact_json_literal(config.get("execution"), _EXECUTION):
        raise ValueError("fixed denominator, failure, or completion contract differs")
    if not _matches_exact_json_literal(config.get("limitations"), _LIMITATIONS):
        raise ValueError("descriptive-only limitation contract differs")
    units = tuple(FrequencyResponseUnit(**entry) for entry in roster)
    if len(units) != 8:
        raise ValueError("frequency-response roster must contain exactly 8 units")
    for name, values in {
        "unit": [unit.unit_id for unit in units], "source": [unit.source_id for unit in units],
        "prompt": [unit.prompt for unit in units], "seed": [unit.seed for unit in units],
    }.items():
        if len(values) != len(set(values)):
            raise ValueError(f"frequency-response roster has duplicate {name} identity")
    if any(not unit.prompt.strip() or unit.height < 256 or unit.width < 256 or unit.seed < 0 for unit in units):
        raise ValueError("frequency-response roster has invalid generation identity")
    return FrequencyResponsePlan(
        protocol_id=config["protocol_id"],
        protocol_digest=_canonical_digest(config),
        roster_digest=_canonical_digest(roster),
        model_id=config["generation_runtime"]["model_id"],
        method_identities=config["methods"],
        units=units,
    )


def expected_pairs() -> tuple[tuple[str, str], ...]:
    return tuple((condition, arm) for condition in CONDITIONS for arm in RECORD_ARMS)
