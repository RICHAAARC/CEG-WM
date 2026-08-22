"""Finite, descriptive-only plan for the standalone LF/HF frequency response."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

EVIDENCE_CONTRACT = "STANDALONE_LF_HF_FREQUENCY_RESPONSE_EVIDENCE"
CONDITIONS = (
    "identity", "jpeg_q90", "jpeg_q75", "jpeg_q50", "gaussian_blur_sigma_0_5",
    "gaussian_blur_sigma_1", "gaussian_blur_sigma_2", "gaussian_noise_std_0_005",
    "gaussian_noise_std_0_01", "gaussian_noise_std_0_02",
)
HF_ARM = "hf_tail_rademacher_v1_rankgate_v2"
LF_ARM = "lf_shell_balanced_blocks_v2_blocknorm_median_v1"
RECORD_ARMS = (HF_ARM, f"primary_null__{HF_ARM}", LF_ARM, f"primary_null__{LF_ARM}")


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
    config_digest: str
    model_id: str
    units: tuple[FrequencyResponseUnit, ...]


def _canonical_digest(config: dict[str, Any], roster: list[dict[str, Any]]) -> str:
    payload = json.dumps({"config": config, "roster": roster}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_plan(config_path: Path, roster_path: Path) -> FrequencyResponsePlan:
    """Load the fixed plan before runtime or ordinary-image observation begins."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    roster = [json.loads(line) for line in roster_path.read_text(encoding="utf-8").splitlines() if line]
    if config.get("evidence_contract") != EVIDENCE_CONTRACT:
        raise ValueError("frequency-response evidence contract differs")
    if tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError("frequency-response condition order differs")
    if tuple(config.get("record_arms_in_exact_condition_order", ())) != RECORD_ARMS:
        raise ValueError("frequency-response detector arm order differs")
    access = config.get("detection_access", {})
    if access.get("allowed_inputs") != ["image", "detection_key", "frozen_public_assets"]:
        raise ValueError("blind detection inputs differ")
    forbidden = set(access.get("forbidden_inputs", ()))
    if not {"original_image", "embed_record", "private_latent", "embed_side_route"}.issubset(forbidden):
        raise ValueError("private-state exclusion differs")
    if config.get("budget", {}).get("actual_callback_dtype_relative_l2_per_method_max") != 0.012:
        raise ValueError("actual-callback-dtype budget differs")
    if config["budget"].get("independent_full_budget") is not True or config["budget"].get("never_coinject_or_fuse") is not True:
        raise ValueError("method independence differs")
    if config.get("keying", {}).get("wrong_key_count") != 16:
        raise ValueError("wrong-key count differs")
    if config.get("execution", {}).get("fixed_units") != 8 or config["execution"].get("records_per_unit") != 40 or config["execution"].get("fixed_records") != 320:
        raise ValueError("fixed 8-unit 320-record denominator differs")
    if config["execution"].get("failures_remain_in_denominator") is not True or config["execution"].get("replacement_units_allowed") is not False or config["execution"].get("complete_rc") != 0:
        raise ValueError("failure or completion rule differs")
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
        protocol_id=config["protocol_id"], config_digest=_canonical_digest(config, roster),
        model_id=config["generation_runtime"]["model_id"], units=units,
    )


def expected_pairs() -> tuple[tuple[str, str], ...]:
    return tuple((condition, arm) for condition in CONDITIONS for arm in RECORD_ARMS)
