"""Loader and fail-closed validation for the finite Stage-A protocol."""

from __future__ import annotations

import hashlib
import json
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
