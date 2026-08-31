"""Method-neutral, evidence-preserving records for baseline observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from cegwm.baselines.registry import baseline_by_id


@dataclass(frozen=True)
class BaselineObservation:
    """One observed detector score; not a scientific result by itself.

    Source and adapter metadata are optional per-record context. A method may
    decide only with its own calibrated threshold provenance; no shared numeric
    threshold is represented by this schema.
    """

    baseline_id: str
    source_exact: str | None
    adapter_exact: str | None
    prompt_id: str
    seed: int
    base_latent_commitment: str
    split: str
    protocol_partition: str
    sample_role: str
    attack_family: str
    attack_condition: str
    continuous_score: float | None
    score_direction: str | None
    threshold_provenance: str | None
    decision: bool | None
    quality: Mapping[str, float]
    runtime_seconds: float | None
    status: str
    failure_reason: str | None
    artifact_digests: Mapping[str, str]
    attack_provenance: Mapping[str, object] | None

    def as_dict(self) -> dict[str, object]:
        """Validate before returning data suitable for an append-only JSONL writer."""

        return asdict(validate_observation(self))


def validate_observation(observation: BaselineObservation) -> BaselineObservation:
    """Fail closed on method, score, partition, and attack-boundary violations."""

    baseline = baseline_by_id(observation.baseline_id)
    if not observation.prompt_id or not observation.base_latent_commitment:
        raise ValueError("prompt_id and base_latent_commitment are required")
    if observation.seed < 0:
        raise ValueError("seed must be non-negative")
    if observation.status not in {"calibration_observed", "confirmation_observed", "observed", "failed", "not_available"}:
        raise ValueError("status is not recognized")
    if observation.sample_role not in {
        "calibration_unwatermarked_negative",
        "confirmation_unwatermarked_negative",
        "evaluation_unwatermarked_negative",
        "evaluation_watermarked",
        "wrong_key_diagnostic",
    }:
        raise ValueError("sample_role is not part of the baseline protocol")
    if not observation.attack_family or not observation.attack_condition:
        raise ValueError("attack_family and attack_condition are required")
    if observation.protocol_partition not in {"threshold_freeze", "clean_confirmation", "evaluation"}:
        raise ValueError("protocol_partition is not recognized")
    if observation.status == "calibration_observed":
        if observation.sample_role != "calibration_unwatermarked_negative" or observation.protocol_partition != "threshold_freeze":
            raise ValueError("calibration records require unwatermarked-negative role")
        if observation.continuous_score is None or observation.decision is not None:
            raise ValueError("calibration records require a score and no decision")
        if observation.threshold_provenance is not None:
            raise ValueError("calibration records cannot claim a frozen threshold")
        if observation.score_direction != baseline.score_direction:
            raise ValueError("score_direction must equal the method-declared direction")
    if observation.status == "confirmation_observed":
        if observation.sample_role != "confirmation_unwatermarked_negative" or observation.protocol_partition != "clean_confirmation":
            raise ValueError("confirmation records require clean-confirmation unwatermarked-negative role")
        if observation.attack_family != "clean" or observation.attack_condition != "clean_no_attack":
            raise ValueError("confirmation records must be clean")
        if observation.continuous_score is None or not isinstance(observation.decision, bool):
            raise ValueError("confirmation records require a score and decision")
    if observation.status in {"observed", "confirmation_observed"}:
        if observation.protocol_partition != "evaluation":
            if observation.status == "observed":
                raise ValueError("observed records require evaluation partition")
        if observation.status == "observed" and observation.sample_role not in {
            "evaluation_watermarked", "evaluation_unwatermarked_negative", "wrong_key_diagnostic",
        }:
            raise ValueError("observed records require an evaluation sample role")
        if not observation.threshold_provenance:
            raise ValueError("observed records require a method-specific threshold identity")
        if observation.continuous_score is None or not isinstance(observation.decision, bool):
            raise ValueError("observed records require a continuous score and decision")
        if baseline.score_direction is None:
            raise ValueError("method detector score direction is unresolved")
        if observation.score_direction != baseline.score_direction:
            raise ValueError("score_direction must equal the method-declared direction")
        expected_prefix = f"{observation.baseline_id}:calibration:"
        if not observation.threshold_provenance.startswith(expected_prefix):
            raise ValueError("threshold provenance must be bound to the baseline")
    elif observation.status in {"failed", "not_available"}:
        if any(value is not None for value in (observation.continuous_score, observation.score_direction,
                                                observation.threshold_provenance, observation.decision)):
            raise ValueError("failed or unavailable records cannot carry detection evidence")
    if observation.status == "failed" and not observation.failure_reason:
        raise ValueError("failed records require failure_reason")
    if not isinstance(observation.artifact_digests, Mapping):
        raise ValueError("artifact_digests must be a mapping when supplied")
    if observation.status == "observed" and observation.attack_condition == "rotation_10_bicubic_reflect_center_crop_v1":
        _validate_rotation_provenance(observation.attack_provenance)
    return observation


def _validate_rotation_provenance(provenance: Mapping[str, object] | None) -> None:
    if provenance is None:
        raise ValueError("rotation observations require attack provenance")
    required = {
        "attack_id", "angle_degrees", "angle_convention", "center_formula_id", "padding_x", "padding_y",
        "bicubic_margin_pixels", "padding_mode_rgb", "padding_mode_mask", "rgb_interpolation", "mask_interpolation",
        "crop_box", "positive_negative_pipeline_identical",
    }
    if not required.issubset(provenance):
        raise ValueError("rotation provenance is incomplete")
    if provenance["attack_id"] != "rotation_10_bicubic_reflect_center_crop_v1":
        raise ValueError("rotation provenance attack_id mismatch")
    if provenance["angle_degrees"] != 10.0 or provenance["bicubic_margin_pixels"] != 2:
        raise ValueError("rotation provenance parameters mismatch")
    if provenance["angle_convention"] != "Pillow visual counter-clockwise positive angle":
        raise ValueError("rotation provenance angle convention mismatch")
    if provenance["center_formula_id"] != "pixel_center_w_minus_1_over_2_v1":
        raise ValueError("rotation provenance center formula mismatch")
    if provenance["padding_mode_rgb"] != "numpy.reflect_edge_not_repeated" or provenance["padding_mode_mask"] != "numpy.constant_zero":
        raise ValueError("rotation provenance padding mode mismatch")
    if provenance["rgb_interpolation"] != "PIL.Image.Resampling.BICUBIC" or provenance["mask_interpolation"] != "PIL.Image.Resampling.NEAREST":
        raise ValueError("rotation provenance interpolation mismatch")
    if provenance["positive_negative_pipeline_identical"] is not True:
        raise ValueError("rotation provenance must bind shared positive-negative pipeline")
    if not isinstance(provenance["padding_x"], int) or not isinstance(provenance["padding_y"], int):
        raise ValueError("rotation provenance padding must be Python integers")
    crop_box = provenance["crop_box"]
    if (
        not isinstance(crop_box, (list, tuple))
        or len(crop_box) != 4
        or any(not isinstance(coordinate, int) for coordinate in crop_box)
    ):
        raise ValueError("rotation provenance crop box is invalid")
