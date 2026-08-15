"""Typed observations and fixed-denominator aggregate for the salient pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite, sqrt
import re
from typing import Sequence

import torch

from experiments.protocol.salient_local_lf_mask_write_validation import (
    QUALITY_PIXEL_COUNT,
    QUALITY_SQUARED_CODE_DELTA_LIMIT,
    REQUIRED_QUALITY_SUCCESS_COUNT,
    MINIMUM_MECHANISM_SUCCESS_COUNT,
    SCIENTIFIC_UNIT_COUNT,
)


class SalientLocalLfMaskWriteMetricError(ValueError):
    """A typed public observation or aggregate violated the frozen metric."""


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _digest(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PublicRgb8QualityObservation:
    clean_image_digest: str
    marked_image_digest: str
    pixel_count: int
    squared_code_delta_sum: int
    normalized_mean_squared_error: float
    root_mean_squared_code_delta: float
    relative_l2_auxiliary: float
    quality_pass: bool
    observation_identity: str

    def validate(self) -> None:
        if (self.pixel_count != QUALITY_PIXEL_COUNT
                or type(self.squared_code_delta_sum) is not int
                or self.squared_code_delta_sum < 0
                or any(type(value) is not str or len(value) != 64 for value in (
                    self.clean_image_digest, self.marked_image_digest, self.observation_identity,
                ))
                or not all(isfinite(value) and value >= 0.0 for value in (
                    self.normalized_mean_squared_error,
                    self.root_mean_squared_code_delta,
                    self.relative_l2_auxiliary,
                ))):
            raise SalientLocalLfMaskWriteMetricError("public RGB8 quality observation is invalid")
        if self.quality_pass is not (self.squared_code_delta_sum <= QUALITY_SQUARED_CODE_DELTA_LIMIT):
            raise SalientLocalLfMaskWriteMetricError("public RGB8 quality decision drifted")
        expected = _digest({key: value for key, value in asdict(self).items() if key != "observation_identity"})
        if self.observation_identity != expected:
            raise SalientLocalLfMaskWriteMetricError("public RGB8 quality identity drifted")


def observe_public_rgb8_quality(
    clean: torch.Tensor,
    marked: torch.Tensor,
    *,
    clean_image_digest: str,
    marked_image_digest: str,
) -> PublicRgb8QualityObservation:
    if (type(clean) is not torch.Tensor or type(marked) is not torch.Tensor
            or clean.dtype != torch.uint8 or marked.dtype != torch.uint8
            or tuple(clean.shape) != (1, 3, 512, 512) or tuple(marked.shape) != (1, 3, 512, 512)):
        raise SalientLocalLfMaskWriteMetricError("quality images must be public RGB8 [1,3,512,512]")
    if (_DIGEST.fullmatch(clean_image_digest) is None
            or _DIGEST.fullmatch(marked_image_digest) is None):
        raise SalientLocalLfMaskWriteMetricError("quality image digest identity is invalid")
    left = clean.detach().to(device="cpu").contiguous()
    right = marked.detach().to(device="cpu").contiguous()
    difference = right.to(torch.int64) - left.to(torch.int64)
    squared_sum = int(torch.sum(difference * difference, dtype=torch.int64).item())
    if left.numel() != QUALITY_PIXEL_COUNT:
        raise SalientLocalLfMaskWriteMetricError("quality pixel count drifted")
    clean_norm = float(torch.linalg.vector_norm(left.to(torch.float64)).item())
    if clean_norm <= 0.0 or not isfinite(clean_norm):
        raise SalientLocalLfMaskWriteMetricError("clean RGB8 norm is invalid")
    normalized_mse = squared_sum / float(QUALITY_PIXEL_COUNT * 255 * 255)
    rms = sqrt(squared_sum / float(QUALITY_PIXEL_COUNT))
    relative = sqrt(float(squared_sum)) / clean_norm
    payload = {
        "clean_image_digest": clean_image_digest,
        "marked_image_digest": marked_image_digest,
        "pixel_count": QUALITY_PIXEL_COUNT,
        "squared_code_delta_sum": squared_sum,
        "normalized_mean_squared_error": normalized_mse,
        "root_mean_squared_code_delta": rms,
        "relative_l2_auxiliary": relative,
        "quality_pass": squared_sum <= QUALITY_SQUARED_CODE_DELTA_LIMIT,
    }
    result = PublicRgb8QualityObservation(**payload, observation_identity=_digest(payload))
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class SalientLocalLfMaskWriteObservation:
    cluster_ordinal: int
    source_cluster_id: str
    clean_image_digest: str
    marked_image_digest: str
    embed_saliency_observation_identity: str
    detect_saliency_observation_identity: str
    embed_mask_identity: str
    detect_mask_identity: str
    embed_mask_coverage: int
    detect_mask_coverage: int
    mask_intersection_over_union: float
    nominal_masked_lf_outside_bitwise_zero: bool
    nominal_masked_lf_inside_nonzero: bool
    nominal_masked_lf_consumed_by_materialization: bool
    accepted_materialization_replay_identity: str
    realized_relative_l2: float
    actual_dtype_budget_pass: bool
    identity_pass: bool
    integrity_pass: bool
    quality: PublicRgb8QualityObservation
    observation_identity: str

    @property
    def mechanism_pass(self) -> bool:
        return bool(
            64 <= self.embed_mask_coverage <= 3072
            and 64 <= self.detect_mask_coverage <= 3072
            and self.mask_intersection_over_union >= 0.5
            and self.nominal_masked_lf_outside_bitwise_zero
            and self.nominal_masked_lf_inside_nonzero
            and self.nominal_masked_lf_consumed_by_materialization
            and self.actual_dtype_budget_pass
            and self.identity_pass
            and self.integrity_pass
        )

    def validate(self) -> None:
        if (type(self.cluster_ordinal) is not int or not 0 <= self.cluster_ordinal < SCIENTIFIC_UNIT_COUNT
                or type(self.embed_mask_coverage) is not int or type(self.detect_mask_coverage) is not int
                or not isfinite(self.mask_intersection_over_union)
                or not 0.0 <= self.mask_intersection_over_union <= 1.0
                or not isfinite(self.realized_relative_l2) or self.realized_relative_l2 < 0.0
                or any(type(value) is not bool for value in (
                    self.nominal_masked_lf_outside_bitwise_zero,
                    self.nominal_masked_lf_inside_nonzero,
                    self.nominal_masked_lf_consumed_by_materialization,
                    self.actual_dtype_budget_pass, self.identity_pass, self.integrity_pass,
                ))):
            raise SalientLocalLfMaskWriteMetricError("mask/write observation is invalid")
        for value in (
            self.source_cluster_id, self.clean_image_digest, self.marked_image_digest,
            self.embed_saliency_observation_identity, self.detect_saliency_observation_identity,
            self.embed_mask_identity, self.detect_mask_identity,
            self.accepted_materialization_replay_identity, self.observation_identity,
        ):
            if type(value) is not str or len(value) != 64:
                raise SalientLocalLfMaskWriteMetricError("mask/write observation identity is invalid")
        self.quality.validate()
        if self.quality.clean_image_digest != self.clean_image_digest or self.quality.marked_image_digest != self.marked_image_digest:
            raise SalientLocalLfMaskWriteMetricError("quality image identity drifted")
        expected = _digest({
            **{key: value for key, value in asdict(self).items() if key not in {"quality", "observation_identity"}},
            "quality": asdict(self.quality),
        })
        if self.observation_identity != expected:
            raise SalientLocalLfMaskWriteMetricError("mask/write observation identity drifted")


def create_mask_write_observation(**values: object) -> SalientLocalLfMaskWriteObservation:
    payload = dict(values)
    quality = payload.get("quality")
    if type(quality) is not PublicRgb8QualityObservation:
        raise SalientLocalLfMaskWriteMetricError("typed quality observation is required")
    identity_payload = {**payload, "quality": asdict(quality)}
    result = SalientLocalLfMaskWriteObservation(**payload, observation_identity=_digest(identity_payload))
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class SalientLocalLfTerminalFailure:
    cluster_ordinal: int
    failure_class: str
    failure_reason: str


@dataclass(frozen=True, slots=True)
class SalientLocalLfMaskWriteAggregate:
    scientific_denominator: int
    successful_observation_count: int
    quality_success_count: int
    mechanism_success_count: int
    identity_failure_count: int
    integrity_failure_count: int
    implementation_failure_count: int
    resource_failure_count: int
    environment_failure_count: int
    module_outcome: str
    candidate_recommendation: str
    allow_request_for_independent_masked_lf_null_fit: bool
    aggregate_identity: str


def aggregate_salient_local_lf_mask_write_validation(
    observations: Sequence[SalientLocalLfMaskWriteObservation],
    failures: Sequence[SalientLocalLfTerminalFailure],
) -> SalientLocalLfMaskWriteAggregate:
    observed = tuple(observations)
    failed = tuple(failures)
    for item in observed: item.validate()
    if any(type(item) is not SalientLocalLfTerminalFailure for item in failed):
        raise SalientLocalLfMaskWriteMetricError("terminal failure exact type is required")
    ordinals = tuple(item.cluster_ordinal for item in observed) + tuple(item.cluster_ordinal for item in failed)
    if len(ordinals) != SCIENTIFIC_UNIT_COUNT or set(ordinals) != set(range(SCIENTIFIC_UNIT_COUNT)):
        raise SalientLocalLfMaskWriteMetricError("fixed eight denominator is incomplete")
    counts = {
        name: sum(item.failure_class == name for item in failed)
        for name in ("identity_failure", "integrity_failure", "implementation_failure", "resource_failure", "environment_failure")
    }
    quality_count = sum(item.quality.quality_pass for item in observed)
    mechanism_count = sum(item.mechanism_pass for item in observed)
    all_terminal_success = len(observed) == SCIENTIFIC_UNIT_COUNT and not failed
    allow = bool(all_terminal_success and quality_count == REQUIRED_QUALITY_SUCCESS_COUNT
                 and mechanism_count >= MINIMUM_MECHANISM_SUCCESS_COUNT)
    if counts["identity_failure"]:
        outcome, recommendation = "identity_blocked", "candidate_not_recommended"
    elif counts["integrity_failure"]:
        outcome, recommendation = "integrity_blocked", "candidate_not_recommended"
    elif counts["implementation_failure"]:
        outcome, recommendation = "implementation_blocked", "candidate_not_recommended"
    elif counts["resource_failure"]:
        outcome, recommendation = "resource_blocked", "candidate_not_recommended"
    elif counts["environment_failure"]:
        outcome, recommendation = "environment_blocked", "candidate_not_recommended"
    elif allow:
        outcome, recommendation = "mechanism_signal_observed", "candidate_worth_further_validation"
    else:
        outcome, recommendation = "mechanism_signal_not_observed", "candidate_not_recommended"
    payload = {
        "scientific_denominator": SCIENTIFIC_UNIT_COUNT,
        "successful_observation_count": len(observed),
        "quality_success_count": quality_count,
        "mechanism_success_count": mechanism_count,
        "identity_failure_count": counts["identity_failure"],
        "integrity_failure_count": counts["integrity_failure"],
        "implementation_failure_count": counts["implementation_failure"],
        "resource_failure_count": counts["resource_failure"],
        "environment_failure_count": counts["environment_failure"],
        "module_outcome": outcome,
        "candidate_recommendation": recommendation,
        "allow_request_for_independent_masked_lf_null_fit": allow,
    }
    return SalientLocalLfMaskWriteAggregate(**payload, aggregate_identity=_digest(payload))


__all__ = [
    "PublicRgb8QualityObservation", "SalientLocalLfMaskWriteObservation",
    "SalientLocalLfTerminalFailure", "SalientLocalLfMaskWriteAggregate",
    "SalientLocalLfMaskWriteMetricError", "observe_public_rgb8_quality",
    "create_mask_write_observation", "aggregate_salient_local_lf_mask_write_validation",
]
