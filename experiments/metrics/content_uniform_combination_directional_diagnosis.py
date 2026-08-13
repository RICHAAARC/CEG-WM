"""Frozen statistics for disabled-routing LF/HF combination diagnosis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from struct import pack, unpack
from typing import Sequence

from experiments.protocol.content_uniform_combination_directional_diagnosis import (
    ATTRIBUTION_MARGIN_FLOOR,
    ATTRIBUTION_SUCCESS_COUNT_REQUIREMENT,
    CLAIM_BOUNDARY,
    COMBINATION_FUNCTIONS,
    COMBINATION_WEIGHTS,
    DIRECTIONAL_IMPROVEMENT_COUNT_REQUIREMENT,
    DIRECTIONAL_PROBE_CLUSTER_COUNT,
    IDENTITY_SUCCESS_COUNT_MAXIMUM_LOSS,
    MIXING_COEFFICIENTS,
    NEGATIVE_OUTCOME,
    NEGATIVE_RECOMMENDATION,
    PASSING_OUTCOME,
    PASSING_RECOMMENDATION,
    REFERENCE_FIT_CLUSTER_COUNT,
    canonical_digest,
)


class ContentUniformCombinationDirectionalMetricError(ValueError):
    """Combination diagnosis observations are incomplete or inconsistent."""


class ContentCombinationArmRoleInvalidError(
    ContentUniformCombinationDirectionalMetricError
):
    """The arm role or its embedding coefficient is invalid."""


class ContentCombinationArmMeasurementNonfiniteError(
    ContentUniformCombinationDirectionalMetricError
):
    """An arm budget measurement is nonfinite."""


class ContentCombinationArmRgbQualityBudgetExceededError(
    ContentUniformCombinationDirectionalMetricError
):
    """An arm's RGB quality measurement exceeds the canonical content budget."""

    def __init__(self, arm_id: str) -> None:
        super().__init__("arm RGB quality exceeds the canonical content budget")
        self.arm_id = arm_id


class ContentCombinationArmRealizedContentBudgetExceededError(
    ContentUniformCombinationDirectionalMetricError
):
    """An arm's realized content measurement exceeds the canonical budget."""

    def __init__(self, arm_id: str) -> None:
        super().__init__("arm realized content measurement exceeds the canonical budget")
        self.arm_id = arm_id


class ContentCombinationArmMaterializationRejectedError(
    ContentUniformCombinationDirectionalMetricError
):
    """An arm materialization status is not accepted."""


class ContentCombinationArmImageDigestInvalidError(
    ContentUniformCombinationDirectionalMetricError
):
    """An arm image digest is invalid."""


class ContentCombinationArmObservationIdentityDriftError(
    ContentUniformCombinationDirectionalMetricError
):
    """An arm observation identity does not match its payload."""


def _finite(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ContentUniformCombinationDirectionalMetricError(f"{role} must be finite")
    return float(value)


def _digest(value: object, role: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ContentUniformCombinationDirectionalMetricError(f"{role} must be SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class ContentCombinationReferenceMeasurement:
    cluster_ordinal: int
    fold_index: int
    hf_score: float
    lf_score: float
    hf_detector_identity: str
    lf_detector_identity: str
    whitening_asset_digest: str
    observation_digest: str
    measurement_identity: str

    def validate(self) -> None:
        if (
            type(self.cluster_ordinal) is not int
            or not 0 <= self.cluster_ordinal < REFERENCE_FIT_CLUSTER_COUNT
            or self.fold_index != self.cluster_ordinal % 4
        ):
            raise ContentUniformCombinationDirectionalMetricError("reference identity drifted")
        _finite(self.hf_score, "HF reference score")
        _finite(self.lf_score, "LF reference score")
        for role in ("hf_detector_identity", "lf_detector_identity", "whitening_asset_digest", "observation_digest"):
            _digest(getattr(self, role), role)
        expected = canonical_digest({
            "cluster_ordinal": self.cluster_ordinal,
            "fold_index": self.fold_index,
            "hf_score": self.hf_score,
            "lf_score": self.lf_score,
            "hf_detector_identity": self.hf_detector_identity,
            "lf_detector_identity": self.lf_detector_identity,
            "whitening_asset_digest": self.whitening_asset_digest,
            "observation_digest": self.observation_digest,
        })
        if self.measurement_identity != expected:
            raise ContentUniformCombinationDirectionalMetricError("reference measurement identity drifted")


def create_content_combination_reference_measurement(**values: object) -> ContentCombinationReferenceMeasurement:
    payload = dict(values)
    payload["measurement_identity"] = canonical_digest(payload)
    measurement = ContentCombinationReferenceMeasurement(**payload)
    measurement.validate()
    return measurement


@dataclass(frozen=True, slots=True)
class ContentCombinationFoldReference:
    probe_fold_index: int
    source_cluster_ordinals: tuple[int, ...]
    hf_scores: tuple[float, ...]
    lf_scores: tuple[float, ...]
    hf_detector_identity: str
    lf_detector_identity: str
    whitening_asset_digest: str
    reference_identity: str

    def validate(self) -> None:
        if (
            type(self.probe_fold_index) is not int
            or not 0 <= self.probe_fold_index < 4
            or len(self.source_cluster_ordinals) != 24
            or len(set(self.source_cluster_ordinals)) != 24
            or any(item % 4 == self.probe_fold_index for item in self.source_cluster_ordinals)
            or len(self.hf_scores) != 24
            or len(self.lf_scores) != 24
        ):
            raise ContentUniformCombinationDirectionalMetricError("cross-fit reference drifted")
        for value in (*self.hf_scores, *self.lf_scores):
            _finite(value, "cross-fit score")
        for role in ("hf_detector_identity", "lf_detector_identity", "whitening_asset_digest"):
            _digest(getattr(self, role), role)
        payload = asdict(self)
        identity = payload.pop("reference_identity")
        if identity != canonical_digest(payload):
            raise ContentUniformCombinationDirectionalMetricError("cross-fit identity drifted")


def fit_content_combination_fold_reference(
    measurements: Sequence[ContentCombinationReferenceMeasurement], *, probe_fold_index: int
) -> ContentCombinationFoldReference:
    items = tuple(measurements)
    if len(items) != 32 or any(type(item) is not ContentCombinationReferenceMeasurement for item in items):
        raise ContentUniformCombinationDirectionalMetricError("all 32 reference measurements are required")
    for item in items:
        item.validate()
    if tuple(sorted(item.cluster_ordinal for item in items)) != tuple(range(32)):
        raise ContentUniformCombinationDirectionalMetricError("reference roster drifted")
    selected = tuple(item for item in sorted(items, key=lambda value: value.cluster_ordinal) if item.fold_index != probe_fold_index)
    if len(selected) != 24:
        raise ContentUniformCombinationDirectionalMetricError("cross-fit count drifted")
    identity_sets = (
        {item.hf_detector_identity for item in selected},
        {item.lf_detector_identity for item in selected},
        {item.whitening_asset_digest for item in selected},
    )
    if any(len(values) != 1 for values in identity_sets):
        raise ContentUniformCombinationDirectionalMetricError("reference detector identity drifted")
    payload = {
        "probe_fold_index": probe_fold_index,
        "source_cluster_ordinals": tuple(item.cluster_ordinal for item in selected),
        "hf_scores": tuple(item.hf_score for item in selected),
        "lf_scores": tuple(item.lf_score for item in selected),
        "hf_detector_identity": selected[0].hf_detector_identity,
        "lf_detector_identity": selected[0].lf_detector_identity,
        "whitening_asset_digest": selected[0].whitening_asset_digest,
    }
    reference = ContentCombinationFoldReference(**payload, reference_identity=canonical_digest(payload))
    reference.validate()
    return reference


_CONTROL_ROLES = ("registered", "paired_clean_primary_null", "wrong_key_control")
_ARM_IDENTITIES = ("hf_only", "lf_only", "uniform_combined_quarter", "uniform_combined_half", "uniform_combined_three_quarters")
_BINARY32_CONTENT_LIMIT = unpack(">f", pack(">f", 3.0 / 250.0))[0]


@dataclass(frozen=True, slots=True)
class ContentCombinationScoreRow:
    arm_id: str
    embedding_coefficient: float | None
    control_role: str
    wrong_key_index: int | None
    key_role: str
    combination_function: str
    detector_weight: float | None
    hf_raw_score: float
    lf_raw_score: float | None
    hf_standardized_score: float
    lf_standardized_score: float | None
    content_score: float
    content_detector_identity: str
    content_config_digest: str
    hf_detector_identity: str
    lf_detector_identity: str | None
    whitening_asset_digest: str | None
    input_image_digest: str
    hf_observation_digest: str
    lf_observation_digest: str | None
    hf_template_digest: str
    lf_template_digest: str | None
    root_key_public_digest: str
    row_identity: str

    def validate(self) -> None:
        if self.arm_id not in _ARM_IDENTITIES or self.control_role not in _CONTROL_ROLES:
            raise ContentUniformCombinationDirectionalMetricError("score row role drifted")
        expected_coefficient = {
            "hf_only": None, "lf_only": None, "uniform_combined_quarter": 0.25,
            "uniform_combined_half": 0.50, "uniform_combined_three_quarters": 0.75,
        }[self.arm_id]
        if self.embedding_coefficient != expected_coefficient:
            raise ContentUniformCombinationDirectionalMetricError("embedding coefficient drifted")
        if self.control_role == "wrong_key_control":
            if self.key_role != "wrong" or type(self.wrong_key_index) is not int or not 0 <= self.wrong_key_index < 4:
                raise ContentUniformCombinationDirectionalMetricError("wrong-key row drifted")
        elif self.key_role != "registered" or self.wrong_key_index is not None:
            raise ContentUniformCombinationDirectionalMetricError("registered row drifted")
        if self.combination_function not in COMBINATION_FUNCTIONS:
            raise ContentUniformCombinationDirectionalMetricError("combination function drifted")
        if self.combination_function == "weighted_hf_lf_standardized_score":
            if self.detector_weight not in COMBINATION_WEIGHTS:
                raise ContentUniformCombinationDirectionalMetricError("detector weight drifted")
        elif self.detector_weight is not None:
            raise ContentUniformCombinationDirectionalMetricError("nonweighted row carries weight")
        _finite(self.hf_raw_score, "HF raw score")
        _finite(self.hf_standardized_score, "HF standardized score")
        _finite(self.content_score, "content score")
        consumes_lf = self.combination_function != "hf_only_standardized_score"
        if consumes_lf:
            _finite(self.lf_raw_score, "LF raw score")
            _finite(self.lf_standardized_score, "LF standardized score")
            for role in ("lf_detector_identity", "whitening_asset_digest", "lf_observation_digest", "lf_template_digest"):
                _digest(getattr(self, role), role)
        elif any(getattr(self, role) is not None for role in ("lf_raw_score", "lf_standardized_score", "lf_detector_identity", "whitening_asset_digest", "lf_observation_digest", "lf_template_digest")):
            raise ContentUniformCombinationDirectionalMetricError("HF-only row consumed LF")
        for role in ("content_detector_identity", "content_config_digest", "hf_detector_identity", "input_image_digest", "hf_observation_digest", "hf_template_digest", "root_key_public_digest"):
            _digest(getattr(self, role), role)
        payload = asdict(self)
        identity = payload.pop("row_identity")
        if identity != canonical_digest(payload):
            raise ContentUniformCombinationDirectionalMetricError("score row identity drifted")


def create_content_combination_score_row(**values: object) -> ContentCombinationScoreRow:
    payload = dict(values)
    payload["row_identity"] = canonical_digest(payload)
    row = ContentCombinationScoreRow(**payload)
    row.validate()
    return row


@dataclass(frozen=True, slots=True)
class ContentCombinationArmObservation:
    arm_id: str
    embedding_coefficient: float | None
    clean_to_watermarked_rgb_relative_l2: float
    realized_relative_l2: float
    materialization_integrity_status: str
    materialization_budget_status: str
    image_digest: str
    arm_identity: str

    def _validate_payload(self) -> None:
        expected_coefficient = {
            "hf_only": None,
            "lf_only": None,
            "uniform_combined_quarter": 0.25,
            "uniform_combined_half": 0.50,
            "uniform_combined_three_quarters": 0.75,
        }.get(self.arm_id, object())
        if self.arm_id not in _ARM_IDENTITIES or self.embedding_coefficient != expected_coefficient:
            raise ContentCombinationArmRoleInvalidError(
                "content combination arm role is invalid"
            )
        try:
            _finite(self.clean_to_watermarked_rgb_relative_l2, "RGB quality")
            _finite(self.realized_relative_l2, "realized relative L2")
        except ContentUniformCombinationDirectionalMetricError as exc:
            if type(exc) is not ContentUniformCombinationDirectionalMetricError:
                raise
            raise ContentCombinationArmMeasurementNonfiniteError(
                "content combination arm measurement is nonfinite"
            ) from exc
        if self.clean_to_watermarked_rgb_relative_l2 > _BINARY32_CONTENT_LIMIT:
            raise ContentCombinationArmRgbQualityBudgetExceededError(
                self.arm_id
            )
        if self.realized_relative_l2 > _BINARY32_CONTENT_LIMIT:
            raise ContentCombinationArmRealizedContentBudgetExceededError(
                self.arm_id
            )
        if self.materialization_integrity_status != "passed" or self.materialization_budget_status != "accepted":
            raise ContentCombinationArmMaterializationRejectedError(
                "arm materialization was not accepted"
            )
        try:
            _digest(self.image_digest, "arm image digest")
        except ContentUniformCombinationDirectionalMetricError as exc:
            if type(exc) is not ContentUniformCombinationDirectionalMetricError:
                raise
            raise ContentCombinationArmImageDigestInvalidError(
                "content combination arm image digest is invalid"
            ) from exc
    def _validate_identity(self) -> None:
        payload = asdict(self); identity = payload.pop("arm_identity")
        if identity != canonical_digest(payload):
            raise ContentCombinationArmObservationIdentityDriftError(
                "arm observation identity drifted"
            )

    def validate(self) -> None:
        self._validate_payload()
        self._validate_identity()


def create_content_combination_arm_observation(**values: object) -> ContentCombinationArmObservation:
    payload = dict(values)
    provisional = ContentCombinationArmObservation(
        **payload,
        arm_identity="0" * 64,
    )
    provisional._validate_payload()
    payload["arm_identity"] = canonical_digest(payload)
    arm = ContentCombinationArmObservation(**payload)
    arm._validate_identity()
    return arm


@dataclass(frozen=True, slots=True)
class ContentUniformCombinationDirectionalObservation:
    cluster_ordinal: int
    fold_index: int
    fold_reference_identity: str
    whitening_asset_digest: str
    score_rows: tuple[ContentCombinationScoreRow, ...]
    arm_observations: tuple[ContentCombinationArmObservation, ...]
    failure_class: str | None
    observation_identity: str

    def validate(self) -> None:
        if self.cluster_ordinal not in range(8) or self.fold_index != self.cluster_ordinal % 4:
            raise ContentUniformCombinationDirectionalMetricError("probe identity drifted")
        _digest(self.fold_reference_identity, "fold reference")
        _digest(self.whitening_asset_digest, "whitening asset")
        if self.failure_class is not None:
            if self.failure_class not in {"implementation_failure", "resource_failure"} or self.score_rows or self.arm_observations:
                raise ContentUniformCombinationDirectionalMetricError("failed probe payload drifted")
        else:
            if len(self.arm_observations) != 5 or tuple(item.arm_id for item in self.arm_observations) != _ARM_IDENTITIES:
                raise ContentUniformCombinationDirectionalMetricError("five watermarked arms are required")
            for arm in self.arm_observations: arm.validate()
            expected = []
            for arm_id in _ARM_IDENTITIES:
                for role in ("registered", "paired_clean_primary_null"):
                    for function in COMBINATION_FUNCTIONS:
                        weights = COMBINATION_WEIGHTS if function == "weighted_hf_lf_standardized_score" else (None,)
                        expected.extend((arm_id, role, None, function, weight) for weight in weights)
                for wrong_index in range(4):
                    for function in COMBINATION_FUNCTIONS:
                        weights = COMBINATION_WEIGHTS if function == "weighted_hf_lf_standardized_score" else (None,)
                        expected.extend((arm_id, "wrong_key_control", wrong_index, function, weight) for weight in weights)
            observed = []
            for row in self.score_rows:
                row.validate(); observed.append((row.arm_id, row.control_role, row.wrong_key_index, row.combination_function, row.detector_weight))
                if row.whitening_asset_digest not in {None, self.whitening_asset_digest}:
                    raise ContentUniformCombinationDirectionalMetricError("row whitening asset drifted")
            if tuple(observed) != tuple(expected):
                raise ContentUniformCombinationDirectionalMetricError("complete score row roster drifted")
            clean_images = {
                row.input_image_digest for row in self.score_rows
                if row.control_role == "paired_clean_primary_null"
            }
            if len(clean_images) != 1:
                raise ContentUniformCombinationDirectionalMetricError(
                    "paired clean image identity drifted"
                )
            registered_hf_template: str | None = None
            registered_lf_template: str | None = None
            wrong_hf_templates: dict[int, str] = {}
            wrong_lf_templates: dict[int, str] = {}
            for arm_id in _ARM_IDENTITIES:
                candidate_rows = [
                    row for row in self.score_rows
                    if row.arm_id == arm_id
                    and row.control_role in {"registered", "wrong_key_control"}
                ]
                if len({row.input_image_digest for row in candidate_rows}) != 1:
                    raise ContentUniformCombinationDirectionalMetricError(
                        "registered and wrong-key controls must share one image"
                    )
                for role, wrong_index in (
                    ("registered", None),
                    ("paired_clean_primary_null", None),
                    *(("wrong_key_control", index) for index in range(4)),
                ):
                    control_rows = [
                        row for row in self.score_rows
                        if row.arm_id == arm_id
                        and row.control_role == role
                        and row.wrong_key_index == wrong_index
                    ]
                    for identity_role in (
                        "input_image_digest",
                        "hf_observation_digest",
                        "hf_template_digest",
                        "root_key_public_digest",
                    ):
                        if len({getattr(row, identity_role) for row in control_rows}) != 1:
                            raise ContentUniformCombinationDirectionalMetricError(
                                "control identity drifted across diagnostic functions"
                            )
                    lf_rows = [row for row in control_rows if row.lf_detector_identity is not None]
                    if any(
                        row.lf_observation_digest != row.hf_observation_digest
                        for row in lf_rows
                    ):
                        raise ContentUniformCombinationDirectionalMetricError(
                            "HF/LF public observation identity drifted"
                        )
                    for identity_role in (
                        "lf_observation_digest",
                        "lf_template_digest",
                        "whitening_asset_digest",
                    ):
                        if len({getattr(row, identity_role) for row in lf_rows}) != 1:
                            raise ContentUniformCombinationDirectionalMetricError(
                                "LF control identity drifted across diagnostic functions"
                            )
                    hf_template = control_rows[0].hf_template_digest
                    lf_template = lf_rows[0].lf_template_digest
                    if role in {"registered", "paired_clean_primary_null"}:
                        registered_hf_template = registered_hf_template or hf_template
                        registered_lf_template = registered_lf_template or lf_template
                        if hf_template != registered_hf_template or lf_template != registered_lf_template:
                            raise ContentUniformCombinationDirectionalMetricError(
                                "registered template drifted across arms or clean control"
                            )
                    else:
                        assert wrong_index is not None
                        wrong_hf_templates.setdefault(wrong_index, hf_template)
                        wrong_lf_templates.setdefault(wrong_index, lf_template)
                        if (
                            hf_template != wrong_hf_templates[wrong_index]
                            or lf_template != wrong_lf_templates[wrong_index]
                        ):
                            raise ContentUniformCombinationDirectionalMetricError(
                                "wrong-key template drifted across arms"
                            )
        payload=asdict(self); identity=payload.pop("observation_identity")
        if identity != canonical_digest(payload):
            raise ContentUniformCombinationDirectionalMetricError("probe observation identity drifted")


def create_content_uniform_combination_directional_observation(**values: object) -> ContentUniformCombinationDirectionalObservation:
    payload=dict(values)
    identity_payload = {
        **payload,
        "score_rows": tuple(
            item if type(item) is dict else asdict(item)
            for item in payload["score_rows"]
        ),
        "arm_observations": tuple(
            item if type(item) is dict else asdict(item)
            for item in payload["arm_observations"]
        ),
    }
    payload["observation_identity"]=canonical_digest(identity_payload)
    observation=ContentUniformCombinationDirectionalObservation(**payload); observation.validate(); return observation


@dataclass(frozen=True, slots=True)
class ContentUniformCombinationDirectionalAggregate:
    scientific_cluster_count: int
    successful_cluster_count: int
    failed_cluster_count: int
    implementation_failure_count: int
    resource_failure_count: int
    identity_violation_count: int
    integrity_violation_count: int
    nonfinite_violation_count: int
    budget_violation_count: int
    qualifying_candidate_count: int
    candidate_statistics: tuple[dict[str, object], ...]
    outcome: str
    candidate_recommendation: str
    allow_request_for_content_combination_candidate_selection: bool
    claim_boundary: str
    aggregate_identity: str

    def validate(self) -> None:
        if self.scientific_cluster_count != 8 or self.successful_cluster_count + self.failed_cluster_count != 8 or self.failed_cluster_count != self.implementation_failure_count + self.resource_failure_count:
            raise ContentUniformCombinationDirectionalMetricError("fixed denominator drifted")
        for item in self.candidate_statistics:
            if set(item) != {"embedding_coefficient", "combination_function", "detector_weight", "registered_primary_null_success_count", "registered_maximum_wrong_success_count", "directional_improvement_count", "identity_success_count", "baseline_identity_success_count"}:
                raise ContentUniformCombinationDirectionalMetricError("candidate statistic schema drifted")
        passing = self.qualifying_candidate_count > 0 and self.failed_cluster_count == 0 and self.identity_violation_count == 0 and self.integrity_violation_count == 0 and self.nonfinite_violation_count == 0 and self.budget_violation_count == 0
        if (self.outcome == PASSING_OUTCOME) is not passing or (self.candidate_recommendation == PASSING_RECOMMENDATION) is not passing or self.allow_request_for_content_combination_candidate_selection is not passing or self.claim_boundary != CLAIM_BOUNDARY:
            raise ContentUniformCombinationDirectionalMetricError("aggregate decision drifted")
        payload=asdict(self); identity=payload.pop("aggregate_identity")
        if identity != canonical_digest(payload):
            raise ContentUniformCombinationDirectionalMetricError("aggregate identity drifted")


def _row(observation: ContentUniformCombinationDirectionalObservation, *, arm_id: str, role: str, wrong_index: int | None, function: str, weight: float | None) -> ContentCombinationScoreRow:
    matches=[item for item in observation.score_rows if (item.arm_id,item.control_role,item.wrong_key_index,item.combination_function,item.detector_weight)==(arm_id,role,wrong_index,function,weight)]
    if len(matches)!=1: raise ContentUniformCombinationDirectionalMetricError("score row lookup drifted")
    return matches[0]


def aggregate_content_uniform_combination_directional_diagnosis(
    observations: Sequence[ContentUniformCombinationDirectionalObservation], *,
    identity_violation_count: int = 0, integrity_violation_count: int = 0,
    nonfinite_violation_count: int = 0, budget_violation_count: int = 0,
) -> ContentUniformCombinationDirectionalAggregate:
    items=tuple(observations)
    if len(items)!=8 or tuple(sorted(item.cluster_ordinal for item in items))!=tuple(range(8)):
        raise ContentUniformCombinationDirectionalMetricError("all eight fixed probes are required")
    for item in items: item.validate()
    implementation=sum(item.failure_class=="implementation_failure" for item in items)
    resource=sum(item.failure_class=="resource_failure" for item in items)
    successful=8-implementation-resource
    statistics=[]
    for embedding_coefficient, arm_id in zip(MIXING_COEFFICIENTS, _ARM_IDENTITIES[2:], strict=True):
        baseline_identity=0
        for observation in items:
            if observation.failure_class is not None: continue
            registered=_row(observation,arm_id=arm_id,role="registered",wrong_index=None,function="hf_only_standardized_score",weight=None).content_score
            primary=_row(observation,arm_id=arm_id,role="paired_clean_primary_null",wrong_index=None,function="hf_only_standardized_score",weight=None).content_score
            wrong=max(_row(observation,arm_id=arm_id,role="wrong_key_control",wrong_index=index,function="hf_only_standardized_score",weight=None).content_score for index in range(4))
            baseline_identity += int(registered-primary>ATTRIBUTION_MARGIN_FLOOR and registered-wrong>ATTRIBUTION_MARGIN_FLOOR)
        candidates=(("maximum_hf_lf_standardized_score",None),*(("weighted_hf_lf_standardized_score",weight) for weight in COMBINATION_WEIGHTS))
        for function,weight in candidates:
            primary_count=wrong_count=improvement_count=identity_count=0
            for observation in items:
                if observation.failure_class is not None: continue
                registered=_row(observation,arm_id=arm_id,role="registered",wrong_index=None,function=function,weight=weight).content_score
                primary=_row(observation,arm_id=arm_id,role="paired_clean_primary_null",wrong_index=None,function=function,weight=weight).content_score
                wrong=max(_row(observation,arm_id=arm_id,role="wrong_key_control",wrong_index=index,function=function,weight=weight).content_score for index in range(4))
                base_registered=_row(observation,arm_id=arm_id,role="registered",wrong_index=None,function="hf_only_standardized_score",weight=None).content_score
                base_primary=_row(observation,arm_id=arm_id,role="paired_clean_primary_null",wrong_index=None,function="hf_only_standardized_score",weight=None).content_score
                base_wrong=max(_row(observation,arm_id=arm_id,role="wrong_key_control",wrong_index=index,function="hf_only_standardized_score",weight=None).content_score for index in range(4))
                primary_margin=registered-primary; wrong_margin=registered-wrong
                primary_count+=int(primary_margin>ATTRIBUTION_MARGIN_FLOOR)
                wrong_count+=int(wrong_margin>ATTRIBUTION_MARGIN_FLOOR)
                identity_count+=int(primary_margin>ATTRIBUTION_MARGIN_FLOOR and wrong_margin>ATTRIBUTION_MARGIN_FLOOR)
                improvement_count+=int(min(primary_margin,wrong_margin)>min(base_registered-base_primary,base_registered-base_wrong))
            statistics.append({"embedding_coefficient":embedding_coefficient,"combination_function":function,"detector_weight":weight,"registered_primary_null_success_count":primary_count,"registered_maximum_wrong_success_count":wrong_count,"directional_improvement_count":improvement_count,"identity_success_count":identity_count,"baseline_identity_success_count":baseline_identity})
    qualifying=sum(item["registered_primary_null_success_count"]>=ATTRIBUTION_SUCCESS_COUNT_REQUIREMENT and item["registered_maximum_wrong_success_count"]>=ATTRIBUTION_SUCCESS_COUNT_REQUIREMENT and item["directional_improvement_count"]>DIRECTIONAL_IMPROVEMENT_COUNT_REQUIREMENT and item["identity_success_count"]>=item["baseline_identity_success_count"]-IDENTITY_SUCCESS_COUNT_MAXIMUM_LOSS for item in statistics)
    blocked=implementation or resource or identity_violation_count or integrity_violation_count or nonfinite_violation_count or budget_violation_count
    passing=not blocked and successful==8 and qualifying>0
    payload={"scientific_cluster_count":8,"successful_cluster_count":successful,"failed_cluster_count":implementation+resource,"implementation_failure_count":implementation,"resource_failure_count":resource,"identity_violation_count":identity_violation_count,"integrity_violation_count":integrity_violation_count,"nonfinite_violation_count":nonfinite_violation_count,"budget_violation_count":budget_violation_count,"qualifying_candidate_count":qualifying,"candidate_statistics":tuple(statistics),"outcome":PASSING_OUTCOME if passing else "implementation_blocked" if implementation or identity_violation_count or integrity_violation_count or nonfinite_violation_count else "resource_blocked" if resource else NEGATIVE_OUTCOME,"candidate_recommendation":PASSING_RECOMMENDATION if passing else NEGATIVE_RECOMMENDATION,"allow_request_for_content_combination_candidate_selection":passing,"claim_boundary":CLAIM_BOUNDARY}
    aggregate=ContentUniformCombinationDirectionalAggregate(**payload,aggregate_identity=canonical_digest(payload)); aggregate.validate(); return aggregate


__all__=["ContentCombinationArmImageDigestInvalidError","ContentCombinationArmMaterializationRejectedError","ContentCombinationArmMeasurementNonfiniteError","ContentCombinationArmObservation","ContentCombinationArmObservationIdentityDriftError","ContentCombinationArmRealizedContentBudgetExceededError","ContentCombinationArmRgbQualityBudgetExceededError","ContentCombinationArmRoleInvalidError","ContentCombinationFoldReference","ContentCombinationReferenceMeasurement","ContentCombinationScoreRow","ContentUniformCombinationDirectionalAggregate","ContentUniformCombinationDirectionalMetricError","ContentUniformCombinationDirectionalObservation","aggregate_content_uniform_combination_directional_diagnosis","create_content_combination_arm_observation","create_content_combination_reference_measurement","create_content_combination_score_row","create_content_uniform_combination_directional_observation","fit_content_combination_fold_reference"]
