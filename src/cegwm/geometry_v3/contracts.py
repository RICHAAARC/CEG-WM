"""Pure contract layer for the Geometry-V3 keyed Q/K canonical anchor route.

This module deliberately contains no model, image, tensor, or writer runtime.
It freezes the information boundaries that a later SD3.5 implementation must
obey before any model-backed work is authorized.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Sequence

from cegwm.shared.keys import normalize_detection_key


METHOD_ID = "geometry_v3_keyed_qk_canonical_anchor"
ANCHOR_DOMAIN_ID = "cegwm/geometry-v3/keyed-qk-canonical-relation/v1"
WRITER_CONTRACT_ID = "cegwm/geometry-v3/qk-attention-writer-declaration/v1"
RECOVERABILITY_CONTRACT_ID = "cegwm/geometry-v3/fresh-qk-recoverability/v1"
GEOMETRY_DECISION_CEILING = "coordinates_only_no_positive_watermark_authority"

_ANCHOR_PRF_DOMAIN = b"CEG-WM/geometry-v3/keyed-qk-canonical-relation/v1\x00"
_MIN_ANCHOR_POINTS = 4
_MAX_ANCHOR_POINTS = 64


class FeatureRole(str, Enum):
    """Feature family in which a later writer may place an anchor."""

    QUERY = "q"
    KEY = "k"
    ATTENTION = "attention"


class PlacementBasis(str, Enum):
    """The only accepted placement authority for this route."""

    INDEPENDENT_PREDECLARED = "independent_predeclared"


class ObservationProvenance(str, Enum):
    """Admissible detector-side Q/K provenance."""

    FRESH_ATTACKED_RGB_QK = "fresh_qk_from_current_attacked_rgb"


class RecoverabilityPhase(str, Enum):
    WRITER_ADMITTED = "writer_admitted"
    FINAL_RGB_RECORDED = "final_rgb_recorded"
    ATTACKED_RGB_RECORDED = "attacked_rgb_recorded"
    FRESH_QK_OBSERVED = "fresh_qk_observed"
    GEOMETRY_ESTIMATED = "geometry_estimated"
    RECTIFICATION_AUTHORIZED = "rectification_authorized"
    STOPPED_UNRELIABLE = "stopped_unreliable"


@dataclass(frozen=True, slots=True)
class CanonicalRelationAnchor:
    method_id: str
    domain_id: str
    points: tuple[tuple[float, float], ...]
    public_digest: str


@dataclass(frozen=True, slots=True)
class WriterPlacement:
    layer_path: str
    feature_role: FeatureRole

    def __post_init__(self) -> None:
        _public_id(self.layer_path, name="layer_path")
        if not isinstance(self.feature_role, FeatureRole):
            raise TypeError("feature_role must be a FeatureRole")


@dataclass(frozen=True, slots=True)
class WriterBudget:
    max_total_relative_l2: float
    max_per_placement_relative_l2: float

    def __post_init__(self) -> None:
        total = _finite_float(self.max_total_relative_l2, name="max_total_relative_l2")
        per_placement = _finite_float(
            self.max_per_placement_relative_l2,
            name="max_per_placement_relative_l2",
        )
        if total <= 0.0 or per_placement <= 0.0:
            raise ValueError("writer budgets must be strictly positive")
        if per_placement > total:
            raise ValueError("per-placement budget cannot exceed total budget")


@dataclass(frozen=True, slots=True)
class WriterDeclaration:
    contract_id: str
    method_id: str
    placement_basis: PlacementBasis
    placement_protocol_id: str
    interference_test_protocol_id: str
    placements: tuple[WriterPlacement, ...]
    budget: WriterBudget

    def __post_init__(self) -> None:
        _validate_writer_declaration(self)


@dataclass(frozen=True, slots=True)
class ContentDetectorBinding:
    detector_id: str
    key_semantics_id: str
    preprocessing_id: str
    threshold_id: str

    def __post_init__(self) -> None:
        for name in (
            "detector_id",
            "key_semantics_id",
            "preprocessing_id",
            "threshold_id",
        ):
            _public_id(getattr(self, name), name=name)


@dataclass(frozen=True, slots=True)
class GeometryEstimate:
    observation_id: str
    reliability_protocol_id: str
    corners: tuple[tuple[float, float], ...]
    homography: tuple[tuple[float, float, float], ...]
    support_fraction: float
    reliability_score: float
    minimum_support_fraction: float
    minimum_reliability: float
    reliable: bool

    def __post_init__(self) -> None:
        _public_id(self.observation_id, name="observation_id")
        _public_id(self.reliability_protocol_id, name="reliability_protocol_id")
        if self.corners != _corners(self.corners):
            raise ValueError("corners must use the canonical tuple representation")
        if self.homography != _homography(self.homography):
            raise ValueError("homography must use the canonical tuple representation")
        support = _unit_interval(self.support_fraction, name="support_fraction")
        score = _unit_interval(self.reliability_score, name="reliability_score")
        minimum_support = _unit_interval(
            self.minimum_support_fraction, name="minimum_support_fraction"
        )
        minimum_reliability = _unit_interval(
            self.minimum_reliability, name="minimum_reliability"
        )
        if not isinstance(self.reliable, bool):
            raise TypeError("reliable must be boolean")
        if self.reliable is not (
            support >= minimum_support and score >= minimum_reliability
        ):
            raise ValueError("reliable must be derived from the frozen reliability rule")


@dataclass(frozen=True, slots=True)
class RecoverabilityState:
    contract_id: str
    method_id: str
    phase: RecoverabilityPhase
    writer_declaration: WriterDeclaration
    detector_binding: ContentDetectorBinding
    final_rgb_id: str | None = None
    attacked_rgb_id: str | None = None
    fresh_qk_observation_id: str | None = None
    estimate: GeometryEstimate | None = None


@dataclass(frozen=True, slots=True)
class RectificationAuthorization:
    method_id: str
    geometry_authority: str
    homography: tuple[tuple[float, float, float], ...]
    detector_binding: ContentDetectorBinding


def derive_canonical_relation_anchor(
    geometry_key: str | bytes | bytearray | memoryview,
    *,
    point_count: int,
) -> CanonicalRelationAnchor:
    """Derive an ordered public 2-D relation anchor without exposing the key."""

    if (
        not isinstance(point_count, int)
        or isinstance(point_count, bool)
        or not _MIN_ANCHOR_POINTS <= point_count <= _MAX_ANCHOR_POINTS
    ):
        raise ValueError(
            f"point_count must be between {_MIN_ANCHOR_POINTS} and {_MAX_ANCHOR_POINTS}"
        )
    key = normalize_detection_key(geometry_key)
    frame = point_count.to_bytes(2, "big")
    points: list[tuple[float, float]] = []
    for index in range(point_count):
        digest = hmac.new(
            key,
            _ANCHOR_PRF_DOMAIN + frame + index.to_bytes(2, "big"),
            hashlib.sha256,
        ).digest()
        x_raw = int.from_bytes(digest[:8], "big")
        y_raw = int.from_bytes(digest[8:16], "big")
        # Keep points away from the boundary while retaining deterministic
        # continuous coordinates. The half-bin term avoids exact endpoints.
        x = 0.1 + 0.8 * ((x_raw + 0.5) / 2**64)
        y = 0.1 + 0.8 * ((y_raw + 0.5) / 2**64)
        points.append((x, y))
    encoded = json.dumps(points, separators=(",", ":"), allow_nan=False).encode("ascii")
    return CanonicalRelationAnchor(
        method_id=METHOD_ID,
        domain_id=ANCHOR_DOMAIN_ID,
        points=tuple(points),
        public_digest=hashlib.sha256(encoded).hexdigest(),
    )


def declare_writer_contract(
    *,
    placements: Sequence[WriterPlacement],
    budget: WriterBudget,
    placement_basis: PlacementBasis,
    placement_protocol_id: str,
    interference_test_protocol_id: str,
) -> WriterDeclaration:
    """Freeze explicit writer placements and budget; there are no default layers."""

    frozen = tuple(placements)
    return WriterDeclaration(
        contract_id=WRITER_CONTRACT_ID,
        method_id=METHOD_ID,
        placement_basis=placement_basis,
        placement_protocol_id=_public_id(
            placement_protocol_id, name="placement_protocol_id"
        ),
        interference_test_protocol_id=_public_id(
            interference_test_protocol_id,
            name="interference_test_protocol_id",
        ),
        placements=frozen,
        budget=budget,
    )


def start_recoverability(
    writer_declaration: WriterDeclaration | None,
    *,
    detector_binding: ContentDetectorBinding,
) -> RecoverabilityState:
    """Enter the route only with a complete, explicit writer declaration."""

    if not isinstance(writer_declaration, WriterDeclaration):
        raise ValueError("a validated writer declaration is required before writer admission")
    _validate_writer_declaration(writer_declaration)
    if not isinstance(detector_binding, ContentDetectorBinding):
        raise TypeError("detector_binding must be a ContentDetectorBinding")
    return RecoverabilityState(
        contract_id=RECOVERABILITY_CONTRACT_ID,
        method_id=METHOD_ID,
        phase=RecoverabilityPhase.WRITER_ADMITTED,
        writer_declaration=writer_declaration,
        detector_binding=detector_binding,
    )


def record_final_rgb(state: RecoverabilityState, *, final_rgb_id: str) -> RecoverabilityState:
    _require_phase(state, RecoverabilityPhase.WRITER_ADMITTED)
    return replace(
        state,
        phase=RecoverabilityPhase.FINAL_RGB_RECORDED,
        final_rgb_id=_public_id(final_rgb_id, name="final_rgb_id"),
    )


def record_attacked_rgb(
    state: RecoverabilityState,
    *,
    attacked_rgb_id: str,
) -> RecoverabilityState:
    _require_phase(state, RecoverabilityPhase.FINAL_RGB_RECORDED)
    attacked = _public_id(attacked_rgb_id, name="attacked_rgb_id")
    if attacked == state.final_rgb_id:
        raise ValueError("attacked RGB identity must be distinct from final RGB identity")
    return replace(
        state,
        phase=RecoverabilityPhase.ATTACKED_RGB_RECORDED,
        attacked_rgb_id=attacked,
    )


def record_fresh_qk_observation(
    state: RecoverabilityState,
    *,
    observation_id: str,
    attacked_rgb_id: str,
    provenance: ObservationProvenance,
) -> RecoverabilityState:
    """Admit only Q/K recomputed from the current attacked RGB image."""

    _require_phase(state, RecoverabilityPhase.ATTACKED_RGB_RECORDED)
    if _public_id(attacked_rgb_id, name="attacked_rgb_id") != state.attacked_rgb_id:
        raise ValueError("fresh Q/K observation must bind the current attacked RGB")
    if provenance is not ObservationProvenance.FRESH_ATTACKED_RGB_QK:
        raise ValueError("detector Q/K must be freshly observed from current attacked RGB")
    return replace(
        state,
        phase=RecoverabilityPhase.FRESH_QK_OBSERVED,
        fresh_qk_observation_id=_public_id(observation_id, name="observation_id"),
    )


def make_geometry_estimate(
    state: RecoverabilityState,
    *,
    corners: Sequence[Sequence[float]],
    homography: Sequence[Sequence[float]],
    reliability_protocol_id: str,
    support_fraction: float,
    reliability_score: float,
    minimum_support_fraction: float,
    minimum_reliability: float,
) -> GeometryEstimate:
    """Validate corners/H and compute reliability without accepting a caller verdict."""

    _require_phase(state, RecoverabilityPhase.FRESH_QK_OBSERVED)
    frozen_corners = _corners(corners)
    frozen_h = _homography(homography)
    support = _unit_interval(support_fraction, name="support_fraction")
    score = _unit_interval(reliability_score, name="reliability_score")
    min_support = _unit_interval(
        minimum_support_fraction, name="minimum_support_fraction"
    )
    min_reliability = _unit_interval(minimum_reliability, name="minimum_reliability")
    return GeometryEstimate(
        observation_id=state.fresh_qk_observation_id or "",
        reliability_protocol_id=_public_id(
            reliability_protocol_id, name="reliability_protocol_id"
        ),
        corners=frozen_corners,
        homography=frozen_h,
        support_fraction=support,
        reliability_score=score,
        minimum_support_fraction=min_support,
        minimum_reliability=min_reliability,
        reliable=support >= min_support and score >= min_reliability,
    )


def record_geometry_estimate(
    state: RecoverabilityState,
    estimate: GeometryEstimate,
) -> RecoverabilityState:
    _require_phase(state, RecoverabilityPhase.FRESH_QK_OBSERVED)
    if not isinstance(estimate, GeometryEstimate):
        raise TypeError("estimate must be a GeometryEstimate")
    if estimate.observation_id != state.fresh_qk_observation_id:
        raise ValueError("estimate must bind the admitted fresh Q/K observation")
    phase = (
        RecoverabilityPhase.GEOMETRY_ESTIMATED
        if estimate.reliable
        else RecoverabilityPhase.STOPPED_UNRELIABLE
    )
    return replace(state, phase=phase, estimate=estimate)


def authorize_rectification(
    state: RecoverabilityState,
    *,
    detector_binding_after_rectification: ContentDetectorBinding,
) -> tuple[RecoverabilityState, RectificationAuthorization]:
    """Fail closed unless reliability passed and the content detector is unchanged."""

    _require_phase(state, RecoverabilityPhase.GEOMETRY_ESTIMATED)
    if state.estimate is None or not state.estimate.reliable:
        raise ValueError("unreliable geometry cannot authorize rectification")
    if detector_binding_after_rectification != state.detector_binding:
        raise ValueError(
            "rectification must preserve detector, key semantics, preprocessing, and threshold"
        )
    authorization = RectificationAuthorization(
        method_id=METHOD_ID,
        geometry_authority=GEOMETRY_DECISION_CEILING,
        homography=state.estimate.homography,
        detector_binding=state.detector_binding,
    )
    return (
        replace(state, phase=RecoverabilityPhase.RECTIFICATION_AUTHORIZED),
        authorization,
    )


def _public_id(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if not value or value != value.strip() or len(value.encode("utf-8")) > 256:
        raise ValueError(f"{name} must be non-empty bounded canonical text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} cannot contain control characters")
    return value


def _validate_writer_declaration(declaration: WriterDeclaration) -> None:
    if declaration.contract_id != WRITER_CONTRACT_ID or declaration.method_id != METHOD_ID:
        raise ValueError("writer declaration identity differs")
    if declaration.placement_basis is not PlacementBasis.INDEPENDENT_PREDECLARED:
        raise ValueError("writer placements require independent predeclaration")
    _public_id(declaration.placement_protocol_id, name="placement_protocol_id")
    _public_id(
        declaration.interference_test_protocol_id,
        name="interference_test_protocol_id",
    )
    if not isinstance(declaration.placements, tuple):
        raise TypeError("writer placements must use the frozen tuple representation")
    if not declaration.placements:
        raise ValueError("at least one independently predeclared placement is required")
    if len(declaration.placements) > 64:
        raise ValueError("writer declaration cannot exceed 64 placements")
    if any(
        not isinstance(placement, WriterPlacement)
        for placement in declaration.placements
    ):
        raise TypeError("placements must contain WriterPlacement values")
    identities = tuple(
        (item.layer_path, item.feature_role) for item in declaration.placements
    )
    if len(set(identities)) != len(identities):
        raise ValueError("writer placements must be unique")
    if not isinstance(declaration.budget, WriterBudget):
        raise TypeError("budget must be a WriterBudget")


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit_interval(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _corners(value: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    rows = tuple(tuple(row) for row in value)
    if len(rows) != 4 or any(len(row) != 2 for row in rows):
        raise ValueError("corners must contain exactly four normalized 2-D points")
    frozen = tuple(
        tuple(_unit_interval(coordinate, name="corner coordinate") for coordinate in row)
        for row in rows
    )
    if len(set(frozen)) != 4:
        raise ValueError("corners must be distinct")
    return frozen


def _homography(
    value: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    rows = tuple(tuple(row) for row in value)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("homography must have shape 3x3")
    frozen = tuple(
        tuple(_finite_float(coordinate, name="homography coefficient") for coordinate in row)
        for row in rows
    )
    if any(abs(coordinate) > 16.0 for row in frozen for coordinate in row):
        raise ValueError("normalized homography coefficients exceed the public bound")
    if not math.isclose(frozen[2][2], 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("homography must be normalized with H[2][2] == 1")
    a, b, c = frozen[0]
    d, e, f = frozen[1]
    g, h, i = frozen[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(determinant) <= 1e-12:
        raise ValueError("homography must be invertible")
    return frozen


def _require_phase(state: RecoverabilityState, expected: RecoverabilityPhase) -> None:
    if not isinstance(state, RecoverabilityState):
        raise TypeError("state must be a RecoverabilityState")
    if state.phase is not expected:
        raise ValueError(f"expected phase {expected.value}, received {state.phase.value}")
