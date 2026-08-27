"""Pure-CPU contracts for Geometry-V2 keyed neural corner synchronization.

This module freezes identities and data boundaries.  It deliberately does not
implement a neural embedder/extractor, model loading, or a watermark decision.
Geometry output can authorize coordinate rectification only; the unchanged
content detector remains the sole positive watermark authority.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import struct


METHOD_IDENTITY = "geometry_v2_keyed_neural_corner_sync"
PROTOCOL_IDENTITY = "geometry-v2-keyed-neural-corner-sync-contract-v1"
GEOMETRY_KEY_DOMAIN = b"CEG-WM/geometry-v2/keyed-neural-corner-sync/v1\x00"
GEOMETRY_AUTHORITY = "coordinates_only"

CANONICAL_CORNERS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
)
CORNER_BOUND_MIN = -0.25
CORNER_BOUND_MAX = 1.25
MIN_QUADRILATERAL_AREA = 1.0e-4
MIN_HOMOGRAPHY_DETERMINANT = 1.0e-8
MAX_NORMALIZED_HOMOGRAPHY_COEFFICIENT = 16.0
HOMOGRAPHY_CORNER_TOLERANCE = 1.0e-7


Point = tuple[float, float]
Corners = tuple[Point, Point, Point, Point]
Homography = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _require_plain_bytes(value: object, *, name: str, minimum: int, maximum: int) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if not minimum <= len(value) <= maximum:
        raise ValueError(f"{name} length must be in [{minimum}, {maximum}]")
    return value


def _expand_keyed_bytes(geometry_key: bytes, public_context: bytes, count: int) -> bytes:
    output = bytearray()
    counter = 0
    context_frame = struct.pack(">I", len(public_context)) + public_context
    while len(output) < count:
        message = GEOMETRY_KEY_DOMAIN + context_frame + struct.pack(">I", counter)
        output.extend(hmac.new(geometry_key, message, hashlib.sha256).digest())
        counter += 1
    return bytes(output[:count])


@dataclass(frozen=True, slots=True)
class KeyedSyncTarget:
    """Public keyed target consumed by a future weak-signal embedder.

    The target exposes a bipolar code and a digest of public context, never the
    geometry key or key-derived reusable secret material.
    """

    method_identity: str
    protocol_identity: str
    public_context_digest: str
    bipolar_code: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.method_identity != METHOD_IDENTITY:
            raise ValueError("keyed sync target method identity differs")
        if self.protocol_identity != PROTOCOL_IDENTITY:
            raise ValueError("keyed sync target protocol identity differs")
        if len(self.public_context_digest) != 64:
            raise ValueError("public context digest must be SHA-256 hex")
        if not self.bipolar_code or set(self.bipolar_code) - {-1, 1}:
            raise ValueError("keyed sync code must be non-empty and bipolar")


def derive_keyed_sync_target(
    geometry_key: bytes,
    public_context: bytes,
    *,
    code_length: int = 64,
) -> KeyedSyncTarget:
    """Derive a deterministic, domain-separated weak synchronization target."""

    key = _require_plain_bytes(
        geometry_key,
        name="geometry_key",
        minimum=16,
        maximum=4096,
    )
    context = _require_plain_bytes(
        public_context,
        name="public_context",
        minimum=1,
        maximum=4096,
    )
    if isinstance(code_length, bool) or not isinstance(code_length, int):
        raise TypeError("code_length must be an integer")
    if not 8 <= code_length <= 512:
        raise ValueError("code_length must be in [8, 512]")
    packed = _expand_keyed_bytes(key, context, (code_length + 7) // 8)
    code = tuple(
        1 if packed[index // 8] & (1 << (7 - index % 8)) else -1
        for index in range(code_length)
    )
    return KeyedSyncTarget(
        method_identity=METHOD_IDENTITY,
        protocol_identity=PROTOCOL_IDENTITY,
        public_context_digest=hashlib.sha256(context).hexdigest(),
        bipolar_code=code,
    )


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalize_corners(value: Sequence[Sequence[float]]) -> Corners:
    if len(value) != 4:
        raise ValueError("corners must contain exactly four ordered points")
    points: list[Point] = []
    for index, point in enumerate(value):
        if len(point) != 2:
            raise ValueError(f"corner[{index}] must contain exactly x and y")
        x = _finite_float(point[0], name=f"corner[{index}].x")
        y = _finite_float(point[1], name=f"corner[{index}].y")
        if not (
            CORNER_BOUND_MIN <= x <= CORNER_BOUND_MAX
            and CORNER_BOUND_MIN <= y <= CORNER_BOUND_MAX
        ):
            raise ValueError("corners exceed the frozen normalized boundary")
        points.append((x, y))

    crosses: list[float] = []
    for index in range(4):
        current = points[index]
        following = points[(index + 1) % 4]
        after = points[(index + 2) % 4]
        edge_one = (following[0] - current[0], following[1] - current[1])
        edge_two = (after[0] - following[0], after[1] - following[1])
        crosses.append(edge_one[0] * edge_two[1] - edge_one[1] * edge_two[0])
    if any(cross <= 0.0 for cross in crosses):
        raise ValueError("corners must be strictly convex in canonical order")

    area_twice = sum(
        points[index][0] * points[(index + 1) % 4][1]
        - points[(index + 1) % 4][0] * points[index][1]
        for index in range(4)
    )
    if area_twice / 2.0 < MIN_QUADRILATERAL_AREA:
        raise ValueError("corner quadrilateral is degenerate")
    return (points[0], points[1], points[2], points[3])


def _determinant(matrix: Homography) -> float:
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _normalize_homography(value: Sequence[Sequence[float]]) -> Homography:
    if len(value) != 3 or any(len(row) != 3 for row in value):
        raise ValueError("homography must be 3x3")
    rows = tuple(
        tuple(
            _finite_float(value[row][column], name=f"homography[{row}][{column}]")
            for column in range(3)
        )
        for row in range(3)
    )
    scale = rows[2][2]
    if abs(scale) < MIN_HOMOGRAPHY_DETERMINANT:
        raise ValueError("homography normalization coefficient is degenerate")
    normalized = tuple(tuple(item / scale for item in row) for row in rows)
    matrix: Homography = (normalized[0], normalized[1], normalized[2])
    if max(abs(item) for row in matrix for item in row) > MAX_NORMALIZED_HOMOGRAPHY_COEFFICIENT:
        raise ValueError("homography exceeds the frozen coefficient bound")
    if abs(_determinant(matrix)) < MIN_HOMOGRAPHY_DETERMINANT:
        raise ValueError("homography is singular or degenerate")
    return matrix


def _project(matrix: Homography, point: Point) -> Point:
    x, y = point
    denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if not math.isfinite(denominator) or abs(denominator) < MIN_HOMOGRAPHY_DETERMINANT:
        raise ValueError("homography maps a canonical corner to infinity")
    projected = (
        (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator,
        (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator,
    )
    if not all(math.isfinite(item) for item in projected):
        raise ValueError("homography projection must be finite")
    return projected


def _invert_homography(matrix: Homography) -> Homography:
    (a, b, c), (d, e, f), (g, h, i) = matrix
    determinant = _determinant(matrix)
    if abs(determinant) < MIN_HOMOGRAPHY_DETERMINANT:
        raise ValueError("homography is singular or degenerate")
    inverse = (
        ((e * i - f * h) / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant),
        ((f * g - d * i) / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant),
        ((d * h - e * g) / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant),
    )
    return _normalize_homography(inverse)


def _estimate_binding(corners: Corners, homography: Homography) -> str:
    payload = json.dumps(
        {"corners": corners, "homography": homography},
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class GeometryEstimate:
    """Extractor output: ordered attacked-image corners and canonical-to-attacked H."""

    corners: Corners
    homography: Homography

    def __post_init__(self) -> None:
        corners = _normalize_corners(self.corners)
        homography = _normalize_homography(self.homography)
        projected = tuple(_project(homography, point) for point in CANONICAL_CORNERS)
        for expected, actual in zip(corners, projected, strict=True):
            if max(abs(expected[0] - actual[0]), abs(expected[1] - actual[1])) > HOMOGRAPHY_CORNER_TOLERANCE:
                raise ValueError("corners and homography do not describe the same transform")
        object.__setattr__(self, "corners", corners)
        object.__setattr__(self, "homography", homography)

    @property
    def binding(self) -> str:
        return _estimate_binding(self.corners, self.homography)


@dataclass(frozen=True, slots=True)
class ReliabilityPolicy:
    min_confidence: float
    min_support: float

    def __post_init__(self) -> None:
        confidence = _finite_float(self.min_confidence, name="min_confidence")
        support = _finite_float(self.min_support, name="min_support")
        if not (0.0 < confidence <= 1.0 and 0.0 < support <= 1.0):
            raise ValueError("reliability thresholds must be in (0, 1]")
        object.__setattr__(self, "min_confidence", confidence)
        object.__setattr__(self, "min_support", support)


@dataclass(frozen=True, slots=True)
class ReliabilityAssessment:
    reliable: bool
    confidence: float | None
    support: float | None
    reason: str
    geometry_binding: str | None
    policy: ReliabilityPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.policy, ReliabilityPolicy):
            raise TypeError("reliability assessment must bind its policy")
        if self.reliable and (
            self.reason != "reliable"
            or self.geometry_binding is None
            or self.confidence is None
            or self.support is None
        ):
            raise ValueError("reliable assessment must be complete and geometry-bound")
        if self.reliable and (
            self.confidence < self.policy.min_confidence
            or self.support < self.policy.min_support
        ):
            raise ValueError("reliable assessment does not satisfy its bound policy")
        if not self.reliable and self.reason not in {
            "invalid_geometry",
            "invalid_measurement",
            "below_threshold",
        }:
            raise ValueError("unreliable assessment reason is not fail-closed")


def assess_reliability(
    estimate: GeometryEstimate | None,
    *,
    confidence: object,
    support: object,
    policy: ReliabilityPolicy,
) -> ReliabilityAssessment:
    """Apply an independent fail-closed reliability rule to extractor output."""

    if not isinstance(policy, ReliabilityPolicy):
        raise TypeError("policy must be a ReliabilityPolicy")
    if estimate is None:
        return ReliabilityAssessment(False, None, None, "invalid_geometry", None, policy)
    if not isinstance(estimate, GeometryEstimate):
        raise TypeError("estimate must be a GeometryEstimate or None")
    try:
        confidence_value = _finite_float(confidence, name="confidence")
        support_value = _finite_float(support, name="support")
    except (TypeError, ValueError):
        return ReliabilityAssessment(
            False,
            None,
            None,
            "invalid_measurement",
            estimate.binding,
            policy,
        )
    if not (0.0 <= confidence_value <= 1.0 and 0.0 <= support_value <= 1.0):
        return ReliabilityAssessment(
            False,
            None,
            None,
            "invalid_measurement",
            estimate.binding,
            policy,
        )
    reliable = (
        confidence_value >= policy.min_confidence
        and support_value >= policy.min_support
    )
    return ReliabilityAssessment(
        reliable=reliable,
        confidence=confidence_value,
        support=support_value,
        reason="reliable" if reliable else "below_threshold",
        geometry_binding=estimate.binding,
        policy=policy,
    )


def _identity_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if not value or len(value) > 256 or value.strip() != value or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must be non-empty bounded public text")
    return value


@dataclass(frozen=True, slots=True)
class ContentDetectorIdentity:
    """Frozen content detector identity reused before and after rectification."""

    detector_identity: str
    detection_key_identity: str
    preprocessing_identity: str
    tau: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "detector_identity", _identity_text(self.detector_identity, name="detector_identity"))
        object.__setattr__(self, "detection_key_identity", _identity_text(self.detection_key_identity, name="detection_key_identity"))
        object.__setattr__(self, "preprocessing_identity", _identity_text(self.preprocessing_identity, name="preprocessing_identity"))
        tau = _finite_float(self.tau, name="tau")
        object.__setattr__(self, "tau", tau)


@dataclass(frozen=True, slots=True)
class RectificationRequest:
    """A coordinate-only request; it cannot carry a watermark decision."""

    source_corners: Corners
    attacked_to_canonical_homography: Homography
    content_detector_identity: ContentDetectorIdentity
    geometry_authority: str = GEOMETRY_AUTHORITY
    positive_watermark_authority: bool = False

    def __post_init__(self) -> None:
        if self.geometry_authority != GEOMETRY_AUTHORITY:
            raise ValueError("Geometry-V2 authority must remain coordinates-only")
        if self.positive_watermark_authority is not False:
            raise ValueError("geometry cannot create positive watermark authority")
        _normalize_corners(self.source_corners)
        _normalize_homography(self.attacked_to_canonical_homography)
        if not isinstance(self.content_detector_identity, ContentDetectorIdentity):
            raise TypeError("content detector identity must be frozen")


def build_rectification_request(
    estimate: GeometryEstimate,
    reliability: ReliabilityAssessment,
    content_detector_identity: ContentDetectorIdentity,
) -> RectificationRequest | None:
    """Return no request unless independent reliability passes fail-closed."""

    if not isinstance(estimate, GeometryEstimate):
        raise TypeError("estimate must be a GeometryEstimate")
    if not isinstance(reliability, ReliabilityAssessment):
        raise TypeError("reliability must be a ReliabilityAssessment")
    if not isinstance(content_detector_identity, ContentDetectorIdentity):
        raise TypeError("content_detector_identity must be frozen")
    if not reliability.reliable:
        return None
    if reliability.geometry_binding != estimate.binding:
        raise ValueError("reliability assessment belongs to a different geometry estimate")
    return RectificationRequest(
        source_corners=estimate.corners,
        attacked_to_canonical_homography=_invert_homography(estimate.homography),
        content_detector_identity=content_detector_identity,
    )


def assert_content_detector_identity_preserved(
    request: RectificationRequest,
    detector_identity_after_rectification: ContentDetectorIdentity,
) -> ContentDetectorIdentity:
    """Fail if detector, detection key, preprocessing, or tau changed."""

    if request.content_detector_identity != detector_identity_after_rectification:
        raise ValueError("content detector identity changed across rectification")
    return detector_identity_after_rectification


__all__ = [
    "CANONICAL_CORNERS",
    "ContentDetectorIdentity",
    "GEOMETRY_AUTHORITY",
    "GEOMETRY_KEY_DOMAIN",
    "GeometryEstimate",
    "KeyedSyncTarget",
    "METHOD_IDENTITY",
    "PROTOCOL_IDENTITY",
    "RectificationRequest",
    "ReliabilityAssessment",
    "ReliabilityPolicy",
    "assert_content_detector_identity_preserved",
    "assess_reliability",
    "build_rectification_request",
    "derive_keyed_sync_target",
]
