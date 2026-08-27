"""Geometry-V2 keyed neural corner synchronization contracts."""

from cegwm.geometry_v2.contracts import (
    CANONICAL_CORNERS,
    ContentDetectorIdentity,
    GEOMETRY_AUTHORITY,
    GEOMETRY_KEY_DOMAIN,
    GeometryEstimate,
    KeyedSyncTarget,
    METHOD_IDENTITY,
    PROTOCOL_IDENTITY,
    RectificationRequest,
    ReliabilityAssessment,
    ReliabilityPolicy,
    assert_content_detector_identity_preserved,
    assess_reliability,
    build_rectification_request,
    derive_keyed_sync_target,
)

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
