"""Frozen Geometry-V5 P0 governance and validation contract.

This module deliberately contains no synchronization, inversion, latent-write,
detector, or content-integration runtime. It only freezes interfaces for later
exact-bound work.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import InitVar, dataclass
from enum import Enum
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


GEOMETRY_V5_METHOD_ID = "geometry_v5_training_free_initial_noise_sync"
GEOMETRY_V5_P0_CONTRACT_PATH = "configs/geometry_v5/geometry_v5_p0_contract_v1.json"
GEOMETRY_V5_P0_CONTRACT_SHA256 = "079b1de98ead81e867bf7f9ea3d1f23e6ad29da2a95eaddf31bcd91e2ad1c80a"
GEOMETRY_V5_P0_CLAIM_CEILING = "P0_local_static_engineering_only_science_denominator_0"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_KEY_DOMAINS = ("k_search", "k_fit", "k_validate")
_DETECTOR_ALLOWED_INPUTS = (
    "attacked_ordinary_RGB",
    "geometry_root_key",
    "frozen_model_scheduler_inversion_identities",
)
_HKDF_SALT = b"CEG-WM/geometry-v5/HKDF-SHA256/salt/v1"
_HKDF_INFO_PREFIX = b"CEG-WM/geometry-v5/key-domain/v1\x00"


@dataclass(frozen=True, slots=True)
class GeometryV5Contract:
    """Immutable canonical P0 contract and its byte binding."""

    config: Mapping[str, Any]
    byte_sha256: str


class GeometryV5Status(str, Enum):
    RELIABLE = "RELIABLE"
    UNRELIABLE = "UNRELIABLE"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class ReliabilityConditions:
    """Structural, non-numeric prerequisites for safe rectification only."""

    search_candidate: bool
    fit_support: bool
    macro_region_coverage: bool
    residual: bool
    holdout_correlation_psr: bool
    cross_scale_rs_consistency: bool
    holdout_disjoint: bool
    legal_conditioning: bool

    def __post_init__(self) -> None:
        for name in (
            "search_candidate",
            "fit_support",
            "macro_region_coverage",
            "residual",
            "holdout_correlation_psr",
            "cross_scale_rs_consistency",
            "holdout_disjoint",
            "legal_conditioning",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")

    @property
    def satisfied(self) -> bool:
        return all(
            (
                self.search_candidate,
                self.fit_support,
                self.macro_region_coverage,
                self.residual,
                self.holdout_correlation_psr,
                self.cross_scale_rs_consistency,
                self.holdout_disjoint,
                self.legal_conditioning,
            )
        )


@dataclass(frozen=True, slots=True)
class GeometryV5Observation:
    """The complete public Geometry-V5 output, with no positive-vote field."""

    H_hat: tuple[tuple[float, float, float], ...] | None
    corners_hat: tuple[tuple[float, float], ...] | None
    support: int
    reliability: float
    status: GeometryV5Status
    reliability_conditions: InitVar[ReliabilityConditions | None] = None

    def __post_init__(
        self, reliability_conditions: ReliabilityConditions | None
    ) -> None:
        if isinstance(self.support, bool) or not isinstance(self.support, int):
            raise TypeError("support must be a non-bool integer")
        if self.support < 0:
            raise ValueError("support must be non-negative")
        reliability = _finite_real(self.reliability, "reliability")
        if not 0.0 <= reliability <= 1.0:
            raise ValueError("reliability must be in [0, 1]")
        try:
            status = GeometryV5Status(self.status)
        except (TypeError, ValueError) as error:
            raise ValueError("status must be RELIABLE, UNRELIABLE, or STOPPED") from error
        object.__setattr__(self, "reliability", reliability)
        object.__setattr__(self, "status", status)

        if status is GeometryV5Status.STOPPED:
            if (
                self.H_hat is not None
                or self.corners_hat is not None
                or self.support != 0
                or reliability != 0.0
            ):
                raise ValueError("STOPPED must not export a geometry estimate")
            return
        if status is GeometryV5Status.UNRELIABLE:
            if self.H_hat is not None or self.corners_hat is not None:
                raise ValueError("UNRELIABLE must not export a rectification estimate")
            return
        if reliability_conditions is None or not reliability_conditions.satisfied:
            raise ValueError("RELIABLE requires complete legal structural conditions")
        matrix = _similarity_matrix(self.H_hat)
        corners = _consistent_corners(self.corners_hat, matrix)
        object.__setattr__(self, "H_hat", matrix)
        object.__setattr__(self, "corners_hat", corners)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the only permitted serialization for the P0 JSON contract."""

    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def validate_lowercase_sha256(value: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError("SHA-256 digest must be lowercase 64-hex")
    return value


def load_geometry_v5_p0_contract(repo_root: str | Path) -> GeometryV5Contract:
    """Load P0 only when its canonical bytes match the frozen digest."""

    expected = validate_lowercase_sha256(GEOMETRY_V5_P0_CONTRACT_SHA256)
    raw = (Path(repo_root) / GEOMETRY_V5_P0_CONTRACT_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Geometry-V5 P0 contract byte digest differs")
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Geometry-V5 P0 contract must be UTF-8 JSON") from error
    if not isinstance(config, dict) or canonical_json_bytes(config) != raw:
        raise ValueError("Geometry-V5 P0 contract must use canonical JSON bytes")
    _validate_contract(config)
    return GeometryV5Contract(config=_freeze(config), byte_sha256=expected)


def derive_geometry_v5_key_domain_digests(
    geometry_root_key: bytes | bytearray | memoryview,
) -> Mapping[str, str]:
    """Derive internal HKDF domains while exporting only public digests."""

    if not isinstance(geometry_root_key, (bytes, bytearray, memoryview)):
        raise TypeError("geometry root key must be bytes-like")
    root = bytes(geometry_root_key)
    if not root:
        raise ValueError("geometry root key must be non-empty")
    prk = hmac.new(_HKDF_SALT, root, hashlib.sha256).digest()
    digests = {
        domain: hashlib.sha256(_hkdf_expand(prk, domain.encode("ascii"))).hexdigest()
        for domain in _KEY_DOMAINS
    }
    if len(set(digests.values())) != len(_KEY_DOMAINS):
        raise RuntimeError("Geometry-V5 HKDF domains must remain distinct")
    return MappingProxyType(digests)


def validate_detector_input_names(input_names: Sequence[str]) -> tuple[str, ...]:
    """Freeze the blind detector boundary before detector runtime exists."""

    if isinstance(input_names, (str, bytes)):
        raise TypeError("detector input names must be a sequence of names")
    received = tuple(input_names)
    if received != _DETECTOR_ALLOWED_INPUTS:
        raise ValueError("Geometry-V5 detector inputs must be the exact blind boundary")
    return received


def _hkdf_expand(prk: bytes, domain: bytes) -> bytes:
    info = _HKDF_INFO_PREFIX + domain
    previous = b""
    output = bytearray()
    counter = 1
    while len(output) < 32:
        previous = hmac.new(
            prk, previous + info + bytes((counter,)), hashlib.sha256
        ).digest()
        output.extend(previous)
        counter += 1
    return bytes(output[:32])


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _validate_contract(config: Mapping[str, Any]) -> None:
    expected_keys = {
        "contract_version", "contract_id", "method_id", "stage", "method_semantics",
        "key_hierarchy", "writer_conceptual_flow", "detector_boundary", "roles",
        "public_output", "reliable_structure", "failure_policy", "content_integration",
        "later_pre_run_contracts", "evidence_ceiling", "base_identity",
    }
    if set(config) != expected_keys:
        raise ValueError("Geometry-V5 P0 contract fields differ")
    if config["method_id"] != GEOMETRY_V5_METHOD_ID:
        raise ValueError("Geometry-V5 method identity differs")
    if config["evidence_ceiling"] != GEOMETRY_V5_P0_CLAIM_CEILING:
        raise ValueError("Geometry-V5 P0 evidence ceiling differs")
    stage = _mapping(config, "stage")
    if stage.get("current_stage") != "V5-P0" or stage.get("kind") != "governance_validation_plane_only":
        raise ValueError("Geometry-V5 P0 stage identity differs")
    if tuple(stage.get("preferred_progression", ())) != (
        "V5-M0_SD2_1_faithful_global_RST_reproduction",
        "V5-M1_key_domain_separation_and_holdout_safety",
        "V5-C0_keyed_local_latent_tiles_for_crop_crop_rescale",
        "V5-I0_unchanged_content_detector_integration",
        "V5-SD35_after_method_freeze_and_separately_proven_fixed_repeatable_inversion",
    ):
        raise ValueError("Geometry-V5 stage progression differs")
    hierarchy = _mapping(config, "key_hierarchy")
    if hierarchy.get("derivation") != "HKDF-SHA256" or tuple(hierarchy.get("domains", ())) != _KEY_DOMAINS:
        raise ValueError("Geometry-V5 key domains differ")
    if hierarchy.get("source_forbidden") != ["content_subkey"]:
        raise ValueError("Geometry-V5 key separation differs")
    boundary = _mapping(config, "detector_boundary")
    if tuple(boundary.get("allowed_inputs", ())) != _DETECTOR_ALLOWED_INPUTS:
        raise ValueError("Geometry-V5 blind detector inputs differ")
    forbidden = boundary.get("forbidden_inputs")
    required_forbidden = {
        "clean_RGB", "pre_attack_RGB", "original_z_T", "writer_tensors", "writer_residuals",
        "true_transform_parameters", "true_crop_parameters", "true_attack_parameters",
        "content_scores", "content_keys", "evaluation_truth", "retry", "fallback",
    }
    if not isinstance(forbidden, list) or set(forbidden) != required_forbidden:
        raise ValueError("Geometry-V5 forbidden detector inputs differ")
    roles = _mapping(config, "roles")
    if set(roles) != set(_KEY_DOMAINS):
        raise ValueError("Geometry-V5 role domains differ")
    if _mapping(roles, "k_search").get("may_make_RELIABLE") is not False:
        raise ValueError("Geometry-V5 search must not make RELIABLE")
    validate = _mapping(roles, "k_validate")
    if validate.get("may_participate_in") != [] or set(validate.get("forbidden_participation", ())) != {
        "candidate_proposal", "correspondence", "parameter_estimation", "tie_break",
        "threshold_tuning", "fallback",
    }:
        raise ValueError("Geometry-V5 holdout disjointness semantics differ")
    output = _mapping(config, "public_output")
    if tuple(output.get("fields", ())) != (
        "H_hat", "corners_hat", "support", "reliability", "status"
    ) or tuple(output.get("status_enum", ())) != tuple(status.value for status in GeometryV5Status):
        raise ValueError("Geometry-V5 public output differs")
    content = _mapping(config, "content_integration")
    if (
        content.get("content_and_geometry_keys") != "strictly_separated"
        or content.get("geometry_may_add_positive_evidence") is not False
        or content.get("s1") != "exact_same_content_path_and_tau"
        or content.get("tau_or_delta") != "unbound"
    ):
        raise ValueError("Geometry-V5 content boundary differs")
    if not isinstance(config["later_pre_run_contracts"], list) or any(
        not isinstance(item, str) for item in config["later_pre_run_contracts"]
    ):
        raise ValueError("Geometry-V5 unbound contracts differ")


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-bool real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _similarity_matrix(
    value: tuple[tuple[float, float, float], ...] | None,
) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("H_hat must be a 3x3 matrix")
    rows: list[tuple[float, float, float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 3:
            raise ValueError("H_hat must be a 3x3 matrix")
        rows.append(tuple(_finite_real(entry, "H_hat") for entry in row))
    matrix = tuple(rows)
    h00, h01, _ = matrix[0]
    h10, h11, _ = matrix[1]
    h20, h21, h22 = matrix[2]
    if not (
        _same(h20, 0.0) and _same(h21, 0.0) and _same(h22, 1.0)
        and _same(h00, h11) and _same(h01, -h10)
    ):
        raise ValueError("H_hat must be a normalized attacked_to_canonical similarity")
    if h00 * h00 + h10 * h10 <= 0.0:
        raise ValueError("H_hat similarity must be non-singular and non-reflective")
    return matrix


def _consistent_corners(
    value: tuple[tuple[float, float], ...] | None,
    matrix: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError("corners_hat must contain TL/TR/BR/BL")
    corners: list[tuple[float, float]] = []
    for corner in value:
        if not isinstance(corner, Sequence) or isinstance(corner, (str, bytes)) or len(corner) != 2:
            raise ValueError("each corner must be a normalized coordinate pair")
        x, y = (_finite_real(component, "corner") for component in corner)
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("corners_hat must use normalized coordinates")
        corners.append((x, y))
    frozen = tuple(corners)
    _validate_strict_convex_tl_tr_br_bl(frozen)
    canonical = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for corner, expected in zip(frozen, canonical, strict=True):
        mapped = _apply_similarity(matrix, corner)
        if not (_same(mapped[0], expected[0]) and _same(mapped[1], expected[1])):
            raise ValueError("H_hat and corners_hat are inconsistent")
    return frozen


def _apply_similarity(
    matrix: tuple[tuple[float, float, float], ...], point: tuple[float, float]
) -> tuple[float, float]:
    x, y = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2],
    )


def _validate_strict_convex_tl_tr_br_bl(corners: tuple[tuple[float, float], ...]) -> None:
    cross_products: list[float] = []
    for index in range(4):
        first = corners[index]
        second = corners[(index + 1) % 4]
        third = corners[(index + 2) % 4]
        cross_products.append(
            (second[0] - first[0]) * (third[1] - second[1])
            - (second[1] - first[1]) * (third[0] - second[0])
        )
    if any(value <= 0.0 for value in cross_products):
        raise ValueError("corners_hat must be strict convex TL/TR/BR/BL")


def _same(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
