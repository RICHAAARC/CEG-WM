"""Deterministic internal identity/crop/scale/rotation image attacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import cos, isfinite, pi, sin, sqrt
from pathlib import Path

import torch
import torch.nn.functional as functional

from experiments.protocol.internal_splits import AnalysisUnitIdentity


ATTACK_IDS = (
    "identity",
    "crop",
    "scale",
    "rotation",
    "crop_scale_rotation",
)
CROP_FRACTION_BOUNDS = (0.45, 1.0)
SCALE_FACTOR_BOUNDS = (0.7071067811865476, 1.4142135623730951)
ROTATION_DEGREES_BOUNDS = (-32.0, 32.0)
DEFAULT_COMPONENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "experiments"
    / "internal_execution_components.json"
)


class GeometricAttackError(ValueError):
    """An image artifact or attack specification failed closed."""


def _canonical_digest(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _image_digest(image: torch.Tensor) -> str:
    cpu = image.detach().to(device="cpu").contiguous()
    payload = {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "values_sha256": sha256(cpu.numpy().tobytes()).hexdigest(),
    }
    return _canonical_digest(payload)


def _validated_rgb8(image: object) -> torch.Tensor:
    if not _rgb8_valid(image):
        raise GeometricAttackError(
            "attack image must be RGB uint8 [1,3,H,W] with H,W > 1"
        )
    return image.detach().clone()


def _rgb8_valid(image: object) -> bool:
    return bool(
        isinstance(image, torch.Tensor)
        and image.dtype is torch.uint8
        and image.ndim == 4
        and tuple(image.shape[:2]) == (1, 3)
        and int(image.shape[2]) > 1
        and int(image.shape[3]) > 1
    )


@dataclass(frozen=True, slots=True)
class AttackRegistry:
    schema_version: str
    registry_version: str
    attack_ids: tuple[str, ...]
    image_interpolation: str
    image_padding: str
    align_corners: bool
    output_quantization: str
    output_size_policy: str
    crop_fraction_bounds: tuple[float, float]
    scale_factor_bounds: tuple[float, float]
    rotation_degrees_bounds: tuple[float, float]
    registry_digest: str = field(init=False)

    def __post_init__(self) -> None:
        violations = _attack_registry_semantic_violations(self)
        if "attack_registry_identity_or_order_drifted" in violations:
            raise GeometricAttackError("attack registry identity or order drifted")
        if violations:
            raise GeometricAttackError("attack registry semantics drifted")
        object.__setattr__(
            self,
            "registry_digest",
            _canonical_digest(_attack_registry_payload(self)),
        )


def validate_attack_registry(registry: AttackRegistry) -> tuple[str, ...]:
    """Revalidate current attack semantics and their construction-time digest."""

    if type(registry) is not AttackRegistry:
        return ("attack_registry_exact_type_required",)
    violations = list(_attack_registry_semantic_violations(registry))
    if registry.registry_digest != _canonical_digest(
        _attack_registry_payload(registry)
    ):
        violations.append("attack_registry_digest_drifted")
    return tuple(dict.fromkeys(violations))


def _attack_registry_semantic_violations(
    registry: AttackRegistry,
) -> tuple[str, ...]:
    violations: list[str] = []
    if registry.attack_ids != ATTACK_IDS:
        violations.append("attack_registry_identity_or_order_drifted")
    if (
        registry.schema_version
        != "ceg_wm_internal_execution_components_v1"
        or registry.registry_version != "ceg_wm_internal_attack_registry_v1"
        or registry.image_interpolation != "bilinear"
        or registry.image_padding != "zeros"
        or registry.align_corners is not True
        or registry.output_quantization != "clamp_floor_uint8"
        or registry.output_size_policy != "preserve_source_size"
        or registry.crop_fraction_bounds != CROP_FRACTION_BOUNDS
        or registry.scale_factor_bounds != SCALE_FACTOR_BOUNDS
        or registry.rotation_degrees_bounds != ROTATION_DEGREES_BOUNDS
    ):
        violations.append("attack_registry_semantics_drifted")
    return tuple(violations)


def _attack_registry_payload(registry: AttackRegistry) -> dict[str, object]:
    return {
        "align_corners": registry.align_corners,
        "attack_ids": list(registry.attack_ids),
        "image_interpolation": registry.image_interpolation,
        "image_padding": registry.image_padding,
        "output_quantization": registry.output_quantization,
        "output_size_policy": registry.output_size_policy,
        "parameter_bounds": {
            "crop_fraction": list(registry.crop_fraction_bounds),
            "rotation_degrees": list(registry.rotation_degrees_bounds),
            "scale_factor": list(registry.scale_factor_bounds),
        },
        "registry_version": registry.registry_version,
        "schema_version": registry.schema_version,
    }


def load_attack_registry(
    path: str | Path = DEFAULT_COMPONENT_CONFIG_PATH,
) -> AttackRegistry:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if type(document) is not dict or set(document) != {
        "attack_registry",
        "method_adapter",
        "metric_registry",
        "schema_version",
    }:
        raise GeometricAttackError(
            "execution component configuration fields drifted"
        )
    raw = document.get("attack_registry")
    if type(raw) is not dict or set(raw) != {
        "align_corners",
        "attack_ids",
        "image_interpolation",
        "image_padding",
        "output_quantization",
        "output_size_policy",
        "parameter_bounds",
        "registry_version",
    }:
        raise GeometricAttackError("attack_registry configuration missing")
    try:
        bounds = raw["parameter_bounds"]
        if type(bounds) is not dict or set(bounds) != {
            "crop_fraction",
            "rotation_degrees",
            "scale_factor",
        }:
            raise GeometricAttackError(
                "attack parameter bound fields drifted"
            )
        return AttackRegistry(
            schema_version=document["schema_version"],
            registry_version=raw["registry_version"],
            attack_ids=tuple(raw["attack_ids"]),
            image_interpolation=raw["image_interpolation"],
            image_padding=raw["image_padding"],
            align_corners=raw["align_corners"],
            output_quantization=raw["output_quantization"],
            output_size_policy=raw["output_size_policy"],
            crop_fraction_bounds=tuple(bounds["crop_fraction"]),
            scale_factor_bounds=tuple(bounds["scale_factor"]),
            rotation_degrees_bounds=tuple(bounds["rotation_degrees"]),
        )
    except (KeyError, TypeError) as exc:
        raise GeometricAttackError("attack registry is incomplete") from exc


@dataclass(frozen=True, slots=True)
class AttackArtifact:
    analysis_unit_identity: AnalysisUnitIdentity
    image: torch.Tensor
    image_digest: str = field(init=False)

    def __post_init__(self) -> None:
        image = _validated_rgb8(self.image)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "image_digest", _image_digest(image))
        violations = validate_attack_artifact(self)
        if violations:
            raise GeometricAttackError(
                f"attack artifact invalid: {','.join(violations)}"
            )


def validate_attack_artifact(artifact: AttackArtifact) -> tuple[str, ...]:
    """Revalidate artifact identity, current RGB8 pixels, and their digest."""

    if type(artifact) is not AttackArtifact:
        return ("attack_artifact_exact_type_required",)
    violations: list[str] = []
    if type(artifact.analysis_unit_identity) is not AnalysisUnitIdentity:
        violations.append("analysis_unit_identity_exact_type_required")
    else:
        violations.extend(artifact.analysis_unit_identity.validate())
    if not _rgb8_valid(artifact.image):
        violations.append("attack_artifact_rgb8_invalid")
    elif artifact.image_digest != _image_digest(artifact.image):
        violations.append("attack_artifact_image_digest_drifted")
    return tuple(dict.fromkeys(violations))


@dataclass(frozen=True, slots=True)
class GeometricAttackSpec:
    attack_id: str
    crop_fraction: float = 1.0
    scale_factor: float = 1.0
    rotation_degrees: float = 0.0
    attack_config_digest: str = field(init=False)

    def __post_init__(self) -> None:
        values = (self.crop_fraction, self.scale_factor, self.rotation_degrees)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in values
        ):
            raise GeometricAttackError("attack parameters must be finite numbers")
        crop = float(self.crop_fraction)
        scale = float(self.scale_factor)
        rotation = float(self.rotation_degrees)
        object.__setattr__(self, "crop_fraction", crop)
        object.__setattr__(self, "scale_factor", scale)
        object.__setattr__(self, "rotation_degrees", rotation)
        object.__setattr__(
            self,
            "attack_config_digest",
            _attack_spec_digest(self),
        )
        violations = validate_geometric_attack_spec(self)
        if violations:
            raise GeometricAttackError(_attack_spec_violation_message(violations[0]))


def validate_geometric_attack_spec(
    specification: GeometricAttackSpec,
) -> tuple[str, ...]:
    """Revalidate current attack parameters, active transforms, and digest."""

    if type(specification) is not GeometricAttackSpec:
        return ("geometric_attack_spec_exact_type_required",)
    violations: list[str] = []
    values = (
        specification.crop_fraction,
        specification.scale_factor,
        specification.rotation_degrees,
    )
    if any(type(value) is not float or not isfinite(value) for value in values):
        return ("attack_parameters_finite_float_required",)
    crop, scale, rotation = values
    if not CROP_FRACTION_BOUNDS[0] <= crop <= CROP_FRACTION_BOUNDS[1]:
        violations.append("crop_fraction_out_of_bounds")
    if not SCALE_FACTOR_BOUNDS[0] <= scale <= SCALE_FACTOR_BOUNDS[1]:
        violations.append("scale_factor_out_of_bounds")
    if not ROTATION_DEGREES_BOUNDS[0] <= rotation <= ROTATION_DEGREES_BOUNDS[1]:
        violations.append("rotation_degrees_out_of_bounds")
    changed = sum((crop != 1.0, scale != 1.0, rotation != 0.0))
    expected_changed = {
        "identity": 0,
        "crop": 1,
        "scale": 1,
        "rotation": 1,
    }
    if specification.attack_id in expected_changed:
        if changed != expected_changed[specification.attack_id]:
            violations.append("attack_active_parameter_count_invalid")
        if specification.attack_id == "crop" and crop == 1.0:
            violations.append("crop_attack_inactive")
        if specification.attack_id == "scale" and scale == 1.0:
            violations.append("scale_attack_inactive")
        if specification.attack_id == "rotation" and rotation == 0.0:
            violations.append("rotation_attack_inactive")
    elif specification.attack_id == "crop_scale_rotation":
        if changed < 2:
            violations.append("combined_attack_active_parameter_count_invalid")
    else:
        violations.append("attack_id_unregistered")
    if specification.attack_config_digest != _attack_spec_digest(specification):
        violations.append("attack_config_digest_drifted")
    return tuple(dict.fromkeys(violations))


def _attack_spec_digest(specification: GeometricAttackSpec) -> str:
    return _canonical_digest(
        {
            "attack_id": specification.attack_id,
            "crop_fraction": specification.crop_fraction.hex(),
            "rotation_degrees": specification.rotation_degrees.hex(),
            "scale_factor": specification.scale_factor.hex(),
        }
    )


def _attack_spec_violation_message(violation: str) -> str:
    messages = {
        "attack_active_parameter_count_invalid": (
            "attack parameters do not match its semantics"
        ),
        "attack_config_digest_drifted": "attack configuration digest drifted",
        "attack_id_unregistered": "attack_id is not registered",
        "attack_parameters_finite_float_required": (
            "attack parameters must be finite numbers"
        ),
        "combined_attack_active_parameter_count_invalid": (
            "combined geometric attack requires at least two active transforms"
        ),
        "crop_attack_inactive": "crop attack must remove border content",
        "crop_fraction_out_of_bounds": "crop_fraction must be in [0.45,1]",
        "rotation_attack_inactive": "rotation attack must rotate",
        "rotation_degrees_out_of_bounds": "rotation_degrees must be in [-32,32]",
        "scale_attack_inactive": "scale attack must change scale",
        "scale_factor_out_of_bounds": (
            "scale_factor must be in [1/sqrt(2),sqrt(2)]"
        ),
    }
    return messages[violation]


@dataclass(frozen=True, slots=True)
class GeometricAttackResult:
    source_artifact_digest: str
    attacked_artifact: AttackArtifact
    attack_id: str
    attack_config_digest: str
    attack_registry_digest: str
    output_to_input_matrix: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]
    interpolation: str
    padding: str
    align_corners: bool
    output_quantization: str


def apply_geometric_attack(
    artifact: AttackArtifact,
    specification: GeometricAttackSpec,
    *,
    registry: AttackRegistry,
) -> GeometricAttackResult:
    """Apply one registered, deterministic output-to-input affine attack."""

    registry_violations = validate_attack_registry(registry)
    artifact_violations = validate_attack_artifact(artifact)
    specification_violations = validate_geometric_attack_spec(specification)
    if registry_violations:
        raise GeometricAttackError(
            f"attack registry invalid: {','.join(registry_violations)}"
        )
    if artifact_violations:
        raise GeometricAttackError(
            f"attack artifact invalid: {','.join(artifact_violations)}"
        )
    if specification_violations:
        raise GeometricAttackError(
            "attack specification invalid: "
            + ",".join(specification_violations)
        )

    angle = specification.rotation_degrees * pi / 180.0
    sampling_scale = specification.crop_fraction / specification.scale_factor
    matrix = (
        (
            sampling_scale * cos(angle),
            sampling_scale * sin(angle),
            0.0,
        ),
        (
            -sampling_scale * sin(angle),
            sampling_scale * cos(angle),
            0.0,
        ),
    )
    if specification.attack_id == "identity":
        attacked_image = artifact.image.detach().clone()
    else:
        theta = torch.tensor(
            matrix,
            dtype=torch.float32,
            device=artifact.image.device,
        ).unsqueeze(0)
        grid = functional.affine_grid(
            theta,
            artifact.image.shape,
            align_corners=registry.align_corners,
        )
        sampled = functional.grid_sample(
            artifact.image.to(dtype=torch.float32) / 255.0,
            grid,
            mode=registry.image_interpolation,
            padding_mode=registry.image_padding,
            align_corners=registry.align_corners,
        )
        attacked_image = torch.floor(
            torch.clamp(sampled, 0.0, 1.0) * 255.0
        ).to(dtype=torch.uint8)
    attacked_artifact = AttackArtifact(
        analysis_unit_identity=artifact.analysis_unit_identity,
        image=attacked_image,
    )
    return GeometricAttackResult(
        source_artifact_digest=artifact.image_digest,
        attacked_artifact=attacked_artifact,
        attack_id=specification.attack_id,
        attack_config_digest=specification.attack_config_digest,
        attack_registry_digest=registry.registry_digest,
        output_to_input_matrix=matrix,
        interpolation=registry.image_interpolation,
        padding=registry.image_padding,
        align_corners=registry.align_corners,
        output_quantization=registry.output_quantization,
    )
