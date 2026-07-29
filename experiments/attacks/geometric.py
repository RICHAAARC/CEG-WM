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
    if (
        not isinstance(image, torch.Tensor)
        or image.dtype is not torch.uint8
        or image.ndim != 4
        or tuple(image.shape[:2]) != (1, 3)
        or int(image.shape[2]) <= 1
        or int(image.shape[3]) <= 1
    ):
        raise GeometricAttackError(
            "attack image must be RGB uint8 [1,3,H,W] with H,W > 1"
        )
    return image.detach().clone()


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
        if self.attack_ids != ATTACK_IDS:
            raise GeometricAttackError("attack registry identity or order drifted")
        if (
            self.schema_version != "ceg_wm_internal_execution_components_v1"
            or self.registry_version != "ceg_wm_internal_attack_registry_v1"
            or self.image_interpolation != "bilinear"
            or self.image_padding != "zeros"
            or self.align_corners is not True
            or self.output_quantization != "clamp_floor_uint8"
            or self.output_size_policy != "preserve_source_size"
            or self.crop_fraction_bounds != CROP_FRACTION_BOUNDS
            or self.scale_factor_bounds != SCALE_FACTOR_BOUNDS
            or self.rotation_degrees_bounds != ROTATION_DEGREES_BOUNDS
        ):
            raise GeometricAttackError("attack registry semantics drifted")
        object.__setattr__(
            self,
            "registry_digest",
            _canonical_digest(
                {
                    "align_corners": self.align_corners,
                    "attack_ids": list(self.attack_ids),
                    "image_interpolation": self.image_interpolation,
                    "image_padding": self.image_padding,
                    "output_quantization": self.output_quantization,
                    "output_size_policy": self.output_size_policy,
                    "parameter_bounds": {
                        "crop_fraction": list(self.crop_fraction_bounds),
                        "rotation_degrees": list(
                            self.rotation_degrees_bounds
                        ),
                        "scale_factor": list(self.scale_factor_bounds),
                    },
                    "registry_version": self.registry_version,
                    "schema_version": self.schema_version,
                }
            ),
        )


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
        if type(self.analysis_unit_identity) is not AnalysisUnitIdentity:
            raise GeometricAttackError(
                "attack artifact requires AnalysisUnitIdentity"
            )
        violations = self.analysis_unit_identity.validate()
        if violations:
            raise GeometricAttackError(
                f"analysis unit identity invalid: {','.join(violations)}"
            )
        image = _validated_rgb8(self.image)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "image_digest", _image_digest(image))


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
        if not CROP_FRACTION_BOUNDS[0] <= crop <= CROP_FRACTION_BOUNDS[1]:
            raise GeometricAttackError("crop_fraction must be in [0.45,1]")
        if not SCALE_FACTOR_BOUNDS[0] <= scale <= SCALE_FACTOR_BOUNDS[1]:
            raise GeometricAttackError(
                "scale_factor must be in [1/sqrt(2),sqrt(2)]"
            )
        if not ROTATION_DEGREES_BOUNDS[0] <= rotation <= ROTATION_DEGREES_BOUNDS[1]:
            raise GeometricAttackError("rotation_degrees must be in [-32,32]")
        changed = sum(
            (crop != 1.0, scale != 1.0, rotation != 0.0)
        )
        expected_changed = {
            "identity": 0,
            "crop": 1,
            "scale": 1,
            "rotation": 1,
        }
        if self.attack_id in expected_changed:
            if changed != expected_changed[self.attack_id]:
                raise GeometricAttackError(
                    f"{self.attack_id} parameters do not match its semantics"
                )
            if self.attack_id == "crop" and crop == 1.0:
                raise GeometricAttackError("crop attack must remove border content")
            if self.attack_id == "scale" and scale == 1.0:
                raise GeometricAttackError("scale attack must change scale")
            if self.attack_id == "rotation" and rotation == 0.0:
                raise GeometricAttackError("rotation attack must rotate")
        elif self.attack_id == "crop_scale_rotation":
            if changed < 2:
                raise GeometricAttackError(
                    "combined geometric attack requires at least two active transforms"
                )
        else:
            raise GeometricAttackError("attack_id is not registered")
        object.__setattr__(self, "crop_fraction", crop)
        object.__setattr__(self, "scale_factor", scale)
        object.__setattr__(self, "rotation_degrees", rotation)
        object.__setattr__(
            self,
            "attack_config_digest",
            _canonical_digest(
                {
                    "attack_id": self.attack_id,
                    "crop_fraction": crop.hex(),
                    "rotation_degrees": rotation.hex(),
                    "scale_factor": scale.hex(),
                }
            ),
        )


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

    if type(artifact) is not AttackArtifact:
        raise GeometricAttackError("artifact must be AttackArtifact")
    if type(specification) is not GeometricAttackSpec:
        raise GeometricAttackError("specification must be GeometricAttackSpec")
    if type(registry) is not AttackRegistry:
        raise GeometricAttackError("registry must be AttackRegistry")
    if specification.attack_id not in registry.attack_ids:
        raise GeometricAttackError("attack is absent from the frozen registry")
    if (
        not registry.crop_fraction_bounds[0]
        <= specification.crop_fraction
        <= registry.crop_fraction_bounds[1]
        or not registry.scale_factor_bounds[0]
        <= specification.scale_factor
        <= registry.scale_factor_bounds[1]
        or not registry.rotation_degrees_bounds[0]
        <= specification.rotation_degrees
        <= registry.rotation_degrees_bounds[1]
    ):
        raise GeometricAttackError(
            "attack parameters exceed the frozen registry bounds"
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
