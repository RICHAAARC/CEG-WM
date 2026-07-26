"""CEG-WM 冻结的 S/T/R/Q 内容路由与同预算禁用对照。"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import floor, isfinite, prod
from struct import pack, unpack
from typing import Literal, Sequence

from main.shared.key_schedule import DEFAULT_CONFIG, stable_json_utf8

ROUTING_CANDIDATE_ID = "routing_stqr"
UNIFORM_CONTROL_CANDIDATE_ID = "routing_uniform_control"
ROUTING_KEY_CANDIDATE_ID = "key_schedule_sha256_counter"
RoutingMode = Literal["routing_stqr", "routing_uniform_control"]


class ContentRouterError(ValueError):
    """路由观测、插值、mask 或候选身份无效。"""


def _float32(value: float) -> float:
    if not isfinite(value):
        raise ContentRouterError("routing numeric value must be finite")
    try:
        rounded = unpack(">f", pack(">f", value))[0]
    except (OverflowError, ValueError) as exc:
        raise ContentRouterError("routing value is outside binary32 range") from exc
    if not isfinite(rounded):
        raise ContentRouterError("routing binary32 value must be finite")
    return rounded


def _digest_float32(values: Sequence[float]) -> str:
    return sha256(b"".join(pack(">f", value) for value in values)).hexdigest()


def _validate_latent_shape(
    shape: Sequence[int],
) -> tuple[int, int, int, int]:
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        raise ContentRouterError("latent_shape must be positive [1,C,H,W]")
    normalized = tuple(shape)
    if (
        len(normalized) != 4
        or normalized[0] != 1
        or any(type(size) is not int or size <= 0 for size in normalized)
    ):
        raise ContentRouterError("latent_shape must be positive [1,C,H,W]")
    return normalized


@dataclass(frozen=True, slots=True)
class SpatialRoutingObservation:
    """runtime 已计算的单通道生成时空间观测。"""

    values: tuple[float, ...]
    spatial_shape: tuple[int, int]
    source_identity_digest: str
    values_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            len(self.spatial_shape) != 2
            or any(type(size) is not int or size <= 0 for size in self.spatial_shape)
        ):
            raise ContentRouterError("observation shape must be positive [H,W]")
        expected = prod(self.spatial_shape)
        if len(self.values) != expected:
            raise ContentRouterError("observation length does not match [H,W]")
        if (
            type(self.source_identity_digest) is not str
            or len(self.source_identity_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_identity_digest
            )
        ):
            raise ContentRouterError(
                "observation source identity must be a lowercase SHA-256 digest"
            )
        normalized = []
        for value in self.values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContentRouterError(
                    "routing observation must contain only finite numbers"
                )
            converted = _float32(float(value))
            if not 0.0 <= converted <= 1.0:
                raise ContentRouterError("routing observation values must be in [0,1]")
            normalized.append(converted)
        normalized_values = tuple(normalized)
        object.__setattr__(self, "values", normalized_values)
        object.__setattr__(
            self,
            "values_digest",
            _digest_float32(normalized_values),
        )


@dataclass(frozen=True, slots=True)
class RoutingObservations:
    """S/T/R/Q_sens 四个已物化数值观测。"""

    semantic: SpatialRoutingObservation
    texture: SpatialRoutingObservation
    response: SpatialRoutingObservation
    sensitivity: SpatialRoutingObservation


@dataclass(frozen=True, slots=True)
class ContentRoutingResult:
    """按 latent CHW 广播后的 A 与两分支 mask。"""

    candidate_id: str
    candidate_ids: tuple[str, ...]
    mode: RoutingMode
    latent_shape: tuple[int, int, int, int]
    routing_observations: RoutingObservations | None
    routing_map: tuple[float, ...]
    mask_lf: tuple[float, ...]
    mask_hf: tuple[float, ...]
    routing_map_digest: str
    mask_lf_digest: str
    mask_hf_digest: str
    observation_digests: tuple[tuple[str, str, str], ...]
    mean_routing_map: float
    mean_mask_lf: float
    mean_mask_hf: float
    route_config_digest: str
    route_identity: str


def _resize_bilinear_align_corners_false(
    observation: SpatialRoutingObservation,
    target_height: int,
    target_width: int,
) -> tuple[float, ...]:
    source_height, source_width = observation.spatial_shape
    output: list[float] = []
    for target_row in range(target_height):
        source_y = (target_row + 0.5) * source_height / target_height - 0.5
        y0_unclamped = floor(source_y)
        y1_unclamped = y0_unclamped + 1
        weight_y = source_y - y0_unclamped
        y0 = min(max(y0_unclamped, 0), source_height - 1)
        y1 = min(max(y1_unclamped, 0), source_height - 1)
        for target_column in range(target_width):
            source_x = (
                (target_column + 0.5) * source_width / target_width - 0.5
            )
            x0_unclamped = floor(source_x)
            x1_unclamped = x0_unclamped + 1
            weight_x = source_x - x0_unclamped
            x0 = min(max(x0_unclamped, 0), source_width - 1)
            x1 = min(max(x1_unclamped, 0), source_width - 1)
            top = (
                observation.values[y0 * source_width + x0] * (1.0 - weight_x)
                + observation.values[y0 * source_width + x1] * weight_x
            )
            bottom = (
                observation.values[y1 * source_width + x0] * (1.0 - weight_x)
                + observation.values[y1 * source_width + x1] * weight_x
            )
            output.append(_float32(top * (1.0 - weight_y) + bottom * weight_y))
    return tuple(output)


def _broadcast_spatial(
    values: Sequence[float],
    channels: int,
) -> tuple[float, ...]:
    return tuple(value for _ in range(channels) for value in values)


def _identity_digest(value: object) -> str:
    return sha256(stable_json_utf8(value)).hexdigest()


def _route_config(
    mode: RoutingMode,
    latent_shape: tuple[int, int, int, int],
) -> dict[str, object]:
    candidate_ids = (
        ROUTING_KEY_CANDIDATE_ID,
        mode,
    )
    return {
        "candidate_ids": list(candidate_ids),
        "formula": (
            "A=((1-S)*(1-R)*(1-Q_sens))^(1/3);"
            "mask_lf=A*(1-T);mask_hf=A*T"
            if mode == ROUTING_CANDIDATE_ID
            else "A=mask_lf=mask_hf=1"
        ),
        "interpolation": "bilinear_align_corners_false",
        "key_schedule_config_digest": DEFAULT_CONFIG.config_digest,
        "latent_shape": list(latent_shape),
        "mode": mode,
        "uniform_control_reads_observations": False,
    }


def _validate_result_vector(
    values: object,
    element_count: int,
    role: str,
) -> tuple[float, ...]:
    if type(values) is not tuple or len(values) != element_count:
        raise ContentRouterError(f"{role} length does not match latent shape")
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContentRouterError(f"{role} must contain finite float32 values")
        converted = _float32(float(value))
        if converted != value or not 0.0 <= converted <= 1.0:
            raise ContentRouterError(f"{role} must contain float32 values in [0,1]")
        normalized.append(converted)
    return tuple(normalized)


def _is_sha256_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_spatial_observation(
    observation: object,
    role: str,
) -> SpatialRoutingObservation:
    if type(observation) is not SpatialRoutingObservation:
        raise ContentRouterError(
            f"{role} must be an immutable SpatialRoutingObservation"
        )
    if (
        type(observation.spatial_shape) is not tuple
        or len(observation.spatial_shape) != 2
        or any(
            type(size) is not int or size <= 0
            for size in observation.spatial_shape
        )
    ):
        raise ContentRouterError(f"{role} spatial shape is invalid")
    element_count = prod(observation.spatial_shape)
    values = _validate_result_vector(
        observation.values,
        element_count,
        f"{role} values",
    )
    if not _is_sha256_digest(observation.source_identity_digest):
        raise ContentRouterError(f"{role} source identity is invalid")
    if observation.values_digest != _digest_float32(values):
        raise ContentRouterError(f"{role} values digest mismatch")
    return observation


def _validate_routing_observations(
    observations: object,
) -> RoutingObservations:
    if type(observations) is not RoutingObservations:
        raise ContentRouterError(
            "routed result requires immutable RoutingObservations"
        )
    for role, observation in (
        ("S", observations.semantic),
        ("T", observations.texture),
        ("R", observations.response),
        ("Q_sens", observations.sensitivity),
    ):
        _validate_spatial_observation(observation, role)
    return observations


def _routing_observation_digests(
    observations: RoutingObservations,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            role,
            observation.values_digest,
            observation.source_identity_digest,
        )
        for role, observation in (
            ("S", observations.semantic),
            ("T", observations.texture),
            ("R", observations.response),
            ("Q_sens", observations.sensitivity),
        )
    )


def _derive_routed_tensors(
    observations: object,
    latent_shape: tuple[int, int, int, int],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    validated = _validate_routing_observations(observations)
    _, channels, target_height, target_width = latent_shape
    spatial_maps = {
        "S": _resize_bilinear_align_corners_false(
            validated.semantic,
            target_height,
            target_width,
        ),
        "T": _resize_bilinear_align_corners_false(
            validated.texture,
            target_height,
            target_width,
        ),
        "R": _resize_bilinear_align_corners_false(
            validated.response,
            target_height,
            target_width,
        ),
        "Q_sens": _resize_bilinear_align_corners_false(
            validated.sensitivity,
            target_height,
            target_width,
        ),
    }
    routing_spatial = tuple(
        _float32(
            (
                (1.0 - semantic)
                * (1.0 - response)
                * (1.0 - sensitivity)
            )
            ** (1.0 / 3.0)
        )
        for semantic, response, sensitivity in zip(
            spatial_maps["S"],
            spatial_maps["R"],
            spatial_maps["Q_sens"],
            strict=True,
        )
    )
    mask_lf_spatial = tuple(
        _float32(routing_value * (1.0 - texture))
        for routing_value, texture in zip(
            routing_spatial,
            spatial_maps["T"],
            strict=True,
        )
    )
    mask_hf_spatial = tuple(
        _float32(routing_value * texture)
        for routing_value, texture in zip(
            routing_spatial,
            spatial_maps["T"],
            strict=True,
        )
    )
    return (
        _broadcast_spatial(routing_spatial, channels),
        _broadcast_spatial(mask_lf_spatial, channels),
        _broadcast_spatial(mask_hf_spatial, channels),
    )


def _derive_route_outputs(
    mode: RoutingMode,
    latent_shape: tuple[int, int, int, int],
    routing_observations: RoutingObservations | None,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if mode == ROUTING_CANDIDATE_ID:
        return _derive_routed_tensors(
            routing_observations,
            latent_shape,
        )
    if mode == UNIFORM_CONTROL_CANDIDATE_ID:
        if routing_observations is not None:
            raise ContentRouterError(
                "uniform control result must not retain observations"
            )
        ones = (1.0,) * prod(latent_shape)
        return ones, ones, ones
    raise ContentRouterError("routing result mode is unsupported")


def validate_content_routing_result(
    result: object,
) -> ContentRoutingResult:
    """重算并验证 route 的 shape、partition、digests 与完整 identity。"""

    if type(result) is not ContentRoutingResult:
        raise ContentRouterError("routing binding requires ContentRoutingResult")
    latent_shape = _validate_latent_shape(result.latent_shape)
    if result.mode not in {
        ROUTING_CANDIDATE_ID,
        UNIFORM_CONTROL_CANDIDATE_ID,
    }:
        raise ContentRouterError("routing result mode is unsupported")
    if result.candidate_id != result.mode or result.candidate_ids != (
        ROUTING_KEY_CANDIDATE_ID,
        result.mode,
    ):
        raise ContentRouterError("routing result candidate identity mismatch")

    element_count = prod(latent_shape)
    routing_map = _validate_result_vector(
        result.routing_map,
        element_count,
        "routing_map",
    )
    mask_lf = _validate_result_vector(result.mask_lf, element_count, "mask_lf")
    mask_hf = _validate_result_vector(result.mask_hf, element_count, "mask_hf")
    expected_routing_map, expected_mask_lf, expected_mask_hf = (
        _derive_route_outputs(
            result.mode,
            latent_shape,
            result.routing_observations,
        )
    )
    if result.mode == ROUTING_CANDIDATE_ID:
        expected_observation_digests = _routing_observation_digests(
            result.routing_observations
        )
        if result.observation_digests != expected_observation_digests:
            raise ContentRouterError("routing observation digest identity mismatch")
    elif result.observation_digests != ():
        raise ContentRouterError("uniform control must not retain observation digests")
    if (
        routing_map != expected_routing_map
        or mask_lf != expected_mask_lf
        or mask_hf != expected_mask_hf
    ):
        raise ContentRouterError(
            "routing result does not match the authoritative routing formula"
        )

    expected_routing_map_digest = _digest_float32(routing_map)
    expected_mask_lf_digest = _digest_float32(mask_lf)
    expected_mask_hf_digest = _digest_float32(mask_hf)
    if (
        result.routing_map_digest != expected_routing_map_digest
        or result.mask_lf_digest != expected_mask_lf_digest
        or result.mask_hf_digest != expected_mask_hf_digest
    ):
        raise ContentRouterError("routing result mask digest mismatch")
    expected_route_config_digest = _identity_digest(
        _route_config(result.mode, latent_shape)
    )
    if result.route_config_digest != expected_route_config_digest:
        raise ContentRouterError("routing result config digest mismatch")
    expected_route_identity = _identity_digest(
        {
            "routing_map_digest": expected_routing_map_digest,
            "mask_hf_digest": expected_mask_hf_digest,
            "mask_lf_digest": expected_mask_lf_digest,
            "observation_digests": [
                list(item) for item in result.observation_digests
            ],
            "route_config_digest": expected_route_config_digest,
        }
    )
    if result.route_identity != expected_route_identity:
        raise ContentRouterError("routing result identity mismatch")

    count = len(routing_map)
    if (
        result.mean_routing_map != sum(routing_map) / count
        or result.mean_mask_lf != sum(mask_lf) / count
        or result.mean_mask_hf != sum(mask_hf) / count
    ):
        raise ContentRouterError("routing result summary statistic mismatch")
    return result


def _build_result(
    *,
    mode: RoutingMode,
    latent_shape: tuple[int, int, int, int],
    routing_observations: RoutingObservations | None,
) -> ContentRoutingResult:
    routing_map, mask_lf, mask_hf = _derive_route_outputs(
        mode,
        latent_shape,
        routing_observations,
    )
    observation_digests = (
        _routing_observation_digests(routing_observations)
        if routing_observations is not None
        else ()
    )
    candidate_ids = (ROUTING_KEY_CANDIDATE_ID, mode)
    route_config = _route_config(mode, latent_shape)
    route_config_digest = _identity_digest(route_config)
    routing_map_digest = _digest_float32(routing_map)
    mask_lf_digest = _digest_float32(mask_lf)
    mask_hf_digest = _digest_float32(mask_hf)
    route_identity = _identity_digest(
        {
            "routing_map_digest": routing_map_digest,
            "mask_hf_digest": mask_hf_digest,
            "mask_lf_digest": mask_lf_digest,
            "observation_digests": [list(item) for item in observation_digests],
            "route_config_digest": route_config_digest,
        }
    )
    count = len(routing_map)
    result = ContentRoutingResult(
        candidate_id=mode,
        candidate_ids=candidate_ids,
        mode=mode,
        latent_shape=latent_shape,
        routing_observations=routing_observations,
        routing_map=routing_map,
        mask_lf=mask_lf,
        mask_hf=mask_hf,
        routing_map_digest=routing_map_digest,
        mask_lf_digest=mask_lf_digest,
        mask_hf_digest=mask_hf_digest,
        observation_digests=observation_digests,
        mean_routing_map=sum(routing_map) / count,
        mean_mask_lf=sum(mask_lf) / count,
        mean_mask_hf=sum(mask_hf) / count,
        route_config_digest=route_config_digest,
        route_identity=route_identity,
    )
    return validate_content_routing_result(result)


def content_router(
    latent_shape: Sequence[int],
    *,
    mode: RoutingMode,
    observations: object | None = None,
) -> ContentRoutingResult:
    """执行冻结 routed 公式或完全不读取 observations 的 uniform control。"""

    normalized_shape = _validate_latent_shape(latent_shape)
    if mode == UNIFORM_CONTROL_CANDIDATE_ID:
        return _build_result(
            mode=mode,
            latent_shape=normalized_shape,
            routing_observations=None,
        )
    if mode != ROUTING_CANDIDATE_ID:
        raise ContentRouterError("unsupported routing mode")
    if type(observations) is not RoutingObservations:
        raise ContentRouterError("routing_stqr requires numeric S/T/R/Q observations")
    return _build_result(
        mode=mode,
        latent_shape=normalized_shape,
        routing_observations=observations,
    )
