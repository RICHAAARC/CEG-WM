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
SEMANTIC_TEXTURE_ROUTING_CANDIDATE_ID = "routing_semantic_texture_soft"
SEMANTIC_TEXTURE_EMBEDDING_CANDIDATE_ID = (
    "content_embedding_semantic_texture_soft_lf_hf"
)
SEMANTIC_TEXTURE_CANDIDATE_STATUS = "implemented_not_scientifically_validated"
SEMANTIC_TEXTURE_ROUTE_DISABLED_MODE = "semantic_texture_route_disabled"
SemanticTextureRoutingMode = Literal[
    "routing_semantic_texture_soft",
    "semantic_texture_route_disabled",
]


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


@dataclass(frozen=True, slots=True)
class SemanticTextureRoutingObservations:
    """Public RGB8-derived semantic probability and texture maps."""

    semantic_probability: SpatialRoutingObservation
    texture_complexity: SpatialRoutingObservation


@dataclass(frozen=True, slots=True)
class SemanticTextureRoutingResult:
    """Positive sum-one soft maps for the unpromoted five-candidate family."""

    candidate_id: str
    candidate_status: str
    mode: SemanticTextureRoutingMode
    latent_shape: tuple[int, int, int, int]
    observations: SemanticTextureRoutingObservations | None
    semantic_probability: tuple[float, ...] | None
    texture_complexity: tuple[float, ...] | None
    mask_hf: tuple[float, ...]
    mask_lf: tuple[float, ...]
    semantic_probability_digest: str | None
    texture_complexity_digest: str | None
    semantic_source_identity_digest: str | None
    texture_source_identity_digest: str | None
    mask_hf_digest: str
    mask_lf_digest: str
    route_config_digest: str
    route_identity: str
    route_disabled_reads_observations: bool


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


def _semantic_texture_route_config(
    mode: SemanticTextureRoutingMode,
    latent_shape: tuple[int, int, int, int],
) -> dict[str, object]:
    return {
        "candidate_id": SEMANTIC_TEXTURE_ROUTING_CANDIDATE_ID,
        "candidate_status": SEMANTIC_TEXTURE_CANDIDATE_STATUS,
        "formula": (
            "m_hf=(1+M*T)/(2+M);m_lf=(1+M*(1-T))/(2+M)"
            if mode == SEMANTIC_TEXTURE_ROUTING_CANDIDATE_ID
            else "m_hf=m_lf=0.5_without_reading_M_or_T"
        ),
        "latent_shape": list(latent_shape),
        "mode": mode,
        "route_disabled_reads_observations": False,
    }


def _validate_semantic_texture_observations(
    observations: object,
    spatial_shape: tuple[int, int],
) -> SemanticTextureRoutingObservations:
    if type(observations) is not SemanticTextureRoutingObservations:
        raise ContentRouterError(
            "semantic-texture routing requires immutable public observations"
        )
    semantic = _validate_spatial_observation(
        observations.semantic_probability,
        "semantic probability M",
    )
    texture = _validate_spatial_observation(
        observations.texture_complexity,
        "texture complexity T",
    )
    if semantic.spatial_shape != spatial_shape or texture.spatial_shape != spatial_shape:
        raise ContentRouterError(
            "semantic-texture maps must match the latent spatial shape"
        )
    return observations


def validate_semantic_texture_routing_result(
    result: object,
) -> SemanticTextureRoutingResult:
    """Replay the soft formula, digests, disabled isolation, and identity."""

    if type(result) is not SemanticTextureRoutingResult:
        raise ContentRouterError(
            "semantic-texture binding requires SemanticTextureRoutingResult"
        )
    shape = _validate_latent_shape(result.latent_shape)
    if (
        result.candidate_id != SEMANTIC_TEXTURE_ROUTING_CANDIDATE_ID
        or result.candidate_status != SEMANTIC_TEXTURE_CANDIDATE_STATUS
        or result.mode
        not in {
            SEMANTIC_TEXTURE_ROUTING_CANDIDATE_ID,
            SEMANTIC_TEXTURE_ROUTE_DISABLED_MODE,
        }
        or result.route_disabled_reads_observations is not False
    ):
        raise ContentRouterError("semantic-texture route identity drifted")
    element_count = prod(shape)
    mask_hf = _validate_result_vector(result.mask_hf, element_count, "m_hf")
    mask_lf = _validate_result_vector(result.mask_lf, element_count, "m_lf")
    if any(value <= 0.0 for value in (*mask_hf, *mask_lf)):
        raise ContentRouterError("semantic-texture route maps must be positive")
    for hf_value, lf_value in zip(mask_hf, mask_lf, strict=True):
        if _float32(hf_value + lf_value) != _float32(1.0):
            raise ContentRouterError("semantic-texture route maps must sum to one")
    if result.mode == SEMANTIC_TEXTURE_ROUTE_DISABLED_MODE:
        if (
            result.observations is not None
            or result.semantic_probability is not None
            or result.texture_complexity is not None
            or result.semantic_probability_digest is not None
            or result.texture_complexity_digest is not None
            or result.semantic_source_identity_digest is not None
            or result.texture_source_identity_digest is not None
            or any(value != _float32(0.5) for value in (*mask_hf, *mask_lf))
        ):
            raise ContentRouterError(
                "route-disabled control must retain no M/T state"
            )
    else:
        observations = _validate_semantic_texture_observations(
            result.observations,
            (shape[2], shape[3]),
        )
        semantic = observations.semantic_probability.values
        texture = observations.texture_complexity.values
        expected_hf_spatial: list[float] = []
        expected_lf_spatial: list[float] = []
        for semantic_value, texture_value in zip(semantic, texture, strict=True):
            denominator = _float32(2.0 + semantic_value)
            hf_value = _float32(
                _float32(1.0 + _float32(semantic_value * texture_value))
                / denominator
            )
            # The complementary construction preserves the exact sum-one invariant
            # while remaining the registered algebraic LF expression.
            lf_value = _float32(1.0 - hf_value)
            expected_hf_spatial.append(hf_value)
            expected_lf_spatial.append(lf_value)
        expected_hf = _broadcast_spatial(expected_hf_spatial, shape[1])
        expected_lf = _broadcast_spatial(expected_lf_spatial, shape[1])
        if (
            result.semantic_probability != tuple(semantic)
            or result.texture_complexity != tuple(texture)
            or mask_hf != expected_hf
            or mask_lf != expected_lf
            or result.semantic_probability_digest
            != observations.semantic_probability.values_digest
            or result.texture_complexity_digest
            != observations.texture_complexity.values_digest
            or result.semantic_source_identity_digest
            != observations.semantic_probability.source_identity_digest
            or result.texture_source_identity_digest
            != observations.texture_complexity.source_identity_digest
        ):
            raise ContentRouterError(
                "semantic-texture result does not match public M/T observations"
            )
    if result.mask_hf_digest != _digest_float32(mask_hf) or (
        result.mask_lf_digest != _digest_float32(mask_lf)
    ):
        raise ContentRouterError("semantic-texture mask digest mismatch")
    config_digest = _identity_digest(
        _semantic_texture_route_config(result.mode, shape)
    )
    if result.route_config_digest != config_digest:
        raise ContentRouterError("semantic-texture route config digest mismatch")
    identity = _identity_digest(
        {
            "candidate_status": result.candidate_status,
            "mask_hf_digest": result.mask_hf_digest,
            "mask_lf_digest": result.mask_lf_digest,
            "route_config_digest": config_digest,
            "semantic_probability_digest": result.semantic_probability_digest,
            "semantic_source_identity_digest": result.semantic_source_identity_digest,
            "texture_complexity_digest": result.texture_complexity_digest,
            "texture_source_identity_digest": result.texture_source_identity_digest,
        }
    )
    if result.route_identity != identity:
        raise ContentRouterError("semantic-texture route identity mismatch")
    return result


def semantic_texture_content_router(
    latent_shape: Sequence[int],
    *,
    mode: SemanticTextureRoutingMode,
    observations: object | None = None,
) -> SemanticTextureRoutingResult:
    """Build the soft route or a causal control that never reads ``observations``."""

    shape = _validate_latent_shape(latent_shape)
    config_digest = _identity_digest(_semantic_texture_route_config(mode, shape))
    if mode == SEMANTIC_TEXTURE_ROUTE_DISABLED_MODE:
        mask_hf = (_float32(0.5),) * prod(shape)
        mask_lf = mask_hf
        retained_observations = None
        semantic = None
        texture = None
        semantic_digest = None
        texture_digest = None
        semantic_source_digest = None
        texture_source_digest = None
    elif mode == SEMANTIC_TEXTURE_ROUTING_CANDIDATE_ID:
        retained_observations = _validate_semantic_texture_observations(
            observations,
            (shape[2], shape[3]),
        )
        semantic = retained_observations.semantic_probability.values
        texture = retained_observations.texture_complexity.values
        hf_spatial: list[float] = []
        lf_spatial: list[float] = []
        for semantic_value, texture_value in zip(semantic, texture, strict=True):
            denominator = _float32(2.0 + semantic_value)
            hf_value = _float32(
                _float32(1.0 + _float32(semantic_value * texture_value))
                / denominator
            )
            hf_spatial.append(hf_value)
            lf_spatial.append(_float32(1.0 - hf_value))
        mask_hf = _broadcast_spatial(hf_spatial, shape[1])
        mask_lf = _broadcast_spatial(lf_spatial, shape[1])
        semantic_digest = retained_observations.semantic_probability.values_digest
        texture_digest = retained_observations.texture_complexity.values_digest
        semantic_source_digest = (
            retained_observations.semantic_probability.source_identity_digest
        )
        texture_source_digest = (
            retained_observations.texture_complexity.source_identity_digest
        )
    else:
        raise ContentRouterError("unsupported semantic-texture routing mode")
    mask_hf_digest = _digest_float32(mask_hf)
    mask_lf_digest = _digest_float32(mask_lf)
    route_identity = _identity_digest(
        {
            "candidate_status": SEMANTIC_TEXTURE_CANDIDATE_STATUS,
            "mask_hf_digest": mask_hf_digest,
            "mask_lf_digest": mask_lf_digest,
            "route_config_digest": config_digest,
            "semantic_probability_digest": semantic_digest,
            "semantic_source_identity_digest": semantic_source_digest,
            "texture_complexity_digest": texture_digest,
            "texture_source_identity_digest": texture_source_digest,
        }
    )
    return validate_semantic_texture_routing_result(
        SemanticTextureRoutingResult(
            candidate_id=SEMANTIC_TEXTURE_ROUTING_CANDIDATE_ID,
            candidate_status=SEMANTIC_TEXTURE_CANDIDATE_STATUS,
            mode=mode,
            latent_shape=shape,
            observations=retained_observations,
            semantic_probability=tuple(semantic) if semantic is not None else None,
            texture_complexity=tuple(texture) if texture is not None else None,
            mask_hf=mask_hf,
            mask_lf=mask_lf,
            semantic_probability_digest=semantic_digest,
            texture_complexity_digest=texture_digest,
            semantic_source_identity_digest=semantic_source_digest,
            texture_source_identity_digest=texture_source_digest,
            mask_hf_digest=mask_hf_digest,
            mask_lf_digest=mask_lf_digest,
            route_config_digest=config_digest,
            route_identity=route_identity,
            route_disabled_reads_observations=False,
        )
    )
