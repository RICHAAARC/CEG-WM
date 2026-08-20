"""CEG-WM `lf_low_pass` 低频载体模板与 masked unit direction。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, prod, sqrt
from struct import pack, unpack
from typing import Sequence

from main.shared.key_schedule import (
    DEFAULT_CONFIG as DEFAULT_KEY_SCHEDULE_CONFIG,
    DerivedInternalLfDecoyMaterial,
    DerivedWrongKeyMaterial,
    KeyScheduleError,
    derive_internal_lf_decoy_stream,
    derive_wrong_key_stream,
    identify_root_key,
    key_schedule_sha256_counter,
    stable_json_utf8,
)

from .routing import (
    ContentRouterError,
    ContentRoutingResult,
    SemanticTextureRoutingResult,
    validate_content_routing_result,
    validate_semantic_texture_routing_result,
)

LF_CANDIDATE_ID = "lf_low_pass"
KEY_SCHEDULE_CANDIDATE_ID = "key_schedule_sha256_counter"
AVERAGE_KERNEL_SIZE = 5
MULTISCALE_CONTRASTIVE_CANDIDATE_ID = "lf_multiscale_lowpass_contrastive"
SINGLE_SCALE_CONTRASTIVE_CANDIDATE_ID = "lf_five_by_five_lowpass_contrastive"
CONTRASTIVE_LF_CANDIDATE_IDS = (
    MULTISCALE_CONTRASTIVE_CANDIDATE_ID,
    SINGLE_SCALE_CONTRASTIVE_CANDIDATE_ID,
)


class LfCarrierError(ValueError):
    """LF 模板、mask 或单位方向不符合冻结候选。"""


@dataclass(frozen=True, slots=True)
class LfCarrierResult:
    """LF 载体的不可变未遮罩模板、masked direction 与身份。"""

    candidate_id: str
    candidate_ids: tuple[str, ...]
    shape: tuple[int, int, int, int]
    template: tuple[float, ...]
    direction: tuple[float, ...]
    template_digest: str
    direction_digest: str
    mask_digest: str
    route_identity: str | None
    route_config_digest: str | None
    root_key_public_digest: str
    key_role: str
    wrong_key_index: int | None
    key_domain_digest: str
    carrier_config_digest: str


@dataclass(frozen=True, slots=True)
class ContrastiveLfCarrierResult:
    """Registered Stage-A LF template(s) and exact embedding direction."""

    candidate_id: str
    shape: tuple[int, int, int, int]
    scale_five_template: tuple[float, ...]
    scale_nine_template: tuple[float, ...] | None
    direction: tuple[float, ...]
    scale_five_domain_digest: str
    scale_nine_domain_digest: str | None
    scale_five_template_digest: str
    scale_nine_template_digest: str | None
    direction_digest: str
    root_key_public_digest: str
    key_role: str
    wrong_key_index: int | None
    carrier_config_digest: str

    def as_embedding_carrier(self) -> LfCarrierResult:
        return LfCarrierResult(
            candidate_id=self.candidate_id,
            candidate_ids=(KEY_SCHEDULE_CANDIDATE_ID, self.candidate_id),
            shape=self.shape,
            template=self.direction,
            direction=self.direction,
            template_digest=self.direction_digest,
            direction_digest=self.direction_digest,
            mask_digest=_digest((1.0,) * prod(self.shape)),
            route_identity=None,
            route_config_digest=None,
            root_key_public_digest=self.root_key_public_digest,
            key_role=self.key_role,
            wrong_key_index=self.wrong_key_index,
            key_domain_digest=self.scale_five_domain_digest,
            carrier_config_digest=self.carrier_config_digest,
        )


def _float32(value: float) -> float:
    if not isfinite(value):
        raise LfCarrierError("LF numeric value must be finite")
    try:
        rounded = unpack(">f", pack(">f", value))[0]
    except (OverflowError, ValueError) as exc:
        raise LfCarrierError("LF numeric value is outside binary32 range") from exc
    if not isfinite(rounded):
        raise LfCarrierError("LF binary32 value must be finite")
    return rounded


def _validate_shape(shape: Sequence[int]) -> tuple[int, int, int, int]:
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        raise LfCarrierError("LF shape must be [1,C,H,W]")
    normalized = tuple(shape)
    if (
        len(normalized) != 4
        or normalized[0] != 1
        or any(type(size) is not int or size <= 0 for size in normalized)
    ):
        raise LfCarrierError("LF shape must be positive [1,C,H,W]")
    return normalized


def _vector(
    values: Sequence[float],
    expected_size: int,
    role: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise LfCarrierError(f"{role} must be a numeric sequence")
    if len(values) != expected_size:
        raise LfCarrierError(f"{role} length does not match carrier shape")
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LfCarrierError(f"{role} must contain only finite numbers")
        normalized.append(_float32(float(value)))
    return tuple(normalized)


def _l2(values: Sequence[float]) -> float:
    squared_sum = 0.0
    for value in values:
        squared_sum = _float32(squared_sum + _float32(value * value))
    return _float32(sqrt(squared_sum))


def _normalize(values: Sequence[float], role: str) -> tuple[float, ...]:
    norm = _l2(values)
    if norm == 0.0:
        raise LfCarrierError(f"{role} has zero L2 energy")
    return tuple(_float32(value / norm) for value in values)


def _digest(values: Sequence[float]) -> str:
    return sha256(b"".join(pack(">f", value) for value in values)).hexdigest()


def _zero_padded_average_5x5(
    values: Sequence[float],
    shape: tuple[int, int, int, int],
) -> tuple[float, ...]:
    _, channels, height, width = shape
    pooled = [0.0] * len(values)
    for channel in range(channels):
        channel_offset = channel * height * width
        for row in range(height):
            for column in range(width):
                window_sum = 0.0
                for row_offset in range(-2, 3):
                    source_row = row + row_offset
                    for column_offset in range(-2, 3):
                        source_column = column + column_offset
                        if 0 <= source_row < height and 0 <= source_column < width:
                            source_index = (
                                channel_offset + source_row * width + source_column
                            )
                            window_sum = _float32(
                                window_sum + values[source_index]
                            )
                pooled[channel_offset + row * width + column] = _float32(
                    window_sum / 25.0
                )
    return tuple(pooled)


def contrastive_lowpass(
    values: Sequence[float],
    shape: Sequence[int],
    kernel_size: int,
) -> tuple[float, ...]:
    """Per-channel stride-one zero-padded binary32 average, padding included."""

    normalized_shape = _validate_shape(shape)
    normalized_values = _vector(values, prod(normalized_shape), "contrastive LF values")
    if kernel_size not in {5, 9}:
        raise LfCarrierError("contrastive LF kernel is not registered")
    _, channels, height, width = normalized_shape
    radius = kernel_size // 2
    denominator = float(kernel_size * kernel_size)
    pooled = [0.0] * len(normalized_values)
    for channel in range(channels):
        channel_offset = channel * height * width
        for row in range(height):
            for column in range(width):
                total = _float32(0.0)
                for row_offset in range(-radius, radius + 1):
                    source_row = row + row_offset
                    for column_offset in range(-radius, radius + 1):
                        source_column = column + column_offset
                        if 0 <= source_row < height and 0 <= source_column < width:
                            total = _float32(
                                total
                                + normalized_values[
                                    channel_offset + source_row * width + source_column
                                ]
                            )
                pooled[channel_offset + row * width + column] = _float32(
                    total / denominator
                )
    return tuple(pooled)


def _contrastive_domain(
    candidate_id: str,
    kernel_size: int,
    *,
    internal_decoy: bool,
) -> dict[str, object]:
    scale = "five_by_five" if kernel_size == 5 else "nine_by_nine"
    return {
        "candidate_id": candidate_id,
        "operator": (
            f"internal_decoy_carrier_template_lowpass_{scale}"
            if internal_decoy
            else f"carrier_template_lowpass_{scale}"
        ),
        "responsibility_domain": (
            "lf_detector_internal_decoy" if internal_decoy else "lf_carrier"
        ),
        "tensor_role": "base_gaussian",
    }


def contrastive_lf_carrier(
    detection_key: str | DerivedWrongKeyMaterial | DerivedInternalLfDecoyMaterial,
    shape: Sequence[int],
    *,
    candidate_id: str,
) -> ContrastiveLfCarrierResult:
    """Build the independent 5/9 Stage-A LF carrier family."""

    if candidate_id not in CONTRASTIVE_LF_CANDIDATE_IDS:
        raise LfCarrierError("contrastive LF candidate is not registered")
    normalized_shape = _validate_shape(shape)
    if normalized_shape[1] != 16:
        raise LfCarrierError("contrastive LF observation must have 16 channels")
    kernels = (
        (5, 9)
        if candidate_id == MULTISCALE_CONTRASTIVE_CANDIDATE_ID
        else (5,)
    )
    internal = type(detection_key) is DerivedInternalLfDecoyMaterial
    if internal and detection_key.lf_candidate_id != candidate_id:
        raise LfCarrierError("internal LF decoy candidate mismatch")
    streams = []
    try:
        for kernel in kernels:
            domain = _contrastive_domain(candidate_id, kernel, internal_decoy=internal)
            if type(detection_key) is str:
                stream = key_schedule_sha256_counter(
                    detection_key, domain, normalized_shape, distribution="gaussian"
                )
            elif type(detection_key) is DerivedWrongKeyMaterial:
                stream = derive_wrong_key_stream(
                    detection_key, domain, normalized_shape, distribution="gaussian"
                )
            elif type(detection_key) is DerivedInternalLfDecoyMaterial:
                stream = derive_internal_lf_decoy_stream(
                    detection_key, domain, normalized_shape, distribution="gaussian"
                )
            else:
                raise LfCarrierError("contrastive LF key capability is invalid")
            streams.append(stream)
    except KeyScheduleError as exc:
        raise LfCarrierError("contrastive LF key schedule failed") from exc
    templates = tuple(
        _normalize(
            contrastive_lowpass(stream.values, normalized_shape, kernel),
            f"contrastive LF scale {kernel} template",
        )
        for stream, kernel in zip(streams, kernels, strict=True)
    )
    direction = (
        templates[0]
        if len(templates) == 1
        else _normalize(
            tuple(
                _float32(a + b)
                for a, b in zip(templates[0], templates[1], strict=True)
            ),
            "contrastive LF multiscale carrier direction",
        )
    )
    if type(detection_key) is str:
        root_digest = identify_root_key(detection_key).root_key_public_digest
        key_role, wrong_index = "registered", None
    elif type(detection_key) is DerivedWrongKeyMaterial:
        root_digest = detection_key.registered_root_key_public_digest
        key_role, wrong_index = "wrong", detection_key.wrong_key_index
    else:
        root_digest = detection_key.registered_root_key_public_digest
        key_role, wrong_index = "internal_decoy", detection_key.internal_decoy_index
    identity = {
        "candidate_id": candidate_id,
        "carrier_direction": (
            "normalize32(scale_five+scale_nine)"
            if len(templates) == 2
            else "scale_five"
        ),
        "dtype": "binary32",
        "kernel_sizes": list(kernels),
        "lowpass": "per_channel_stride_one_zero_padding_count_include_pad",
        "scale_domain_digests": [stream.domain_digest for stream in streams],
        "scale_template_digests": [_digest(value) for value in templates],
    }
    return ContrastiveLfCarrierResult(
        candidate_id=candidate_id,
        shape=normalized_shape,
        scale_five_template=templates[0],
        scale_nine_template=templates[1] if len(templates) == 2 else None,
        direction=direction,
        scale_five_domain_digest=streams[0].domain_digest,
        scale_nine_domain_digest=(streams[1].domain_digest if len(streams) == 2 else None),
        scale_five_template_digest=_digest(templates[0]),
        scale_nine_template_digest=(_digest(templates[1]) if len(templates) == 2 else None),
        direction_digest=_digest(direction),
        root_key_public_digest=root_digest,
        key_role=key_role,
        wrong_key_index=wrong_index,
        carrier_config_digest=sha256(stable_json_utf8(identity)).hexdigest(),
    )


def _derive_lf_gaussian(
    detection_key: str | DerivedWrongKeyMaterial,
    shape: tuple[int, int, int, int],
):
    domain_fields = {
        "candidate_id": LF_CANDIDATE_ID,
        "operator": "carrier_template",
        "responsibility_domain": "lf_carrier",
        "tensor_role": "base_gaussian",
    }
    try:
        if type(detection_key) is str:
            stream = key_schedule_sha256_counter(
                detection_key,
                domain_fields,
                shape,
                distribution="gaussian",
            )
            root_digest = identify_root_key(
                detection_key
            ).root_key_public_digest
            key_role = "registered"
            wrong_key_index = None
        elif type(detection_key) is DerivedWrongKeyMaterial:
            stream = derive_wrong_key_stream(
                detection_key,
                domain_fields,
                shape,
                distribution="gaussian",
            )
            root_digest = detection_key.registered_root_key_public_digest
            key_role = "wrong"
            wrong_key_index = detection_key.wrong_key_index
        else:
            raise LfCarrierError(
                "detection_key must be root text or DerivedWrongKeyMaterial"
            )
    except KeyScheduleError as exc:
        raise LfCarrierError("LF key stream derivation failed") from exc
    return stream, root_digest, key_role, wrong_key_index


def _carrier_config_identity(
    *,
    mask_digest: str,
    normalized_shape: tuple[int, int, int, int],
    route_identity: str | None,
    route_config_digest: str | None,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "average_kernel_size": AVERAGE_KERNEL_SIZE,
        "candidate_ids": [KEY_SCHEDULE_CANDIDATE_ID, LF_CANDIDATE_ID],
        "center_per_sample": True,
        "count_include_pad": True,
        "dtype": "float32",
        "key_schedule_config_digest": DEFAULT_KEY_SCHEDULE_CONFIG.config_digest,
        "mask_digest": mask_digest,
        "shape": list(normalized_shape),
    }
    if route_identity is not None:
        identity["route_config_digest"] = route_config_digest
        identity["route_identity"] = route_identity
    return identity


def lf_carrier(
    detection_key: str | DerivedWrongKeyMaterial,
    shape: Sequence[int],
    *,
    mask_lf: Sequence[float] | None = None,
    routing_result: ContentRoutingResult | SemanticTextureRoutingResult | None = None,
) -> LfCarrierResult:
    """构造 center-per-sample 的 LF 单位模板与 mask 后单位方向。"""

    normalized_shape = _validate_shape(shape)
    element_count = prod(normalized_shape)
    route_identity: str | None = None
    route_config_digest: str | None = None
    if routing_result is not None:
        if mask_lf is not None:
            raise LfCarrierError(
                "LF carrier accepts either routing_result or mask_lf, not both"
            )
        try:
            if type(routing_result) is ContentRoutingResult:
                validated_route = validate_content_routing_result(routing_result)
            elif type(routing_result) is SemanticTextureRoutingResult:
                validated_route = validate_semantic_texture_routing_result(
                    routing_result
                )
            else:
                raise ContentRouterError("LF routing result type is unsupported")
        except ContentRouterError as exc:
            raise LfCarrierError("LF routing result validation failed") from exc
        if validated_route.latent_shape != normalized_shape:
            raise LfCarrierError("LF routing result shape mismatch")
        mask = validated_route.mask_lf
        route_identity = validated_route.route_identity
        route_config_digest = validated_route.route_config_digest
    elif mask_lf is None:
        mask = (1.0,) * element_count
    else:
        mask = _vector(mask_lf, element_count, "mask_lf")
        if any(value < 0.0 or value > 1.0 for value in mask):
            raise LfCarrierError("mask_lf values must be in [0,1]")

    gaussian_stream, root_digest, key_role, wrong_key_index = _derive_lf_gaussian(
        detection_key,
        normalized_shape,
    )
    low_pass = _zero_padded_average_5x5(
        gaussian_stream.values,
        normalized_shape,
    )
    total = 0.0
    for value in low_pass:
        total = _float32(total + value)
    mean = _float32(total / element_count)
    centered = tuple(_float32(value - mean) for value in low_pass)
    template = _normalize(centered, "LF centered low-pass template")
    masked_template = tuple(
        _float32(value * weight)
        for value, weight in zip(template, mask, strict=True)
    )
    if all(weight == 1.0 for weight in mask):
        direction = template
    else:
        direction = _normalize(masked_template, "LF masked direction")

    mask_digest = _digest(mask)
    carrier_identity = _carrier_config_identity(
        mask_digest=mask_digest,
        normalized_shape=normalized_shape,
        route_identity=route_identity,
        route_config_digest=route_config_digest,
    )
    return LfCarrierResult(
        candidate_id=LF_CANDIDATE_ID,
        candidate_ids=(KEY_SCHEDULE_CANDIDATE_ID, LF_CANDIDATE_ID),
        shape=normalized_shape,
        template=template,
        direction=direction,
        template_digest=_digest(template),
        direction_digest=_digest(direction),
        mask_digest=mask_digest,
        route_identity=route_identity,
        route_config_digest=route_config_digest,
        root_key_public_digest=root_digest,
        key_role=key_role,
        wrong_key_index=wrong_key_index,
        key_domain_digest=gaussian_stream.domain_digest,
        carrier_config_digest=sha256(
            stable_json_utf8(carrier_identity)
        ).hexdigest(),
    )


def validate_lf_carrier_routing_binding(
    carrier: object,
    routing_result: object,
) -> LfCarrierResult:
    """验证 LF carrier 确由给定 route 的 LF mask 生成。"""

    if type(carrier) is not LfCarrierResult:
        raise LfCarrierError("LF routing binding requires LfCarrierResult")
    try:
        route = validate_content_routing_result(routing_result)
    except ContentRouterError as exc:
        raise LfCarrierError("LF routing result validation failed") from exc
    if (
        carrier.candidate_id != LF_CANDIDATE_ID
        or carrier.shape != route.latent_shape
        or carrier.route_identity != route.route_identity
        or carrier.route_config_digest != route.route_config_digest
        or carrier.mask_digest != route.mask_lf_digest
    ):
        raise LfCarrierError("LF carrier route binding mismatch")
    template = _vector(
        carrier.template,
        prod(carrier.shape),
        "LF carrier template",
    )
    direction = _vector(
        carrier.direction,
        prod(carrier.shape),
        "LF carrier direction",
    )
    if (
        _digest(template) != carrier.template_digest
        or _digest(direction) != carrier.direction_digest
    ):
        raise LfCarrierError("LF carrier template or direction digest mismatch")
    masked_template = tuple(
        _float32(value * weight)
        for value, weight in zip(
            template,
            route.mask_lf,
            strict=True,
        )
    )
    expected_direction = (
        template
        if all(weight == 1.0 for weight in route.mask_lf)
        else _normalize(masked_template, "LF routed direction")
    )
    if direction != expected_direction:
        raise LfCarrierError("LF carrier direction does not match routed LF mask")
    expected_config_digest = sha256(
        stable_json_utf8(
            _carrier_config_identity(
                mask_digest=carrier.mask_digest,
                normalized_shape=carrier.shape,
                route_identity=carrier.route_identity,
                route_config_digest=carrier.route_config_digest,
            )
        )
    ).hexdigest()
    if carrier.carrier_config_digest != expected_config_digest:
        raise LfCarrierError("LF carrier config digest does not match route binding")
    return carrier


def validate_lf_carrier_semantic_texture_binding(
    carrier: object,
    routing_result: object,
) -> LfCarrierResult:
    """Verify that the LF direction consumes this exact public soft route."""

    if type(carrier) is not LfCarrierResult:
        raise LfCarrierError("LF soft-route binding requires LfCarrierResult")
    try:
        route = validate_semantic_texture_routing_result(routing_result)
    except ContentRouterError as exc:
        raise LfCarrierError("LF semantic-texture route validation failed") from exc
    if (
        carrier.shape != route.latent_shape
        or carrier.route_identity != route.route_identity
        or carrier.route_config_digest != route.route_config_digest
        or carrier.mask_digest != route.mask_lf_digest
    ):
        raise LfCarrierError("LF semantic-texture route binding mismatch")
    template = _vector(
        carrier.template,
        prod(carrier.shape),
        "LF carrier template",
    )
    expected = _normalize(
        tuple(
            _float32(value * weight)
            for value, weight in zip(template, route.mask_lf, strict=True)
        ),
        "LF semantic-texture direction",
    )
    if carrier.direction != expected or carrier.direction_digest != _digest(expected):
        raise LfCarrierError("LF semantic-texture direction mismatch")
    return carrier
