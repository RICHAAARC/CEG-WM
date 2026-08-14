"""CEG-WM LF/HF 内容方向组合与共同 target total budget。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite, sqrt
from struct import pack, unpack
from typing import Literal, Protocol, Sequence

from main.shared.key_schedule import stable_json_utf8

from .hf_carrier import (
    HfCarrierError,
    HfCarrierResult,
    validate_hf_carrier_routing_binding,
)
from .lf_carrier import (
    LfCarrierError,
    LfCarrierResult,
    validate_lf_carrier_routing_binding,
)
from .routing import (
    ContentRouterError,
    ContentRoutingResult,
    SalientLocalLfRoutingResult,
    validate_content_routing_result,
    validate_salient_local_lf_routing_result,
)

CONTENT_RELATIVE_L2_NUMERATOR = 3
CONTENT_RELATIVE_L2_DENOMINATOR = 250
FROZEN_MIXING_COEFFICIENTS = (0.25, 0.50, 0.75)
EMBEDDER_CANDIDATE_IDS = (
    "runtime_sd35_flowmatch",
    "hf_sparse_tail",
    "lf_low_pass",
    "routing_stqr",
    "routing_uniform_control",
)
SALIENT_LOCAL_LF_EMBEDDER_CANDIDATE_ID = (
    "content_embedding_global_hf_local_lf"
)
EmbeddingMode = Literal["hf_only", "lf_only", "combined"]
ContentMaterializationIntegrityStatus = Literal[
    "passed",
    "write_disappeared",
]
ContentBudgetStatus = Literal["accepted"]


class ContentEmbedderError(ValueError):
    """内容方向、混合系数或共同 target total budget 无效。"""


@dataclass(frozen=True, slots=True)
class ContentEmbeddingResult:
    """未物化 actual dtype 的理论内容更新与可重建分支诊断。"""

    candidate_ids: tuple[str, ...]
    mode: EmbeddingMode
    shape: tuple[int, int, int, int]
    delta_content: tuple[float, ...]
    delta_content_digest: str
    latent_norm: float
    target_total_norm: float
    target_relative_l2: float
    content_direction: tuple[float, ...]
    active_lf_direction: tuple[float, ...] | None
    active_hf_direction: tuple[float, ...] | None
    mixing_coefficient: float | None
    gamma_lh: float | None
    combined_pre_normalization_norm: float
    target_component_lf: tuple[float, ...] | None
    target_component_hf: tuple[float, ...] | None
    target_component_lf_norm: float | None
    target_component_hf_norm: float | None
    lf_carrier_config_digest: str | None
    hf_carrier_config_digest: str | None
    route_identity: str | None
    route_config_digest: str | None
    embedder_config_digest: str
    embedding_result_identity: str


@dataclass(frozen=True, slots=True)
class SalientLocalLfEmbeddingResult:
    """Fixed global-HF plus local-LF unit direction and causal witness."""

    candidate_id: str
    shape: tuple[int, int, int, int]
    delta_content: tuple[float, ...]
    delta_content_digest: str
    content_direction: tuple[float, ...]
    masked_lf_direction: tuple[float, ...]
    masked_lf_direction_digest: str
    hf_direction_digest: str
    route_identity: str
    route_config_digest: str
    latent_norm: float
    target_total_norm: float
    target_relative_l2: float
    lf_delta_nonzero: bool
    mask_outside_bitwise_zero: bool
    mask_inside_has_energy: bool
    embedder_config_digest: str
    embedding_result_identity: str


@dataclass(frozen=True, slots=True)
class _NominalEmbeddingDerivation:
    gamma_lh: float | None
    combined_pre_normalization_norm: float
    content_direction: tuple[float, ...]
    delta_content: tuple[float, ...]
    target_component_lf: tuple[float, ...] | None
    target_component_hf: tuple[float, ...] | None
    target_component_lf_norm: float | None
    target_component_hf_norm: float | None


@dataclass(frozen=True, slots=True)
class ContentMaterializationObservation:
    """Runtime 对一个 binary32 scale 的 actual-dtype 物化观测。"""

    materialization_scale: float
    baseline_norm: float
    scaled_nominal_delta_digest: str
    delta_content_actual: tuple[float, ...]
    realized_total_l2: float
    integrity_status: ContentMaterializationIntegrityStatus
    deterministic_binary16_replay_passed: bool
    materialization_replay_identity: str


class ContentMaterializer(Protocol):
    """由 runtime 实现、由 content embedder 驱动的物化边界。"""

    def __call__(
        self,
        embedding_result: ContentEmbeddingResult,
        materialization_scale: float,
        /,
    ) -> ContentMaterializationObservation: ...


@dataclass(frozen=True, slots=True)
class ContentMaterializationResult:
    """通过 actual-dtype 完整性门和硬预算门的最终内容写入。"""

    embedding_result: ContentEmbeddingResult
    observation: ContentMaterializationObservation
    content_relative_l2_nominal: float
    content_relative_l2_limit: float
    realized_total_l2: float
    realized_relative_l2: float
    budget_utilization: float
    materialization_scale: float
    attempt_count: int
    integrity_status: Literal["passed"]
    budget_status: ContentBudgetStatus


def _float32(value: float) -> float:
    if not isfinite(value):
        raise ContentEmbedderError("content embedder value must be finite")
    try:
        rounded = unpack(">f", pack(">f", value))[0]
    except (OverflowError, ValueError) as exc:
        raise ContentEmbedderError(
            "content embedder value is outside binary32 range"
        ) from exc
    if not isfinite(rounded):
        raise ContentEmbedderError(
            "content embedder binary32 value must be finite"
        )
    return rounded


def _vector(
    values: Sequence[float],
    size: int,
    role: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContentEmbedderError(f"{role} must be a numeric sequence")
    if len(values) != size:
        raise ContentEmbedderError(f"{role} length does not match carrier shape")
    converted = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContentEmbedderError(f"{role} must contain only finite numbers")
        converted.append(_float32(float(value)))
    return tuple(converted)


def _l2_norm(values: Sequence[float]) -> float:
    squared_sum = 0.0
    for value in values:
        squared_sum = _float32(squared_sum + _float32(value * value))
    return _float32(sqrt(squared_sum))


def _digest(values: Sequence[float]) -> str:
    return sha256(b"".join(pack(">f", value) for value in values)).hexdigest()


def _optional_text(value: str | None) -> str:
    return "<none>" if value is None else value


def _optional_float32_bits(value: float | None) -> str:
    return "<none>" if value is None else pack(">f", _float32(value)).hex()


def _optional_vector_digest(
    values: tuple[float, ...] | None,
) -> str:
    return "<none>" if values is None else _digest(values)


def _embedding_result_identity(result: ContentEmbeddingResult) -> str:
    identity = (
        "content-embedding-result-v1\0"
        f"{','.join(result.candidate_ids)}\0"
        f"{result.mode}\0"
        f"{','.join(str(value) for value in result.shape)}\0"
        f"{result.delta_content_digest}\0"
        f"{pack('>f', _float32(result.latent_norm)).hex()}\0"
        f"{pack('>f', _float32(result.target_total_norm)).hex()}\0"
        f"{pack('>f', _float32(result.target_relative_l2)).hex()}\0"
        f"{_digest(result.content_direction)}\0"
        f"{_optional_vector_digest(result.active_lf_direction)}\0"
        f"{_optional_vector_digest(result.active_hf_direction)}\0"
        f"{_optional_float32_bits(result.mixing_coefficient)}\0"
        f"{_optional_float32_bits(result.gamma_lh)}\0"
        f"{pack('>f', _float32(result.combined_pre_normalization_norm)).hex()}\0"
        f"{_optional_vector_digest(result.target_component_lf)}\0"
        f"{_optional_vector_digest(result.target_component_hf)}\0"
        f"{_optional_float32_bits(result.target_component_lf_norm)}\0"
        f"{_optional_float32_bits(result.target_component_hf_norm)}\0"
        f"{_optional_text(result.lf_carrier_config_digest)}\0"
        f"{_optional_text(result.hf_carrier_config_digest)}\0"
        f"{_optional_text(result.route_identity)}\0"
        f"{_optional_text(result.route_config_digest)}\0"
        f"{result.embedder_config_digest}"
    )
    return sha256(identity.encode("ascii")).hexdigest()


def _embedder_config_digest_from_fields(
    *,
    shape: tuple[int, int, int, int],
    mode: EmbeddingMode,
    mixing_coefficient: float | None,
    lf_carrier_config_digest: str | None,
    hf_carrier_config_digest: str | None,
    route_identity: str | None,
    route_config_digest: str | None,
) -> str:
    identity = {
        "candidate_ids": list(EMBEDDER_CANDIDATE_IDS),
        "hf_carrier_config_digest": hf_carrier_config_digest,
        "lf_carrier_config_digest": lf_carrier_config_digest,
        "mixing_coefficient": (
            int(mixing_coefficient * 100)
            if mixing_coefficient is not None
            else None
        ),
        "mixing_coefficient_denominator": (
            100 if mixing_coefficient is not None else None
        ),
        "mode": mode,
        "rho_content_denominator": CONTENT_RELATIVE_L2_DENOMINATOR,
        "rho_content_numerator": CONTENT_RELATIVE_L2_NUMERATOR,
        "shape": list(shape),
    }
    if route_identity is not None:
        identity["route_config_digest"] = route_config_digest
        identity["route_identity"] = route_identity
    return sha256(stable_json_utf8(identity)).hexdigest()


def _expected_embedder_config_digest(
    result: ContentEmbeddingResult,
) -> str:
    return _embedder_config_digest_from_fields(
        shape=result.shape,
        mode=result.mode,
        mixing_coefficient=result.mixing_coefficient,
        lf_carrier_config_digest=result.lf_carrier_config_digest,
        hf_carrier_config_digest=result.hf_carrier_config_digest,
        route_identity=result.route_identity,
        route_config_digest=result.route_config_digest,
    )


def _validate_content_embedding_result(
    result: ContentEmbeddingResult,
) -> ContentEmbeddingResult:
    if type(result) is not ContentEmbeddingResult:
        raise ContentEmbedderError(
            "content embedding result has an invalid type"
        )
    if result.candidate_ids != EMBEDDER_CANDIDATE_IDS:
        raise ContentEmbedderError(
            "content embedding candidate identity drifted"
        )
    if (
        type(result.shape) is not tuple
        or len(result.shape) != 4
        or any(type(value) is not int or value <= 0 for value in result.shape)
    ):
        raise ContentEmbedderError("content embedding shape is invalid")
    expected_size = 1
    for value in result.shape:
        expected_size *= value
    delta = _vector(result.delta_content, expected_size, "delta_content")
    if _digest(delta) != result.delta_content_digest:
        raise ContentEmbedderError("delta_content digest mismatch")
    latent_norm = _float32(result.latent_norm)
    if latent_norm <= 0.0:
        raise ContentEmbedderError("content embedding latent norm must be positive")
    relative_l2 = _float32(result.target_relative_l2)
    if not _same_float32(relative_l2, _content_relative_l2()):
        raise ContentEmbedderError(
            "content embedding nominal relative L2 drifted"
        )
    expected_target = _float32(relative_l2 * latent_norm)
    if not _same_float32(
        _float32(result.target_total_norm),
        expected_target,
    ):
        raise ContentEmbedderError(
            "content embedding target total norm drifted"
        )
    if result.mode not in ("hf_only", "lf_only", "combined"):
        raise ContentEmbedderError("content embedding mode is invalid")
    content_direction = _vector(
        result.content_direction,
        expected_size,
        "content_direction",
    )
    lf_direction = (
        _vector(
            result.active_lf_direction,
            expected_size,
            "active_lf_direction",
        )
        if result.active_lf_direction is not None
        else None
    )
    hf_direction = (
        _vector(
            result.active_hf_direction,
            expected_size,
            "active_hf_direction",
        )
        if result.active_hf_direction is not None
        else None
    )
    if result.mode == "hf_only":
        consistent = (
            lf_direction is None
            and hf_direction is not None
            and result.mixing_coefficient is None
            and result.gamma_lh is None
            and result.target_component_lf is None
            and result.target_component_lf_norm is None
            and result.lf_carrier_config_digest is None
            and result.target_component_hf is not None
            and result.target_component_hf_norm is not None
            and result.hf_carrier_config_digest is not None
            and result.route_identity is None
            and result.route_config_digest is None
        )
    elif result.mode == "lf_only":
        consistent = (
            lf_direction is not None
            and hf_direction is None
            and result.mixing_coefficient is None
            and result.gamma_lh is None
            and result.target_component_hf is None
            and result.target_component_hf_norm is None
            and result.hf_carrier_config_digest is None
            and result.target_component_lf is not None
            and result.target_component_lf_norm is not None
            and result.lf_carrier_config_digest is not None
            and result.route_identity is None
            and result.route_config_digest is None
        )
    else:
        consistent = (
            lf_direction is not None
            and hf_direction is not None
            and result.mixing_coefficient in FROZEN_MIXING_COEFFICIENTS
            and result.gamma_lh is not None
            and result.target_component_lf is not None
            and result.target_component_hf is not None
            and result.target_component_lf_norm is not None
            and result.target_component_hf_norm is not None
            and result.lf_carrier_config_digest is not None
            and result.hf_carrier_config_digest is not None
            and result.route_identity is not None
            and result.route_config_digest is not None
        )
    if not consistent:
        raise ContentEmbedderError(
            "content embedding component fields do not match mode"
        )
    expected = _derive_nominal_embedding(
        mode=result.mode,
        lf_direction=lf_direction,
        hf_direction=hf_direction,
        mixing_coefficient=result.mixing_coefficient,
        target_total_norm=expected_target,
    )
    if content_direction != expected.content_direction:
        raise ContentEmbedderError(
            "content direction failed frozen formula replay"
        )
    direction_tolerance = 2e-5 if result.mode == "combined" else 1e-5
    if abs(_l2_norm(content_direction) - 1.0) > direction_tolerance:
        raise ContentEmbedderError(
            "content direction failed frozen unit-L2 structure"
        )
    if result.gamma_lh != expected.gamma_lh:
        raise ContentEmbedderError(
            "content direction inner product identity drifted"
        )
    if (
        result.combined_pre_normalization_norm
        != expected.combined_pre_normalization_norm
    ):
        raise ContentEmbedderError(
            "content pre-normalization norm identity drifted"
        )
    if delta != expected.delta_content:
        raise ContentEmbedderError(
            "delta_content failed frozen nominal formula replay"
        )
    if abs(_l2_norm(delta) / expected_target - 1.0) > direction_tolerance:
        raise ContentEmbedderError(
            "delta_content failed frozen nominal unit-direction structure"
        )
    for component, expected_component, norm, expected_norm, role in (
        (
            result.target_component_lf,
            expected.target_component_lf,
            result.target_component_lf_norm,
            expected.target_component_lf_norm,
            "target_component_lf",
        ),
        (
            result.target_component_hf,
            expected.target_component_hf,
            result.target_component_hf_norm,
            expected.target_component_hf_norm,
            "target_component_hf",
        ),
    ):
        if component is None and expected_component is None:
            continue
        if component is None or expected_component is None:
            raise ContentEmbedderError(f"{role} presence identity drifted")
        vector = _vector(component, expected_size, role)
        if vector != expected_component:
            raise ContentEmbedderError(f"{role} formula identity drifted")
        if not _same_float32(_float32(expected_norm), _float32(norm)):
            raise ContentEmbedderError(f"{role} norm identity drifted")
    if result.embedder_config_digest != _expected_embedder_config_digest(result):
        raise ContentEmbedderError(
            "content embedder configuration digest mismatch"
        )
    if result.embedding_result_identity != _embedding_result_identity(result):
        raise ContentEmbedderError(
            "content embedding result identity mismatch"
        )
    return result


def _same_float32(left: float, right: float) -> bool:
    return pack(">f", left) == pack(">f", right)


def _content_relative_l2() -> float:
    return _float32(
        CONTENT_RELATIVE_L2_NUMERATOR / CONTENT_RELATIVE_L2_DENOMINATOR
    )


def _actual_budget_limit_norm(baseline_norm: float) -> float:
    baseline = _float32(baseline_norm)
    if baseline <= 0.0:
        raise ContentEmbedderError(
            "actual callback baseline L2 must be positive"
        )
    return _float32(baseline * _content_relative_l2())


def content_actual_budget_accepts(
    baseline_norm: float,
    realized_total_l2: float,
) -> bool:
    """以冻结 binary32 直接比较判定 actual content 总扰动上限。"""

    realized = _float32(realized_total_l2)
    if realized < 0.0:
        raise ContentEmbedderError(
            "actual content realized total L2 must be nonnegative"
        )
    return realized <= _actual_budget_limit_norm(baseline_norm)


def scale_content_delta_binary32(
    embedding_result: ContentEmbeddingResult,
    materialization_scale: float,
) -> tuple[float, ...]:
    """逐项执行 `f32(delta_content_i * f32(scale))`。"""

    _validate_content_embedding_result(embedding_result)
    scale = _float32(materialization_scale)
    if not 0.0 < scale <= 1.0:
        raise ContentEmbedderError(
            "content materialization scale must be in (0,1]"
        )
    return tuple(
        _float32(value * scale)
        for value in embedding_result.delta_content
    )


def content_materialization_replay_identity(
    embedding_result: ContentEmbeddingResult,
    *,
    materialization_scale: float,
    scaled_nominal_delta_digest: str,
    baseline_norm: float,
    delta_content_actual: Sequence[float],
    realized_total_l2: float,
    integrity_status: ContentMaterializationIntegrityStatus,
) -> str:
    """绑定 scale、scaled nominal 输入与 actual-delta 重放结果。"""

    scale = _float32(materialization_scale)
    baseline = _float32(baseline_norm)
    realized = _float32(realized_total_l2)
    actual = _vector(
        delta_content_actual,
        len(embedding_result.delta_content),
        "delta_content_actual",
    )
    if (
        not isinstance(scaled_nominal_delta_digest, str)
        or len(scaled_nominal_delta_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in scaled_nominal_delta_digest
        )
    ):
        raise ContentEmbedderError(
            "scaled nominal delta digest is invalid"
        )
    if integrity_status not in ("passed", "write_disappeared"):
        raise ContentEmbedderError(
            "content materialization integrity status is invalid"
        )
    identity = (
        "content-materialization-replay-v1\0"
        f"{embedding_result.embedder_config_digest}\0"
        f"{pack('>f', scale).hex()}\0"
        f"{scaled_nominal_delta_digest}\0"
        f"{pack('>f', baseline).hex()}\0"
        f"{_digest(actual)}\0"
        f"{pack('>f', realized).hex()}\0"
        f"{integrity_status}"
    )
    return sha256(identity.encode("ascii")).hexdigest()


def _materialization_observation(
    embedding_result: ContentEmbeddingResult,
    materializer: ContentMaterializer,
    materialization_scale: float,
) -> ContentMaterializationObservation:
    scale = _float32(materialization_scale)
    if not 0.0 < scale <= 1.0:
        raise ContentEmbedderError(
            "content materialization scale must be in (0,1]"
        )
    try:
        observation = materializer(embedding_result, scale)
    except ContentEmbedderError:
        raise
    except Exception as exc:
        raise ContentEmbedderError(
            "content actual-dtype materialization failed"
        ) from exc
    if type(observation) is not ContentMaterializationObservation:
        raise ContentEmbedderError(
            "content materializer returned an invalid observation"
        )
    observed_scale = _float32(observation.materialization_scale)
    if not _same_float32(observed_scale, scale):
        raise ContentEmbedderError(
            "content materialization scale identity drifted"
        )
    scaled_nominal_delta = scale_content_delta_binary32(
        embedding_result,
        scale,
    )
    scaled_nominal_delta_digest = _digest(scaled_nominal_delta)
    if observation.scaled_nominal_delta_digest != scaled_nominal_delta_digest:
        raise ContentEmbedderError(
            "scaled nominal delta digest drifted"
        )
    baseline_norm = _float32(observation.baseline_norm)
    if baseline_norm <= 0.0:
        raise ContentEmbedderError(
            "actual callback baseline L2 must be positive"
        )
    if not _same_float32(baseline_norm, embedding_result.latent_norm):
        raise ContentEmbedderError(
            "actual callback baseline norm does not match embedder input"
        )
    delta_actual = _vector(
        observation.delta_content_actual,
        len(embedding_result.delta_content),
        "delta_content_actual",
    )
    realized_total_l2 = _float32(observation.realized_total_l2)
    replayed_total_l2 = _l2_norm(delta_actual)
    if not _same_float32(realized_total_l2, replayed_total_l2):
        raise ContentEmbedderError(
            "actual content norm failed row-major binary32 replay"
        )
    if observation.deterministic_binary16_replay_passed is not True:
        raise ContentEmbedderError(
            "actual content write failed deterministic binary16 replay"
        )
    replay_identity = observation.materialization_replay_identity
    if not isinstance(replay_identity, str):
        raise ContentEmbedderError(
            "content materialization replay identity is invalid"
        )
    if observation.integrity_status == "write_disappeared":
        if realized_total_l2 != 0.0:
            raise ContentEmbedderError(
                "write-disappeared observation has nonzero actual delta"
            )
    elif observation.integrity_status == "passed":
        if realized_total_l2 == 0.0:
            raise ContentEmbedderError(
                "passed observation has zero actual content delta"
            )
    else:
        raise ContentEmbedderError(
            "content materialization integrity status is invalid"
        )
    expected_replay_identity = content_materialization_replay_identity(
        embedding_result,
        materialization_scale=scale,
        scaled_nominal_delta_digest=scaled_nominal_delta_digest,
        baseline_norm=baseline_norm,
        delta_content_actual=delta_actual,
        realized_total_l2=realized_total_l2,
        integrity_status=observation.integrity_status,
    )
    if replay_identity != expected_replay_identity:
        raise ContentEmbedderError(
            "content materialization replay identity mismatch"
        )
    return ContentMaterializationObservation(
        materialization_scale=scale,
        baseline_norm=baseline_norm,
        scaled_nominal_delta_digest=scaled_nominal_delta_digest,
        delta_content_actual=delta_actual,
        realized_total_l2=realized_total_l2,
        integrity_status=observation.integrity_status,
        deterministic_binary16_replay_passed=True,
        materialization_replay_identity=replay_identity,
    )


def _binary32_midpoint(lower: float, upper: float) -> float:
    return _float32(_float32(_float32(lower) + _float32(upper)) * 0.5)


def _accepted_materialization_result(
    embedding_result: ContentEmbeddingResult,
    observation: ContentMaterializationObservation,
    attempt_count: int,
) -> ContentMaterializationResult:
    limit_norm = _actual_budget_limit_norm(observation.baseline_norm)
    realized_relative_l2 = _float32(
        observation.realized_total_l2 / observation.baseline_norm
    )
    budget_utilization = _float32(
        observation.realized_total_l2 / limit_norm
    )
    return ContentMaterializationResult(
        embedding_result=embedding_result,
        observation=observation,
        content_relative_l2_nominal=_content_relative_l2(),
        content_relative_l2_limit=_content_relative_l2(),
        realized_total_l2=observation.realized_total_l2,
        realized_relative_l2=realized_relative_l2,
        budget_utilization=budget_utilization,
        materialization_scale=observation.materialization_scale,
        attempt_count=attempt_count,
        integrity_status="passed",
        budget_status="accepted",
    )


def reconcile_content_materialization_budget(
    embedding_result: ContentEmbeddingResult,
    materializer: ContentMaterializer,
) -> ContentMaterializationResult:
    """选择满足 actual-dtype 硬上限的最大非零 binary32 scale。"""

    _validate_content_embedding_result(embedding_result)
    if not callable(materializer):
        raise ContentEmbedderError("content materializer must be callable")

    attempt_count = 1
    full_scale = _materialization_observation(
        embedding_result,
        materializer,
        _float32(1.0),
    )
    if full_scale.integrity_status == "write_disappeared":
        raise ContentEmbedderError(
            "full-scale actual-dtype content write disappeared"
        )
    if content_actual_budget_accepts(
        full_scale.baseline_norm,
        full_scale.realized_total_l2,
    ):
        return _accepted_materialization_result(
            embedding_result,
            full_scale,
            attempt_count,
        )

    lower = _float32(0.0)
    upper = _float32(1.0)
    greatest_nonzero_feasible_observation: (
        ContentMaterializationObservation | None
    ) = None
    while True:
        midpoint = _binary32_midpoint(lower, upper)
        if _same_float32(midpoint, lower) or _same_float32(midpoint, upper):
            break
        attempt_count += 1
        observation = _materialization_observation(
            embedding_result,
            materializer,
            midpoint,
        )
        if observation.integrity_status == "write_disappeared":
            lower = midpoint
            continue
        if content_actual_budget_accepts(
            observation.baseline_norm,
            observation.realized_total_l2,
        ):
            lower = midpoint
            greatest_nonzero_feasible_observation = observation
        else:
            upper = midpoint

    if greatest_nonzero_feasible_observation is None:
        raise ContentEmbedderError(
            "no nonzero actual-dtype content write satisfies the hard budget"
        )
    return _accepted_materialization_result(
        embedding_result,
        greatest_nonzero_feasible_observation,
        attempt_count,
    )


def _validated_hf_direction(
    carrier: object,
) -> tuple[HfCarrierResult, tuple[float, ...]]:
    if type(carrier) is not HfCarrierResult:
        raise ContentEmbedderError("HF branch requires HfCarrierResult")
    direction = _vector(
        carrier.direction,
        len(carrier.direction),
        "hf_direction",
    )
    if _digest(direction) != carrier.direction_digest:
        raise ContentEmbedderError("HF carrier direction digest mismatch")
    norm = _l2_norm(direction)
    if norm == 0.0:
        raise ContentEmbedderError("active HF direction has zero L2 energy")
    if abs(norm - 1.0) > 1e-5:
        raise ContentEmbedderError("active HF direction must be unit L2")
    return carrier, direction


def _validated_lf_direction(
    carrier: object,
) -> tuple[LfCarrierResult, tuple[float, ...]]:
    if type(carrier) is not LfCarrierResult:
        raise ContentEmbedderError("LF branch requires LfCarrierResult")
    direction = _vector(
        carrier.direction,
        len(carrier.direction),
        "lf_direction",
    )
    if _digest(direction) != carrier.direction_digest:
        raise ContentEmbedderError("LF carrier direction digest mismatch")
    norm = _l2_norm(direction)
    if norm == 0.0:
        raise ContentEmbedderError("active LF direction has zero L2 energy")
    if abs(norm - 1.0) > 1e-5:
        raise ContentEmbedderError("active LF direction must be unit L2")
    return carrier, direction


def _target_component(
    scale: float,
    direction: Sequence[float],
) -> tuple[float, ...]:
    return tuple(_float32(scale * value) for value in direction)


def _derive_nominal_embedding(
    *,
    mode: EmbeddingMode,
    lf_direction: tuple[float, ...] | None,
    hf_direction: tuple[float, ...] | None,
    mixing_coefficient: float | None,
    target_total_norm: float,
) -> _NominalEmbeddingDerivation:
    if mode == "hf_only":
        if (
            lf_direction is not None
            or hf_direction is None
            or mixing_coefficient is not None
        ):
            raise ContentEmbedderError(
                "HF-only nominal derivation inputs are inconsistent"
            )
        gamma_lh = None
        combined_norm = 1.0
        content_direction = hf_direction
        lf_scale = None
        hf_scale = 1.0
    elif mode == "lf_only":
        if (
            lf_direction is None
            or hf_direction is not None
            or mixing_coefficient is not None
        ):
            raise ContentEmbedderError(
                "LF-only nominal derivation inputs are inconsistent"
            )
        gamma_lh = None
        combined_norm = 1.0
        content_direction = lf_direction
        lf_scale = 1.0
        hf_scale = None
    elif mode == "combined":
        if (
            lf_direction is None
            or hf_direction is None
            or mixing_coefficient not in FROZEN_MIXING_COEFFICIENTS
        ):
            raise ContentEmbedderError(
                "combined nominal derivation inputs are inconsistent"
            )
        gamma_accumulator = 0.0
        for lf_value, hf_value in zip(
            lf_direction,
            hf_direction,
            strict=True,
        ):
            gamma_accumulator += lf_value * hf_value
        gamma_lh = gamma_accumulator
        if not isfinite(gamma_lh) or not -1.00002 <= gamma_lh <= 1.00002:
            raise ContentEmbedderError(
                "LF/HF direction inner product is invalid"
            )
        one_minus_a = 1.0 - mixing_coefficient
        c_squared = (
            mixing_coefficient * mixing_coefficient
            + one_minus_a * one_minus_a
            + 2.0 * mixing_coefficient * one_minus_a * gamma_lh
        )
        if not isfinite(c_squared) or c_squared <= 0.0:
            raise ContentEmbedderError(
                "combined content direction has zero or invalid norm"
            )
        combined_norm = sqrt(c_squared)
        combined_raw = tuple(
            _float32(
                mixing_coefficient * lf_value
                + one_minus_a * hf_value
            )
            for lf_value, hf_value in zip(
                lf_direction,
                hf_direction,
                strict=True,
            )
        )
        if _l2_norm(combined_raw) == 0.0:
            raise ContentEmbedderError(
                "combined content direction vanished in float32"
            )
        content_direction = tuple(
            _float32(value / combined_norm)
            for value in combined_raw
        )
        if abs(_l2_norm(content_direction) - 1.0) > 2e-5:
            raise ContentEmbedderError(
                "combined content direction failed unit normalization"
            )
        lf_scale = mixing_coefficient / combined_norm
        hf_scale = one_minus_a / combined_norm
    else:
        raise ContentEmbedderError("content embedding mode is invalid")

    delta_content = _target_component(
        target_total_norm,
        content_direction,
    )
    if _l2_norm(delta_content) == 0.0:
        raise ContentEmbedderError("delta_content has zero L2 energy")
    target_component_lf = (
        _target_component(target_total_norm * lf_scale, lf_direction)
        if lf_scale is not None and lf_direction is not None
        else None
    )
    target_component_hf = (
        _target_component(target_total_norm * hf_scale, hf_direction)
        if hf_scale is not None and hf_direction is not None
        else None
    )
    return _NominalEmbeddingDerivation(
        gamma_lh=gamma_lh,
        combined_pre_normalization_norm=combined_norm,
        content_direction=content_direction,
        delta_content=delta_content,
        target_component_lf=target_component_lf,
        target_component_hf=target_component_hf,
        target_component_lf_norm=(
            _l2_norm(target_component_lf)
            if target_component_lf is not None
            else None
        ),
        target_component_hf_norm=(
            _l2_norm(target_component_hf)
            if target_component_hf is not None
            else None
        ),
    )


def content_embedder(
    latent_values: Sequence[float],
    hf_carrier_result: HfCarrierResult | None = None,
    *,
    lf_carrier_result: LfCarrierResult | None = None,
    mixing_coefficient: float | None = None,
    routing_result: ContentRoutingResult | None = None,
) -> ContentEmbeddingResult:
    """产生 HF-only、LF-only 或冻结 `a` 的 combined 理论更新。"""

    if hf_carrier_result is None and lf_carrier_result is None:
        raise ContentEmbedderError("at least one content carrier must be active")

    hf_carrier_value: HfCarrierResult | None = None
    hf_direction: tuple[float, ...] | None = None
    lf_carrier_value: LfCarrierResult | None = None
    lf_direction: tuple[float, ...] | None = None
    if hf_carrier_result is not None:
        hf_carrier_value, hf_direction = _validated_hf_direction(
            hf_carrier_result
        )
    if lf_carrier_result is not None:
        lf_carrier_value, lf_direction = _validated_lf_direction(
            lf_carrier_result
        )

    reference_shape = (
        hf_carrier_value.shape
        if hf_carrier_value is not None
        else lf_carrier_value.shape
    )
    element_count = len(
        hf_direction if hf_direction is not None else lf_direction
    )
    if (
        hf_carrier_value is not None
        and lf_carrier_value is not None
        and (
            hf_carrier_value.shape != lf_carrier_value.shape
            or hf_carrier_value.root_key_public_digest
            != lf_carrier_value.root_key_public_digest
            or hf_carrier_value.key_role != lf_carrier_value.key_role
            or hf_carrier_value.wrong_key_index
            != lf_carrier_value.wrong_key_index
        )
    ):
        raise ContentEmbedderError(
            "active LF/HF carriers must share shape and key semantics"
        )

    latent = _vector(latent_values, element_count, "latent_values")
    latent_norm = _l2_norm(latent)
    if latent_norm == 0.0:
        raise ContentEmbedderError("callback latent has zero L2 energy")

    mixing_value: float | None
    if lf_direction is not None and hf_direction is not None:
        if routing_result is None:
            raise ContentEmbedderError(
                "combined mode requires one validated ContentRoutingResult"
            )
        try:
            route = validate_content_routing_result(routing_result)
            validate_hf_carrier_routing_binding(
                hf_carrier_value,
                route,
            )
            validate_lf_carrier_routing_binding(
                lf_carrier_value,
                route,
            )
        except (ContentRouterError, HfCarrierError, LfCarrierError) as exc:
            raise ContentEmbedderError(
                "combined carrier routing binding validation failed"
            ) from exc
        if route.latent_shape != reference_shape:
            raise ContentEmbedderError("combined routing result shape mismatch")
        route_identity = route.route_identity
        route_config_digest = route.route_config_digest
        if (
            isinstance(mixing_coefficient, bool)
            or not isinstance(mixing_coefficient, (int, float))
            or float(mixing_coefficient) not in FROZEN_MIXING_COEFFICIENTS
        ):
            raise ContentEmbedderError(
                "combined mode requires a in {0.25,0.50,0.75}"
            )
        mode: EmbeddingMode = "combined"
        mixing_value = float(mixing_coefficient)
    elif hf_direction is not None:
        if routing_result is not None:
            raise ContentEmbedderError(
                "HF-only mode does not consume a combined routing result"
            )
        if mixing_coefficient is not None:
            raise ContentEmbedderError(
                "HF-only mode does not accept a mixing coefficient"
            )
        mode = "hf_only"
        mixing_value = None
        route_identity = None
        route_config_digest = None
    else:
        if routing_result is not None:
            raise ContentEmbedderError(
                "LF-only mode does not consume a combined routing result"
            )
        if mixing_coefficient is not None:
            raise ContentEmbedderError(
                "LF-only mode does not accept a mixing coefficient"
            )
        mode = "lf_only"
        mixing_value = None
        route_identity = None
        route_config_digest = None

    target_relative_l2 = _float32(
        CONTENT_RELATIVE_L2_NUMERATOR / CONTENT_RELATIVE_L2_DENOMINATOR
    )
    target_total_norm = _float32(target_relative_l2 * latent_norm)
    if target_total_norm == 0.0:
        raise ContentEmbedderError(
            "target content update vanished in float32"
        )
    nominal = _derive_nominal_embedding(
        mode=mode,
        lf_direction=lf_direction,
        hf_direction=hf_direction,
        mixing_coefficient=mixing_value,
        target_total_norm=target_total_norm,
    )
    lf_carrier_config_digest = (
        lf_carrier_value.carrier_config_digest
        if lf_carrier_value is not None
        else None
    )
    hf_carrier_config_digest = (
        hf_carrier_value.carrier_config_digest
        if hf_carrier_value is not None
        else None
    )
    embedder_config_digest = _embedder_config_digest_from_fields(
        shape=reference_shape,
        mode=mode,
        mixing_coefficient=mixing_value,
        lf_carrier_config_digest=lf_carrier_config_digest,
        hf_carrier_config_digest=hf_carrier_config_digest,
        route_identity=route_identity,
        route_config_digest=route_config_digest,
    )
    result = ContentEmbeddingResult(
        candidate_ids=EMBEDDER_CANDIDATE_IDS,
        mode=mode,
        shape=reference_shape,
        delta_content=nominal.delta_content,
        delta_content_digest=_digest(nominal.delta_content),
        latent_norm=latent_norm,
        target_total_norm=target_total_norm,
        target_relative_l2=target_relative_l2,
        content_direction=nominal.content_direction,
        active_lf_direction=lf_direction,
        active_hf_direction=hf_direction,
        mixing_coefficient=mixing_value,
        gamma_lh=nominal.gamma_lh,
        combined_pre_normalization_norm=(
            nominal.combined_pre_normalization_norm
        ),
        target_component_lf=nominal.target_component_lf,
        target_component_hf=nominal.target_component_hf,
        target_component_lf_norm=nominal.target_component_lf_norm,
        target_component_hf_norm=nominal.target_component_hf_norm,
        lf_carrier_config_digest=lf_carrier_config_digest,
        hf_carrier_config_digest=hf_carrier_config_digest,
        route_identity=route_identity,
        route_config_digest=route_config_digest,
        embedder_config_digest=embedder_config_digest,
        embedding_result_identity="",
    )
    result = replace(
        result,
        embedding_result_identity=_embedding_result_identity(result),
    )
    return _validate_content_embedding_result(result)


def salient_local_lf_content_embedder(
    latent_values: Sequence[float],
    hf_carrier_result: HfCarrierResult,
    lf_carrier_result: LfCarrierResult,
    routing_result: SalientLocalLfRoutingResult,
) -> SalientLocalLfEmbeddingResult:
    """Write normalize(normalize(T_hf)+normalize(M_embed*T_lf)) at 3/250."""

    hf_carrier_value, hf_direction = _validated_hf_direction(hf_carrier_result)
    lf_carrier_value, lf_direction = _validated_lf_direction(lf_carrier_result)
    try:
        route = validate_salient_local_lf_routing_result(routing_result)
    except ContentRouterError as exc:
        raise ContentEmbedderError("salient LF routing validation failed") from exc
    if (
        route.observation_role
        != "embed_nonterminal_content_write_callback_latent_rgb8"
    ):
        raise ContentEmbedderError("content writing requires the embed-side saliency mask")
    if (
        hf_carrier_value.shape != lf_carrier_value.shape
        or hf_carrier_value.shape != route.latent_shape
        or hf_carrier_value.root_key_public_digest
        != lf_carrier_value.root_key_public_digest
        or hf_carrier_value.key_role != lf_carrier_value.key_role
        or hf_carrier_value.wrong_key_index != lf_carrier_value.wrong_key_index
    ):
        raise ContentEmbedderError("salient LF carriers and route identities differ")
    if (
        hf_carrier_value.route_identity is not None
        or lf_carrier_value.route_identity is not None
        or hf_carrier_value.route_config_digest is not None
        or lf_carrier_value.route_config_digest is not None
    ):
        raise ContentEmbedderError(
            "salient LF embedding requires unmasked carrier templates"
        )
    latent = _vector(latent_values, len(hf_direction), "latent_values")
    latent_norm = _l2_norm(latent)
    if latent_norm == 0.0:
        raise ContentEmbedderError("callback latent has zero L2 energy")
    masked_raw = tuple(
        _float32(mask * value)
        for mask, value in zip(route.mask_lf, lf_direction, strict=True)
    )
    masked_norm = _l2_norm(masked_raw)
    if masked_norm == 0.0:
        raise ContentEmbedderError("saliency-masked LF direction has zero L2 energy")
    masked_lf_direction = tuple(
        _float32(value / masked_norm) for value in masked_raw
    )
    combined_raw = tuple(
        _float32(hf_value + lf_value)
        for hf_value, lf_value in zip(
            hf_direction,
            masked_lf_direction,
            strict=True,
        )
    )
    combined_norm = _l2_norm(combined_raw)
    if combined_norm == 0.0:
        raise ContentEmbedderError("global-HF plus local-LF direction vanished")
    content_direction = tuple(
        _float32(value / combined_norm) for value in combined_raw
    )
    target_relative_l2 = _float32(
        CONTENT_RELATIVE_L2_NUMERATOR / CONTENT_RELATIVE_L2_DENOMINATOR
    )
    target_total_norm = _float32(target_relative_l2 * latent_norm)
    if target_total_norm == 0.0:
        raise ContentEmbedderError("target content update vanished in float32")
    delta_content = _target_component(target_total_norm, content_direction)
    outside_zero = all(
        mask != 0.0 or masked == 0.0
        for mask, masked in zip(route.mask_lf, masked_raw, strict=True)
    )
    inside_energy = any(
        mask == 1.0 and masked != 0.0
        for mask, masked in zip(route.mask_lf, masked_raw, strict=True)
    )
    if not outside_zero or not inside_energy:
        raise ContentEmbedderError("saliency-masked LF causal witness failed")
    config = {
        "candidate_id": SALIENT_LOCAL_LF_EMBEDDER_CANDIDATE_ID,
        "formula": "normalize(normalize(T_hf)+normalize(M_embed*T_lf))",
        "hf_carrier_config_digest": hf_carrier_value.carrier_config_digest,
        "lf_carrier_config_digest": lf_carrier_value.carrier_config_digest,
        "route_config_digest": route.route_config_digest,
        "target_relative_l2_binary32": pack(">f", target_relative_l2).hex(),
    }
    config_digest = sha256(stable_json_utf8(config)).hexdigest()
    masked_digest = _digest(masked_lf_direction)
    delta_digest = _digest(delta_content)
    identity = sha256(
        stable_json_utf8(
            {
                "candidate_id": SALIENT_LOCAL_LF_EMBEDDER_CANDIDATE_ID,
                "delta_content_digest": delta_digest,
                "embedder_config_digest": config_digest,
                "hf_direction_digest": hf_carrier_value.direction_digest,
                "masked_lf_direction_digest": masked_digest,
                "route_identity": route.route_identity,
            }
        )
    ).hexdigest()
    return SalientLocalLfEmbeddingResult(
        candidate_id=SALIENT_LOCAL_LF_EMBEDDER_CANDIDATE_ID,
        shape=route.latent_shape,
        delta_content=delta_content,
        delta_content_digest=delta_digest,
        content_direction=content_direction,
        masked_lf_direction=masked_lf_direction,
        masked_lf_direction_digest=masked_digest,
        hf_direction_digest=hf_carrier_value.direction_digest,
        route_identity=route.route_identity,
        route_config_digest=route.route_config_digest,
        latent_norm=latent_norm,
        target_total_norm=target_total_norm,
        target_relative_l2=target_relative_l2,
        lf_delta_nonzero=any(value != 0.0 for value in masked_raw),
        mask_outside_bitwise_zero=outside_zero,
        mask_inside_has_energy=inside_energy,
        embedder_config_digest=config_digest,
        embedding_result_identity=identity,
    )
