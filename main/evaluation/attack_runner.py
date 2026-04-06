"""
File purpose: 协议驱动攻击执行器唯一入口。
Module type: Core innovation module

设计边界：
1. 本模块只负责协议驱动执行编排与审计追踪，不改变算法统计口径。
2. 所有参数必须来自 protocol_loader 标准化协议对象，禁止硬编码参数表。
3. 失败必须 fail-fast 并返回证据路径，便于审计与复算。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import io
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image, ImageFilter

from main.core import digests


def apply_attack_transform(
    image_or_latent: Any,
    attack_spec: Dict[str, Any],
    rng: Any,
) -> Dict[str, Any]:
    """
    功能：应用攻击变换并返回可复算追踪信息。

    Apply attack transform in deterministic audit mode.
    Supports image/latent payload with append-only trace anchors.

    Args:
        image_or_latent: Input payload.
        attack_spec: Attack specification mapping.
        rng: Random generator handle used for seed binding.

    Returns:
        Dict containing transformed payload and attack trace anchors.

    Raises:
        TypeError: If attack_spec is invalid.
    """
    if not isinstance(attack_spec, dict):
        # attack_spec 类型不合法，必须 fail-fast。
        raise TypeError("attack_spec must be dict")

    seed_value = _resolve_attack_seed(attack_spec)
    local_rng = np.random.default_rng(seed_value)

    transformed_payload, trace_core = _apply_attack_family(
        image_or_latent,
        attack_spec,
        local_rng,
    )

    attack_digest = digests.canonical_sha256(attack_spec)
    trace_payload = {
        "attack_family": trace_core.get("attack_family", attack_spec.get("attack_family", "<absent>")),
        "params_version": attack_spec.get("params_version", "<absent>"),
        "seed": int(seed_value),
        "attack_random_seed": int(seed_value),
        "params": trace_core.get("params", {}),
        "interpolation": trace_core.get("interpolation", "<absent>"),
        "transform_order": trace_core.get("transform_order", []),
        "library_versions": {
            "numpy": np.__version__,
            "pillow": Image.__version__,
        },
        "attack_digest": attack_digest,
        "rng_type": type(rng).__name__ if rng is not None else "<absent>",
    }
    attack_trace_digest = digests.canonical_sha256(trace_payload)
    trace_payload["attack_trace_digest"] = attack_trace_digest

    return {
        "payload": transformed_payload,
        "attack_trace": trace_payload,
        "attack_trace_digest": attack_trace_digest,
        "attack_digest": attack_digest,
        "seed": int(seed_value),
    }


def should_enable_geometry_chain(
    attack_spec: Dict[str, Any],
    cfg: Dict[str, Any],
    plan: Dict[str, Any],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    功能：判断是否启用几何链并输出路由摘要。

    Decide geometry chain routing and return routing decisions plus routing digest.

    Args:
        attack_spec: Attack specification mapping.
        cfg: Runtime config mapping.
        plan: Plan mapping.
        evidence: Evidence mapping.

    Returns:
        Dict with enable flag, routing_decisions and routing_digest.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(attack_spec, dict):
        # attack_spec 类型不合法，必须 fail-fast。
        raise TypeError("attack_spec must be dict")
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(plan, dict):
        # plan 类型不合法，必须 fail-fast。
        raise TypeError("plan must be dict")
    if not isinstance(evidence, dict):
        # evidence 类型不合法，必须 fail-fast。
        raise TypeError("evidence must be dict")

    family = attack_spec.get("attack_family")
    if not isinstance(family, str) or not family:
        family = attack_spec.get("family") if isinstance(attack_spec.get("family"), str) else "<absent>"

    geometry_attack_families = {
        "rotate",
        "resize",
        "crop",
        "translate",
        "composite",
    }
    enabled_by_family = family in geometry_attack_families
    cfg_geometry_enabled = bool(
        isinstance(cfg.get("detect"), dict)
        and isinstance(cfg.get("detect", {}).get("geometry"), dict)
        and cfg.get("detect", {}).get("geometry", {}).get("enabled", False)
    )

    routing_decisions = {
        "attack_family": family,
        "enabled_by_family": enabled_by_family,
        "cfg_geometry_enabled": cfg_geometry_enabled,
        "plan_digest": plan.get("plan_digest", "<absent>"),
        "evidence_status": evidence.get("status", "<absent>"),
        "geo_chain_enabled": bool(enabled_by_family and cfg_geometry_enabled),
    }
    routing_digest = digests.canonical_sha256(routing_decisions)

    return {
        "geo_chain_enabled": bool(routing_decisions["geo_chain_enabled"]),
        "routing_decisions": routing_decisions,
        "routing_digest": routing_digest,
    }


class AttackFailureReason(str, Enum):
    """
    功能：攻击执行失败原因枚举。

    Enumerated failure reasons for attack runner execution.
    """

    OK = "ok"
    INVALID_CONDITION_SPEC = "invalid_condition_spec"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_FIELD_TYPE = "invalid_field_type"
    PROTOCOL_PARAMS_MISMATCH = "protocol_params_mismatch"
    RECORD_HOOK_ERROR = "record_hook_error"
    ATTACK_EXECUTION_ERROR = "attack_execution_error"


@dataclass(frozen=True)
class AttackRunResult:
    """
    功能：攻击执行结果结构体。

    Structured attack execution result for reproducible evaluation.

    Attributes:
        attack_condition_key: Canonical key in format family::params_version.
        params_canon_sha256: Canonical SHA256 digest of params object.
        input_digest: Stable digest of input payload.
        output_digest: Stable digest of output payload.
        attack_status: Execution status, "ok" or "fail".
        failure_reason: Enumerated failure reason string.
        attack_trace: Minimal reproducibility trace.
    """

    attack_condition_key: str
    params_canon_sha256: str
    input_digest: str
    output_digest: str
    attack_status: str
    failure_reason: str
    attack_trace: Dict[str, Any]
    attack_trace_digest: str


def resolve_condition_spec_from_protocol(
    protocol_spec: Dict[str, Any],
    condition_key: str,
) -> Dict[str, Any]:
    """
    功能：从协议对象解析条件规范。

    Resolve condition specification from standardized attack protocol object.

    Args:
        protocol_spec: Protocol dictionary loaded by protocol_loader.
        condition_key: Canonical condition key in format family::params_version.

    Returns:
        Condition specification dict with fields attack_family, params_version, params, and source metadata.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If condition_key or protocol fields are missing.
    """
    if not isinstance(protocol_spec, dict):
        # protocol_spec 类型不合法，必须 fail-fast。
        raise TypeError("protocol_spec must be dict")
    if not isinstance(condition_key, str) or "::" not in condition_key:
        # condition_key 非 canonical 格式，必须 fail-fast。
        raise ValueError(f"invalid condition_key format: {condition_key}")

    family_name, params_version = condition_key.split("::", 1)
    if not family_name or not params_version:
        # family/version 为空，必须 fail-fast。
        raise ValueError(f"invalid condition_key components: {condition_key}")

    params_versions = protocol_spec.get("params_versions")
    if not isinstance(params_versions, dict):
        # protocol 字段类型错误，必须 fail-fast。
        raise ValueError("protocol_spec.params_versions must be dict")

    condition_obj = params_versions.get(condition_key)
    if not isinstance(condition_obj, dict):
        # condition 未定义，必须 fail-fast。
        raise ValueError(f"condition_key not found in protocol_spec.params_versions: {condition_key}")

    params_obj = condition_obj.get("params")
    if not isinstance(params_obj, dict):
        # params 缺失或类型不合法，必须 fail-fast。
        raise ValueError(
            "invalid protocol field: path=params_versions"
            f"[{condition_key}].params must be dict"
        )

    return {
        "attack_family": family_name,
        "params_version": params_version,
        "params": params_obj,
        "condition_key": condition_key,
        "protocol_version": protocol_spec.get("version", "<absent>"),
        "protocol_digest": protocol_spec.get("attack_protocol_digest", "<absent>"),
        "params_canon_sha256": digests.canonical_sha256(params_obj),
    }


def run_attacks_for_condition(
    images_or_latents: Any,
    condition_spec: Dict[str, Any],
    seed: int,
    *,
    record_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> AttackRunResult:
    """
    功能：运行单个条件攻击并输出可审计结果。

    Run attacks for one protocol condition and return reproducible audit result.

    Args:
        images_or_latents: Input payload to attack stage.
        condition_spec: Condition specification resolved from protocol.
        seed: Reproducibility seed.
        record_hook: Optional callback for trace records.

    Returns:
        AttackRunResult with canonical condition key, digests, status, reason, and trace.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If condition_spec misses required fields.
    """
    if not isinstance(condition_spec, dict):
        # condition_spec 类型不合法，必须 fail-fast。
        raise TypeError("condition_spec must be dict")
    if not isinstance(seed, int):
        # seed 类型不合法，必须 fail-fast。
        raise TypeError("seed must be int")
    if record_hook is not None and not callable(record_hook):
        # record_hook 类型不合法，必须 fail-fast。
        raise TypeError("record_hook must be callable or None")

    required_fields = ["attack_family", "params_version", "params"]
    for field_name in required_fields:
        if field_name not in condition_spec:
            # 必填字段缺失，必须 fail-fast。
            raise ValueError(
                "condition_spec missing required field: "
                f"path={field_name}, condition_key={condition_spec.get('condition_key', '<absent>')}"
            )

    attack_family = condition_spec.get("attack_family")
    params_version = condition_spec.get("params_version")
    params_obj = condition_spec.get("params")

    if not isinstance(attack_family, str) or not attack_family:
        # family 类型不合法，必须 fail-fast。
        raise ValueError("condition_spec.attack_family must be non-empty str")
    if not isinstance(params_version, str) or not params_version:
        # params_version 类型不合法，必须 fail-fast。
        raise ValueError("condition_spec.params_version must be non-empty str")
    if not isinstance(params_obj, dict):
        # params 类型不合法，必须 fail-fast。
        raise ValueError("condition_spec.params must be dict")

    attack_condition_key = f"{attack_family}::{params_version}"
    params_canon_sha256 = digests.canonical_sha256(params_obj)
    input_digest = _stable_payload_digest(images_or_latents)

    trace: Dict[str, Any] = {
        "seed": seed,
        "attack_family": attack_family,
        "params_version": params_version,
        "attack_condition_key": attack_condition_key,
        "params_canon_sha256": params_canon_sha256,
        "execution_order": [attack_condition_key],
        "protocol_version": condition_spec.get("protocol_version", "<absent>"),
        "protocol_digest": condition_spec.get("protocol_digest", "<absent>"),
    }

    try:
        attack_spec = {
            "attack_family": attack_family,
            "params_version": params_version,
            "params": params_obj,
            "seed": seed,
        }
        transform_result = apply_attack_transform(
            images_or_latents,
            attack_spec,
            random.Random(seed),
        )
        attacked_payload = transform_result["payload"]
        output_digest = _stable_payload_digest(attacked_payload)
        attack_trace = transform_result.get("attack_trace") if isinstance(transform_result, dict) else {}
        attack_trace_digest = transform_result.get("attack_trace_digest", "<absent>")
        if isinstance(attack_trace, dict):
            trace.update(attack_trace)
        trace["attack_trace_digest"] = attack_trace_digest

        event_record = {
            "attack_condition_key": attack_condition_key,
            "seed": seed,
            "params_canon_sha256": params_canon_sha256,
            "input_digest": input_digest,
            "output_digest": output_digest,
            "attack_status": "ok",
            "attack_trace_digest": attack_trace_digest,
        }
        if record_hook is not None:
            record_hook(event_record)

        return AttackRunResult(
            attack_condition_key=attack_condition_key,
            params_canon_sha256=params_canon_sha256,
            input_digest=input_digest,
            output_digest=output_digest,
            attack_status="ok",
            failure_reason=AttackFailureReason.OK.value,
            attack_trace=trace,
            attack_trace_digest=attack_trace_digest,
        )
    except Exception as exc:
        failure_reason = AttackFailureReason.ATTACK_EXECUTION_ERROR
        if record_hook is not None:
            try:
                record_hook(
                    {
                        "attack_condition_key": attack_condition_key,
                        "seed": seed,
                        "params_canon_sha256": params_canon_sha256,
                        "input_digest": input_digest,
                        "attack_status": "fail",
                        "failure_reason": failure_reason.value,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            except Exception as hook_exc:
                failure_reason = AttackFailureReason.RECORD_HOOK_ERROR
                trace["record_hook_error"] = f"{type(hook_exc).__name__}: {hook_exc}"

        return AttackRunResult(
            attack_condition_key=attack_condition_key,
            params_canon_sha256=params_canon_sha256,
            input_digest=input_digest,
            output_digest="<absent>",
            attack_status="fail",
            failure_reason=failure_reason.value,
            attack_trace={
                **trace,
                "error": f"{type(exc).__name__}: {exc}",
            },
                attack_trace_digest="<absent>",
        )


def _stable_payload_digest(payload: Any) -> str:
    """
    功能：计算 payload 的稳定摘要。

    Compute stable digest for generic payload.

    Args:
        payload: Any serializable/non-serializable object.

    Returns:
        Stable digest string.
    """
    try:
        return digests.canonical_sha256(payload)
    except Exception:
        # 非 JSON 可序列化对象回退到 repr，保证审计链可追踪。
        return digests.canonical_sha256({"repr": repr(payload)})


def _resolve_attack_seed(attack_spec: Dict[str, Any]) -> int:
    """
    功能：解析攻击随机种子。

    Resolve deterministic attack seed from attack specification.

    Args:
        attack_spec: Attack specification mapping.

    Returns:
        Integer seed.
    """
    seed_value = attack_spec.get("seed")
    if isinstance(seed_value, int):
        return int(seed_value)
    rng_seed = attack_spec.get("rng_seed")
    if isinstance(rng_seed, int):
        return int(rng_seed)
    return 0


def _apply_attack_family(
    payload: Any,
    attack_spec: Dict[str, Any],
    local_rng: np.random.Generator,
) -> tuple[Any, Dict[str, Any]]:
    """
    功能：按攻击族执行真实变换。

    Apply concrete transform for one attack family.

    Args:
        payload: Input payload.
        attack_spec: Attack specification.
        local_rng: Deterministic numpy random generator.

    Returns:
        Tuple of (transformed_payload, trace_core).

    Raises:
        ValueError: If attack family or params are invalid.
    """
    family = attack_spec.get("attack_family")
    if not isinstance(family, str) or not family:
        raise ValueError("attack_spec.attack_family must be non-empty str")

    params = attack_spec.get("params")
    if not isinstance(params, dict):
        raise ValueError("attack_spec.params must be dict")

    family_normalized = _normalize_attack_family(family)
    if family_normalized == "composite":
        transformed_payload = payload
        transform_order: List[str] = []
        interpolation_values: List[str] = []
        steps = params.get("steps")
        if not isinstance(steps, list) or len(steps) == 0:
            raise ValueError("composite attack requires non-empty params.steps")
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ValueError(f"composite step[{index}] must be dict")
            step_family = step.get("family")
            step_params = step.get("params")
            if not isinstance(step_family, str) or not isinstance(step_params, dict):
                raise ValueError(f"composite step[{index}] missing family/params")
            step_spec = {
                "attack_family": step_family,
                "params": step_params,
                "seed": int(local_rng.integers(0, 2**31 - 1)),
            }
            step_payload, step_trace = _apply_attack_family(
                transformed_payload,
                step_spec,
                np.random.default_rng(_resolve_attack_seed(step_spec)),
            )
            transformed_payload = step_payload
            transform_order.extend(step_trace.get("transform_order", [step_family]))
            interpolation_values.append(step_trace.get("interpolation", "<absent>"))
        return transformed_payload, {
            "attack_family": "composite",
            "params": params,
            "interpolation": interpolation_values,
            "transform_order": transform_order,
        }

    transformed_payload = _apply_single_transform(payload, family_normalized, params, local_rng)
    interpolation_value = _resolve_interpolation(params)
    return transformed_payload, {
        "attack_family": family_normalized,
        "params": params,
        "interpolation": interpolation_value,
        "transform_order": [family_normalized],
    }


def _normalize_attack_family(family: str) -> str:
    """
    功能：规范化攻击族名称。

    Normalize attack family aliases.

    Args:
        family: Raw family name.

    Returns:
        Normalized family name.
    """
    mapping = {
        "jpeg": "jpeg_compression",
        "jpeg_compression": "jpeg_compression",
    }
    normalized = mapping.get(family, family)
    return normalized


def _apply_single_transform(
    payload: Any,
    family: str,
    params: Dict[str, Any],
    local_rng: np.random.Generator,
) -> Any:
    """
    功能：执行单步攻击变换。

    Apply one concrete attack transformation.

    Args:
        payload: Input payload.
        family: Normalized family name.
        params: Attack parameters.
        local_rng: Deterministic random generator.

    Returns:
        Transformed payload.

    Raises:
        ValueError: If family is unsupported.
    """
    if isinstance(payload, dict):
        key_candidates = ["image", "latent", "payload", "image_or_latent"]
        for key_name in key_candidates:
            if key_name in payload:
                transformed_payload = dict(payload)
                transformed_payload[key_name] = _apply_single_transform(
                    payload[key_name],
                    family,
                    params,
                    local_rng,
                )
                return transformed_payload
        return payload

    if family == "rotate":
        degrees = _resolve_float(params, ["degrees", "degree"], default=0.0)
        return _transform_rotate(payload, degrees, _resolve_interpolation(params))
    if family == "resize":
        scale_factor = _resolve_float(params, ["scale_factor", "scale", "scale_factors"], default=1.0)
        return _transform_resize(payload, scale_factor, _resolve_interpolation(params))
    if family == "crop":
        crop_ratio = _resolve_float(params, ["crop_ratio", "crop_ratios"], default=1.0)
        return _transform_crop(payload, crop_ratio, _resolve_interpolation(params))
    if family == "translate":
        x_shift = int(_resolve_float(params, ["x_shift"], default=0.0))
        y_shift = int(_resolve_float(params, ["y_shift"], default=0.0))
        return _transform_translate(payload, x_shift, y_shift)
    if family == "jpeg_compression":
        quality = int(_resolve_float(params, ["quality"], default=85.0))
        return _transform_jpeg_compression(payload, quality)
    if family == "gaussian_noise":
        sigma = _resolve_float(params, ["sigma"], default=0.01)
        return _transform_gaussian_noise(payload, sigma, local_rng)
    if family == "gaussian_blur":
        sigma = _resolve_float(params, ["sigma"], default=1.0)
        kernel_size = int(_resolve_float(params, ["kernel_size"], default=3.0))
        return _transform_gaussian_blur(payload, sigma, kernel_size)
    raise ValueError(f"unsupported attack family: {family}")


def _transform_rotate(payload: Any, degrees: float, interpolation: str) -> Any:
    if _is_image_payload(payload):
        image = _to_pil_image(payload)
        rotated = image.rotate(float(degrees), resample=_pil_resample(interpolation), expand=False)
        return _from_pil_image(payload, rotated)
    array = _to_numpy_array(payload)
    if abs(degrees) % 90.0 != 0.0:
        raise ValueError("rotate on latent payload supports only multiples of 90 degrees")
    rotate_k = int((degrees / 90.0) % 4)
    return np.rot90(array, k=rotate_k, axes=(-2, -1))


def _transform_resize(payload: Any, scale_factor: float, interpolation: str) -> Any:
    if scale_factor <= 0:
        raise ValueError("resize scale_factor must be > 0")
    if _is_image_payload(payload):
        image = _to_pil_image(payload)
        target_width = max(1, int(round(image.width * scale_factor)))
        target_height = max(1, int(round(image.height * scale_factor)))
        resized = image.resize((target_width, target_height), resample=_pil_resample(interpolation))
        restored = resized.resize((image.width, image.height), resample=_pil_resample(interpolation))
        return _from_pil_image(payload, restored)
    array = _to_numpy_array(payload)
    return _resize_ndarray(array, scale_factor)


def _transform_crop(payload: Any, crop_ratio: float, interpolation: str) -> Any:
    if crop_ratio <= 0 or crop_ratio > 1:
        raise ValueError("crop_ratio must be in (0, 1]")
    if _is_image_payload(payload):
        image = _to_pil_image(payload)
        crop_width = max(1, int(round(image.width * crop_ratio)))
        crop_height = max(1, int(round(image.height * crop_ratio)))
        left = max(0, (image.width - crop_width) // 2)
        upper = max(0, (image.height - crop_height) // 2)
        cropped = image.crop((left, upper, left + crop_width, upper + crop_height))
        restored = cropped.resize((image.width, image.height), resample=_pil_resample(interpolation))
        return _from_pil_image(payload, restored)
    array = _to_numpy_array(payload)
    return _center_crop_ndarray(array, crop_ratio)


def _transform_translate(payload: Any, x_shift: int, y_shift: int) -> Any:
    if _is_image_payload(payload):
        image = _to_pil_image(payload)
        canvas = Image.new(image.mode, image.size)
        canvas.paste(image, (int(x_shift), int(y_shift)))
        return _from_pil_image(payload, canvas)
    array = _to_numpy_array(payload)
    return _translate_ndarray(array, x_shift, y_shift)


def _transform_jpeg_compression(payload: Any, quality: int) -> Any:
    if not _is_image_payload(payload):
        raise ValueError("jpeg_compression requires image payload")
    quality_clamped = max(1, min(100, int(quality)))
    image = _to_pil_image(payload)
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=quality_clamped)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert(image.mode)
        return _from_pil_image(payload, recompressed)


def _transform_gaussian_noise(payload: Any, sigma: float, local_rng: np.random.Generator) -> Any:
    array = _to_numpy_array(payload).astype(np.float32)
    noise = local_rng.normal(0.0, float(sigma), size=array.shape).astype(np.float32)
    transformed = array + noise
    return _restore_array_dtype(payload, transformed)


def _transform_gaussian_blur(payload: Any, sigma: float, kernel_size: int) -> Any:
    sigma_value = float(sigma)
    kernel_size_value = max(1, int(kernel_size))
    if _is_image_payload(payload):
        image = _to_pil_image(payload)
        radius = max(0.1, sigma_value)
        blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return _from_pil_image(payload, blurred)
    array = _to_numpy_array(payload)
    return _gaussian_blur_ndarray(array, sigma_value, kernel_size_value)


def _resolve_float(params: Dict[str, Any], keys: List[str], default: float) -> float:
    for key in keys:
        value = params.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
            return float(value[0])
    return float(default)


def _resolve_interpolation(params: Dict[str, Any]) -> str:
    interpolation = params.get("interpolation")
    if isinstance(interpolation, str) and interpolation:
        return interpolation.lower()
    return "bilinear"


def _pil_resample(interpolation: str) -> int:
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return mapping.get(interpolation.lower(), Image.Resampling.BILINEAR)


def _is_image_payload(payload: Any) -> bool:
    if isinstance(payload, Image.Image):
        return True
    if isinstance(payload, np.ndarray) and payload.ndim in (2, 3):
        return True
    return False


def _to_pil_image(payload: Any) -> Image.Image:
    if isinstance(payload, Image.Image):
        return payload
    if isinstance(payload, np.ndarray):
        array = payload
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)
    raise TypeError("payload is not image-like")


def _from_pil_image(original_payload: Any, image: Image.Image) -> Any:
    if isinstance(original_payload, Image.Image):
        return image
    if isinstance(original_payload, np.ndarray):
        array = np.array(image)
        if original_payload.dtype != np.uint8:
            return array.astype(original_payload.dtype)
        return array
    return image


def _to_numpy_array(payload: Any) -> np.ndarray:
    if isinstance(payload, np.ndarray):
        return payload
    if hasattr(payload, "detach") and callable(payload.detach):
        detached = payload.detach()
        if hasattr(detached, "cpu") and callable(detached.cpu):
            detached = detached.cpu()
        if hasattr(detached, "numpy") and callable(detached.numpy):
            return detached.numpy()
    if isinstance(payload, Image.Image):
        return np.array(payload)
    raise TypeError("payload must be numpy array, PIL image, or tensor-like")


def _restore_array_dtype(original_payload: Any, transformed: np.ndarray) -> Any:
    if isinstance(original_payload, Image.Image):
        restored_image = _to_pil_image(transformed)
        if restored_image.mode != original_payload.mode:
            restored_image = restored_image.convert(original_payload.mode)
        return _from_pil_image(original_payload, restored_image)
    if isinstance(original_payload, np.ndarray):
        if np.issubdtype(original_payload.dtype, np.integer):
            return np.clip(transformed, 0, 255).astype(original_payload.dtype)
        return transformed.astype(original_payload.dtype)
    return transformed


def _resize_ndarray(array: np.ndarray, scale_factor: float) -> np.ndarray:
    if array.ndim < 2:
        raise ValueError("ndarray resize requires at least 2 dimensions")
    h_axis = array.ndim - 2
    w_axis = array.ndim - 1
    src_h = array.shape[h_axis]
    src_w = array.shape[w_axis]
    dst_h = max(1, int(round(src_h * scale_factor)))
    dst_w = max(1, int(round(src_w * scale_factor)))
    y_indices = np.linspace(0, src_h - 1, dst_h).astype(np.int32)
    x_indices = np.linspace(0, src_w - 1, dst_w).astype(np.int32)
    resized = np.take(array, y_indices, axis=h_axis)
    resized = np.take(resized, x_indices, axis=w_axis)

    restore_y = np.linspace(0, dst_h - 1, src_h).astype(np.int32)
    restore_x = np.linspace(0, dst_w - 1, src_w).astype(np.int32)
    restored = np.take(resized, restore_y, axis=h_axis)
    restored = np.take(restored, restore_x, axis=w_axis)
    return restored


def _center_crop_ndarray(array: np.ndarray, crop_ratio: float) -> np.ndarray:
    if array.ndim < 2:
        raise ValueError("ndarray crop requires at least 2 dimensions")
    h_axis = array.ndim - 2
    w_axis = array.ndim - 1
    src_h = array.shape[h_axis]
    src_w = array.shape[w_axis]
    crop_h = max(1, int(round(src_h * crop_ratio)))
    crop_w = max(1, int(round(src_w * crop_ratio)))
    top = max(0, (src_h - crop_h) // 2)
    left = max(0, (src_w - crop_w) // 2)

    slicer = [slice(None)] * array.ndim
    slicer[h_axis] = slice(top, top + crop_h)
    slicer[w_axis] = slice(left, left + crop_w)
    cropped = array[tuple(slicer)]

    scale_h = src_h / max(1, crop_h)
    return _resize_ndarray(cropped, scale_h)


def _translate_ndarray(array: np.ndarray, x_shift: int, y_shift: int) -> np.ndarray:
    if array.ndim < 2:
        raise ValueError("ndarray translate requires at least 2 dimensions")
    h_axis = array.ndim - 2
    w_axis = array.ndim - 1
    translated = np.zeros_like(array)
    src_h = array.shape[h_axis]
    src_w = array.shape[w_axis]

    dst_y_start = max(0, y_shift)
    dst_y_end = min(src_h, src_h + y_shift)
    dst_x_start = max(0, x_shift)
    dst_x_end = min(src_w, src_w + x_shift)

    src_y_start = max(0, -y_shift)
    src_y_end = src_y_start + max(0, dst_y_end - dst_y_start)
    src_x_start = max(0, -x_shift)
    src_x_end = src_x_start + max(0, dst_x_end - dst_x_start)

    if dst_y_start >= dst_y_end or dst_x_start >= dst_x_end:
        return translated

    src_slicer = [slice(None)] * array.ndim
    dst_slicer = [slice(None)] * array.ndim
    src_slicer[h_axis] = slice(src_y_start, src_y_end)
    src_slicer[w_axis] = slice(src_x_start, src_x_end)
    dst_slicer[h_axis] = slice(dst_y_start, dst_y_end)
    dst_slicer[w_axis] = slice(dst_x_start, dst_x_end)
    translated[tuple(dst_slicer)] = array[tuple(src_slicer)]
    return translated


def _gaussian_blur_ndarray(array: np.ndarray, sigma: float, kernel_size: int) -> np.ndarray:
    if array.ndim < 2:
        raise ValueError("ndarray gaussian_blur requires at least 2 dimensions")
    if sigma <= 0:
        return array
    kernel = _build_gaussian_kernel(max(3, kernel_size), sigma)
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=-1, arr=array)
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=-2, arr=blurred)
    return _restore_array_dtype(array, blurred)


def _build_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma * sigma))
    kernel_sum = float(np.sum(kernel))
    if kernel_sum == 0.0:
        return np.array([1.0], dtype=np.float32)
    return kernel / kernel_sum
