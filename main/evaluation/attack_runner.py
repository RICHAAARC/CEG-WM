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
from typing import Any, Callable, Dict, List, Optional

from main.core import digests


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
        # 真实攻击执行由上游实现接入；当前仅做协议驱动执行占位与审计封装。
        attacked_payload = images_or_latents
        output_digest = _stable_payload_digest(attacked_payload)

        event_record = {
            "attack_condition_key": attack_condition_key,
            "seed": seed,
            "params_canon_sha256": params_canon_sha256,
            "input_digest": input_digest,
            "output_digest": output_digest,
            "attack_status": "ok",
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
