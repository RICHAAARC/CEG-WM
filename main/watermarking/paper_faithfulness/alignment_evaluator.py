"""
Paper Alignment Evaluator

功能：
- 把 paper_spec 的"必须对齐项"映射为可自动判定的 checks。
- 验证 pipeline_fingerprint、trajectory、injection_site 是否满足 paper_spec 约束。
- 生成 alignment_report 与 alignment_digest。

Module type: Core innovation module
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from main.core import digests


IMPL_ID = "paper_alignment_evaluator"
IMPL_VERSION = "v1"


def evaluate_alignment(
    paper_spec: Dict[str, Any],
    pipeline_fingerprint: Optional[Dict[str, Any]],
    trajectory_evidence: Optional[Dict[str, Any]],
    injection_site_spec: Optional[Dict[str, Any]],
    cfg: Dict[str, Any]
) -> Tuple[Dict[str, Any], str]:
    """
    功能：评估对齐检查。

    Evaluate paper faithfulness alignment checks.

    Args:
        paper_spec: Paper faithfulness spec dict (from paper_faithfulness_spec.yaml).
        pipeline_fingerprint: SD3 pipeline fingerprint dict.
        trajectory_evidence: Diffusion trajectory evidence dict.
        injection_site_spec: Injection site spec dict.
        cfg: Configuration mapping (包含 enable_paper_faithfulness 等开关).

    Returns:
        Tuple of (alignment_report dict, alignment_digest str).
        alignment_report 包含逐项 PASS/FAIL/NA + 证据路径。
        alignment_digest 是 alignment_report 的 canonical sha256。

    Raises:
        TypeError: If inputs types invalid.
    """
    if not isinstance(paper_spec, dict):
        raise TypeError("paper_spec must be dict")

    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    # 确定是否启用 paper_faithfulness（用于决定 absent 状态是否升级为 FAIL）
    paper_faithfulness_cfg = cfg.get("paper_faithfulness", {})
    is_paper_faithfulness_enabled = False
    if isinstance(paper_faithfulness_cfg, dict):
        is_paper_faithfulness_enabled = paper_faithfulness_cfg.get("enabled", False)

    alignment_checks = []

    # 检查 1：pipeline_fingerprint 必须存在且非缺省标记值。
    check_1 = _check_pipeline_fingerprint_presence(
        pipeline_fingerprint,
        enable_paper_faithfulness=is_paper_faithfulness_enabled
    )
    alignment_checks.append(check_1)

    # 检查 2：trajectory_digest 必须存在且可复算。
    check_2 = _check_trajectory_digest_reproducibility(
        trajectory_evidence,
        enable_paper_faithfulness=is_paper_faithfulness_enabled
    )
    alignment_checks.append(check_2)

    # 检查 3：injection_site_spec 必须与 paper_spec 对齐。
    check_3 = _check_injection_site_alignment(paper_spec, injection_site_spec, cfg)
    alignment_checks.append(check_3)

    # 检查 4：方法特定参数必须绑定。
    check_4 = _check_method_specific_parameters(paper_spec, cfg)
    alignment_checks.append(check_4)

    # 汇总结果。
    total_checks = len(alignment_checks)
    pass_count = sum(1 for c in alignment_checks if c["result"] == "PASS")
    fail_count = sum(1 for c in alignment_checks if c["result"] == "FAIL")
    na_count = sum(1 for c in alignment_checks if c["result"] == "NA")

    overall_status = "PASS" if fail_count == 0 else "FAIL"

    alignment_report = {
        "overall_status": overall_status,
        "total_checks": total_checks,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "na_count": na_count,
        "checks": alignment_checks,
        "evaluator_version": IMPL_VERSION
    }

    # 生成 alignment_digest。
    alignment_digest = digests.canonical_sha256(alignment_report)

    return alignment_report, alignment_digest


def _check_pipeline_fingerprint_presence(
    pipeline_fingerprint: Optional[Dict[str, Any]],
    enable_paper_faithfulness: bool = False
) -> Dict[str, Any]:
    """
    功能：检查 pipeline_fingerprint 是否存在且非缺省标记值。

    Check pipeline fingerprint presence and non-empty value.

    Args:
        pipeline_fingerprint: Pipeline fingerprint dict.
        enable_paper_faithfulness: Whether paper faithfulness is enabled (影响 absent 是否升级为 FAIL).

    Returns:
        Check result dict with result in [PASS, FAIL, NA].
    """
    check_name = "pipeline_fingerprint_presence"
    check_rule = "SD3 pipeline fingerprint 必须存在且非缺省标记值"

    if pipeline_fingerprint is None:
        return {
            "check_name": check_name,
            "check_rule": check_rule,
            "result": "FAIL",
            "failure_message": "pipeline_fingerprint 缺失（None）"
        }

    if not isinstance(pipeline_fingerprint, dict):
        return {
            "check_name": check_name,
            "check_rule": check_rule,
            "result": "FAIL",
            "failure_message": "pipeline_fingerprint 类型错误（非 dict）"
        }

    # 检查 pipeline 是否处于不可评估的状态（absent 或 failed）
    pipeline_status = pipeline_fingerprint.get("status")
    if pipeline_status in ["absent", "failed"]:
        # 当启用 paper_faithfulness 时，absent/failed 状态必须升级为 FAIL（不允许 NA）
        # 因为 paper_faithfulness 模式下 fingerprint 是必达的
        if enable_paper_faithfulness:
            reason = pipeline_fingerprint.get("reason", "unknown")
            return {
                "check_name": check_name,
                "check_rule": check_rule,
                "result": "FAIL",
                "failure_message": f"paper_faithfulness enabled 但 pipeline_fingerprint 为 {pipeline_status}（reason: {reason}）"
            }
        else:
            return {
                "check_name": check_name,
                "check_rule": check_rule,
                "result": "NA",
                "na_reason": f"无法检查：pipeline_fingerprint 状态为 {pipeline_status}"
            }

    # 检查关键字段是否存在。
    required_fields = [
        "transformer_num_blocks",
        "scheduler_class_name",
        "vae_latent_channels"
    ]

    missing_fields = []
    for field in required_fields:
        value = pipeline_fingerprint.get(field)
        # 只有当字段完全缺失（None）时才视为缺失
        # "<absent>" 是合法的缺省标记值，表示该模块不存在或无法提取
        if value is None:
            missing_fields.append(field)

    if missing_fields:
        return {
            "check_name": check_name,
            "check_rule": check_rule,
            "result": "FAIL",
            "failure_message": f"pipeline_fingerprint 缺少关键字段: {missing_fields}"
        }

    return {
        "check_name": check_name,
        "check_rule": check_rule,
        "result": "PASS",
        "evidence_fields": required_fields
    }


def _check_trajectory_digest_reproducibility(
    trajectory_evidence: Optional[Dict[str, Any]],
    enable_paper_faithfulness: bool = False
) -> Dict[str, Any]:
    """
    功能：检查 trajectory_digest 是否可复算。

    Check trajectory digest reproducibility.

    Args:
        trajectory_evidence: Trajectory evidence dict.
        enable_paper_faithfulness: Whether paper faithfulness is enabled (影响 absent 是否升级为 FAIL).

    Returns:
        Check result dict.
    """
    check_name = "trajectory_digest_reproducibility"
    check_rule = "trajectory_digest 必须存在且可复算"

    if trajectory_evidence is None:
        # 当启用 paper_faithfulness 时，缺失 trajectory_evidence 必须升级为 FAIL
        if enable_paper_faithfulness:
            return {
                "check_name": check_name,
                "check_rule": check_rule,
                "result": "FAIL",
                "failure_message": "paper_faithfulness enabled 但 trajectory_evidence 为 None"
            }
        else:
            return {
                "check_name": check_name,
                "check_rule": check_rule,
                "result": "NA",
                "na_reason": "trajectory_evidence 未启用"
            }

    status = trajectory_evidence.get("status")
    if status == "absent":
        # 当启用 paper_faithfulness 时，absent 状态必须升级为 FAIL
        absent_reason = trajectory_evidence.get("trajectory_absent_reason", "unknown")
        if enable_paper_faithfulness:
            return {
                "check_name": check_name,
                "check_rule": check_rule,
                "result": "FAIL",
                "failure_message": f"paper_faithfulness enabled 但 trajectory tracing 为 absent（reason: {absent_reason}）"
            }
        else:
            return {
                "check_name": check_name,
                "check_rule": check_rule,
                "result": "NA",
                "na_reason": f"trajectory tracing 未启用（reason: {absent_reason}）"
            }

    trajectory_spec_digest = trajectory_evidence.get("trajectory_spec_digest")
    trajectory_digest = trajectory_evidence.get("trajectory_digest")

    if trajectory_spec_digest == "<absent>" or trajectory_digest == "<absent>":
        return {
            "check_name": check_name,
            "check_rule": check_rule,
            "result": "FAIL",
            "failure_message": "trajectory digest 为缺省标记值（<absent>）"
        }

    if not isinstance(trajectory_spec_digest, str) or len(trajectory_spec_digest) != 64:
        return {
            "check_name": check_name,
            "check_rule": check_rule,
            "result": "FAIL",
            "failure_message": "trajectory_spec_digest 格式错误（非 64 位十六进制）"
        }

    if not isinstance(trajectory_digest, str) or len(trajectory_digest) != 64:
        return {
            "check_name": check_name,
            "check_rule": check_rule,
            "result": "FAIL",
            "failure_message": "trajectory_digest 格式错误（非 64 位十六进制）"
        }

    return {
        "check_name": check_name,
        "check_rule": check_rule,
        "result": "PASS",
        "evidence_fields": ["trajectory_spec_digest", "trajectory_digest"]
    }


def _check_injection_site_alignment(
    paper_spec: Dict[str, Any],
    injection_site_spec: Optional[Dict[str, Any]],
    cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    功能：检查 injection_site_spec 是否与 paper_spec 对齐。

    Check injection site spec conformance with paper spec.

    Args:
        paper_spec: Paper spec dict.
        injection_site_spec: Injection site spec dict.
        cfg: Configuration mapping.

    Returns:
        Check result dict.
    """
    check_name = "injection_site_alignment"
    check_rule = "injection_site_spec 必须与 paper_spec 对齐"

    if injection_site_spec is None:
        return {
            "check_name": check_name,
            "check_rule": check_rule,
            "result": "FAIL",
            "failure_message": "injection_site_spec 缺失（None）"
        }

    status = injection_site_spec.get("status")
    if status == "absent":
        return {
            "check_name": check_name,
            "check_rule": check_rule,
            "result": "FAIL",
            "failure_message": "injection_site_spec 为 absent 状态"
        }

    hook_type = injection_site_spec.get("hook_type")
    if hook_type == "<absent>":
        return {
            "check_name": check_name,
            "check_rule": check_rule,
            "result": "FAIL",
            "failure_message": "hook_type 为缺省标记值（<absent>）"
        }

    # 检查是否符合 SD3 适配要求（从 paper_spec 读取）。
    sd3_adaptation = paper_spec.get("sd3_adaptation", {})
    injection_binding = sd3_adaptation.get("injection_site_binding", {})
    required_fields = injection_binding.get("required_fields", [])

    missing_fields = []
    for field in required_fields:
        value = injection_site_spec.get(field)
        if value is None or value == "<absent>":
            missing_fields.append(field)

    if missing_fields:
        return {
            "check_name": check_name,
            "check_rule": check_rule,
            "result": "FAIL",
            "failure_message": f"injection_site_spec 缺少 SD3 必需字段: {missing_fields}"
        }

    return {
        "check_name": check_name,
        "check_rule": check_rule,
        "result": "PASS",
        "evidence_fields": required_fields
    }


def _check_method_specific_parameters(
    paper_spec: Dict[str, Any],
    cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    功能：检查方法特定参数是否绑定。

    Check method-specific parameter binding.

    Args:
        paper_spec: Paper spec dict.
        cfg: Configuration mapping.

    Returns:
        Check result dict.
    """
    check_name = "method_specific_parameter_binding"
    check_rule = "方法特定参数必须正确绑定"

    # 从 paper_spec 读取对齐检查规则。
    alignment_rules = paper_spec.get("alignment_check_rules", {})
    param_binding_rule = alignment_rules.get("method_specific_parameter_binding", {})
    bindings = param_binding_rule.get("bindings", {})

    if not bindings:
        return {
            "check_name": check_name,
            "check_rule": check_rule,
            "result": "NA",
            "na_reason": "无方法特定参数绑定要求"
        }

    # 检查每个方法的参数绑定。
    failures = []

    # HF 模板编码器绑定检查。
    hf_bindings = bindings.get("hf_truncation_codec", [])
    for binding in hf_bindings:
        field_path = binding.get("field_path")
        required_value = binding.get("required_value")
        actual_value = _get_nested_value(cfg, field_path)
        if actual_value != required_value:
            failures.append(f"{field_path}: expected {required_value}, got {actual_value}")

    # LF 模板编码器绑定检查。
    lf_bindings = bindings.get("lf_template_codec", [])
    for binding in lf_bindings:
        field_path = binding.get("field_path")
        required_value = binding.get("required_value")
        actual_value = _get_nested_value(cfg, field_path)
        if actual_value != required_value:
            failures.append(f"{field_path}: expected {required_value}, got {actual_value}")

    if failures:
        return {
            "check_name": check_name,
            "check_rule": check_rule,
            "result": "FAIL",
            "failure_message": f"参数绑定不匹配: {failures}"
        }

    return {
        "check_name": check_name,
        "check_rule": check_rule,
        "result": "PASS"
    }


def _get_nested_value(obj: Dict[str, Any], field_path: str) -> Any:
    """
    功能：从嵌套字典提取值。

    Get nested value from dict by dot-separated path.

    Args:
        obj: Target dict.
        field_path: Dot-separated field path (e.g., "watermark.hf.enabled").

    Returns:
        Value or None if not found.
    """
    parts = field_path.split(".")
    current = obj
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def get_evaluator_impl_identity() -> Dict[str, str]:
    """
    功能：返回 evaluator 的实现身份标识。

    Get evaluator implementation identity.

    Returns:
        Dict with impl_id, impl_version, impl_digest.
    """
    impl_digest = digests.canonical_sha256({
        "impl_id": IMPL_ID,
        "impl_version": IMPL_VERSION,
        "source_module": "main.watermarking.paper_faithfulness.alignment_evaluator"
    })
    return {
        "impl_id": IMPL_ID,
        "impl_version": IMPL_VERSION,
        "impl_digest": impl_digest
    }
