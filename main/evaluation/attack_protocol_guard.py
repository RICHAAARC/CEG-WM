"""
File purpose: 攻击协议可实现性门禁（protocol-implementation consistency gate）。
Module type: Core innovation module

设计边界：
1. 本模块仅做协议—实现一致性校验，不改变 attack_runner 实现逻辑。
2. 失败必须 fail-fast，错误信息包含证据路径（条件键与字段路径）。
3. 协议加载与覆盖计算必须使用仓库标准接口，禁止硬编码。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from main.core import digests


def assert_attack_protocol_is_implementable(
    protocol_spec: Dict[str, Any],
    coverage_manifest: Dict[str, Any],
) -> None:
    """
    功能：校验攻击协议可实现性（评测前 fail-fast）。

    Assert attack protocol is fully implementable before evaluation execution.
    Validates that all declared families, params_versions, and composite steps
    are supported by attack_runner implementation.

    Args:
        protocol_spec: Attack protocol specification from protocol_loader.
        coverage_manifest: Coverage manifest from attack_coverage.compute_attack_coverage_manifest().

    Returns:
        None (succeeds silently).

    Raises:
        TypeError: If inputs are invalid.
        RuntimeError: If protocol contains unimplementable attacks (fail-fast with evidence path).
    """
    if not isinstance(protocol_spec, dict):
        # protocol_spec 类型不合法，必须 fail-fast。
        raise TypeError("protocol_spec must be dict")
    if not isinstance(coverage_manifest, dict):
        # coverage_manifest 类型不合法，必须 fail-fast。
        raise TypeError("coverage_manifest must be dict")

    supported_families = _extract_supported_families(coverage_manifest)
    protocol_families = _extract_protocol_families(protocol_spec)

    # (1) 校验所有协议族在实现支持列表中
    unsupported_families = _find_unsupported_families(protocol_families, supported_families)
    if unsupported_families:
        evidence = {
            "unsupported_families": sorted(unsupported_families),
            "supported_families": sorted(supported_families),
            "protocol_families": sorted(protocol_families),
            "evidence_path": "protocol_spec.families",
        }
        raise RuntimeError(
            f"Attack protocol implementability validation FAILED: "
            f"protocol declares unsupported families: {sorted(unsupported_families)}. "
            f"Supported families: {sorted(supported_families)}. "
            f"Evidence: {evidence}"
        )

    # (2) 校验 params_versions 必需字段存在性与类型正确性
    params_versions = protocol_spec.get("params_versions", {})
    if not isinstance(params_versions, dict):
        # params_versions 结构不合法，必须 fail-fast。
        raise TypeError(
            f"protocol_spec.params_versions must be dict, got {type(params_versions).__name__}"
        )

    for condition_key, params_spec in params_versions.items():
        if not isinstance(condition_key, str) or not condition_key:
            # 条件键不合法，必须 fail-fast。
            raise ValueError(
                f"protocol_spec.params_versions condition_key must be non-empty str, "
                f"got {type(condition_key).__name__}: {condition_key}"
            )
        if not isinstance(params_spec, dict):
            # params_spec 类型不合法，必须 fail-fast。
            raise TypeError(
                f"protocol_spec.params_versions[{condition_key!r}] must be dict, "
                f"got {type(params_spec).__name__}"
            )

        _validate_params_spec_schema(condition_key, params_spec, supported_families)

    # (3) 校验 composite 的递归可实现性
    families_dict = protocol_spec.get("families", {})
    if isinstance(families_dict, dict):
        composite_spec = families_dict.get("composite")
        if isinstance(composite_spec, dict):
            _validate_composite_implementability(composite_spec, supported_families)


def _extract_supported_families(coverage_manifest: Dict[str, Any]) -> Set[str]:
    """
    功能：提取实现支持的攻击族集合。

    Extract set of supported attack families from coverage manifest.

    Args:
        coverage_manifest: Coverage manifest mapping.

    Returns:
        Set of supported family names (normalized to lowercase).

    Raises:
        TypeError: If coverage_manifest structure is invalid.
    """
    if not isinstance(coverage_manifest, dict):
        raise TypeError("coverage_manifest must be dict")

    supported = coverage_manifest.get("supported_families")
    if not isinstance(supported, list):
        raise TypeError(
            f"coverage_manifest.supported_families must be list, "
            f"got {type(supported).__name__}"
        )

    families_set: Set[str] = set()
    for item in supported:
        if isinstance(item, str) and item:
            normalized = _normalize_family_name(item)
            families_set.add(normalized)

    return families_set


def _extract_protocol_families(protocol_spec: Dict[str, Any]) -> Set[str]:
    """
    功能：提取协议声明的攻击族集合。

    Extract set of declared attack families from protocol spec.

    Args:
        protocol_spec: Attack protocol specification.

    Returns:
        Set of declared family names (normalized to lowercase).

    Raises:
        TypeError: If protocol_spec structure is invalid.
    """
    if not isinstance(protocol_spec, dict):
        raise TypeError("protocol_spec must be dict")

    families_dict = protocol_spec.get("families", {})
    if not isinstance(families_dict, dict):
        return set()

    families_set: Set[str] = set()
    for family_name in families_dict.keys():
        if isinstance(family_name, str) and family_name:
            normalized = _normalize_family_name(family_name)
            families_set.add(normalized)

    return families_set


def _normalize_family_name(family: str) -> str:
    """
    功能：规范化攻击族名称。

    Normalize family name for consistent comparison.

    Args:
        family: Raw family name string.

    Returns:
        Normalized family name (lowercase, underscores preserved).
    """
    if not isinstance(family, str) or not family:
        return "<absent>"
    return family.strip().lower()


def _find_unsupported_families(
    protocol_families: Set[str],
    supported_families: Set[str],
) -> Set[str]:
    """
    功能：查找协议中未被实现支持的攻击族。

    Find protocol families that are not supported by implementation.

    Args:
        protocol_families: Set of protocol-declared families.
        supported_families: Set of implementation-supported families.

    Returns:
        Set of unsupported family names.
    """
    # 特殊处理 jpeg 别名：协议使用 "jpeg"，实现使用 "jpeg_compression"
    normalized_supported = set(supported_families)
    normalized_supported.add("jpeg")  # 允许协议使用 "jpeg" 别名

    return protocol_families - normalized_supported


def _validate_params_spec_schema(
    condition_key: str,
    params_spec: Dict[str, Any],
    supported_families: Set[str],
) -> None:
    """
    功能：校验 params_version 的 schema 必需字段存在性与类型正确性。

    Validate params_version schema compliance for one condition_key.

    Args:
        condition_key: Condition key in format family::version.
        params_spec: Params specification mapping.
        supported_families: Set of supported family names.

    Returns:
        None (succeeds silently).

    Raises:
        RuntimeError: If params_spec is invalid (fail-fast with evidence path).
    """
    if not isinstance(params_spec, dict):
        raise TypeError(f"params_spec for {condition_key!r} must be dict")

    family_value = params_spec.get("family")
    if not isinstance(family_value, str) or not family_value:
        evidence = {
            "condition_key": condition_key,
            "evidence_path": f"params_versions.{condition_key}.family",
            "reason": "missing_or_invalid_family_field",
        }
        raise RuntimeError(
            f"Attack protocol params_version validation FAILED: "
            f"params_versions[{condition_key!r}].family must be non-empty str. "
            f"Evidence: {evidence}"
        )

    normalized_family = _normalize_family_name(family_value)
    if normalized_family not in supported_families and normalized_family != "jpeg":
        evidence = {
            "condition_key": condition_key,
            "family": family_value,
            "normalized_family": normalized_family,
            "supported_families": sorted(supported_families),
            "evidence_path": f"params_versions.{condition_key}.family",
        }
        raise RuntimeError(
            f"Attack protocol params_version validation FAILED: "
            f"params_versions[{condition_key!r}].family={family_value!r} is not supported. "
            f"Supported families: {sorted(supported_families)}. "
            f"Evidence: {evidence}"
        )

    params_obj = params_spec.get("params")
    if not isinstance(params_obj, dict):
        evidence = {
            "condition_key": condition_key,
            "evidence_path": f"params_versions.{condition_key}.params",
            "reason": "missing_or_invalid_params_field",
        }
        raise RuntimeError(
            f"Attack protocol params_version validation FAILED: "
            f"params_versions[{condition_key!r}].params must be dict. "
            f"Evidence: {evidence}"
        )


def _validate_composite_implementability(
    composite_spec: Dict[str, Any],
    supported_families: Set[str],
) -> None:
    """
    功能：校验 composite 攻击的递归可实现性。

    Validate composite attack implementability (recursive family checks).

    Args:
        composite_spec: Composite family specification.
        supported_families: Set of supported family names.

    Returns:
        None (succeeds silently).

    Raises:
        RuntimeError: If composite steps contain unsupported families (fail-fast with evidence).
    """
    if not isinstance(composite_spec, dict):
        raise TypeError("composite_spec must be dict")

    params_versions = composite_spec.get("params_versions", {})
    if not isinstance(params_versions, dict):
        return

    for version_key, version_spec in params_versions.items():
        if not isinstance(version_spec, dict):
            continue

        params_obj = version_spec.get("params", {})
        if not isinstance(params_obj, dict):
            continue

        steps = params_obj.get("steps", [])
        if not isinstance(steps, list):
            continue

        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue

            step_family = step.get("family")
            if not isinstance(step_family, str) or not step_family:
                evidence = {
                    "composite_version": version_key,
                    "step_index": step_index,
                    "evidence_path": f"families.composite.params_versions.{version_key}.params.steps[{step_index}].family",
                    "reason": "missing_or_invalid_step_family",
                }
                raise RuntimeError(
                    f"Attack protocol composite validation FAILED: "
                    f"composite step {step_index} in version {version_key!r} missing family. "
                    f"Evidence: {evidence}"
                )

            normalized_step_family = _normalize_family_name(step_family)
            if normalized_step_family not in supported_families and normalized_step_family != "jpeg":
                evidence = {
                    "composite_version": version_key,
                    "step_index": step_index,
                    "step_family": step_family,
                    "normalized_family": normalized_step_family,
                    "supported_families": sorted(supported_families),
                    "evidence_path": f"families.composite.params_versions.{version_key}.params.steps[{step_index}].family",
                }
                raise RuntimeError(
                    f"Attack protocol composite validation FAILED: "
                    f"composite step {step_index} family={step_family!r} is not supported. "
                    f"Supported families: {sorted(supported_families)}. "
                    f"Evidence: {evidence}"
                )
