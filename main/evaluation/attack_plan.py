"""
File purpose: 攻击计划生成与可复算性管理。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from main.core import digests

from . import protocol_loader


class AttackPlan:
    """
    功能：评测用攻击计划标准化表示。

    Standardized attack plan for reproducible evaluation.
    
    Contains versioned protocol and all unique attack conditions.
    """

    def __init__(
        self,
        protocol_version: str,
        conditions: List[str],
        protocol_digest: Optional[str] = None,
    ):
        """
        Args:
            protocol_version: Protocol version string.
            conditions: List of unique conditions (format: "family::params_version").
            protocol_digest: Optional protocol digest for audit.
        """
        if not isinstance(protocol_version, str):
            raise TypeError("protocol_version must be str")
        if not isinstance(conditions, list):
            raise TypeError("conditions must be list")

        self.protocol_version = protocol_version
        self.conditions = sorted(conditions)  # 确保拓扑顺序稳定。
        self.protocol_digest = protocol_digest or "<absent>"

    def as_dict(self) -> Dict[str, Any]:
        """
        功能：将计划转换为字典表示。

        Convert plan to dictionary representation.

        Returns:
            Plan dict with stable ordering.
        """
        return {
            "protocol_version": self.protocol_version,
            "conditions": self.conditions,
            "protocol_digest": self.protocol_digest,
        }

    def compute_digest(self) -> str:
        """
        功能：计算计划的 canonical digest。

        Compute canonical SHA256 digest of plan.
        Digest includes protocol_version and sorted conditions.

        Returns:
            Canonical SHA256 hex string.
        """
        canon_obj = {
            "protocol_version": self.protocol_version,
            "conditions": sorted(self.conditions),
        }
        return digests.canonical_sha256(canon_obj)


def generate_attack_plan(
    protocol_spec: Dict[str, Any],
) -> AttackPlan:
    """
    功能：从协议规范生成可复算的攻击计划。

    Generate deterministic attack plan from protocol specification.
    Extracts all unique attack_family::params_version combinations.

    Args:
        protocol_spec: Attack protocol spec from protocol_loader.

    Returns:
        AttackPlan object.

    Raises:
        TypeError: If protocol_spec is invalid.
    """
    if not isinstance(protocol_spec, dict):
        raise TypeError("protocol_spec must be dict")

    protocol_version = protocol_loader.get_protocol_version(protocol_spec)
    protocol_digest = protocol_spec.get("attack_protocol_digest", "<absent>")

    # 从 params_versions 提取唯一的 family::params_version 条件。
    conditions: List[str] = []
    params_versions = protocol_spec.get("params_versions")
    if isinstance(params_versions, dict):
        # params_versions 结构：{family::version: {...}}
        for condition_key in params_versions.keys():
            if isinstance(condition_key, str) and "::" in condition_key:
                # 条件键已为 family::params_version 格式。
                if condition_key not in conditions:
                    conditions.append(condition_key)

    # 若 params_versions 为空，从 families 推导（兼容模式）。
    if not conditions:
        families = protocol_spec.get("families")
        if isinstance(families, dict):
            for family_name, family_spec in families.items():
                if isinstance(family_spec, dict):
                    params_versions_in_family = family_spec.get("params_versions", {})
                    if isinstance(params_versions_in_family, dict):
                        for param_version_name in params_versions_in_family.keys():
                            condition_key = f"{family_name}::{param_version_name}"
                            if condition_key not in conditions:
                                conditions.append(condition_key)

    plan = AttackPlan(
        protocol_version=protocol_version,
        conditions=conditions,
        protocol_digest=protocol_digest,
    )
    return plan


def validate_attack_plan(plan: AttackPlan) -> bool:
    """
    功能：验证攻击计划的基本有效性。

    Validate attack plan format and content.

    Args:
        plan: AttackPlan object.

    Returns:
        True if plan is valid.

    Raises:
        ValueError: If plan is invalid.
    """
    if not isinstance(plan, AttackPlan):
        raise TypeError("plan must be AttackPlan instance")

    if not plan.protocol_version or plan.protocol_version == "<absent>":
        raise ValueError("plan.protocol_version must be non-empty")

    if not isinstance(plan.conditions, list) or len(plan.conditions) == 0:
        raise ValueError("plan.conditions must be non-empty list")

    for condition in plan.conditions:
        if not isinstance(condition, str) or "::" not in condition:
            raise ValueError(f"Invalid condition format: {condition}; expected family::params_version")

    return True
