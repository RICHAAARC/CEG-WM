"""
File purpose: 攻击协议事实源加载与标准化。
Module type: General module
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from main.core import config_loader
from main.core import digests


def load_attack_protocol_spec(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    功能：从 configs 事实源加载并标准化攻击协议。

    Load attack protocol specification from fact source (configs/attack_protocol.yaml).
    Uses config_loader as the single YAML loading entry point.

    Args:
        cfg: Optional configuration mapping. If provided, can override protocol path via
             cfg["evaluate"]["attack_protocol_path"].

    Returns:
        AttackProtocolSpec as dict containing:
            - version: Protocol version string (e.g., "attack_protocol_v1")
            - families: Dict of family definitions
            - params_versions: Dict of versioned parameter sets
            - attack_protocol_digest: Canonical SHA256 of protocol object
            - attack_protocol_file_sha256: SHA256 of raw YAML file

    Raises:
        ValueError: If protocol cannot be loaded or parsed.
    """
    # 构造默认规范结构。
    default_spec = {
        "version": "<absent>",
        "family_field_candidates": ["attack_family", "attack.family", "attack.type"],
        "params_version_field_candidates": ["attack_params_version", "attack.params_version"],
        "families": {},
        "params_versions": {},
        "attack_protocol_digest": "<absent>",
        "attack_protocol_file_sha256": "<absent>",
    }

    # 解析加载路径（支持 cfg 覆盖）。
    fact_source_path = config_loader.ATTACK_PROTOCOL_PATH
    if isinstance(cfg, dict):
        evaluate_cfg = cfg.get("evaluate") if isinstance(cfg.get("evaluate"), dict) else {}
        cfg_path = evaluate_cfg.get("attack_protocol_path")
        if isinstance(cfg_path, str) and cfg_path:
            fact_source_path = cfg_path

    # 加载 YAML 事实源（使用唯一入口）。
    merged_spec = dict(default_spec)
    try:
        loaded_obj, provenance = config_loader.load_yaml_with_provenance(fact_source_path)
        if isinstance(loaded_obj, dict):
            # 仅授权字段可从 YAML 合并到 spec（append-only 策略）。
            for key in [
                "version",
                "family_field_candidates",
                "params_version_field_candidates",
                "families",
                "params_versions",
            ]:
                if key in loaded_obj:
                    merged_spec[key] = loaded_obj[key]
            # 注入 digest 信息（溯源字段）。
            merged_spec["attack_protocol_digest"] = provenance.canon_sha256
            merged_spec["attack_protocol_file_sha256"] = provenance.file_sha256
    except Exception as e:
        # 加载失败时返回默认值，不中断流程。
        # 注意：真实场景应记录警告日志。
        pass

    # 应用 cfg 层补充（可覆盖版本与字段候选，但保留 digest）。
    if isinstance(cfg, dict):
        evaluate_cfg = cfg.get("evaluate") if isinstance(cfg.get("evaluate"), dict) else {}

        cfg_version = evaluate_cfg.get("attack_protocol_version")
        if isinstance(cfg_version, str) and cfg_version:
            merged_spec["version"] = cfg_version

        cfg_family_candidates = evaluate_cfg.get("attack_family_field_candidates")
        if isinstance(cfg_family_candidates, list) and len(cfg_family_candidates) > 0:
            merged_spec["family_field_candidates"] = cfg_family_candidates

        cfg_params_candidates = evaluate_cfg.get("attack_params_version_field_candidates")
        if isinstance(cfg_params_candidates, list) and len(cfg_params_candidates) > 0:
            merged_spec["params_version_field_candidates"] = cfg_params_candidates

    return merged_spec


def compute_attack_protocol_digest(protocol_spec: Dict[str, Any]) -> str:
    """
    功能：计算协议规范的 canonical digest。

    Compute canonical SHA256 digest of attack protocol specification.
    Includes version, families, and params_versions.

    Args:
        protocol_spec: Attack protocol spec dict.

    Returns:
        Canonical SHA256 hex string.

    Raises:
        TypeError: If protocol_spec is invalid.
    """
    if not isinstance(protocol_spec, dict):
        raise TypeError("protocol_spec must be dict")

    # 包含: version, families, params_versions（不包含 digest 自身以避免循环）。
    canon_obj = {
        "version": protocol_spec.get("version", "<absent>"),
        "families": protocol_spec.get("families", {}),
        "params_versions": protocol_spec.get("params_versions", {}),
    }
    return digests.canonical_sha256(canon_obj)


def get_protocol_version(protocol_spec: Dict[str, Any]) -> str:
    """
    功能：安全提取协议版本字符串。

    Get protocol version string from spec with safety check.

    Args:
        protocol_spec: Attack protocol spec dict.

    Returns:
        Version string (e.g., "attack_protocol_v1") or "<absent>".
    """
    if not isinstance(protocol_spec, dict):
        return "<absent>"
    version = protocol_spec.get("version")
    if isinstance(version, str) and version:
        return version
    return "<absent>"


def get_family_candidates(protocol_spec: Dict[str, Any]) -> list[str]:
    """
    功能：安全提取 family 字段候选列表。

    Get family field candidates from spec with fallback to defaults.

    Args:
        protocol_spec: Attack protocol spec dict.

    Returns:
        List of field path candidates (e.g., ["attack_family", "attack.family"]).
    """
    if not isinstance(protocol_spec, dict):
        return ["attack_family", "attack.family", "attack.type"]
    candidates = protocol_spec.get("family_field_candidates")
    if isinstance(candidates, list) and len(candidates) > 0:
        return candidates
    return ["attack_family", "attack.family", "attack.type"]


def get_params_version_candidates(protocol_spec: Dict[str, Any]) -> list[str]:
    """
    功能：安全提取 params_version 字段候选列表。

    Get params_version field candidates from spec with fallback to defaults.

    Args:
        protocol_spec: Attack protocol spec dict.

    Returns:
        List of field path candidates.
    """
    if not isinstance(protocol_spec, dict):
        return ["attack_params_version", "attack.params_version"]
    candidates = protocol_spec.get("params_version_field_candidates")
    if isinstance(candidates, list) and len(candidates) > 0:
        return candidates
    return ["attack_params_version", "attack.params_version"]
