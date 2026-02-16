"""
身份与元数据权威定义

功能说明：
- ImplIdentity: 定义实现选择的字段集合，表示“选择了哪些实现”。
- ImplMeta: 在 ImplIdentity 基础上增加版本和摘要字段，表示“选择了哪些实现以及它们的版本/摘要”。
- compute_impl_meta_digest: 计算 ImplMeta 的确定性摘要，用于审计和可追溯性。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from main.core import digests
from main.core.errors import MissingRequiredFieldError


@dataclass(frozen=True)
class ImplIdentity:
    """
    功能：实现身份字段集合。
    
    Frozen mapping of impl_id for each domain. This represents "which implementations are selected".
    
    Attributes:
        content_extractor_id: Content extractor impl_id.
        geometry_extractor_id: Geometry extractor impl_id.
        fusion_rule_id: Fusion rule impl_id.
        subspace_planner_id: Subspace planner impl_id.
        sync_module_id: Sync module impl_id.
    
    Raises:
        ValueError: If any field is invalid or empty.
    """
    content_extractor_id: str
    geometry_extractor_id: str
    fusion_rule_id: str
    subspace_planner_id: str
    sync_module_id: str
    
    def __post_init__(self) -> None:
        for field_name, field_value in asdict(self).items():
            if not isinstance(field_value, str) or not field_value:
                raise ValueError(f"{field_name} must be non-empty str, got {field_value}")
    
    def as_dict(self) -> Dict[str, str]:
        """
        功能：导出 impl_identity 映射。
        
        Export ImplIdentity as dict.
        
        Args:
            None.
        
        Returns:
            Dict with all impl_id fields.
        """
        return asdict(self)


@dataclass(frozen=True)
class ImplMeta:
    """
    功能：实现元数据。
    
    Frozen augmentation of ImplIdentity with version and digest for audit and reproducibility.
    This represents "which implementations and what versions/digests".
    
    Attributes:
        impl_identity: ImplIdentity selection.
        content_extractor_version: Version string of content extractor impl.
        content_extractor_digest: Deterministic digest of content extractor code.
        geometry_extractor_version: Version string of geometry extractor impl.
        geometry_extractor_digest: Deterministic digest of geometry extractor code.
        fusion_rule_version: Version string of fusion rule impl.
        fusion_rule_digest: Deterministic digest of fusion rule code.
        subspace_planner_version: Version string of subspace planner impl.
        subspace_planner_digest: Deterministic digest of subspace planner code.
        sync_module_version: Version string of sync module impl.
        sync_module_digest: Deterministic digest of sync module code.
    
    Raises:
        TypeError: If impl_identity is not ImplIdentity.
        ValueError: If any version or digest is empty.
    """
    impl_identity: ImplIdentity
    content_extractor_version: str
    content_extractor_digest: str
    geometry_extractor_version: str
    geometry_extractor_digest: str
    fusion_rule_version: str
    fusion_rule_digest: str
    subspace_planner_version: str
    subspace_planner_digest: str
    sync_module_version: str
    sync_module_digest: str
    
    def __post_init__(self) -> None:
        if not isinstance(self.impl_identity, ImplIdentity):
            raise TypeError(f"impl_identity must be ImplIdentity, got {type(self.impl_identity)}")
        
        version_digest_pairs = [
            ("content_extractor_version", self.content_extractor_version),
            ("content_extractor_digest", self.content_extractor_digest),
            ("geometry_extractor_version", self.geometry_extractor_version),
            ("geometry_extractor_digest", self.geometry_extractor_digest),
            ("fusion_rule_version", self.fusion_rule_version),
            ("fusion_rule_digest", self.fusion_rule_digest),
            ("subspace_planner_version", self.subspace_planner_version),
            ("subspace_planner_digest", self.subspace_planner_digest),
            ("sync_module_version", self.sync_module_version),
            ("sync_module_digest", self.sync_module_digest),
        ]
        
        for field_name, field_value in version_digest_pairs:
            if not isinstance(field_value, str) or not field_value:
                raise ValueError(f"{field_name} must be non-empty str, got {field_value}")
    
    def as_dict(self) -> Dict[str, Any]:
        """
        功能：导出 impl_meta 映射。
        
        Export ImplMeta as dict.
        
        Args:
            None.
        
        Returns:
            Dict with all impl_meta fields.
        """
        result = asdict(self)
        result["impl_identity"] = self.impl_identity.as_dict()
        return result


def compute_impl_meta_digest(impl_meta: ImplMeta) -> str:
    """
    功能：计算 impl_meta 的权威摘要。
    
    Compute deterministic digest of implementation metadata for audit and reproducibility.
    This digest is frozen in records and used to trace impl configuration changes.
    
    Input domain is strictly limited to:
    - impl_id (str) for each domain
    - impl_version (str) for each domain
    - impl_source_canon_sha256 (str) for each domain (currently impl_digest)
    
    Args:
        impl_meta: ImplMeta object containing all impl versions and digests.
    
    Returns:
        SHA256 hex digest of canonical JSON representation.
    
    Raises:
        TypeError: If impl_meta is not ImplMeta.
        ValueError: If canonical representation fails.
    """
    if not isinstance(impl_meta, ImplMeta):
        raise TypeError(f"impl_meta must be ImplMeta, got {type(impl_meta)}")
    
    # 输入域硬封闭：仅包含 impl_id、version、digest。
    # 禁止任何环境相关字段。
    closed_input = {
        "content_extractor_id": impl_meta.impl_identity.content_extractor_id,
        "content_extractor_version": impl_meta.content_extractor_version,
        "content_extractor_digest": impl_meta.content_extractor_digest,
        "geometry_extractor_id": impl_meta.impl_identity.geometry_extractor_id,
        "geometry_extractor_version": impl_meta.geometry_extractor_version,
        "geometry_extractor_digest": impl_meta.geometry_extractor_digest,
        "fusion_rule_id": impl_meta.impl_identity.fusion_rule_id,
        "fusion_rule_version": impl_meta.fusion_rule_version,
        "fusion_rule_digest": impl_meta.fusion_rule_digest,
        "subspace_planner_id": impl_meta.impl_identity.subspace_planner_id,
        "subspace_planner_version": impl_meta.subspace_planner_version,
        "subspace_planner_digest": impl_meta.subspace_planner_digest,
        "sync_module_id": impl_meta.impl_identity.sync_module_id,
        "sync_module_version": impl_meta.sync_module_version,
        "sync_module_digest": impl_meta.sync_module_digest,
    }
    
    # 委托给统一的 canonical_sha256 入口。
    return digests.canonical_sha256(closed_input)


def compute_impl_identity_digest(impl_identity: ImplIdentity) -> str:
    """
    功能：计算 impl_identity 的权威摘要。
    
    Compute deterministic digest of impl identity selections (which impls are selected).
    This digest represents the "selection fingerprint" before considering versions.
    
    Args:
        impl_identity: ImplIdentity object containing all impl_id selections.
    
    Returns:
        SHA256 hex digest of canonical JSON representation.
    
    Raises:
        TypeError: If impl_identity is not ImplIdentity.
        ValueError: If canonical representation fails.
    """
    if not isinstance(impl_identity, ImplIdentity):
        raise TypeError(f"impl_identity must be ImplIdentity, got {type(impl_identity)}")
    
    identity_dict = impl_identity.as_dict()
    return digests.canonical_sha256(identity_dict)
