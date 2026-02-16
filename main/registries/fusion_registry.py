"""
融合规则 registry

功能说明：
- 提供融合规则实现的运行时注册表。
- 定义融合规则占位实现，供测试和示例使用。
"""

from __future__ import annotations

from typing import Any, Dict

from main.core import digests

from .registry_base import FactoryType, RegistryBase
from .capabilities import ImplCapabilities


FUSION_BASELINE_IDENTITY_ID = "fusion_baseline_identity_v1"


class FusionBaselineIdentity:
    """
    功能：融合规则占位实现。

    Placeholder fusion rule that returns fixed decision values.

    Args:
        impl_id: Implementation identifier.
        impl_version: Implementation version string.
        impl_digest: Implementation digest string.

    Returns:
        None.

    Raises:
        ValueError: If any input is invalid.
    """

    def __init__(self, impl_id: str, impl_version: str, impl_digest: str) -> None:
        if not isinstance(impl_id, str) or not impl_id:
            # impl_id 输入不合法，必须 fail-fast。
            raise ValueError("impl_id must be non-empty str")
        if not isinstance(impl_version, str) or not impl_version:
            # impl_version 输入不合法，必须 fail-fast。
            raise ValueError("impl_version must be non-empty str")
        if not isinstance(impl_digest, str) or not impl_digest:
            # impl_digest 输入不合法，必须 fail-fast。
            raise ValueError("impl_digest must be non-empty str")
        self.impl_id = impl_id
        self.impl_version = impl_version
        self.impl_digest = impl_digest

    def fuse(
        self,
        content_result: Dict[str, Any],
        geometry_result: Dict[str, Any],
        cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        功能：输出占位融合结果。

        Return placeholder fusion result.

        Args:
            content_result: Content evidence mapping.
            geometry_result: Geometry evidence mapping.
            cfg: Config mapping.

        Returns:
            Placeholder fusion result mapping.

        Raises:
            TypeError: If inputs are not dict.
        """
        if not isinstance(content_result, dict):
            # content_result 类型不合法，必须 fail-fast。
            raise TypeError("content_result must be dict")
        if not isinstance(geometry_result, dict):
            # geometry_result 类型不合法，必须 fail-fast。
            raise TypeError("geometry_result must be dict")
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        return {
            "decision": {
                "is_watermarked": False
            },
            "score": 0.0,
            "threshold": 0.5,
            "fusion_rule": "identity"
        }


_FUSION_REGISTRY = RegistryBase("fusion_rule")


def _derive_impl_digest(impl_id: str, impl_version: str) -> str:
    """
    功能：计算 impl_digest。

    Compute impl digest from impl_id and impl_version.

    Args:
        impl_id: Implementation identifier.
        impl_version: Implementation version string.

    Returns:
        Canonical digest string.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(impl_id, str) or not impl_id:
        # impl_id 输入不合法，必须 fail-fast。
        raise ValueError("impl_id must be non-empty str")
    if not isinstance(impl_version, str) or not impl_version:
        # impl_version 输入不合法，必须 fail-fast。
        raise ValueError("impl_version must be non-empty str")
    return digests.canonical_sha256({
        "impl_id": impl_id,
        "impl_version": impl_version
    })


def _build_fusion_baseline_identity(cfg: Dict[str, Any]) -> FusionBaselineIdentity:
    """
    功能：构造融合规则占位实现。

    Build placeholder fusion rule.

    Args:
        cfg: Config mapping.

    Returns:
        FusionBaselineIdentity instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = "v1"
    impl_digest = _derive_impl_digest(FUSION_BASELINE_IDENTITY_ID, impl_version)
    return FusionBaselineIdentity(FUSION_BASELINE_IDENTITY_ID, impl_version, impl_digest)


_FUSION_REGISTRY.register_factory(
    FUSION_BASELINE_IDENTITY_ID,
    _build_fusion_baseline_identity,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=None
    )
)

# 静态注册完成后立即冻结，禁止运行期修改。
_FUSION_REGISTRY.seal()


def resolve_fusion_rule(impl_id: str) -> FactoryType:
    """
    功能：解析融合规则 impl_id。

    Resolve fusion rule factory.

    Args:
        impl_id: Implementation identifier.

    Returns:
        Factory callable.

    Raises:
        ValueError: If impl_id is invalid or unknown.
    """
    return _FUSION_REGISTRY.resolve_factory(impl_id)


def list_fusion_impl_ids() -> list[str]:
    """
    功能：列出融合规则 impl_id。

    List fusion rule impl_id values.

    Args:
        None.

    Returns:
        List of impl_id values.
    """
    return _FUSION_REGISTRY.list_impl_ids()
