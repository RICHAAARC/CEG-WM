"""
内容链与子空间 registry

功能说明：
- 内容链（content extractor）与子空间规划器（subspace planner）的运行时注册表。
- 提供占位实现（baseline implementations）以支持系统的基本功能和测试。
"""

from __future__ import annotations

from typing import Any, Dict

from main.core import digests

from .registry_base import FactoryType, RegistryBase
from .capabilities import ImplCapabilities
from main.watermarking.content_chain.semantic_mask_provider import (
    SemanticMaskProvider,
    SEMANTIC_MASK_PROVIDER_ID,
    SEMANTIC_MASK_PROVIDER_VERSION
)
from main.watermarking.content_chain.subspace.placeholder_planner import (
    SubspacePlannerImpl,
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION
)


CONTENT_BASELINE_NOOP_ID = "content_baseline_noop_v1"
SUBSPACE_BASELINE_FULL_ID = "subspace_baseline_full_v1"
CONTENT_SEMANTIC_MASK_PROVIDER_ID = SEMANTIC_MASK_PROVIDER_ID


class ContentBaselineNoop:
    """
    功能：内容链占位实现。

    Placeholder content extractor that emits fixed evidence.

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

    def extract(self, cfg: Dict[str, Any], inputs: Dict[str, Any] | None = None, cfg_digest: str | None = None) -> Dict[str, Any]:
        """
        功能：输出占位内容证据。

        Return placeholder content evidence.

        Args:
            cfg: Config mapping.
            inputs: Optional input mapping (unused in placeholder).
            cfg_digest: Optional cfg digest (unused in placeholder).

        Returns:
            Placeholder content evidence mapping.

        Raises:
            TypeError: If cfg is not dict.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        return {
            "content_placeholder": True,
            "content_evidence": "absent",
            "content_signal": None
        }


class SubspaceBaselineFull:
    """
    功能：子空间规划占位实现。

    Placeholder subspace planner that returns fixed metadata.

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

    def plan(
        self,
        cfg: Dict[str, Any],
        mask_digest: str | None = None,
        cfg_digest: str | None = None,
        inputs: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        功能：输出占位子空间规划。

        Return placeholder subspace plan.

        Args:
            cfg: Config mapping.
            mask_digest: Optional mask digest (unused in placeholder).
            cfg_digest: Optional cfg digest (unused in placeholder).
            inputs: Optional input mapping (unused in placeholder).

        Returns:
            Placeholder subspace plan mapping.

        Raises:
            TypeError: If cfg is not dict.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        return {
            "subspace_method": "absent",
            "subspace_source": "absent",
            "subspace_frame": "full"
        }


_CONTENT_REGISTRY = RegistryBase("content_extractor")
_SUBSPACE_REGISTRY = RegistryBase("subspace_planner")


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


def _build_content_baseline_noop(cfg: Dict[str, Any]) -> ContentBaselineNoop:
    """
    功能：构造内容链占位实现。

    Build placeholder content extractor.

    Args:
        cfg: Config mapping.

    Returns:
        ContentBaselineNoop instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = "v1"
    impl_digest = _derive_impl_digest(CONTENT_BASELINE_NOOP_ID, impl_version)
    return ContentBaselineNoop(CONTENT_BASELINE_NOOP_ID, impl_version, impl_digest)


def _build_subspace_baseline_full(cfg: Dict[str, Any]) -> SubspaceBaselineFull:
    """
    功能：构造子空间占位实现。

    Build placeholder subspace planner.

    Args:
        cfg: Config mapping.

    Returns:
        SubspaceBaselineFull instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = "v1"
    impl_digest = _derive_impl_digest(SUBSPACE_BASELINE_FULL_ID, impl_version)
    return SubspaceBaselineFull(SUBSPACE_BASELINE_FULL_ID, impl_version, impl_digest)


def _build_content_semantic_mask_provider(cfg: Dict[str, Any]) -> SemanticMaskProvider:
    """
    功能：构造语义掩码提供器实现。

    Build semantic mask provider content extractor.

    Args:
        cfg: Config mapping.

    Returns:
        SemanticMaskProvider instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = SEMANTIC_MASK_PROVIDER_VERSION
    impl_digest = _derive_impl_digest(CONTENT_SEMANTIC_MASK_PROVIDER_ID, impl_version)
    return SemanticMaskProvider(CONTENT_SEMANTIC_MASK_PROVIDER_ID, impl_version, impl_digest)


def _build_subspace_planner(cfg: Dict[str, Any]) -> SubspacePlannerImpl:
    """
    功能：构造子空间规划器实现。

    Build subspace planner implementation.

    Args:
        cfg: Config mapping.

    Returns:
        SubspacePlannerImpl instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = SUBSPACE_PLANNER_VERSION
    impl_digest = _derive_impl_digest(SUBSPACE_PLANNER_ID, impl_version)
    return SubspacePlannerImpl(SUBSPACE_PLANNER_ID, impl_version, impl_digest)


_CONTENT_REGISTRY.register_factory(
    CONTENT_BASELINE_NOOP_ID,
    _build_content_baseline_noop,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=None
    )
)
_CONTENT_REGISTRY.register_factory(
    CONTENT_SEMANTIC_MASK_PROVIDER_ID,
    _build_content_semantic_mask_provider,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=(2048, 2048),
        supported_models=None
    )
)
_SUBSPACE_REGISTRY.register_factory(
    SUBSPACE_BASELINE_FULL_ID,
    _build_subspace_baseline_full,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=None
    )
)
_SUBSPACE_REGISTRY.register_factory(
    SUBSPACE_PLANNER_ID,
    _build_subspace_planner,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=None
    )
)

# 静态注册完成后立即冻结，禁止运行期修改。
_CONTENT_REGISTRY.seal()
_SUBSPACE_REGISTRY.seal()


def resolve_content_extractor(impl_id: str) -> FactoryType:
    """
    功能：解析内容链 impl_id。

    Resolve content extractor factory.

    Args:
        impl_id: Implementation identifier.

    Returns:
        Factory callable.

    Raises:
        ValueError: If impl_id is invalid or unknown.
    """
    return _CONTENT_REGISTRY.resolve_factory(impl_id)


def resolve_subspace_planner(impl_id: str) -> FactoryType:
    """
    功能：解析子空间 impl_id。

    Resolve subspace planner factory.

    Args:
        impl_id: Implementation identifier.

    Returns:
        Factory callable.

    Raises:
        ValueError: If impl_id is invalid or unknown.
    """
    return _SUBSPACE_REGISTRY.resolve_factory(impl_id)


def list_content_impl_ids() -> list[str]:
    """
    功能：列出内容链 impl_id。

    List content extractor impl_id values.

    Args:
        None.

    Returns:
        List of impl_id values.
    """
    return _CONTENT_REGISTRY.list_impl_ids()


def list_subspace_impl_ids() -> list[str]:
    """
    功能：列出子空间 impl_id。

    List subspace planner impl_id values.

    Args:
        None.

    Returns:
        List of impl_id values.
    """
    return _SUBSPACE_REGISTRY.list_impl_ids()
