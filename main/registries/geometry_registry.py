"""
几何链与同步模块 registry

功能说明：
- 提供几何链提取器和同步模块实现的运行时注册表。
- 定义几何链和同步模块基线实现，供测试和示例使用。
"""

from __future__ import annotations

from typing import Any, Dict

from main.core import digests
from main.watermarking.geometry_chain.align_invariance_extractor import GeometryAlignInvarianceExtractor
from main.watermarking.geometry_chain.attention_anchor_extractor import AttentionAnchorExtractor
from main.watermarking.geometry_chain.sync.latent_sync_template import LatentSyncGeometryExtractor

from .registry_base import FactoryType, RegistryBase
from .capabilities import ImplCapabilities


GEOMETRY_BASELINE_IDENTITY_ID = "geometry_baseline_identity_v1"
GEOMETRY_ATTENTION_ANCHOR_SD3_ID = "geometry_attention_anchor_sd3_v1"
GEOMETRY_LATENT_SYNC_SD3_ID = "geometry_latent_sync_sd3_v1"
GEOMETRY_ALIGN_INVARIANCE_SD3_ID = "geometry_align_invariance_sd3_v1"
SYNC_BASELINE_ID = "geometry_sync_baseline_v1"


class GeometryBaselineIdentity:
    """
    功能：几何链基线实现。

    Baseline geometry extractor that emits fixed evidence.

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

    def extract(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：输出基线几何证据。

        Return baseline geometry evidence.

        Args:
            cfg: Config mapping.

        Returns:
            Baseline geometry evidence mapping.

        Raises:
            TypeError: If cfg is not dict.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        return {
            "geometry_baseline": True,
            "geometry_evidence": "absent",
            "geometry_signal": None
        }


class SyncBaseline:
    """
    功能：同步模块基线实现。

    Baseline sync module that returns fixed status.

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

    def sync(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：输出基线同步状态。

        Return baseline sync status.

        Args:
            cfg: Config mapping.

        Returns:
            Baseline sync status mapping.

        Raises:
            TypeError: If cfg is not dict.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        return {
            "sync_status": "unavailable",
            "sync_success": False
        }


_GEOMETRY_REGISTRY = RegistryBase("geometry_extractor")
_SYNC_REGISTRY = RegistryBase("sync_module")


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


def _build_geometry_baseline_identity(cfg: Dict[str, Any]) -> GeometryBaselineIdentity:
    """
    功能：构造几何链基线实现。

    Build baseline geometry extractor.

    Args:
        cfg: Config mapping.

    Returns:
        GeometryBaselineIdentity instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = "v1"
    impl_digest = _derive_impl_digest(GEOMETRY_BASELINE_IDENTITY_ID, impl_version)
    return GeometryBaselineIdentity(GEOMETRY_BASELINE_IDENTITY_ID, impl_version, impl_digest)


def _build_geometry_attention_anchor_sd3(cfg: Dict[str, Any]) -> AttentionAnchorExtractor:
    """
    功能：构造 SD3 attention 锚点提取实现。

    Build SD3 transformer attention anchor extractor.

    Args:
        cfg: Config mapping.

    Returns:
        AttentionAnchorExtractor instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = "v1"
    impl_digest = _derive_impl_digest(GEOMETRY_ATTENTION_ANCHOR_SD3_ID, impl_version)
    return AttentionAnchorExtractor(GEOMETRY_ATTENTION_ANCHOR_SD3_ID, impl_version, impl_digest)


def _build_geometry_latent_sync_sd3(cfg: Dict[str, Any]) -> LatentSyncGeometryExtractor:
    """
    功能：构造 SD3 latent 同步模板几何实现。

    Build SD3 latent sync geometry extractor.

    Args:
        cfg: Config mapping.

    Returns:
        LatentSyncGeometryExtractor instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = "v1"
    impl_digest = _derive_impl_digest(GEOMETRY_LATENT_SYNC_SD3_ID, impl_version)
    return LatentSyncGeometryExtractor(GEOMETRY_LATENT_SYNC_SD3_ID, impl_version, impl_digest)


def _build_geometry_align_invariance_sd3(cfg: Dict[str, Any]) -> GeometryAlignInvarianceExtractor:
    """
    功能：构造 SD3 几何对齐与不变性评分实现。

    Build SD3 geometry align-and-invariance extractor.

    Args:
        cfg: Config mapping.

    Returns:
        GeometryAlignInvarianceExtractor instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = "v2"
    impl_digest = _derive_impl_digest(GEOMETRY_ALIGN_INVARIANCE_SD3_ID, impl_version)
    return GeometryAlignInvarianceExtractor(GEOMETRY_ALIGN_INVARIANCE_SD3_ID, impl_version, impl_digest)


def _build_sync_baseline(cfg: Dict[str, Any]) -> SyncBaseline:
    """
    功能：构造同步基线实现。

    Build baseline sync module.

    Args:
        cfg: Config mapping.

    Returns:
        SyncBaseline instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = "v1"
    impl_digest = _derive_impl_digest(SYNC_BASELINE_ID, impl_version)
    return SyncBaseline(SYNC_BASELINE_ID, impl_version, impl_digest)


_GEOMETRY_REGISTRY.register_factory(
    GEOMETRY_BASELINE_IDENTITY_ID,
    _build_geometry_baseline_identity,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
    )
)
_GEOMETRY_REGISTRY.register_factory(
    GEOMETRY_ATTENTION_ANCHOR_SD3_ID,
    _build_geometry_attention_anchor_sd3,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
    )
)
_GEOMETRY_REGISTRY.register_factory(
    GEOMETRY_LATENT_SYNC_SD3_ID,
    _build_geometry_latent_sync_sd3,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
    )
)
_GEOMETRY_REGISTRY.register_factory(
    GEOMETRY_ALIGN_INVARIANCE_SD3_ID,
    _build_geometry_align_invariance_sd3,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
    )
)
_SYNC_REGISTRY.register_factory(
    SYNC_BASELINE_ID,
    _build_sync_baseline,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
    )
)

# 静态注册完成后立即冻结，禁止运行期修改。
_GEOMETRY_REGISTRY.seal()
_SYNC_REGISTRY.seal()


def resolve_geometry_extractor(impl_id: str) -> FactoryType:
    """
    功能：解析几何链 impl_id。

    Resolve geometry extractor factory.

    Args:
        impl_id: Implementation identifier.

    Returns:
        Factory callable.

    Raises:
        ValueError: If impl_id is invalid or unknown.
    """
    return _GEOMETRY_REGISTRY.resolve_factory(impl_id)


def resolve_sync_module(impl_id: str) -> FactoryType:
    """
    功能：解析同步模块 impl_id。

    Resolve sync module factory.

    Args:
        impl_id: Implementation identifier.

    Returns:
        Factory callable.

    Raises:
        ValueError: If impl_id is invalid or unknown.
    """
    return _SYNC_REGISTRY.resolve_factory(impl_id)


def list_geometry_impl_ids() -> list[str]:
    """
    功能：列出几何链 impl_id。

    List geometry extractor impl_id values.

    Args:
        None.

    Returns:
        List of impl_id values.
    """
    return _GEOMETRY_REGISTRY.list_impl_ids()


def list_sync_impl_ids() -> list[str]:
    """
    功能：列出同步模块 impl_id。

    List sync module impl_id values.

    Args:
        None.

    Returns:
        List of impl_id values.
    """
    return _SYNC_REGISTRY.list_impl_ids()
