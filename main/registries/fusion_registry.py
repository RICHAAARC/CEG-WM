"""
融合规则注册表

功能说明：
- 定义了一个融合规则注册表，用于管理不同的融合规则实现。
- 仅注册正式 v2 实现，不含测试脚手架基线。
"""

from __future__ import annotations

from typing import Any, Dict

from main.core import digests
from main.watermarking.fusion.decision import NeumanPearsonFusionRule

from .registry_base import FactoryType, RegistryBase
from .capabilities import ImplCapabilities


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



def _build_fusion_neyman_pearson_v2(cfg: Dict[str, Any]) -> NeumanPearsonFusionRule:
    """
    功能：构造融合规则 v2 实现。

    Build Neyman-Pearson fusion rule v2 with formal impl_id binding.

    Args:
        cfg: Config mapping.

    Returns:
        NeumanPearsonFusionRule instance bound to fusion_neyman_pearson_v2.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    impl_version = "v2"
    impl_id = "fusion_neyman_pearson_v2"
    impl_digest = _derive_impl_digest(impl_id, impl_version)
    return NeumanPearsonFusionRule(impl_id, impl_version, impl_digest)


_FUSION_REGISTRY.register_factory(
    "fusion_neyman_pearson_v2",
    _build_fusion_neyman_pearson_v2,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
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
