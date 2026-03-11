"""
内容链与子空间 registry

功能说明：
- 内容链（content extractor）与子空间规划器（subspace planner）的运行时注册表。
- 仅注册正式 v2/v3 实现，不含测试脚手架基线。
"""

from __future__ import annotations

from typing import Any, Dict

from main.core import digests

from .registry_base import FactoryType, RegistryBase
from .capabilities import ImplCapabilities
from main.watermarking.content_chain.semantic_mask_provider import (
    SemanticMaskProvider,
    SEMANTIC_MASK_PROVIDER_ID,
    SEMANTIC_MASK_PROVIDER_VERSION,
    SEMANTIC_MASK_PROVIDER_SALIENCY_POLICY_ID,
    SEMANTIC_MASK_PROVIDER_SALIENCY_POLICY_VERSION,
)
from main.watermarking.content_chain.low_freq_coder import (
    LowFreqTemplateCodecV2,
    LOW_FREQ_TEMPLATE_CODEC_V2_ID,
    LOW_FREQ_TEMPLATE_CODEC_V2_VERSION,
)
from main.watermarking.content_chain.high_freq_embedder import (
    HighFreqTemplateCodecV2,
    HIGH_FREQ_TEMPLATE_CODEC_V2_ID,
    HIGH_FREQ_TEMPLATE_CODEC_V2_VERSION,
)
from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    SubspacePlannerV2,
    SUBSPACE_PLANNER_V2_ID,
    SUBSPACE_PLANNER_V2_VERSION,
)
from main.watermarking.content_chain.unified_content_extractor import (
    UnifiedContentExtractorV2,
    UNIFIED_CONTENT_EXTRACTOR_V2_ID,
    UNIFIED_CONTENT_EXTRACTOR_V2_VERSION,
)


CONTENT_SEMANTIC_MASK_PROVIDER_ID = SEMANTIC_MASK_PROVIDER_ID


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


def _build_content_semantic_mask_provider_saliency_policy(cfg: Dict[str, Any]) -> SemanticMaskProvider:
    """
    功能：构造 saliency_source 策略版语义掩码提供器实现。

    Build saliency-source-policy semantic mask provider implementation.

    Args:
        cfg: Config mapping.

    Returns:
        SemanticMaskProvider instance.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    impl_version = SEMANTIC_MASK_PROVIDER_SALIENCY_POLICY_VERSION
    impl_digest = _derive_impl_digest(SEMANTIC_MASK_PROVIDER_SALIENCY_POLICY_ID, impl_version)
    return SemanticMaskProvider(SEMANTIC_MASK_PROVIDER_SALIENCY_POLICY_ID, impl_version, impl_digest)


def _build_low_freq_template_codec_v2(cfg: Dict[str, Any]) -> LowFreqTemplateCodecV2:
    """
    功能：构造 LF 低频模板编码器 v2 实现。

    Build LF low-frequency template codec v2 (additive pseudogaussian template injection).

    Args:
        cfg: Config mapping.

    Returns:
        LowFreqTemplateCodecV2 instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    impl_version = LOW_FREQ_TEMPLATE_CODEC_V2_VERSION
    impl_digest = _derive_impl_digest(LOW_FREQ_TEMPLATE_CODEC_V2_ID, impl_version)
    return LowFreqTemplateCodecV2(LOW_FREQ_TEMPLATE_CODEC_V2_ID, impl_version, impl_digest)


def _build_high_freq_template_codec_v2(cfg: Dict[str, Any]) -> HighFreqTemplateCodecV2:
    """
    功能：构造 HF 高频模板编码器 v2 实现。

    Build HF high-frequency template codec v2 (keyed Rademacher template + correlation detection).

    Args:
        cfg: Config mapping.

    Returns:
        HighFreqTemplateCodecV2 instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    impl_version = HIGH_FREQ_TEMPLATE_CODEC_V2_VERSION
    impl_digest = _derive_impl_digest(HIGH_FREQ_TEMPLATE_CODEC_V2_ID, impl_version)
    return HighFreqTemplateCodecV2(HIGH_FREQ_TEMPLATE_CODEC_V2_ID, impl_version, impl_digest)


def _build_unified_content_extractor_v2(cfg: Dict[str, Any]) -> UnifiedContentExtractorV2:
    """
    功能：构造统一内容链提取器 v2 实现。

    Build unified content extractor v2 implementation.

    Args:
        cfg: Config mapping.

    Returns:
        UnifiedContentExtractorV2 instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    impl_version = UNIFIED_CONTENT_EXTRACTOR_V2_VERSION
    impl_digest = _derive_impl_digest(UNIFIED_CONTENT_EXTRACTOR_V2_ID, impl_version)
    return UnifiedContentExtractorV2(UNIFIED_CONTENT_EXTRACTOR_V2_ID, impl_version, impl_digest)


def _build_subspace_planner_v2(cfg: Dict[str, Any]) -> SubspacePlannerV2:
    """
    功能：构造子空间规划器 v2 实现。

    Build subspace planner v2 (with semantic domain annotations).

    Args:
        cfg: Config mapping.

    Returns:
        SubspacePlannerV2 instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    impl_version = SUBSPACE_PLANNER_V2_VERSION
    impl_digest = _derive_impl_digest(SUBSPACE_PLANNER_V2_ID, impl_version)
    return SubspacePlannerV2(SUBSPACE_PLANNER_V2_ID, impl_version, impl_digest)


_CONTENT_REGISTRY.register_factory(
    CONTENT_SEMANTIC_MASK_PROVIDER_ID,
    _build_content_semantic_mask_provider,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution="2048x2048",
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
    )
)
_CONTENT_REGISTRY.register_factory(
    SEMANTIC_MASK_PROVIDER_SALIENCY_POLICY_ID,
    _build_content_semantic_mask_provider_saliency_policy,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution="2048x2048",
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
    )
)
_CONTENT_REGISTRY.register_factory(
    LOW_FREQ_TEMPLATE_CODEC_V2_ID,
    _build_low_freq_template_codec_v2,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
    )
)
_CONTENT_REGISTRY.register_factory(
    HIGH_FREQ_TEMPLATE_CODEC_V2_ID,
    _build_high_freq_template_codec_v2,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
    )
)
_CONTENT_REGISTRY.register_factory(
    UNIFIED_CONTENT_EXTRACTOR_V2_ID,
    _build_unified_content_extractor_v2,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution="2048x2048",
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
    )
)
_SUBSPACE_REGISTRY.register_factory(
    SUBSPACE_PLANNER_V2_ID,
    _build_subspace_planner_v2,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
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
