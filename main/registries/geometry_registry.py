"""  
几何链与同步模块 registry

功能说明：
- 提供几何链提取器和同步模块实现的运行时注册表。
- 仅注册正式 v2/v3 实现，不含测试脚手架基线。
"""

from __future__ import annotations

from typing import Any, Dict

from main.core import digests
from main.watermarking.geometry_chain.attention_anchor_extractor import (
    AttentionAnchorMapRelationExtractor,
    ATTENTION_ANCHOR_EXTRACTOR_ID,
    ATTENTION_ANCHOR_EXTRACTOR_VERSION,
)
from main.watermarking.geometry_chain.sync.latent_sync_template import (
    GeometryLatentSyncSD3,
    GEOMETRY_LATENT_SYNC_SD3_ID,
    GEOMETRY_LATENT_SYNC_SD3_VERSION,
    SyncResult,
    SyncRuntimeContext,
)

from .registry_base import FactoryType, RegistryBase
from .capabilities import ImplCapabilities


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


def _build_attention_anchor_extractor(cfg: Dict[str, Any]) -> AttentionAnchorMapRelationExtractor:
    """
    功能：构造 attention anchor map relation 实现（无 proxy 模式）。

    Build attention anchor map relation geometry extractor (no proxy mode).

    Args:
        cfg: Config mapping.

    Returns:
        AttentionAnchorMapRelationExtractor instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    impl_version = ATTENTION_ANCHOR_EXTRACTOR_VERSION
    impl_digest = _derive_impl_digest(ATTENTION_ANCHOR_EXTRACTOR_ID, impl_version)
    return AttentionAnchorMapRelationExtractor(ATTENTION_ANCHOR_EXTRACTOR_ID, impl_version, impl_digest)


class SyncGeometryLatentSyncSD3:
    """
    功能：SD3 latent sync 同步模块包装器。

    Sync module wrapper for GeometryLatentSyncSD3 extractor providing
    sync() and sync_with_context() interfaces.

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
            raise ValueError("impl_id must be non-empty str")
        if not isinstance(impl_version, str) or not impl_version:
            raise ValueError("impl_version must be non-empty str")
        if not isinstance(impl_digest, str) or not impl_digest:
            raise ValueError("impl_digest must be non-empty str")
        self.impl_id = impl_id
        self.impl_version = impl_version
        self.impl_digest = impl_digest
        self._extractor = GeometryLatentSyncSD3(impl_id, impl_version, impl_digest)

    def sync(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：返回结构化同步状态（无上下文）。

        Return sync status without runtime context.

        Args:
            cfg: Config mapping.

        Returns:
            Sync status mapping.

        Raises:
            TypeError: If cfg is not dict.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        result = self._extractor.extract(cfg, inputs=None, sync_ctx=None)
        status = self._normalize_status_token(result.get("status"))
        return {
            "sync_status": status,
            "sync_success": status == "ok",
            "sync_digest": result.get("sync_digest"),
            "geo_score": result.get("geo_score"),
            "geometry_absent_reason": result.get("geometry_absent_reason"),
            "geometry_failure_reason": result.get("geometry_failure_reason"),
            "sync_quality_semantics": result.get("sync_quality_semantics"),
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
        }

    def sync_with_context(
        self,
        cfg: Dict[str, Any],
        context: SyncRuntimeContext,
        runtime_inputs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        功能：使用运行期上下文提取同步状态（v3）。

        Extract sync status using runtime context with template_match_score as geo_score.

        Args:
            cfg: Config mapping.
            context: Sync runtime context.
            runtime_inputs: Optional runtime inputs dict.

        Returns:
            Sync status mapping with template match metrics.

        Raises:
            TypeError: If inputs are invalid.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if not isinstance(context, SyncRuntimeContext):
            raise TypeError("context must be SyncRuntimeContext")
        if runtime_inputs is not None and not isinstance(runtime_inputs, dict):
            raise TypeError("runtime_inputs must be dict or None")
        result = self._extractor.extract(cfg, inputs=runtime_inputs, sync_ctx=context)
        status = self._normalize_status_token(result.get("status"))
        payload = {
            "status": status,
            "sync_status": status,
            "sync_success": status == "ok",
            "sync_digest": result.get("sync_digest"),
            "geo_score": result.get("geo_score"),
            "geometry_absent_reason": result.get("geometry_absent_reason"),
            "geometry_failure_reason": result.get("geometry_failure_reason"),
            "template_match_metrics": result.get("template_match_metrics"),
            "sync_quality_metrics": result.get("sync_quality_metrics"),
            "sync_quality_semantics": result.get("sync_quality_semantics"),
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
        }
        relation_digest_bound = result.get("relation_digest_bound")
        if isinstance(relation_digest_bound, str) and relation_digest_bound:
            payload["relation_digest_bound"] = relation_digest_bound
        if isinstance(context.trajectory_evidence, dict):
            payload["trajectory_spec_digest"] = context.trajectory_evidence.get("trajectory_spec_digest")
            payload["trajectory_digest"] = context.trajectory_evidence.get("trajectory_digest")
            payload["trajectory_tap_version"] = context.trajectory_evidence.get("trajectory_tap_version")
        return payload

    def _normalize_status_token(self, raw_status: Any) -> str:
        """
        功能：归一化同步状态 token。

        Normalize raw sync status token to frozen status enum domain.

        Args:
            raw_status: Raw status token from extractor.

        Returns:
            Normalized status token in {ok, absent, mismatch, failed}.
        """
        if not isinstance(raw_status, str) or not raw_status:
            return "absent"
        lowered = raw_status.lower()
        if lowered == "fail":
            return "failed"
        if lowered in {"ok", "absent", "mismatch", "failed"}:
            return lowered
        return "failed"


def _build_geometry_latent_sync_sd3(cfg: Dict[str, Any]) -> SyncGeometryLatentSyncSD3:
    """
    功能：构造 geometry latent sync SD3 同步模块。

    Build geometry latent sync SD3 sync module wrapper.

    Args:
        cfg: Config mapping.

    Returns:
        SyncGeometryLatentSyncSD3 instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    impl_version = GEOMETRY_LATENT_SYNC_SD3_VERSION
    impl_digest = _derive_impl_digest(GEOMETRY_LATENT_SYNC_SD3_ID, impl_version)
    return SyncGeometryLatentSyncSD3(GEOMETRY_LATENT_SYNC_SD3_ID, impl_version, impl_digest)


_GEOMETRY_REGISTRY.register_factory(
    ATTENTION_ANCHOR_EXTRACTOR_ID,
    _build_attention_anchor_extractor,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
    )
)
_SYNC_REGISTRY.register_factory(
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
