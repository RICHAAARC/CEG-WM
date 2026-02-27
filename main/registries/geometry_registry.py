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
from main.watermarking.geometry_chain.attention_anchor_extractor import (
    AttentionAnchorExtractor,
    AttentionAnchorMapRelation,
    ATTENTION_ANCHOR_MAP_RELATION_ID,
    ATTENTION_ANCHOR_MAP_RELATION_VERSION
)
from main.watermarking.geometry_chain.sync.latent_sync_template import (
    LatentSyncGeometryExtractor,
    LatentSyncTemplate,
    GeometryLatentSyncSD3V2,
    GEOMETRY_LATENT_SYNC_SD3_V2_ID,
    GEOMETRY_LATENT_SYNC_SD3_V2_VERSION,
    SyncResult,
    SyncRuntimeContext,
)

from .registry_base import FactoryType, RegistryBase
from .capabilities import ImplCapabilities


GEOMETRY_BASELINE_IDENTITY_ID = "geometry_baseline_identity_v1"
GEOMETRY_ATTENTION_ANCHOR_SD3_ID = "geometry_attention_anchor_sd3_v1"
GEOMETRY_LATENT_SYNC_SD3_ID = "geometry_latent_sync_sd3_v1"  # 同名 impl_id 同时用于 geometry_extractor 与 sync_module 域。
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

    def sync_with_context(
        self,
        cfg: Dict[str, Any],
        context: SyncRuntimeContext,
        runtime_inputs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        功能：返回基线同步状态（忽略上下文）。

        Return baseline sync status while ignoring runtime context.

        Args:
            cfg: Config mapping.
            context: Sync runtime context.
            runtime_inputs: Optional runtime input mapping.

        Returns:
            Baseline sync status mapping.

        Raises:
            TypeError: If inputs are invalid.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        if not isinstance(context, SyncRuntimeContext):
            # context 类型不合法，必须 fail-fast。
            raise TypeError("context must be SyncRuntimeContext")
        return self.sync(cfg)


class SyncLatentSyncSd3:
    """
    功能：SD3 latent 同步模块最小实现。

    Minimal SD3 latent sync module that emits structured status.

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
        self._sync_template = LatentSyncTemplate(impl_id, impl_version, impl_digest)

    def _build_sync_payload(self, sync_result: SyncResult) -> Dict[str, Any]:
        """
        功能：构造统一的同步结果字段。

        Build unified sync result payload from SyncResult.

        Args:
            sync_result: SyncResult instance.

        Returns:
            Sync result mapping with audit fields.

        Raises:
            TypeError: If sync_result is invalid.
        """
        if not isinstance(sync_result, SyncResult):
            # sync_result 类型不合法，必须 fail-fast。
            raise TypeError("sync_result must be SyncResult")
        status = sync_result.status
        sync_success = status == "ok"
        return {
            "sync_status": status,
            "sync_success": sync_success,
            "failure_reason": sync_result.failure_reason,
            "sync_digest": sync_result.sync_digest,
            "sync_config_digest": sync_result.sync_config_digest,
            "sync_quality_metrics": sync_result.sync_quality_metrics,
            "resolution_binding": sync_result.resolution_binding,
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
        }

    def sync(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：返回结构化同步状态，不执行实际同步。

        Return structured sync status without performing heavy runtime sync.

        Args:
            cfg: Config mapping.

        Returns:
            Sync status mapping with audit fields.

        Raises:
            TypeError: If cfg is not dict.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")

        sync_result = self._sync_template.extract_sync(None, None, cfg=cfg, rng=None)
        return self._build_sync_payload(sync_result)

    def sync_with_context(self, cfg: Dict[str, Any], context: SyncRuntimeContext) -> Dict[str, Any]:
        """
        功能：在运行期上下文下提取同步状态。

        Extract sync status using runtime context inputs.

        Args:
            cfg: Config mapping.
            context: Sync runtime context.

        Returns:
            Sync status mapping with audit fields.

        Raises:
            TypeError: If inputs are invalid.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        if not isinstance(context, SyncRuntimeContext):
            # context 类型不合法，必须 fail-fast。
            raise TypeError("context must be SyncRuntimeContext")
        sync_result = self._sync_template.extract_sync(
            context.pipeline,
            context.latents,
            cfg=cfg,
            rng=context.rng,
        )
        payload = self._build_sync_payload(sync_result)
        if isinstance(context.trajectory_evidence, dict):
            payload["trajectory_spec_digest"] = context.trajectory_evidence.get("trajectory_spec_digest")
            payload["trajectory_digest"] = context.trajectory_evidence.get("trajectory_digest")
            payload["trajectory_tap_version"] = context.trajectory_evidence.get("trajectory_tap_version")
        return payload


class SyncGeometryLatentSyncSD3V2:
    """
    功能：SD3 latent sync v2 同步模块包装器。

    Sync module wrapper for GeometryLatentSyncSD3V2 extractor that provides
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
        self._extractor = GeometryLatentSyncSD3V2(impl_id, impl_version, impl_digest)

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
        功能：使用运行期上下文提取同步状态。

        Extract sync status using runtime context.

        Args:
            cfg: Config mapping.
            context: Sync runtime context.

        Returns:
            Sync status mapping with trajectory bindings.

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
        功能：归一化同步状态到冻结枚举口径。

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


def _build_sync_geometry_latent_sync_sd3(cfg: Dict[str, Any]) -> SyncLatentSyncSd3:
    """
    功能：构造 SD3 latent 同步模块实现。

    Build SD3 latent sync module implementation.

    Args:
        cfg: Config mapping.

    Returns:
        SyncLatentSyncSd3 instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = "v1"
    impl_digest = _derive_impl_digest(GEOMETRY_LATENT_SYNC_SD3_ID, impl_version)
    return SyncLatentSyncSd3(GEOMETRY_LATENT_SYNC_SD3_ID, impl_version, impl_digest)


def _build_attention_anchor_map_relation(cfg: Dict[str, Any]) -> AttentionAnchorMapRelation:
    """
    功能：构造 attention anchor map relation 实现。

    Build attention anchor map relation geometry extractor.

    Args:
        cfg: Config mapping.

    Returns:
        AttentionAnchorMapRelation instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    impl_version = ATTENTION_ANCHOR_MAP_RELATION_VERSION
    impl_digest = _derive_impl_digest(ATTENTION_ANCHOR_MAP_RELATION_ID, impl_version)
    return AttentionAnchorMapRelation(ATTENTION_ANCHOR_MAP_RELATION_ID, impl_version, impl_digest)


def _build_geometry_latent_sync_sd3_v2(cfg: Dict[str, Any]) -> SyncGeometryLatentSyncSD3V2:
    """
    功能：构造 geometry latent sync SD3 v2 同步模块。

    Build geometry latent sync SD3 v2 sync module wrapper.

    Args:
        cfg: Config mapping.

    Returns:
        SyncGeometryLatentSyncSD3V2 instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    impl_version = GEOMETRY_LATENT_SYNC_SD3_V2_VERSION
    impl_digest = _derive_impl_digest(GEOMETRY_LATENT_SYNC_SD3_V2_ID, impl_version)
    return SyncGeometryLatentSyncSD3V2(GEOMETRY_LATENT_SYNC_SD3_V2_ID, impl_version, impl_digest)


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
_GEOMETRY_REGISTRY.register_factory(
    ATTENTION_ANCHOR_MAP_RELATION_ID,
    _build_attention_anchor_map_relation,
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
_SYNC_REGISTRY.register_factory(
    GEOMETRY_LATENT_SYNC_SD3_ID,
    _build_sync_geometry_latent_sync_sd3,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
    )
)
_SYNC_REGISTRY.register_factory(
    GEOMETRY_LATENT_SYNC_SD3_V2_ID,
    _build_geometry_latent_sync_sd3_v2,
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
