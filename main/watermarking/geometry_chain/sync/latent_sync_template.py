"""
File purpose: 基于 SD3 latent 频域摘要生成同步模板证据。
Module type: General module
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from main.core import digests


@dataclass(frozen=True)
class SyncResult:
    """
    功能：同步模板提取结果载体。

    Structured result for latent sync template extraction.

    Args:
        status: Sync status token.
        sync_digest: Canonical sync digest.
        sync_config_digest: Sync config domain digest.
        sync_quality_metrics: Compact quality metrics.
        resolution_binding: Resolution binding payload.
        failure_reason: Structured failure reason.

    Returns:
        None.
    """

    status: str
    sync_digest: Optional[str]
    sync_config_digest: Optional[str]
    sync_quality_metrics: Optional[Dict[str, Any]]
    resolution_binding: Optional[Dict[str, Any]]
    failure_reason: Optional[str]


@dataclass(frozen=True)
class SyncRuntimeContext:
    """
    功能：运行期同步上下文载体。

    Runtime context container for sync module execution.

    Args:
        pipeline: Runtime pipeline object.
        latents: Runtime latents.
        rng: Optional random generator handle.
        trajectory_evidence: Optional trajectory tap evidence mapping.

    Returns:
        None.
    """

    pipeline: Any
    latents: Any
    rng: Any
    trajectory_evidence: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.trajectory_evidence is not None and not isinstance(self.trajectory_evidence, dict):
            # trajectory_evidence 类型不符合预期，必须 fail-fast。
            raise TypeError("trajectory_evidence must be dict or None")


def resolve_enable_latent_sync(cfg: Dict[str, Any]) -> bool:
    """
    功能：解析同步模板开关（embed/detect 双向兼容）。

    Resolve latent sync enable switch from config.
    Prioritizes embed.geometry.enable_latent_sync over detect.geometry.enable_latent_sync.

    Args:
        cfg: Configuration mapping.

    Returns:
        True when latent sync template is enabled.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    
    # (1) 优先读取 embed.geometry.enable_latent_sync（append-only v1.0）
    embed_cfg = cfg.get("embed")
    if isinstance(embed_cfg, dict):
        embed_geometry_cfg = embed_cfg.get("geometry")
        if isinstance(embed_geometry_cfg, dict):
            embed_explicit = embed_geometry_cfg.get("enable_latent_sync")
            if isinstance(embed_explicit, bool):
                return embed_explicit
    
    # (2) 回退到 detect.geometry.enable_latent_sync（既有逻辑）
    detect_cfg = cfg.get("detect")
    if not isinstance(detect_cfg, dict):
        return False
    geometry_cfg = detect_cfg.get("geometry")
    if not isinstance(geometry_cfg, dict):
        return False
    explicit = geometry_cfg.get("enable_latent_sync")
    if isinstance(explicit, bool):
        return explicit
    enabled = geometry_cfg.get("enabled")
    return bool(enabled) if isinstance(enabled, bool) else False


class LatentSyncTemplate:
    """
    功能：提取几何同步模板摘要与质量指标。

    Extract latent-domain sync template digest and quality metrics.

    Args:
        impl_id: Implementation identifier.
        impl_version: Implementation version.
        impl_digest: Implementation digest.

    Returns:
        None.
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

    def validate_sync_inputs(self, pipeline: Any, latents: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：校验同步输入并返回结构化状态。

        Validate sync extraction inputs and return structured status.

        Args:
            pipeline: Runtime pipeline object.
            latents: Runtime latents.
            cfg: Configuration mapping.

        Returns:
            Validation result with keys: valid, status, failure_reason.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")

        if not self._resolve_enable_latent_sync(cfg):
            return {"valid": False, "status": "absent", "failure_reason": "sync_disabled_by_policy"}

        if pipeline is None:
            return {"valid": False, "status": "absent", "failure_reason": "detect_pipeline_unavailable"}

        if not self._is_sd3_transformer_pipeline(pipeline, cfg):
            return {"valid": False, "status": "absent", "failure_reason": "sync_unsupported_model_or_pipeline"}

        if latents is None:
            return {"valid": False, "status": "mismatch", "failure_reason": "detect_sync_resolution_binding_missing"}

        return {"valid": True, "status": "ok", "failure_reason": None}

    def extract_sync(self, pipeline: Any, latents: Any, *, cfg: Dict[str, Any], rng: Any) -> SyncResult:
        """
        功能：提取同步摘要与质量指标。

        Extract sync digest and quality metrics from latent frequency summary.

        Args:
            pipeline: Runtime pipeline object.
            latents: Runtime latents.
            cfg: Configuration mapping.
            rng: Optional random generator handle.

        Returns:
            SyncResult for latent sync template.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        _ = rng

        validation = self.validate_sync_inputs(pipeline, latents, cfg)
        if not bool(validation.get("valid", False)):
            return SyncResult(
                status=str(validation.get("status", "fail")),
                sync_digest=None,
                sync_config_digest=None,
                sync_quality_metrics=None,
                resolution_binding=None,
                failure_reason=str(validation.get("failure_reason") or "sync_input_invalid"),
            )

        latents_np = _to_numpy_latents(latents)
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            return SyncResult(
                status="absent",
                sync_digest=None,
                sync_config_digest=None,
                sync_quality_metrics=None,
                resolution_binding=None,
                failure_reason="sync_transformer_absent",
            )

        try:
            resolution_binding = self.build_resolution_binding(transformer, latents_np, cfg)
            sync_summary = self._build_sync_summary(latents_np, cfg)
            sync_quality_metrics = self.compute_sync_quality_metrics(sync_summary)
            sync_config_digest = digests.canonical_sha256(self._build_sync_config_domain(cfg))

            sync_payload = {
                "summary_version": sync_summary.get("summary_version"),
                "sync_signature": sync_summary.get("sync_signature"),
                "sync_config_digest": sync_config_digest,
                "resolution_binding": resolution_binding,
                "impl_identity": self._impl_identity(),
            }
            sync_digest = self.compute_sync_digest(sync_payload)
            return SyncResult(
                status="ok",
                sync_digest=sync_digest,
                sync_config_digest=sync_config_digest,
                sync_quality_metrics=sync_quality_metrics,
                resolution_binding=resolution_binding,
                failure_reason=None,
            )
        except Exception:
            # 同步模板提取异常，必须结构化标记 fail。
            return SyncResult(
                status="fail",
                sync_digest=None,
                sync_config_digest=None,
                sync_quality_metrics=None,
                resolution_binding=None,
                failure_reason="latent_sync_extraction_failed",
            )

    def compute_sync_digest(self, payload: Dict[str, Any]) -> str:
        """
        功能：计算同步摘要 digest。

        Compute sync digest from canonical payload.

        Args:
            payload: Sync payload mapping.

        Returns:
            SHA256 digest string.
        """
        if not isinstance(payload, dict):
            # payload 类型不合法，必须 fail-fast。
            raise TypeError("payload must be dict")
        return digests.canonical_sha256(payload)

    def compute_sync_quality_metrics(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：计算同步质量指标。

        Compute compact sync quality metrics from summary.

        Args:
            summary: Sync summary mapping.

        Returns:
            Compact sync quality metrics mapping.
        """
        if not isinstance(summary, dict):
            # summary 类型不合法，必须 fail-fast。
            raise TypeError("summary must be dict")
        peak_ratio = float(summary.get("peak_ratio", 0.0))
        residual_score = float(summary.get("residual_score", 0.0))
        radial_entropy = float(summary.get("radial_entropy", 0.0))
        match_confidence = max(0.0, min(1.0, 0.6 * peak_ratio + 0.4 * (1.0 - residual_score)))
        return {
            "peak_ratio": round(peak_ratio, 6),
            "residual_score": round(residual_score, 6),
            "radial_entropy": round(radial_entropy, 6),
            "rotation_bin": int(summary.get("rotation_bin", 0)),
            "scale_bin": int(summary.get("scale_bin", 0)),
            "match_confidence": round(match_confidence, 6),
            "availability_score": 1.0,
            "sync_evidence_level": "primary",
        }

    def build_resolution_binding(self, transformer: Any, latents_np: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：构建同步分辨率绑定信息。

        Build reproducible resolution binding for sync template.

        Args:
            transformer: SD3 transformer object.
            latents_np: Latent array in [batch, channels, h, w].
            cfg: Configuration mapping.

        Returns:
            Resolution binding mapping.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        if latents_np.ndim != 4:
            # latents 维度不符合预期，必须 fail-fast。
            raise ValueError("latents must be rank-4")

        latent_height = int(latents_np.shape[-2])
        latent_width = int(latents_np.shape[-1])
        transformer_config = getattr(transformer, "config", None)
        patch_size = getattr(transformer_config, "patch_size", 1)
        if not isinstance(patch_size, int) or patch_size <= 0:
            patch_size = 1

        image_height = int(cfg.get("inference_height", 0) or 0)
        image_width = int(cfg.get("inference_width", 0) or 0)
        scale_h = float(image_height) / float(latent_height) if latent_height > 0 and image_height > 0 else 0.0
        scale_w = float(image_width) / float(latent_width) if latent_width > 0 and image_width > 0 else 0.0

        return {
            "binding_version": "latent_sync_resolution_binding_v1",
            "latent_height": latent_height,
            "latent_width": latent_width,
            "token_grid_height": latent_height,
            "token_grid_width": latent_width,
            "token_count": latent_height * latent_width,
            "patch_size": patch_size,
            "image_height": image_height,
            "image_width": image_width,
            "latent_to_image_scale_h": round(scale_h, 6),
            "latent_to_image_scale_w": round(scale_w, 6),
            "model_binding": "sd3_transformer_only",
        }

    def _build_sync_summary(self, latents_np: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：构建不可逆同步摘要。

        Build non-reversible latent frequency sync summary.

        Args:
            latents_np: Latent array.
            cfg: Configuration mapping.

        Returns:
            Sync summary mapping for digesting.
        """
        latent_first = latents_np[0]
        latent_map = np.mean(np.abs(latent_first), axis=0)

        spectrum = np.fft.fftshift(np.abs(np.fft.fft2(latent_map)))
        spectrum_center = spectrum.copy()
        center_y = spectrum_center.shape[0] // 2
        center_x = spectrum_center.shape[1] // 2
        spectrum_center[center_y, center_x] = 0.0

        flat = spectrum_center.reshape(-1)
        top_index = int(np.argmax(flat))
        top_value = float(flat[top_index])
        mean_value = float(np.mean(flat) + 1e-12)
        peak_ratio = top_value / mean_value

        top_y, top_x = np.unravel_index(top_index, spectrum_center.shape)
        delta_y = float(top_y - center_y)
        delta_x = float(top_x - center_x)
        angle = float(np.arctan2(delta_y, delta_x))
        radius = float(np.sqrt(delta_x * delta_x + delta_y * delta_y))

        rotation_bins = self._resolve_rotation_bins(cfg)
        scale_bins = self._resolve_scale_bins(cfg)
        rotation_bin = int(((angle + np.pi) / (2.0 * np.pi)) * rotation_bins) % rotation_bins
        max_radius = float(np.sqrt(center_x * center_x + center_y * center_y) + 1e-12)
        norm_radius = radius / max_radius
        scale_bin = int(min(scale_bins - 1, max(0, np.floor(norm_radius * scale_bins))))

        hist_bins = self._resolve_fft_bins(cfg)
        hist_values, _ = np.histogram(flat, bins=hist_bins, density=True)
        hist_values = hist_values.astype(np.float64)
        probs = hist_values / (float(np.sum(hist_values)) + 1e-12)
        radial_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        sorted_values = np.sort(flat)
        second_peak = float(sorted_values[-2]) if sorted_values.size > 1 else 0.0
        residual_score = 0.0
        if top_value > 0:
            residual_score = float(min(1.0, max(0.0, second_peak / top_value)))

        sync_signature = {
            "rotation_bin": rotation_bin,
            "scale_bin": scale_bin,
            "peak_ratio_q": round(float(peak_ratio), 4),
            "residual_q": round(float(residual_score), 4),
            "histogram_q": [round(float(v), 6) for v in probs.tolist()],
        }
        return {
            "summary_version": "latent_sync_template_summary_v1",
            "sync_signature": sync_signature,
            "peak_ratio": float(peak_ratio),
            "residual_score": float(residual_score),
            "radial_entropy": float(radial_entropy),
            "rotation_bin": rotation_bin,
            "scale_bin": scale_bin,
        }

    def _build_sync_config_domain(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：构建同步配置输入域。

        Build canonical sync config input domain.

        Args:
            cfg: Configuration mapping.

        Returns:
            Sync config domain mapping.
        """
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        return {
            "domain_version": "latent_sync_cfg_domain_v1",
            "enable_latent_sync": bool(self._resolve_enable_latent_sync(cfg)),
            "sync_fft_bins": self._resolve_fft_bins(cfg),
            "sync_rotation_bins": self._resolve_rotation_bins(cfg),
            "sync_scale_bins": self._resolve_scale_bins(cfg),
            "model_id": cfg.get("model_id"),
        }

    def _resolve_enable_latent_sync(self, cfg: Dict[str, Any]) -> bool:
        """
        功能：解析同步模板开关。

        Resolve latent sync enable switch from config.

        Args:
            cfg: Configuration mapping.

        Returns:
            True when latent sync template is enabled.
        """
        return resolve_enable_latent_sync(cfg)

    def _resolve_fft_bins(self, cfg: Dict[str, Any]) -> int:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("sync_fft_bins", 16)
        if not isinstance(value, int):
            return 16
        return max(8, min(64, value))

    def _resolve_rotation_bins(self, cfg: Dict[str, Any]) -> int:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("sync_rotation_bins", 36)
        if not isinstance(value, int):
            return 36
        return max(8, min(72, value))

    def _resolve_scale_bins(self, cfg: Dict[str, Any]) -> int:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("sync_scale_bins", 16)
        if not isinstance(value, int):
            return 16
        return max(4, min(32, value))

    def _is_sd3_transformer_pipeline(self, pipeline: Any, cfg: Dict[str, Any]) -> bool:
        """
        功能：判断 pipeline 是否符合 SD3 transformer 要求。

        Check whether runtime pipeline matches SD3 transformer requirements.

        Args:
            pipeline: Runtime pipeline object.
            cfg: Configuration mapping.

        Returns:
            True when pipeline/model is SD3 transformer compatible.
        """
        if getattr(pipeline, "transformer", None) is None:
            return False
        model_id = cfg.get("model_id")
        if isinstance(model_id, str) and "stable-diffusion-3" in model_id:
            return True
        pipeline_name = type(pipeline).__name__.lower()
        return "sd3" in pipeline_name

    def _impl_identity(self) -> Dict[str, str]:
        return {
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
        }


class LatentSyncGeometryExtractor:
    """
    功能：几何链同步模板提取器。

    Geometry extractor wrapper that emits sync template evidence.

    Args:
        impl_id: Implementation identifier.
        impl_version: Implementation version.
        impl_digest: Implementation digest.

    Returns:
        None.
    """

    def __init__(self, impl_id: str, impl_version: str, impl_digest: str) -> None:
        self.impl_id = impl_id
        self.impl_version = impl_version
        self.impl_digest = impl_digest
        self._sync_template = LatentSyncTemplate(impl_id, impl_version, impl_digest)

    def extract(self, cfg: Dict[str, Any], inputs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        功能：提取几何同步证据。

        Extract geometry evidence containing sync template fields.

        Args:
            cfg: Configuration mapping.
            inputs: Runtime inputs mapping.

        Returns:
            Geometry evidence mapping.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        runtime_inputs = inputs if isinstance(inputs, dict) else {}
        pipeline_obj = runtime_inputs.get("pipeline")
        latents = runtime_inputs.get("latents")
        rng = runtime_inputs.get("rng")

        sync_result = self._sync_template.extract_sync(pipeline_obj, latents, cfg=cfg, rng=rng)

        trace_payload = {
            "status": sync_result.status,
            "sync_digest": sync_result.sync_digest,
            "sync_config_digest": sync_result.sync_config_digest,
            "resolution_binding": sync_result.resolution_binding,
            "failure_reason": sync_result.failure_reason,
        }
        trace_digest = digests.canonical_sha256(trace_payload)
        return {
            "status": sync_result.status,
            "geo_score": None,
            "anchor_digest": None,
            "anchor_config_digest": None,
            "anchor_metrics": None,
            "stability_metrics": None,
            "sync_digest": sync_result.sync_digest,
            "sync_config_digest": sync_result.sync_config_digest,
            "sync_metrics": sync_result.sync_quality_metrics,
            "sync_quality_metrics": sync_result.sync_quality_metrics,
            "resolution_binding": sync_result.resolution_binding,
            "align_trace_digest": None,
            "geo_failure_reason": sync_result.failure_reason,
            "geometry_failure_reason": sync_result.failure_reason,
            "audit": {
                "impl_identity": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "trace_digest": trace_digest,
                "sync_status_detail": sync_result.status,
            },
        }


def _to_numpy_latents(latents: Any) -> np.ndarray:
    """
    功能：将 latent 输入转换为 numpy array。

    Convert latent tensor-like object to numpy array.

    Args:
        latents: Numpy or tensor-like latent input.

    Returns:
        Numpy array with shape [batch, channels, h, w].
    """
    if isinstance(latents, np.ndarray):
        result = latents
    elif hasattr(latents, "detach") and callable(latents.detach):
        detached = latents.detach()
        if hasattr(detached, "cpu") and callable(detached.cpu):
            detached = detached.cpu()
        if hasattr(detached, "numpy") and callable(detached.numpy):
            result = detached.numpy()
        else:
            # tensor 转换接口缺失，必须 fail-fast。
            raise TypeError("latents.detach() result must expose numpy()")
    else:
        # latents 类型不符合预期，必须 fail-fast。
        raise TypeError("latents must be numpy array or tensor-like")

    if result.ndim != 4:
        # latents 维度不符合预期，必须 fail-fast。
        raise ValueError(f"latents must be rank-4, got {result.shape}")
    return result


# Paper-faithful geometry latent sync SD3 v2 with relation digest binding
GEOMETRY_LATENT_SYNC_SD3_V2_ID = "geometry_latent_sync_sd3_v2"
GEOMETRY_LATENT_SYNC_SD3_V2_VERSION = "v2"


class GeometryLatentSyncSD3V2:
    """
    功能：SD3 latent sync v2 with relation_digest binding.

    Upgraded sync module that uses relation_digest from attention anchor
    to improve alignment certainty and detect mismatches.

    Args:
        impl_id: Implementation identifier (must be geometry_latent_sync_sd3_v2).
        impl_version: Implementation version string.
        impl_digest: Implementation digest string.

    Returns:
        None.

    Raises:
        ValueError: If constructor inputs are invalid.
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

    def extract(
        self,
        cfg: Dict[str, Any],
        inputs: Dict[str, Any] | None = None,
        sync_ctx: SyncRuntimeContext | None = None
    ) -> Dict[str, Any]:
        """
        功能：提取 sync 证据并绑定 relation_digest。

        Extract sync evidence with relation_digest binding for enhanced certainty.

        Args:
            cfg: Configuration mapping.
            inputs: Optional runtime inputs with relation_digest from anchor extractor.
            sync_ctx: Optional sync runtime context.

        Returns:
            Geometry evidence mapping with sync_digest and relation binding.

        Raises:
            TypeError: If inputs are invalid.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if inputs is not None and not isinstance(inputs, dict):
            raise TypeError("inputs must be dict or None")

        if not resolve_enable_latent_sync(cfg):
            return {
                "status": "absent",
                "geo_score": None,
                "sync_digest": None,
                "geometry_absent_reason": "latent_sync_disabled",
            }

        runtime_inputs = inputs or {}
        relation_digest = runtime_inputs.get("relation_digest")

        # If relation_digest is missing and we require it for v2, fail with mismatch
        if relation_digest is None:
            return {
                "status": "mismatch",
                "geo_score": None,
                "sync_digest": None,
                "geometry_failure_reason": "relation_digest_missing_for_v2",
                "sync_quality_semantics": {
                    "score_type": "interpretable_geometry_consistency",
                    "score_version": "latent_sync_geometry_consistency_v2",
                    "trusted_as_primary_geometry_evidence": False,
                    "evidence_level": "quantitative_secondary",
                },
            }

        if not isinstance(relation_digest, str) or not relation_digest:
            return {
                "status": "mismatch",
                "geo_score": None,
                "sync_digest": None,
                "geometry_failure_reason": "relation_digest_invalid",
                "sync_quality_semantics": {
                    "score_type": "interpretable_geometry_consistency",
                    "score_version": "latent_sync_geometry_consistency_v2",
                    "trusted_as_primary_geometry_evidence": False,
                    "evidence_level": "quantitative_secondary",
                },
            }

        # Extract latents from sync_ctx if available
        if sync_ctx is None or sync_ctx.latents is None:
            return {
                "status": "absent",
                "geo_score": None,
                "sync_digest": None,
                "geometry_absent_reason": "latents_missing",
                "sync_quality_semantics": {
                    "score_type": "interpretable_geometry_consistency",
                    "score_version": "latent_sync_geometry_consistency_v2",
                    "trusted_as_primary_geometry_evidence": False,
                    "evidence_level": "quantitative_secondary",
                },
            }

        try:
            latents_np = _to_numpy_latents(sync_ctx.latents)
        except Exception as e:
            return {
                "status": "failed",
                "geo_score": None,
                "sync_digest": None,
                "geometry_failure_reason": f"latents_conversion_failed: {str(e)}",
            }

        # Compute sync quality metrics
        try:
            sync_quality_metrics = self._compute_sync_quality(latents_np, relation_digest)
        except Exception as e:
            return {
                "status": "failed",
                "geo_score": None,
                "sync_digest": None,
                "geometry_failure_reason": f"sync_quality_computation_failed: {str(e)}",
            }

        # Check uncertainty threshold - if uncertain, report mismatch instead of silent fallback
        uncertainty = sync_quality_metrics.get("uncertainty", 1.0)
        if uncertainty > 0.5:
            return {
                "status": "mismatch",
                "geo_score": None,
                "sync_digest": None,
                "sync_quality_metrics": sync_quality_metrics,
                "sync_quality_semantics": {
                    "score_type": "interpretable_geometry_consistency",
                    "score_version": "latent_sync_geometry_consistency_v2",
                    "trusted_as_primary_geometry_evidence": False,
                    "evidence_level": "quantitative_secondary",
                },
                "geometry_failure_reason": "sync_uncertainty_too_high",
            }

        # Compute sync_digest binding relation_digest
        sync_config_digest = digests.canonical_sha256({
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
        })
        sync_digest = digests.canonical_sha256({
            "relation_digest": relation_digest,
            "sync_config_digest": sync_config_digest,
            "sync_quality_digest": digests.canonical_sha256(sync_quality_metrics),
        })

        return {
            "status": "ok",
            "geo_score": None,  # Will be computed by align module
            "sync_digest": sync_digest,
            "sync_config_digest": sync_config_digest,
            "sync_quality_metrics": sync_quality_metrics,
            "sync_quality_semantics": {
                "score_type": "interpretable_geometry_consistency",
                "score_version": "latent_sync_geometry_consistency_v2",
                "trusted_as_primary_geometry_evidence": False,
                "evidence_level": "quantitative_secondary",
            },
            "relation_digest_bound": relation_digest,
            "geometry_failure_reason": None,
        }

    def _compute_sync_quality(
        self,
        latents_np: np.ndarray,
        relation_digest: str
    ) -> Dict[str, Any]:
        """
        功能：计算同步质量指标。

        Compute sync quality metrics using latents and relation_digest.

        Args:
            latents_np: Latents numpy array.
            relation_digest: Relation digest from anchor extractor.

        Returns:
            Sync quality metrics mapping.
        """
        if not isinstance(latents_np, np.ndarray) or latents_np.ndim != 4:
            raise ValueError("latents_np must be rank-4 numpy array")
        if not isinstance(relation_digest, str) or not relation_digest:
            raise ValueError("relation_digest must be non-empty str")

        latent_first = latents_np[0]
        spatial_energy = np.mean(np.abs(latent_first), axis=0)
        spatial_std = float(np.std(spatial_energy))
        spatial_mean = float(np.mean(spatial_energy) + 1e-12)
        contrast_ratio = float(spatial_std / spatial_mean)

        flat_energy = spatial_energy.reshape(-1)
        peak_value = float(np.max(flat_energy))
        median_value = float(np.median(flat_energy) + 1e-12)
        peak_sharpness = float(peak_value / median_value)

        spectrum = np.fft.fftshift(np.abs(np.fft.fft2(spatial_energy)))
        center_y = spectrum.shape[0] // 2
        center_x = spectrum.shape[1] // 2
        spectrum[center_y, center_x] = 0.0
        sorted_spectrum = np.sort(spectrum.reshape(-1))
        top1 = float(sorted_spectrum[-1]) if sorted_spectrum.size > 0 else 0.0
        top2 = float(sorted_spectrum[-2]) if sorted_spectrum.size > 1 else 0.0
        spectral_peak_ratio = float(top1 / (top2 + 1e-12)) if top1 > 0.0 else 0.0

        relation_norm = float(sum(bytearray(relation_digest.encode("utf-8"))) % 1024) / 1024.0
        relation_alignment = float(1.0 - abs(relation_norm - min(1.0, contrast_ratio / 4.0)))
        relation_alignment = float(max(0.0, min(1.0, relation_alignment)))

        quality_score = 0.35 * min(1.0, contrast_ratio / 2.0) + 0.35 * min(1.0, spectral_peak_ratio / 3.0) + 0.30 * relation_alignment
        quality_score = float(max(0.0, min(1.0, quality_score)))
        uncertainty = float(max(0.0, min(1.0, 1.0 - quality_score)))

        return {
            "contrast_ratio": float(round(contrast_ratio, 6)),
            "peak_sharpness": float(round(peak_sharpness, 6)),
            "spectral_peak_ratio": float(round(spectral_peak_ratio, 6)),
            "relation_alignment": float(round(relation_alignment, 6)),
            "quality_score": float(round(quality_score, 6)),
            "uncertainty": float(round(uncertainty, 6)),
            "relation_digest_bound": relation_digest,
            "quality_method": "interpretable_latent_spectrum_relation_consistency_v2",
            "quality_components_v2": {
                "version": "latent_sync_quality_components_v2",
                "contrast_component": float(round(min(1.0, contrast_ratio / 2.0), 6)),
                "spectral_component": float(round(min(1.0, spectral_peak_ratio / 3.0), 6)),
                "relation_component": float(round(relation_alignment, 6)),
                "weights": {
                    "contrast_component": 0.35,
                    "spectral_component": 0.35,
                    "relation_component": 0.30,
                },
                "constraints": {
                    "quality_score_in_unit_interval": bool(0.0 <= quality_score <= 1.0),
                    "uncertainty_is_one_minus_quality_score": float(round(abs((1.0 - quality_score) - uncertainty), 6)),
                },
                "primary_evidence": False,
                "evidence_level": "supporting",
            },
        }
