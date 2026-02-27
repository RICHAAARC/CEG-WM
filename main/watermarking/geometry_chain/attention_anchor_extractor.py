"""
File purpose: 基于 SD3 Transformer token 关系生成几何锚点摘要与稳定性指标。
Module type: General module
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from main.core import digests


@dataclass(frozen=True)
class AnchorResult:
    """
    功能：注意力锚点提取结果。

    Structured anchor extraction result.

    Args:
        status: Geometry anchor status.
        anchor_digest: Canonical anchor digest.
        anchor_config_digest: Digest of anchor config domain.
        stability_metrics: Stable scalar/vector metrics.
        resolution_binding: Resolution binding payload.
        failure_reason: Failure reason token.

    Returns:
        None.
    """

    status: str
    anchor_digest: Optional[str]
    anchor_config_digest: Optional[str]
    stability_metrics: Optional[Dict[str, Any]]
    resolution_binding: Optional[Dict[str, Any]]
    failure_reason: Optional[str]

    def as_geometry_evidence(self, impl_identity: Dict[str, str]) -> Dict[str, Any]:
        """
        功能：转换为 geometry evidence 载荷。

        Convert anchor result to geometry evidence payload.

        Args:
            impl_identity: Implementation identity mapping.

        Returns:
            Geometry evidence mapping.
        """
        trace_payload = {
            "status": self.status,
            "anchor_digest": self.anchor_digest,
            "anchor_config_digest": self.anchor_config_digest,
            "resolution_binding": self.resolution_binding,
            "failure_reason": self.failure_reason,
        }
        trace_digest = digests.canonical_sha256(trace_payload)
        return {
            "status": self.status,
            "geo_score": None,
            "anchor_digest": self.anchor_digest,
            "anchor_config_digest": self.anchor_config_digest,
            "anchor_source_semantics": "attention_like_proxy_from_latent_correlation",
            "anchor_evidence_level": "proxy",
            "anchor_blocking_reason": "real_self_attention_map_not_wired_in_runtime" if self.status in {"ok", "mismatch", "absent", "fail", "failed"} else None,
            "anchor_metrics": self.stability_metrics,
            "stability_metrics": self.stability_metrics,
            "resolution_binding": self.resolution_binding,
            "sync_digest": None,
            "sync_metrics": None,
            "align_trace_digest": None,
            "geo_failure_reason": self.failure_reason,
            "geometry_failure_reason": self.failure_reason,
            "audit": {
                "impl_identity": impl_identity.get("impl_id"),
                "impl_version": impl_identity.get("impl_version"),
                "impl_digest": impl_identity.get("impl_digest"),
                "trace_digest": trace_digest,
                "sync_status_detail": self.status,
                "anchor_status_detail": self.status,
            },
        }


class AttentionAnchorExtractor:
    """
    功能：从 SD3 Transformer token 空间提取关系型几何锚点。

    Extract relation-only geometry anchors from SD3 transformer token space.

    Args:
        impl_id: Implementation identifier.
        impl_version: Implementation version.
        impl_digest: Implementation digest.

    Returns:
        None.
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

    def extract(self, cfg: Dict[str, Any], inputs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        功能：按检测阶段上下文提取锚点证据。

        Extract geometry anchor evidence from runtime detect context.

        Args:
            cfg: Configuration mapping.
            inputs: Optional runtime inputs with pipeline and latents.

        Returns:
            Geometry evidence mapping with absent/mismatch/fail semantics.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if inputs is not None and not isinstance(inputs, dict):
            raise TypeError("inputs must be dict or None")

        if not self._resolve_enable_attention_anchor(cfg):
            result = AnchorResult(
                status="absent",
                anchor_digest=None,
                anchor_config_digest=None,
                stability_metrics=None,
                resolution_binding=None,
                failure_reason="anchor_disabled_by_policy",
            )
            return result.as_geometry_evidence(self._impl_identity())

        runtime_inputs = inputs or {}
        pipeline_obj = runtime_inputs.get("pipeline")
        latents = runtime_inputs.get("latents")

        if pipeline_obj is None:
            result = AnchorResult(
                status="absent",
                anchor_digest=None,
                anchor_config_digest=None,
                stability_metrics=None,
                resolution_binding=None,
                failure_reason="detect_pipeline_unavailable",
            )
            return result.as_geometry_evidence(self._impl_identity())

        if latents is None:
            result = AnchorResult(
                status="mismatch",
                anchor_digest=None,
                anchor_config_digest=None,
                stability_metrics=None,
                resolution_binding=None,
                failure_reason="detect_anchor_resolution_binding_missing",
            )
            return result.as_geometry_evidence(self._impl_identity())

        try:
            anchor_result = self.extract_anchors(
                pipeline_obj,
                latents,
                cfg=cfg,
                rng=runtime_inputs.get("rng"),
            )
        except Exception:
            result = AnchorResult(
                status="fail",
                anchor_digest=None,
                anchor_config_digest=None,
                stability_metrics=None,
                resolution_binding=None,
                failure_reason="attention_anchor_extraction_failed",
            )
            return result.as_geometry_evidence(self._impl_identity())

        return anchor_result.as_geometry_evidence(self._impl_identity())

    def extract_anchors(
        self,
        pipeline: Any,
        latents: Any,
        *,
        cfg: Dict[str, Any],
        rng: Any,
    ) -> AnchorResult:
        """
        功能：执行锚点摘要提取与摘要计算。

        Extract anchor relation summary and produce reproducible digest.

        Args:
            pipeline: Runtime pipeline object.
            latents: Final latent tensor.
            cfg: Configuration mapping.
            rng: Optional random generator handle.

        Returns:
            AnchorResult with status="ok" on success.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        _ = rng
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise ValueError("pipeline.transformer is required for SD3 attention anchor extraction")

        latents_np = _to_numpy_latents(latents)
        resolution_binding = self.build_resolution_binding(transformer, latents_np, cfg)
        relation_summary = self._build_relation_summary(latents_np, cfg)
        stability_metrics = self.compute_stability_metrics(relation_summary)
        anchor_config_digest = digests.canonical_sha256(self._build_anchor_config_domain(cfg))

        anchor_payload = {
            "relation_summary": relation_summary,
            "anchor_config_digest": anchor_config_digest,
            "resolution_binding": resolution_binding,
            "impl_identity": self._impl_identity(),
        }
        anchor_digest = self.compute_anchor_digest(anchor_payload)
        return AnchorResult(
            status="ok",
            anchor_digest=anchor_digest,
            anchor_config_digest=anchor_config_digest,
            stability_metrics=stability_metrics,
            resolution_binding=resolution_binding,
            failure_reason=None,
        )

    def compute_anchor_digest(self, anchor_payload: Dict[str, Any]) -> str:
        """
        功能：计算锚点摘要。

        Compute anchor digest using canonical JSON serialization.

        Args:
            anchor_payload: Anchor payload mapping.

        Returns:
            SHA256 digest string.
        """
        if not isinstance(anchor_payload, dict):
            raise TypeError("anchor_payload must be dict")
        return digests.canonical_sha256(anchor_payload)

    def compute_stability_metrics(self, attn_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：计算稳定性指标。

        Compute compact scalar/vector stability metrics.

        Args:
            attn_summary: Relation summary mapping.

        Returns:
            JSON-serializable stability metrics mapping.
        """
        if not isinstance(attn_summary, dict):
            raise TypeError("attn_summary must be dict")
        topk_hist = attn_summary.get("neighbor_hist", [])
        values = np.asarray(topk_hist, dtype=np.float64)
        if values.size == 0:
            entropy = 0.0
            concentration = 0.0
        else:
            probs = values / (float(values.sum()) + 1e-12)
            entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
            concentration = float(np.max(probs))
        return {
            "neighbor_entropy": round(entropy, 6),
            "top1_concentration": round(concentration, 6),
            "spectral_signature": attn_summary.get("spectral_signature", []),
            "token_count": int(attn_summary.get("token_count", 0)),
            "availability_score": 1.0,
        }

    def build_resolution_binding(self, transformer: Any, latents_np: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：构造分辨率绑定信息。

        Build reproducible resolution binding payload.

        Args:
            transformer: SD3 transformer object.
            latents_np: Latent tensor as numpy array.
            cfg: Configuration mapping.

        Returns:
            Resolution binding mapping.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if latents_np.ndim < 4:
            raise ValueError("latents must have shape [batch, channels, h, w]")

        latent_height = int(latents_np.shape[-2])
        latent_width = int(latents_np.shape[-1])

        transformer_config = getattr(transformer, "config", None)
        patch_size = getattr(transformer_config, "patch_size", 1)
        if not isinstance(patch_size, int) or patch_size <= 0:
            patch_size = 1

        image_height = int(cfg.get("inference_height", 0) or 0)
        image_width = int(cfg.get("inference_width", 0) or 0)
        token_grid_height = latent_height
        token_grid_width = latent_width
        latent_to_image_scale_h = 0.0
        latent_to_image_scale_w = 0.0
        if latent_height > 0 and image_height > 0:
            latent_to_image_scale_h = float(image_height) / float(latent_height)
        if latent_width > 0 and image_width > 0:
            latent_to_image_scale_w = float(image_width) / float(latent_width)

        return {
            "binding_version": "attention_anchor_resolution_binding_v1",
            "token_grid_height": token_grid_height,
            "token_grid_width": token_grid_width,
            "token_count": token_grid_height * token_grid_width,
            "patch_size": patch_size,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "image_height": image_height,
            "image_width": image_width,
            "latent_to_image_scale_h": round(latent_to_image_scale_h, 6),
            "latent_to_image_scale_w": round(latent_to_image_scale_w, 6),
            "model_binding": "sd3_transformer_only",
        }

    def _build_relation_summary(self, latents_np: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：构造 token 关系摘要（不可逆）。

        Build non-reversible token relation summary from latent token vectors.

        Args:
            latents_np: Latent tensor as numpy array.
            cfg: Configuration mapping.

        Returns:
            Compact relation summary for digesting.
        """
        latents_first = latents_np[0]
        channels = int(latents_first.shape[0])
        token_vectors = np.transpose(latents_first, (1, 2, 0)).reshape(-1, channels)

        top_k = self._resolve_anchor_top_k(cfg)
        token_count = int(token_vectors.shape[0])
        if token_count <= 1:
            return {
                "summary_version": "attention_anchor_relation_summary_v1",
                "top_k": top_k,
                "token_count": token_count,
                "neighbor_hist": [],
                "spectral_signature": [],
            }

        norms = np.linalg.norm(token_vectors, axis=1, keepdims=True)
        token_vectors = token_vectors / (norms + 1e-12)
        similarity = token_vectors @ token_vectors.T
        np.fill_diagonal(similarity, -1.0)
        effective_k = min(top_k, max(1, token_count - 1))

        neighbor_indices = np.argpartition(-similarity, effective_k - 1, axis=1)[:, :effective_k]
        neighbor_indices = np.sort(neighbor_indices, axis=1)

        hist_bins = min(64, token_count)
        hist = np.zeros(hist_bins, dtype=np.int64)
        for row in neighbor_indices:
            for neighbor in row:
                hist[int(neighbor % hist_bins)] += 1

        gram = token_vectors.T @ token_vectors
        eigenvalues = np.linalg.eigvalsh(gram)
        eigenvalues = np.sort(eigenvalues)[::-1]
        spectral_signature = [round(float(v), 6) for v in eigenvalues[:4]]

        return {
            "summary_version": "attention_anchor_relation_summary_v1",
            "top_k": effective_k,
            "token_count": token_count,
            "neighbor_hist": [int(v) for v in hist.tolist()],
            "spectral_signature": spectral_signature,
        }

    def _resolve_enable_attention_anchor(self, cfg: Dict[str, Any]) -> bool:
        """
        功能：解析锚点开关。 

        Resolve enable_attention_anchor switch from config.

        Args:
            cfg: Configuration mapping.

        Returns:
            True when attention anchor extraction is enabled.
        """
        detect_cfg = cfg.get("detect")
        if not isinstance(detect_cfg, dict):
            return False
        geometry_cfg = detect_cfg.get("geometry")
        if not isinstance(geometry_cfg, dict):
            return False
        explicit_flag = geometry_cfg.get("enable_attention_anchor")
        if isinstance(explicit_flag, bool):
            return explicit_flag
        enabled = geometry_cfg.get("enabled")
        return bool(enabled) if isinstance(enabled, bool) else False

    def _resolve_anchor_top_k(self, cfg: Dict[str, Any]) -> int:
        """
        功能：解析锚点关系 top-k 参数。

        Resolve top-k neighbor count for relation summary.

        Args:
            cfg: Configuration mapping.

        Returns:
            Integer top-k value in [1, 16].
        """
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        top_k = geometry_cfg.get("anchor_top_k", 4)
        if not isinstance(top_k, int):
            return 4
        if top_k < 1:
            return 1
        if top_k > 16:
            return 16
        return top_k

    def _build_anchor_config_domain(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：构造锚点配置输入域。

        Build canonical anchor config input domain.

        Args:
            cfg: Configuration mapping.

        Returns:
            Config domain mapping included in anchor digest discipline.
        """
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        return {
            "domain_version": "attention_anchor_cfg_domain_v1",
            "enable_attention_anchor": bool(self._resolve_enable_attention_anchor(cfg)),
            "anchor_top_k": self._resolve_anchor_top_k(cfg),
            "model_id": cfg.get("model_id"),
        }

    def _impl_identity(self) -> Dict[str, str]:
        """
        功能：导出实现身份。 

        Export implementation identity mapping.

        Args:
            None.

        Returns:
            Implementation identity mapping.
        """
        return {
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
        }


def _to_numpy_latents(latents: Any) -> np.ndarray:
    """
    功能：将 latents 转为 numpy。 

    Convert latent tensor-like object to numpy array.

    Args:
        latents: Tensor-like object.

    Returns:
        Numpy array with shape [batch, channels, h, w].

    Raises:
        TypeError: If conversion fails.
        ValueError: If shape is invalid.
    """
    if isinstance(latents, np.ndarray):
        latents_np = latents
    elif hasattr(latents, "detach") and callable(latents.detach):
        detached = latents.detach()
        if hasattr(detached, "cpu") and callable(detached.cpu):
            detached = detached.cpu()
        if hasattr(detached, "numpy") and callable(detached.numpy):
            latents_np = detached.numpy()
        else:
            raise TypeError("latents.detach() result does not expose numpy()")
    else:
        raise TypeError("latents must be numpy array or tensor-like with detach().cpu().numpy()")

    if latents_np.ndim != 4:
        raise ValueError(f"latents must be rank-4, got shape={latents_np.shape}")
    return latents_np


# Paper-faithful attention anchor map relation impl
ATTENTION_ANCHOR_MAP_RELATION_ID = "attention_anchor_map_relation_v1"
ATTENTION_ANCHOR_MAP_RELATION_VERSION = "v1"


class AttentionAnchorMapRelation:
    """
    功能：基于 attention map 关系图构建几何锚点。

    Extract geometry anchors from attention map relation graph (not token vectors).
    Implements relation_graph_topk and relation_spectral_hash binding.

    Args:
        impl_id: Implementation identifier (must be attention_anchor_map_relation_v1).
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

    def extract(self, cfg: Dict[str, Any], inputs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        功能：从 attention maps 提取关系图锚点。

        Extract relation-based geometry anchors from attention maps.

        Args:
            cfg: Configuration mapping.
            inputs: Optional runtime inputs with pipeline and attention_maps.

        Returns:
            Geometry evidence mapping with relation_digest.

        Raises:
            TypeError: If inputs are invalid.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if inputs is not None and not isinstance(inputs, dict):
            raise TypeError("inputs must be dict or None")

        detect_cfg = cfg.get("detect", {})
        geometry_cfg = detect_cfg.get("geometry", {})
        enable_attention_anchor = bool(geometry_cfg.get("enable_attention_anchor", False))

        if not enable_attention_anchor:
            return {
                "status": "absent",
                "geo_score": None,
                "relation_digest": None,
                "anchor_digest": None,
                "geometry_failure_reason": None,
                "geometry_absent_reason": "attention_anchor_disabled",
            }

        runtime_inputs = inputs or {}
        attention_maps = runtime_inputs.get("attention_maps")

        if attention_maps is None:
            return {
                "status": "absent",
                "geo_score": None,
                "relation_digest": None,
                "anchor_digest": None,
                "geometry_failure_reason": None,
                "geometry_absent_reason": "attention_maps_missing",
            }

        # Build relation graph from attention maps
        try:
            relation_graph_topk, relation_spectral_hash = self._build_relation_graph(attention_maps, cfg)
        except Exception as e:
            return {
                "status": "failed",
                "geo_score": None,
                "relation_digest": None,
                "anchor_digest": None,
                "geometry_failure_reason": f"relation_graph_failed: {str(e)}",
            }

        # Compute relation_digest from graph structure
        relation_digest = digests.canonical_sha256({
            "relation_graph_topk": relation_graph_topk,
            "relation_spectral_hash": relation_spectral_hash,
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
        })

        anchor_config_digest = digests.canonical_sha256({
            "impl_id": self.impl_id,
            "enable_attention_anchor": enable_attention_anchor,
        })

        anchor_digest = digests.canonical_sha256({
            "relation_digest": relation_digest,
            "anchor_config_digest": anchor_config_digest,
        })

        return {
            "status": "ok",
            "geo_score": None,  # Geometry score computed by sync module
            "relation_digest": relation_digest,
            "anchor_digest": anchor_digest,
            "anchor_config_digest": anchor_config_digest,
            "relation_graph_topk": relation_graph_topk,
            "relation_spectral_hash": relation_spectral_hash,
            "anchor_source_semantics": "attention_like_proxy_from_latent_correlation",
            "anchor_evidence_level": "proxy",
            "anchor_blocking_reason": "real_self_attention_map_not_wired_in_runtime",
            "anchor_semantics": {
                "attention_source": "latent_correlation_proxy",
                "attention_like": True,
                "self_attention_authentic": False,
                "proxy_version": "attention_proxy_v1",
            },
            "geometry_failure_reason": None,
        }

    def _build_relation_graph(
        self,
        attention_maps: Any,
        cfg: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """
        功能：从 attention maps 构建关系图。

        Build relation graph from attention maps.

        Args:
            attention_maps: Attention maps tensor or mapping.
            cfg: Configuration.

        Returns:
            Tuple of (relation_graph_topk, relation_spectral_hash).

        Raises:
            ValueError: If attention_maps is invalid.
        """
        # Convert attention_maps to numpy if needed
        if hasattr(attention_maps, "detach"):
            attn_np = attention_maps.detach().cpu().numpy()
        elif isinstance(attention_maps, np.ndarray):
            attn_np = attention_maps
        else:
            raise ValueError("attention_maps must be tensor or numpy array")

        # Build top-k relation graph
        # Simplified: compute pairwise correlation matrix
        if attn_np.ndim < 2:
            raise ValueError("attention_maps must be at least 2D")

        # Flatten to 2D for correlation computation
        if attn_np.ndim > 2:
            shape = attn_np.shape
            attn_2d = attn_np.reshape(shape[0], -1)
        else:
            attn_2d = attn_np

        # Compute correlation matrix
        # Simplified: top-k edges by correlation strength
        num_nodes = min(attn_2d.shape[0], 50)  # Limit to top 50 nodes
        correlation_matrix = np.corrcoef(attn_2d[:num_nodes])

        # Extract top-k edges
        k = 10
        flat_corr = correlation_matrix.flatten()
        top_k_indices = np.argsort(-np.abs(flat_corr))[:k]
        top_k_edges = [(int(idx // num_nodes), int(idx % num_nodes), float(flat_corr[idx])) 
                       for idx in top_k_indices]

        relation_graph_topk = {
            "num_nodes": num_nodes,
            "top_k": k,
            "edges_digest": digests.canonical_sha256({"edges": top_k_edges}),
        }

        # Compute spectral hash (simplified: eigenvalue-based hash)
        try:
            eigenvalues = np.linalg.eigvalsh(correlation_matrix)
            top_eigenvalues = sorted(eigenvalues[-5:], reverse=True)
            relation_spectral_hash = digests.canonical_sha256({
                "top_eigenvalues": [float(ev) for ev in top_eigenvalues]
            })
        except Exception:
            relation_spectral_hash = digests.canonical_sha256({"spectral": "failed"})

        return relation_graph_topk, relation_spectral_hash
