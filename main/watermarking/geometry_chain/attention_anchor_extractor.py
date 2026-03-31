"""
File purpose: 基于 SD3 Transformer token 关系生成几何锚点摘要与稳定性指标。
Module type: General module
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from main.core import digests


def _resolve_attestation_event_digest(cfg: Dict[str, Any]) -> Optional[str]:
    """
    功能：解析锚点链 attestation 事件摘要。

    Resolve the event-level attestation digest for anchor conditioning.

    Args:
        cfg: Configuration mapping.

    Returns:
        Event binding digest or None.
    """
    runtime_node = cfg.get("attestation_runtime")
    runtime_cfg = runtime_node if isinstance(runtime_node, dict) else {}
    candidate = cfg.get("attestation_event_digest") or runtime_cfg.get("event_binding_digest") or cfg.get("attestation_digest")
    if isinstance(candidate, str) and candidate:
        return candidate
    return None


def _resolve_geo_anchor_seed(cfg: Dict[str, Any]) -> Optional[int]:
    """
    功能：解析 attestation 派生的几何锚点种子。

    Resolve the deterministic geometry anchor seed from the attestation runtime.

    Args:
        cfg: Configuration mapping.

    Returns:
        Integer seed or None.
    """
    runtime_node = cfg.get("attestation_runtime")
    runtime_cfg = runtime_node if isinstance(runtime_node, dict) else {}
    candidate = cfg.get("geo_anchor_seed")
    if isinstance(candidate, int):
        return int(candidate)
    candidate = runtime_cfg.get("geo_anchor_seed")
    if isinstance(candidate, int):
        return int(candidate)
    return None


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
    anchor_observations: Optional[Dict[str, Any]] = None

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
            "anchor_source_semantics": "token_latent_relation_summary",
            "anchor_evidence_level": "real",
            "anchor_metrics": self.stability_metrics,
            "stability_metrics": self.stability_metrics,
            "anchor_observations": self.anchor_observations,
            "resolution_binding": self.resolution_binding,
            "sync_digest": None,
            "sync_metrics": None,
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
            anchor_observations={
                "observed_anchor_candidates": relation_summary.get("observed_anchor_candidates"),
                "anchor_match_candidates": relation_summary.get("anchor_match_candidates"),
                "anchor_observation_summary": relation_summary.get("anchor_observation_summary"),
            },
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
            "candidate_confidence_mean": round(float(attn_summary.get("candidate_confidence_mean", 0.0)), 6),
            "visible_candidate_count": int(attn_summary.get("visible_candidate_count", 0)),
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
        token_vectors = np.transpose(latents_first, (1, 2, 0)).reshape(-1, channels).astype(np.float32, copy=False)

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
        similarity = np.sum(token_vectors[:, None, :] * token_vectors[None, :, :], axis=2)
        if not np.isfinite(similarity).all():
            similarity = np.nan_to_num(similarity, nan=-1.0, posinf=-1.0, neginf=-1.0)
        np.fill_diagonal(similarity, -1.0)
        effective_k = min(top_k, max(1, token_count - 1))

        neighbor_indices = np.argpartition(-similarity, effective_k - 1, axis=1)[:, :effective_k]
        neighbor_indices = np.sort(neighbor_indices, axis=1)
        attestation_event_digest = _resolve_attestation_event_digest(cfg)
        geo_anchor_seed = _resolve_geo_anchor_seed(cfg)
        anchor_offset = 0
        if attestation_event_digest is not None or geo_anchor_seed is not None:
            offset_payload = {
                "anchor_binding_version": "attention_anchor_binding_v1",
                "attestation_event_digest": attestation_event_digest if isinstance(attestation_event_digest, str) and attestation_event_digest else "<absent>",
                "geo_anchor_seed": int(geo_anchor_seed) if isinstance(geo_anchor_seed, int) else "<absent>",
                "token_count": token_count,
            }
            anchor_offset = int(digests.canonical_sha256(offset_payload)[:8], 16) % max(1, token_count)
            if anchor_offset > 0:
                neighbor_indices = (neighbor_indices + anchor_offset) % token_count
                token_vectors = np.roll(token_vectors, shift=anchor_offset, axis=0)

        hist_bins = min(64, token_count)
        hist = np.zeros(hist_bins, dtype=np.int64)
        for row in neighbor_indices:
            for neighbor in row:
                hist[int(neighbor % hist_bins)] += 1

        channel_energy = np.sum(np.square(token_vectors), axis=0)
        if not np.isfinite(channel_energy).all():
            channel_energy = np.nan_to_num(channel_energy, nan=0.0, posinf=0.0, neginf=0.0)
        spectral_signature = [round(float(v), 6) for v in np.sort(channel_energy)[::-1][:4]]

        centrality_scores = np.mean(
            similarity[np.arange(token_count)[:, None], neighbor_indices],
            axis=1,
        )
        candidate_count = min(self._resolve_anchor_candidate_count(cfg), token_count)
        candidate_indices = np.argsort(-centrality_scores)[:candidate_count]
        grid_height = int(latents_first.shape[1])
        grid_width = int(latents_first.shape[2])

        observed_anchor_candidates: list[Dict[str, Any]] = []
        anchor_match_candidates: list[Dict[str, Any]] = []
        candidate_confidences: list[float] = []
        visible_candidate_count = 0

        for rank_index, observed_index in enumerate(candidate_indices.tolist()):
            reference_index = int((observed_index - anchor_offset) % token_count)
            candidate_confidence = float(max(0.0, min(1.0, (centrality_scores[observed_index] + 1.0) / 2.0)))
            visibility = bool(candidate_confidence >= 0.15)
            if visibility:
                visible_candidate_count += 1
            candidate_confidences.append(candidate_confidence)
            descriptor = {
                "mean_similarity": round(float(centrality_scores[observed_index]), 8),
                "max_similarity": round(float(np.max(similarity[observed_index, neighbor_indices[observed_index]])), 8),
                "channel_energy": round(float(np.linalg.norm(token_vectors[observed_index])), 8),
                "neighbor_dispersion": round(float(np.std(similarity[observed_index, neighbor_indices[observed_index]])), 8),
            }
            observed_anchor_candidates.append(
                {
                    "rank": int(rank_index),
                    "observed_token_index": int(observed_index),
                    "reference_token_index": int(reference_index),
                    "observed_coord": self._index_to_normalized_coord(int(observed_index), grid_height, grid_width),
                    "reference_coord": self._index_to_normalized_coord(int(reference_index), grid_height, grid_width),
                    "confidence": round(candidate_confidence, 8),
                    "visibility": visibility,
                    "descriptor": descriptor,
                }
            )

            match_limit = min(2, int(neighbor_indices.shape[1]))
            for relation_rank, observed_neighbor in enumerate(neighbor_indices[observed_index, :match_limit].tolist()):
                reference_neighbor = int((int(observed_neighbor) - anchor_offset) % token_count)
                match_score = float(max(0.0, min(1.0, (float(similarity[observed_index, int(observed_neighbor)]) + 1.0) / 2.0)))
                anchor_match_candidates.append(
                    {
                        "candidate_rank": int(rank_index),
                        "relation_rank": int(relation_rank),
                        "source_reference_coord": self._index_to_normalized_coord(reference_neighbor, grid_height, grid_width),
                        "observed_target_coord": self._index_to_normalized_coord(int(observed_neighbor), grid_height, grid_width),
                        "match_score": round(match_score, 8),
                        "visibility": bool(match_score >= 0.1),
                    }
                )

        return {
            "summary_version": "attention_anchor_relation_summary_v1",
            "top_k": effective_k,
            "token_count": token_count,
            "neighbor_hist": [int(v) for v in hist.tolist()],
            "spectral_signature": spectral_signature,
            "attestation_event_digest": attestation_event_digest,
            "geo_anchor_seed": geo_anchor_seed,
            "anchor_offset": int(anchor_offset),
            "observed_anchor_candidates": observed_anchor_candidates,
            "anchor_match_candidates": anchor_match_candidates,
            "visible_candidate_count": int(visible_candidate_count),
            "candidate_confidence_mean": round(float(np.mean(candidate_confidences)) if candidate_confidences else 0.0, 8),
            "anchor_observation_summary": {
                "candidate_count": int(len(observed_anchor_candidates)),
                "match_count": int(len(anchor_match_candidates)),
                "visible_candidate_count": int(visible_candidate_count),
                "candidate_confidence_mean": round(float(np.mean(candidate_confidences)) if candidate_confidences else 0.0, 8),
            },
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

    def _resolve_anchor_candidate_count(self, cfg: Dict[str, Any]) -> int:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("anchor_candidate_count", 6)
        if not isinstance(value, int):
            return 6
        return max(3, min(12, value))

    def _index_to_normalized_coord(self, token_index: int, grid_height: int, grid_width: int) -> Dict[str, float]:
        row_index = int(token_index // max(1, grid_width))
        col_index = int(token_index % max(1, grid_width))
        center_y = float(max(1, grid_height - 1)) / 2.0
        center_x = float(max(1, grid_width - 1)) / 2.0
        norm_y = 0.0 if center_y <= 0.0 else (float(row_index) - center_y) / center_y
        norm_x = 0.0 if center_x <= 0.0 else (float(col_index) - center_x) / center_x
        return {
            "y": round(float(norm_y), 6),
            "x": round(float(norm_x), 6),
        }

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
            "anchor_candidate_count": self._resolve_anchor_candidate_count(cfg),
            "model_id": cfg.get("model_id"),
            "attestation_event_digest": _resolve_attestation_event_digest(cfg),
            "geo_anchor_seed": _resolve_geo_anchor_seed(cfg),
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


# 旧 v1 实现标识符，不对外导出
_ATTENTION_ANCHOR_MAP_RELATION_LEGACY_ID = "attention_anchor_map_relation_v1"
_ATTENTION_ANCHOR_MAP_RELATION_LEGACY_VERSION = "v1"

# v2：no-proxy，无真实 self-attention 时硬失败，不允许降级。
ATTENTION_ANCHOR_EXTRACTOR_ID = "attention_anchor_extractor"
ATTENTION_ANCHOR_EXTRACTOR_VERSION = "v2"


def _to_json_safe_value(value: Any) -> Any:
    """
    功能：将 numpy/tuple 等对象规范化为 JSON-safe 值。

    Normalize nested payload to JSON-safe python primitives.

    Args:
        value: Arbitrary nested value.

    Returns:
        JSON-safe value composed of dict/list/str/int/float/bool/None.
    """
    if isinstance(value, dict):
        normalized: Dict[str, Any] = {}
        for key, item in value.items():
            normalized[str(key)] = _to_json_safe_value(item)
        return normalized
    if isinstance(value, tuple):
        return [_to_json_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_to_json_safe_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_json_safe_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class AttentionAnchorMapRelation:
    """
    功能：基于 attention map 关系图构建几何锚点（内部辅助类，不在正式注册路径中）。

    Internal helper providing _build_relation_graph() for AttentionAnchorMapRelationExtractor.
    Not registered in geometry_registry; proxy semantics have been removed.
    Use AttentionAnchorMapRelationExtractor for all formal pipeline calls.

    Args:
        impl_id: Implementation identifier.
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
        top_k_edges: List[Dict[str, Any]] = []
        for idx in top_k_indices:
            src = int(idx // num_nodes)
            dst = int(idx % num_nodes)
            weight = float(flat_corr[idx])
            top_k_edges.append(
                {
                    "src": src,
                    "dst": dst,
                    "weight": weight,
                }
            )

        top_k_edges = _to_json_safe_value(top_k_edges)

        relation_graph_topk = {
            "num_nodes": num_nodes,
            "top_k": k,
            "edges": top_k_edges,
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


class AttentionAnchorMapRelationExtractor:
    """
    功能：Attention anchor map relation 提取器 —— 无 proxy，无真实 self-attention 时硬失败。

    Implements relation-graph-based geometry anchor extraction with strict
    authenticity enforcement. If runtime_self_attention is unavailable,
    returns status=failed; proxy mode is forbidden.

    Args:
        impl_id: Implementation identifier (must be attention_anchor_extractor).
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
        # v1 实现用于共享关系图构建逻辑。
        self._relation_extractor = AttentionAnchorMapRelation(impl_id, impl_version, impl_digest)

    def extract(self, cfg: Dict[str, Any], inputs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        功能：从 runtime self-attention 提取关系图锚点；无真实 attention 时硬失败。

        Extract relation-based geometry anchors from authentic self-attention maps.
        Proxy mode is permanently forbidden in v2.

        Args:
            cfg: Configuration mapping.
            inputs: Optional runtime inputs with attention_maps and attention_maps_source.

        Returns:
            Geometry evidence mapping with relation_digest.
            status=failed if runtime_self_attention is unavailable.

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
                "geometry_absent_reason": "attention_anchor_disabled",
            }

        runtime_inputs = inputs or {}
        attention_maps = runtime_inputs.get("attention_maps")
        attention_maps_source = runtime_inputs.get("attention_maps_source")
        is_authentic = isinstance(attention_maps_source, str) and attention_maps_source == "runtime_self_attention"

        if attention_maps is None:
            # v2：attention maps 缺失即视为硬失败，不允许降级到 proxy。
            return {
                "status": "failed",
                "geo_score": None,
                "relation_digest": None,
                "anchor_digest": None,
                "geometry_failure_reason": "runtime_self_attention_unavailable_proxy_forbidden_in_v2",
            }

        if not is_authentic:
            # v2：来源不是 runtime_self_attention 则视为硬失败。
            return {
                "status": "failed",
                "geo_score": None,
                "relation_digest": None,
                "anchor_digest": None,
                "geometry_failure_reason": "attention_source_not_authentic_proxy_forbidden_in_v2",
            }

        try:
            relation_graph_topk, relation_spectral_hash = self._relation_extractor._build_relation_graph(
                attention_maps, cfg
            )
        except Exception as e:
            return {
                "status": "failed",
                "geo_score": None,
                "relation_digest": None,
                "anchor_digest": None,
                "geometry_failure_reason": f"relation_graph_failed: {str(e)}",
            }

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
            "geo_score": None,
            "relation_digest": relation_digest,
            "anchor_digest": anchor_digest,
            "anchor_config_digest": anchor_config_digest,
            "relation_graph_topk": relation_graph_topk,
            "relation_spectral_hash": relation_spectral_hash,
            "anchor_source_semantics": "authentic_self_attention_from_runtime_pipeline",
            "anchor_evidence_level": "real",
            "anchor_semantics": {
                "attention_source": "runtime_self_attention",
                "attention_like": False,
                "self_attention_authentic": True,
            },
            "anchor_metrics": {
                "extraction_source": "attention_map_relation",
                "n_edges": len(relation_graph_topk.get("edges", [])) if isinstance(relation_graph_topk, dict) else 0,
                "relation_digest": relation_digest,
                "anchor_evidence_level": "real",
            },
            "geometry_failure_reason": None,
        }
