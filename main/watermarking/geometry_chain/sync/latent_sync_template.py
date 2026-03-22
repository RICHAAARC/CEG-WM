"""
File purpose: 基于 SD3 latent 频域摘要生成同步模板证据。
Module type: General module
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from main.core import digests


def _clamp_unit_interval(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


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
    sync_observations: Optional[Dict[str, Any]] = None


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


def _resolve_attestation_event_digest(cfg: Dict[str, Any]) -> Optional[str]:
    """
    功能：解析几何链 attestation 事件摘要。

    Resolve the event-level attestation digest for geometry conditioning.

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
    功能：解析几何链事件锚点种子。

    Resolve the deterministic event-conditioned geometry seed.

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


def _resolve_geo_score_repair_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：解析 GEO 主分重绑定配置。

    Resolve the optional geometry score rebinding configuration.

    Args:
        cfg: Configuration mapping.

    Returns:
        Canonical geometry score repair config mapping.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    detect_cfg = cfg.get("detect")
    detect_mapping = detect_cfg if isinstance(detect_cfg, dict) else {}
    geometry_cfg = detect_mapping.get("geometry")
    geometry_mapping = geometry_cfg if isinstance(geometry_cfg, dict) else {}
    repair_node = geometry_mapping.get("geo_score_repair")
    repair_cfg = repair_node if isinstance(repair_node, dict) else {}
    enabled_value = repair_cfg.get("enabled")
    enabled = bool(enabled_value) if isinstance(enabled_value, bool) else False
    mode_value = repair_cfg.get("mode")
    mode = mode_value.strip() if isinstance(mode_value, str) and mode_value.strip() else "template_confidence"
    return {"enabled": enabled, "mode": mode}


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

    def embed_inject(self, latents: Any, cfg: Dict[str, Any], seed: int) -> tuple[Any, Dict[str, Any]]:
        """
        功能：在 embed 侧注入 FFT 同步模板。

        Inject deterministic sync template in latent frequency domain.

        Args:
            latents: Input latent payload (numpy or tensor-like).
            cfg: Configuration mapping.
            seed: Deterministic seed.

        Returns:
            Tuple of (modified_latents, inject_trace).

        Raises:
            TypeError: If inputs are invalid.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if not isinstance(seed, int):
            raise TypeError("seed must be int")

        if not resolve_enable_latent_sync(cfg):
            return latents, {
                "status": "absent",
                "sync_inject_status": "absent",
                "sync_inject_reason": "embed_sync_disabled",
            }

        latents_np = _to_numpy_latents(latents)
        sync_strength = self._resolve_sync_strength(cfg)
        template = self._build_sync_template(latents_np.shape, cfg, seed)

        modified = latents_np.copy().astype(np.float32)
        for batch_index in range(modified.shape[0]):
            for channel_index in range(modified.shape[1]):
                fft_map = np.fft.fft2(modified[batch_index, channel_index])
                fft_map = fft_map + template * sync_strength
                modified[batch_index, channel_index] = np.real(np.fft.ifft2(fft_map)).astype(np.float32)

        template_digest = digests.canonical_sha256({
            "shape": list(template.shape),
            "seed": int(seed),
            "fft_bins": self._resolve_fft_bins(cfg),
            "attestation_event_digest": _resolve_attestation_event_digest(cfg),
            "geo_anchor_seed": _resolve_geo_anchor_seed(cfg),
        })

        inject_trace = {
            "status": "ok",
            "sync_inject_status": "ok",
            "sync_inject_strength": float(sync_strength),
            "template_digest": template_digest,
            "sync_config_digest": digests.canonical_sha256(self._build_sync_config_domain(cfg)),
        }

        if isinstance(latents, np.ndarray):
            return modified, inject_trace
        if hasattr(latents, "detach") and hasattr(latents, "device"):
            try:
                import torch

                out_tensor = torch.from_numpy(modified).to(device=latents.device)
                return out_tensor.type_as(latents), inject_trace
            except Exception:
                return modified, inject_trace
        return modified, inject_trace

    def _build_sync_template(self, shape: tuple[int, ...], cfg: Dict[str, Any], seed: int) -> np.ndarray:
        """
        功能：构建确定性 FFT 同步模板。

        Build deterministic conjugate-symmetric FFT template.

        Args:
            shape: Latent shape.
            cfg: Configuration mapping.
            seed: Deterministic seed.

        Returns:
            Complex FFT template.
        """
        if len(shape) < 4:
            raise ValueError("shape must be rank-4")

        height = int(shape[-2])
        width = int(shape[-1])
        rng = np.random.RandomState(seed)
        fft_bins = self._resolve_fft_bins(cfg)
        template = np.zeros((height, width), dtype=np.complex64)
        for _ in range(fft_bins):
            row = int(rng.randint(1, max(2, height // 2)))
            col = int(rng.randint(1, max(2, width // 2)))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            value = np.exp(1j * phase)
            template[row, col] = value
            template[-row % height, -col % width] = np.conj(value)
        return template

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
        seed_value = self._resolve_sync_seed(cfg, rng)

        validation = self.validate_sync_inputs(pipeline, latents, cfg)
        if not bool(validation.get("valid", False)):
            return SyncResult(
                status=str(validation.get("status", "fail")),
                sync_digest=None,
                sync_config_digest=None,
                sync_quality_metrics=None,
                resolution_binding=None,
                failure_reason=str(validation.get("failure_reason") or "sync_input_invalid"),
                sync_observations=None,
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
                sync_observations=None,
            )

        try:
            resolution_binding = self.build_resolution_binding(transformer, latents_np, cfg)
            sync_summary = self._build_sync_summary(latents_np, cfg)
            sync_quality_metrics = self.compute_sync_quality_metrics(sync_summary)
            template_match_metrics = self._compute_template_match_metrics(latents_np, cfg, seed_value)
            sync_quality_metrics["template_match_score"] = template_match_metrics["template_match_score"]
            sync_quality_metrics["template_match_p95"] = template_match_metrics["template_match_p95"]
            sync_quality_metrics["template_match_detected"] = template_match_metrics["template_match_detected"]
            sync_quality_metrics["template_match_threshold"] = template_match_metrics["template_match_threshold"]
            sync_quality_metrics["template_seed"] = template_match_metrics["template_seed"]
            sync_quality_metrics["template_digest"] = template_match_metrics["template_digest"]
            existing_confidence = float(sync_quality_metrics.get("match_confidence", 0.0))
            template_confidence = float(min(1.0, template_match_metrics["template_match_score"] * 6.0))
            sync_quality_metrics["template_confidence"] = round(template_confidence, 6)
            sync_quality_metrics["match_confidence"] = round(max(existing_confidence, template_confidence), 6)
            sync_config_digest = digests.canonical_sha256(self._build_sync_config_domain(cfg))

            sync_payload = {
                "summary_version": sync_summary.get("summary_version"),
                "sync_signature": sync_summary.get("sync_signature"),
                "sync_config_digest": sync_config_digest,
                "resolution_binding": resolution_binding,
                "impl_identity": self._impl_identity(),
                "template_digest": template_match_metrics["template_digest"],
                "template_match_score": template_match_metrics["template_match_score"],
                "observed_sync_peaks": sync_summary.get("observed_sync_peaks"),
                "template_support_points": sync_summary.get("template_support_points"),
                "sync_response_summary": sync_summary.get("sync_response_summary"),
            }
            sync_digest = self.compute_sync_digest(sync_payload)
            return SyncResult(
                status="ok",
                sync_digest=sync_digest,
                sync_config_digest=sync_config_digest,
                sync_quality_metrics=sync_quality_metrics,
                resolution_binding=resolution_binding,
                failure_reason=None,
                sync_observations={
                    "observed_sync_peaks": sync_summary.get("observed_sync_peaks"),
                    "template_support_points": sync_summary.get("template_support_points"),
                    "sync_response_summary": sync_summary.get("sync_response_summary"),
                    "template_support_summary": sync_summary.get("template_support_summary"),
                },
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
                sync_observations=None,
            )

    def _resolve_sync_seed(self, cfg: Dict[str, Any], rng: Any) -> int:
        """
        功能：解析同步模板重建种子。

        Resolve deterministic seed used for sync template reconstruction.

        Args:
            cfg: Configuration mapping.
            rng: Optional runtime rng handle.

        Returns:
            Deterministic integer seed.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")

        seed_value = cfg.get("seed")
        if isinstance(seed_value, int):
            base_seed = int(seed_value)
        else:
            base_seed = None

        if base_seed is None and isinstance(rng, int):
            base_seed = int(rng)
        if base_seed is None and hasattr(rng, "randint") and callable(getattr(rng, "randint")):
            try:
                sampled = rng.randint(0, 2 ** 31 - 1)
                if isinstance(sampled, int):
                    base_seed = int(sampled)
            except Exception:
                pass
        if base_seed is None:
            base_seed = 42

        attestation_event_digest = _resolve_attestation_event_digest(cfg)
        geo_anchor_seed = _resolve_geo_anchor_seed(cfg)
        if attestation_event_digest is None and geo_anchor_seed is None:
            return base_seed

        seed_binding_payload = {
            "seed_binding_version": "latent_sync_attestation_seed_v1",
            "base_seed": int(base_seed),
            "attestation_event_digest": attestation_event_digest if isinstance(attestation_event_digest, str) and attestation_event_digest else "<absent>",
            "geo_anchor_seed": int(geo_anchor_seed) if isinstance(geo_anchor_seed, int) else "<absent>",
        }
        return int(digests.canonical_sha256(seed_binding_payload)[:8], 16) & 0x7FFFFFFF

    def _compute_template_match_metrics(self, latents_np: np.ndarray, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """
        功能：计算注入模板与运行期频谱的相关性匹配指标。

        Compute correlation-based match metrics between reconstructed template and runtime FFT maps.

        Args:
            latents_np: Runtime latent array in shape [B, C, H, W].
            cfg: Configuration mapping.
            seed: Deterministic seed used for template reconstruction.

        Returns:
            Mapping with template match metrics and digest.
        """
        if not isinstance(latents_np, np.ndarray) or latents_np.ndim != 4:
            raise ValueError("latents_np must be rank-4 numpy array")
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if not isinstance(seed, int):
            raise TypeError("seed must be int")

        template = self._build_sync_template(latents_np.shape, cfg, seed)
        support_mask = np.abs(template) > 1e-8
        support_indices = np.argwhere(support_mask)
        if support_indices.size == 0:
            template_digest = digests.canonical_sha256(
                {
                    "shape": list(template.shape),
                    "seed": int(seed),
                    "fft_bins": self._resolve_fft_bins(cfg),
                    "attestation_event_digest": _resolve_attestation_event_digest(cfg),
                    "geo_anchor_seed": _resolve_geo_anchor_seed(cfg),
                }
            )
            return {
                "template_match_score": 0.0,
                "template_match_p95": 0.0,
                "template_match_detected": False,
                "template_match_threshold": 0.0,
                "template_seed": int(seed),
                "template_digest": template_digest,
            }

        support_template = template[support_mask]
        support_template_norm = float(np.linalg.norm(support_template))
        if support_template_norm <= 1e-12:
            template_digest = digests.canonical_sha256(
                {
                    "shape": list(template.shape),
                    "seed": int(seed),
                    "fft_bins": self._resolve_fft_bins(cfg),
                    "attestation_event_digest": _resolve_attestation_event_digest(cfg),
                    "geo_anchor_seed": _resolve_geo_anchor_seed(cfg),
                }
            )
            return {
                "template_match_score": 0.0,
                "template_match_p95": 0.0,
                "template_match_detected": False,
                "template_match_threshold": 0.0,
                "template_seed": int(seed),
                "template_digest": template_digest,
            }

        match_scores = []
        for batch_index in range(latents_np.shape[0]):
            for channel_index in range(latents_np.shape[1]):
                fft_map = np.fft.fft2(latents_np[batch_index, channel_index].astype(np.float32))
                support_values = fft_map[support_mask]
                support_norm = float(np.linalg.norm(support_values))
                if support_norm <= 1e-12:
                    continue
                magnitude_map = np.abs(fft_map)
                total_energy = float(np.sum(magnitude_map))
                support_energy = float(np.sum(np.abs(support_values)))
                support_energy_ratio = 0.0 if total_energy <= 1e-12 else support_energy / total_energy
                support_alignment = float(
                    np.abs(np.vdot(support_template, support_values))
                    / (support_template_norm * support_norm + 1e-12)
                )
                baseline_scale = float(np.median(magnitude_map) + 1e-12)
                support_prominence = float(
                    np.mean(np.abs(support_values) / (np.abs(support_values) + baseline_scale))
                )
                score = float(
                    max(
                        0.0,
                        min(
                            1.0,
                            0.7 * support_alignment
                            + 0.2 * np.sqrt(max(support_energy_ratio, 0.0))
                            + 0.1 * support_prominence,
                        ),
                    )
                )
                match_scores.append(score)

        if match_scores:
            match_score = float(np.mean(match_scores))
            match_p95 = float(np.percentile(np.asarray(match_scores, dtype=np.float64), 95.0))
        else:
            match_score = 0.0
            match_p95 = 0.0

        threshold = float(max(0.002, min(0.08, self._resolve_sync_strength(cfg) * 0.5 + 0.002)))
        template_digest = digests.canonical_sha256(
            {
                "shape": list(template.shape),
                "seed": int(seed),
                "fft_bins": self._resolve_fft_bins(cfg),
                "attestation_event_digest": _resolve_attestation_event_digest(cfg),
                "geo_anchor_seed": _resolve_geo_anchor_seed(cfg),
            }
        )

        return {
            "template_match_score": round(match_score, 8),
            "template_match_p95": round(match_p95, 8),
            "template_match_detected": bool(match_score >= threshold),
            "template_match_threshold": round(threshold, 8),
            "template_seed": int(seed),
            "template_digest": template_digest,
        }

    def _resolve_sync_peak_top_k(self, cfg: Dict[str, Any]) -> int:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("sync_peak_top_k", 6)
        if not isinstance(value, int):
            return 6
        return max(2, min(16, value))

    def _normalize_frequency_coordinate(self, row_index: int, col_index: int, height: int, width: int) -> Dict[str, float]:
        center_y = float(max(1, height - 1)) / 2.0
        center_x = float(max(1, width - 1)) / 2.0
        norm_y = 0.0 if center_y <= 0.0 else (float(row_index) - center_y) / center_y
        norm_x = 0.0 if center_x <= 0.0 else (float(col_index) - center_x) / center_x
        return {
            "y": round(float(norm_y), 6),
            "x": round(float(norm_x), 6),
        }

    def _extract_sync_peak_candidates(self, spectrum_center: np.ndarray, cfg: Dict[str, Any]) -> list[Dict[str, Any]]:
        if not isinstance(spectrum_center, np.ndarray) or spectrum_center.ndim != 2:
            raise TypeError("spectrum_center must be rank-2 np.ndarray")

        peak_top_k = min(self._resolve_sync_peak_top_k(cfg), int(spectrum_center.size))
        flat = spectrum_center.reshape(-1)
        if peak_top_k <= 0 or flat.size == 0:
            return []

        selected = np.argpartition(-flat, peak_top_k - 1)[:peak_top_k]
        selected = selected[np.argsort(-flat[selected])]
        global_mean = float(np.mean(flat) + 1e-12)
        global_std = float(np.std(flat) + 1e-12)
        height, width = spectrum_center.shape
        candidates: list[Dict[str, Any]] = []

        for rank_index, flat_index in enumerate(selected.tolist()):
            peak_value = float(flat[flat_index])
            row_index, col_index = np.unravel_index(int(flat_index), spectrum_center.shape)
            patch = spectrum_center[
                max(0, row_index - 1):min(height, row_index + 2),
                max(0, col_index - 1):min(width, col_index + 2),
            ]
            local_mean = float(np.mean(patch) + 1e-12)
            local_std = float(np.std(patch) + 1e-12)
            local_contrast = float((peak_value - local_mean) / local_std)
            confidence = max(
                0.0,
                min(
                    1.0,
                    0.55 * (peak_value / (peak_value + global_mean))
                    + 0.45 * (peak_value / (peak_value + global_std)),
                ),
            )
            candidates.append(
                {
                    "rank": int(rank_index),
                    "coord": {"row": int(row_index), "col": int(col_index)},
                    "normalized_coord": self._normalize_frequency_coordinate(int(row_index), int(col_index), height, width),
                    "peak_strength": round(peak_value, 8),
                    "local_mean": round(local_mean, 8),
                    "local_std": round(local_std, 8),
                    "local_contrast": round(local_contrast, 8),
                    "confidence": round(float(confidence), 8),
                    "visibility": bool(peak_value > global_mean),
                }
            )
        return candidates

    def _extract_template_support_points(self, shape: tuple[int, ...], cfg: Dict[str, Any], seed: int) -> list[Dict[str, Any]]:
        template = self._build_sync_template(shape, cfg, seed)
        support_coords = np.argwhere(np.abs(template) > 1e-8)
        if support_coords.size == 0:
            return []

        height, width = template.shape
        scored_points = []
        for row_index, col_index in support_coords.tolist():
            radius = float(np.sqrt((float(row_index) - float(height) / 2.0) ** 2 + (float(col_index) - float(width) / 2.0) ** 2))
            scored_points.append((radius, int(row_index), int(col_index)))
        scored_points.sort(key=lambda item: (item[0], item[1], item[2]))

        template_points: list[Dict[str, Any]] = []
        for rank_index, (_, row_index, col_index) in enumerate(scored_points[:self._resolve_sync_peak_top_k(cfg)]):
            template_points.append(
                {
                    "rank": int(rank_index),
                    "coord": {"row": int(row_index), "col": int(col_index)},
                    "reference_coord": self._normalize_frequency_coordinate(int(row_index), int(col_index), height, width),
                    "visibility": True,
                    "confidence": 1.0,
                }
            )
        return template_points

    def revalidate_recovered_sync(
        self,
        sync_observations: Dict[str, Any],
        inverse_transform: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        功能：在恢复坐标系中复验 sync 观测。 

        Revalidate sync observations in the recovered coordinate system using
        recovered peak coordinates against the template support points.

        Args:
            sync_observations: Sync observation payload from the attacked sample.
            inverse_transform: Inverse transform that maps observed coordinates to the recovered domain.
            cfg: Configuration mapping.

        Returns:
            Recovered-domain sync observation summary and consistency metrics.
        """
        if not isinstance(sync_observations, dict):
            raise TypeError("sync_observations must be dict")
        if not isinstance(inverse_transform, dict):
            raise TypeError("inverse_transform must be dict")
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")

        observed_sync_peaks = sync_observations.get("observed_sync_peaks") if isinstance(sync_observations.get("observed_sync_peaks"), list) else []
        template_support_points = sync_observations.get("template_support_points") if isinstance(sync_observations.get("template_support_points"), list) else []
        if not observed_sync_peaks or not template_support_points:
            return {
                "recovered_sync_peak_strength": 0.0,
                "recovered_sync_local_contrast": 0.0,
                "recovered_sync_confidence": 0.0,
                "recovered_sync_visibility": 0.0,
                "recovered_sync_support_overlap": 0.0,
                "recovered_sync_match_score": 0.0,
                "recovered_sync_consistency": 0.0,
            }

        matching_radius = max(0.08, min(0.45, float(self._resolve_inverse_recovery_radius(cfg))))
        template_points = []
        for template_point in template_support_points:
            reference_coord = template_point.get("reference_coord")
            if not isinstance(reference_coord, dict):
                continue
            template_x = reference_coord.get("x")
            template_y = reference_coord.get("y")
            if not isinstance(template_x, (int, float)) or not isinstance(template_y, (int, float)):
                continue
            template_points.append(
                {
                    "coord": np.asarray([float(template_x), float(template_y)], dtype=np.float64),
                    "visibility": bool(template_point.get("visibility", True)),
                }
            )

        if not template_points:
            return {
                "recovered_sync_peak_strength": 0.0,
                "recovered_sync_local_contrast": 0.0,
                "recovered_sync_confidence": 0.0,
                "recovered_sync_visibility": 0.0,
                "recovered_sync_support_overlap": 0.0,
                "recovered_sync_match_score": 0.0,
                "recovered_sync_consistency": 0.0,
            }

        used_template_indices: set[int] = set()
        match_scores: list[float] = []
        confidence_scores: list[float] = []
        visibility_scores: list[float] = []
        local_contrast_scores: list[float] = []
        peak_strength_scores: list[float] = []

        for peak_payload in observed_sync_peaks:
            if not isinstance(peak_payload, dict):
                continue
            normalized_coord = peak_payload.get("normalized_coord")
            recovered_coord = self._transform_normalized_coord(normalized_coord, inverse_transform)
            if recovered_coord is None:
                continue
            best_index = None
            best_distance = None
            for template_index, template_point in enumerate(template_points):
                if template_index in used_template_indices:
                    continue
                distance = float(np.linalg.norm(recovered_coord - template_point["coord"]))
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_index = template_index
            if best_index is None or best_distance is None or best_distance > matching_radius:
                continue

            used_template_indices.add(best_index)
            peak_confidence = float(peak_payload.get("confidence", 0.0))
            peak_local_contrast = _clamp_unit_interval(float(peak_payload.get("local_contrast", 0.0)) / 4.0)
            peak_strength = float(peak_payload.get("peak_strength", 0.0))
            peak_strength_norm = peak_strength / (1.0 + abs(peak_strength))
            peak_visibility = 1.0 if bool(peak_payload.get("visibility", False)) else 0.0
            distance_score = _clamp_unit_interval(1.0 - best_distance / matching_radius)
            match_score = _clamp_unit_interval(
                0.45 * distance_score
                + 0.25 * _clamp_unit_interval(peak_confidence)
                + 0.15 * peak_local_contrast
                + 0.15 * peak_visibility
            )
            match_scores.append(match_score)
            confidence_scores.append(_clamp_unit_interval(peak_confidence))
            visibility_scores.append(peak_visibility)
            local_contrast_scores.append(peak_local_contrast)
            peak_strength_scores.append(_clamp_unit_interval(peak_strength_norm))

        support_overlap = float(len(used_template_indices)) / float(max(1, len(template_points)))
        recovered_sync_match_score = float(np.mean(match_scores)) if match_scores else 0.0
        recovered_sync_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
        recovered_sync_visibility = float(np.mean(visibility_scores)) if visibility_scores else 0.0
        recovered_sync_local_contrast = float(np.mean(local_contrast_scores)) if local_contrast_scores else 0.0
        recovered_sync_peak_strength = float(np.mean(peak_strength_scores)) if peak_strength_scores else 0.0
        recovered_sync_consistency = _clamp_unit_interval(
            0.50 * recovered_sync_match_score
            + 0.30 * recovered_sync_confidence
            + 0.20 * support_overlap
        )
        return {
            "recovered_sync_peak_strength": round(recovered_sync_peak_strength, 6),
            "recovered_sync_local_contrast": round(recovered_sync_local_contrast, 6),
            "recovered_sync_confidence": round(recovered_sync_confidence, 6),
            "recovered_sync_visibility": round(recovered_sync_visibility, 6),
            "recovered_sync_support_overlap": round(_clamp_unit_interval(support_overlap), 6),
            "recovered_sync_match_score": round(recovered_sync_match_score, 6),
            "recovered_sync_consistency": round(recovered_sync_consistency, 6),
        }

    def _transform_normalized_coord(self, coord_payload: Any, transform: Dict[str, Any]) -> Optional[np.ndarray]:
        if not isinstance(coord_payload, dict):
            return None
        x_value = coord_payload.get("x")
        y_value = coord_payload.get("y")
        if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
            return None

        src_point = np.asarray([[float(x_value), float(y_value)]], dtype=np.float64)
        model_type = str(transform.get("model_type", "similarity")).lower()
        quantized = transform.get("transform_quantized") if isinstance(transform.get("transform_quantized"), dict) else {}
        theta = np.deg2rad(float(quantized.get("rotation_degree_q", 0.0)))
        scale_factor = float(quantized.get("scale_factor_q", 1.0))
        translation_x = float(quantized.get("translation_x_q", 0.0))
        translation_y = float(quantized.get("translation_y_q", 0.0))
        cos_value = float(np.cos(theta))
        sin_value = float(np.sin(theta))

        if model_type == "affine":
            transformed = np.stack(
                [
                    scale_factor * (cos_value * src_point[:, 0] - sin_value * src_point[:, 1]) + translation_x,
                    scale_factor * (sin_value * src_point[:, 0] + cos_value * src_point[:, 1]) + translation_y,
                ],
                axis=1,
            )
        else:
            transformed = np.stack(
                [
                    scale_factor * (cos_value * src_point[:, 0] - sin_value * src_point[:, 1]) + translation_x,
                    scale_factor * (sin_value * src_point[:, 0] + cos_value * src_point[:, 1]) + translation_y,
                ],
                axis=1,
            )
        return transformed[0]

    def _resolve_inverse_recovery_radius(self, cfg: Dict[str, Any]) -> float:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("align_inverse_max_residual", 0.18)
        if not isinstance(value, (int, float)):
            return 0.18 * 2.0
        return float(value) * 2.0

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
        observed_sync_peaks = summary.get("observed_sync_peaks") if isinstance(summary.get("observed_sync_peaks"), list) else []
        peak_confidence_mean = float(np.mean([float(item.get("confidence", 0.0)) for item in observed_sync_peaks])) if observed_sync_peaks else 0.0
        peak_local_contrast_mean = float(
            np.mean(
                [
                    max(0.0, min(1.0, float(item.get("local_contrast", 0.0)) / 4.0))
                    for item in observed_sync_peaks
                ]
            )
        ) if observed_sync_peaks else 0.0
        match_confidence = max(
            0.0,
            min(
                1.0,
                0.35 * max(0.0, min(1.0, peak_ratio / 8.0))
                + 0.25 * (1.0 - max(0.0, min(1.0, residual_score)))
                + 0.25 * max(0.0, min(1.0, peak_confidence_mean))
                + 0.15 * max(0.0, min(1.0, peak_local_contrast_mean)),
            ),
        )
        return {
            "peak_ratio": round(peak_ratio, 6),
            "residual_score": round(residual_score, 6),
            "radial_entropy": round(radial_entropy, 6),
            "rotation_bin": int(summary.get("rotation_bin", 0)),
            "scale_bin": int(summary.get("scale_bin", 0)),
            "match_confidence": round(match_confidence, 6),
            "observed_peak_count": int(len(observed_sync_peaks)),
            "peak_confidence_mean": round(max(0.0, min(1.0, peak_confidence_mean)), 6),
            "peak_local_contrast_mean": round(max(0.0, min(1.0, peak_local_contrast_mean)), 6),
            "availability_score": 1.0,
            "sync_evidence_level": "primary",
            "quality_components": {
                "version": "latent_sync_quality_components",
                "peak_ratio_term": round(max(0.0, min(1.0, peak_ratio / 8.0)), 6),
                "residual_term": round(1.0 - max(0.0, min(1.0, residual_score)), 6),
                "peak_confidence_term": round(max(0.0, min(1.0, peak_confidence_mean)), 6),
                "local_contrast_term": round(max(0.0, min(1.0, peak_local_contrast_mean)), 6),
            },
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
        observed_sync_peaks = self._extract_sync_peak_candidates(spectrum_center, cfg)
        template_support_points = self._extract_template_support_points(latents_np.shape, cfg, self._resolve_sync_seed(cfg, None))
        top_peak = observed_sync_peaks[0] if observed_sync_peaks else {}
        return {
            "summary_version": "latent_sync_template_summary_v1",
            "sync_signature": sync_signature,
            "peak_ratio": float(peak_ratio),
            "residual_score": float(residual_score),
            "radial_entropy": float(radial_entropy),
            "rotation_bin": rotation_bin,
            "scale_bin": scale_bin,
            "observed_sync_peaks": observed_sync_peaks,
            "template_support_points": template_support_points,
            "sync_response_summary": {
                "map_shape": [int(spectrum_center.shape[0]), int(spectrum_center.shape[1])],
                "response_mean": round(float(np.mean(spectrum_center)), 8),
                "response_std": round(float(np.std(spectrum_center)), 8),
                "response_energy": round(float(np.sum(np.square(spectrum_center))), 8),
                "top_peak_strength": round(float(top_peak.get("peak_strength", 0.0)), 8),
                "top_peak_confidence": round(float(top_peak.get("confidence", 0.0)), 8),
                "top_peak_local_contrast": round(float(top_peak.get("local_contrast", 0.0)), 8),
            },
            "template_support_summary": {
                "support_count": int(len(template_support_points)),
                "peak_limit": int(self._resolve_sync_peak_top_k(cfg)),
            },
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
            "sync_peak_top_k": self._resolve_sync_peak_top_k(cfg),
            "sync_rotation_bins": self._resolve_rotation_bins(cfg),
            "sync_scale_bins": self._resolve_scale_bins(cfg),
            "model_id": cfg.get("model_id"),
            "attestation_event_digest": _resolve_attestation_event_digest(cfg),
            "geo_anchor_seed": _resolve_geo_anchor_seed(cfg),
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

    def _resolve_sync_strength(self, cfg: Dict[str, Any]) -> float:
        embed_cfg = cfg.get("embed") if isinstance(cfg.get("embed"), dict) else {}
        geometry_cfg = embed_cfg.get("geometry") if isinstance(embed_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("sync_strength", 0.01)
        if not isinstance(value, (int, float)):
            return 0.01
        return float(max(0.0, min(0.2, value)))

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
            "sync_observations": sync_result.sync_observations,
            "resolution_binding": sync_result.resolution_binding,
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


# 内部辅助类常量：v2 quality helper（不对外导出）
_GEOMETRY_LATENT_SYNC_SD3_QUALITY_ID = "geometry_latent_sync_sd3_v2"
_GEOMETRY_LATENT_SYNC_SD3_QUALITY_VERSION = "v2"


class _GeometryLatentSyncSD3QualityHelper:
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

        # embed 侧不提供 relation_digest（无 anchor），语义为 absent 而非 mismatch。
        # detect 侧通过 precomputed_relation_digest 注入，因此不会走此分支。
        if relation_digest is None:
            return {
                "status": "absent",
                "geo_score": None,
                "sync_digest": None,
                "geometry_absent_reason": "relation_digest_absent_embed_mode",
                "sync_quality_semantics": {
                    "score_type": "interpretable_geometry_consistency",
                    "score_version": "latent_sync_geometry_consistency",
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
                    "score_version": "latent_sync_geometry_consistency",
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
                    "score_version": "latent_sync_geometry_consistency",
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
            sync_quality_metrics = self._compute_sync_quality(
                latents_np, relation_digest,
                embed_latent_stats=runtime_inputs.get("embed_latent_stats"),
            )
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
                    "score_version": "latent_sync_geometry_consistency",
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
            "geo_score": float(sync_quality_metrics.get("quality_score", 0.0)),  # 用 sync quality_score 作为几何分数
            "sync_digest": sync_digest,
            "sync_config_digest": sync_config_digest,
            "sync_quality_metrics": sync_quality_metrics,
            "sync_quality_semantics": {
                "score_type": "interpretable_geometry_consistency",
                "score_version": "latent_sync_geometry_consistency",
                "trusted_as_primary_geometry_evidence": False,
                "evidence_level": "quantitative_secondary",
            },
            "relation_digest_bound": relation_digest,
            "geometry_failure_reason": None,
        }

    def _compute_sync_quality(
        self,
        latents_np: np.ndarray,
        relation_digest: str,
        embed_latent_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        功能：计算同步质量指标。

        Compute sync quality metrics using latents and relation_digest.
        When embed_latent_stats is provided, cross-comparison between embed
        and detect spatial statistics is used as the primary quality signal,
        providing genuine discriminability between watermarked and non-watermarked
        samples. Falls back to internal-statistics mode when absent.

        Args:
            latents_np: Latents numpy array.
            relation_digest: Relation digest from anchor extractor.
            embed_latent_stats: Optional embed-side latent spatial statistics
                from embed_record.latent_spatial_stats (contrast_ratio,
                spectral_peak_ratio, peak_sharpness, stats_version).

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

        # embed vs detect cross-comparison：当 embed_latent_stats 可用时，
        # 用 embed/detect 空间统计的相似度替代单侧统计，
        # 从而提供真实的判别力：水印图像与非水印图像的 latent 空间统计应显著差异。
        if isinstance(embed_latent_stats, dict):
            embed_contrast = float(embed_latent_stats.get("contrast_ratio", 0.0))
            embed_spr = float(embed_latent_stats.get("spectral_peak_ratio", 0.0))
            # 对比相似度：1 - 差异率，差异率用 max 归一化避免除以零。
            max_cr = max(embed_contrast, contrast_ratio, 0.01)
            contrast_sim = float(max(0.0, 1.0 - abs(embed_contrast - contrast_ratio) / max_cr))
            max_spr = max(embed_spr, spectral_peak_ratio, 0.01)
            spr_sim = float(max(0.0, 1.0 - abs(embed_spr - spectral_peak_ratio) / max_spr))
            quality_score = 0.45 * contrast_sim + 0.45 * spr_sim + 0.10 * relation_alignment
            quality_score = float(max(0.0, min(1.0, quality_score)))
            uncertainty = float(max(0.0, min(1.0, 1.0 - quality_score)))
            quality_components: Dict[str, Any] = {
                "version": "latent_sync_quality_components_cross",
                "quality_mode": "cross_similarity",
                "contrast_sim": round(contrast_sim, 6),
                "spr_sim": round(spr_sim, 6),
                "relation_component": round(relation_alignment, 6),
                "embed_contrast_ratio": round(embed_contrast, 6),
                "embed_spectral_peak_ratio": round(embed_spr, 6),
                "detect_contrast_ratio": round(contrast_ratio, 6),
                "detect_spectral_peak_ratio": round(spectral_peak_ratio, 6),
                "weights": {
                    "contrast_sim": 0.45,
                    "spr_sim": 0.45,
                    "relation_component": 0.10,
                },
                "constraints": {
                    "quality_score_in_unit_interval": bool(0.0 <= quality_score <= 1.0),
                    "uncertainty_is_one_minus_quality_score": float(round(abs((1.0 - quality_score) - uncertainty), 6)),
                },
                "primary_evidence": True,
                "evidence_level": "cross_comparison",
            }
        else:
            quality_score = 0.35 * min(1.0, contrast_ratio / 2.0) + 0.35 * min(1.0, spectral_peak_ratio / 3.0) + 0.30 * relation_alignment
            quality_score = float(max(0.0, min(1.0, quality_score)))
            uncertainty = float(max(0.0, min(1.0, 1.0 - quality_score)))
            quality_components = {
                "version": "latent_sync_quality_components",
                "quality_mode": "internal_statistics",
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
            }

        return {
            "contrast_ratio": float(round(contrast_ratio, 6)),
            "peak_sharpness": float(round(peak_sharpness, 6)),
            "spectral_peak_ratio": float(round(spectral_peak_ratio, 6)),
            "relation_alignment": float(round(relation_alignment, 6)),
            "quality_score": float(round(quality_score, 6)),
            "uncertainty": float(round(uncertainty, 6)),
            "relation_digest_bound": relation_digest,
            "quality_method": "interpretable_latent_spectrum_relation_consistency",
            "quality_components": quality_components,
        }


# Paper-faithful geometry latent sync SD3 v3：geo_score 改为 template_match_score
GEOMETRY_LATENT_SYNC_SD3_ID = "geometry_latent_sync_sd3"
GEOMETRY_LATENT_SYNC_SD3_VERSION = "v3"


class GeometryLatentSyncSD3:
    """
    功能：SD3 latent sync v3，geo_score 使用 template_match_score。

    Upgraded sync module over v2 that returns template_match_score as geo_score
    instead of quality_score, closing the geometry chain to a single template
    correlation loop.

    Args:
        impl_id: Implementation identifier (must be geometry_latent_sync_sd3).
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
        # 内部 LatentSyncTemplate 实例用于 template_match_score 计算
        self._template_engine = LatentSyncTemplate(impl_id, impl_version, impl_digest)
        # v2 底层实例提供 quality/uncertainty 辅助指标  
        self._quality_extractor = _GeometryLatentSyncSD3QualityHelper(impl_id, impl_version, impl_digest)

    def extract(
        self,
        cfg: Dict[str, Any],
        inputs: Dict[str, Any] | None = None,
        sync_ctx: SyncRuntimeContext | None = None
    ) -> Dict[str, Any]:
        """
        功能：提取 sync 证据，geo_score = template_match_score。

        Extract sync evidence using template correlation as the primary geometry score.

        Args:
            cfg: Configuration mapping.
            inputs: Optional runtime inputs with relation_digest from anchor extractor.
            sync_ctx: Optional sync runtime context.

        Returns:
            Geometry evidence mapping with geo_score from template_match_score.

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

        if relation_digest is None:
            return {
                "status": "absent",
                "geo_score": None,
                "sync_digest": None,
                "geometry_absent_reason": "relation_digest_absent_embed_mode",
            }

        if not isinstance(relation_digest, str) or not relation_digest:
            return {
                "status": "mismatch",
                "geo_score": None,
                "sync_digest": None,
                "geometry_failure_reason": "relation_digest_invalid",
            }

        if sync_ctx is None or sync_ctx.latents is None:
            return {
                "status": "absent",
                "geo_score": None,
                "sync_digest": None,
                "geometry_absent_reason": "latents_missing",
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

        # (1) 用 LatentSyncTemplate 计算模板匹配分（主几何证据）
        try:
            _rng = sync_ctx.rng if sync_ctx is not None else None
            seed = self._template_engine._resolve_sync_seed(cfg, _rng)
            template_match_metrics = self._template_engine._compute_template_match_metrics(
                latents_np, cfg, seed
            )
            template_match_score = float(template_match_metrics.get("template_match_score", 0.0))
            template_confidence = float(min(1.0, template_match_score * 6.0))
            template_match_metrics["template_confidence"] = round(template_confidence, 6)
        except Exception as e:
            return {
                "status": "failed",
                "geo_score": None,
                "sync_digest": None,
                "geometry_failure_reason": f"template_match_computation_failed: {str(e)}",
            }

        # (2) 用 v2 的 quality 指标作为辅助诊断（不影响 geo_score）
        try:
            sync_quality_metrics = self._quality_extractor._compute_sync_quality(
                latents_np, relation_digest,
                embed_latent_stats=runtime_inputs.get("embed_latent_stats"),
            )
            sync_quality_metrics["template_match_score"] = template_match_score
            sync_quality_metrics["template_confidence"] = round(template_confidence, 6)
            uncertainty = sync_quality_metrics.get("uncertainty", 1.0)
        except Exception:
            sync_quality_metrics = {
                "template_match_score": template_match_score,
                "template_confidence": round(template_confidence, 6),
            }
            uncertainty = 0.0

        if uncertainty > 0.5:
            return {
                "status": "mismatch",
                "geo_score": None,
                "sync_digest": None,
                "sync_quality_metrics": sync_quality_metrics,
                "sync_quality_semantics": {
                    "score_type": "template_correlation_geometry_score",
                    "score_version": "geometry_latent_sync_sd3",
                    "trusted_as_primary_geometry_evidence": False,
                    "evidence_level": "primary",
                },
                "geometry_failure_reason": "sync_uncertainty_too_high",
            }

        sync_config_digest = digests.canonical_sha256({
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
        })
        repair_cfg = _resolve_geo_score_repair_cfg(cfg)
        geo_score_source = "template_match_score"
        geo_score = template_match_score
        geo_score_repair_active = False
        geo_score_repair_summary: Dict[str, Any] = {
            "status": "disabled",
            "mode": repair_cfg.get("mode"),
            "raw_template_match_score": template_match_score,
            "template_confidence": round(template_confidence, 6),
            "mapping": "template_match_score_clamped_linear_x6",
        }
        if bool(repair_cfg.get("enabled", False)) and repair_cfg.get("mode") == "template_confidence":
            geo_score_source = "template_confidence"
            geo_score = template_confidence
            geo_score_repair_active = True
            geo_score_repair_summary = {
                "status": "applied",
                "mode": repair_cfg.get("mode"),
                "raw_template_match_score": template_match_score,
                "template_confidence": round(template_confidence, 6),
                "mapping": "template_match_score_clamped_linear_x6",
            }
        sync_quality_metrics["geo_score_source"] = geo_score_source
        sync_quality_metrics["geo_score_repair_enabled"] = bool(repair_cfg.get("enabled", False))
        sync_quality_metrics["geo_score_repair_mode"] = repair_cfg.get("mode")
        sync_quality_metrics["geo_score_repair_active"] = geo_score_repair_active
        sync_quality_metrics["geo_score_repair_summary"] = geo_score_repair_summary
        sync_digest = digests.canonical_sha256({
            "relation_digest": relation_digest,
            "sync_config_digest": sync_config_digest,
            "geo_score_source": geo_score_source,
            "geo_score": round(geo_score, 8),
        })

        return {
            "status": "ok",
            "geo_score": geo_score,
            "sync_digest": sync_digest,
            "sync_config_digest": sync_config_digest,
            "template_match_metrics": template_match_metrics,
            "sync_quality_metrics": sync_quality_metrics,
            "sync_quality_semantics": {
                "score_type": "template_correlation_geometry_score",
                "score_version": "geometry_latent_sync_sd3",
                "trusted_as_primary_geometry_evidence": True,
                "evidence_level": "primary",
            },
            "relation_digest_bound": relation_digest,
            "geometry_failure_reason": None,
        }
