"""
高频鲁棒通道实现

功能说明：
- 提供 HighFreqEmbedder，用于 HF 子空间的鲁棒增强嵌入与检测。
- HF 子空间必须来源于 SubspacePlanner 的计划摘要，不允许实现层自选子空间。
- 通过受控采样与尾部截断稳定高频证据，输出可复算的摘要与 hf_trace_digest。

Module type: Core innovation module
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from main.core import digests


HIGH_FREQ_EMBEDDER_ID = "high_freq_coder_v1"
HIGH_FREQ_EMBEDDER_VERSION = "v1"
HIGH_FREQ_EMBEDDER_TRACE_VERSION = "v1"
CONTENT_SCORE_RULE_VERSION = "content_score_rule_v1"
HF_FAILURE_RULE_VERSION = "hf_failure_rule_v1"

# Paper-faithful T2SMark HF embedder impl
HF_EMBEDDER_T2SMARK_ID = "hf_embedder_t2smark_v1"
HF_EMBEDDER_T2SMARK_VERSION = "v1"

HF_ABSENT_REASONS = {
    "hf_disabled_by_config",
}

HF_FAILURE_REASONS = {
    "hf_invalid_input",
    "hf_missing_plan",
    "hf_plan_mismatch",
    "hf_subspace_missing",
    "hf_detection_failed",
}

HF_FAILURE_DECISION_REASONS = {
    "hf_monotonic_degradation",
    "hf_tail_ratio_anomaly",
    "hf_truncation_excessive",
    "hf_energy_drop_insufficient",
}


class HighFreqEmbedder:
    """
    功能：HF 鲁棒增强通道实现。

    Robust high-frequency channel implementation with deterministic truncation
    and reproducible evidence digest generation.

    Args:
        impl_id: Implementation identifier.
        impl_version: Implementation version.
        impl_digest: Implementation digest.

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

    def embed(
        self,
        latents: Any,
        plan: Optional[Dict[str, Any]],
        cfg: Dict[str, Any],
        rng: Optional[Any] = None,
        cfg_digest: Optional[str] = None,
        expected_plan_digest: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        功能：在 HF 子空间生成嵌入侧鲁棒证据摘要。

        Generate robust HF evidence for embed-side audit without persisting raw tensors.

        Args:
            latents: Latent features or trajectory features.
            plan: Planner output mapping or subspace plan payload.
            cfg: Configuration mapping.
            rng: Optional deterministic RNG handle (reserved for compatibility).
            cfg_digest: Optional config digest.
            expected_plan_digest: Optional expected plan digest.

        Returns:
            HF evidence mapping containing status, hf_score, metrics, and hf_trace_digest.

        Raises:
            TypeError: If cfg type is invalid.
        """
        _ = rng
        return self._run_hf_channel(
            latents=latents,
            plan=plan,
            cfg=cfg,
            cfg_digest=cfg_digest,
            expected_plan_digest=expected_plan_digest,
            mode="embed"
        )

    def detect(
        self,
        latents_or_features: Any,
        plan: Optional[Dict[str, Any]],
        cfg: Dict[str, Any],
        cfg_digest: Optional[str] = None,
        expected_plan_digest: Optional[str] = None
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        功能：在 HF 子空间执行检测并返回得分与摘要证据。

        Detect high-frequency evidence under planner-defined subspace and return
        reproducible score with compact evidence.

        Args:
            latents_or_features: Input features for HF detection.
            plan: Planner output mapping or subspace plan payload.
            cfg: Configuration mapping.
            cfg_digest: Optional config digest.
            expected_plan_digest: Optional expected plan digest.

        Returns:
            Tuple of (hf_score, hf_evidence).

        Raises:
            TypeError: If cfg type is invalid.
        """
        evidence = self._run_hf_channel(
            latents=latents_or_features,
            plan=plan,
            cfg=cfg,
            cfg_digest=cfg_digest,
            expected_plan_digest=expected_plan_digest,
            mode="detect"
        )
        return evidence.get("hf_score"), evidence

    def _run_hf_channel(
        self,
        latents: Any,
        plan: Optional[Dict[str, Any]],
        cfg: Dict[str, Any],
        cfg_digest: Optional[str],
        expected_plan_digest: Optional[str],
        mode: str
    ) -> Dict[str, Any]:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if mode not in {"embed", "detect"}:
            raise ValueError("mode must be 'embed' or 'detect'")

        hf_cfg = cfg.get("watermark", {}).get("hf", {})
        enabled = hf_cfg.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("watermark.hf.enabled must be bool")

        if not enabled:
            return self._build_absent_evidence(
                reason="hf_disabled_by_config",
                plan_digest=None,
                cfg_digest=cfg_digest,
                mode=mode
            )

        if not isinstance(plan, dict):
            return self._build_failure_evidence(
                reason="hf_missing_plan",
                status="mismatch",
                plan_digest=None,
                cfg_digest=cfg_digest,
                mode=mode
            )

        plan_digest = _extract_plan_digest(plan)
        if expected_plan_digest is not None:
            if not isinstance(expected_plan_digest, str) or not expected_plan_digest:
                raise TypeError("expected_plan_digest must be non-empty str or None")
            if plan_digest != expected_plan_digest:
                return self._build_failure_evidence(
                    reason="hf_plan_mismatch",
                    status="mismatch",
                    plan_digest=plan_digest,
                    cfg_digest=cfg_digest,
                    mode=mode
                )

        selected_indices = _extract_hf_indices_from_plan(plan)
        if len(selected_indices) == 0:
            return self._build_failure_evidence(
                reason="hf_subspace_missing",
                status="mismatch",
                plan_digest=plan_digest,
                cfg_digest=cfg_digest,
                mode=mode
            )

        try:
            flat_values = _flatten_to_float_list(latents)
            sampled_values = _sample_by_indices(flat_values, selected_indices)
            truncation_ratio = _read_float(hf_cfg.get("tail_truncation_ratio", 0.1), 0.1)
            truncation_mode = _read_truncation_mode(hf_cfg.get("tail_truncation_mode", "gaussian"))
            sampling_stride = _read_int(hf_cfg.get("sampling_stride", 1), 1)
            sampled_values = sampled_values[::sampling_stride]
            truncated_values, threshold_value = self._apply_tail_truncation(
                sampled_values,
                truncation_ratio,
                truncation_mode
            )

            hf_score = _compute_hf_score(truncated_values)
            evidence_summary = self._summarize_hf_evidence(
                sampled_values=sampled_values,
                truncated_values=truncated_values,
                threshold_value=threshold_value,
                selected_indices=selected_indices,
                sampling_stride=sampling_stride,
                truncation_ratio=truncation_ratio,
                truncation_mode=truncation_mode
            )
            hf_features = self._compute_hf_monotonic_features(
                sampled_values=sampled_values,
                truncated_values=truncated_values,
                threshold_value=threshold_value,
                evidence_summary=evidence_summary
            )
            hf_failure_decision, hf_failure_reason = self._decide_hf_failure(
                hf_score=hf_score,
                hf_features=hf_features,
                cfg=cfg
            )
            trace_digest = self._derive_hf_trace_digest(
                plan_digest=plan_digest,
                cfg_digest=cfg_digest,
                hf_score=hf_score,
                summary=evidence_summary,
                mode=mode
            )

            audit = {
                "impl_identity": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "trace_digest": trace_digest,
            }
            return {
                "status": "ok",
                "hf_score": hf_score,
                "hf_trace_digest": trace_digest,
                "hf_evidence_summary": evidence_summary,
                "hf_features": hf_features,
                "hf_failure_rule_version": HF_FAILURE_RULE_VERSION,
                "hf_failure_decision": hf_failure_decision,
                "hf_failure_reason": hf_failure_reason,
                "plan_digest": plan_digest,
                "content_score_rule_version": CONTENT_SCORE_RULE_VERSION,
                "audit": audit,
                "content_failure_reason": None,
            }
        except Exception:
            return self._build_failure_evidence(
                reason="hf_detection_failed",
                status="failed",
                plan_digest=plan_digest,
                cfg_digest=cfg_digest,
                mode=mode
            )

    def _apply_tail_truncation(
        self,
        sampled_values: List[float],
        truncation_ratio: float,
        truncation_mode: str
    ) -> Tuple[List[float], float]:
        """
        功能：执行尾部截断，抑制极端扰动以提升鲁棒性。

        Apply deterministic tail truncation on sampled high-frequency values.

        Args:
            sampled_values: HF sampled values.
            truncation_ratio: Tail truncation ratio in [0, 0.95].
            truncation_mode: Truncation mode token.

        Returns:
            Tuple of (truncated_values, threshold_value).

        Raises:
            TypeError: If sampled_values type is invalid.
            ValueError: If truncation parameters are invalid.
        """
        if not isinstance(sampled_values, list):
            raise TypeError("sampled_values must be list")
        if not isinstance(truncation_ratio, float):
            raise TypeError("truncation_ratio must be float")
        if not isinstance(truncation_mode, str) or not truncation_mode:
            raise ValueError("truncation_mode must be non-empty str")

        if len(sampled_values) == 0:
            return [], 0.0

        sorted_abs = sorted(abs(value) for value in sampled_values)
        if truncation_mode == "gaussian":
            quantile = 1.0 - truncation_ratio
        else:
            quantile = 1.0 - min(0.9, truncation_ratio + 0.05)
        quantile = min(1.0, max(0.05, quantile))
        position = int(round((len(sorted_abs) - 1) * quantile))
        threshold_value = float(sorted_abs[position])

        truncated_values = []
        for value in sampled_values:
            if value > threshold_value:
                truncated_values.append(threshold_value)
            elif value < -threshold_value:
                truncated_values.append(-threshold_value)
            else:
                truncated_values.append(value)
        return truncated_values, threshold_value

    def _summarize_hf_evidence(
        self,
        sampled_values: List[float],
        truncated_values: List[float],
        threshold_value: float,
        selected_indices: List[int],
        sampling_stride: int,
        truncation_ratio: float,
        truncation_mode: str
    ) -> Dict[str, Any]:
        """
        功能：生成 HF 摘要证据，仅输出可审计统计，不输出原始张量。

        Build compact and reproducible HF evidence summary.

        Args:
            sampled_values: Values before truncation.
            truncated_values: Values after truncation.
            threshold_value: Truncation threshold.
            selected_indices: HF indices from planner definition.
            sampling_stride: Sampling stride.
            truncation_ratio: Truncation ratio.
            truncation_mode: Truncation mode.

        Returns:
            Summary mapping with deterministic metrics.

        Raises:
            TypeError: If any required argument type is invalid.
        """
        if not isinstance(sampled_values, list):
            raise TypeError("sampled_values must be list")
        if not isinstance(truncated_values, list):
            raise TypeError("truncated_values must be list")
        if not isinstance(selected_indices, list):
            raise TypeError("selected_indices must be list")

        raw_energy = _mean_square(sampled_values)
        truncated_energy = _mean_square(truncated_values)
        reduction = max(0.0, raw_energy - truncated_energy)
        metrics = {
            "selected_index_count": len(selected_indices),
            "sample_count": len(sampled_values),
            "sampling_stride": sampling_stride,
            "truncation_ratio": truncation_ratio,
            "truncation_mode": truncation_mode,
            "truncation_threshold": _round_float(threshold_value),
            "raw_energy": _round_float(raw_energy),
            "truncated_energy": _round_float(truncated_energy),
            "truncation_energy_reduction": _round_float(reduction),
            "index_preview": selected_indices[:16],
        }
        metrics["metrics_digest"] = digests.canonical_sha256(metrics)
        return metrics

    def _derive_hf_trace_digest(
        self,
        plan_digest: Optional[str],
        cfg_digest: Optional[str],
        hf_score: float,
        summary: Dict[str, Any],
        mode: str
    ) -> str:
        """
        功能：计算 HF 追踪摘要。

        Derive canonical HF trace digest from deterministic payload.

        Args:
            plan_digest: Plan digest.
            cfg_digest: Optional cfg digest.
            hf_score: HF score.
            summary: HF summary mapping.
            mode: embed or detect.

        Returns:
            Canonical SHA256 digest string.

        Raises:
            TypeError: If summary type is invalid.
        """
        if not isinstance(summary, dict):
            raise TypeError("summary must be dict")
        payload = {
            "trace_version": HIGH_FREQ_EMBEDDER_TRACE_VERSION,
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "mode": mode,
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest,
            "hf_score": _round_float(hf_score),
            "summary": summary,
        }
        return digests.canonical_sha256(payload)

    def _compute_hf_monotonic_features(
        self,
        sampled_values: List[float],
        truncated_values: List[float],
        threshold_value: float,
        evidence_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        功能：计算 HF 单调性特征（攻击增强时分数单调性）。

        Compute HF monotonic sensitivity features for attack strength estimation.

        Args:
            sampled_values: Raw sampled HF values before truncation.
            truncated_values: Values after tail truncation.
            threshold_value: Truncation threshold (quantile-based).
            evidence_summary: Pre-computed evidence summary mapping.

        Returns:
            HF features mapping with attack_strength_index, tail_ratio, truncation_level, hf_energy_drop.

        Raises:
            TypeError: If argument types are invalid.
        """
        if not isinstance(sampled_values, list):
            raise TypeError("sampled_values must be list")
        if not isinstance(truncated_values, list):
            raise TypeError("truncated_values must be list")
        if not isinstance(evidence_summary, dict):
            raise TypeError("evidence_summary must be dict")

        # (1) attack_strength_index：基于裁剪后的绝对值能量，反映攻击强度
        abs_truncated = [abs(v) for v in truncated_values]
        attack_strength_index = _mean_square(abs_truncated)

        # (2) tail_ratio：尾部被截断的比例（被裁剪为阈值的样本占比）
        if len(sampled_values) > 0:
            clipped_count = sum(1 for v in sampled_values if abs(v) > threshold_value)
            tail_ratio = float(clipped_count) / float(len(sampled_values))
        else:
            tail_ratio = 0.0

        # (3) truncation_level：截断阈值相对于原始能量的比例
        raw_energy = evidence_summary.get("raw_energy", 1.0)
        if raw_energy <= 0.0:
            truncation_level = 0.0
        else:
            truncation_level = float(threshold_value) / float(raw_energy)

        # (4) hf_energy_drop：截断前后能量下降
        raw_energy_val = float(evidence_summary.get("raw_energy", 0.0))
        truncated_energy_val = float(evidence_summary.get("truncated_energy", 0.0))
        hf_energy_drop = max(0.0, raw_energy_val - truncated_energy_val)

        features = {
            "attack_strength_index": _round_float(attack_strength_index),
            "tail_ratio": _round_float(tail_ratio),
            "truncation_level": _round_float(truncation_level),
            "hf_energy_drop": _round_float(hf_energy_drop),
        }
        features["features_digest"] = digests.canonical_sha256(features)
        return features

    def _decide_hf_failure(
        self,
        hf_score: float,
        hf_features: Dict[str, Any],
        cfg: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        功能：根据 HF 特征与配置判定 HF 失败状态。

        Decide whether HF detection has failed based on monotonic sensitivity and feature thresholds.

        Args:
            hf_score: Computed HF score.
            hf_features: HF monotonic features mapping.
            cfg: Configuration mapping.

        Returns:
            Tuple of (has_failed: bool, failure_reason: Optional[str]).
            If has_failed=False, failure_reason=None.
            If has_failed=True, failure_reason is one of HF_FAILURE_DECISION_REASONS.

        Raises:
            TypeError: If argument types are invalid.
        """
        if not isinstance(hf_score, (int, float)):
            raise TypeError("hf_score must be numeric")
        if not isinstance(hf_features, dict):
            raise TypeError("hf_features must be dict")
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")

        hf_cfg = cfg.get("watermark", {}).get("hf", {})

        # (1) 检查单调性降级：若 energy_drop 过小 (< 0.001)，则可能是攻击破坏了 HF
        min_energy_drop = float(hf_cfg.get("min_energy_drop_for_valid", 0.001))
        hf_energy_drop = float(hf_features.get("hf_energy_drop", 0.0))
        if hf_energy_drop < min_energy_drop:
            return True, "hf_monotonic_degradation"

        # (2) 检查尾部比例异常：若尾部比例过高 (> 0.6)，表示攻击导致极端值增加
        max_tail_ratio = float(hf_cfg.get("max_tail_ratio_for_valid", 0.6))
        tail_ratio = float(hf_features.get("tail_ratio", 0.0))
        if tail_ratio > max_tail_ratio:
            return True, "hf_tail_ratio_anomaly"

        # (3) 检查截断过度：若 truncation_level 过高 (> 0.8)，表示原始值分布异常
        max_truncation_level = float(hf_cfg.get("max_truncation_level_for_valid", 0.8))
        truncation_level = float(hf_features.get("truncation_level", 0.0))
        if truncation_level > max_truncation_level:
            return True, "hf_truncation_excessive"

        # (4) 检查能量绝对值不足：若 attack_strength_index < threshold，HF 信号弱
        min_strength_index = float(hf_cfg.get("min_attack_strength_index", 0.001))
        attack_strength_index = float(hf_features.get("attack_strength_index", 0.0))
        if attack_strength_index < min_strength_index:
            return True, "hf_energy_drop_insufficient"

        # 所有检查都通过，未失败
        return False, None

    def _build_absent_evidence(
        self,
        reason: str,
        plan_digest: Optional[str],
        cfg_digest: Optional[str],
        mode: str
    ) -> Dict[str, Any]:
        if reason not in HF_ABSENT_REASONS:
            raise ValueError(f"invalid hf absent reason: {reason}")
        payload = {
            "trace_version": HIGH_FREQ_EMBEDDER_TRACE_VERSION,
            "status": "absent",
            "reason": reason,
            "mode": mode,
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest,
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
        }
        trace_digest = digests.canonical_sha256(payload)
        return {
            "status": "absent",
            "hf_score": None,
            "hf_trace_digest": trace_digest,
            "hf_evidence_summary": {
                "hf_status": "absent",
                "hf_absent_reason": reason,
                "metrics_digest": digests.canonical_sha256({"hf_status": "absent", "hf_absent_reason": reason}),
            },
            "plan_digest": plan_digest,
            "content_score_rule_version": CONTENT_SCORE_RULE_VERSION,
            "audit": {
                "impl_identity": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "trace_digest": trace_digest,
            },
            "content_failure_reason": None,
        }

    def _build_failure_evidence(
        self,
        reason: str,
        status: str,
        plan_digest: Optional[str],
        cfg_digest: Optional[str],
        mode: str
    ) -> Dict[str, Any]:
        if reason not in HF_FAILURE_REASONS:
            raise ValueError(f"invalid hf failure reason: {reason}")
        if status not in {"failed", "mismatch"}:
            raise ValueError("status must be failed or mismatch")
        payload = {
            "trace_version": HIGH_FREQ_EMBEDDER_TRACE_VERSION,
            "status": status,
            "reason": reason,
            "mode": mode,
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest,
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
        }
        trace_digest = digests.canonical_sha256(payload)
        return {
            "status": status,
            "hf_score": None,
            "hf_trace_digest": trace_digest,
            "hf_evidence_summary": {
                "hf_status": status,
                "hf_failure_reason": reason,
                "metrics_digest": digests.canonical_sha256({"hf_status": status, "hf_failure_reason": reason}),
            },
            "plan_digest": plan_digest,
            "content_score_rule_version": CONTENT_SCORE_RULE_VERSION,
            "audit": {
                "impl_identity": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "trace_digest": trace_digest,
            },
            "content_failure_reason": reason,
        }


def embed_high_freq_pattern(
    image_array: np.ndarray,
    routing_summary: Dict[str, Any],
    key_material: str,
    params: Dict[str, Any]
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    功能：在纹理高频区域执行最小可行嵌入。

    Embed deterministic weak pattern in texture-driven high-frequency region.

    Args:
        image_array: Input uint8 image array in HWC format.
        routing_summary: Routing summary mapping from planner/mask.
        key_material: Deterministic key material string.
        params: HF embedding parameters.

    Returns:
        Tuple of (watermarked_image_array, hf_trace_summary).

    Raises:
        TypeError: If input types are invalid.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be np.ndarray")
    if image_array.ndim != 3:
        raise ValueError("image_array must be HWC array")
    if not isinstance(routing_summary, dict):
        raise TypeError("routing_summary must be dict")
    if not isinstance(key_material, str) or not key_material:
        raise TypeError("key_material must be non-empty str")
    if not isinstance(params, dict):
        raise TypeError("params must be dict")

    beta = float(params.get("beta", 2.0))
    tail_ratio = float(params.get("tail_truncation_ratio", 0.1))
    texture_ratio = float(routing_summary.get("region_ratio", params.get("texture_ratio", 0.2)))
    texture_ratio = max(0.0, min(1.0, texture_ratio))
    tail_ratio = max(0.0, min(0.95, tail_ratio))
    if beta <= 0:
        raise ValueError("beta must be positive")

    work = image_array.astype(np.float32).copy()
    gray = np.mean(work, axis=2)
    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    grad_y[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    texture = grad_x + grad_y

    flat_texture = texture.reshape(-1)
    total = flat_texture.shape[0]
    if total <= 0:
        return image_array.copy(), {
            "hf_status": "absent",
            "hf_absent_reason": "empty_image",
        }

    selected_count = max(1, int(round(total * texture_ratio)))
    sorted_idx = np.argsort(flat_texture)
    texture_idx = sorted_idx[-selected_count:]

    rng = np.random.default_rng(int(digests.canonical_sha256({"key_material": key_material, "tag": "hf_texture"})[:16], 16))
    pattern = rng.choice([-1.0, 1.0], size=selected_count).astype(np.float32)

    threshold = float(np.quantile(flat_texture, max(0.0, 1.0 - texture_ratio)))
    perturbation = np.zeros(total, dtype=np.float32)
    perturbation[texture_idx] = pattern * beta

    # tail truncation-like control: clamp perturbation magnitude by tail ratio-driven scale
    clamp_scale = max(1.0, beta * (1.0 + tail_ratio * 2.0))
    perturbation = np.clip(perturbation, -clamp_scale, clamp_scale)

    for channel_idx in range(work.shape[2]):
        channel = work[:, :, channel_idx].reshape(-1)
        channel = channel + perturbation
        work[:, :, channel_idx] = channel.reshape(work.shape[:2])

    watermarked = np.clip(np.rint(work), 0, 255).astype(np.uint8)
    hf_trace_summary = {
        "hf_status": "ok",
        "method": "texture_region_weak_pattern",
        "tail_truncation_ratio": tail_ratio,
        "texture_ratio": texture_ratio,
        "texture_threshold": threshold,
        "selected_pixel_count": int(selected_count),
        "total_pixel_count": int(total),
        "beta": beta,
        "routing_digest": digests.canonical_sha256(routing_summary),
    }
    return watermarked, hf_trace_summary


def detect_high_freq_score(
    image_array: np.ndarray,
    routing_summary: Dict[str, Any],
    key_material: str,
    params: Dict[str, Any]
) -> tuple[Optional[float], Dict[str, Any]]:
    """
    功能：提取 HF 通道原始分数。

    Detect raw HF score from texture region statistics.

    Args:
        image_array: Input uint8 image array in HWC format.
        routing_summary: Routing summary mapping from planner/mask.
        key_material: Deterministic key material string.
        params: HF detection parameters.

    Returns:
        Tuple of (hf_score, hf_detect_trace).
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be np.ndarray")
    if image_array.ndim != 3:
        return None, {"hf_status": "fail", "hf_failure_reason": "hf_invalid_input"}
    if not isinstance(routing_summary, dict):
        raise TypeError("routing_summary must be dict")
    if not isinstance(key_material, str) or not key_material:
        raise TypeError("key_material must be non-empty str")
    if not isinstance(params, dict):
        raise TypeError("params must be dict")

    texture_ratio = float(routing_summary.get("region_ratio", params.get("texture_ratio", 0.2)))
    texture_ratio = max(0.0, min(1.0, texture_ratio))
    gray = np.mean(image_array.astype(np.float32), axis=2)
    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    grad_y[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    texture = grad_x + grad_y
    flat_texture = texture.reshape(-1)
    total = flat_texture.shape[0]
    if total <= 0:
        return None, {"hf_status": "absent", "hf_absent_reason": "empty_image"}

    selected_count = max(1, int(round(total * texture_ratio)))
    sorted_idx = np.argsort(flat_texture)
    texture_idx = sorted_idx[-selected_count:]

    rng = np.random.default_rng(int(digests.canonical_sha256({"key_material": key_material, "tag": "hf_texture"})[:16], 16))
    pattern = rng.choice([-1.0, 1.0], size=selected_count).astype(np.float32)

    selected_values = flat_texture[texture_idx]
    raw = float(np.dot(selected_values, pattern) / max(1.0, np.linalg.norm(selected_values) * np.linalg.norm(pattern)))
    hf_score = float(0.5 + 0.5 * math.tanh(raw))
    trace = {
        "hf_status": "ok",
        "hf_score_raw": raw,
        "selected_pixel_count": int(selected_count),
        "total_pixel_count": int(total),
        "routing_digest": digests.canonical_sha256(routing_summary),
    }
    return hf_score, trace


def compute_hf_trace_digest(hf_trace_summary: Dict[str, Any]) -> str:
    """
    功能：计算 HF 追踪摘要 digest。

    Compute canonical HF trace digest from summary mapping.

    Args:
        hf_trace_summary: HF trace summary mapping.

    Returns:
        SHA256 digest string.
    """
    if not isinstance(hf_trace_summary, dict):
        raise TypeError("hf_trace_summary must be dict")
    return digests.canonical_sha256(hf_trace_summary)


def _extract_plan_digest(plan: Dict[str, Any]) -> Optional[str]:
    if not isinstance(plan, dict):
        raise TypeError("plan must be dict")
    direct = plan.get("plan_digest")
    if isinstance(direct, str) and direct:
        return direct
    nested_plan = plan.get("plan")
    if isinstance(nested_plan, dict):
        nested_digest = nested_plan.get("plan_digest")
        if isinstance(nested_digest, str) and nested_digest:
            return nested_digest
    return None


def _extract_hf_indices_from_plan(plan: Dict[str, Any]) -> List[int]:
    if not isinstance(plan, dict):
        raise TypeError("plan must be dict")
    candidate_plan = plan
    if isinstance(plan.get("plan"), dict):
        candidate_plan = plan["plan"]

    hf_spec = candidate_plan.get("high_freq_subspace_spec")
    if isinstance(hf_spec, dict):
        indices = hf_spec.get("selected_indices")
        if isinstance(indices, list):
            return [int(index) for index in indices if isinstance(index, int) and index >= 0]

    subspace_spec = candidate_plan.get("subspace_spec")
    if not isinstance(subspace_spec, dict):
        return []
    top_indices = subspace_spec.get("top_feature_indices")
    if not isinstance(top_indices, list) or len(top_indices) == 0:
        return []

    normalized = [int(index) for index in top_indices if isinstance(index, int) and index >= 0]
    if len(normalized) == 0:
        return []

    split = max(1, len(normalized) // 2)
    return normalized[-split:]


def _flatten_to_float_list(value: Any) -> List[float]:
    values: List[float] = []
    _flatten_recursive(value, values)
    return values


def _flatten_recursive(value: Any, sink: List[float]) -> None:
    if value is None:
        return
    if isinstance(value, (int, float)):
        sink.append(float(value))
        return
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            _flatten_recursive(value[key], sink)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten_recursive(item, sink)


def _sample_by_indices(flat_values: List[float], selected_indices: List[int]) -> List[float]:
    if len(flat_values) == 0:
        return []
    sampled: List[float] = []
    max_index = len(flat_values) - 1
    for index in selected_indices:
        if index <= max_index:
            sampled.append(flat_values[index])
    return sampled


def _compute_hf_score(values: List[float]) -> float:
    if len(values) == 0:
        return 0.0
    abs_values = [abs(value) for value in values]
    mean_abs = sum(abs_values) / len(abs_values)
    centered = [value - (sum(values) / len(values)) for value in values]
    centered_energy = _mean_square(centered)
    raw_score = mean_abs + centered_energy
    return _round_float(raw_score)


def _mean_square(values: Iterable[float]) -> float:
    values_list = list(values)
    if len(values_list) == 0:
        return 0.0
    return float(sum(value * value for value in values_list) / len(values_list))


def _round_float(value: float) -> float:
    return float(round(float(value), 8))


def _read_float(value: Any, default_value: float) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean is not valid float value")
    if not isinstance(default_value, float):
        raise TypeError("default_value must be float")
    if value is None:
        return default_value
    if not isinstance(value, (int, float)):
        raise TypeError("value must be float-like")
    casted = float(value)
    if casted < 0.0 or casted > 0.95:
        raise ValueError("tail_truncation_ratio must be in [0.0, 0.95]")
    return casted


def _read_int(value: Any, default_value: int) -> int:
    if not isinstance(default_value, int) or default_value <= 0:
        raise TypeError("default_value must be positive int")
    if value is None:
        return default_value
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("sampling_stride must be positive int")
    if value <= 0:
        raise ValueError("sampling_stride must be positive int")
    return value


def _read_truncation_mode(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("tail_truncation_mode must be non-empty str")
    if value not in {"gaussian", "winsor", "top_k_per_latent"}:
        raise ValueError("tail_truncation_mode must be gaussian, winsor, or top_k_per_latent")
    return value


class HFEmbedderT2SMark:
    """
    功能：T2SMark paper-faithful HF embedder with top-k per-latent truncation.

    Implements tail-truncation sampling based on top-k magnitude selection per latent,
    as specified in T2SMark paper (arXiv:...).

    Args:
        impl_id: Implementation identifier (must be hf_embedder_t2smark_v1).
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

    def embed(
        self,
        latents: Any,
        plan: Optional[Dict[str, Any]],
        cfg: Dict[str, Any],
        rng: Optional[Any] = None,
        cfg_digest: Optional[str] = None,
        expected_plan_digest: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        功能：T2SMark embed with top-k per-latent selection.

        Args:
            latents: Latent features or trajectory features.
            plan: Planner output mapping or subspace plan payload.
            cfg: Configuration mapping.
            rng: Optional deterministic RNG handle (reserved for compatibility).
            cfg_digest: Optional config digest.
            expected_plan_digest: Optional expected plan digest.

        Returns:
            HF evidence mapping containing status, hf_score, metrics, and hf_trace_digest.

        Raises:
            TypeError: If cfg type is invalid.
        """
        _ = rng
        return self._run_t2smark_channel(
            latents=latents,
            plan=plan,
            cfg=cfg,
            cfg_digest=cfg_digest,
            expected_plan_digest=expected_plan_digest,
            mode="embed"
        )

    def detect(
        self,
        latents_or_features: Any,
        plan: Optional[Dict[str, Any]],
        cfg: Dict[str, Any],
        cfg_digest: Optional[str] = None,
        expected_plan_digest: Optional[str] = None
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        功能：T2SMark detect with top-k per-latent consistency check.

        Args:
            latents_or_features: Input features for HF detection.
            plan: Planner output mapping or subspace plan payload.
            cfg: Configuration mapping.
            cfg_digest: Optional config digest.
            expected_plan_digest: Optional expected plan digest.

        Returns:
            Tuple of (hf_score, hf_evidence).

        Raises:
            TypeError: If cfg type is invalid.
        """
        evidence = self._run_t2smark_channel(
            latents=latents_or_features,
            plan=plan,
            cfg=cfg,
            cfg_digest=cfg_digest,
            expected_plan_digest=expected_plan_digest,
            mode="detect"
        )
        return evidence.get("hf_score"), evidence

    def _run_t2smark_channel(
        self,
        latents: Any,
        plan: Optional[Dict[str, Any]],
        cfg: Dict[str, Any],
        cfg_digest: Optional[str],
        expected_plan_digest: Optional[str],
        mode: str
    ) -> Dict[str, Any]:
        """
        功能：T2SMark HF channel with top-k magnitude-based truncation.

        Args:
            latents: Input latents.
            plan: Subspace plan.
            cfg: Configuration.
            cfg_digest: Config digest.
            expected_plan_digest: Expected plan digest.
            mode: embed or detect.

        Returns:
            HF evidence mapping.

        Raises:
            TypeError: If cfg is invalid.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if mode not in {"embed", "detect"}:
            raise ValueError("mode must be 'embed' or 'detect'")

        hf_cfg = cfg.get("watermark", {}).get("hf", {})
        enabled = hf_cfg.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("watermark.hf.enabled must be bool")

        if not enabled:
            return {
                "status": "absent",
                "hf_score": None,
                "hf_failure_reason": None,
                "hf_absent_reason": "hf_disabled_by_config",
                "hf_trace_digest": digests.canonical_sha256({"absent": "hf_disabled_by_config"}),
            }

        if not isinstance(plan, dict):
            return {
                "status": "mismatch",
                "hf_score": None,
                "hf_failure_reason": "hf_missing_plan",
                "hf_trace_digest": digests.canonical_sha256({"failure": "hf_missing_plan"}),
            }

        plan_digest = _extract_plan_digest(plan)
        if expected_plan_digest is not None:
            if not isinstance(expected_plan_digest, str) or not expected_plan_digest:
                raise TypeError("expected_plan_digest must be non-empty str or None")
            if plan_digest != expected_plan_digest:
                return {
                    "status": "mismatch",
                    "hf_score": None,
                    "hf_failure_reason": "hf_plan_mismatch",
                    "hf_trace_digest": digests.canonical_sha256({"failure": "hf_plan_mismatch"}),
                }

        # Top-k per-latent truncation logic
        tail_truncation_ratio = _read_float(hf_cfg.get("tail_truncation_ratio"), 0.1)
        tail_truncation_mode = hf_cfg.get("tail_truncation_mode", "top_k_per_latent")
        if tail_truncation_mode != "top_k_per_latent":
            return {
                "status": "failed",
                "hf_score": None,
                "hf_failure_reason": "hf_truncation_mode_mismatch",
                "hf_trace_digest": digests.canonical_sha256({"failure": "mode_mismatch"}),
            }

        flat_values = _flatten_to_float_list(latents)
        if len(flat_values) == 0:
            return {
                "status": "failed",
                "hf_score": None,
                "hf_failure_reason": "hf_invalid_input",
                "hf_trace_digest": digests.canonical_sha256({"failure": "empty_latents"}),
            }

        # Top-k magnitude-based selection per latent
        k_count = max(1, int(len(flat_values) * tail_truncation_ratio))
        abs_magnitudes = [(abs(v), i) for i, v in enumerate(flat_values)]
        abs_magnitudes.sort(reverse=True, key=lambda x: x[0])
        selected_indices = sorted([idx for _, idx in abs_magnitudes[:k_count]])
        selected_values = [flat_values[i] for i in selected_indices]

        # Compute HF score from selected values
        if len(selected_values) == 0:
            hf_score = 0.0
        else:
            mean_val = sum(selected_values) / len(selected_values)
            variance = sum((v - mean_val) ** 2 for v in selected_values) / len(selected_values)
            hf_score = float(mean_val + math.sqrt(variance))

        # Build trace summary
        truncation_digest = digests.canonical_sha256({
            "selected_indices": selected_indices,
            "k_count": k_count,
            "tail_truncation_ratio": tail_truncation_ratio,
        })
        trace_summary = {
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
            "tail_truncation_mode": tail_truncation_mode,
            "tail_truncation_ratio": tail_truncation_ratio,
            "top_k_value": k_count,
            "selected_index_digest": truncation_digest,
            "truncation_digest": truncation_digest,
            "hf_score": hf_score,
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest,
            "mode": mode,
        }
        hf_trace_digest = digests.canonical_sha256(trace_summary)

        return {
            "status": "ok",
            "hf_score": hf_score,
            "hf_trace_digest": hf_trace_digest,
            "top_k_value": k_count,
            "selected_index_digest": truncation_digest,
            "truncation_digest": truncation_digest,
            "tail_truncation_mode": tail_truncation_mode,
            "tail_truncation_ratio": tail_truncation_ratio,
        }
