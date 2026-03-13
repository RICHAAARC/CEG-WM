"""
高频鲁棒通道实现

功能说明：
- 提供 HighFreqEmbedder，用于 HF 子空间的鲁棒增强嵌入与检测。
- HF 子空间必须来源于 SubspacePlanner 的计划摘要，不允许实现层自选子空间。
- 通过受控采样与尾部截断稳定高频证据，输出可复算的摘要与 hf_trace_digest。

Module type: Core innovation module
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from main.core import digests
from main.watermarking.content_chain import channel_hf



# 项目内生命名（project-internal naming）
HIGH_FREQ_TRUNCATION_CODEC_ID = "high_freq_truncation_codec"
HIGH_FREQ_TRUNCATION_CODEC_VERSION = "v2"

# 评分规则版本（与版本化决策规则绑定，非 impl ID）
CONTENT_SCORE_RULE_VERSION = "content_score_rule_v1"
HF_FAILURE_RULE_VERSION = "hf_failure_rule_v1"


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



class HighFreqTruncationCodec:
    """
        功能：HF 高频截断编解码 v2 —— planner-bounded subspace projection + tail truncation + constrained energy scoring。

        Implements a closed-loop HF truncation family:
            embed: project to planner-provided HF basis and enforce deterministic tail truncation.
            detect: project to the same basis and score constrained HF energy.
        The formal path does not derive or correlate any keyed template.

    Args:
        impl_id: Implementation identifier (must be high_freq_truncation_codec).
        impl_version: Implementation version string.
        impl_digest: Implementation digest string.

    Returns:
        None.

    Raises:
        ValueError: If constructor inputs are invalid.
    """

    def __init__(self, impl_id: str, impl_version: str, impl_digest: str) -> None:
        if not impl_id:
            raise ValueError("impl_id must be non-empty str")
        if not impl_version:
            raise ValueError("impl_version must be non-empty str")
        if not impl_digest:
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
        expected_plan_digest: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        功能：HF truncation codec v2 embed。

        Apply planner-bounded HF projection and deterministic tail truncation.

        Args:
            latents: Latent features.
            plan: Planner output mapping.
            cfg: Configuration mapping.
            rng: Reserved (unused).
            cfg_digest: Optional config digest.
            expected_plan_digest: Optional expected plan digest.

        Returns:
            HF evidence mapping with status and hf_trace_digest.

        Raises:
            TypeError: If cfg is invalid.
        """
        _ = rng
        return self._run_channel(
            latents=latents, plan=plan, cfg=cfg,
            cfg_digest=cfg_digest, expected_plan_digest=expected_plan_digest, mode="embed",
        )

    def detect(
        self,
        latents_or_features: Any,
        plan: Optional[Dict[str, Any]],
        cfg: Dict[str, Any],
        cfg_digest: Optional[str] = None,
        expected_plan_digest: Optional[str] = None,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        功能：HF truncation codec v2 detect。

        Detect HF watermark by projecting to the planner-bounded HF basis and
        scoring constrained HF energy after deterministic truncation.

        Args:
            latents_or_features: Input features.
            plan: Planner output mapping.
            cfg: Configuration mapping.
            cfg_digest: Optional config digest.
            expected_plan_digest: Optional expected plan digest.

        Returns:
            Tuple of (hf_score, hf_evidence).
            hf_score is non-negative; larger indicates stronger HF truncation evidence.

        Raises:
            TypeError: If cfg is invalid.
        """
        evidence = self._run_channel(
            latents=latents_or_features, plan=plan, cfg=cfg,
            cfg_digest=cfg_digest, expected_plan_digest=expected_plan_digest, mode="detect",
        )
        return evidence.get("hf_score"), evidence

    def _run_channel(
        self,
        latents: Any,
        plan: Optional[Dict[str, Any]],
        cfg: Dict[str, Any],
        cfg_digest: Optional[str],
        expected_plan_digest: Optional[str],
        mode: str,
    ) -> Dict[str, Any]:
        if mode not in {"embed", "detect"}:
            raise ValueError("mode must be 'embed' or 'detect'")

        watermark_cfg_raw = cfg.get("watermark", {})
        watermark_cfg: Dict[str, Any] = cast(Dict[str, Any], watermark_cfg_raw) if isinstance(watermark_cfg_raw, dict) else {}
        hf_cfg_raw = watermark_cfg.get("hf", {})
        hf_cfg: Dict[str, Any] = cast(Dict[str, Any], hf_cfg_raw) if isinstance(hf_cfg_raw, dict) else {}
        enabled = hf_cfg.get("enabled", False)
        if not enabled:
            return {
                "status": "absent",
                "hf_score": None,
                "hf_absent_reason": "hf_disabled_by_config",
                "hf_trace_digest": digests.canonical_sha256({"absent": "hf_disabled_by_config"}),
            }

        if not isinstance(plan, dict):
            return {
                "status": "mismatch",
                "hf_score": None,
                "hf_failure_reason": "hf_missing_plan",
                "hf_trace_digest": digests.canonical_sha256({"failure": "hf_missing_plan"}),
                "hf_evidence_summary": {
                    "hf_status": "mismatch",
                    "hf_failure_reason": "hf_missing_plan",
                },
            }

        plan_digest = _extract_plan_digest(plan)
        if expected_plan_digest is not None:
            if plan_digest != expected_plan_digest:
                return {
                    "status": "mismatch",
                    "hf_score": None,
                    "hf_failure_reason": "hf_plan_mismatch",
                    "hf_trace_digest": digests.canonical_sha256({"failure": "hf_plan_mismatch"}),
                    "hf_evidence_summary": {
                        "hf_status": "mismatch",
                        "hf_failure_reason": "hf_plan_mismatch",
                    },
                }

        hf_basis = _extract_hf_basis(plan)
        if not isinstance(hf_basis, dict):
            return {
                "status": "mismatch",
                "hf_score": None,
                "hf_failure_reason": "hf_subspace_missing",
                "hf_trace_digest": digests.canonical_sha256({"failure": "hf_subspace_missing"}),
                "hf_evidence_summary": {
                    "hf_status": "mismatch",
                    "hf_failure_reason": "hf_subspace_missing",
                },
            }

        channel_cfg = _build_hf_channel_cfg(cfg)

        try:
            feature_vector = _prepare_hf_feature_vector(latents, hf_basis)
        except Exception:
            return {
                "status": "failed",
                "hf_score": None,
                "hf_failure_reason": "hf_invalid_input",
                "hf_trace_digest": digests.canonical_sha256({"failure": "hf_invalid_input"}),
                "hf_evidence_summary": {
                    "hf_status": "failed",
                    "hf_failure_reason": "hf_invalid_input",
                },
            }

        coeffs = channel_hf.compute_hf_basis_projection(feature_vector, hf_basis)
        constrained_coeffs, constraint_evidence = channel_hf.apply_hf_truncation_constraint(coeffs, channel_cfg)
        threshold_percentile = float(channel_cfg.get("hf_threshold_percentile", 75.0))
        tail_ratio = float(round(1.0 - threshold_percentile / 100.0, 8))

        if mode == "embed":
            constrained_features = channel_hf.reconstruct_from_hf_coeffs(
                constrained_coeffs,
                hf_basis,
                feature_vector.shape,
            )
            trace_summary: Dict[str, Any] = {
                "impl_id": self.impl_id,
                "impl_version": self.impl_version,
                "coding_mode": "hf_projection_tail_truncation",
                "plan_digest": plan_digest,
                "cfg_digest": cfg_digest,
                "basis_rank": int(hf_basis.get("basis_rank", len(constrained_coeffs))),
                "tail_truncation_ratio": tail_ratio,
                "threshold_percentile_applied": threshold_percentile,
                **constraint_evidence,
                "mode": "embed",
            }
            hf_trace_digest = digests.canonical_sha256(trace_summary)
            return {
                "status": "ok",
                "hf_score": None,
                "hf_trace_digest": hf_trace_digest,
                "tail_truncation_ratio": tail_ratio,
                "threshold_percentile_applied": threshold_percentile,
                "latent_features_embedded": constrained_features.reshape(-1).tolist(),
                "hf_evidence_summary": {
                    "hf_status": "ok",
                    "hf_detect_variant": "projection_tail_truncation",
                    "basis_rank": int(hf_basis.get("basis_rank", len(constrained_coeffs))),
                    "tail_truncation_ratio": tail_ratio,
                    "threshold_percentile_applied": threshold_percentile,
                    **constraint_evidence,
                },
            }

        hf_score = channel_hf.extract_hf_score(feature_vector, hf_basis, channel_cfg)
        trace_summary: Dict[str, Any] = {
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
            "coding_mode": "hf_projection_tail_truncation",
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest,
            "basis_rank": int(hf_basis.get("basis_rank", len(constrained_coeffs))),
            "tail_truncation_ratio": tail_ratio,
            "threshold_percentile_applied": threshold_percentile,
            "hf_score": float(hf_score),
            "higher_is_watermarked": True,
            **constraint_evidence,
            "mode": "detect",
        }
        hf_trace_digest = digests.canonical_sha256(trace_summary)
        return {
            "status": "ok",
            "hf_score": float(hf_score),
            "hf_trace_digest": hf_trace_digest,
            "tail_truncation_ratio": tail_ratio,
            "threshold_percentile_applied": threshold_percentile,
            "hf_evidence_summary": {
                "hf_status": "ok",
                "hf_detect_variant": "projection_tail_truncation",
                "basis_rank": int(hf_basis.get("basis_rank", len(constrained_coeffs))),
                "tail_truncation_ratio": tail_ratio,
                "threshold_percentile_applied": threshold_percentile,
                **constraint_evidence,
                "higher_is_watermarked": True,
            }
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
    if image_array.ndim != 3:
        raise ValueError("image_array must be HWC array")
    if not key_material:
        raise TypeError("key_material must be non-empty str")

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
    hf_trace_summary: Dict[str, Any] = {
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


def compute_hf_trace_digest(hf_trace_summary: Dict[str, Any]) -> str:
    """
    功能：计算 HF 追踪摘要 digest。

    Compute canonical HF trace digest from summary mapping.

    Args:
        hf_trace_summary: HF trace summary mapping.

    Returns:
        SHA256 digest string.
    """
    return digests.canonical_sha256(hf_trace_summary)


def _extract_plan_digest(plan: Dict[str, Any]) -> Optional[str]:
    direct = plan.get("plan_digest")
    if isinstance(direct, str) and direct:
        return direct
    nested_plan = plan.get("plan")
    if isinstance(nested_plan, dict):
        nested_plan = cast(Dict[str, Any], nested_plan)
        nested_digest = nested_plan.get("plan_digest")
        if isinstance(nested_digest, str) and nested_digest:
            return nested_digest
    return None


def _extract_hf_basis(plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidate_plan = plan
    nested_plan = plan.get("plan")
    if isinstance(nested_plan, dict):
        candidate_plan = cast(Dict[str, Any], nested_plan)
    hf_basis = candidate_plan.get("hf_basis")
    if isinstance(hf_basis, dict):
        return cast(Dict[str, Any], hf_basis)
    return None


def _prepare_hf_feature_vector(latents: Any, hf_basis: Dict[str, Any]) -> np.ndarray:
    latents_np = np.asarray(latents, dtype=np.float32)
    feature_vector = latents_np.reshape(-1)
    basis_matrix = hf_basis.get("hf_projection_matrix")
    if basis_matrix is None:
        raise ValueError("hf_projection_matrix is required")
    basis_matrix_np = np.asarray(basis_matrix, dtype=np.float32)
    if basis_matrix_np.ndim != 2:
        raise ValueError("hf_projection_matrix must be rank-2")
    if feature_vector.shape[0] == basis_matrix_np.shape[0]:
        return feature_vector

    trajectory_feature_spec_raw = hf_basis.get("trajectory_feature_spec")
    trajectory_feature_spec = (
        cast(Dict[str, Any], trajectory_feature_spec_raw)
        if isinstance(trajectory_feature_spec_raw, dict)
        else None
    )
    if isinstance(trajectory_feature_spec, dict) and trajectory_feature_spec.get("feature_operator") == "masked_normalized_random_projection":
        from main.watermarking.content_chain.subspace.trajectory_feature_space import extract_trajectory_feature_np

        return np.asarray(extract_trajectory_feature_np(latents_np.astype(np.float64), trajectory_feature_spec), dtype=np.float32)

    raise ValueError("hf feature dimension mismatches basis and no trajectory_feature_spec is available")


def _build_hf_channel_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    watermark_cfg_raw = cfg.get("watermark", {})
    watermark_cfg: Dict[str, Any] = cast(Dict[str, Any], watermark_cfg_raw) if isinstance(watermark_cfg_raw, dict) else {}
    hf_cfg_raw = watermark_cfg.get("hf", {})
    hf_cfg: Dict[str, Any] = cast(Dict[str, Any], hf_cfg_raw) if isinstance(hf_cfg_raw, dict) else {}
    threshold_percentile = hf_cfg.get("threshold_percentile")
    if isinstance(threshold_percentile, (int, float)):
        percentile_value = float(threshold_percentile)
    else:
        tail_ratio = float(hf_cfg.get("tail_truncation_ratio", 0.1))
        tail_ratio = max(0.0, min(0.95, tail_ratio))
        percentile_value = (1.0 - tail_ratio) * 100.0
    percentile_value = max(0.0, min(100.0, percentile_value))
    return {"hf_threshold_percentile": percentile_value}


def compute_hf_attestation_score(
    hf_values: Any,
    k_hf: str,
    template_size: int = 256,
) -> Dict[str, Any]:
    """
    功能：计算 HF 通道的 attestation 得分（key-conditioned template 相关性）。

    Compute the HF channel attestation score by measuring correlation between
    the image's high-frequency feature values and a key-conditioned template.

    Template formula:
        T_HF = HKDF-Expand(k_HF, context="hf_template", length=template_size)
        t_signs = [(b >> i) & 1 for b in T_HF for i in range(8)]  → ±1 encoding

    The attestation score is the normalized dot product of HF values with t_signs.
    Score direction: higher value indicates stronger key-conditioned pattern match.

    Args:
        hf_values: HF feature values (list of float or numpy array).
        k_hf: HF channel derived key (hex str).
        template_size: Template size in bytes (default 256 = 2048 bits).

    Returns:
        Dict with keys:
        - "hf_attestation_score": float in [0, 1], higher indicates attestation match.
        - "status": "ok" | "failed".
        - "n_values_used": number of values correlated.
        - "hf_attestation_trace_digest": reproducible audit digest.

    Raises:
        TypeError: If inputs are of invalid type.
        ValueError: If k_hf is empty.
    """
    import hashlib
    import hmac as _hmac

    if not k_hf:
        raise ValueError("k_hf must be non-empty str")
    if template_size <= 0:
        raise ValueError("template_size must be positive int")

    # 派生 k_HF 字节。
    try:
        key_bytes = bytes.fromhex(k_hf)
    except ValueError:
        key_bytes = k_hf.encode("utf-8")

    # HKDF-Expand 生成 key-conditioned template（与 key_derivation.generate_hf_key_template 一致）。
    template_raw = b""
    t_prev = b""
    counter = 1
    context_info = b"hf_template"
    while len(template_raw) < template_size:
        t_i = _hmac.new(key_bytes, t_prev + context_info + bytes([counter]), hashlib.sha256).digest()
        template_raw += t_i
        t_prev = t_i
        counter += 1
    template_raw = template_raw[:template_size]

    # 将 template 展开为 ±1 符号序列。
    template_signs: List[float] = []
    for byte_val in template_raw:
        for bit_pos in range(8):
            bit = (byte_val >> bit_pos) & 1
            template_signs.append(1.0 if bit == 1 else -1.0)

    # 规范化 hf_values 为列表。
    try:
        if hasattr(hf_values, "cpu"):
            flat_hf: List[float] = [float(v) for v in hf_values.cpu().detach().reshape(-1).tolist()]
        elif hasattr(hf_values, "flatten"):
            import numpy as _np
            flat_hf = [float(v) for v in _np.asarray(hf_values, dtype=float).flatten().tolist()]
        elif isinstance(hf_values, list):
            flat_hf = [float(v) for v in cast(List[Any], hf_values)]
        else:
            flat_hf = [float(v) for v in list(hf_values)]
    except Exception:
        return {
            "hf_attestation_score": None,
            "status": "failed",
            "n_values_used": 0,
            "hf_attestation_trace_digest": digests.canonical_sha256({
                "error": "flatten_failed", "k_hf_prefix": k_hf[:8]
            }),
        }

    n_compare = min(len(flat_hf), len(template_signs))
    if n_compare <= 0:
        return {
            "hf_attestation_score": None,
            "status": "failed",
            "n_values_used": 0,
            "hf_attestation_trace_digest": digests.canonical_sha256({
                "error": "empty_hf_values", "k_hf_prefix": k_hf[:8]
            }),
        }

    # 归一化点积：sign correlation。
    dot = sum(
        float(flat_hf[i]) * template_signs[i]
        for i in range(n_compare)
    )
    abs_sum = sum(abs(float(flat_hf[i])) for i in range(n_compare))
    if abs_sum < 1e-12:
        hf_attestation_score = 0.5
    else:
        # 归一化到 [0, 1]（点积方向：正相关 → 得分高）。
        hf_attestation_score = float(0.5 + 0.5 * (dot / abs_sum))
        hf_attestation_score = max(0.0, min(1.0, hf_attestation_score))

    # 构造审计摘要（可复算）。
    trace_payload: Dict[str, Any] = {
        "channel": "hf",
        "k_hf_prefix": k_hf[:16],
        "template_size": template_size,
        "n_values_used": n_compare,
        "hf_attestation_score": round(hf_attestation_score, 6),
    }
    trace_digest = digests.canonical_sha256(trace_payload)

    return {
        "hf_attestation_score": hf_attestation_score,
        "status": "ok",
        "n_values_used": n_compare,
        "hf_attestation_trace_digest": trace_digest,
    }