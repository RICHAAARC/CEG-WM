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



# 项目内生命名（project-internal naming）
HIGH_FREQ_TEMPLATE_CODEC_V2_ID = "high_freq_template_codec_v2"
HIGH_FREQ_TEMPLATE_CODEC_V2_VERSION = "v2"

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



class HighFreqTemplateCodecV2:
    """
    功能：HF 高频模板编解码 v2 —— keyed Rademacher template + truncation-constrained additive injection + template-matched detection。

    Implements a closed-loop HF encode/detect cycle:
      embed: derive Rademacher template from plan_digest; add alpha * template at hf_basis positions.
      detect: measure correlation between runtime values at same positions and the same template.
    Removes top_k_magnitude_based and energy-statistics detection.

    Args:
        impl_id: Implementation identifier (must be high_freq_template_codec_v2).
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

    def _derive_hf_template(self, plan_digest: str, n_positions: int) -> np.ndarray:
        """
        功能：从 plan_digest 派生 Rademacher ±1 模板，embed/detect 共用同一推导路径。

        Derive Rademacher (±1) template deterministically from plan_digest.
        Shared computation ensures embed and detect use identical template.

        Args:
            plan_digest: Plan digest binding.
            n_positions: Number of HF positions (template length).

        Returns:
            numpy array of float64 Rademacher values in {-1, +1}.
        """
        seed = int(digests.canonical_sha256({"plan_digest": plan_digest, "tag": "hf_template_v2"})[:16], 16)
        rng = np.random.default_rng(seed % (2 ** 32))
        return rng.choice([-1.0, 1.0], size=max(1, n_positions)).astype(np.float64)

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
        功能：HF template codec v2 embed with keyed template additive injection.

        Apply keyed Rademacher template at HF basis positions with truncation constraint.

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
        功能：HF template codec v2 detect with keyed template matched correlation.

        Detect HF watermark by computing whitened Pearson correlation between
        runtime values at hf_basis positions and the keyed Rademacher template.

        Args:
            latents_or_features: Input features.
            plan: Planner output mapping.
            cfg: Configuration mapping.
            cfg_digest: Optional config digest.
            expected_plan_digest: Optional expected plan digest.

        Returns:
            Tuple of (hf_score, hf_evidence).
            hf_score in [0, 1]; higher values indicate template match.

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
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if mode not in {"embed", "detect"}:
            raise ValueError("mode must be 'embed' or 'detect'")

        hf_cfg = cfg.get("watermark", {}).get("hf", {})
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
            }

        plan_digest = _extract_plan_digest(plan)
        if expected_plan_digest is not None:
            if plan_digest != expected_plan_digest:
                return {
                    "status": "mismatch",
                    "hf_score": None,
                    "hf_failure_reason": "hf_plan_mismatch",
                    "hf_trace_digest": digests.canonical_sha256({"failure": "hf_plan_mismatch"}),
                }

        selected_indices = _extract_hf_indices_from_plan(plan)
        n_positions = len(selected_indices)
        if n_positions == 0:
            return {
                "status": "mismatch",
                "hf_score": None,
                "hf_failure_reason": "hf_subspace_missing",
                "hf_trace_digest": digests.canonical_sha256({"failure": "hf_subspace_missing"}),
            }

        template = self._derive_hf_template(plan_digest, n_positions)
        alpha = float(hf_cfg.get("variance", hf_cfg.get("tau", 1.0)))
        # truncation constraint：限制注入幅值上限，避免破坏潜变量分布。
        alpha = max(0.01, min(5.0, alpha))
        correlation_scale = float(hf_cfg.get("correlation_scale", 10.0))

        flat_values = _flatten_to_float_list(latents)

        if mode == "embed":
            # （1）加性 template 注入，在 hf_basis 位置上叠加 alpha * template[i]。
            embedded = list(flat_values) if flat_values else []
            for i, idx in enumerate(selected_indices):
                if idx < len(embedded):
                    embedded[idx] += alpha * float(template[i])
            template_digest = digests.canonical_sha256({
                "plan_digest": plan_digest,
                "n_positions": n_positions,
                "alpha": alpha,
                "tag": "hf_template_v2",
            })
            trace_summary = {
                "impl_id": self.impl_id,
                "impl_version": self.impl_version,
                "coding_mode": "keyed_rademacher_template_additive",
                "n_positions": n_positions,
                "alpha": alpha,
                "plan_digest": plan_digest,
                "cfg_digest": cfg_digest,
                "template_digest": template_digest,
                "mode": "embed",
            }
            hf_trace_digest = digests.canonical_sha256(trace_summary)
            return {
                "status": "ok",
                "hf_score": None,  # embed 端不产出检测分数
                "latent_features_embedded": embedded,
                "hf_trace_digest": hf_trace_digest,
                "template_digest": template_digest,
                "n_positions": n_positions,
                "alpha": alpha,
            }
        else:
            # （2）detect 模式：计算运行期值与模板的 whitened Pearson correlation。
            if len(flat_values) == 0:
                return {
                    "status": "failed",
                    "hf_score": None,
                    "hf_failure_reason": "hf_invalid_input",
                    "hf_trace_digest": digests.canonical_sha256({"failure": "empty_latents"}),
                }
            measured = np.array(
                [flat_values[idx] if idx < len(flat_values) else 0.0 for idx in selected_indices],
                dtype=np.float64,
            )
            t_norm = float(np.linalg.norm(template))
            m_mean = float(np.mean(measured))
            m_std = max(float(np.std(measured)), 1e-8)
            m_whitened = (measured - m_mean) / m_std
            t_normalized = template / (t_norm + 1e-8)
            raw_corr = float(np.dot(m_whitened, t_normalized))
            hf_score = 1.0 / (1.0 + math.exp(-correlation_scale * raw_corr))

            template_digest = digests.canonical_sha256({
                "plan_digest": plan_digest,
                "n_positions": n_positions,
                "alpha": alpha,
                "tag": "hf_template_v2",
            })
            trace_summary = {
                "impl_id": self.impl_id,
                "impl_version": self.impl_version,
                "coding_mode": "keyed_rademacher_template_matched",
                "n_positions": n_positions,
                "raw_correlation": raw_corr,
                "hf_score": hf_score,
                "correlation_scale": correlation_scale,
                "plan_digest": plan_digest,
                "cfg_digest": cfg_digest,
                "template_digest": template_digest,
                "higher_is_watermarked": True,
                "mode": "detect",
            }
            hf_trace_digest = digests.canonical_sha256(trace_summary)
            return {
                "status": "ok",
                "hf_score": hf_score,
                "hf_trace_digest": hf_trace_digest,
                "hf_template_match_score": hf_score,
                "raw_correlation": raw_corr,
                "template_digest": template_digest,
                "n_positions": n_positions,
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

    if not isinstance(k_hf, str) or not k_hf:
        raise ValueError("k_hf must be non-empty str")
    if not isinstance(template_size, int) or template_size <= 0:
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
    template_signs = []
    for byte_val in template_raw:
        for bit_pos in range(8):
            bit = (byte_val >> bit_pos) & 1
            template_signs.append(1.0 if bit == 1 else -1.0)

    # 规范化 hf_values 为列表。
    try:
        if hasattr(hf_values, "cpu"):
            flat_hf = hf_values.cpu().detach().reshape(-1).tolist()
        elif hasattr(hf_values, "flatten"):
            import numpy as _np
            flat_hf = _np.asarray(hf_values, dtype=float).flatten().tolist()
        elif isinstance(hf_values, list):
            flat_hf = [float(v) for v in hf_values]
        else:
            flat_hf = list(hf_values)
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
    trace_payload = {
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