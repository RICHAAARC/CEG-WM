"""
低频水印编码器

功能说明：
- 低频 (LF) 子空间水印编码核心实现。
- 实现伪高斯采样（Pseudogaussian sampling）：codeword * |randn()|。
- 实现 erf-based 后验概率恢复用于检测。
- 生成可复算的 lf_trace_digest，绑定配置、计划与编码参数。
- 严格区分 absent（未启用）、failed（异常）、mismatch（参数不一致）语义。

Module type: Core innovation module
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from main.core import digests

from . import channel_lf
from .interfaces import ContentEvidence
from .ldpc_codec import build_ldpc_spec, decode_soft_llr


def erf(x: float) -> float:
    """
    功能：误差函数（Error Function）近似实现。

    Approximation of error function erf(x) for posterior recovery.
    Uses standard approximation with max error < 1.5e-7.

    Args:
        x: Input value.

    Returns:
        erf(x) approximation.

    Raises:
        None.
    """
    # 标准 erf 近似公式（Abramowitz and Stegun）。
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return sign * y


def sample_pseudogaussian(codeword: list, seed: int) -> list:
    """
    功能：伪高斯采样（Pseudogaussian sampling）。

    Implements pseudogaussian sampling: codeword * |randn()|.
    Uses deterministic pseudo-random generation for reproducibility.

    Args:
        codeword: Binary codeword (±1 values).
        seed: Random seed for reproducibility.

    Returns:
        Pseudogaussian samples (codeword weighted by absolute Gaussian).

    Raises:
        None.
    """
    import random
    random.seed(seed)

    pseudogaussian = []
    for c in codeword:
        # 使用 Box-Muller 变换生成伪高斯随机数。
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        # 取绝对值并乘以码字。
        pseudogaussian.append(c * abs(z))

    return pseudogaussian


def recover_posteriors_erf(latents: list, variance: float = 1.5) -> list:
    """
    功能：基于 erf 从 latents 恢复后验概率。

    Recover posterior probabilities from latents using error function.
    Implements: erf(z / sqrt(2 * variance * (1 + variance))).

    Args:
        latents: Latent features (flattened vector).
        variance: Variance parameter (default 1.5, for the pseudogaussian detection mechanism).

    Returns:
        Posterior probabilities for each latent.

    Raises:
        None.
    """
    denominator = math.sqrt(2 * variance * (1 + variance))
    posteriors = []
    for z in latents:
        posterior = erf(z / denominator)
        posteriors.append(posterior)

    return posteriors



# 项目内生命名（project-internal naming）
LOW_FREQ_TEMPLATE_CODEC_ID = "low_freq_template_codec"
LOW_FREQ_TEMPLATE_CODEC_VERSION = "v2"

def encode_low_freq_dct(
    image_array: np.ndarray,
    band_spec: Dict[str, Any],
    key_material: str,
    params: Dict[str, Any]
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    功能：在 LF 子域执行块 DCT 的最小可行嵌入。

    Apply minimal viable LF embedding using block DCT coefficient modulation.

    Args:
        image_array: Input uint8 image array in HWC format.
        band_spec: Planner band specification mapping.
        key_material: Deterministic key material string.
        params: LF embedding parameters.

    Returns:
        Tuple of (watermarked_image_array, lf_trace_summary).

    Raises:
        TypeError: If input types are invalid.
        ValueError: If parameter values are invalid.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be np.ndarray")
    if image_array.ndim != 3:
        raise ValueError("image_array must be HWC array")
    if not isinstance(band_spec, dict):
        raise TypeError("band_spec must be dict")
    if not isinstance(key_material, str) or not key_material:
        raise TypeError("key_material must be non-empty str")
    if not isinstance(params, dict):
        raise TypeError("params must be dict")

    block_size = int(params.get("dct_block_size", 8))
    alpha = float(params.get("alpha", 1.5))
    redundancy = int(params.get("redundancy", 1))
    coeff_indices_raw = params.get("lf_coeff_indices", [(1, 1), (1, 2), (2, 1)])
    if block_size <= 1:
        raise ValueError("dct_block_size must be > 1")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if redundancy <= 0:
        raise ValueError("redundancy must be positive")
    if not isinstance(coeff_indices_raw, list) or len(coeff_indices_raw) == 0:
        raise ValueError("lf_coeff_indices must be non-empty list")

    coeff_indices = []
    for item in coeff_indices_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("each lf_coeff_indices item must be 2-length sequence")
        ci = int(item[0])
        cj = int(item[1])
        if ci < 0 or cj < 0 or ci >= block_size or cj >= block_size:
            raise ValueError("lf_coeff_indices values out of block range")
        coeff_indices.append((ci, cj))

    height, width, channels = image_array.shape
    blocks_h = height // block_size
    blocks_w = width // block_size
    total_blocks = blocks_h * blocks_w
    if total_blocks <= 0:
        return image_array.copy(), {
            "lf_status": "absent",
            "lf_absent_reason": "image_too_small_for_lf_blocks",
        }

    lf_selector = band_spec.get("lf_selector_summary") if isinstance(band_spec.get("lf_selector_summary"), dict) else {}
    region_ratio = float(lf_selector.get("region_ratio", 1.0))
    region_ratio = max(0.0, min(1.0, region_ratio))

    import random
    rng_seed = int(digests.canonical_sha256({"key_material": key_material, "tag": "lf_dct"})[:16], 16)
    rng = random.Random(rng_seed)

    all_block_ids = list(range(total_blocks))
    rng.shuffle(all_block_ids)
    selected_count = max(1, int(round(total_blocks * region_ratio)))
    selected_count = min(total_blocks, selected_count * redundancy)
    selected_blocks = all_block_ids[:selected_count]

    work = image_array.astype(np.float32).copy()
    dct_matrix = _build_dct_matrix(block_size)
    idct_matrix = dct_matrix.T

    coeff_ops = 0
    bit_values = []
    for block_id in selected_blocks:
        block_row = block_id // blocks_w
        block_col = block_id % blocks_w
        r0 = block_row * block_size
        c0 = block_col * block_size
        bit = -1.0 if rng.random() < 0.5 else 1.0
        bit_values.append(bit)
        for channel_idx in range(channels):
            # np.ascontiguousarray 确保切片内存连续，避免非连续视图传入 BLAS 矩阵乘法时触发进程崩溃。
            block = np.ascontiguousarray(
                work[r0:r0 + block_size, c0:c0 + block_size, channel_idx]
            )
            coeff = _dct2(block, dct_matrix)
            for ci, cj in coeff_indices:
                coeff[ci, cj] = coeff[ci, cj] + bit * alpha
                coeff_ops += 1
            restored = _idct2(coeff, idct_matrix)
            work[r0:r0 + block_size, c0:c0 + block_size, channel_idx] = restored

    watermarked = np.clip(np.rint(work), 0, 255).astype(np.uint8)
    lf_trace_summary = {
        "lf_status": "ok",
        "method": "block_dct_lf_modulation",
        "block_size": block_size,
        "coeff_indices": [[ci, cj] for ci, cj in coeff_indices],
        "alpha": alpha,
        "redundancy": redundancy,
        "lf_region_ratio": region_ratio,
        "selected_block_count": selected_count,
        "total_block_count": total_blocks,
        "coeff_operation_count": coeff_ops,
        "bit_mean": float(sum(bit_values) / len(bit_values)) if bit_values else 0.0,
        "band_spec_digest": digests.canonical_sha256(band_spec),
    }
    return watermarked, lf_trace_summary


def detect_low_freq_score(
    image_array: np.ndarray,
    band_spec: Dict[str, Any],
    key_material: str,
    params: Dict[str, Any]
) -> tuple[Optional[float], Dict[str, Any]]:
    """
    功能：提取 LF 通道原始分数，用于后续 NP 校准。

    Extract LF raw score from block DCT coefficients for calibration readiness.

    Args:
        image_array: Input uint8 image array in HWC format.
        band_spec: Planner band specification mapping.
        key_material: Deterministic key material string.
        params: LF detection parameters.

    Returns:
        Tuple of (lf_score, lf_detect_trace).

    Raises:
        TypeError: If input types are invalid.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be np.ndarray")
    if image_array.ndim != 3:
        return None, {"lf_status": "fail", "lf_failure_reason": "lf_invalid_input"}
    if not isinstance(band_spec, dict):
        raise TypeError("band_spec must be dict")
    if not isinstance(key_material, str) or not key_material:
        raise TypeError("key_material must be non-empty str")
    if not isinstance(params, dict):
        raise TypeError("params must be dict")

    block_size = int(params.get("dct_block_size", 8))
    coeff_indices_raw = params.get("lf_coeff_indices", [(1, 1), (1, 2), (2, 1)])
    coeff_indices = [(int(v[0]), int(v[1])) for v in coeff_indices_raw if isinstance(v, (list, tuple)) and len(v) == 2]
    if block_size <= 1 or len(coeff_indices) == 0:
        return None, {"lf_status": "fail", "lf_failure_reason": "lf_invalid_params"}

    height, width, channels = image_array.shape
    blocks_h = height // block_size
    blocks_w = width // block_size
    total_blocks = blocks_h * blocks_w
    if total_blocks <= 0:
        return None, {"lf_status": "absent", "lf_absent_reason": "image_too_small_for_lf_blocks"}

    lf_selector = band_spec.get("lf_selector_summary") if isinstance(band_spec.get("lf_selector_summary"), dict) else {}
    region_ratio = float(lf_selector.get("region_ratio", 1.0))
    region_ratio = max(0.0, min(1.0, region_ratio))

    import random
    rng_seed = int(digests.canonical_sha256({"key_material": key_material, "tag": "lf_dct"})[:16], 16)
    rng = random.Random(rng_seed)
    all_block_ids = list(range(total_blocks))
    rng.shuffle(all_block_ids)
    selected_count = max(1, int(round(total_blocks * region_ratio)))
    selected_blocks = all_block_ids[:selected_count]
    bit_signs = {block_id: (-1.0 if rng.random() < 0.5 else 1.0) for block_id in selected_blocks}

    dct_matrix = _build_dct_matrix(block_size)
    score_acc = 0.0
    score_count = 0
    work = image_array.astype(np.float32)
    for block_id in selected_blocks:
        block_row = block_id // blocks_w
        block_col = block_id % blocks_w
        r0 = block_row * block_size
        c0 = block_col * block_size
        expected_sign = bit_signs[block_id]
        for channel_idx in range(channels):
            block = work[r0:r0 + block_size, c0:c0 + block_size, channel_idx]
            coeff = _dct2(block, dct_matrix)
            for ci, cj in coeff_indices:
                score_acc += expected_sign * float(coeff[ci, cj])
                score_count += 1

    if score_count <= 0:
        return None, {"lf_status": "absent", "lf_absent_reason": "no_lf_coefficients"}

    raw = score_acc / float(score_count)
    lf_score = float(0.5 + 0.5 * math.tanh(raw / 10.0))
    trace = {
        "lf_status": "ok",
        "lf_score_raw": float(raw),
        "lf_score_count": score_count,
        "selected_block_count": selected_count,
        "total_block_count": total_blocks,
        "band_spec_digest": digests.canonical_sha256(band_spec),
    }
    return lf_score, trace


def compute_lf_trace_digest(lf_trace_summary: Dict[str, Any]) -> str:
    """
    功能：计算 LF 追踪摘要 digest。

    Compute canonical LF trace digest from summary mapping.

    Args:
        lf_trace_summary: LF trace summary mapping.

    Returns:
        SHA256 digest string.
    """
    if not isinstance(lf_trace_summary, dict):
        raise TypeError("lf_trace_summary must be dict")
    return digests.canonical_sha256(lf_trace_summary)


def _build_dct_matrix(block_size: int) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    matrix = np.zeros((block_size, block_size), dtype=np.float32)
    scale0 = math.sqrt(1.0 / block_size)
    scale = math.sqrt(2.0 / block_size)
    for i in range(block_size):
        alpha = scale0 if i == 0 else scale
        for j in range(block_size):
            matrix[i, j] = alpha * math.cos(math.pi * (2 * j + 1) * i / (2.0 * block_size))
    return matrix


def _dct2(block: np.ndarray, dct_matrix: np.ndarray) -> np.ndarray:
    return dct_matrix @ block @ dct_matrix.T


def _idct2(coeff: np.ndarray, idct_matrix: np.ndarray) -> np.ndarray:
    return idct_matrix @ coeff @ idct_matrix.T


def _flatten_to_list(value: Any) -> list:
    """
    功能：递归扁平化为浮点列表。

    Flatten nested structure to float list.

    Args:
        value: Input value (can be nested).

    Returns:
        Flattened list of floats.
    """
    result = []
    _flatten_recursive_to_list(value, result)
    return result


def _flatten_recursive_to_list(value: Any, sink: list) -> None:
    """
    功能：递归扁平化辅助函数。

    Recursive helper for flattening.

    Args:
        value: Current value.
        sink: Output list.

    Returns:
        None.
    """
    if value is None:
        return
    if isinstance(value, (int, float)):
        sink.append(float(value))
        return
    if isinstance(value, np.ndarray):
        for item in value.reshape(-1).tolist():
            _flatten_recursive_to_list(item, sink)
        return
    # 兼容 torch.Tensor 输入但不引入硬依赖。
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        try:
            tensor_np = value.detach().cpu().numpy()
            if isinstance(tensor_np, np.ndarray):
                for item in tensor_np.reshape(-1).tolist():
                    _flatten_recursive_to_list(item, sink)
                return
        except Exception:
            # tensor 转换失败时，回退到后续通用分支。
            pass
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            _flatten_recursive_to_list(value[key], sink)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten_recursive_to_list(item, sink)


# ————————————————————————————
# Cryptographic generation attestation 扩展（附加，不破坏既有接口）
# ————————————————————————————

def compute_lf_attestation_score(
    latent_features: Any,
    k_lf: str,
    attestation_digest: str,
    lf_params: Optional[Dict[str, Any]] = None,
    payload_length: int = 48,
) -> Dict[str, Any]:
    """
    功能：计算 LF 通道的 attestation 得分（检测侧）。

    Compute the LF channel attestation score by measuring correlation between
    the image's LF latent features and the expected attestation payload.

    The attestation payload is derived as:
        payload = truncate(HMAC(k_LF, d_A), payload_length)
        expected_bits = [(b >> i) & 1 for b in payload for i in range(8)]
    The score is the fraction of LF posteriors whose sign agrees with expected bits.
    Score direction: higher value indicates stronger attestation evidence.

    Args:
        latent_features: LF latent features (list or array) from the image.
        k_lf: LF channel derived key (hex str or bytes).
        attestation_digest: Attestation digest d_A (hex str).
        lf_params: Optional LF parameter overrides (variance, block_length, etc.).
        payload_length: Attestation payload length in bytes (default 48 = 384 bits).

    Returns:
        Dict with keys:
        - "lf_attestation_score": float in [0, 1], higher indicates attestation match.
        - "status": "ok" | "failed".
        - "n_bits_compared": number of bits compared.
        - "attestation_digest": echoed d_A.
        - "lf_attestation_trace_digest": reproducible audit digest.

    Raises:
        TypeError: If inputs are of invalid type.
        ValueError: If inputs are empty or invalid.
    """
    from .low_freq_coder import recover_posteriors_erf
    import hashlib
    import hmac as _hmac

    if not isinstance(k_lf, str) or not k_lf:
        raise ValueError("k_lf must be non-empty str")
    if not isinstance(attestation_digest, str) or not attestation_digest:
        raise ValueError("attestation_digest must be non-empty str")

    # 解析 LF 参数（支持覆盖）。
    params = lf_params or {}
    variance = float(params.get("variance", 1.5))
    block_length = int(params.get("block_length", min(payload_length * 8, 256)))

    # 派生 k_LF 字节。
    try:
        key_bytes = bytes.fromhex(k_lf)
    except ValueError:
        key_bytes = k_lf.encode("utf-8")

    # 计算 attestation payload bits：HMAC(k_LF, d_A)[:payload_length]。
    message = attestation_digest.encode("utf-8")
    if payload_length <= 32:
        raw_payload = _hmac.new(key_bytes, message, hashlib.sha256).digest()[:payload_length]
    else:
        # HKDF-Expand 延伸（与 key_derivation.compute_lf_attestation_payload 一致）。
        prk = _hmac.new(key_bytes, message, hashlib.sha256).digest()
        raw_payload = b""
        t_prev = b""
        counter = 1
        while len(raw_payload) < payload_length:
            t_i = _hmac.new(prk, t_prev + b"lf_payload" + bytes([counter]), hashlib.sha256).digest()
            raw_payload += t_i
            t_prev = t_i
            counter += 1
        raw_payload = raw_payload[:payload_length]

    # 将 payload 展开为 ±1 比特序列。
    expected_bit_signs = []
    for byte_val in raw_payload:
        for bit_pos in range(8):
            bit = (byte_val >> bit_pos) & 1
            expected_bit_signs.append(1 if bit == 1 else -1)

    # 扁平化 latent features。
    try:
        flat = _flatten_to_list(latent_features)
    except Exception:
        return {
            "lf_attestation_score": None,
            "status": "failed",
            "n_bits_compared": 0,
            "attestation_digest": attestation_digest,
            "lf_attestation_trace_digest": digests.canonical_sha256({
                "error": "flatten_failed", "d_A": attestation_digest
            }),
        }

    # 取前 block_length 个潜变量（与期望比特数对齐）。
    n_compare = min(block_length, len(flat), len(expected_bit_signs))
    if n_compare <= 0:
        return {
            "lf_attestation_score": None,
            "status": "failed",
            "n_bits_compared": 0,
            "attestation_digest": attestation_digest,
            "lf_attestation_trace_digest": digests.canonical_sha256({
                "error": "insufficient_latents", "d_A": attestation_digest
            }),
        }

    # 计算后验概率（erf-based）。
    posteriors = recover_posteriors_erf(flat[:n_compare], variance)

    # 计算符号一致率：posterior_i * expected_bit_sign_i > 0 则一致。
    agreements = sum(
        1 for p, e in zip(posteriors, expected_bit_signs[:n_compare])
        if p * e > 0
    )
    lf_attestation_score = float(agreements) / float(n_compare)

    # 构造审计摘要（可复算）。
    trace_payload = {
        "channel": "lf",
        "attestation_digest": attestation_digest,
        "variance": variance,
        "block_length": block_length,
        "payload_length": payload_length,
        "n_bits_compared": n_compare,
        "lf_attestation_score": round(lf_attestation_score, 6),
    }
    trace_digest = digests.canonical_sha256(trace_payload)

    return {
        "lf_attestation_score": lf_attestation_score,
        "status": "ok",
        "n_bits_compared": n_compare,
        "attestation_digest": attestation_digest,
        "lf_attestation_trace_digest": trace_digest,
    }



class LowFreqTemplateCodec:
    """
    功能：LF 低频模板编解码 v2 —— keyed pseudogaussian template + additive injection 闭环。

    Implements additive pseudogaussian template embedding and matched-correlation detection.
    Removes latent_space_sign_flipping and belief_propagation; all evidence flows
    through the same coefficient-domain template on both embed and detect sides.

    Args:
        impl_id: Implementation identifier (must be low_freq_template_codec).
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

    def _build_runtime_cfg(
        self,
        cfg: Dict[str, Any],
        plan_digest: str,
        *,
        basis_digest: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        功能：构造绑定 plan/basis 的 LF 运行时配置。

        Build LF runtime config enriched with plan and basis anchors.

        Args:
            cfg: Base runtime config.
            plan_digest: Bound plan digest.
            basis_digest: Optional basis digest.

        Returns:
            Runtime config mapping for shared LF operators.
        """
        runtime_cfg = dict(cfg)
        watermark_cfg = dict(cfg.get("watermark", {})) if isinstance(cfg.get("watermark"), dict) else {}
        watermark_cfg["plan_digest"] = plan_digest
        if isinstance(basis_digest, str) and basis_digest:
            runtime_cfg["lf_basis_digest"] = basis_digest
            watermark_cfg["basis_digest"] = basis_digest
        runtime_cfg["watermark"] = watermark_cfg
        return runtime_cfg

    def embed_apply(
        self,
        cfg: Dict[str, Any],
        latent_features: Any,
        plan_digest: str,
        cfg_digest: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        功能：在潜变量系数域应用加性伪高斯模板注入。

        Apply additive pseudogaussian template injection in latent coefficient domain.
        Replaces sign-flipping. Template direction: codeword ∈ {-1, +1}, scale = variance.

        Args:
            cfg: Configuration dict with watermark.lf.* parameters.
            latent_features: Input latent features.
            plan_digest: Plan digest binding.
            cfg_digest: Optional cfg canonical digest.

        Returns:
            Dict with latent_features_embedded, lf_trace_summary, lf_trace_digest.

        Raises:
            TypeError: If inputs are invalid.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if not isinstance(plan_digest, str) or not plan_digest:
            raise TypeError("plan_digest must be non-empty str")

        lf_cfg = cfg.get("watermark", {}).get("lf", {})
        enabled = lf_cfg.get("enabled", False)
        if not enabled:
            return {
                "status": "absent",
                "latent_features_embedded": latent_features,
                "lf_trace_summary": {"absent": "lf_disabled"},
                "lf_trace_digest": digests.canonical_sha256({"absent": "lf_disabled"}),
            }

        variance = float(lf_cfg.get("variance", 1.5))
        runtime_cfg = self._build_runtime_cfg(cfg, plan_digest)
        template_bundle = channel_lf.derive_lf_template_bundle(runtime_cfg, len(_flatten_to_list(latent_features)))
        ldpc_spec = template_bundle["ldpc_spec"]
        block_length = int(ldpc_spec.get("n", len(template_bundle["codeword_bipolar"])))
        parity_check_digest = ldpc_spec["parity_check_digest"]

        flat_latents = _flatten_to_list(latent_features)
        if len(flat_latents) < block_length:
            return {
                "status": "failed",
                "latent_features_embedded": latent_features,
                "lf_failure_reason": "lf_insufficient_latent_dimension",
                "lf_trace_digest": digests.canonical_sha256({"failure": "insufficient_dim"}),
            }

        coeffs = np.asarray(flat_latents[:block_length], dtype=np.float32)
        encoded_coeffs, encoding_evidence = channel_lf.apply_low_freq_encoding(
            coeffs=coeffs,
            key=0,
            cfg=runtime_cfg,
        )
        embedded_latents = list(flat_latents)
        for index in range(block_length):
            embedded_latents[index] = float(encoded_coeffs[index])

        trace_summary = {
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
            "coding_mode": "pseudogaussian_template_additive",
            "variance": variance,
            "message_length": int(lf_cfg.get("message_length", 64)),
            "block_length": block_length,
            "ecc_sparsity": int(lf_cfg.get("ecc_sparsity", 3)),
            "parity_check_digest": parity_check_digest,
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest,
            "basis_digest": template_bundle.get("basis_digest"),
            "attestation_event_digest": template_bundle.get("attestation_event_digest"),
            "message_source": template_bundle.get("message_source"),
            "pseudogaussian_seed": template_bundle.get("pseudogaussian_seed"),
            "decoder_mode": template_bundle.get("decoder_mode"),
            "bp_iterations": template_bundle.get("bp_iterations"),
            "encoding_evidence": encoding_evidence,
            "mode": "embed",
        }
        lf_trace_digest = digests.canonical_sha256(trace_summary)

        return {
            "status": "ok",
            "latent_features_embedded": embedded_latents,
            "lf_trace_summary": trace_summary,
            "lf_trace_digest": lf_trace_digest,
            "parity_check_digest": parity_check_digest,
        }

    def detect_score(
        self,
        cfg: Dict[str, Any],
        latent_features: Any,
        plan_digest: str,
        cfg_digest: Optional[str] = None,
        lf_basis: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        功能：基于 whitened Pearson correlation 的 LF 水印检测（v2）。

        Detect LF watermark using whitened template correlation in lf_basis coefficient domain.
        Score is defined such that higher values indicate stronger watermark evidence.

        Args:
            cfg: Configuration dict.
            latent_features: Input latent features.
            plan_digest: Plan digest binding (must match embedding side).
            cfg_digest: Optional cfg digest.
            lf_basis: LF subspace basis dict with projection_matrix (required).

        Returns:
            Tuple of (lf_score, lf_detect_trace).
            lf_score in [0, 1]; 0.5 is null hypothesis, >0.5 indicates watermark evidence.

        Raises:
            TypeError: If inputs are invalid.

        Notes:
            detect_variant: correlation_v2.
            higher_is_watermarked: True.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if not isinstance(plan_digest, str) or not plan_digest:
            raise TypeError("plan_digest must be non-empty str")

        lf_cfg = cfg.get("watermark", {}).get("lf", {})
        enabled = lf_cfg.get("enabled", False)
        if not enabled:
            return None, {
                "status": "absent",
                "lf_absent_reason": "lf_disabled",
                "lf_trace_digest": digests.canonical_sha256({"absent": "lf_disabled"}),
            }

        correlation_scale = float(lf_cfg.get("correlation_scale", 10.0))

        if lf_basis is None or not isinstance(lf_basis, dict):
            return None, {
                "status": "failed",
                "lf_failure_reason": "lf_basis_required_but_absent",
                "lf_trace_digest": digests.canonical_sha256({"failure": "lf_basis_absent"}),
            }

        basis_matrix_raw = lf_basis.get("projection_matrix")
        if basis_matrix_raw is None:
            return None, {
                "status": "failed",
                "lf_failure_reason": "lf_basis_projection_matrix_missing",
                "lf_trace_digest": digests.canonical_sha256({"failure": "no_projection_matrix"}),
            }
        basis_matrix_np = np.asarray(basis_matrix_raw, dtype=np.float64)
        basis_rank = int(lf_basis.get("basis_rank", basis_matrix_np.shape[1]))
        basis_digest = None
        if isinstance(lf_basis.get("basis_digest"), str) and lf_basis.get("basis_digest"):
            basis_digest = str(lf_basis.get("basis_digest"))
        else:
            basis_digest = digests.canonical_sha256(
                {
                    "basis_rank": basis_rank,
                    "projection_matrix": basis_matrix_np.tolist(),
                }
            )

        runtime_cfg = self._build_runtime_cfg(cfg, plan_digest, basis_digest=basis_digest)
        template_bundle = channel_lf.derive_lf_template_bundle(runtime_cfg, max(basis_rank, 1))
        ldpc_spec = template_bundle["ldpc_spec"]
        block_length = int(ldpc_spec.get("n", basis_rank))
        parity_check_digest = ldpc_spec["parity_check_digest"]

        if hasattr(latent_features, "detach"):
            latents_flat = latent_features.detach().cpu().numpy().astype(np.float64).reshape(-1)
        else:
            latents_flat = np.asarray(latent_features, dtype=np.float64).reshape(-1)

        if latents_flat.shape[0] != basis_matrix_np.shape[0]:
            latent_proj_spec = lf_basis.get("latent_projection_spec")
            if not isinstance(latent_proj_spec, dict):
                return None, {
                    "status": "failed",
                    "lf_failure_reason": "lf_dim_mismatch_no_projection_spec",
                    "lf_trace_digest": digests.canonical_sha256({"failure": "dim_mismatch_no_spec"}),
                }
            feature_dim = int(latent_proj_spec.get("feature_dim", basis_matrix_np.shape[0]))
            proj_seed = int(latent_proj_spec.get("seed", 0))
            t_idx = int(latent_proj_spec.get("edit_timestep", 0))
            sample_idx = int(latent_proj_spec.get("sample_idx", 0))
            projection_seed = proj_seed + 7919 + t_idx * 131 + sample_idx
            index_rng = np.random.default_rng(projection_seed)
            projection_indices = index_rng.integers(0, max(1, latents_flat.shape[0]), size=feature_dim)
            latents_flat = latents_flat[projection_indices]

        if latents_flat.shape[0] != basis_matrix_np.shape[0]:
            return None, {
                "status": "failed",
                "lf_failure_reason": "lf_dim_mismatch_after_index_selection",
                "lf_trace_digest": digests.canonical_sha256({"failure": "dim_mismatch_after_sel"}),
            }

        coeffs_arr = np.dot(latents_flat, basis_matrix_np)
        template = np.asarray(template_bundle["template"], dtype=np.float64)
        codeword = np.asarray(template_bundle["codeword_bipolar"], dtype=np.float64)

        c = coeffs_arr[:basis_rank]
        eps = 1e-8
        c_mean = float(np.mean(c))
        c_std = float(np.std(c))
        c_whitened = (c - c_mean) / (c_std + eps)
        template_head = template[:basis_rank]
        t_norm = float(np.linalg.norm(template_head))
        t_normalized = template_head / (t_norm + eps)
        raw_corr = float(np.dot(c_whitened, t_normalized))

        variance = float(lf_cfg.get("variance", 1.5))
        llr_values = channel_lf.build_lf_soft_llr(
            coeffs=np.asarray(c, dtype=np.float32),
            template_bundle=template_bundle,
            variance=variance,
        )
        bp_iterations = int(template_bundle["bp_iterations"])
        decode_result = decode_soft_llr(llr_values, ldpc_spec, bp_iterations)
        decoded_bits = decode_result["decoded_bits"]
        expected_codeword = np.asarray(template_bundle["codeword_bipolar"], dtype=np.int32)
        agreement_count = 0
        compare_count = min(len(decoded_bits), int(expected_codeword.shape[0]))
        for index in range(compare_count):
            if int(decoded_bits[index]) == int(expected_codeword[index]):
                agreement_count += 1
        codeword_agreement = float(agreement_count / compare_count) if compare_count > 0 else 0.0
        correlation_score = 1.0 / (1.0 + math.exp(-correlation_scale * raw_corr))
        lf_score = float(round(0.5 * correlation_score + 0.5 * codeword_agreement, 8))

        trace = {
            "status": "ok",
            "lf_score": lf_score,
            "raw_correlation": raw_corr,
            "correlation_score": float(correlation_score),
            "codeword_agreement": round(codeword_agreement, 8),
            "c_mean": c_mean,
            "c_std": c_std,
            "basis_rank": basis_rank,
            "correlation_dim": int(len(c)),
            "parity_check_digest": parity_check_digest,
            "correlation_scale": correlation_scale,
            "detect_variant": "correlation_v2",
            "higher_is_watermarked": True,
            "decoder_mode": template_bundle.get("decoder_mode"),
            "soft_decode_called": True,
            "bp_converged": bool(decode_result.get("bp_converged")),
            "bp_iteration_count": int(decode_result.get("bp_iteration_count", 0)),
            "syndrome_weight": int(decode_result.get("syndrome_weight", 0)),
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest,
            "basis_digest": basis_digest,
            "attestation_event_digest": template_bundle.get("attestation_event_digest"),
            "message_source": template_bundle.get("message_source"),
            "pseudogaussian_seed": template_bundle.get("pseudogaussian_seed"),
        }
        trace["lf_trace_digest"] = digests.canonical_sha256(trace)
        return lf_score, trace
