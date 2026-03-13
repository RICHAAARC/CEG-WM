"""
低频子空间编码通道实现

功能说明：
- 在 LF（低频）子空间实现隐蔽的水印编码。
- 支持投影、编码、恢复的完整张量流程。
- 生成可复算的 lf_trace_digest 与固定精度指标。
- 严格区分 absent / failed / mismatch 语义。

Module type: Core innovation module
"""

from __future__ import annotations

import math
import hmac
import random as _random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from main.core import digests
from .ldpc_codec import build_ldpc_spec, encode_message_bits


LF_CHANNEL_IMPL_ID = "low_freq_template_codec"
LF_CHANNEL_VERSION = "v2"
LF_TRACE_VERSION = "v2"


def _stable_vector_matrix_dot(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """功能：使用非 BLAS 路径计算向量与矩阵乘法。"""
    vector_fp32 = np.asarray(vector, dtype=np.float32)
    matrix_fp32 = np.asarray(matrix, dtype=np.float32)
    if vector_fp32.ndim != 1:
        raise ValueError("vector must be rank-1")
    if matrix_fp32.ndim != 2:
        raise ValueError("matrix must be rank-2")
    if vector_fp32.shape[0] != matrix_fp32.shape[0]:
        raise ValueError(
            f"vector dimension {vector_fp32.shape[0]} != matrix dimension {matrix_fp32.shape[0]}"
        )
    return np.einsum("i,ij->j", vector_fp32, matrix_fp32, dtype=np.float32)


def _stable_vector_dot(left: np.ndarray, right: np.ndarray) -> float:
    """功能：使用非 BLAS 路径计算两个向量的点积。"""
    left_fp32 = np.asarray(left, dtype=np.float32)
    right_fp32 = np.asarray(right, dtype=np.float32)
    if left_fp32.ndim != 1 or right_fp32.ndim != 1:
        raise ValueError("left and right must be rank-1")
    if left_fp32.shape[0] != right_fp32.shape[0]:
        raise ValueError(
            f"left dimension {left_fp32.shape[0]} != right dimension {right_fp32.shape[0]}"
        )
    return float(np.einsum("i,i->", left_fp32, right_fp32, dtype=np.float32))


def _resolve_lf_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """功能：规范化 LF 配置视图。"""
    watermark_cfg = cfg.get("watermark") if isinstance(cfg.get("watermark"), dict) else {}
    lf_cfg = watermark_cfg.get("lf") if isinstance(watermark_cfg.get("lf"), dict) else {}
    return dict(lf_cfg)


def _resolve_plan_digest(cfg: Dict[str, Any]) -> str:
    """功能：统一解析 LF 绑定的 plan_digest。"""
    watermark_cfg = cfg.get("watermark") if isinstance(cfg.get("watermark"), dict) else {}
    candidate = cfg.get("lf_plan_digest") or watermark_cfg.get("plan_digest") or cfg.get("plan_digest")
    return candidate if isinstance(candidate, str) else ""


def _resolve_basis_digest(cfg: Dict[str, Any]) -> Optional[str]:
    """功能：统一解析 basis_digest。"""
    candidate = cfg.get("lf_basis_digest") or cfg.get("basis_digest")
    if isinstance(candidate, str) and candidate:
        return candidate
    watermark_cfg = cfg.get("watermark") if isinstance(cfg.get("watermark"), dict) else {}
    candidate = watermark_cfg.get("basis_digest")
    return candidate if isinstance(candidate, str) and candidate else None


def _resolve_attestation_event_digest(cfg: Dict[str, Any]) -> Optional[str]:
    """功能：统一解析 attestation 事件摘要。"""
    direct_candidate = (
        cfg.get("lf_attestation_event_digest")
        or cfg.get("attestation_event_digest")
        or cfg.get("attestation_digest")
    )
    if isinstance(direct_candidate, str) and direct_candidate:
        return direct_candidate
    watermark_cfg = cfg.get("watermark") if isinstance(cfg.get("watermark"), dict) else {}
    watermark_candidate = watermark_cfg.get("attestation_event_digest") or watermark_cfg.get("attestation_digest")
    if isinstance(watermark_candidate, str) and watermark_candidate:
        return watermark_candidate
    attestation_cfg = cfg.get("attestation") if isinstance(cfg.get("attestation"), dict) else {}
    candidate = attestation_cfg.get("event_digest") or attestation_cfg.get("attestation_digest")
    return candidate if isinstance(candidate, str) and candidate else None


def _resolve_lf_attestation_key(cfg: Dict[str, Any]) -> Optional[str]:
    """功能：统一解析 LF attestation 子密钥。"""
    candidate = cfg.get("lf_attestation_key") or cfg.get("k_lf")
    if isinstance(candidate, str) and candidate:
        return candidate
    attestation_runtime = cfg.get("attestation_runtime") if isinstance(cfg.get("attestation_runtime"), dict) else {}
    candidate = attestation_runtime.get("k_lf")
    if isinstance(candidate, str) and candidate:
        return candidate
    watermark_cfg = cfg.get("watermark") if isinstance(cfg.get("watermark"), dict) else {}
    candidate = watermark_cfg.get("k_lf")
    return candidate if isinstance(candidate, str) and candidate else None


def _resolve_decoder_mode(cfg: Dict[str, Any]) -> str:
    """功能：解析 LF decoder mode。"""
    lf_cfg = _resolve_lf_cfg(cfg)
    decoder = lf_cfg.get("decoder", "bp_soft_llr")
    if not isinstance(decoder, str) or not decoder:
        return "bp_soft_llr"
    return decoder


def _resolve_bp_iterations(cfg: Dict[str, Any]) -> int:
    """功能：解析 BP 迭代次数。"""
    lf_cfg = _resolve_lf_cfg(cfg)
    value = lf_cfg.get("bp_iterations", 50)
    if not isinstance(value, int):
        return 50
    return max(1, value)


def _payload_bytes_to_bipolar_bits(payload: bytes, required_length: int) -> List[int]:
    """功能：将 payload bytes 展开为双极性比特。"""
    bits: List[int] = []
    for byte_value in payload:
        for bit_index in range(8):
            bit = (byte_value >> bit_index) & 1
            bits.append(1 if bit == 1 else -1)
            if len(bits) >= required_length:
                return bits
    while len(bits) < required_length:
        bits.append(1)
    return bits


def derive_lf_template_bundle(cfg: Dict[str, Any], n: int) -> Dict[str, Any]:
    """
    功能：统一派生 LF 模板闭环所需的码字、幅度与审计锚点。

    Derive the formal LF template bundle shared by embed and detect.

    Args:
        cfg: Runtime configuration mapping.
        n: Number of coefficients required.

    Returns:
        Template bundle containing codeword, amplitudes, LDPC spec, and audit anchors.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be positive int")

    lf_cfg = _resolve_lf_cfg(cfg)
    plan_digest = _resolve_plan_digest(cfg)
    message_length = int(lf_cfg.get("message_length", 64))
    ecc_sparsity = int(lf_cfg.get("ecc_sparsity", 3))
    basis_digest = _resolve_basis_digest(cfg)
    attestation_event_digest = _resolve_attestation_event_digest(cfg)
    k_lf = _resolve_lf_attestation_key(cfg)

    message_source = "plan_digest"
    if attestation_event_digest and k_lf:
        from main.watermarking.provenance.key_derivation import compute_lf_attestation_payload

        payload_length = max(1, int(math.ceil(float(message_length) / 8.0)))
        payload = compute_lf_attestation_payload(k_lf, attestation_event_digest, payload_length=payload_length)
        message_bits = _payload_bytes_to_bipolar_bits(payload, message_length)
        message_source = "attestation_event_digest"
    else:
        seed_material = {
            "plan_digest": plan_digest,
            "tag": "lf_message",
            "basis_digest": basis_digest,
            "attestation_event_digest": attestation_event_digest,
        }
        seed = int(digests.canonical_sha256(seed_material)[:16], 16)
        rng = _random.Random(seed)
        message_bits = [1 if rng.random() < 0.5 else -1 for _ in range(message_length)]

    ldpc_seed_key = digests.canonical_sha256(
        {
            "plan_digest": plan_digest,
            "basis_digest": basis_digest,
            "attestation_event_digest": attestation_event_digest,
            "message_length": message_length,
            "ecc_sparsity": ecc_sparsity,
            "message_source": message_source,
            "channel": "lf",
        }
    )
    ldpc_spec = build_ldpc_spec(
        message_length=message_length,
        ecc_sparsity=ecc_sparsity,
        seed_key=ldpc_seed_key,
    )
    code_bits = encode_message_bits(message_bits, ldpc_spec)
    codeword = np.array(code_bits[:n] + [1] * max(0, n - len(code_bits)), dtype=np.float32)

    template_seed_material = {
        "channel": "lf",
        "plan_digest": plan_digest,
        "basis_digest": basis_digest,
        "attestation_event_digest": attestation_event_digest,
        "message_source": message_source,
        "parity_check_digest": ldpc_spec.get("parity_check_digest"),
    }
    pseudogaussian_seed = int(digests.canonical_sha256(template_seed_material)[:16], 16)
    rng_np = np.random.default_rng(pseudogaussian_seed)
    amplitudes = np.abs(rng_np.standard_normal(n)).astype(np.float32)
    signed_template = codeword * amplitudes

    return {
        "message_bits": message_bits,
        "message_source": message_source,
        "codeword_bipolar": codeword,
        "ldpc_spec": ldpc_spec,
        "parity_check_digest": ldpc_spec.get("parity_check_digest"),
        "pseudogaussian_seed": pseudogaussian_seed,
        "amplitudes": amplitudes,
        "template": signed_template,
        "plan_digest": plan_digest,
        "basis_digest": basis_digest,
        "attestation_event_digest": attestation_event_digest,
        "decoder_mode": _resolve_decoder_mode(cfg),
        "bp_iterations": _resolve_bp_iterations(cfg),
    }


def build_lf_soft_llr(coeffs: np.ndarray, template_bundle: Dict[str, Any], variance: float) -> List[float]:
    """
    功能：根据模板方向与幅度构造 LF 软判决 LLR。

    Build LF soft-decision LLR values from coefficient-domain observations.

    Args:
        coeffs: Observed LF coefficients.
        template_bundle: Shared LF template bundle.
        variance: LF variance used during embedding.

    Returns:
        LLR list aligned with LDPC block length.
    """
    if not isinstance(coeffs, np.ndarray):
        raise TypeError("coeffs must be np.ndarray")
    if not isinstance(template_bundle, dict):
        raise TypeError("template_bundle must be dict")
    if not isinstance(variance, (int, float)) or variance <= 0:
        raise ValueError("variance must be positive number")

    ldpc_spec = template_bundle.get("ldpc_spec") if isinstance(template_bundle.get("ldpc_spec"), dict) else {}
    block_length = int(ldpc_spec.get("n", coeffs.shape[0]))
    template = np.asarray(template_bundle.get("template", []), dtype=np.float64)
    llr_values: List[float] = []
    effective_count = min(int(coeffs.shape[0]), int(template.shape[0]), block_length)
    for index in range(effective_count):
        observed = float(coeffs[index])
        template_value = float(template[index])
        amplitude = max(abs(template_value), 1e-6)
        sign = 1.0 if template_value >= 0.0 else -1.0
        llr_values.append(float((observed * sign) / (float(variance) * amplitude)))
    while len(llr_values) < block_length:
        llr_values.append(0.0)
    return llr_values


def _has_plan_digest(cfg: Dict[str, Any]) -> bool:
    """判断 cfg 中是否存在有效 plan_digest。"""
    pd = cfg.get("lf_plan_digest") or cfg.get("watermark", {}).get("plan_digest", "")
    return isinstance(pd, str) and bool(pd)


def _get_ldpc_codeword_bits(cfg: Dict[str, Any], n: int) -> List[int]:
    """
    功能：从 plan_digest 派生 LDPC 码字（±1 列表）。

    Derive deterministic LDPC codeword from plan_digest.
    Matches the seed derivation used in LowFreqTemplateCodec.detect_score().

    Args:
        cfg: Injection config containing lf_plan_digest / lf_message_length / lf_ecc_sparsity.
        n: Number of codeword bits required (= block_length = 96).

    Returns:
        List of ±1 integers of length n. Falls back to ones if plan_digest absent.
    """
    template_bundle = derive_lf_template_bundle(cfg, n)
    return [int(v) for v in np.asarray(template_bundle["codeword_bipolar"], dtype=np.int32).tolist()]


def _derive_ldpc_codeword_from_cfg(cfg: Dict[str, Any], n: int, device: Any) -> Any:
    """
    功能：生成 torch 格式的 LDPC ±1 码字张量。

    Generate torch LDPC codeword tensor for embed-side LF encoding.

    Args:
        cfg: Injection config.
        n: Number of codeword bits required.
        device: Torch device.

    Returns:
        Torch float32 tensor of ±1, shape (n,).
    """
    import torch
    bits = _get_ldpc_codeword_bits(cfg, n)
    return torch.tensor(bits, dtype=torch.float32, device=device)


def _derive_ldpc_codeword_from_cfg_np(cfg: Dict[str, Any], n: int) -> np.ndarray:
    """
    功能：生成 numpy 格式的 LDPC ±1 码字数组（numpy 路径）。

    Generate numpy LDPC codeword array for embed-side LF encoding.

    Args:
        cfg: Injection config.
        n: Number of codeword bits required.

    Returns:
        Numpy float32 array of ±1, shape (n,).
    """
    bits = _get_ldpc_codeword_bits(cfg, n)
    return np.array(bits, dtype=np.float32)


def compute_lf_basis_projection_torch(latents: Any, basis: Dict[str, Any]) -> Any:
    """
    功能：使用 torch 在 LF 基向量上计算投影系数。

    Compute LF projection coefficients using torch tensor operations.

    Args:
        latents: Torch tensor latents.
        basis: Basis mapping with projection_matrix.

    Returns:
        Torch tensor coefficients in LF subspace.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If basis shape mismatches latent dimension.
    """
    import torch

    if not torch.is_tensor(latents):
        raise TypeError("latents must be torch.Tensor")
    if not isinstance(basis, dict):
        raise TypeError("basis must be dict mapping")

    basis_matrix = basis.get("projection_matrix")
    if basis_matrix is None:
        raise ValueError("basis must contain projection_matrix")

    latents_flat = latents.reshape(-1).to(dtype=torch.float32)
    basis_matrix_t = torch.as_tensor(
        basis_matrix,
        dtype=torch.float32,
        device=latents_flat.device
    )
    if basis_matrix_t.ndim != 2:
        raise ValueError("projection_matrix must be rank-2")
    if latents_flat.shape[0] != basis_matrix_t.shape[0]:
        # 维度不匹配：使用 trajectory_feature_spec（TFSW 主路径）将 latent 降至 feature_dim。
        # 公式：projection_seed = planner_seed + edit_timestep（不含 sample_idx）。
        tfs = basis.get("trajectory_feature_spec")
        if isinstance(tfs, dict) and tfs.get("feature_operator") == "masked_normalized_random_projection":
            from .subspace.trajectory_feature_space import extract_trajectory_feature_torch
            latents_flat = extract_trajectory_feature_torch(latents, tfs)
        else:
            # backward-compat：latent_projection_spec 随机索引路径（将被废弃）。
            latent_proj_spec = basis.get("latent_projection_spec")
            if not isinstance(latent_proj_spec, dict) or latent_proj_spec.get("method") != "random_index_selection":
                raise ValueError(
                    f"latents dimension {latents_flat.shape[0]} != basis dimension {basis_matrix_t.shape[0]}: "
                    "neither trajectory_feature_spec nor latent_projection_spec found in basis"
                )
            import numpy as _np
            feature_dim = int(latent_proj_spec.get("feature_dim", basis_matrix_t.shape[0]))
            seed = int(latent_proj_spec.get("seed", 0))
            t_idx = int(latent_proj_spec.get("edit_timestep", 0))
            sample_idx = int(latent_proj_spec.get("sample_idx", 0))
            projection_seed = seed + 7919 + t_idx * 131 + sample_idx
            rng = _np.random.default_rng(projection_seed)
            indices = rng.integers(0, max(1, int(latents_flat.shape[0])), size=feature_dim)
            indices_t = torch.as_tensor(indices, dtype=torch.long, device=latents_flat.device)
            latents_flat = latents_flat[indices_t]
    return torch.matmul(latents_flat, basis_matrix_t)


def apply_low_freq_encoding_torch(coeffs: Any, key: int, cfg: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    功能：使用 torch 在 LF 系数上施加确定性编码。

    Apply deterministic LF encoding on torch coefficients.

    Args:
        coeffs: Torch coefficients tensor.
        key: Encoding seed.
        cfg: Config mapping.

    Returns:
        Tuple of (encoded_coeffs, encoding_evidence).

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If cfg values are invalid.
    """
    import torch

    if not torch.is_tensor(coeffs):
        raise TypeError("coeffs must be torch.Tensor")
    if coeffs.ndim != 1:
        raise TypeError("coeffs must be 1-D torch.Tensor")
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    lf_cfg = _resolve_lf_cfg(cfg)
    strength = cfg.get("lf_strength")
    if not isinstance(strength, (int, float)):
        strength = lf_cfg.get("strength", lf_cfg.get("variance", 1.5))
    if not isinstance(strength, (int, float)) or strength < 0:
        raise ValueError(f"lf_strength must be non-negative, got {strength}")

    n = int(coeffs.shape[0])
    template_bundle = derive_lf_template_bundle(cfg, n)
    watermark_pattern = torch.as_tensor(
        template_bundle["template"],
        dtype=torch.float32,
        device=coeffs.device,
    )

    coeffs_fp32 = coeffs.to(dtype=torch.float32)
    encoded_coeffs = coeffs_fp32 + float(strength) * watermark_pattern

    encoding_evidence = {
        "strength_applied": float(strength),
        "pattern_seed": int(template_bundle["pseudogaussian_seed"]),
        "runtime_step_key": int(key),
        "codeword_source": str(template_bundle["message_source"]),
        "parity_check_digest": template_bundle["parity_check_digest"],
        "basis_digest": template_bundle["basis_digest"],
        "attestation_event_digest": template_bundle["attestation_event_digest"],
        "decoder_mode": template_bundle["decoder_mode"],
        "bp_iterations": int(template_bundle["bp_iterations"]),
        "coeffs_before_norm": float(torch.linalg.vector_norm(coeffs_fp32).item()),
        "coeffs_after_norm": float(torch.linalg.vector_norm(encoded_coeffs).item()),
        "pattern_norm": float(torch.linalg.vector_norm(watermark_pattern).item()),
        "coeffs_count": n,
    }
    return encoded_coeffs, encoding_evidence


def reconstruct_from_lf_coeffs_torch(
    coeffs: Any,
    basis: Dict[str, Any],
    latents_shape: Tuple[int, ...],
    dtype: Any
) -> Any:
    """
    功能：使用 torch 从 LF 系数重建张量。

    Reconstruct latent tensor from LF coefficients using torch.

    Args:
        coeffs: Torch coefficients tensor.
        basis: Basis mapping with projection_matrix.
        latents_shape: Target shape.
        dtype: Target dtype.

    Returns:
        Torch tensor with requested shape and dtype.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If basis is invalid.
    """
    import torch

    if not torch.is_tensor(coeffs):
        raise TypeError("coeffs must be torch.Tensor")
    if not isinstance(basis, dict):
        raise TypeError("basis must be dict")
    if not isinstance(latents_shape, tuple) or len(latents_shape) == 0:
        raise TypeError("latents_shape must be non-empty tuple")

    basis_matrix = basis.get("projection_matrix")
    if basis_matrix is None:
        raise ValueError("basis must contain projection_matrix")
    basis_matrix_t = torch.as_tensor(
        basis_matrix,
        dtype=torch.float32,
        device=coeffs.device
    )
    if basis_matrix_t.ndim != 2:
        raise ValueError("projection_matrix must be rank-2")

    # latents_sub: basis B (basis_dim, rank) @ coeffs (rank,) = 特征空间向量 (basis_dim,)
    latents_sub = torch.matmul(basis_matrix_t, coeffs.to(dtype=torch.float32))
    basis_dim = int(basis_matrix_t.shape[0])
    full_dim = 1
    for s in latents_shape:
        full_dim *= s

    if basis_dim < full_dim:
        # basis_dim < full_dim：需把特征向量映射回完整 latent 维度。
        # 优先使用 trajectory_feature_spec（TFSW pullback）。
        tfs = basis.get("trajectory_feature_spec")
        if isinstance(tfs, dict) and tfs.get("feature_operator") == "masked_normalized_random_projection":
            from .subspace.trajectory_feature_space import pullback_feature_delta_torch
            reconstructed = pullback_feature_delta_torch(latents_sub, tfs, latents_shape)
        else:
            # backward-compat：latent_projection_spec scatter-back 路径。
            latent_proj_spec = basis.get("latent_projection_spec")
            if not isinstance(latent_proj_spec, dict) or latent_proj_spec.get("method") != "random_index_selection":
                raise ValueError(
                    f"reconstruction dim mismatch: basis_dim={basis_dim} < full_dim={full_dim}, "
                    "but neither trajectory_feature_spec nor latent_projection_spec found in basis"
                )
            import numpy as _np
            feature_dim = int(latent_proj_spec.get("feature_dim", basis_dim))
            seed = int(latent_proj_spec.get("seed", 0))
            t_idx = int(latent_proj_spec.get("edit_timestep", 0))
            sample_idx = int(latent_proj_spec.get("sample_idx", 0))
            projection_seed = seed + 7919 + t_idx * 131 + sample_idx
            rng = _np.random.default_rng(projection_seed)
            indices = rng.integers(0, max(1, full_dim), size=feature_dim)
            indices_t = torch.as_tensor(indices, dtype=torch.long, device=coeffs.device)
            output = torch.zeros(full_dim, dtype=torch.float32, device=coeffs.device)
            output[indices_t] = latents_sub
            reconstructed = output.reshape(latents_shape)
    else:
        reconstructed = latents_sub.reshape(latents_shape)
    return reconstructed.to(dtype=dtype)


def compute_lf_basis_projection(latents: np.ndarray | Any, basis: Dict[str, Any]) -> np.ndarray:
    """
    功能：计算 latent 在 LF 基向量上的投影系数。
    
    Compute projection coefficients of latents onto LF basis vectors.
    Projects high-dim latents to low-freq subspace using basis matrix.
    
    Args:
        latents: Input latent array (typically Gaussian, may be torch.Tensor).
        basis: Basis mapping with shape and projection matrix info.
    
    Returns:
        Projection coefficients (np.ndarray) in LF subspace.
    
    Raises:
        TypeError: If inputs types are invalid.
        ValueError: If basis or latents shapes are incompatible.
    """
    # 支持 torch.Tensor 转换。
    if hasattr(latents, "cpu"):
        latents_np = latents.cpu().numpy()
    else:
        latents_np = np.asarray(latents, dtype=np.float32)
    
    if not isinstance(basis, dict):
        raise TypeError("basis must be dict mapping")
    
    # 从 basis 中提取投影方向矩阵。
    basis_matrix = basis.get("projection_matrix")  # shape: (latent_dim, lf_rank)
    if basis_matrix is None:
        raise ValueError("basis must contain projection_matrix")
    
    if hasattr(basis_matrix, "cpu"):
        basis_matrix_np = basis_matrix.cpu().numpy()
    else:
        basis_matrix_np = np.asarray(basis_matrix, dtype=np.float32)
    
    # 扁平化 latents（若多维）。
    latents_flat = latents_np.reshape(-1)
    
    # 投影计算：coeffs = latents_flat @ basis_matrix。
    if latents_flat.shape[0] != basis_matrix_np.shape[0]:
        raise ValueError(
            f"latents dimension {latents_flat.shape[0]} != basis dimension {basis_matrix_np.shape[0]}"
        )
    
    coeffs = _stable_vector_matrix_dot(latents_flat, basis_matrix_np)  # shape: (lf_rank,)
    return coeffs.astype(np.float32)


def apply_low_freq_encoding(
    coeffs: np.ndarray,
    key: int,
    cfg: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    功能：在 LF 投影系数上施加水印编码（伪高斯采样）。
    
    Apply pseudogaussian watermark encoding to LF coefficients.
    Modifies coefficients using deterministic pseudo-random pattern.
    
    Args:
        coeffs: Projected coefficients (LF subspace, 1D array).
        key: Random seed for encoding pattern generation.
        cfg: Config dict with encoding parameters (strength, etc).
    
    Returns:
        Tuple of (encoded_coeffs, encoding_evidence).
        encoding_evidence contains metrics and trace components.
    
    Raises:
        TypeError: If coeffs array type is invalid.
        ValueError: If cfg parameters out of bounds.
    """
    if not isinstance(coeffs, np.ndarray):
        raise TypeError("coeffs must be np.ndarray")
    
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    
    # 获取编码强度参数。
    lf_cfg = _resolve_lf_cfg(cfg)
    strength = cfg.get("lf_strength")
    if not isinstance(strength, (int, float)):
        strength = lf_cfg.get("strength", lf_cfg.get("variance", 1.5))
    if not isinstance(strength, (int, float)) or strength < 0:
        raise ValueError(f"lf_strength must be non-negative, got {strength}")
    
    n = int(coeffs.shape[0])
    template_bundle = derive_lf_template_bundle(cfg, n)
    watermark_pattern = np.asarray(template_bundle["template"], dtype=np.float32)

    # 以 strength 系数应用水印。
    encoded_coeffs = coeffs + strength * watermark_pattern

    # 构造编码证据（摘要不含原始张量）。
    encoding_evidence = {
        "strength_applied": float(strength),
        "pattern_seed": int(template_bundle["pseudogaussian_seed"]),
        "runtime_step_key": int(key),
        "codeword_source": str(template_bundle["message_source"]),
        "parity_check_digest": template_bundle["parity_check_digest"],
        "basis_digest": template_bundle["basis_digest"],
        "attestation_event_digest": template_bundle["attestation_event_digest"],
        "decoder_mode": template_bundle["decoder_mode"],
        "bp_iterations": int(template_bundle["bp_iterations"]),
        "coeffs_before_norm": float(np.linalg.norm(coeffs)),
        "coeffs_after_norm": float(np.linalg.norm(encoded_coeffs)),
        "pattern_norm": float(np.linalg.norm(watermark_pattern)),
        "coeffs_count": n,
    }

    return encoded_coeffs.astype(np.float32), encoding_evidence


def reconstruct_from_lf_coeffs(
    coeffs: np.ndarray,
    basis: Dict[str, Any],
    latents_shape: Tuple[int, ...],
    is_torch: bool = False
) -> np.ndarray | Any:
    """
    功能：从 LF 投影系数恢复张量（投影反演）。
    
    Reconstruct latents from LF coefficients via basis inverse projection.
    Restores original space from projected subspace coefficients.
    
    Args:
        coeffs: Projected coefficients (1D array, LF subspace).
        basis: Basis mapping with inverse projection info.
        latents_shape: Original latents shape for reshaping.
        is_torch: If True, return torch.Tensor; else np.ndarray.
    
    Returns:
        Reconstructed latents (shape matches input).
    
    Raises:
        TypeError: If inputs types are invalid.
        ValueError: If basis or shape incompatible.
    """
    if not isinstance(coeffs, np.ndarray):
        raise TypeError("coeffs must be np.ndarray")
    
    if not isinstance(basis, dict):
        raise TypeError("basis must be dict")
    
    # 提取投影矩阵用于反演。
    basis_matrix = basis.get("projection_matrix")
    if basis_matrix is None:
        raise ValueError("basis must contain projection_matrix")
    
    if hasattr(basis_matrix, "cpu"):
        basis_matrix_np = basis_matrix.cpu().numpy()
    else:
        basis_matrix_np = np.asarray(basis_matrix, dtype=np.float32)
    
    # 反演：latents_reconstructed = coeffs @ basis_matrix.T。
    latents_flat = _stable_vector_matrix_dot(coeffs, basis_matrix_np.T)  # shape: (latent_dim,)
    
    # 恢复原始形状。
    latents_reconstructed = latents_flat.reshape(latents_shape).astype(np.float32)
    
    if is_torch:
        import torch
        return torch.from_numpy(latents_reconstructed)
    else:
        return latents_reconstructed


def extract_lf_score(
    latents: np.ndarray | Any,
    basis: Dict[str, Any],
    expected_pattern_seed: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None
) -> float:
    """
    功能：从 latent 中提取 LF 检测分数（相关性统计）。
    
    Extract LF detection score via projection correlation analysis.
    Computes likelihood of watermark presence in LF subspace.
    
    Args:
        latents: Input latent array for detection.
        basis: LF basis mapping (same as used during encoding).
        expected_pattern_seed: Expected seed for pattern (for verification).
        cfg: Optional config with detection parameters.
    
    Returns:
        LF score (float, non-negative; higher indicates stronger watermark evidence).
    
    Raises:
        TypeError: If inputs types are invalid.
    """
    if not isinstance(basis, dict):
        raise TypeError("basis must be dict")
    
    # 支持 torch.Tensor 转换。
    if hasattr(latents, "cpu"):
        latents_np = latents.cpu().numpy()
    else:
        latents_np = np.asarray(latents, dtype=np.float32)
    
    # 投影到 LF 子空间。
    coeffs = compute_lf_basis_projection(latents_np, basis)
    
    # 在检测侧，若已知 expected_pattern_seed，重建参考模式。
    if expected_pattern_seed is not None:
        np.random.seed(expected_pattern_seed)
        pseudogaussian_factors = np.random.randn(coeffs.shape[0]).astype(np.float32)
        codeword = 2 * (np.random.binomial(1, 0.5, coeffs.shape[0]) - 0.5)
        reference_pattern = codeword * np.abs(pseudogaussian_factors)
        
        # 计算相关系数。
        normalization = np.linalg.norm(coeffs) * np.linalg.norm(reference_pattern)
        if normalization > 1e-8:
            score = _stable_vector_dot(coeffs, reference_pattern) / normalization
            score = max(0.0, score)  # 约束非负。
        else:
            score = 0.0
    else:
        # 若无期望模式，计算 coeffs 的统计量。
        coeffs_norm = np.linalg.norm(coeffs)
        score = float(coeffs_norm) if coeffs_norm > 0 else 0.0
    
    return float(score)


def generate_lf_evidence_digest(
    trace_components: list,
    params_digest: str
) -> str:
    """
    功能：生成 lf_trace_digest（可复算摘要）。
    
    Generate LF trace digest from ordered step evidence and parameters.
    Uses canonical SHA256 for reproducibility.
    
    Args:
        trace_components: Ordered list of step evidence dicts.
        params_digest: Parameter digest (already computed).
    
    Returns:
        LF trace digest (64-char hex sha256).
    
    Raises:
        TypeError: If inputs types invalid.
    """
    if not isinstance(trace_components, list):
        raise TypeError("trace_components must be list")
    
    if not isinstance(params_digest, str):
        raise TypeError("params_digest must be str")
    
    # 构造可复算的摘要输入：参数摘要 + 顺序敏感的 trace list。
    trace_payload = {
        "params_digest": params_digest,
        "trace_count": len(trace_components),
        "traces": trace_components
    }
    
    # 使用规范化的摘要入口。
    digest = digests.canonical_sha256(trace_payload)
    
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"Computed digest invalid: {digest}")
    
    return digest


def build_lf_embed_evidence(
    latents_before: np.ndarray,
    latents_after: np.ndarray,
    trace_components: list,
    encoding_evidence: Dict[str, Any],
    cfg: Dict[str, Any],
    plan_digest: Optional[str] = None
) -> Dict[str, Any]:
    """
    功能：构造 LF 嵌入侧证据（完整审计载体）。
    
    Build LF evidence structure for embed-side audit.
    Includes status, metrics, digest, and failure semantics.
    
    Args:
        latents_before: Latents before LF encoding.
        latents_after: Latents after LF encoding.
        trace_components: Ordered list of step traces.
        encoding_evidence: Evidence from apply_low_freq_encoding().
        cfg: Configuration mapping.
        plan_digest: Optional plan digest for binding.
    
    Returns:
        LF evidence dict with status, metrics, digest, etc.
    
    Raises:
        TypeError: If inputs types invalid.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    
    lf_enabled = cfg.get("lf_enabled", True)
    if not lf_enabled:
        return {
            "status": "absent",
            "absent_reason": "lf_disabled_by_config",
            "lf_score": None,
            "lf_trace_digest": None,
            "lf_metrics": None,
            "lf_params_digest": None
        }
    
    # 构造参数摘要（固定字段集）。
    lf_params = {
        "impl_id": LF_CHANNEL_IMPL_ID,
        "impl_version": LF_CHANNEL_VERSION,
        "lf_strength": cfg.get("lf_strength", 1.5),
        "lf_enabled": True
    }
    lf_params_digest = digests.canonical_sha256(lf_params)
    
    # 生成 lf_trace_digest。
    lf_trace_digest = generate_lf_evidence_digest(trace_components, lf_params_digest)
    
    # 计算固定精度指标（fp32 精度）。
    latents_before_np = np.asarray(latents_before, dtype=np.float32)
    latents_after_np = np.asarray(latents_after, dtype=np.float32)
    
    lf_metrics = {
        "before_norm": float(np.linalg.norm(latents_before_np)),
        "after_norm": float(np.linalg.norm(latents_after_np)),
        "delta_norm": float(np.linalg.norm(latents_after_np - latents_before_np)),
        "trace_count": len(trace_components),
        "impl_version": LF_TRACE_VERSION
    }
    if isinstance(encoding_evidence, dict):
        lf_metrics.update(encoding_evidence)
    
    evidence = {
        "status": "ok",
        "lf_score": None,  # embed 端不计算 score，仅生成证据。
        "lf_trace_digest": lf_trace_digest,
        "lf_metrics": lf_metrics,
        "lf_params_digest": lf_params_digest,
        "plan_digest": plan_digest if plan_digest else None
    }
    
    return evidence
