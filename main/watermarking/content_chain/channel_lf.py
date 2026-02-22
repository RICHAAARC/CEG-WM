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
from typing import Any, Dict, Optional, Tuple

import numpy as np

from main.core import digests


LF_CHANNEL_IMPL_ID = "low_freq_coder_v1"
LF_CHANNEL_VERSION = "v1"
LF_TRACE_VERSION = "v1"


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
        raise ValueError(
            f"latents dimension {latents_flat.shape[0]} != basis dimension {basis_matrix_t.shape[0]}"
        )
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

    strength = cfg.get("lf_strength", 1.5)
    if not isinstance(strength, (int, float)) or strength < 0:
        raise ValueError(f"lf_strength must be non-negative, got {strength}")

    generator = torch.Generator(device=coeffs.device)
    generator.manual_seed(int(key))
    pseudogaussian_factors = torch.randn(
        coeffs.shape[0],
        generator=generator,
        device=coeffs.device,
        dtype=torch.float32
    )
    code_bits = torch.randint(
        low=0,
        high=2,
        size=(coeffs.shape[0],),
        generator=generator,
        device=coeffs.device
    )
    codeword = code_bits.to(dtype=torch.float32) * 2.0 - 1.0
    watermark_pattern = codeword * torch.abs(pseudogaussian_factors)

    coeffs_fp32 = coeffs.to(dtype=torch.float32)
    encoded_coeffs = coeffs_fp32 + float(strength) * watermark_pattern

    encoding_evidence = {
        "strength_applied": float(strength),
        "pattern_seed": int(key),
        "coeffs_before_norm": float(torch.linalg.vector_norm(coeffs_fp32).item()),
        "coeffs_after_norm": float(torch.linalg.vector_norm(encoded_coeffs).item()),
        "pattern_norm": float(torch.linalg.vector_norm(watermark_pattern).item()),
        "coeffs_count": int(coeffs.shape[0])
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

    latents_flat = torch.matmul(basis_matrix_t, coeffs.to(dtype=torch.float32))
    reconstructed = latents_flat.reshape(latents_shape)
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
    
    coeffs = np.dot(latents_flat, basis_matrix_np)  # shape: (lf_rank,)
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
    strength = cfg.get("lf_strength", 1.5)
    if not isinstance(strength, (int, float)) or strength < 0:
        raise ValueError(f"lf_strength must be non-negative, got {strength}")
    
    # 生成伪高斯采样向量：基于 key seed 与 coeffs 维度。
    np.random.seed(key)
    pseudogaussian_factors = np.random.randn(coeffs.shape[0]).astype(np.float32)
    
    # 生成二进制码字（±1）。
    codeword = 2 * (np.random.binomial(1, 0.5, coeffs.shape[0]) - 0.5)
    
    # 伪高斯采样：codeword * |randn()|。
    watermark_pattern = codeword * np.abs(pseudogaussian_factors)
    
    # 以 strength 系数应用水印。
    encoded_coeffs = coeffs + strength * watermark_pattern
    
    # 构造编码证据（摘要不含原始张量）。
    encoding_evidence = {
        "strength_applied": float(strength),
        "pattern_seed": int(key),
        "coeffs_before_norm": float(np.linalg.norm(coeffs)),
        "coeffs_after_norm": float(np.linalg.norm(encoded_coeffs)),
        "pattern_norm": float(np.linalg.norm(watermark_pattern)),
        "coeffs_count": int(coeffs.shape[0])
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
    latents_flat = np.dot(coeffs, basis_matrix_np.T)  # shape: (latent_dim,)
    
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
            score = np.dot(coeffs, reference_pattern) / normalization
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
