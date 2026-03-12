"""
高频子空间鲁棒约束通道实现

功能说明：
- 在 HF（高频）子空间实现鲁棒增强编码。
- 支持尾部截断与受控约束策略。
- 生成可复算的 hf_trace_digest 与固定精度指标。
- 严格区分 absent / failed / mismatch 语义。
- 确保 HF 与 LF 正交，HF 失败不污染 LF。

Module type: Core innovation module
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from main.core import digests


HF_CHANNEL_IMPL_ID = "high_freq_template_codec"
HF_CHANNEL_VERSION = "v2"
HF_TRACE_VERSION = "v2"

HF_ABSENT_REASONS = {
    "hf_disabled_by_config",
}

HF_FAILURE_REASONS = {
    "hf_invalid_input",
    "hf_missing_basis",
    "hf_basis_mismatch",
    "hf_encoding_failed",
}


def compute_hf_basis_projection_torch(latents: Any, basis: Dict[str, Any]) -> Any:
    """
    功能：使用 torch 在 HF 基向量上计算投影系数。

    Compute HF projection coefficients using torch tensor operations.

    Args:
        latents: Torch tensor latents.
        basis: Basis mapping with hf_projection_matrix.

    Returns:
        Torch tensor coefficients.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If shape mismatches.
    """
    import torch

    if not torch.is_tensor(latents):
        raise TypeError("latents must be torch.Tensor")
    if not isinstance(basis, dict):
        raise TypeError("basis must be dict mapping")

    basis_matrix = basis.get("hf_projection_matrix")
    if basis_matrix is None:
        raise ValueError("basis must contain hf_projection_matrix for HF channel")

    latents_flat = latents.reshape(-1).to(dtype=torch.float32)
    basis_matrix_t = torch.as_tensor(
        basis_matrix,
        dtype=torch.float32,
        device=latents_flat.device
    )
    if basis_matrix_t.ndim != 2:
        raise ValueError("hf_projection_matrix must be rank-2")
    # 维度不匹配时，使用 trajectory_feature_spec（TFSW 主路径）将 latent 降至 feature_dim。
    # 与 LF 使用相同的 projection_seed 保证两通道在同一特征空间内。
    if latents_flat.shape[0] != basis_matrix_t.shape[0]:
        tfs = basis.get("trajectory_feature_spec")
        if isinstance(tfs, dict) and tfs.get("feature_operator") == "masked_normalized_random_projection":
            from .subspace.trajectory_feature_space import extract_trajectory_feature_torch
            latents_flat = extract_trajectory_feature_torch(latents, tfs)
        else:
            # backward-compat：latent_projection_spec 随机索引路径。
            latent_proj_spec = basis.get("latent_projection_spec")
            if latent_proj_spec is None or latent_proj_spec.get("method") != "random_index_selection":
                raise ValueError(
                    f"latents dimension {latents_flat.shape[0]} != basis dimension {basis_matrix_t.shape[0]}: "
                    "neither trajectory_feature_spec nor latent_projection_spec found in basis"
                )
            feature_dim = int(latent_proj_spec.get("feature_dim", 128))
            seed = int(latent_proj_spec.get("seed", 0))
            t_idx = int(latent_proj_spec.get("edit_timestep", 0))
            sample_idx = int(latent_proj_spec.get("sample_idx", 0))
            projection_seed = seed + 7919 + t_idx * 131 + sample_idx
            rng = np.random.default_rng(projection_seed)
            indices = rng.integers(0, max(1, int(latents_flat.shape[0])), size=feature_dim)
            indices_t = torch.as_tensor(indices, dtype=torch.long, device=latents_flat.device)
            latents_flat = latents_flat[indices_t]
    return torch.matmul(latents_flat, basis_matrix_t)


def apply_hf_truncation_constraint_torch(coeffs: Any, cfg: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    功能：使用 torch 对 HF 系数执行尾部截断约束。

    Apply deterministic tail truncation on torch HF coefficients.

    Args:
        coeffs: Torch coefficients tensor.
        cfg: Config mapping.

    Returns:
        Tuple of (constrained_coeffs, constraint_evidence).

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

    hf_threshold_percentile = cfg.get("hf_threshold_percentile", 75.0)
    if not isinstance(hf_threshold_percentile, (int, float)) or not (0 <= hf_threshold_percentile <= 100):
        raise ValueError(f"hf_threshold_percentile must be in [0, 100], got {hf_threshold_percentile}")

    coeffs_fp32 = coeffs.to(dtype=torch.float32)
    abs_coeffs = torch.abs(coeffs_fp32)
    quantile = float(hf_threshold_percentile) / 100.0
    threshold_value = torch.quantile(abs_coeffs, quantile)

    mask = abs_coeffs >= threshold_value
    constrained_coeffs = torch.where(mask, coeffs_fp32, torch.zeros_like(coeffs_fp32))

    retained_count = int(mask.sum().item())
    total_count = int(coeffs.shape[0])
    retention_ratio = float(retained_count / total_count) if total_count > 0 else 0.0

    constraint_evidence = {
        "threshold_percentile_applied": float(hf_threshold_percentile),
        "threshold_value": float(threshold_value.item()),
        "coeffs_before_norm": float(torch.linalg.vector_norm(coeffs_fp32).item()),
        "coeffs_after_norm": float(torch.linalg.vector_norm(constrained_coeffs).item()),
        "coeffs_retained_count": retained_count,
        "coeffs_total_count": total_count,
        "retention_ratio": retention_ratio
    }
    return constrained_coeffs, constraint_evidence


def reconstruct_from_hf_coeffs_torch(
    coeffs: Any,
    basis: Dict[str, Any],
    latents_shape: Tuple[int, ...],
    dtype: Any
) -> Any:
    """
    功能：使用 torch 从 HF 系数重建张量。

    Reconstruct latent tensor from HF coefficients using torch.

    Args:
        coeffs: Torch coefficients tensor.
        basis: Basis mapping with hf_projection_matrix.
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

    basis_matrix = basis.get("hf_projection_matrix")
    if basis_matrix is None:
        raise ValueError("basis must contain hf_projection_matrix for HF channel")
    basis_matrix_t = torch.as_tensor(
        basis_matrix,
        dtype=torch.float32,
        device=coeffs.device
    )
    if basis_matrix_t.ndim != 2:
        raise ValueError("hf_projection_matrix must be rank-2")

    basis_dim = basis_matrix_t.shape[0]
    full_dim = int(np.prod(latents_shape))
    latents_sub = torch.matmul(basis_matrix_t, coeffs.to(dtype=torch.float32))
    # basis_dim < full_dim：优先使用 trajectory_feature_spec TFSW pullback。
    if basis_dim < full_dim:
        tfs = basis.get("trajectory_feature_spec")
        if isinstance(tfs, dict) and tfs.get("feature_operator") == "masked_normalized_random_projection":
            from .subspace.trajectory_feature_space import pullback_feature_delta_torch
            reconstructed = pullback_feature_delta_torch(latents_sub, tfs, latents_shape)
        else:
            # backward-compat：latent_projection_spec scatter-back。
            latent_proj_spec = basis.get("latent_projection_spec")
            if latent_proj_spec is None or latent_proj_spec.get("method") != "random_index_selection":
                raise ValueError(
                    "HF basis_dim < full_dim but neither trajectory_feature_spec nor latent_projection_spec found in basis"
                )
            feature_dim = int(latent_proj_spec.get("feature_dim", 128))
            seed = int(latent_proj_spec.get("seed", 0))
            t_idx = int(latent_proj_spec.get("edit_timestep", 0))
            sample_idx = int(latent_proj_spec.get("sample_idx", 0))
            projection_seed = seed + 7919 + t_idx * 131 + sample_idx
            rng = np.random.default_rng(projection_seed)
            indices = rng.integers(0, max(1, full_dim), size=feature_dim)
            indices_t = torch.as_tensor(indices, dtype=torch.long, device=coeffs.device)
            output = torch.zeros(full_dim, dtype=torch.float32, device=coeffs.device)
            output[indices_t] = latents_sub
            reconstructed = output.reshape(latents_shape)
    else:
        reconstructed = latents_sub.reshape(latents_shape)
    return reconstructed.to(dtype=dtype)


def compute_hf_basis_projection(latents: np.ndarray | Any, basis: Dict[str, Any]) -> np.ndarray:
    """
    功能：计算 latent 在 HF 基向量上的投影系数。
    
    Compute projection coefficients of latents onto HF basis vectors.
    Projects to high-freq subspace (orthogonal to LF).
    
    Args:
        latents: Input latent array (same spatial layout as LF).
        basis: HF basis mapping with shape and projection matrix info.
    
    Returns:
        Projection coefficients (np.ndarray) in HF subspace.
    
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
    
    # 从 basis 中提取 HF 投影方向矩阵。
    basis_matrix = basis.get("hf_projection_matrix")
    if basis_matrix is None:
        raise ValueError("basis must contain hf_projection_matrix for HF channel")
    
    if hasattr(basis_matrix, "cpu"):
        basis_matrix_np = basis_matrix.cpu().numpy()
    else:
        basis_matrix_np = np.asarray(basis_matrix, dtype=np.float32)
    
    # 扁平化 latents。
    latents_flat = latents_np.reshape(-1)
    
    # 投影计算。
    if latents_flat.shape[0] != basis_matrix_np.shape[0]:
        raise ValueError(
            f"latents dimension {latents_flat.shape[0]} != basis dimension {basis_matrix_np.shape[0]}"
        )
    
    coeffs = np.dot(latents_flat, basis_matrix_np)  # shape: (hf_rank,)
    return coeffs.astype(np.float32)


def apply_hf_truncation_constraint(
    coeffs: np.ndarray,
    cfg: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    功能：在 HF 投影系数上运用尾部截断与约束。
    
    Apply tail-truncation and clipping constraint to HF coefficients.
    Stabilizes high-frequency evidence via deterministic threshold-based removal.
    
    Args:
        coeffs: Projected coefficients (HF subspace, 1D array).
        cfg: Config dict with HF constraint parameters (threshold, etc).
    
    Returns:
        Tuple of (constrained_coeffs, constraint_evidence).
        constraint_evidence contains metrics and applied parameters.
    
    Raises:
        TypeError: If coeffs array type is invalid.
        ValueError: If cfg parameters out of bounds.
    """
    if not isinstance(coeffs, np.ndarray):
        raise TypeError("coeffs must be np.ndarray")
    
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    
    # 获取截断参数。
    hf_threshold_percentile = cfg.get("hf_threshold_percentile", 75.0)
    if not isinstance(hf_threshold_percentile, (int, float)) or not (0 <= hf_threshold_percentile <= 100):
        raise ValueError(f"hf_threshold_percentile must be in [0, 100], got {hf_threshold_percentile}")
    
    # 计算尾部阈值（基于绝对值百分位数）。
    abs_coeffs = np.abs(coeffs)
    threshold_value = np.percentile(abs_coeffs, hf_threshold_percentile)
    
    # 应用截断：保留超过阈值的系数，其余置零。
    mask = abs_coeffs >= threshold_value
    constrained_coeffs = np.where(mask, coeffs, 0.0).astype(np.float32)
    
    # 构造约束证据。
    constraint_evidence = {
        "threshold_percentile_applied": float(hf_threshold_percentile),
        "threshold_value": float(threshold_value),
        "coeffs_before_norm": float(np.linalg.norm(coeffs)),
        "coeffs_after_norm": float(np.linalg.norm(constrained_coeffs)),
        "coeffs_retained_count": int(np.sum(mask)),
        "coeffs_total_count": int(coeffs.shape[0]),
        "retention_ratio": float(np.sum(mask) / coeffs.shape[0])
    }
    
    return constrained_coeffs.astype(np.float32), constraint_evidence


def reconstruct_from_hf_coeffs(
    coeffs: np.ndarray,
    basis: Dict[str, Any],
    latents_shape: Tuple[int, ...],
    is_torch: bool = False
) -> np.ndarray | Any:
    """
    功能：从 HF 投影系数恢复张量（投影反演）。
    
    Reconstruct latents from HF coefficients via basis inverse projection.
    Restores original space from HF projected subspace coefficients.
    
    Args:
        coeffs: Projected coefficients (1D array, HF subspace).
        basis: HF basis mapping with inverse projection info.
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
    
    # 提取 HF 投影矩阵用于反演。
    basis_matrix = basis.get("hf_projection_matrix")
    if basis_matrix is None:
        raise ValueError("basis must contain hf_projection_matrix for HF channel")
    
    if hasattr(basis_matrix, "cpu"):
        basis_matrix_np = basis_matrix.cpu().numpy()
    else:
        basis_matrix_np = np.asarray(basis_matrix, dtype=np.float32)
    
    # 反演。
    latents_flat = np.dot(coeffs, basis_matrix_np.T)
    
    # 恢复原始形状。
    latents_reconstructed = latents_flat.reshape(latents_shape).astype(np.float32)
    
    if is_torch:
        import torch
        return torch.from_numpy(latents_reconstructed)
    else:
        return latents_reconstructed


def extract_hf_score(
    latents: np.ndarray | Any,
    basis: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None
) -> float:
    """
    功能：从 latent 中提取 HF 检测分数（能量统计）。
    
    Extract HF detection score via projection energy analysis.
    Computes energy of HF coefficients after constraint.
    
    Args:
        latents: Input latent array for detection.
        basis: HF basis mapping (same as used during encoding).
        cfg: Optional config with detection parameters.
    
    Returns:
        HF score (float, non-negative; higher indicates stronger HF evidence).
    
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
    
    # 投影到 HF 子空间。
    coeffs = compute_hf_basis_projection(latents_np, basis)
    
    # 应用相同的约束获取检测分体系。
    if cfg is not None:
        constrained_coeffs, _ = apply_hf_truncation_constraint(coeffs, cfg)
    else:
        constrained_coeffs = coeffs
    
    # 计算能量（L2 范数）作为分数。
    score = float(np.linalg.norm(constrained_coeffs))
    
    return score


def generate_hf_evidence_digest(
    trace_components: list,
    params_digest: str
) -> str:
    """
    功能：生成 hf_trace_digest（可复算摘要）。
    
    Generate HF trace digest from ordered step evidence and parameters.
    Uses canonical SHA256 for reproducibility.
    
    Args:
        trace_components: Ordered list of step evidence dicts.
        params_digest: Parameter digest (already computed).
    
    Returns:
        HF trace digest (64-char hex sha256).
    
    Raises:
        TypeError: If inputs types invalid.
    """
    if not isinstance(trace_components, list):
        raise TypeError("trace_components must be list")
    
    if not isinstance(params_digest, str):
        raise TypeError("params_digest must be str")
    
    # 构造可复算的摘要输入。
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


def build_hf_embed_evidence(
    latents_before: np.ndarray,
    latents_after: np.ndarray,
    trace_components: list,
    constraint_evidence: Dict[str, Any],
    cfg: Dict[str, Any],
    plan_digest: Optional[str] = None
) -> Dict[str, Any]:
    """
    功能：构造 HF 嵌入侧证据（完整审计载体）。
    
    Build HF evidence structure for embed-side audit.
    Includes status, metrics, digest, and failure semantics.
    
    Args:
        latents_before: Latents before HF constraint.
        latents_after: Latents after HF constraint.
        trace_components: Ordered list of step traces.
        constraint_evidence: Evidence from apply_hf_truncation_constraint().
        cfg: Configuration mapping.
        plan_digest: Optional plan digest for binding.
    
    Returns:
        HF evidence dict with status, metrics, digest, etc.
    
    Raises:
        TypeError: If inputs types invalid.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    
    hf_enabled = cfg.get("hf_enabled", True)
    if not hf_enabled:
        return {
            "status": "absent",
            "absent_reason": "hf_disabled_by_config",
            "hf_score": None,
            "hf_trace_digest": None,
            "hf_metrics": None,
            "hf_params_digest": None
        }
    
    # 构造参数摘要（固定字段集）。
    hf_params = {
        "impl_id": HF_CHANNEL_IMPL_ID,
        "impl_version": HF_CHANNEL_VERSION,
        "hf_threshold_percentile": cfg.get("hf_threshold_percentile", 75.0),
        "hf_enabled": True
    }
    hf_params_digest = digests.canonical_sha256(hf_params)
    
    # 生成 hf_trace_digest。
    hf_trace_digest = generate_hf_evidence_digest(trace_components, hf_params_digest)
    
    # 计算固定精度指标（fp32 精度）。
    latents_before_np = np.asarray(latents_before, dtype=np.float32)
    latents_after_np = np.asarray(latents_after, dtype=np.float32)
    
    hf_metrics = {
        "before_norm": float(np.linalg.norm(latents_before_np)),
        "after_norm": float(np.linalg.norm(latents_after_np)),
        "delta_norm": float(np.linalg.norm(latents_after_np - latents_before_np)),
        "trace_count": len(trace_components),
        "impl_version": HF_TRACE_VERSION
    }
    if isinstance(constraint_evidence, dict):
        hf_metrics.update(constraint_evidence)
    
    evidence = {
        "status": "ok",
        "hf_score": None,  # embed 端不计算 score，仅生成证据。
        "hf_trace_digest": hf_trace_digest,
        "hf_metrics": hf_metrics,
        "hf_params_digest": hf_params_digest,
        "plan_digest": plan_digest if plan_digest else None
    }
    
    return evidence
