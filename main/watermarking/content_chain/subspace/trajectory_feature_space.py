"""
轨迹特征空间算子（Trajectory Feature Space Operator，TFSW）

功能说明：
- 定义特征算子 Φ：z_t → φ = Pᵀ vec(normalize(M ⊙ z_t))
- P ∈ R^(latent_dim × feature_dim) 为随机高斯投影矩阵，缩放保证近等距性。
- 支持 torch 与 numpy 双路径，设计对称。
- pullback 实现 Δφ → Δz_t，用于 watermark 注入。

Module type: Core innovation module

Innovation boundary:
    以固定随机投影矩阵 P 将高维 latent 映射到低维特征空间，使规划期与
    运行期共享同一特征空间（trajectory feature subspace）。规划期学习的
    LF/HF basis 在此特征空间内有效，embed/detect 通过相同的 P 可重现
    完全一致的特征提取。

Dependency assumptions:
    - projection_seed 由 planner_seed + edit_timestep 确定，不依赖 sample_idx，
      保证特征空间跨样本一致性（同一 edit_timestep 下所有样本共享 P）。
    - 对 mask_summary 的支持是可选的；当前版本令 M = 1（全 latent），
      channel-wise 归一化作为隐式的掩码等效约束。
    - P 从 np.random.default_rng(projection_seed) 派生，可完全复现。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


TFSW_SPEC_VERSION = "v1"
TFSW_FEATURE_OPERATOR = "masked_normalized_random_projection"


# ---------------------------------------------------------------------------
# Numpy 路径（规划期，CPU）
# ---------------------------------------------------------------------------

def build_projection_matrix_np(
    spec: Dict[str, Any],
    latent_shape: Any,
) -> np.ndarray:
    """
    Build random projection matrix P ∈ R^(latent_dim × feature_dim) using numpy.

    P is drawn from N(0, 1/latent_dim) to provide approximate isometric embedding
    (Johnson-Lindenstrauss property). The same seed always produces the same P,
    ensuring deterministic consistency between planning and embedding/detection.

    Args:
        spec: trajectory_feature_spec dict with keys:
            - "projection_seed" (int): Deterministic seed (planner_seed + edit_timestep).
            - "feature_dim" (int): Target feature dimension (e.g., 128).
        latent_shape: Shape tuple (or scalar) used to compute latent_dim.

    Returns:
        numpy.ndarray of shape (latent_dim, feature_dim), dtype float32.

    Raises:
        TypeError: If spec is not a dict.
        ValueError: If feature_dim > latent_dim or required keys are missing.
    """
    if not isinstance(spec, dict):
        raise TypeError("spec must be dict")
    feature_dim = int(spec.get("feature_dim", 128))
    projection_seed = int(spec.get("projection_seed", 0))
    if hasattr(latent_shape, "__len__"):
        latent_dim = int(np.prod(latent_shape))
    else:
        latent_dim = int(latent_shape)
    if feature_dim > latent_dim:
        raise ValueError(
            f"feature_dim {feature_dim} > latent_dim {latent_dim}; "
            "cannot project to higher dimension"
        )
    rng = np.random.default_rng(projection_seed)
    # 缩放因子 1/sqrt(latent_dim) 保证 |Pᵀx|₂ ≈ |x|₂ 的近似等距性。
    P = rng.standard_normal((latent_dim, feature_dim)).astype(np.float32)
    P /= math.sqrt(latent_dim)
    return P


def normalize_masked_latent_np(
    z_t: Any,
    mask_summary: Optional[Dict[str, Any]],
    spec: Dict[str, Any],
) -> np.ndarray:
    """
    Apply channel-wise normalization to z_t (M=1), then flatten to 1-D.

    Channel normalization: per-channel mean subtraction and std division
    over spatial dimensions, stabilizing the feature space across diverse
    latent magnitudes.

    Args:
        z_t: Input latent array; shape (..., C, H, W) or flat.
        mask_summary: Optional mask metadata (currently unused; M=1).
        spec: trajectory_feature_spec dict (unused in normalization).

    Returns:
        numpy float32 array of shape (latent_dim,).

    Raises:
        TypeError: If z_t cannot be converted to array.
    """
    z_np = np.asarray(z_t, dtype=np.float32)
    # 对 ndim >= 3 的张量做 channel-wise 归一化（最后两维为空间维）。
    if z_np.ndim >= 3:
        spatial_axes = tuple(range(z_np.ndim - 2, z_np.ndim))
        ch_mean = np.mean(z_np, axis=spatial_axes, keepdims=True)
        ch_std = np.std(z_np, axis=spatial_axes, keepdims=True) + 1e-8
        z_np = (z_np - ch_mean) / ch_std
    elif z_np.ndim == 2:
        # (C, pixels) 形式：按行（channel）均值归一化。
        ch_mean = np.mean(z_np, axis=-1, keepdims=True)
        ch_std = np.std(z_np, axis=-1, keepdims=True) + 1e-8
        z_np = (z_np - ch_mean) / ch_std
    # ndim == 1 时跳过归一化（已经是扁平向量）。
    return z_np.reshape(-1).astype(np.float32)


def extract_trajectory_feature_np(
    z_t: Any,
    trajectory_feature_spec: Dict[str, Any],
    mask_summary: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Compute trajectory feature vector φ = Pᵀ vec(normalize(M ⊙ z_t)) using numpy.

    Args:
        z_t: Input latent array (any shape).
        trajectory_feature_spec: Feature spec with "projection_seed" and "feature_dim".
        mask_summary: Optional mask metadata (currently unused, M=1).

    Returns:
        numpy float32 array of shape (feature_dim,).

    Raises:
        TypeError: If trajectory_feature_spec is not dict.
        ValueError: If spec is incompatible with z_t shape.
    """
    v = normalize_masked_latent_np(z_t, mask_summary, trajectory_feature_spec)
    P = build_projection_matrix_np(trajectory_feature_spec, v.shape)
    phi = P.T @ v  # (feature_dim,)
    return phi.astype(np.float32)


def pullback_feature_delta_np(
    delta_phi: np.ndarray,
    trajectory_feature_spec: Dict[str, Any],
    latent_shape: Any,
    mask_summary: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Compute latent-space delta Δz_t = P Δφ (pull-back / adjoint of Pᵀ).

    Given a desired modification Δφ in feature space (e.g., after watermark
    embedding), this maps it back to the latent space via the left adjoint P.

    Args:
        delta_phi: Feature-space delta, shape (feature_dim,), dtype float32.
        trajectory_feature_spec: Feature spec matching embed-side spec.
        latent_shape: Target latent tensor shape (e.g., (1, 16, 64, 64)).
        mask_summary: Optional mask metadata (currently unused).

    Returns:
        numpy float32 array of latent_shape.

    Raises:
        TypeError: If delta_phi is not array-like.
        ValueError: If spec is incompatible.
    """
    delta_phi_np = np.asarray(delta_phi, dtype=np.float32)
    if hasattr(latent_shape, "__len__"):
        latent_dim = int(np.prod(latent_shape))
    else:
        latent_dim = int(latent_shape)
    P = build_projection_matrix_np(trajectory_feature_spec, (latent_dim,))
    delta_z_flat = P @ delta_phi_np  # (latent_dim,)
    return delta_z_flat.reshape(latent_shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Torch 路径（运行期，支持 CUDA）
# ---------------------------------------------------------------------------

def build_projection_matrix_torch(
    spec: Dict[str, Any],
    latent_shape: Any,
    device: Any,
) -> Any:
    """
    Build random projection matrix P as torch.Tensor on specified device.

    Shares the same seed and scaling convention as build_projection_matrix_np,
    guaranteeing that numpy and torch paths produce identical matrices.

    Args:
        spec: trajectory_feature_spec dict.
        latent_shape: Shape of the input latent tensor.
        device: Torch device (cpu or cuda).

    Returns:
        torch.Tensor P of shape (latent_dim, feature_dim), float32.

    Raises:
        TypeError: If spec is not dict.
        ValueError: If feature_dim > latent_dim.
    """
    import torch

    P_np = build_projection_matrix_np(spec, latent_shape)
    return torch.from_numpy(P_np).to(device=device)


def normalize_masked_latent_torch(
    z_t: Any,
    mask_summary: Optional[Dict[str, Any]],
    spec: Dict[str, Any],
) -> Any:
    """
    Apply channel-wise normalization to z_t (M=1), then flatten. Torch path.

    Args:
        z_t: Torch latent tensor (any shape, will be cast to float32).
        mask_summary: Optional mask metadata (currently unused, M=1).
        spec: trajectory_feature_spec dict (unused in normalization).

    Returns:
        Flat torch float32 tensor of shape (latent_dim,).

    Raises:
        TypeError: If z_t is not a torch.Tensor.
    """
    import torch

    if not torch.is_tensor(z_t):
        raise TypeError("z_t must be torch.Tensor")
    z = z_t.to(dtype=torch.float32)
    if z.ndim >= 3:
        spatial_dims = tuple(range(z.ndim - 2, z.ndim))
        ch_mean = z.mean(dim=spatial_dims, keepdim=True)
        ch_std = z.std(dim=spatial_dims, keepdim=True) + 1e-8
        z = (z - ch_mean) / ch_std
    elif z.ndim == 2:
        ch_mean = z.mean(dim=-1, keepdim=True)
        ch_std = z.std(dim=-1, keepdim=True) + 1e-8
        z = (z - ch_mean) / ch_std
    return z.reshape(-1)


def extract_trajectory_feature_torch(
    z_t: Any,
    trajectory_feature_spec: Dict[str, Any],
    mask_summary: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Compute trajectory feature vector φ = Pᵀ vec(normalize(M ⊙ z_t)) via torch.

    Projection seed formula: projection_seed = planner_seed + edit_timestep.
    Sample index (sample_idx) must NOT be included in the seed to preserve
    feature space consistency across different images/samples.

    Args:
        z_t: Torch latent tensor.
        trajectory_feature_spec: Feature spec with "projection_seed" and "feature_dim".
        mask_summary: Optional mask metadata (currently unused, M=1).

    Returns:
        torch.Tensor φ of shape (feature_dim,), float32.

    Raises:
        TypeError: If z_t is not a torch.Tensor or spec is not dict.
        ValueError: If spec is incompatible with z_t shape.
    """
    import torch

    if not torch.is_tensor(z_t):
        raise TypeError("z_t must be torch.Tensor")
    if not isinstance(trajectory_feature_spec, dict):
        raise TypeError("trajectory_feature_spec must be dict")

    v = normalize_masked_latent_torch(z_t, mask_summary, trajectory_feature_spec)
    P = build_projection_matrix_torch(trajectory_feature_spec, tuple(v.shape), v.device)
    phi = torch.matmul(P.T, v)  # (feature_dim,)
    return phi


def pullback_feature_delta_torch(
    delta_phi: Any,
    trajectory_feature_spec: Dict[str, Any],
    latent_shape: Tuple[int, ...],
    mask_summary: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Compute latent-space delta Δz_t = P Δφ (pull-back of feature-space delta).

    Pull-back maps Δφ ∈ R^feature_dim to Δz_t ∈ R^latent_dim via the left
    adjoint P of Pᵀ. This ensures that the injected modification operates in
    the planned feature subspace.

    Args:
        delta_phi: Feature-space delta torch.Tensor, shape (feature_dim,).
        trajectory_feature_spec: Feature spec matching the embed-side spec.
        latent_shape: Target latent tensor shape (e.g., (1, 16, 64, 64)).
        mask_summary: Optional mask metadata (currently unused).

    Returns:
        torch.Tensor of latent_shape, float32.

    Raises:
        TypeError: If delta_phi is not a torch.Tensor or spec is not dict.
        ValueError: If spec is incompatible.
    """
    import torch

    if not torch.is_tensor(delta_phi):
        raise TypeError("delta_phi must be torch.Tensor")
    if not isinstance(trajectory_feature_spec, dict):
        raise TypeError("trajectory_feature_spec must be dict")

    latent_dim = 1
    for s in latent_shape:
        latent_dim *= s
    P = build_projection_matrix_torch(
        trajectory_feature_spec, (latent_dim,), delta_phi.device
    )
    delta_z_flat = torch.matmul(P, delta_phi.to(dtype=torch.float32))  # (latent_dim,)
    return delta_z_flat.reshape(latent_shape)
