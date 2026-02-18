"""
扩散轨迹采样规范与可验证闭包

功能说明：
- 定义轨迹采样的完整规范（可验证输入域）。
- 支持轨迹采样、摘要锚定、跨流程一致性校验。
- 确保 embed 与 detect 侧的轨迹规范一致性绑定。

Module type: Core innovation module。
Innovation boundary: 以可复算、可验证的方式绑定扩散轨迹采样规范，
避免外部直接喂入的特征矩阵。确立"真实轨迹"与"特征子空间"的因果关系。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import json
import numpy as np

from main.core import digests


@dataclass(frozen=True)
class TrajectoryAcquisitionSpec:
    """
    功能：轨迹采样规范。
    
    Specifies deterministic trajectory sampling from diffusion inference,
    enabling reproducible feature extraction and validation across embed/detect.
    
    Attributes:
        trajectory_method: 采样方法标识（latent_trajectory / noise_residual / latent_flow）。
        timestep_window: [timestep_start, timestep_end] 闭区间。
        timestep_stride: 采样步长（1=每步采样, 2=隔步采样）。
        sample_count: 目标采样点数。
        sample_padding_mode: 样本数不足时补齐模式（repeat_last / zero_pad / interpolate）。
        feature_projection_method: 特征投影方法（variance_topk / pca / randomized_projection）。
        feature_dim: 投影后的特征维度。
        acquisition_seed: 采样派生种子（用于 probe、投影等）。
        trajectory_acquisition_tag: 采样场景标签（e.g. "inference_t2i" / "inference_i2i"）。
    """
    trajectory_method: str
    timestep_window: Tuple[int, int]
    timestep_stride: int
    sample_count: int
    sample_padding_mode: str
    feature_projection_method: str
    feature_dim: int
    acquisition_seed: int
    trajectory_acquisition_tag: str = "default_acquisition"
    
    def __post_init__(self) -> None:
        # 校验 trajectory_method。
        allowed_methods = {"latent_trajectory", "noise_residual", "latent_flow"}
        if self.trajectory_method not in allowed_methods:
            raise ValueError(
                f"trajectory_method must be one of {allowed_methods}, got {self.trajectory_method}"
            )
        
        # 校验 timestep_window。
        if not isinstance(self.timestep_window, Tuple) or len(self.timestep_window) != 2:
            raise ValueError("timestep_window must be tuple of 2 ints")
        start, end = self.timestep_window
        if start > end or start < 0:
            raise ValueError(f"timestep_window must be [start, end] with 0 <= start <= end, got {self.timestep_window}")
        
        # 校验 timestep_stride。
        if self.timestep_stride <= 0:
            raise ValueError("timestep_stride must be positive")
        
        # 校验 sample_count。
        if self.sample_count < 2:
            raise ValueError("sample_count must be >= 2")
        
        # 校验 sample_padding_mode。
        allowed_padding = {"repeat_last", "zero_pad", "interpolate"}
        if self.sample_padding_mode not in allowed_padding:
            raise ValueError(
                f"sample_padding_mode must be one of {allowed_padding}, got {self.sample_padding_mode}"
            )
        
        # 校验 feature_projection_method。
        allowed_projections = {"variance_topk", "pca", "randomized_projection"}
        if self.feature_projection_method not in allowed_projections:
            raise ValueError(
                f"feature_projection_method must be one of {allowed_projections}, got {self.feature_projection_method}"
            )
        
        # 校验 feature_dim。
        if self.feature_dim < 2:
            raise ValueError("feature_dim must be >= 2")
        
        # 校验 acquisition_seed。
        if self.acquisition_seed < 0:
            raise ValueError("acquisition_seed must be non-negative")
    
    def as_dict(self) -> Dict[str, Any]:
        """
        功能：将规范序列化为字典。
        
        Serialize specification to canonical dict for digest computation.
        
        Args:
            None.
        
        Returns:
            Canonical specification dict.
        """
        return {
            "trajectory_method": self.trajectory_method,
            "timestep_window": list(self.timestep_window),
            "timestep_stride": self.timestep_stride,
            "sample_count": self.sample_count,
            "sample_padding_mode": self.sample_padding_mode,
            "feature_projection_method": self.feature_projection_method,
            "feature_dim": self.feature_dim,
            "acquisition_seed": self.acquisition_seed,
            "trajectory_acquisition_tag": self.trajectory_acquisition_tag
        }


@dataclass(frozen=True)
class TrajectoryAcquisitionAnchor:
    """
    功能：轨迹采样摘要锚点。
    
    Digest anchors for trajectory verification across embed/detect.
    Computed deterministically from actual sampled trajectory.
    
    Attributes:
        acquisition_spec_digest: SHA256 of TrajectoryAcquisitionSpec.as_dict()。
        trajectory_sample_digests: 轨迹各采样点的摘要列表（用于逐点校验）。
        trajectory_overall_digest: 整体轨迹矩阵的摘要。
        trajectory_shape: (sample_count, latent_dim)。
        feature_matrix_digest: 投影后特征矩阵的摘要。
        feature_matrix_shape: (sample_count, feature_dim)。
        feature_statistics: 特征矩阵的统计摘要（均值、标差等）。
    """
    acquisition_spec_digest: str
    trajectory_sample_digests: List[str]
    trajectory_overall_digest: str
    trajectory_shape: Tuple[int, int]
    feature_matrix_digest: str
    feature_matrix_shape: Tuple[int, int]
    feature_statistics: Dict[str, Any]
    
    def __post_init__(self) -> None:
        # 校验各字段格式。
        if not isinstance(self.acquisition_spec_digest, str) or len(self.acquisition_spec_digest) != 64:
            raise ValueError("acquisition_spec_digest must be 64-char hex SHA256")
        
        if not isinstance(self.trajectory_sample_digests, list):
            raise ValueError("trajectory_sample_digests must be list")
        for d in self.trajectory_sample_digests:
            if not isinstance(d, str) or len(d) != 64:
                raise ValueError("each trajectory_sample_digest must be 64-char hex SHA256")
        
        if not isinstance(self.trajectory_overall_digest, str) or len(self.trajectory_overall_digest) != 64:
            raise ValueError("trajectory_overall_digest must be 64-char hex SHA256")
        
        if not isinstance(self.trajectory_shape, Tuple) or len(self.trajectory_shape) != 2:
            raise ValueError("trajectory_shape must be tuple of 2 ints")
        
        if not isinstance(self.feature_matrix_digest, str) or len(self.feature_matrix_digest) != 64:
            raise ValueError("feature_matrix_digest must be 64-char hex SHA256")
        
        if not isinstance(self.feature_matrix_shape, Tuple) or len(self.feature_matrix_shape) != 2:
            raise ValueError("feature_matrix_shape must be tuple of 2 ints")
        
        if not isinstance(self.feature_statistics, dict):
            raise ValueError("feature_statistics must be dict")
    
    def as_dict(self) -> Dict[str, Any]:
        """
        功能：将锚点序列化为字典。
        
        Serialize anchor to canonical dict.
        
        Args:
            None.
        
        Returns:
            Canonical anchor dict.
        """
        return {
            "acquisition_spec_digest": self.acquisition_spec_digest,
            "trajectory_sample_digests": self.trajectory_sample_digests,
            "trajectory_overall_digest": self.trajectory_overall_digest,
            "trajectory_shape": list(self.trajectory_shape),
            "feature_matrix_digest": self.feature_matrix_digest,
            "feature_matrix_shape": list(self.feature_matrix_shape),
            "feature_statistics": self.feature_statistics
        }


class TrajectoryAnchorBuilder:
    """
    功能：轨迹摘要锚点构造器。
    
    Builds digest anchors from actual sampled trajectory and feature matrix.
    Enables verification of trajectory provenance and consistency.
    """
    
    @staticmethod
    def build_anchor(
        spec: TrajectoryAcquisitionSpec,
        trajectory_matrix: np.ndarray,
        feature_matrix: np.ndarray,
        feature_statistics: Optional[Dict[str, Any]] = None
    ) -> TrajectoryAcquisitionAnchor:
        """
        功能：从采样轨迹和特征矩阵构造锚点。
        
        Build anchor from trajectory and feature matrix.
        
        Args:
            spec: Trajectory acquisition specification.
            trajectory_matrix: Raw trajectory matrix (sample_count, latent_dim).
            feature_matrix: Projected feature matrix (sample_count, feature_dim).
            feature_statistics: Optional pre-computed statistics dict.
        
        Returns:
            TrajectoryAcquisitionAnchor instance.
        
        Raises:
            ValueError: If matrices are invalid.
        """
        if not isinstance(spec, TrajectoryAcquisitionSpec):
            raise ValueError("spec must be TrajectoryAcquisitionSpec")
        
        if not isinstance(trajectory_matrix, np.ndarray) or trajectory_matrix.ndim != 2:
            raise ValueError("trajectory_matrix must be 2D numpy array")
        
        if not isinstance(feature_matrix, np.ndarray) or feature_matrix.ndim != 2:
            raise ValueError("feature_matrix must be 2D numpy array")
        
        if trajectory_matrix.shape[0] != feature_matrix.shape[0]:
            raise ValueError(
                f"trajectory and feature matrices must have same sample_count, "
                f"got trajectory {trajectory_matrix.shape[0]} vs feature {feature_matrix.shape[0]}"
            )
        
        # 计算 spec 摘要。
        spec_digest = digests.canonical_sha256(spec.as_dict())
        
        # 逐采样点计算摘要。
        sample_digests: List[str] = []
        for i in range(trajectory_matrix.shape[0]):
            sample_dict = {
                "sample_index": i,
                "trajectory_row": trajectory_matrix[i, :].tolist(),
                "feature_row": feature_matrix[i, :].tolist()
            }
            sample_digest = digests.canonical_sha256(sample_dict)
            sample_digests.append(sample_digest)
        
        # 计算整体轨迹摘要。
        trajectory_dict = {"trajectory": trajectory_matrix.tolist()}
        trajectory_overall_digest = digests.canonical_sha256(trajectory_dict)
        
        # 计算特征矩阵摘要。
        feature_dict = {"feature_matrix": feature_matrix.tolist()}
        feature_matrix_digest = digests.canonical_sha256(feature_dict)
        
        # 计算或使用提供的统计摘要。
        if feature_statistics is None:
            feature_statistics = TrajectoryAnchorBuilder._compute_feature_statistics(feature_matrix)
        
        return TrajectoryAcquisitionAnchor(
            acquisition_spec_digest=spec_digest,
            trajectory_sample_digests=sample_digests,
            trajectory_overall_digest=trajectory_overall_digest,
            trajectory_shape=tuple(trajectory_matrix.shape),
            feature_matrix_digest=feature_matrix_digest,
            feature_matrix_shape=tuple(feature_matrix.shape),
            feature_statistics=feature_statistics
        )
    
    @staticmethod
    def _compute_feature_statistics(feature_matrix: np.ndarray) -> Dict[str, Any]:
        """
        功能：计算特征矩阵的统计摘要。
        
        Compute canonical feature statistics for verification.
        
        Args:
            feature_matrix: Feature matrix (sample_count, feature_dim).
        
        Returns:
            Statistics dict.
        """
        mean = np.mean(feature_matrix, axis=0).tolist()
        std = np.std(feature_matrix, axis=0).tolist()
        min_vals = np.min(feature_matrix, axis=0).tolist()
        max_vals = np.max(feature_matrix, axis=0).tolist()
        
        return {
            "mean": mean,
            "std": std,
            "min": min_vals,
            "max": max_vals,
            "shape": list(feature_matrix.shape),
            "dtype": str(feature_matrix.dtype)
        }
    
    @staticmethod
    def verify_anchor_consistency(
        stored_anchor: TrajectoryAcquisitionAnchor,
        spec: TrajectoryAcquisitionSpec,
        trajectory_matrix: np.ndarray,
        feature_matrix: np.ndarray,
        statistics: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        功能：验证存储的锚点与当前轨迹的一致性。
        
        Verify that stored anchor matches current trajectory and feature matrix.
        Used by detect-side to validate trajectory provenance.
        
        Args:
            stored_anchor: Previously stored anchor (e.g. from run_closure).
            spec: Current trajectory specification.
            trajectory_matrix: Current trajectory matrix.
            feature_matrix: Current feature matrix.
            statistics: Optional current statistics.
        
        Returns:
            (is_consistent, mismatch_reason_or_none).
                - (True, None) if all digests match.
                - (False, reason_str) if mismatch detected.
        """
        # 验证 spec 摘要。
        current_spec_digest = digests.canonical_sha256(spec.as_dict())
        if current_spec_digest != stored_anchor.acquisition_spec_digest:
            return False, "acquisition_spec_digest_mismatch"
        
        # 验证轨迹形状。
        if tuple(trajectory_matrix.shape) != stored_anchor.trajectory_shape:
            return False, "trajectory_shape_mismatch"
        
        # 验证特征形状。
        if tuple(feature_matrix.shape) != stored_anchor.feature_matrix_shape:
            return False, "feature_matrix_shape_mismatch"
        
        # 验证整体轨迹摘要。
        trajectory_dict = {"trajectory": trajectory_matrix.tolist()}
        current_trajectory_digest = digests.canonical_sha256(trajectory_dict)
        if current_trajectory_digest != stored_anchor.trajectory_overall_digest:
            return False, "trajectory_overall_digest_mismatch"
        
        # 验证特征矩阵摘要。
        feature_dict = {"feature_matrix": feature_matrix.tolist()}
        current_feature_digest = digests.canonical_sha256(feature_dict)
        if current_feature_digest != stored_anchor.feature_matrix_digest:
            return False, "feature_matrix_digest_mismatch"
        
        # 可选：验证统计摘要（用于快速预检查）。
        if statistics is not None:
            stored_stats = stored_anchor.feature_statistics
            if isinstance(statistics, dict):
                for key in ["mean", "std", "shape"]:
                    if key in stored_stats and key in statistics:
                        if stored_stats[key] != statistics[key]:
                            return False, f"feature_statistics_{key}_mismatch"
        
        return True, None
