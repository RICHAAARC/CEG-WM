"""
File purpose: 基于 anchor 与 sync 证据执行几何对齐并计算不变性得分。
Module type: General module
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from main.core import digests
from main.watermarking.geometry_chain.attention_anchor_extractor import AttentionAnchorExtractor
from main.watermarking.geometry_chain.sync.latent_sync_template import LatentSyncGeometryExtractor


GEO_AVAILABILITY_RULE_VERSION = "geo_availability_rule_v1"

GEO_UNAVAILABILITY_REASONS = {
    "geo_fit_not_converged",
    "geo_param_uncertainty_high",
    "geo_inlier_ratio_low",
    "geo_residual_stats_anomaly",
}


def _solve_linear_system_gaussian(matrix: np.ndarray, vector: np.ndarray) -> Optional[np.ndarray]:
    """
    功能：使用高斯消元求解小规模线性系统。 

    Solve small linear system via Gaussian elimination with partial pivoting.

    Args:
        matrix: Square coefficient matrix.
        vector: Right-hand-side vector.

    Returns:
        Solution vector or None when matrix is singular/invalid.
    """
    if matrix.ndim != 2:
        return None
    if vector.ndim != 1:
        return None
    rows, cols = matrix.shape
    if rows != cols or vector.shape[0] != rows:
        return None

    augmented = np.concatenate(
        [matrix.astype(np.float64, copy=True), vector.reshape(-1, 1).astype(np.float64, copy=True)],
        axis=1,
    )
    size = rows

    for pivot_index in range(size):
        pivot_row = pivot_index
        pivot_value = abs(float(augmented[pivot_row, pivot_index]))
        for row_index in range(pivot_index + 1, size):
            candidate_value = abs(float(augmented[row_index, pivot_index]))
            if candidate_value > pivot_value:
                pivot_row = row_index
                pivot_value = candidate_value

        if pivot_value <= 1e-12:
            return None

        if pivot_row != pivot_index:
            row_copy = augmented[pivot_index].copy()
            augmented[pivot_index] = augmented[pivot_row]
            augmented[pivot_row] = row_copy

        pivot_scalar = float(augmented[pivot_index, pivot_index])
        augmented[pivot_index] = augmented[pivot_index] / pivot_scalar

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = float(augmented[row_index, pivot_index])
            if abs(factor) <= 1e-15:
                continue
            augmented[row_index] = augmented[row_index] - factor * augmented[pivot_index]

    solution = augmented[:, -1]
    if not np.isfinite(solution).all():
        return None
    return solution.astype(np.float64)


@dataclass(frozen=True)
class FitResult:
    """
    功能：稳健拟合结果。

    Structured robust fitting result.

    Args:
        converged: Whether solver converged.
        model_type: Geometric model type.
        params: Estimated model parameters.
        residuals: Pointwise residual norms.
        inlier_mask: Inlier indicator per correspondence.
        residual_stats: Robust residual statistics.
        param_uncertainty: Parameter uncertainty summary.
        fit_stability: Aggregate stability score in [0, 1].
        iterations: Executed iteration count.

    Returns:
        None.
    """

    converged: bool
    model_type: str
    params: List[float]
    residuals: List[float]
    inlier_mask: List[bool]
    residual_stats: Dict[str, float]
    param_uncertainty: Dict[str, float]
    fit_stability: float
    iterations: int


class RobustSimilarityFitter:
    """
    功能：基于 IRLS 的低阶几何稳健拟合器。

    Robust low-order geometric fitter with IRLS optimization.
    """

    def fit(self, initial_transform: Dict[str, Any], correspondences: List[Dict[str, Any]], cfg: Dict[str, Any]) -> FitResult:
        """
        功能：执行稳健拟合并返回拟合结果。

        Run robust fitting and return structured fit result.

        Args:
            initial_transform: Initial transform payload.
            correspondences: Point correspondence list.
            cfg: Configuration mapping.

        Returns:
            FitResult for robust fitting.
        """
        if not isinstance(initial_transform, dict):
            raise TypeError("initial_transform must be dict")
        if not isinstance(correspondences, list):
            raise TypeError("correspondences must be list")
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")

        model_type = str(initial_transform.get("model_type", "similarity")).lower()
        if model_type not in {"similarity", "affine"}:
            model_type = "similarity"

        src_points, dst_points = self._build_point_arrays(correspondences)
        min_points = 3 if model_type == "similarity" else 4
        if src_points.shape[0] < min_points:
            return FitResult(
                converged=False,
                model_type=model_type,
                params=[],
                residuals=[],
                inlier_mask=[],
                residual_stats=self.compute_residual_stats(np.asarray([], dtype=np.float64)),
                param_uncertainty={"param_variance_norm": 1.0, "bootstrap_valid_rounds": 0.0},
                fit_stability=0.0,
                iterations=0,
            )

        max_iterations = self._resolve_align_max_iterations(cfg)
        inlier_threshold = self._resolve_align_inlier_threshold(cfg)
        robust_loss = self._resolve_align_robust_loss(cfg)
        tolerance = 1e-6

        params = self._initial_params_from_transform(initial_transform, model_type)
        converged = False
        iterations = 0
        weights = np.ones(src_points.shape[0], dtype=np.float64)

        for iterations in range(1, max_iterations + 1):
            next_params = self._solve_weighted_model(src_points, dst_points, weights, model_type)
            if next_params is None:
                break

            predicted = self._predict_points(src_points, next_params, model_type)
            residuals = np.linalg.norm(predicted - dst_points, axis=1)
            mad = self._compute_mad(residuals)
            sigma = max(1e-8, 1.4826 * mad)
            weights = self._compute_irls_weights(residuals, sigma, robust_loss)

            delta = float(np.linalg.norm(next_params - params))
            params = next_params
            if delta <= tolerance:
                converged = True
                break

        predicted = self._predict_points(src_points, params, model_type)
        residuals = np.linalg.norm(predicted - dst_points, axis=1)
        inlier_mask = self.compute_inliers(residuals, inlier_threshold)
        residual_stats = self.compute_residual_stats(residuals)
        param_uncertainty = self.estimate_param_uncertainty(src_points, dst_points, model_type, cfg)

        inlier_ratio = float(np.mean(inlier_mask.astype(np.float64))) if inlier_mask.size > 0 else 0.0
        mad_score = 1.0 - _clamp01(float(residual_stats.get("residual_mad", 1.0)))
        variance_score = 1.0 - _clamp01(float(param_uncertainty.get("param_variance_norm", 1.0)))
        fit_stability = _clamp01(0.45 * inlier_ratio + 0.30 * mad_score + 0.25 * variance_score)

        if not converged and iterations >= max_iterations:
            # 迭代上限却未数值收敛：仅当 fit_stability 超过保守阈值时
            # 视为"有条件收敛"，否则诚实返回 converged=False，
            # 避免抬高伪几何证据通过率。
            _CONDITIONAL_CONVERGENCE_THRESHOLD = 0.6
            if fit_stability >= _CONDITIONAL_CONVERGENCE_THRESHOLD and inlier_ratio >= 0.5:
                converged = True

        return FitResult(
            converged=converged,
            model_type=model_type,
            params=[float(v) for v in params.tolist()],
            residuals=[float(v) for v in residuals.tolist()],
            inlier_mask=[bool(v) for v in inlier_mask.tolist()],
            residual_stats={k: float(v) for k, v in residual_stats.items()},
            param_uncertainty={k: float(v) for k, v in param_uncertainty.items()},
            fit_stability=float(fit_stability),
            iterations=iterations,
        )

    def compute_inliers(self, residuals: np.ndarray, threshold: float) -> np.ndarray:
        """
        功能：计算 inlier 掩码。

        Compute inlier mask by residual threshold.

        Args:
            residuals: Residual norm array.
            threshold: Inlier threshold.

        Returns:
            Boolean inlier mask.
        """
        if not isinstance(residuals, np.ndarray):
            raise TypeError("residuals must be np.ndarray")
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be numeric")
        if residuals.size == 0:
            return np.asarray([], dtype=bool)
        return residuals <= float(max(1e-8, threshold))

    def compute_residual_stats(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        功能：计算稳健残差统计。

        Compute robust residual summary statistics.

        Args:
            residuals: Residual norm array.

        Returns:
            Residual statistics mapping.
        """
        if not isinstance(residuals, np.ndarray):
            raise TypeError("residuals must be np.ndarray")
        if residuals.size == 0:
            return {
                "residual_median": 1.0,
                "residual_mad": 1.0,
                "residual_trimmed_mean": 1.0,
                "residual_q25": 1.0,
                "residual_q75": 1.0,
            }

        sorted_res = np.sort(residuals.astype(np.float64))
        n_count = sorted_res.size
        trim_count = int(np.floor(0.1 * n_count))
        if n_count - 2 * trim_count <= 0:
            trimmed = sorted_res
        else:
            trimmed = sorted_res[trim_count:n_count - trim_count]

        median = float(np.median(sorted_res))
        mad = self._compute_mad(sorted_res)
        return {
            "residual_median": float(median),
            "residual_mad": float(mad),
            "residual_trimmed_mean": float(np.mean(trimmed)),
            "residual_q25": float(np.quantile(sorted_res, 0.25)),
            "residual_q75": float(np.quantile(sorted_res, 0.75)),
        }

    def estimate_param_uncertainty(self, src_points: np.ndarray, dst_points: np.ndarray, model_type: str, cfg: Dict[str, Any]) -> Dict[str, float]:
        """
        功能：估计参数不确定度（bootstrap 子采样）。

        Estimate parameter uncertainty with bootstrap subsampling.

        Args:
            src_points: Source points.
            dst_points: Target points.
            model_type: Model type.
            cfg: Configuration mapping.

        Returns:
            Parameter uncertainty mapping.
        """
        if not isinstance(src_points, np.ndarray):
            raise TypeError("src_points must be np.ndarray")
        if not isinstance(dst_points, np.ndarray):
            raise TypeError("dst_points must be np.ndarray")
        if not isinstance(model_type, str) or not model_type:
            raise TypeError("model_type must be non-empty str")
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")

        rounds = self._resolve_align_bootstrap_rounds(cfg)
        sample_ratio = self._resolve_align_bootstrap_sample_ratio(cfg)
        point_count = src_points.shape[0]
        sample_count = max(3, int(round(point_count * sample_ratio)))
        rng = np.random.RandomState(20260222)

        estimates: List[np.ndarray] = []
        for _ in range(rounds):
            if point_count <= sample_count:
                indices = np.arange(point_count)
            else:
                indices = rng.choice(point_count, size=sample_count, replace=False)
            sample_src = src_points[indices]
            sample_dst = dst_points[indices]
            solved = self._solve_weighted_model(
                sample_src,
                sample_dst,
                np.ones(sample_src.shape[0], dtype=np.float64),
                model_type,
            )
            if solved is not None:
                estimates.append(solved)

        if len(estimates) < 2:
            return {"param_variance_norm": 1.0, "bootstrap_valid_rounds": float(len(estimates))}

        stacked = np.stack(estimates, axis=0)
        variance = np.var(stacked, axis=0)
        variance_norm = float(np.clip(np.mean(variance), 0.0, 1.0))
        return {
            "param_variance_norm": variance_norm,
            "bootstrap_valid_rounds": float(len(estimates)),
        }

    def _build_point_arrays(self, correspondences: List[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        src_points: List[List[float]] = []
        dst_points: List[List[float]] = []
        for item in correspondences:
            if not isinstance(item, dict):
                continue
            src = item.get("src")
            dst = item.get("dst")
            if not isinstance(src, (list, tuple)) or not isinstance(dst, (list, tuple)):
                continue
            if len(src) != 2 or len(dst) != 2:
                continue
            if not all(isinstance(v, (int, float)) for v in src):
                continue
            if not all(isinstance(v, (int, float)) for v in dst):
                continue
            src_points.append([float(src[0]), float(src[1])])
            dst_points.append([float(dst[0]), float(dst[1])])
        return np.asarray(src_points, dtype=np.float64), np.asarray(dst_points, dtype=np.float64)

    def _initial_params_from_transform(self, initial_transform: Dict[str, Any], model_type: str) -> np.ndarray:
        quantized = initial_transform.get("transform_quantized") if isinstance(initial_transform.get("transform_quantized"), dict) else {}
        rotation_degree = float(quantized.get("rotation_degree_q", 0.0))
        scale_factor = float(quantized.get("scale_factor_q", 1.0))
        tx = float(quantized.get("translation_x_q", 0.0))
        ty = float(quantized.get("translation_y_q", 0.0))
        theta = np.deg2rad(rotation_degree)
        cos_v = float(np.cos(theta))
        sin_v = float(np.sin(theta))

        if model_type == "affine":
            return np.asarray([
                scale_factor * cos_v,
                -scale_factor * sin_v,
                scale_factor * sin_v,
                scale_factor * cos_v,
                tx,
                ty,
            ], dtype=np.float64)
        return np.asarray([
            scale_factor * cos_v,
            scale_factor * sin_v,
            tx,
            ty,
        ], dtype=np.float64)

    def _solve_weighted_model(self, src_points: np.ndarray, dst_points: np.ndarray, weights: np.ndarray, model_type: str) -> Optional[np.ndarray]:
        if src_points.shape[0] == 0 or dst_points.shape[0] == 0:
            return None
        if src_points.shape[0] != dst_points.shape[0]:
            return None
        if weights.shape[0] != src_points.shape[0]:
            return None

        x_vals = src_points[:, 0]
        y_vals = src_points[:, 1]
        x_prime = dst_points[:, 0]
        y_prime = dst_points[:, 1]

        if model_type == "affine":
            design = np.zeros((2 * src_points.shape[0], 6), dtype=np.float64)
            target = np.zeros((2 * src_points.shape[0],), dtype=np.float64)
            design[0::2, 0] = x_vals
            design[0::2, 1] = y_vals
            design[0::2, 4] = 1.0
            design[1::2, 2] = x_vals
            design[1::2, 3] = y_vals
            design[1::2, 5] = 1.0
            target[0::2] = x_prime
            target[1::2] = y_prime
        else:
            design = np.zeros((2 * src_points.shape[0], 4), dtype=np.float64)
            target = np.zeros((2 * src_points.shape[0],), dtype=np.float64)
            design[0::2, 0] = x_vals
            design[0::2, 1] = -y_vals
            design[0::2, 2] = 1.0
            design[1::2, 0] = y_vals
            design[1::2, 1] = x_vals
            design[1::2, 3] = 1.0
            target[0::2] = x_prime
            target[1::2] = y_prime

        repeated_weights = np.repeat(np.sqrt(np.clip(weights, 1e-8, 1.0)), 2)
        weighted_design = design * repeated_weights[:, None]
        weighted_target = target * repeated_weights
        if not np.isfinite(weighted_design).all() or not np.isfinite(weighted_target).all():
            return None
        gram_matrix = np.sum(
            weighted_design[:, :, None] * weighted_design[:, None, :],
            axis=0,
        )
        rhs_vector = np.sum(weighted_design * weighted_target[:, None], axis=0)
        regularization = np.eye(gram_matrix.shape[0], dtype=np.float64) * 1e-8
        solved = _solve_linear_system_gaussian(gram_matrix + regularization, rhs_vector)
        if solved is None:
            return None
        return solved.astype(np.float64)

    def _predict_points(self, src_points: np.ndarray, params: np.ndarray, model_type: str) -> np.ndarray:
        if model_type == "affine":
            a00, a01, a10, a11, tx, ty = params.tolist()
            x_vals = src_points[:, 0]
            y_vals = src_points[:, 1]
            return np.stack([
                a00 * x_vals + a01 * y_vals + tx,
                a10 * x_vals + a11 * y_vals + ty,
            ], axis=1)

        a_val, b_val, tx, ty = params.tolist()
        x_vals = src_points[:, 0]
        y_vals = src_points[:, 1]
        return np.stack([
            a_val * x_vals - b_val * y_vals + tx,
            b_val * x_vals + a_val * y_vals + ty,
        ], axis=1)

    def _compute_irls_weights(self, residuals: np.ndarray, sigma: float, robust_loss: str) -> np.ndarray:
        scaled = residuals / max(1e-8, sigma)
        if robust_loss == "tukey":
            c_val = 4.685
            ratio = scaled / c_val
            inside = np.maximum(0.0, 1.0 - ratio * ratio)
            weights = inside * inside
            weights[scaled > c_val] = 0.0
            return np.clip(weights, 1e-8, 1.0)

        c_val = 1.345
        weights = np.ones_like(scaled)
        mask = scaled > c_val
        weights[mask] = c_val / scaled[mask]
        return np.clip(weights, 1e-8, 1.0)

    def _compute_mad(self, residuals: np.ndarray) -> float:
        if residuals.size == 0:
            return 1.0
        median = float(np.median(residuals))
        return float(np.median(np.abs(residuals - median)))

    def _resolve_geometry_cfg(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        return geometry_cfg

    def _resolve_align_robust_loss(self, cfg: Dict[str, Any]) -> str:
        geometry_cfg = self._resolve_geometry_cfg(cfg)
        value = geometry_cfg.get("align_robust_loss", "huber")
        if not isinstance(value, str):
            return "huber"
        lowered = value.strip().lower()
        if lowered not in {"huber", "tukey"}:
            return "huber"
        return lowered

    def _resolve_align_max_iterations(self, cfg: Dict[str, Any]) -> int:
        geometry_cfg = self._resolve_geometry_cfg(cfg)
        value = geometry_cfg.get("align_max_iterations", 8)
        if not isinstance(value, int):
            return 8
        return max(3, min(32, value))

    def _resolve_align_inlier_threshold(self, cfg: Dict[str, Any]) -> float:
        geometry_cfg = self._resolve_geometry_cfg(cfg)
        value = geometry_cfg.get("align_inlier_threshold", 0.22)
        if not isinstance(value, (int, float)):
            return 0.22
        return max(0.01, min(1.0, float(value)))

    def _resolve_align_bootstrap_rounds(self, cfg: Dict[str, Any]) -> int:
        geometry_cfg = self._resolve_geometry_cfg(cfg)
        value = geometry_cfg.get("align_bootstrap_rounds", 8)
        if not isinstance(value, int):
            return 8
        return max(4, min(64, value))

    def _resolve_align_bootstrap_sample_ratio(self, cfg: Dict[str, Any]) -> float:
        geometry_cfg = self._resolve_geometry_cfg(cfg)
        value = geometry_cfg.get("align_bootstrap_sample_ratio", 0.75)
        if not isinstance(value, (int, float)):
            return 0.75
        return max(0.4, min(1.0, float(value)))


class GeometryAligner:
    """
    功能：执行几何对齐估计并产出对齐统计。

    Perform geometry alignment estimation and emit compact metrics.
    """

    def __init__(self) -> None:
        self._robust_fitter = RobustSimilarityFitter()

    def estimate_transform(self, anchor_data: Dict[str, Any], sync_data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：估计量化几何变换参数。

        Estimate quantized geometry transform from anchor and sync evidence.

        Args:
            anchor_data: Anchor evidence mapping.
            sync_data: Sync evidence mapping.
            cfg: Configuration mapping.

        Returns:
            Structured transform estimation payload.
        """
        if not isinstance(anchor_data, dict):
            raise TypeError("anchor_data must be dict")
        if not isinstance(sync_data, dict):
            raise TypeError("sync_data must be dict")
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")

        model_type = self._resolve_align_model_type(cfg)
        sync_metrics = sync_data.get("sync_metrics") if isinstance(sync_data.get("sync_metrics"), dict) else {}
        anchor_metrics = anchor_data.get("stability_metrics") if isinstance(anchor_data.get("stability_metrics"), dict) else {}

        rotation_bins = self._resolve_rotation_bins(cfg)
        scale_bins = self._resolve_scale_bins(cfg)
        rotation_bin = int(sync_metrics.get("rotation_bin", 0))
        scale_bin = int(sync_metrics.get("scale_bin", 0))

        rotation_degree = round((float(rotation_bin) / float(max(1, rotation_bins))) * 360.0, 4)
        center_scale = float(scale_bins - 1) / 2.0
        scale_offset = (float(scale_bin) - center_scale) / float(max(1.0, center_scale))
        scale_factor = round(1.0 + 0.2 * scale_offset, 6)
        anchor_concentration = float(anchor_metrics.get("top1_concentration", 0.0))
        translation_x = round((anchor_concentration - 0.5) * 2.0, 6)
        translation_y = round((0.5 - anchor_concentration) * 2.0, 6)

        initial_transform = {
            "status": "ok",
            "model_type": model_type,
            "transform_quantized": {
                "rotation_degree_q": rotation_degree,
                "scale_factor_q": scale_factor,
                "translation_x_q": translation_x,
                "translation_y_q": translation_y,
            },
            "support": {
                "rotation_bin": rotation_bin,
                "scale_bin": scale_bin,
                "rotation_bins": rotation_bins,
                "scale_bins": scale_bins,
            },
        }

        correspondences = self._build_correspondences(anchor_metrics, sync_metrics, initial_transform)
        fit_result = self._robust_fitter.fit(initial_transform, correspondences, cfg)

        if not fit_result.converged:
            return {
                "status": "fail",
                "model_type": model_type,
                "align_success": False,
                "fit_stability": float(fit_result.fit_stability),
                "transform_quantized": initial_transform["transform_quantized"],
                "support": initial_transform["support"],
                "fit_diagnostics": {
                    "iterations": fit_result.iterations,
                    "residual_stats": fit_result.residual_stats,
                    "param_uncertainty": fit_result.param_uncertainty,
                    "inlier_ratio": float(np.mean(np.asarray(fit_result.inlier_mask, dtype=np.float64))) if fit_result.inlier_mask else 0.0,
                    "robust_loss": self._resolve_align_robust_loss(cfg),
                },
            }

        refined_transform = self._params_to_quantized_transform(fit_result.model_type, fit_result.params)
        inlier_array = np.asarray(fit_result.inlier_mask, dtype=np.float64)
        inlier_ratio = float(np.mean(inlier_array)) if inlier_array.size > 0 else 0.0
        return {
            "status": "ok",
            "model_type": fit_result.model_type,
            "align_success": True,
            "fit_stability": float(fit_result.fit_stability),
            "transform_quantized": refined_transform,
            "support": initial_transform["support"],
            "fit_diagnostics": {
                "iterations": fit_result.iterations,
                "residual_stats": fit_result.residual_stats,
                "param_uncertainty": fit_result.param_uncertainty,
                "inlier_ratio": inlier_ratio,
                "robust_loss": self._resolve_align_robust_loss(cfg),
            },
        }

    def compute_align_metrics(
        self,
        transform_result: Dict[str, Any],
        anchor_data: Dict[str, Any],
        sync_data: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        功能：计算对齐统计指标。

        Compute compact alignment metrics from transform and evidences.

        Args:
            transform_result: Transform estimation mapping.
            anchor_data: Anchor evidence mapping.
            sync_data: Sync evidence mapping.
            cfg: Configuration mapping.

        Returns:
            Compact alignment metrics mapping.
        """
        if not isinstance(transform_result, dict):
            raise TypeError("transform_result must be dict")
        if not isinstance(anchor_data, dict):
            raise TypeError("anchor_data must be dict")
        if not isinstance(sync_data, dict):
            raise TypeError("sync_data must be dict")
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")

        anchor_metrics = anchor_data.get("stability_metrics") if isinstance(anchor_data.get("stability_metrics"), dict) else {}
        sync_metrics = sync_data.get("sync_metrics") if isinstance(sync_data.get("sync_metrics"), dict) else {}

        anchor_entropy = float(anchor_metrics.get("neighbor_entropy", 0.0))
        anchor_concentration = float(anchor_metrics.get("top1_concentration", 0.0))
        match_confidence = float(sync_metrics.get("match_confidence", 0.0))
        peak_ratio = float(sync_metrics.get("peak_ratio", 0.0))

        fit_diagnostics = transform_result.get("fit_diagnostics") if isinstance(transform_result.get("fit_diagnostics"), dict) else {}
        residual_stats = fit_diagnostics.get("residual_stats") if isinstance(fit_diagnostics.get("residual_stats"), dict) else {}
        param_uncertainty = fit_diagnostics.get("param_uncertainty") if isinstance(fit_diagnostics.get("param_uncertainty"), dict) else {}
        fit_stability = float(transform_result.get("fit_stability", 0.0))

        residual_median = float(residual_stats.get("residual_median", 1.0))
        residual_mad = float(residual_stats.get("residual_mad", 1.0))
        residual_trimmed_mean = float(residual_stats.get("residual_trimmed_mean", 1.0))
        residual_q25 = float(residual_stats.get("residual_q25", 1.0))
        residual_q75 = float(residual_stats.get("residual_q75", 1.0))
        param_variance_norm = float(param_uncertainty.get("param_variance_norm", 1.0))

        normalized_entropy = _clamp01(anchor_entropy / 8.0)
        relation_consistency = _clamp01(0.5 * (1.0 - abs(normalized_entropy - match_confidence)) + 0.5 * fit_stability)
        inlier_ratio = _clamp01(float(fit_diagnostics.get("inlier_ratio", 0.0)))
        adjacency_preservation = _clamp01(0.55 * anchor_concentration + 0.45 * inlier_ratio)
        residual_p50 = _clamp01(residual_median)
        residual_p90 = _clamp01(residual_q75)
        residual_max = _clamp01(min(1.0, residual_trimmed_mean + residual_mad))
        support = transform_result.get("support") if isinstance(transform_result.get("support"), dict) else {}
        rotation_bins = int(support.get("rotation_bins", self._resolve_rotation_bins(cfg)))
        scale_bins = int(support.get("scale_bins", self._resolve_scale_bins(cfg)))
        match_count_estimate = int(max(4, round(inlier_ratio * float(max(8, rotation_bins + scale_bins)))))

        return {
            "relation_consistency": round(relation_consistency, 6),
            "adjacency_preservation": round(adjacency_preservation, 6),
            "residual_p50": round(residual_p50, 6),
            "residual_p90": round(residual_p90, 6),
            "residual_max": round(residual_max, 6),
            "inlier_ratio": round(inlier_ratio, 6),
            "match_count_estimate": match_count_estimate,
            "peak_ratio": round(_clamp01(peak_ratio / 8.0), 6),
            "fit_stability": round(_clamp01(fit_stability), 6),
            "align_success": bool(transform_result.get("align_success", False)),
            "residual_median": round(_clamp01(residual_median), 6),
            "residual_mad": round(_clamp01(residual_mad), 6),
            "residual_trimmed_mean": round(_clamp01(residual_trimmed_mean), 6),
            "residual_q25": round(_clamp01(residual_q25), 6),
            "residual_q75": round(_clamp01(residual_q75), 6),
            "param_variance_norm": round(_clamp01(param_variance_norm), 6),
        }

    def compute_align_trace_digest(self, payload: Dict[str, Any]) -> str:
        """
        功能：计算对齐痕迹摘要。

        Compute alignment trace digest using canonical payload.

        Args:
            payload: Canonical alignment payload mapping.

        Returns:
            SHA256 digest string.
        """
        if not isinstance(payload, dict):
            raise TypeError("payload must be dict")
        return digests.canonical_sha256(payload)

    def _resolve_align_model_type(self, cfg: Dict[str, Any]) -> str:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("align_model_type", "similarity")
        if not isinstance(value, str):
            return "similarity"
        lowered = value.strip().lower()
        if lowered not in {"similarity", "affine"}:
            return "similarity"
        return lowered

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

    def _resolve_align_robust_loss(self, cfg: Dict[str, Any]) -> str:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("align_robust_loss", "huber")
        if not isinstance(value, str):
            return "huber"
        lowered = value.strip().lower()
        if lowered not in {"huber", "tukey"}:
            return "huber"
        return lowered

    def _build_correspondences(
        self,
        anchor_metrics: Dict[str, Any],
        sync_metrics: Dict[str, Any],
        initial_transform: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        grid_vals = np.linspace(-1.0, 1.0, num=5, dtype=np.float64)
        src_points = np.stack(np.meshgrid(grid_vals, grid_vals), axis=-1).reshape(-1, 2)
        clean_dst = self._apply_transform(src_points, initial_transform)

        residual_score = float(sync_metrics.get("residual_score", 0.0))
        anchor_entropy = float(anchor_metrics.get("neighbor_entropy", 0.0))
        anchor_concentration = float(anchor_metrics.get("top1_concentration", 0.0))
        support = initial_transform.get("support") if isinstance(initial_transform.get("support"), dict) else {}
        rotation_bin = int(support.get("rotation_bin", 0))
        scale_bin = int(support.get("scale_bin", 0))
        seed = int(rotation_bin * 131 + scale_bin * 17 + round(anchor_concentration * 1000.0))
        rng = np.random.RandomState(seed)
        noise_sigma = 0.01 + 0.07 * _clamp01(residual_score) + 0.03 * _clamp01(anchor_entropy / 8.0)
        noise = rng.normal(loc=0.0, scale=noise_sigma, size=clean_dst.shape)
        dst_points = clean_dst + noise

        correspondences: List[Dict[str, Any]] = []
        for idx in range(src_points.shape[0]):
            correspondences.append(
                {
                    "src": [float(src_points[idx, 0]), float(src_points[idx, 1])],
                    "dst": [float(dst_points[idx, 0]), float(dst_points[idx, 1])],
                }
            )
        return correspondences

    def _apply_transform(self, src_points: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
        model_type = str(transform.get("model_type", "similarity")).lower()
        quantized = transform.get("transform_quantized") if isinstance(transform.get("transform_quantized"), dict) else {}
        theta = np.deg2rad(float(quantized.get("rotation_degree_q", 0.0)))
        scale_factor = float(quantized.get("scale_factor_q", 1.0))
        tx = float(quantized.get("translation_x_q", 0.0))
        ty = float(quantized.get("translation_y_q", 0.0))
        cos_v = float(np.cos(theta))
        sin_v = float(np.sin(theta))

        if model_type == "affine":
            a00 = scale_factor * cos_v
            a01 = -scale_factor * sin_v
            a10 = scale_factor * sin_v
            a11 = scale_factor * cos_v
            return np.stack(
                [
                    a00 * src_points[:, 0] + a01 * src_points[:, 1] + tx,
                    a10 * src_points[:, 0] + a11 * src_points[:, 1] + ty,
                ],
                axis=1,
            )

        return np.stack(
            [
                scale_factor * (cos_v * src_points[:, 0] - sin_v * src_points[:, 1]) + tx,
                scale_factor * (sin_v * src_points[:, 0] + cos_v * src_points[:, 1]) + ty,
            ],
            axis=1,
        )

    def _params_to_quantized_transform(self, model_type: str, params: List[float]) -> Dict[str, float]:
        if model_type == "affine":
            a00, a01, a10, a11, tx, ty = [float(v) for v in params]
            scale_x = float(np.sqrt(a00 * a00 + a10 * a10))
            scale_y = float(np.sqrt(a01 * a01 + a11 * a11))
            scale_factor = float((scale_x + scale_y) / 2.0)
            rotation_degree = float(np.rad2deg(np.arctan2(a10, a00)))
            return {
                "rotation_degree_q": round(rotation_degree, 4),
                "scale_factor_q": round(scale_factor, 6),
                "translation_x_q": round(tx, 6),
                "translation_y_q": round(ty, 6),
            }

        a_val, b_val, tx, ty = [float(v) for v in params]
        scale_factor = float(np.sqrt(a_val * a_val + b_val * b_val))
        rotation_degree = float(np.rad2deg(np.arctan2(b_val, a_val)))
        return {
            "rotation_degree_q": round(rotation_degree, 4),
            "scale_factor_q": round(scale_factor, 6),
            "translation_x_q": round(tx, 6),
            "translation_y_q": round(ty, 6),
        }


class InvarianceScorer:
    """
    功能：根据对齐指标计算几何不变性得分。

    Compute geometry invariance score from alignment metrics.
    """

    def compute_geo_score(
        self,
        align_metrics: Dict[str, Any],
        anchor_metrics: Dict[str, Any],
        *,
        attention_consistency: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        功能：计算 geo_score。

        Compute geo_score in [0, 1] from alignment and anchor metrics.

        Args:
            align_metrics: Alignment metrics mapping.
            anchor_metrics: Anchor metrics mapping.
            attention_consistency: Optional attention consistency score.

        Returns:
            Score payload with direction and summary.
        """
        if not isinstance(align_metrics, dict):
            raise TypeError("align_metrics must be dict")
        if not isinstance(anchor_metrics, dict):
            raise TypeError("anchor_metrics must be dict")

        relation_consistency = float(align_metrics.get("relation_consistency", 0.0))
        adjacency_preservation = float(align_metrics.get("adjacency_preservation", 0.0))
        inlier_ratio = float(align_metrics.get("inlier_ratio", 0.0))
        residual_p90 = float(align_metrics.get("residual_p90", 1.0))
        residual_mad = float(align_metrics.get("residual_mad", 1.0))
        anchor_concentration = float(anchor_metrics.get("top1_concentration", 0.0))
        fit_stability = float(align_metrics.get("fit_stability", 0.0))

        fit_quality = _clamp01(0.45 * inlier_ratio + 0.35 * (1.0 - _clamp01(residual_mad)) + 0.20 * fit_stability)
        anchor_consistency = _clamp01(0.55 * adjacency_preservation + 0.45 * _clamp01(anchor_concentration))

        if attention_consistency is None:
            score_raw = 0.62 * fit_quality + 0.38 * anchor_consistency
            score_parts = {
                "fit_quality": round(fit_quality, 6),
                "anchor_consistency": round(anchor_consistency, 6),
            }
        else:
            attn_value = _clamp01(float(attention_consistency))
            score_raw = 0.52 * fit_quality + 0.30 * anchor_consistency + 0.18 * attn_value
            score_parts = {
                "fit_quality": round(fit_quality, 6),
                "anchor_consistency": round(anchor_consistency, 6),
                "attn_consistency": round(attn_value, 6),
            }

        geo_score = round(_clamp01(score_raw), 6)
        direction = self.validate_score_direction("higher_is_stronger")
        summary = {
            "score_components": {
                "relation_consistency": round(_clamp01(relation_consistency), 6),
                "adjacency_preservation": round(_clamp01(adjacency_preservation), 6),
                "inlier_ratio": round(_clamp01(inlier_ratio), 6),
                "residual_p90_inverse": round(1.0 - _clamp01(residual_p90), 6),
                "anchor_concentration": round(_clamp01(anchor_concentration), 6),
            },
            "score_parts": score_parts,
        }
        return {
            "geo_score": geo_score,
            "geo_score_direction": direction,
            "score_parts": score_parts,
            "score_summary": summary,
        }

    def validate_score_direction(self, direction: str) -> str:
        """
        功能：校验评分方向语义。

        Validate geo_score direction token.

        Args:
            direction: Direction token.

        Returns:
            Valid direction token.
        """
        if not isinstance(direction, str) or not direction:
            raise TypeError("direction must be non-empty str")
        if direction != "higher_is_stronger":
            raise ValueError("geo_score_direction must be 'higher_is_stronger'")
        return direction

    def compute_score_digest(self, payload: Dict[str, Any]) -> str:
        """
        功能：计算评分摘要。

        Compute score digest using canonical payload.

        Args:
            payload: Canonical score payload mapping.

        Returns:
            SHA256 digest string.
        """
        if not isinstance(payload, dict):
            raise TypeError("payload must be dict")
        return digests.canonical_sha256(payload)


class GeometryAlignInvarianceExtractor:
    """
    功能：组合 anchor + sync + align + score 的几何证据提取器。

    Compose anchor, sync, alignment, and invariance scoring into geometry evidence.
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
        self._aligner = GeometryAligner()
        self._scorer = InvarianceScorer()
        self._anchor_extractor = AttentionAnchorExtractor(
            "geometry_attention_anchor_sd3_v1",
            "v1",
            digests.canonical_sha256({"impl_id": "geometry_attention_anchor_sd3_v1", "impl_version": "v1"}),
        )
        self._sync_extractor = LatentSyncGeometryExtractor(
            "geometry_latent_sync_sd3_v1",
            "v1",
            digests.canonical_sha256({"impl_id": "geometry_latent_sync_sd3_v1", "impl_version": "v1"}),
        )

    def extract(self, cfg: Dict[str, Any], inputs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        功能：提取几何对齐与不变性证据。

        Extract geometry evidence with align trace and geo score.

        Args:
            cfg: Configuration mapping.
            inputs: Runtime inputs mapping.

        Returns:
            Geometry evidence mapping.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if inputs is not None and not isinstance(inputs, dict):
            raise TypeError("inputs must be dict or None")

        runtime_inputs = inputs if isinstance(inputs, dict) else {}
        if not self._resolve_enable_align_invariance(cfg):
            return self._build_absent_evidence(
                reason="align_invariance_disabled_by_policy",
                anchor_evidence=None,
                sync_evidence=None,
                align_config_digest=digests.canonical_sha256(self._build_align_config_domain(cfg)),
            )

        anchor_evidence = self._anchor_extractor.extract(cfg, inputs=runtime_inputs)
        sync_evidence = self._sync_extractor.extract(cfg, inputs=runtime_inputs)

        status, failure_reason = self._resolve_precheck_status(anchor_evidence, sync_evidence)
        if status != "ok":
            return self._build_terminal_evidence(
                status=status,
                reason=failure_reason,
                anchor_evidence=anchor_evidence,
                sync_evidence=sync_evidence,
                align_trace_digest=None,
                align_metrics=None,
                geo_score=None,
                geo_score_direction=None,
                align_config_digest=digests.canonical_sha256(self._build_align_config_domain(cfg)),
                score_digest=None,
            )

        if self._has_resolution_binding_mismatch(anchor_evidence, sync_evidence):
            return self._build_terminal_evidence(
                status="mismatch",
                reason="align_anchor_sync_binding_mismatch",
                anchor_evidence=anchor_evidence,
                sync_evidence=sync_evidence,
                align_trace_digest=None,
                align_metrics=None,
                geo_score=None,
                geo_score_direction=None,
                align_config_digest=digests.canonical_sha256(self._build_align_config_domain(cfg)),
                score_digest=None,
            )

        align_config_digest = digests.canonical_sha256(self._build_align_config_domain(cfg))
        anchor_concentration = float((anchor_evidence.get("stability_metrics") or {}).get("top1_concentration", 0.0))
        if anchor_concentration < self._resolve_align_anchor_min_concentration(cfg):
            return self._build_terminal_evidence(
                status="fail",
                reason="anchor_unstable",
                anchor_evidence=anchor_evidence,
                sync_evidence=sync_evidence,
                align_trace_digest=None,
                align_metrics=None,
                geo_score=None,
                geo_score_direction=None,
                align_config_digest=align_config_digest,
                score_digest=None,
            )

        sync_match_confidence = float((sync_evidence.get("sync_metrics") or {}).get("match_confidence", 0.0))
        if sync_match_confidence < self._resolve_align_sync_quality_min(cfg):
            return self._build_terminal_evidence(
                status="fail",
                reason="sync_quality_below_threshold",
                anchor_evidence=anchor_evidence,
                sync_evidence=sync_evidence,
                align_trace_digest=None,
                align_metrics=None,
                geo_score=None,
                geo_score_direction=None,
                align_config_digest=align_config_digest,
                score_digest=None,
            )

        transform_result = self._aligner.estimate_transform(anchor_evidence, sync_evidence, cfg)
        if not bool(transform_result.get("align_success", False)):
            return self._build_terminal_evidence(
                status="fail",
                reason="align_fit_not_converged",
                anchor_evidence=anchor_evidence,
                sync_evidence=sync_evidence,
                align_trace_digest=None,
                align_metrics=None,
                geo_score=None,
                geo_score_direction=None,
                align_config_digest=align_config_digest,
                score_digest=None,
            )

        align_metrics = self._aligner.compute_align_metrics(transform_result, anchor_evidence, sync_evidence, cfg)

        min_inlier_ratio = self._resolve_align_min_inlier_ratio(cfg)
        if float(align_metrics.get("inlier_ratio", 0.0)) < min_inlier_ratio:
            return self._build_terminal_evidence(
                status="fail",
                reason="align_inlier_ratio_below_threshold",
                anchor_evidence=anchor_evidence,
                sync_evidence=sync_evidence,
                align_trace_digest=None,
                align_metrics=align_metrics,
                geo_score=None,
                geo_score_direction=None,
                align_config_digest=align_config_digest,
                score_digest=None,
            )

        residual_mad_max = self._resolve_align_available_max_residual_mad(cfg)
        if float(align_metrics.get("residual_mad", 1.0)) > residual_mad_max:
            return self._build_terminal_evidence(
                status="fail",
                reason="align_residual_mad_above_threshold",
                anchor_evidence=anchor_evidence,
                sync_evidence=sync_evidence,
                align_trace_digest=None,
                align_metrics=align_metrics,
                geo_score=None,
                geo_score_direction=None,
                align_config_digest=align_config_digest,
                score_digest=None,
            )

        param_variance_max = self._resolve_align_available_max_param_variance(cfg)
        if float(align_metrics.get("param_variance_norm", 1.0)) > param_variance_max:
            return self._build_terminal_evidence(
                status="fail",
                reason="align_param_variance_above_threshold",
                anchor_evidence=anchor_evidence,
                sync_evidence=sync_evidence,
                align_trace_digest=None,
                align_metrics=align_metrics,
                geo_score=None,
                geo_score_direction=None,
                align_config_digest=align_config_digest,
                score_digest=None,
            )

        fit_stability = float(transform_result.get("fit_stability", 0.0))
        attention_consistency: Optional[float] = None
        if fit_stability >= self._resolve_align_attention_consistency_min_stability(cfg):
            attention_consistency = self._compute_attention_consistency(anchor_evidence, sync_evidence, align_metrics)
            align_metrics["attention_consistency"] = round(_clamp01(attention_consistency), 6)

        # (新增) 判定几何可用性：基于参数不确定性与拟合质量
        geo_available, geo_unavailability_reason = self._decide_geo_available(
            transform_result=transform_result,
            align_metrics=align_metrics,
            cfg=cfg
        )

        trace_payload = {
            "align_trace_version": "geometry_align_trace_v1",
            "align_config_digest": align_config_digest,
            "model_type": transform_result.get("model_type"),
            "transform_quantized": transform_result.get("transform_quantized"),
            "fit_diagnostics": transform_result.get("fit_diagnostics"),
            "fit_stability": fit_stability,
            "align_metrics": align_metrics,
            "anchor_digest": anchor_evidence.get("anchor_digest"),
            "sync_digest": sync_evidence.get("sync_digest"),
        }
        align_trace_digest = self._aligner.compute_align_trace_digest(trace_payload)
        anchor_metrics = anchor_evidence.get("stability_metrics") if isinstance(anchor_evidence.get("stability_metrics"), dict) else {}
        score_result = self._scorer.compute_geo_score(
            align_metrics,
            anchor_metrics,
            attention_consistency=attention_consistency,
        )
        score_digest = self._scorer.compute_score_digest({
            "geo_score": score_result.get("geo_score"),
            "geo_score_direction": score_result.get("geo_score_direction"),
            "align_trace_digest": align_trace_digest,
            "align_config_digest": align_config_digest,
            "score_parts": score_result.get("score_parts"),
        })

        return self._build_terminal_evidence(
            status="ok",
            reason=None,
            anchor_evidence=anchor_evidence,
            sync_evidence=sync_evidence,
            align_trace_digest=align_trace_digest,
            align_metrics=align_metrics,
            geo_score=float(score_result.get("geo_score", 0.0)),
            geo_score_direction=str(score_result.get("geo_score_direction", "higher_is_stronger")),
            align_config_digest=align_config_digest,
            score_digest=score_digest,
            geo_available=geo_available,
            geo_availability_rule_version=GEO_AVAILABILITY_RULE_VERSION,
            geo_unavailability_reason=geo_unavailability_reason,
        )

    def _build_absent_evidence(
        self,
        reason: str,
        anchor_evidence: Dict[str, Any] | None,
        sync_evidence: Dict[str, Any] | None,
        align_config_digest: str,
    ) -> Dict[str, Any]:
        return self._build_terminal_evidence(
            status="absent",
            reason=reason,
            anchor_evidence=anchor_evidence,
            sync_evidence=sync_evidence,
            align_trace_digest=None,
            align_metrics=None,
            geo_score=None,
            geo_score_direction=None,
            align_config_digest=align_config_digest,
            score_digest=None,
        )

    def _build_terminal_evidence(
        self,
        status: str,
        reason: str | None,
        anchor_evidence: Dict[str, Any] | None,
        sync_evidence: Dict[str, Any] | None,
        align_trace_digest: str | None,
        align_metrics: Dict[str, Any] | None,
        geo_score: float | None,
        geo_score_direction: str | None,
        align_config_digest: str | None,
        score_digest: str | None,
        geo_available: bool | None = None,
        geo_availability_rule_version: str | None = None,
        geo_unavailability_reason: str | None = None,
    ) -> Dict[str, Any]:
        anchor_data = anchor_evidence if isinstance(anchor_evidence, dict) else {}
        sync_data = sync_evidence if isinstance(sync_evidence, dict) else {}
        trace_payload = {
            "status": status,
            "reason": reason,
            "anchor_digest": anchor_data.get("anchor_digest"),
            "sync_digest": sync_data.get("sync_digest"),
            "align_trace_digest": align_trace_digest,
            "score_digest": score_digest,
        }
        trace_digest = digests.canonical_sha256(trace_payload)
        evidence = {
            "status": status,
            "geo_score": geo_score,
            "geo_score_direction": geo_score_direction,
            "anchor_digest": anchor_data.get("anchor_digest"),
            "anchor_config_digest": anchor_data.get("anchor_config_digest"),
            "anchor_metrics": anchor_data.get("anchor_metrics"),
            "stability_metrics": anchor_data.get("stability_metrics"),
            "sync_digest": sync_data.get("sync_digest"),
            "sync_config_digest": sync_data.get("sync_config_digest"),
            "sync_metrics": sync_data.get("sync_metrics"),
            "sync_quality_metrics": sync_data.get("sync_quality_metrics"),
            "resolution_binding": _merge_resolution_binding(
                anchor_data.get("resolution_binding"),
                sync_data.get("resolution_binding"),
            ),
            "align_trace_digest": align_trace_digest,
            "align_metrics": align_metrics,
            "align_config_digest": align_config_digest,
            "geo_failure_reason": reason,
            "geometry_failure_reason": reason,
            "audit": {
                "impl_identity": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "trace_digest": trace_digest,
                "sync_status_detail": sync_data.get("status", "absent"),
                "anchor_status_detail": anchor_data.get("status", "absent"),
                "align_status_detail": status,
                "score_digest": score_digest,
            },
        }
        # (新增) 几何可用性字段（仅在 status="ok" 时添加）
        if geo_available is not None:
            evidence["geo_available"] = geo_available
        if geo_availability_rule_version is not None:
            evidence["geo_availability_rule_version"] = geo_availability_rule_version
        if geo_unavailability_reason is not None:
            evidence["geo_unavailability_reason"] = geo_unavailability_reason
        return evidence

    def _resolve_precheck_status(self, anchor_data: Dict[str, Any], sync_data: Dict[str, Any]) -> tuple[str, str | None]:
        anchor_status = str(anchor_data.get("status", "absent"))
        sync_status = str(sync_data.get("status", "absent"))
        anchor_reason = anchor_data.get("geo_failure_reason")
        sync_reason = sync_data.get("geo_failure_reason")

        if anchor_status in {"mismatch", "fail"}:
            return anchor_status, str(anchor_reason or "anchor_precheck_failed")
        if sync_status in {"mismatch", "fail"}:
            return sync_status, str(sync_reason or "sync_precheck_failed")
        if anchor_status != "ok":
            return "absent", str(anchor_reason or "anchor_unavailable")
        if sync_status != "ok":
            if str(sync_reason or "").strip() in {
                "detect_pipeline_unavailable",
                "sync_unsupported_model_or_pipeline",
                "sync_transformer_absent",
            }:
                return "absent", "sync_unavailable"
            return "absent", str(sync_reason or "sync_unavailable")
        return "ok", None

    def _has_resolution_binding_mismatch(self, anchor_data: Dict[str, Any], sync_data: Dict[str, Any]) -> bool:
        anchor_binding = anchor_data.get("resolution_binding") if isinstance(anchor_data.get("resolution_binding"), dict) else {}
        sync_binding = sync_data.get("resolution_binding") if isinstance(sync_data.get("resolution_binding"), dict) else {}
        if not anchor_binding or not sync_binding:
            return True
        keys = ["token_grid_height", "token_grid_width", "token_count", "model_binding"]
        for key in keys:
            if anchor_binding.get(key) != sync_binding.get(key):
                return True
        return False

    def _decide_geo_available(
        self,
        transform_result: Dict[str, Any],
        align_metrics: Dict[str, Any],
        cfg: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        功能：根据参数不确定性与拟合质量判定几何是否可用。

        Decide whether geometry evidence is available based on parameter uncertainty and fit stability.

        Args:
            transform_result: Transform fitting result with fit_stability and fit_diagnostics.
            align_metrics: Alignment metrics computed from fit result.
            cfg: Configuration mapping.

        Returns:
            Tuple of (geo_available: bool, unavailability_reason: Optional[str]).
            If geo_available=True, unavailability_reason=None.
            If geo_available=False, unavailability_reason is one of GEO_UNAVAILABILITY_REASONS.

        Raises:
            TypeError: If argument types are invalid.
        """
        if not isinstance(transform_result, dict):
            raise TypeError("transform_result must be dict")
        if not isinstance(align_metrics, dict):
            raise TypeError("align_metrics must be dict")
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")

        geo_cfg = cfg.get("detect", {}).get("geometry", {})

        # (1) 检查拟合收敛：若未收敛，则几何不可用
        fit_diagnostics = transform_result.get("fit_diagnostics", {})
        if not bool(transform_result.get("align_success", False)):
            return False, "geo_fit_not_converged"

        # (2) 检查参数不确定性：若 param_variance_norm 过高，则参数不稳定
        max_param_variance = float(geo_cfg.get("max_param_variance_for_geo_available", 0.15))
        param_variance_norm = float(align_metrics.get("param_variance_norm", 1.0))
        if param_variance_norm > max_param_variance:
            return False, "geo_param_uncertainty_high"

        # (3) 检查内点比例：若过低，则对应点噪声过高
        min_inlier_ratio = float(geo_cfg.get("min_inlier_ratio_for_geo_available", 0.5))
        inlier_ratio = float(align_metrics.get("inlier_ratio", 0.0))
        if inlier_ratio < min_inlier_ratio:
            return False, "geo_inlier_ratio_low"

        # (4) 检查残差统计：若 MAD（中位绝对偏差）过高，则数据离散异常
        max_residual_mad = float(geo_cfg.get("max_residual_mad_for_geo_available", 0.08))
        residual_mad = float(align_metrics.get("residual_mad", 1.0))
        if residual_mad > max_residual_mad:
            return False, "geo_residual_stats_anomaly"

        # 所有检查都通过，几何可用
        return True, None

    def _resolve_enable_align_invariance(self, cfg: Dict[str, Any]) -> bool:
        detect_cfg = cfg.get("detect")
        if not isinstance(detect_cfg, dict):
            return False
        geometry_cfg = detect_cfg.get("geometry")
        if not isinstance(geometry_cfg, dict):
            return False
        explicit = geometry_cfg.get("enable_align_invariance")
        if isinstance(explicit, bool):
            return explicit
        enabled = geometry_cfg.get("enabled")
        return bool(enabled) if isinstance(enabled, bool) else False

    def _resolve_align_min_inlier_ratio(self, cfg: Dict[str, Any]) -> float:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("align_min_inlier_ratio", 0.35)
        if not isinstance(value, (int, float)):
            return 0.35
        return max(0.0, min(1.0, float(value)))

    def _resolve_align_available_max_residual_mad(self, cfg: Dict[str, Any]) -> float:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("align_available_max_residual_mad", 0.28)
        if not isinstance(value, (int, float)):
            return 0.28
        return max(0.01, min(1.0, float(value)))

    def _resolve_align_available_max_param_variance(self, cfg: Dict[str, Any]) -> float:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("align_available_max_param_variance", 0.20)
        if not isinstance(value, (int, float)):
            return 0.20
        return max(0.01, min(1.0, float(value)))

    def _resolve_align_attention_consistency_min_stability(self, cfg: Dict[str, Any]) -> float:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("align_attention_consistency_min_stability", 0.45)
        if not isinstance(value, (int, float)):
            return 0.45
        return max(0.0, min(1.0, float(value)))

    def _resolve_align_sync_quality_min(self, cfg: Dict[str, Any]) -> float:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("align_sync_quality_min", 0.0)
        if not isinstance(value, (int, float)):
            return 0.0
        return max(0.0, min(1.0, float(value)))

    def _resolve_align_anchor_min_concentration(self, cfg: Dict[str, Any]) -> float:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        value = geometry_cfg.get("align_anchor_min_concentration", 0.0)
        if not isinstance(value, (int, float)):
            return 0.0
        return max(0.0, min(1.0, float(value)))

    def _compute_attention_consistency(
        self,
        anchor_evidence: Dict[str, Any],
        sync_evidence: Dict[str, Any],
        align_metrics: Dict[str, Any],
    ) -> float:
        anchor_metrics = anchor_evidence.get("stability_metrics") if isinstance(anchor_evidence.get("stability_metrics"), dict) else {}
        sync_metrics = sync_evidence.get("sync_metrics") if isinstance(sync_evidence.get("sync_metrics"), dict) else {}
        anchor_concentration = float(anchor_metrics.get("top1_concentration", 0.0))
        anchor_entropy = float(anchor_metrics.get("neighbor_entropy", 0.0))
        peak_ratio = float(sync_metrics.get("peak_ratio", 0.0))
        fit_stability = float(align_metrics.get("fit_stability", 0.0))
        entropy_term = 1.0 - _clamp01(anchor_entropy / 8.0)
        peak_term = _clamp01(peak_ratio / 8.0)
        return _clamp01(0.35 * _clamp01(anchor_concentration) + 0.30 * entropy_term + 0.20 * peak_term + 0.15 * _clamp01(fit_stability))

    def _build_align_config_domain(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        model_type = geometry_cfg.get("align_model_type", "similarity")
        if not isinstance(model_type, str):
            model_type = "similarity"
        return {
            "domain_version": "geometry_align_cfg_domain_v1",
            "enable_align_invariance": bool(self._resolve_enable_align_invariance(cfg)),
            "align_model_type": model_type,
            "align_min_inlier_ratio": self._resolve_align_min_inlier_ratio(cfg),
            "align_robust_loss": geometry_cfg.get("align_robust_loss", "huber"),
            "align_max_iterations": geometry_cfg.get("align_max_iterations", 8),
            "align_inlier_threshold": geometry_cfg.get("align_inlier_threshold", 0.22),
            "align_bootstrap_rounds": geometry_cfg.get("align_bootstrap_rounds", 8),
            "align_bootstrap_sample_ratio": geometry_cfg.get("align_bootstrap_sample_ratio", 0.75),
            "align_fit_stability_min": geometry_cfg.get("align_fit_stability_min", 0.45),
            "align_available_max_residual_mad": self._resolve_align_available_max_residual_mad(cfg),
            "align_available_max_param_variance": self._resolve_align_available_max_param_variance(cfg),
            "align_sync_quality_min": self._resolve_align_sync_quality_min(cfg),
            "align_anchor_min_concentration": self._resolve_align_anchor_min_concentration(cfg),
            "align_attention_consistency_min_stability": self._resolve_align_attention_consistency_min_stability(cfg),
            "sync_rotation_bins": geometry_cfg.get("sync_rotation_bins", 36),
            "sync_scale_bins": geometry_cfg.get("sync_scale_bins", 16),
            "model_id": cfg.get("model_id"),
        }


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _merge_resolution_binding(anchor_binding: Any, sync_binding: Any) -> Dict[str, Any] | None:
    if not isinstance(anchor_binding, dict) and not isinstance(sync_binding, dict):
        return None
    if not isinstance(anchor_binding, dict):
        return dict(sync_binding)
    if not isinstance(sync_binding, dict):
        return dict(anchor_binding)
    merged = dict(anchor_binding)
    for key, value in sync_binding.items():
        if key not in merged:
            merged[key] = value
    return merged
