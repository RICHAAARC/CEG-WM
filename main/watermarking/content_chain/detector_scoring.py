"""
内容检测器分数计算扩展模块

功能说明：
- 实现 LF/HF 子空间分数提取逻辑。
- 实现 plan_digest 一致性校验与 mismatch/absent 语义。
- 实现 content_score 的固定组合规则。
- 支持消融：HF absent 不影响 LF。

Module type: Core innovation module
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from main.core import digests
from main.watermarking.content_chain import channel_lf
from main.watermarking.content_chain import channel_hf


CONTENT_SCORE_RULE_VERSION = "v1"


def extract_lf_score_from_evidence(
    lf_evidence: Dict[str, Any],
    latent_or_trajectory: Optional[Any],
    plan: Optional[Dict[str, Any]],
    cfg: Dict[str, Any]
) -> Tuple[Optional[float], str]:
    """
    功能：从 LF 证据与 plan 提取检测分数。
    
    Extract LF detection score from embed-side evidence.
    Validates digest consistency and computes score.
    
    Args:
        lf_evidence: LF evidence dict from embed-side record.
        latent_or_trajectory: Optional latent for recomputation (for validation).
        plan: Subspace plan dict.
        cfg: Configuration mapping.
    
    Returns:
        Tuple of (lf_score, status_reason).
        lf_score=None if absent/failed/mismatch; otherwise non-negative float.
        status_reason: "ok" or failure reason.
    
    Raises:
        TypeError: If inputs types invalid.
    """
    if not isinstance(lf_evidence, dict):
        return None, "lf_evidence_invalid_type"
    
    if not isinstance(cfg, dict):
        return None, "cfg_invalid_type"
    
    lf_enabled = cfg.get("lf_enabled", True)
    if not lf_enabled:
        return None, "lf_disabled_by_config"
    
    # 检查 LF 自身状态。
    lf_status = lf_evidence.get("status")
    if lf_status == "absent":
        return None, f"lf_absent: {lf_evidence.get('absent_reason', 'unknown')}"
    elif lf_status == "failed":
        return None, f"lf_failed: {lf_evidence.get('failure_reason', 'unknown')}"
    elif lf_status == "mismatch":
        return None, "lf_mismatch"
    elif lf_status != "ok":
        return None, f"lf_status_unknown: {lf_status}"
    
    # LF 状态为 ok，尝试计算分数。
    # 简化实现：使用 metrics 中的预计算统计量。
    lf_metrics = lf_evidence.get("lf_metrics")
    if not isinstance(lf_metrics, dict):
        # 若无 metrics，尝试从 latent 重新计算。
        if latent_or_trajectory is not None and plan is not None:
            try:
                basis = plan.get("lf_basis")
                if basis is not None:
                    score = channel_lf.extract_lf_score(
                        latent_or_trajectory,
                        basis,
                        cfg=cfg
                    )
                    return score, "ok"
            except Exception as e:
                return None, f"lf_extraction_failed: {type(e).__name__}"
        else:
            return None, "lf_metrics_missing"
    
    # 基于 metrics 计算分数（差分能量）。
    before_norm = lf_metrics.get("before_norm", 0.0)
    after_norm = lf_metrics.get("after_norm", 0.0)
    if isinstance(before_norm, (int, float)) and isinstance(after_norm, (int, float)):
        # 能量差作为证据强度。
        energy_ratio = max(0.0, (after_norm - before_norm) / (before_norm + 1e-8))
        return float(energy_ratio), "ok"
    
    return None, "lf_metrics_incomplete"


def extract_hf_score_from_evidence(
    hf_evidence: Dict[str, Any],
    latent_or_trajectory: Optional[Any],
    plan: Optional[Dict[str, Any]],
    cfg: Dict[str, Any]
) -> Tuple[Optional[float], str]:
    """
    功能：从 HF 证据与 plan 提取检测分数。
    
    Extract HF detection score from embed-side evidence.
    Validates digest consistency and computes score.
    
    Args:
        hf_evidence: HF evidence dict from embed-side record.
        latent_or_trajectory: Optional latent for recomputation.
        plan: Subspace plan dict.
        cfg: Configuration mapping.
    
    Returns:
        Tuple of (hf_score, status_reason).
        hf_score=None if absent/failed/mismatch; otherwise non-negative float.
        status_reason: "ok" or failure reason.
    
    Raises:
        TypeError: If inputs types invalid.
    """
    if not isinstance(hf_evidence, dict):
        return None, "hf_evidence_invalid_type"
    
    if not isinstance(cfg, dict):
        return None, "cfg_invalid_type"
    
    hf_enabled = cfg.get("hf_enabled", True)
    if not hf_enabled:
        return None, "hf_disabled_by_config"
    
    # 检查 HF 自身状态。
    hf_status = hf_evidence.get("status")
    if hf_status == "absent":
        return None, f"hf_absent: {hf_evidence.get('absent_reason', 'unknown')}"
    elif hf_status == "failed":
        return None, f"hf_failed: {hf_evidence.get('failure_reason', 'unknown')}"
    elif hf_status == "mismatch":
        return None, "hf_mismatch"
    elif hf_status != "ok":
        return None, f"hf_status_unknown: {hf_status}"
    
    # HF 状态为 ok，尝试计算分数。
    hf_metrics = hf_evidence.get("hf_metrics")
    if not isinstance(hf_metrics, dict):
        # 若无 metrics，尝试从 latent 重新计算。
        if latent_or_trajectory is not None and plan is not None:
            try:
                basis = plan.get("hf_basis")
                if basis is not None:
                    score = channel_hf.extract_hf_score(
                        latent_or_trajectory,
                        basis,
                        cfg=cfg
                    )
                    return score, "ok"
            except Exception as e:
                return None, f"hf_extraction_failed: {type(e).__name__}"
        else:
            return None, "hf_metrics_missing"
    
    # 基于 metrics 计算分数（能量）。
    constrained_norm = hf_metrics.get("coeffs_after_norm", 0.0)
    if isinstance(constrained_norm, (int, float)):
        return float(max(0.0, constrained_norm)), "ok"
    
    return None, "hf_metrics_incomplete"


def validate_plan_digest_consistency(
    embed_time_plan_digest: Optional[str],
    detect_time_plan_digest: Optional[str]
) -> Tuple[bool, str]:
    """
    功能：校验 embed 时与 detect 时计划是否一致。
    
    Validate plan digest consistency between embed and detect time.
    
    Args:
        embed_time_plan_digest: Plan digest from embed-side record.
        detect_time_plan_digest: Plan digest computed at detect-side.
    
    Returns:
        Tuple of (is_consistent, reason).
        is_consistent=True if digests match or both absent.
    
    Raises:
        None (all errors logged as mismatch).
    """
    # 两者都缺失 → abandon（等待在更高层处理）。
    if embed_time_plan_digest is None and detect_time_plan_digest is None:
        return False, "plan_digest_both_absent"
    
    # embed 缺失，detect 有 → mismatch（期望 embed 有）。
    if embed_time_plan_digest is None:
        return False, "plan_digest_embed_absent"
    
    # embed 有，detect 缺失 → mismatch（期望 detect 能重新计算）。
    if detect_time_plan_digest is None:
        return False, "plan_digest_detect_absent"
    
    # 两者都有，比较。
    if embed_time_plan_digest == detect_time_plan_digest:
        return True, "plan_digest_consistent"
    else:
        return False, "plan_digest_mismatch"


def compute_content_score(
    lf_score: Optional[float],
    hf_score: Optional[float],
    lf_status: str,
    hf_status: str,
    rule_version: str = CONTENT_SCORE_RULE_VERSION
) -> Tuple[Optional[float], str]:
    """
    功能：使用固定规则组合 LF/HF 分数。
    
    Compute content_score from LF and HF scores using fixed rule.
    Ensures HF absent/failed does not impact LF score.
    
    Args:
        lf_score: LF detection score (or None).
        hf_score: HF detection score (or None).
        lf_status: LF status ("ok" / "absent" / "failed" / "mismatch").
        hf_status: HF status ("ok" / "absent" / "failed" / "mismatch").
        rule_version: Rule version identifier.
    
    Returns:
        Tuple of (content_score, rule_applied).
        content_score=None if critical failure; else non-negative float.
        rule_applied: Description of rule used.
    
    Raises:
        TypeError: If inputs types invalid.
    """
    if rule_version == CONTENT_SCORE_RULE_VERSION:
        # Rule v1: LF主导，HF增强但不阻断。
        # - LF status != "ok" (failed/mismatch/absent): mismatch → content_score = None
        # - LF status == "ok" && lf_score is None: failed → content_score = None
        # - HF status == "ok" && hf_score is not None: combine with boosting
        # - HF status != "ok": ignore, use LF only
        
        if lf_status == "mismatch":
            return None, "lf_mismatch"
        elif lf_status == "failed":
            return None, "lf_failed"
        elif lf_status == "absent":
            return None, "lf_absent"
        elif lf_status != "ok":
            return None, f"lf_status_invalid: {lf_status}"
        
        # LF status = "ok"
        if lf_score is None:
            return None, "lf_score_missing_but_status_ok"
        
        if not isinstance(lf_score, (int, float)):
            return None, "lf_score_invalid_type"
        
        content_score = float(lf_score)
        
        # 尝试增强 HF。
        if hf_status == "ok" and hf_score is not None and isinstance(hf_score, (int, float)):
            # HF 增强因子（简单求和）。
            hf_score_f = float(hf_score)
            # 组合规则：加权和，HF权重较小（增强但不主导）。
            combined = content_score + 0.3 * hf_score_f
            return combined, "lf_primary_hf_boosted"
        elif hf_status in ("absent", "disabled"):
            return content_score, "lf_only_hf_absent"
        else:
            return content_score, "lf_only_hf_unavailable"
    
    else:
        return None, f"unknown_rule_version: {rule_version}"



def validate_basis_digest_consistency(
    embed_basis_digest: Optional[str],
    detect_basis_digest: Optional[str]
) -> Tuple[bool, str]:
    """
    功能：校验 basis_digest 一致性。
    
    Validate subspace basis digest consistency.
    
    Args:
        embed_basis_digest: Basis digest from embed-side.
        detect_basis_digest: Basis digest recomputed at detect-side.
    
    Returns:
        Tuple of (is_consistent, reason).
    """
    if embed_basis_digest is None and detect_basis_digest is None:
        return False, "basis_digest_both_absent"
    
    if embed_basis_digest is None:
        return False, "basis_digest_embed_absent"
    
    if detect_basis_digest is None:
        return False, "basis_digest_detect_absent"
    
    if embed_basis_digest == detect_basis_digest:
        return True, "basis_digest_consistent"
    else:
        return False, "basis_digest_mismatch"


# ---------------------------------------------------------------------------
# Detect-side trajectory-based TFSW scoring（Phase 3：用真实 z_{t_e} 替代 final_latents）
# ---------------------------------------------------------------------------

def resolve_detect_trajectory_latent_for_timestep(
    trajectory_cache: Optional[Any],
    edit_timestep: int,
) -> Tuple[Optional[Any], str]:
    """
    Resolve z_t from detect-side LatentTrajectoryCache at the given edit_timestep.

    Exact-only: no nearest-step fallback. A timestep mismatch is treated as a
    hard failure to enforce trajectory-consistent TFSW (z_{t_e} exact match).

    Args:
        trajectory_cache: LatentTrajectoryCache instance or None.
        edit_timestep: Target diffusion step index (0-based), matching planner edit_timestep.

    Returns:
        Tuple of (z_t_or_none, status_str).
        status_str values: "ok_exact",
        "absent_empty_cache", "absent_exact_timestep_mismatch_edit_*_available_*".
    """
    if trajectory_cache is None or trajectory_cache.is_empty():
        return None, "absent_empty_cache"
    z_t = trajectory_cache.get(edit_timestep)
    if z_t is not None:
        return z_t, "ok_exact"
    available = trajectory_cache.available_steps()
    return None, f"absent_exact_timestep_mismatch_edit_{edit_timestep}_available_{sorted(available)}"


def extract_lf_score_from_detect_trajectory(
    trajectory_cache: Optional[Any],
    lf_basis: Optional[Dict[str, Any]],
    embed_lf_score: Optional[float],
    cfg: Dict[str, Any],
) -> Tuple[Optional[float], str]:
    """
    功能：从 detect 侧 trajectory cache 中取出 z_{t_e}，走 TFSW 路径提取 LF 分数。

    Extract LF score from detect-side trajectory latent via Trajectory Feature Space
    Watermarking (TFSW): phi = P^T vec(normalize(z_{t_e})).

    Uses exact-only timestep resolution; no nearest-step fallback.

    Args:
        trajectory_cache: LatentTrajectoryCache instance capturing detect-side latents.
        lf_basis: LF basis dict with trajectory_feature_spec and projection_matrix.
        embed_lf_score: Embed-side LF score for drift consistency check.
        cfg: Configuration mapping.

    Returns:
        Tuple of (lf_score_or_none, status_str).
    """
    if not isinstance(cfg, dict):
        return None, "cfg_invalid_type"
    if lf_basis is None:
        return None, "lf_basis_missing"
    tfs = lf_basis.get("trajectory_feature_spec")
    if not isinstance(tfs, dict) or tfs.get("feature_operator") != "masked_normalized_random_projection":
        return None, "tfs_spec_missing_or_invalid"
    edit_timestep = int(tfs.get("edit_timestep", 0))
    z_t, resolution_status = resolve_detect_trajectory_latent_for_timestep(
        trajectory_cache, edit_timestep
    )
    if z_t is None:
        return None, f"trajectory_latent_absent: {resolution_status}"
    try:
        from main.watermarking.content_chain.subspace.trajectory_feature_space import (
            extract_trajectory_feature_np,
        )
        phi = extract_trajectory_feature_np(np.asarray(z_t, dtype=np.float64), tfs)
        projection_matrix = lf_basis.get("projection_matrix")
        if projection_matrix is None:
            return None, "projection_matrix_missing"
        proj_np = np.asarray(projection_matrix, dtype=np.float32)
        if phi.shape[0] != proj_np.shape[0]:
            return None, f"phi_dim_mismatch_{phi.shape[0]}_vs_{proj_np.shape[0]}"
        lf_coeffs = np.dot(phi.astype(np.float32), proj_np)
        coeffs_norm = float(np.linalg.norm(lf_coeffs))
        lf_score_threshold = float(cfg.get("lf_score_threshold", 10.0))
        detect_lf_score = min(1.0, max(0.0, coeffs_norm / lf_score_threshold))
        if embed_lf_score is not None and abs(detect_lf_score - float(embed_lf_score)) > 0.15:
            return detect_lf_score, f"lf_score_drift_detected_trajectory_{resolution_status}"
        return detect_lf_score, f"ok_trajectory_{resolution_status}"
    except Exception as exc:
        return None, f"lf_trajectory_score_failed: {type(exc).__name__}"


def extract_hf_score_from_detect_trajectory(
    trajectory_cache: Optional[Any],
    hf_basis: Optional[Dict[str, Any]],
    embed_hf_score: Optional[float],
    cfg: Dict[str, Any],
) -> Tuple[Optional[float], str]:
    """
    功能：从 detect 侧 trajectory cache 中取出 z_{t_e}，走 TFSW 路径提取 HF 分数。

    Extract HF score from detect-side trajectory latent via Trajectory Feature Space
    Watermarking (TFSW): phi = P^T vec(normalize(z_{t_e})), then project with hf_projection_matrix.

    Uses exact-only timestep resolution; no nearest-step fallback.

    Args:
        trajectory_cache: LatentTrajectoryCache instance capturing detect-side latents.
        hf_basis: HF basis dict with trajectory_feature_spec and hf_projection_matrix.
        embed_hf_score: Embed-side HF score for drift consistency check.
        cfg: Configuration mapping.

    Returns:
        Tuple of (hf_score_or_none, status_str).
    """
    if not isinstance(cfg, dict):
        return None, "cfg_invalid_type"
    if hf_basis is None:
        return None, "hf_basis_missing"
    tfs = hf_basis.get("trajectory_feature_spec")
    if not isinstance(tfs, dict) or tfs.get("feature_operator") != "masked_normalized_random_projection":
        return None, "tfs_spec_missing_or_invalid"
    edit_timestep = int(tfs.get("edit_timestep", 0))
    z_t, resolution_status = resolve_detect_trajectory_latent_for_timestep(
        trajectory_cache, edit_timestep
    )
    if z_t is None:
        return None, f"trajectory_latent_absent: {resolution_status}"
    try:
        from main.watermarking.content_chain.subspace.trajectory_feature_space import (
            extract_trajectory_feature_np,
        )
        phi = extract_trajectory_feature_np(np.asarray(z_t, dtype=np.float64), tfs)
        hf_projection_matrix = hf_basis.get("hf_projection_matrix")
        if hf_projection_matrix is None:
            return None, "hf_projection_matrix_missing"
        proj_np = np.asarray(hf_projection_matrix, dtype=np.float32)
        if phi.shape[0] != proj_np.shape[0]:
            return None, f"phi_dim_mismatch_{phi.shape[0]}_vs_{proj_np.shape[0]}"
        hf_coeffs = np.dot(phi.astype(np.float32), proj_np)
        coeffs_norm = float(np.linalg.norm(hf_coeffs))
        hf_score_threshold = float(cfg.get("hf_score_threshold", 5.0))
        detect_hf_score = min(1.0, max(0.0, coeffs_norm / hf_score_threshold))
        if embed_hf_score is not None and abs(detect_hf_score - float(embed_hf_score)) > 0.15:
            return detect_hf_score, f"hf_score_drift_detected_trajectory_{resolution_status}"
        return detect_hf_score, f"ok_trajectory_{resolution_status}"
    except Exception as exc:
        return None, f"hf_trajectory_score_failed: {type(exc).__name__}"
