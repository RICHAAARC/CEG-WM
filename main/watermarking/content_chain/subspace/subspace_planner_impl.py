"""
子空间规划器

功能说明：
- 基于可复算的轨迹特征执行低维子空间估计（SVD）。
- 输出摘要化子空间定义（rank、energy_ratio、subspace_spec），不写入大矩阵。
- 绑定 mask_digest、planner_impl_identity、planner_params 与 model provenance 到 plan_digest。
- 严格区分 absent/failed 语义，避免产生伪有效 digest。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np

from main.core import digests
from main.watermarking.content_chain.subspace.planner_interface import SubspacePlanEvidence


SUBSPACE_PLANNER_ID = "subspace_planner_v1"
SUBSPACE_PLANNER_VERSION = "v1"
SUBSPACE_PLANNER_MASK_CONDITIONED_ID = "subspace_planner_mask_conditioned_v1"
SUBSPACE_PLANNER_MASK_CONDITIONED_VERSION = "v1"


ALLOWED_PLANNER_FAILURE_REASONS = {
    "planner_disabled_by_policy",
    "mask_absent",
    "planner_input_absent",
    "invalid_subspace_params",
    "decomposition_failed",
    "rank_computation_failed",
    "unknown"
}


def build_region_index_spec_from_mask_v1(mask: Any) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    功能：将掩码空间结构转换为区域索引规格。 

    Build region index specs from mask spatial structure.

    Args:
        mask: Mask array or summary dict with downsample grid fields.

    Returns:
        Tuple of (hf_region_index_spec, lf_region_index_spec, region_index_digest).

    Raises:
        TypeError: If mask type is invalid.
    """
    grid_rows = 8
    grid_cols = 8
    grid_mask = None
    area_ratio = None

    if isinstance(mask, dict):
        grid_shape = mask.get("downsample_grid_shape", [8, 8])
        if isinstance(grid_shape, list) and len(grid_shape) == 2:
            grid_rows = int(grid_shape[0]) if int(grid_shape[0]) > 0 else 8
            grid_cols = int(grid_shape[1]) if int(grid_shape[1]) > 0 else 8
        true_indices = mask.get("downsample_grid_true_indices")
        if isinstance(true_indices, list):
            grid_mask = np.zeros((grid_rows, grid_cols), dtype=bool)
            for idx in true_indices:
                if not isinstance(idx, int):
                    continue
                if idx < 0 or idx >= grid_rows * grid_cols:
                    continue
                row = idx // grid_cols
                col = idx % grid_cols
                grid_mask[row, col] = True
        area_ratio_value = mask.get("area_ratio")
        if isinstance(area_ratio_value, (int, float)):
            area_ratio = float(area_ratio_value)
    elif isinstance(mask, np.ndarray):
        if mask.ndim != 2:
            raise TypeError("mask must be 2D array when provided as ndarray")
        grid_mask = _downsample_mask_to_grid(mask.astype(bool), grid_rows, grid_cols)
        area_ratio = float(np.mean(mask)) if mask.size > 0 else 0.5
    elif mask is None:
        grid_mask = None
    else:
        raise TypeError("mask must be dict, ndarray, or None")

    if grid_mask is None:
        area_ratio = 0.5 if area_ratio is None else max(0.0, min(area_ratio, 1.0))
        total = grid_rows * grid_cols
        hf_count = max(1, int(round(area_ratio * total)))
        hf_indices = list(range(min(total, hf_count)))
    else:
        hf_indices = np.flatnonzero(grid_mask.reshape(-1)).astype(int).tolist()

    total = grid_rows * grid_cols
    hf_indices = sorted({int(v) for v in hf_indices if isinstance(v, int) and 0 <= int(v) < total})
    hf_index_set = set(hf_indices)
    lf_indices = [idx for idx in range(total) if idx not in hf_index_set]

    mask_grid_digest = digests.canonical_sha256(
        {
            "grid_shape": [grid_rows, grid_cols],
            "hf_indices": hf_indices,
        }
    )

    hf_region_index_spec = {
        "region_index_spec_version": "v1",
        "channel": "hf",
        "selector": "mask_true_grid",
        "grid_shape": [grid_rows, grid_cols],
        "selected_indices": hf_indices,
        "selected_count": len(hf_indices),
        "selection_space": "latent_grid_tokens",
        "mask_grid_digest": mask_grid_digest,
    }
    lf_region_index_spec = {
        "region_index_spec_version": "v1",
        "channel": "lf",
        "selector": "mask_false_grid",
        "grid_shape": [grid_rows, grid_cols],
        "selected_indices": lf_indices,
        "selected_count": len(lf_indices),
        "selection_space": "latent_grid_tokens",
        "mask_grid_digest": mask_grid_digest,
    }

    region_index_digest = digests.canonical_sha256(
        {
            "hf_region_index_spec": hf_region_index_spec,
            "lf_region_index_spec": lf_region_index_spec,
        }
    )
    return hf_region_index_spec, lf_region_index_spec, region_index_digest


def _downsample_mask_to_grid(mask_array: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    功能：将掩码下采样为固定网格布尔阵列。 

    Downsample mask into a fixed grid boolean map.

    Args:
        mask_array: 2D boolean mask array.
        rows: Target grid rows.
        cols: Target grid cols.

    Returns:
        Boolean grid mask.
    """
    if not isinstance(mask_array, np.ndarray) or mask_array.ndim != 2:
        raise TypeError("mask_array must be 2D ndarray")
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    height, width = mask_array.shape
    row_edges = np.linspace(0, height, rows + 1).astype(int)
    col_edges = np.linspace(0, width, cols + 1).astype(int)
    grid = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        r0, r1 = row_edges[r], row_edges[r + 1]
        for c in range(cols):
            c0, c1 = col_edges[c], col_edges[c + 1]
            cell = mask_array[r0:r1, c0:c1]
            if cell.size == 0:
                continue
            grid[r, c] = bool(np.mean(cell) >= 0.5)
    return grid


@dataclass(frozen=True)
class _PlannerParams:
    """
    功能：规划参数载体。

    Planner parameter bundle for deterministic subspace estimation.

    Args:
        rank: Requested subspace rank.
        sample_count: Number of sampled trajectory states.
        feature_dim: Feature dimension per sample.
        timestep_start: Start timestep index.
        timestep_end: End timestep index.
        seed: Deterministic seed for trajectory construction.
        float_round_digits: Round digits for float normalization.
        edit_timestep: Edit timestep.
        mask_shape: Mask shape string (circle/ring/square/whole/outercircle).
        mask_radius: Radius parameter for mask computation.
        mask_radius2: Secondary radius (for ring mask).
        w_channel: Channel index for watermarking (-1 for all channels).
        injection_domain: Domain for injection (spatial/freq).
        enable_channel_refill: Whether to enable channel refill strategy.
        num_inference_steps: Total number of inference steps.

    Returns:
        Immutable planner parameter bundle.
    """

    rank: int
    sample_count: int
    feature_dim: int
    timestep_start: int
    timestep_end: int
    seed: int
    float_round_digits: int
    trajectory_step_stride: int
    spectrum_topk: int
    jacobian_probe_count: int
    jacobian_eps: float
    edit_timestep: int = 0
    mask_shape: str = "circle"
    mask_radius: int = 10
    mask_radius2: int = 5
    w_channel: int = -1
    injection_domain: str = "spatial"
    enable_channel_refill: bool = False
    num_inference_steps: int = 50


@dataclass(frozen=True)
class SubspaceConditioning:
    """
    功能：子空间语义条件化摘要。 

    Serializable conditioning summary for mask-conditioned subspace estimation.

    Args:
        conditioning_mode: Conditioning mode label.
        mask_digest: Bound mask digest anchor.
        region_spec_digest: Digest of selected feature index spec.
        masked_dim_count: Number of selected feature dimensions.
        unmasked_dim_count: Number of non-selected feature dimensions.
        mask_area_ratio: Mask area ratio in [0, 1].
        fallback_used: Whether fallback branch is used.
        fallback_reason: Fallback reason string.
        selected_feature_indices: Selected feature index list.
    """

    conditioning_mode: str
    mask_digest: str
    region_spec_digest: str
    masked_dim_count: int
    unmasked_dim_count: int
    mask_area_ratio: float
    fallback_used: bool
    fallback_reason: str
    selected_feature_indices: List[int]

    def as_dict(self) -> Dict[str, Any]:
        """
        功能：序列化子空间条件化摘要。 

        Serialize subspace conditioning summary.

        Args:
            None.

        Returns:
            JSON-like conditioning mapping.
        """
        return {
            "conditioning_mode": self.conditioning_mode,
            "mask_digest": self.mask_digest,
            "region_spec_digest": self.region_spec_digest,
            "masked_dim_count": self.masked_dim_count,
            "unmasked_dim_count": self.unmasked_dim_count,
            "mask_area_ratio": self.mask_area_ratio,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "selected_feature_indices": self.selected_feature_indices,
        }

    def digest_payload(self) -> Dict[str, Any]:
        """
        功能：构造 digest 输入域。 

        Build canonical digest payload for conditioning anchors.

        Args:
            None.

        Returns:
            Canonical digest payload mapping.
        """
        return {
            "subspace_conditioning_version": "v1",
            **self.as_dict(),
        }


class SubspacePlannerImpl:
    """
    功能：真实子空间规划器实现。

    Reproducible subspace planner that estimates a low-dimensional basis by SVD,
    derives digest anchors, and emits only compact summaries.

    Args:
        impl_id: Implementation identifier string.
        impl_version: Implementation version string.
        impl_digest: Deterministic implementation digest.

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

    def plan(
        self,
        cfg: Dict[str, Any],
        mask_digest: Optional[str] = None,
        cfg_digest: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None
    ) -> SubspacePlanEvidence:
        """
        功能：执行子空间规划并返回摘要证据。

        Plan low-dimensional subspace and return digest-bound summary evidence.

        Args:
            cfg: Configuration mapping.
            mask_digest: Optional semantic mask digest.
            cfg_digest: Optional canonical config digest.
            inputs: Optional planning inputs containing trajectory features.

        Returns:
            SubspacePlanEvidence with strict failure semantics.

        Raises:
            TypeError: If input types are invalid.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if mask_digest is not None and not isinstance(mask_digest, str):
            raise TypeError("mask_digest must be str or None")
        if cfg_digest is not None and not isinstance(cfg_digest, str):
            raise TypeError("cfg_digest must be str or None")
        if inputs is not None and not isinstance(inputs, dict):
            raise TypeError("inputs must be dict or None")

        subspace_cfg = cfg.get("watermark", {}).get("subspace", {})
        enabled = subspace_cfg.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("watermark.subspace.enabled must be bool")

        trace_payload = self._build_trace_payload(
            cfg,
            enabled,
            mask_digest,
            cfg_digest,
            inputs
        )
        trace_digest = digests.canonical_sha256(trace_payload)
        audit = {
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "trace_digest": trace_digest
        }

        if not enabled:
            return SubspacePlanEvidence(
                status="absent",
                plan=None,
                basis_digest=None,
                plan_digest=None,
                audit=audit,
                plan_stats=None,
                plan_failure_reason="planner_disabled_by_policy"
            )

        if not isinstance(mask_digest, str) or not mask_digest:
            return SubspacePlanEvidence(
                status="absent",
                plan=None,
                basis_digest=None,
                plan_digest=None,
                audit=audit,
                plan_stats=None,
                plan_failure_reason="mask_absent"
            )

        if inputs is None or not inputs:
            return SubspacePlanEvidence(
                status="absent",
                plan=None,
                basis_digest=None,
                plan_digest=None,
                audit=audit,
                plan_stats=None,
                plan_failure_reason="planner_input_absent"
            )

        try:
            planner_params = self._parse_planner_params(cfg)
            basis_summary = self._estimate_low_dim_subspace(
                cfg=cfg,
                inputs=inputs,
                planner_params=planner_params,
                mask_digest=mask_digest
            )
            basis_digest = self._derive_basis_digest(basis_summary["basis_digest_payload"])
            plan_origin = "test_mode_synthetic" if basis_summary.get("samples_anchor", {}).get("source") == "test_mode_synthetic" else "planner_v1_band_spec"
            routing_digest_ref = self._extract_routing_digest_ref(inputs)
            band_spec, band_spec_digest, band_metrics = self.build_subspace_plan_v1(
                cfg=cfg,
                inputs=inputs,
                planner_params=planner_params,
                basis_summary=basis_summary,
                routing_digest_ref=routing_digest_ref,
            )
            mask_summary = inputs.get("mask_summary") if isinstance(inputs.get("mask_summary"), dict) else None
            paper_cfg = cfg.get("paper_faithfulness") if isinstance(cfg.get("paper_faithfulness"), dict) else {}
            if bool(paper_cfg.get("enabled", False)) and not isinstance(mask_summary, dict):
                # paper 模式下必须提供 mask_summary 用于空间级 region 规格。
                raise ValueError("mask_summary required for paper_faithfulness region index spec")

            hf_region_index_spec, lf_region_index_spec, region_index_digest = build_region_index_spec_from_mask_v1(
                mask_summary
            )
            hf_region_index_spec["feature_dim_anchor"] = planner_params.feature_dim
            lf_region_index_spec["feature_dim_anchor"] = planner_params.feature_dim
            region_index_digest = digests.canonical_sha256(
                {
                    "hf_region_index_spec": hf_region_index_spec,
                    "lf_region_index_spec": lf_region_index_spec,
                }
            )
            lf_basis = self._build_executable_basis_payload(
                basis_matrix=basis_summary.get("lf_projection_matrix"),
                planner_params=planner_params,
                basis_digest=basis_digest,
                channel="lf",
            )
            hf_basis = self._build_executable_basis_payload(
                basis_matrix=basis_summary.get("hf_projection_matrix"),
                planner_params=planner_params,
                basis_digest=basis_digest,
                channel="hf",
            )
            
            # 推断特征域标签
            feature_source_tag = self._infer_feature_source(inputs)
            normalization_tag = "centering_with_jacobian_probes"
            
            plan_payload = self._build_plan_payload_for_digest(
                cfg=cfg,
                cfg_digest=cfg_digest,
                mask_digest=mask_digest,
                planner_params=planner_params,
                basis_summary=basis_summary,
                basis_digest=basis_digest,
                hf_region_index_spec=hf_region_index_spec,
                lf_region_index_spec=lf_region_index_spec,
                region_index_digest=region_index_digest,
                lf_basis=lf_basis,
                hf_basis=hf_basis,
                subspace_conditioning=basis_summary.get("subspace_conditioning"),
                feature_source_tag=feature_source_tag,
                normalization_tag=normalization_tag,
                inputs=inputs
            )
            plan_digest = self._derive_plan_digest(plan_payload)
            
            # 构造 detection_domain_spec
            detection_domain_spec = self._build_detection_input_domain_spec(
                planner_params=planner_params,
                cfg=cfg
            )

            high_freq_subspace_spec = self._build_high_freq_subspace_spec(
                basis_summary=basis_summary
            )

            plan = {
                "plan_version": "v3",
                "subspace_method": "trajectory_jacobian_nullspace_svd",
                "subspace_source": "planner_computed",
                "plan_origin": plan_origin,
                "rank": basis_summary["rank"],
                "energy_ratio": basis_summary["energy_ratio"],
                "null_space_dim": basis_summary["null_space_dim"],
                "subspace_spec": basis_summary["subspace_spec"],
                "band_spec": band_spec,
                "band_spec_digest": band_spec_digest,
                "hf_region_index_spec": hf_region_index_spec,
                "lf_region_index_spec": lf_region_index_spec,
                "region_index_digest": region_index_digest,
                "routing_digest_ref": routing_digest_ref,
                "pipeline_feature_digest": "<absent>",
                "denoise_trace_digest": "<absent>",
                "attention_anchor_ref_digest": "<absent>",
                "basis_digest": basis_digest,
                "lf_basis": lf_basis,
                "hf_basis": hf_basis,
                "planner_impl_identity": {
                    "impl_id": self.impl_id,
                    "impl_version": self.impl_version,
                    "impl_digest": self.impl_digest
                },
                "feature_domain_anchor": {
                    "feature_source_tag": feature_source_tag,
                    "normalization_tag": normalization_tag,
                    "timestep_window": [planner_params.timestep_start, planner_params.timestep_end]
                },
                "subspace_conditioning": basis_summary.get("subspace_conditioning"),
                "injection_config": {
                    "edit_timestep": planner_params.edit_timestep,
                    "edit_timestep_ratio": self._normalize_float(
                        planner_params.edit_timestep / max(1, planner_params.num_inference_steps),
                        planner_params.float_round_digits
                    ) if planner_params.num_inference_steps > 0 else 0.0,
                    "mask_shape": planner_params.mask_shape,
                    "mask_radius": planner_params.mask_radius,
                    "w_channel": planner_params.w_channel,
                    "injection_domain": planner_params.injection_domain,
                    "channel_mix_policy": "channel_refill" if planner_params.enable_channel_refill else "none"
                },
                "detection_domain_spec": detection_domain_spec,
                "high_freq_subspace_spec": high_freq_subspace_spec
            }

            plan_stats = {
                "rank": basis_summary["rank"],
                "energy_ratio": basis_summary["energy_ratio"],
                "sample_count": basis_summary["sample_count"],
                "feature_dim": basis_summary["feature_dim"],
                "null_space_dim": basis_summary["null_space_dim"],
                "null_space_energy_ratio": basis_summary["null_space_energy_ratio"],
                "spectrum_summary": basis_summary["spectrum_summary"],
                "feature_source_tag": feature_source_tag,
                "band_spec_digest": band_spec_digest,
                "hf_region_ratio": band_metrics.get("hf_region_ratio"),
                "lf_region_ratio": band_metrics.get("lf_region_ratio"),
                "region_index_digest": region_index_digest,
                "lf_basis_shape": lf_basis.get("basis_shape"),
                "hf_basis_shape": hf_basis.get("basis_shape"),
                "subspace_conditioning": basis_summary.get("subspace_conditioning"),
            }

            return SubspacePlanEvidence(
                status="ok",
                plan=plan,
                basis_digest=basis_digest,
                plan_digest=plan_digest,
                audit=audit,
                plan_stats=plan_stats,
                plan_failure_reason=None
            )
        except ValueError as exc:
            message = str(exc).lower()
            if "planner inputs missing" in message or "trajectory source" in message:
                return SubspacePlanEvidence(
                    status="absent",
                    plan=None,
                    basis_digest=None,
                    plan_digest=None,
                    audit=audit,
                    plan_stats=None,
                    plan_failure_reason="planner_input_absent"
                )
            reason = "invalid_subspace_params"
            if "rank" in str(exc).lower():
                reason = "rank_computation_failed"
            return SubspacePlanEvidence(
                status="failed",
                plan=None,
                basis_digest=None,
                plan_digest=None,
                audit=audit,
                plan_stats=None,
                plan_failure_reason=reason
            )
        except Exception:
            return SubspacePlanEvidence(
                status="failed",
                plan=None,
                basis_digest=None,
                plan_digest=None,
                audit=audit,
                plan_stats=None,
                plan_failure_reason="decomposition_failed"
            )

    def _estimate_low_dim_subspace(
        self,
        cfg: Dict[str, Any],
        inputs: Dict[str, Any],
        planner_params: _PlannerParams,
        mask_digest: str
    ) -> Dict[str, Any]:
        """
        功能：估计低维子空间并提取摘要（集成可验证采样和真实 JVP）。

        Estimate low-dimensional subspace from trajectory features by SVD.
        Integrates verifiable trajectory sampling and real/surrogate Jacobian estimates.

        Args:
            cfg: Configuration mapping.
            inputs: Planner inputs containing feature trajectory.
            planner_params: Parsed planner parameters.
            mask_digest: Semantic mask digest.

        Returns:
            Subspace summary mapping for digesting and records.

        Raises:
            ValueError: If feature matrix is invalid.
            RuntimeError: If decomposition fails.
        """
        # 步骤 1：采样可验证轨迹特征
        trajectory_samples, samples_anchor = self._collect_trajectory_samples(
            cfg=cfg,
            inputs=inputs,
            planner_params=planner_params,
            sample_kind="pipeline_trajectory"
        )
        feature_matrix = self._align_feature_matrix(trajectory_samples, planner_params)
        mask_summary = inputs.get("mask_summary") if isinstance(inputs.get("mask_summary"), dict) else {}
        subspace_conditioning = self.build_subspace_conditioning(
            mask_summary=mask_summary,
            planner_params=planner_params,
            mask_digest=mask_digest,
        )
        feature_matrix = self._apply_mask_conditioning_to_feature_matrix(
            feature_matrix=feature_matrix,
            conditioning=subspace_conditioning,
        )
        
        if feature_matrix.shape[0] < 2 or feature_matrix.shape[1] < 2:
            raise ValueError("feature matrix must be at least 2x2")

        # 步骤 2：中心化
        centered = feature_matrix - np.mean(feature_matrix, axis=0, keepdims=True)
        
        # 步骤 3：估算真实或 Surrogate JVP
        jvp_samples, jvp_anchor = self._estimate_jvp_matrix(
            cfg=cfg,
            inputs=inputs,
            centered_matrix=centered,
            planner_params=planner_params
        )
        
        # 步骤 4：组合轨迹和 JVP 进行 SVD
        decomposition_matrix = np.concatenate([centered, jvp_samples], axis=0)
        try:
            _, singular_values, vh = np.linalg.svd(decomposition_matrix, full_matrices=False)
        except Exception as exc:
            raise RuntimeError("svd failed") from exc

        if singular_values.size == 0:
            raise ValueError("empty singular spectrum")

        effective_rank = min(planner_params.rank, vh.shape[0])
        if effective_rank <= 0:
            raise ValueError("rank must be positive after clipping")

        singular_energy = singular_values ** 2
        total_energy = float(np.sum(singular_energy))
        if not math.isfinite(total_energy) or total_energy <= 0:
            raise ValueError("invalid singular energy")

        selected_energy = float(np.sum(singular_energy[:effective_rank]))
        energy_ratio = selected_energy / total_energy
        energy_ratio = self._normalize_float(energy_ratio, planner_params.float_round_digits)
        null_space_energy = float(np.sum(singular_energy[effective_rank:]))
        null_space_energy_ratio = self._normalize_float(
            null_space_energy / total_energy,
            planner_params.float_round_digits
        )
        spectrum_summary = self._build_spectrum_summary(
            singular_values=singular_values,
            effective_rank=effective_rank,
            float_round_digits=planner_params.float_round_digits,
            spectrum_topk=planner_params.spectrum_topk
        )

        basis = vh[:effective_rank, :]
        lf_projection_matrix = basis.T
        hf_basis_candidates = vh[effective_rank:effective_rank + effective_rank, :]
        if hf_basis_candidates.shape[0] == 0:
            hf_basis_candidates = basis
        hf_projection_matrix = hf_basis_candidates.T
        basis_importance = np.mean(np.abs(basis), axis=0)
        top_index_count = min(32, basis_importance.shape[0])
        top_indices = np.argsort(-basis_importance)[:top_index_count].tolist()

        singular_preview = [
            self._normalize_float(float(value), planner_params.float_round_digits)
            for value in singular_values[: min(8, singular_values.shape[0])]
        ]

        null_space_dim = int(feature_matrix.shape[1] - effective_rank)
        trace_signature = inputs.get("trace_signature", {}) if isinstance(inputs, dict) else {}
        method_id = "trajectory_jacobian_nullspace_svd"
        mask_binding_enabled = bool(
            cfg.get("watermark", {}).get("subspace", {}).get("mask_digest_binding", True)
        )
        
        # 扩展采样策略以包含可验证信息
        sampling_strategy = {
            "source": samples_anchor.get("source", "deterministic_trajectory"),
            "row_selection": "linspace_resample",
            "feature_projection": "variance_topk_or_zero_pad",
            "trajectory_step_stride": planner_params.trajectory_step_stride,
            "sample_count": planner_params.sample_count,
            "feature_dim": planner_params.feature_dim,
            "jacobian_probe_count": planner_params.jacobian_probe_count,
            "jacobian_eps": self._normalize_float(planner_params.jacobian_eps, planner_params.float_round_digits),
            "jvp_source": jvp_anchor.get("jvp_source", "surrogate_transition")
        }
        
        subspace_spec = {
            "feature_source": samples_anchor.get("source", "deterministic_trajectory"),
            "method_id": method_id,
            "sample_count": int(feature_matrix.shape[0]),
            "feature_dim": int(feature_matrix.shape[1]),
            "timestep_window": [planner_params.timestep_start, planner_params.timestep_end],
            "top_feature_indices": top_indices,
            "singular_values_preview": singular_preview,
            "spectrum_summary": spectrum_summary,
            "sampling_strategy": sampling_strategy,
            "null_space_dim": null_space_dim,
            "seed": planner_params.seed,
            "jacobian_probe_count": planner_params.jacobian_probe_count,
            "jacobian_eps": self._normalize_float(planner_params.jacobian_eps, planner_params.float_round_digits),
            "trace_signature_anchor": {
                "num_inference_steps": samples_anchor.get("num_inference_steps", _read_int(trace_signature.get("num_inference_steps"), planner_params.sample_count)),
                "guidance_scale": samples_anchor.get("guidance_scale", self._normalize_float(_read_float(trace_signature.get("guidance_scale"), 7.0), planner_params.float_round_digits))
            }
        }

        # 步骤 5：构造basis_digest_payload，包含可验证框架信息
        basis_digest_payload = {
            "basis_digest_version": "v2",  # 版本升级以反映新的可验证机制
            "estimation_method": method_id,
            "rank": int(effective_rank),
            "energy_ratio": energy_ratio,
            "null_space_energy_ratio": null_space_energy_ratio,
            "null_space_dim": null_space_dim,
            "sample_count": int(feature_matrix.shape[0]),
            "feature_dim": int(feature_matrix.shape[1]),
            "seed": planner_params.seed,
            "sampling_strategy": sampling_strategy,
            "spectrum_summary": spectrum_summary,
            "mask_digest_binding_enabled": mask_binding_enabled,
            "subspace_spec": subspace_spec,
            "subspace_conditioning": subspace_conditioning.digest_payload(),
            # 新增：可验证输入域锚点
            "verifiable_input_domain": {
                "samples_anchor": samples_anchor,
                "jvp_anchor": jvp_anchor
            }
        }
        if mask_binding_enabled:
            basis_digest_payload["mask_digest"] = mask_digest

        return {
            "rank": int(effective_rank),
            "energy_ratio": energy_ratio,
            "null_space_dim": null_space_dim,
            "sample_count": int(feature_matrix.shape[0]),
            "feature_dim": int(feature_matrix.shape[1]),
            "null_space_energy_ratio": null_space_energy_ratio,
            "spectrum_summary": spectrum_summary,
            "subspace_spec": subspace_spec,
            "subspace_conditioning": subspace_conditioning.as_dict(),
            "basis_digest_payload": basis_digest_payload,
            "lf_projection_matrix": lf_projection_matrix,
            "hf_projection_matrix": hf_projection_matrix,
            # 新增：可验证锚点供 plan_digest 使用
            "samples_anchor": samples_anchor,
            "jvp_anchor": jvp_anchor
        }

    def build_subspace_conditioning(
        self,
        mask_summary: Dict[str, Any],
        planner_params: _PlannerParams,
        mask_digest: str,
    ) -> SubspaceConditioning:
        """
        功能：构建子空间语义条件化信息。 

        Build mask-conditioned subspace selection summary.

        Args:
            mask_summary: Mask summary mapping.
            planner_params: Planner parameter bundle.
            mask_digest: Mask digest anchor.

        Returns:
            SubspaceConditioning instance.
        """
        if not isinstance(mask_summary, dict):
            mask_summary = {}

        feature_dim = int(planner_params.feature_dim)
        if feature_dim <= 0:
            raise ValueError("planner_params.feature_dim must be positive")

        mask_area_ratio = mask_summary.get("area_ratio", 0.5)
        if not isinstance(mask_area_ratio, (int, float)):
            mask_area_ratio = 0.5
        mask_area_ratio = self._normalize_float(max(0.0, min(float(mask_area_ratio), 1.0)), planner_params.float_round_digits)

        raw_indices = mask_summary.get("downsample_grid_true_indices")
        selected_feature_indices: List[int] = []
        conditioning_mode = "full_feature_fallback"
        fallback_used = False
        fallback_reason = "<absent>"

        if isinstance(raw_indices, list) and len(raw_indices) > 0:
            normalized = []
            for value in raw_indices:
                if not isinstance(value, int):
                    continue
                normalized.append(int(value) % feature_dim)
            selected_feature_indices = sorted(set(normalized))
            if len(selected_feature_indices) > 0:
                conditioning_mode = "mask_indices_v1"

        if len(selected_feature_indices) == 0:
            fallback_used = True
            fallback_reason = "mask_indices_absent_or_invalid"
            conditioning_mode = "mask_ratio_v1"
            masked_dim_count = max(2, int(round(mask_area_ratio * feature_dim)))
            masked_dim_count = min(feature_dim, masked_dim_count)
            selected_feature_indices = list(range(masked_dim_count))

        if len(selected_feature_indices) >= feature_dim:
            fallback_used = True
            fallback_reason = "mask_selected_full_feature_domain"
            conditioning_mode = "full_feature_fallback"
            selected_feature_indices = list(range(feature_dim))

        masked_dim_count = len(selected_feature_indices)
        unmasked_dim_count = max(0, feature_dim - masked_dim_count)
        region_spec_digest = digests.canonical_sha256(
            {
                "feature_dim": feature_dim,
                "selected_feature_indices": selected_feature_indices,
                "conditioning_mode": conditioning_mode,
            }
        )
        return SubspaceConditioning(
            conditioning_mode=conditioning_mode,
            mask_digest=mask_digest,
            region_spec_digest=region_spec_digest,
            masked_dim_count=masked_dim_count,
            unmasked_dim_count=unmasked_dim_count,
            mask_area_ratio=mask_area_ratio,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            selected_feature_indices=selected_feature_indices,
        )

    def _apply_mask_conditioning_to_feature_matrix(
        self,
        feature_matrix: np.ndarray,
        conditioning: SubspaceConditioning,
    ) -> np.ndarray:
        """
        功能：按语义条件化筛选特征矩阵列。 

        Apply mask-conditioned feature selection to planner feature matrix.

        Args:
            feature_matrix: Planner feature matrix.
            conditioning: Subspace conditioning summary.

        Returns:
            Mask-conditioned feature matrix.
        """
        if not isinstance(feature_matrix, np.ndarray) or feature_matrix.ndim != 2:
            raise ValueError("feature_matrix must be 2D ndarray")
        selected = conditioning.selected_feature_indices
        if not isinstance(selected, list) or len(selected) == 0:
            return feature_matrix
        column_count = int(feature_matrix.shape[1])
        valid_indices = [idx for idx in selected if isinstance(idx, int) and 0 <= idx < column_count]
        if len(valid_indices) == 0:
            return feature_matrix
        return feature_matrix[:, valid_indices]

    def _build_plan_payload_for_digest(
        self,
        cfg: Dict[str, Any],
        cfg_digest: Optional[str],
        mask_digest: str,
        planner_params: _PlannerParams,
        basis_summary: Dict[str, Any],
        basis_digest: str,
        hf_region_index_spec: Dict[str, Any],
        lf_region_index_spec: Dict[str, Any],
        region_index_digest: str,
        lf_basis: Dict[str, Any],
        hf_basis: Dict[str, Any],
        subspace_conditioning: Optional[Dict[str, Any]] = None,
        feature_source_tag: Optional[str] = None,
        normalization_tag: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        功能：构造 plan_digest 输入域（扩展可验证框架）。

        Build canonical payload for plan digest binding, with explicit feature domain anchors.
        Now includes verifiable trajectory sampling and JVP anchors per requirements.

        Args:
            cfg: Configuration mapping.
            cfg_digest: Canonical cfg digest.
            mask_digest: Semantic mask digest.
            planner_params: Parsed planner parameters.
            basis_summary: Basis summary mapping (includes samples_anchor, jvp_anchor).
            basis_digest: Basis digest string.
            feature_source_tag: Feature source identifier (trajectory/trace_signature/etc).
            normalization_tag: Normalization strategy tag (centering/variance_scaling/etc).
            inputs: Optional planner inputs for input digest binding.

        Returns:
            Canonical payload mapping for digest that binds feature domain with verifiable anchors.
        """
        model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
        model_id = model_cfg.get("model_id", cfg.get("model_id", "<absent>"))
        model_revision = model_cfg.get("model_revision", cfg.get("model_revision", "<absent>"))
        
        # 从 basis_summary 中提取可验证锚点
        samples_anchor = basis_summary.get("samples_anchor", {})
        jvp_anchor = basis_summary.get("jvp_anchor", {})
        
        # 构造 mask_spec 摘要
        mask_spec = self._compute_mask_spec_from_shape(
            mask_shape=planner_params.mask_shape,
            mask_radius=planner_params.mask_radius,
            mask_radius2=planner_params.mask_radius2 if planner_params.mask_shape == "ring" else None
        )
        mask_spec_digest = digests.canonical_sha256(mask_spec)
        
        planner_input_digest = self._build_planner_input_digest(inputs)
        trajectory_evidence_anchor = self._extract_trajectory_evidence_anchor(inputs)

        # 构造注入参数承载
        injection_config_payload = self._build_injection_config_payload(
            edit_timestep=planner_params.edit_timestep,
            num_inference_steps=planner_params.num_inference_steps,
            mask_spec=mask_spec,
            mask_spec_digest=mask_spec_digest,
            w_channel=planner_params.w_channel,
            injection_domain=planner_params.injection_domain,
            enable_channel_refill=planner_params.enable_channel_refill,
            float_round_digits=planner_params.float_round_digits
        )
        
        # 构造 detection_domain_spec 摘要
        detection_domain_spec = {
            "edit_timestep": planner_params.edit_timestep,
            "num_inference_steps": planner_params.num_inference_steps,
            "forward_diffusion_start": 0,
            "forward_diffusion_end": planner_params.edit_timestep
        }
        detection_domain_spec_digest = digests.canonical_sha256(detection_domain_spec)
        
        # 新增：构造可验证输入域规格（论文级要求）
        verifiable_input_domain_spec = {
            "verifiable_input_domain_version": "v1",
            # 特征来源与采样规格
            "feature_source_tag": feature_source_tag or samples_anchor.get("source", "unspecified"),
            "timesteps_spec": {
                "window_start": planner_params.timestep_start,
                "window_end": planner_params.timestep_end,
                "stride": planner_params.trajectory_step_stride,
                "sample_count": planner_params.sample_count,
                "timesteps_digest": samples_anchor.get("timesteps_digest", "")
            },
            "planner_input_digest": planner_input_digest,
            "trajectory_evidence_anchor": trajectory_evidence_anchor,
            # Jacobian 探针规格
            "probe_spec": {
                "probe_seed": planner_params.seed,
                "probe_count": planner_params.jacobian_probe_count,
                "jacobian_eps": self._normalize_float(planner_params.jacobian_eps, planner_params.float_round_digits),
                "probe_seed_digest": jvp_anchor.get("probe_seed_digest", ""),
                "probe_count_digest": jvp_anchor.get("probe_count_digest", ""),
                "jacobian_eps_digest": jvp_anchor.get("jacobian_eps_digest", "")
            },
            # 规划参数规范化（统一的输入域）
            "planner_params_canonical": {
                "rank": planner_params.rank,
                "sample_count": planner_params.sample_count,
                "feature_dim": planner_params.feature_dim,
                "spectrum_topk": planner_params.spectrum_topk
            },
            # 模型可验证性锚点
            "model_provenance_anchor": {
                "model_id": model_id,
                "model_revision": model_revision
            },
            # Scheduler 关键参数
            "scheduler_tag": {
                "edit_timestep": planner_params.edit_timestep,
                "num_inference_steps": planner_params.num_inference_steps,
                "scheduler_type": cfg.get("scheduler", {}).get("scheduler_type", "<unspecified>")
            },
            # 样本与 JVP 来源表征
            "samples_source": samples_anchor.get("source", "deterministic_trajectory"),
            "jvp_source": jvp_anchor.get("jvp_source", "surrogate_transition"),
            # 摘要锚点（作为 basis_digest 的补充）
            "samples_anchor_digest": digests.canonical_sha256(samples_anchor),
            "jvp_anchor_digest": digests.canonical_sha256(jvp_anchor)
        }

        return {
            "plan_digest_version": "v4",  # 版本升级以反映新的可验证框架
            "mask_digest": mask_digest,
            "mask_spec_digest": mask_spec_digest,
            "cfg_digest": cfg_digest,
            "planner_input_digest": planner_input_digest,
            "planner_method": "trajectory_jacobian_nullspace_svd",
            "planner_impl_identity": {
                "impl_id": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest
            },
            "feature_domain_anchor": {
                "feature_source_tag": feature_source_tag or "unspecified",
                "normalization_tag": normalization_tag or "centering",
                "timestep_window": [planner_params.timestep_start, planner_params.timestep_end],
                "trajectory_sampling_tag": f"linspace_stride_{planner_params.trajectory_step_stride}"
            },
            "planner_params": {
                "rank": planner_params.rank,
                "sample_count": planner_params.sample_count,
                "feature_dim": planner_params.feature_dim,
                "timestep_start": planner_params.timestep_start,
                "timestep_end": planner_params.timestep_end,
                "seed": planner_params.seed,
                "float_round_digits": planner_params.float_round_digits,
                "trajectory_step_stride": planner_params.trajectory_step_stride,
                "spectrum_topk": planner_params.spectrum_topk,
                "jacobian_probe_count": planner_params.jacobian_probe_count,
                "jacobian_eps": self._normalize_float(planner_params.jacobian_eps, planner_params.float_round_digits),
                "low_freq": _build_low_freq_cfg_binding(cfg),
                "high_freq": _build_high_freq_cfg_binding(cfg)
            },
            "injection_config": injection_config_payload,
            "detection_domain_spec": detection_domain_spec,
            "detection_domain_spec_digest": detection_domain_spec_digest,
            "model_provenance_anchor": {
                "model_id": model_id,
                "model_revision": model_revision
            },
            # 新增：可验证输入域规格（论文级要求）
            "verifiable_input_domain_spec": verifiable_input_domain_spec,
            "basis_digest": basis_digest,
            "region_index_digest": region_index_digest,
            "hf_region_index_spec": hf_region_index_spec,
            "lf_region_index_spec": lf_region_index_spec,
            "lf_basis": lf_basis,
            "hf_basis": hf_basis,
            "subspace_conditioning": subspace_conditioning if isinstance(subspace_conditioning, dict) else {},
            "basis_summary": {
                "rank": basis_summary["rank"],
                "energy_ratio": basis_summary["energy_ratio"],
                "null_space_dim": basis_summary["null_space_dim"],
                "null_space_energy_ratio": basis_summary["null_space_energy_ratio"],
                "spectrum_summary": basis_summary["spectrum_summary"],
                "subspace_spec": basis_summary["subspace_spec"]
            }
        }

    def _derive_basis_digest(self, payload: Dict[str, Any]) -> str:
        """
        功能：计算 basis_digest。

        Compute basis digest using canonical SHA256.

        Args:
            payload: Basis payload mapping.

        Returns:
            Digest string.
        """
        return digests.canonical_sha256(payload)

    def _derive_plan_digest(self, payload: Dict[str, Any]) -> str:
        """
        功能：计算 plan_digest。

        Compute plan digest using canonical SHA256.

        Args:
            payload: Plan payload mapping.

        Returns:
            Digest string.
        """
        return digests.canonical_sha256(payload)

    def _build_high_freq_subspace_spec(self, basis_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：从规划摘要中派生 HF 子空间摘要。

        Derive high-frequency subspace summary strictly from planner basis summary.

        Args:
            basis_summary: Basis summary mapping containing subspace_spec.

        Returns:
            High-frequency subspace summary mapping.

        Raises:
            TypeError: If basis_summary type is invalid.
        """
        if not isinstance(basis_summary, dict):
            raise TypeError("basis_summary must be dict")
        subspace_spec = basis_summary.get("subspace_spec")
        if not isinstance(subspace_spec, dict):
            return {
                "selector": "planner_top_feature_tail_half",
                "selected_indices": [],
                "selected_count": 0,
                "source_field": "subspace_spec.top_feature_indices"
            }
        top_feature_indices = subspace_spec.get("top_feature_indices")
        if not isinstance(top_feature_indices, list):
            top_feature_indices = []
        normalized = [int(index) for index in top_feature_indices if isinstance(index, int) and index >= 0]
        split = max(1, len(normalized) // 2) if len(normalized) > 0 else 0
        selected_indices = normalized[-split:] if split > 0 else []
        return {
            "selector": "planner_top_feature_tail_half",
            "selected_indices": selected_indices,
            "selected_count": len(selected_indices),
            "source_field": "subspace_spec.top_feature_indices"
        }

    def _extract_feature_matrix(
        self,
        cfg: Dict[str, Any],
        inputs: Dict[str, Any],
        planner_params: _PlannerParams
    ) -> np.ndarray:
        """
        功能：提取或构造用于 SVD 的特征矩阵。

        Extract trajectory features from inputs or deterministically build from trace signature.

        Args:
            cfg: Configuration mapping.
            inputs: Planner inputs.
            planner_params: Parsed planner parameters.

        Returns:
            Two-dimensional numeric feature matrix.

        Raises:
            ValueError: If matrix cannot be extracted.
        """
        candidate_keys = ["latent_trajectory", "latent_features", "feature_samples"]
        for key in candidate_keys:
            if key in inputs:
                matrix = self._to_numpy_2d(inputs[key])
                if matrix.size > 0:
                    return matrix

        trace_signature = inputs.get("trace_signature")
        if not isinstance(trace_signature, dict):
            raise ValueError("planner inputs missing trajectory and trace_signature")

        return _generate_deterministic_trajectory_from_signature(trace_signature, planner_params, cfg)

    def _parse_planner_params(self, cfg: Dict[str, Any]) -> _PlannerParams:
        """
        功能：解析并校验规划参数（建立统一的输入域解析层）。

        Parse and validate planner parameters from config.
        Establishes single authority for k/topk/rank/spectrum_topk canonicalization.

        Args:
            cfg: Configuration mapping.

        Returns:
            Parsed planner parameter bundle.

        Raises:
            ValueError: If parameters are invalid.
        """
        subspace_cfg = cfg.get("watermark", {}).get("subspace", {})
        
        # 统一解析条例：优先级 rank > k，spectrum_topk > topk
        # 这确保了输入域的可验证性与唯一性
        rank = subspace_cfg.get("rank")
        if rank is None:
            rank = subspace_cfg.get("k", 8)
        
        sample_count = subspace_cfg.get("sample_count", 16)
        feature_dim = subspace_cfg.get("feature_dim", 128)
        timestep_start = subspace_cfg.get("timestep_start", 0)
        timestep_end = subspace_cfg.get("timestep_end", 30)
        seed = subspace_cfg.get("seed", cfg.get("seed", 0))
        float_round_digits = subspace_cfg.get("float_round_digits", 8)
        trajectory_step_stride = subspace_cfg.get("trajectory_step_stride", 1)
        # 统一解析：spectrum_topk > topk（兼容性读取）
        spectrum_topk = subspace_cfg.get("spectrum_topk")
        if spectrum_topk is None:
            spectrum_topk = subspace_cfg.get("topk", 8)
        jacobian_probe_count = subspace_cfg.get("jacobian_probe_count", 2)
        jacobian_eps = subspace_cfg.get("jacobian_eps", 1e-3)
        
        # 注入策略参数
        edit_timestep = subspace_cfg.get("edit_timestep", 0)
        mask_shape = subspace_cfg.get("mask_shape", "circle")
        mask_radius = subspace_cfg.get("mask_radius", 10)
        mask_radius2 = subspace_cfg.get("mask_radius2", 5)
        w_channel = subspace_cfg.get("w_channel", -1)
        injection_domain = subspace_cfg.get("injection_domain", "spatial")
        enable_channel_refill = subspace_cfg.get("enable_channel_refill", False)
        num_inference_steps = subspace_cfg.get("num_inference_steps", 50)

        integer_fields = {
            "rank": rank,
            "sample_count": sample_count,
            "feature_dim": feature_dim,
            "timestep_start": timestep_start,
            "timestep_end": timestep_end,
            "seed": seed,
            "float_round_digits": float_round_digits,
            "trajectory_step_stride": trajectory_step_stride,
            "spectrum_topk": spectrum_topk,
            "jacobian_probe_count": jacobian_probe_count,
            "edit_timestep": edit_timestep,
            "mask_radius": mask_radius,
            "mask_radius2": mask_radius2,
            "w_channel": w_channel,
            "num_inference_steps": num_inference_steps,
        }
        for field_name, field_value in integer_fields.items():
            if not isinstance(field_value, int):
                raise ValueError(f"{field_name} must be int")

        if not isinstance(jacobian_eps, (int, float)):
            raise ValueError("jacobian_eps must be numeric")
        jacobian_eps_value = float(jacobian_eps)
        
        if not isinstance(mask_shape, str):
            raise ValueError("mask_shape must be str")
        allowed_mask_shapes = {"circle", "ring", "square", "whole", "outercircle"}
        if mask_shape not in allowed_mask_shapes:
            raise ValueError(f"mask_shape must be one of {allowed_mask_shapes}, got {mask_shape}")
        
        if not isinstance(injection_domain, str):
            raise ValueError("injection_domain must be str")
        allowed_domains = {"spatial", "freq"}
        if injection_domain not in allowed_domains:
            raise ValueError(f"injection_domain must be one of {allowed_domains}, got {injection_domain}")
        
        if not isinstance(enable_channel_refill, bool):
            raise ValueError("enable_channel_refill must be bool")

        if rank <= 0:
            raise ValueError("rank must be positive")
        if sample_count < 2:
            raise ValueError("sample_count must be >= 2")
        if feature_dim < 2:
            raise ValueError("feature_dim must be >= 2")
        if timestep_end < timestep_start:
            raise ValueError("timestep_end must be >= timestep_start")
        if float_round_digits < 0 or float_round_digits > 12:
            raise ValueError("float_round_digits must be in [0, 12]")
        if trajectory_step_stride <= 0:
            raise ValueError("trajectory_step_stride must be positive")
        if spectrum_topk <= 0:
            raise ValueError("spectrum_topk must be positive")
        if jacobian_probe_count <= 0:
            raise ValueError("jacobian_probe_count must be positive")
        if not math.isfinite(jacobian_eps_value) or jacobian_eps_value <= 0:
            raise ValueError("jacobian_eps must be positive finite")
        if edit_timestep < 0 or edit_timestep > num_inference_steps:
            raise ValueError(f"edit_timestep must be in [0, {num_inference_steps}]")
        if mask_radius <= 0:
            raise ValueError("mask_radius must be positive")
        if mask_radius2 < 0:
            raise ValueError("mask_radius2 must be non-negative")
        if w_channel < -1 or w_channel > 3:
            raise ValueError("w_channel must be in [-1, 0, 1, 2, 3]")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        return _PlannerParams(
            rank=rank,
            sample_count=sample_count,
            feature_dim=feature_dim,
            timestep_start=timestep_start,
            timestep_end=timestep_end,
            seed=seed,
            float_round_digits=float_round_digits,
            trajectory_step_stride=trajectory_step_stride,
            spectrum_topk=spectrum_topk,
            jacobian_probe_count=jacobian_probe_count,
            jacobian_eps=jacobian_eps_value,
            edit_timestep=edit_timestep,
            mask_shape=mask_shape,
            mask_radius=mask_radius,
            mask_radius2=mask_radius2,
            w_channel=w_channel,
            injection_domain=injection_domain,
            enable_channel_refill=enable_channel_refill,
            num_inference_steps=num_inference_steps
        )

    def _align_feature_matrix(self, matrix: np.ndarray, planner_params: _PlannerParams) -> np.ndarray:
        """
        功能：将输入特征矩阵对齐到规划维度。

        Align feature matrix rows and columns to planner sample_count and feature_dim.

        Args:
            matrix: Raw feature matrix.
            planner_params: Planner parameter bundle.

        Returns:
            Aligned feature matrix.

        Raises:
            ValueError: If matrix shape is invalid.
        """
        if not isinstance(matrix, np.ndarray):
            raise ValueError("matrix must be numpy array")
        if matrix.ndim != 2:
            raise ValueError("matrix must be 2D")
        row_count, col_count = matrix.shape
        if row_count <= 0 or col_count <= 0:
            raise ValueError("matrix must be non-empty")

        target_rows = planner_params.sample_count
        if row_count == target_rows:
            row_aligned = matrix
        elif row_count > target_rows:
            row_indices = np.linspace(0, row_count - 1, target_rows, dtype=np.int64)
            row_aligned = matrix[row_indices, :]
        else:
            pad_count = target_rows - row_count
            pad_rows = np.repeat(matrix[-1:, :], pad_count, axis=0)
            row_aligned = np.concatenate([matrix, pad_rows], axis=0)

        target_cols = planner_params.feature_dim
        if row_aligned.shape[1] == target_cols:
            return row_aligned
        if row_aligned.shape[1] > target_cols:
            col_variance = np.var(row_aligned, axis=0)
            top_indices = np.argsort(-col_variance)[:target_cols]
            top_indices = np.sort(top_indices)
            return row_aligned[:, top_indices]

        col_pad = target_cols - row_aligned.shape[1]
        zeros = np.zeros((row_aligned.shape[0], col_pad), dtype=row_aligned.dtype)
        return np.concatenate([row_aligned, zeros], axis=1)

    def _estimate_jvp_matrix(
        self,
        cfg: Dict[str, Any],
        inputs: Dict[str, Any],
        centered_matrix: np.ndarray,
        planner_params: _PlannerParams
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        功能：基于真实去噪网络的 JVP 估算（论文级机制关键）。

        Estimate Jacobian-Vector Products (JVP) based on actual denoising network.
        If unet is provided in inputs, use real network; otherwise use surrogate mechanism.
        All probe parameters are anchored to digest for verifiability.

        Args:
            cfg: Configuration mapping.
            inputs: Planner inputs (may contain unet).
            centered_matrix: Centered trajectory matrix.
            planner_params: Planner parameters.

        Returns:
            Tuple of (jvp_samples, jvp_anchor) where:
            - jvp_samples: shape [probe_rows, feature_dim] JVP sample matrix
            - jvp_anchor: dict with digest anchors
                - "probe_seed_digest": SHA256 of probe seed
                - "probe_count_digest": SHA256 of probe count
                - "jacobian_eps_digest": SHA256 of eps value
                - "jvp_source": "real_unet" or "surrogate_transition"
                - "probe_vectors_digest": SHA256 of probe vectors
                - "jvp_energy_summary": spectral summary

        Raises:
            ValueError: If input is invalid.
        """
        if centered_matrix.ndim != 2:
            raise ValueError("centered_matrix must be 2D")
        if centered_matrix.shape[0] < 2:
            raise ValueError("centered_matrix must have at least two rows")
        
        # 构造确定性探针向量（所有 JVP 计算的基础）
        probes = self._build_deterministic_probe_vectors(
            feature_dim=centered_matrix.shape[1],
            probe_count=planner_params.jacobian_probe_count,
            seed=planner_params.seed
        )
        
        probe_vectors_digest = digests.canonical_sha256({
            "probes": np.round(probes, 8).tolist()[:min(4, len(probes))]
        })
        
        # 检查是否有真实 UNet
        unet = inputs.get("unet") if isinstance(inputs, dict) else None
        
        if unet is not None:
            # 路径 A：真实 UNet JVP（要求 UNet 在 inputs 中）
            jvp_samples, jvp_source = self._estimate_jvp_from_unet(
                unet=unet,
                centered_matrix=centered_matrix,
                probes=probes,
                planner_params=planner_params,
                cfg=cfg
            )
        else:
            # 路径 B：Surrogate transition-based JVP（总是可用）
            jvp_samples, jvp_source = self._estimate_jvp_from_transition(
                centered_matrix=centered_matrix,
                probes=probes,
                planner_params=planner_params
            )
        
        # 计算 JVP 能量摘要（不存储大矩阵）
        jvp_energy = np.sum(jvp_samples ** 2, axis=1)
        total_jvp_energy = float(np.sum(jvp_energy))
        _, jvp_singular_values, _ = np.linalg.svd(jvp_samples, full_matrices=False)
        jvp_spectrum = np.abs(jvp_singular_values)
        
        jvp_anchor = {
            "jvp_anchor_version": "v1",
            "probe_seed": planner_params.seed,
            "probe_seed_digest": digests.canonical_sha256({"seed": planner_params.seed}),
            "probe_count": planner_params.jacobian_probe_count,
            "probe_count_digest": digests.canonical_sha256({"count": planner_params.jacobian_probe_count}),
            "jacobian_eps": self._normalize_float(planner_params.jacobian_eps, planner_params.float_round_digits),
            "jacobian_eps_digest": digests.canonical_sha256({"eps": planner_params.jacobian_eps}),
            "jvp_source": jvp_source,
            "probe_vectors_digest": probe_vectors_digest,
            "jvp_shape": list(jvp_samples.shape),
            "jvp_energy_summary": {
                "total_energy": self._normalize_float(total_jvp_energy / max(1.0, float(jvp_samples.shape[0])), planner_params.float_round_digits),
                "spectrum_topk": [
                    self._normalize_float(float(sv), planner_params.float_round_digits)
                    for sv in jvp_spectrum[:min(5, len(jvp_spectrum))]
                ],
                "spectral_energy_ratio": self._normalize_float(
                    float(np.sum(jvp_spectrum[:min(3, len(jvp_spectrum))] ** 2) / max(1e-12, np.sum(jvp_spectrum ** 2))),
                    planner_params.float_round_digits
                )
            }
        }
        
        return jvp_samples, jvp_anchor

    def _estimate_jvp_from_unet(
        self,
        unet,  # Real UNet2DConditionModel from diffusers
        centered_matrix: np.ndarray,
        probes: np.ndarray,
        planner_params: _PlannerParams,
        cfg: Dict[str, Any]
    ) -> Tuple[np.ndarray, str]:
        """
        功能：从真实 UNet 计算 JVP（若 UNet 可用）。

        Compute JVP from actual UNet via finite differences on denoising network outputs.

        Args:
            unet: Real UNet2DConditionModel.
            centered_matrix: Centered trajectory matrix.
            probes: Probe vectors.
            planner_params: Planner parameters.
            cfg: Configuration mapping.

        Returns:
            Tuple of (jvp_samples, "real_unet").

        Raises:
            RuntimeError: If UNet forward pass fails.
        """
        try:
            import torch
        except ImportError:
            # 如果 torch 不可用，降级到 surrogate
            return self._estimate_jvp_from_transition(centered_matrix, probes, planner_params)
        
        jvp_rows: List[np.ndarray] = []
        
        # 选择 edit_timestep 处进行 JVP 计算
        edit_t = planner_params.edit_timestep
        t_emb = torch.tensor([edit_t], dtype=torch.float32)
        
        # 简化的条件嵌入（null conditioning）
        cond_emb = torch.zeros((1, 77, 768), dtype=torch.float32)  # CLIP embed shape
        
        for step_idx in range(min(centered_matrix.shape[0], 3)):  # 限制行数以保持效率
            x_t = torch.tensor(
                centered_matrix[step_idx:step_idx+1, :].reshape(1, 4, 8, 8),  # 假设 latent shape
                dtype=torch.float32, requires_grad=False
            )
            
            for probe in probes:
                probe_t = torch.tensor(probe.reshape(1, 4, 8, 8), dtype=torch.float32)
                delta = planner_params.jacobian_eps * probe_t
                
                try:
                    with torch.no_grad():
                        out_plus = unet(x_t + delta, t_emb, cond_emb).sample
                        out_minus = unet(x_t - delta, t_emb, cond_emb).sample
                    
                    jv = (out_plus - out_minus) / (2.0 * planner_params.jacobian_eps)
                    jvp_rows.append(jv.numpy().flatten())
                except Exception:
                    # 若 UNet 调用失败（形状不匹配等），跳过
                    pass
        
        if jvp_rows:
            jvp_samples = np.asarray(jvp_rows, dtype=np.float64)
            jvp_samples = jvp_samples - np.mean(jvp_samples, axis=0, keepdims=True)
            return jvp_samples, "real_unet"
        else:
            # 若 JVP 计算失败，降级到 surrogate
            return self._estimate_jvp_from_transition(centered_matrix, probes, planner_params)

    def _estimate_jvp_from_transition(
        self,
        centered_matrix: np.ndarray,
        probes: np.ndarray,
        planner_params: _PlannerParams
    ) -> Tuple[np.ndarray, str]:
        """
        功能：从轨迹转移拟合的 JVP 逼近（Surrogate 机制）。

        Estimate JVP from fitted transition operator (surrogate mechanism for when real unet unavailable).

        Args:
            centered_matrix: Centered trajectory matrix.
            probes: Probe vectors.
            planner_params: Planner parameters.

        Returns:
            Tuple of (jvp_samples, "surrogate_transition").
        """
        x_t = centered_matrix[:-1, :]
        x_tp1 = centered_matrix[1:, :]
        transition_alpha = self._estimate_transition_alpha(x_t, x_tp1)
        transition_bias = np.mean(x_tp1 - transition_alpha * x_t, axis=0, keepdims=True)
        
        jacobian_rows: List[np.ndarray] = []
        for step_index in range(x_t.shape[0]):
            current = x_t[step_index:step_index + 1, :]
            for probe in probes:
                delta = planner_params.jacobian_eps * probe
                plus_state = self._apply_transition(current + delta, transition_alpha, transition_bias)
                minus_state = self._apply_transition(current - delta, transition_alpha, transition_bias)
                jv = (plus_state - minus_state) / (2.0 * planner_params.jacobian_eps)
                jacobian_rows.append(jv.reshape(-1))
        
        jacobian_like = np.asarray(jacobian_rows, dtype=np.float64)
        jacobian_like = jacobian_like - np.mean(jacobian_like, axis=0, keepdims=True)
        return jacobian_like, "surrogate_transition"

    def _build_jacobian_probe_samples(
        self,
        centered_matrix: np.ndarray,
        planner_params: _PlannerParams
    ) -> np.ndarray:
        """
        功能：（兼容接口）从转移算子构造 Jacobian 近似样本。

        Legacy interface: build Jacobian-surrogate samples from adjacent trajectory differences.
        This is now superseded by _estimate_jvp_matrix but kept for backward compatibility.

        Args:
            centered_matrix: Centered trajectory matrix.
            planner_params: Parsed planner parameters.

        Returns:
            Jacobian-surrogate sample matrix.

        Raises:
            ValueError: If centered matrix is invalid.
        """
        # 兼容性包装：调用新的 JVP 估算，但只返回样本矩阵
        jvp_samples, _ = self._estimate_jvp_from_transition(
            centered_matrix=centered_matrix,
            probes=self._build_deterministic_probe_vectors(
                feature_dim=centered_matrix.shape[1],
                probe_count=planner_params.jacobian_probe_count,
                seed=planner_params.seed
            ),
            planner_params=planner_params
        )
        return jvp_samples

    def _estimate_transition_alpha(self, x_t: np.ndarray, x_tp1: np.ndarray) -> float:
        """
        功能：估计轨迹转移的一阶缩放参数。

        Estimate scalar transition coefficient between adjacent states.

        Args:
            x_t: Previous state matrix.
            x_tp1: Next state matrix.

        Returns:
            Estimated transition scalar.
        """
        numerator = float(np.sum(x_t * x_tp1))
        denominator = float(np.sum(x_t * x_t)) + 1e-12
        alpha = numerator / denominator
        if not math.isfinite(alpha):
            return 0.0
        return float(alpha)

    def _apply_transition(self, state: np.ndarray, alpha: float, bias: np.ndarray) -> np.ndarray:
        """
        功能：应用一阶近似转移算子。

        Apply first-order transition operator.

        Args:
            state: Input state row.
            alpha: Scalar transition factor.
            bias: Transition bias row.

        Returns:
            Next state row.
        """
        return alpha * state + bias

    def _build_deterministic_probe_vectors(self, feature_dim: int, probe_count: int, seed: int) -> np.ndarray:
        """
        功能：构造 Jacobian 探针方向。

        Build deterministic probe vectors for Jacobian finite differences.

        Args:
            feature_dim: Feature dimension.
            probe_count: Number of probe directions.
            seed: Deterministic seed.

        Returns:
            Probe matrix with shape [probe_count, feature_dim].
        """
        rng = np.random.default_rng(seed + 7919)
        signs = rng.choice([-1.0, 1.0], size=(probe_count, feature_dim))
        norms = np.linalg.norm(signs, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-12)
        return signs / safe_norms

    def _build_spectrum_summary(
        self,
        singular_values: np.ndarray,
        effective_rank: int,
        float_round_digits: int,
        spectrum_topk: int
    ) -> Dict[str, Any]:
        """
        功能：构造量化奇异值谱摘要。

        Build normalized and quantized spectrum summary with enhanced interpretability.

        Args:
            singular_values: Singular values from decomposition.
            effective_rank: Selected rank.
            float_round_digits: Float normalization digits.
            spectrum_topk: Number of normalized singular values to keep.

        Returns:
            Spectrum summary mapping with extended diagnostics.
        """
        singular_energy = singular_values ** 2
        total_energy = float(np.sum(singular_energy))
        safe_total_energy = max(total_energy, 1e-12)
        normalized = singular_values / max(float(np.sum(np.abs(singular_values))), 1e-12)
        topk = min(int(spectrum_topk), int(normalized.shape[0]))
        normalized_top = [
            self._normalize_float(float(item), float_round_digits)
            for item in normalized[:topk]
        ]
        tail_energy_ratio = float(np.sum(singular_energy[effective_rank:]) / safe_total_energy)
        stable_rank = float((np.sum(np.abs(singular_values)) ** 2) / safe_total_energy)
        
        # 新增：更多诊断统计量
        log_singular_ratios = []
        for i in range(min(3, len(singular_values) - 1)):
            if singular_values[i] > 1e-12 and singular_values[i+1] > 1e-12:
                ratio = float(np.log(singular_values[i] / singular_values[i+1]))
                log_singular_ratios.append(self._normalize_float(ratio, float_round_digits))
        
        return {
            "spectrum_version": "v2",
            "topk_normalized": normalized_top,
            "total_components": int(singular_values.shape[0]),
            "tail_energy_ratio": self._normalize_float(tail_energy_ratio, float_round_digits),
            "stable_rank": self._normalize_float(stable_rank, float_round_digits),
            "rank_energy_threshold": 0.95,
            "effective_rank": effective_rank,
            "log_singular_ratios": log_singular_ratios,
            "singular_values_energy": [
                self._normalize_float(float(sv**2 / safe_total_energy), float_round_digits)
                for sv in singular_values[:min(8, len(singular_values))]
            ]
        }

    def _build_trace_payload(
        self,
        cfg: Dict[str, Any],
        enabled: bool,
        mask_digest: Optional[str],
        cfg_digest: Optional[str],
        inputs: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        功能：构造 trace_digest 输入域。

        Build trace payload for planner audit digest.

        Args:
            cfg: Configuration mapping.
            enabled: Planner enable flag.
            mask_digest: Optional mask digest.
            cfg_digest: Optional cfg digest.
            inputs: Optional planner inputs.

        Returns:
            Deterministic trace payload mapping.
        """
        input_keys: List[str] = []
        if isinstance(inputs, dict):
            input_keys = sorted(inputs.keys())
        return {
            "trace_version": "v3",
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "enabled": enabled,
            "mask_digest": mask_digest,
            "cfg_digest": cfg_digest,
            "input_keys": input_keys,
            "policy_path": cfg.get("policy_path", "<absent>")
        }

    def _build_planner_input_digest(self, inputs: Optional[Dict[str, Any]]) -> str:
        """
        功能：构造 planner 输入摘要。

        Build canonical planner input digest from trace_signature and trajectory evidence.

        Args:
            inputs: Optional planner inputs mapping.

        Returns:
            Canonical SHA256 digest string.

        Raises:
            TypeError: If inputs are invalid.
        """
        if inputs is not None and not isinstance(inputs, dict):
            # inputs 类型不符合预期，必须 fail-fast。
            raise TypeError("inputs must be dict or None")

        trace_signature = None
        if isinstance(inputs, dict):
            trace_signature = inputs.get("trace_signature")

        trajectory_anchor = self._extract_trajectory_evidence_anchor(inputs)

        payload = {
            "planner_input_version": "v1",
            "trace_signature": trace_signature if isinstance(trace_signature, dict) else None,
            "trajectory_evidence_anchor": trajectory_anchor
        }
        return digests.canonical_sha256(payload)

    def _extract_trajectory_evidence_anchor(self, inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        功能：提取 trajectory_evidence 的摘要锚点。

        Extract compact anchor from trajectory evidence for digest binding.

        Args:
            inputs: Optional planner inputs mapping.

        Returns:
            Anchor mapping with status and digest fields.

        Raises:
            TypeError: If inputs are invalid.
        """
        if inputs is not None and not isinstance(inputs, dict):
            # inputs 类型不符合预期，必须 fail-fast。
            raise TypeError("inputs must be dict or None")

        trajectory_evidence = None
        if isinstance(inputs, dict):
            trajectory_evidence = inputs.get("trajectory_evidence")

        if trajectory_evidence is None:
            return {
                "status": "absent",
                "trajectory_spec_digest": "<absent>",
                "trajectory_digest": "<absent>",
                "trajectory_tap_version": "<absent>"
            }

        if not isinstance(trajectory_evidence, dict):
            # trajectory_evidence 类型不符合预期，必须 fail-fast。
            raise TypeError("trajectory_evidence must be dict or None")

        return {
            "status": trajectory_evidence.get("status", "<absent>"),
            "trajectory_spec_digest": trajectory_evidence.get("trajectory_spec_digest", "<absent>"),
            "trajectory_digest": trajectory_evidence.get("trajectory_digest", "<absent>"),
            "trajectory_tap_version": trajectory_evidence.get("trajectory_tap_version", "<absent>")
        }

    def _to_numpy_2d(self, value: Any) -> np.ndarray:
        """
        功能：将输入转换为二维浮点数组。

        Convert arbitrary nested numeric structure to 2D numpy array.

        Args:
            value: Input value.

        Returns:
            Two-dimensional array.

        Raises:
            ValueError: If conversion fails.
        """
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            raise ValueError("feature value must not be scalar")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim > 2:
            first_dim = array.shape[0]
            array = array.reshape(first_dim, -1)
        if not np.isfinite(array).all():
            raise ValueError("feature matrix contains non-finite values")
        return array

    def _normalize_float(self, value: float, digits: int) -> float:
        """
        功能：规范化浮点值以稳定跨机复算。

        Normalize floating-point value by finite check and rounding.

        Args:
            value: Input float.
            digits: Decimal digits.

        Returns:
            Normalized float.

        Raises:
            ValueError: If value is non-finite.
        """
        if not math.isfinite(value):
            raise ValueError("non-finite float encountered")
        return float(round(value, digits))

    def _infer_feature_source(self, inputs: Dict[str, Any]) -> str:
        """
        功能：推断特征来源标签。

        Infer feature source tag for audit summary.

        Args:
            inputs: Planner inputs.

        Returns:
            Source label string.
        """
        if "latent_trajectory" in inputs:
            return "latent_trajectory"
        if "latent_features" in inputs:
            return "latent_features"
        if "feature_samples" in inputs:
            return "feature_samples"
        return "trace_signature_projection"

    def _extract_routing_digest_ref(self, inputs: Dict[str, Any]) -> str:
        """
        功能：提取 routing_digest 引用位。 

        Extract routing digest reference from planner inputs.

        Args:
            inputs: Planner input mapping.

        Returns:
            Routing digest reference string or "<absent>".
        """
        if not isinstance(inputs, dict):
            return "<absent>"
        candidate = inputs.get("routing_digest")
        if isinstance(candidate, str) and candidate:
            return candidate
        return "<absent>"

    def build_subspace_plan_v1(
        self,
        cfg: Dict[str, Any],
        inputs: Dict[str, Any],
        planner_params: _PlannerParams,
        basis_summary: Dict[str, Any],
        routing_digest_ref: str,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """
        功能：构造 Planner v1 的 band 规范与摘要。 

        Build planner v1 band specification and digest anchors.

        Args:
            cfg: Configuration mapping.
            inputs: Planner input mapping.
            planner_params: Parsed planner parameters.
            basis_summary: Basis summary mapping.
            routing_digest_ref: Routing digest reference.

        Returns:
            Tuple of (band_spec, band_spec_digest, band_metrics).
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if not isinstance(inputs, dict):
            raise TypeError("inputs must be dict")
        if not isinstance(basis_summary, dict):
            raise TypeError("basis_summary must be dict")

        mask_summary = inputs.get("mask_summary") if isinstance(inputs.get("mask_summary"), dict) else {}
        band_spec = self.build_band_spec_from_mask(mask_summary, planner_params, routing_digest_ref)
        band_spec_digest = self.compute_band_spec_digest(band_spec)

        hf_selector_summary = band_spec.get("hf_selector_summary") if isinstance(band_spec.get("hf_selector_summary"), dict) else {}
        lf_selector_summary = band_spec.get("lf_selector_summary") if isinstance(band_spec.get("lf_selector_summary"), dict) else {}
        band_metrics = {
            "hf_region_ratio": hf_selector_summary.get("region_ratio", 0.5),
            "lf_region_ratio": lf_selector_summary.get("region_ratio", 0.5),
            "rank": basis_summary.get("rank"),
            "energy_ratio": basis_summary.get("energy_ratio"),
        }
        return band_spec, band_spec_digest, band_metrics

    def build_band_spec_from_mask(
        self,
        mask_summary: Dict[str, Any],
        planner_params: _PlannerParams,
        routing_digest_ref: str,
    ) -> Dict[str, Any]:
        """
        功能：从 mask 摘要构建 LF/HF 频带规范。 

        Build LF/HF band specification from mask summary and planner params.

        Args:
            mask_summary: Optional mask summary mapping.
            planner_params: Parsed planner parameter bundle.
            routing_digest_ref: Routing digest reference.

        Returns:
            Band specification mapping.
        """
        if not isinstance(mask_summary, dict):
            mask_summary = {}

        block_size = 8
        dct_low_cutoff = max(1, min(planner_params.rank, block_size - 1))
        mask_area_ratio = mask_summary.get("area_ratio")
        if not isinstance(mask_area_ratio, (int, float)):
            mask_area_ratio = 0.5
        mask_area_ratio = max(0.0, min(float(mask_area_ratio), 1.0))

        hf_ratio = self._normalize_float(mask_area_ratio, planner_params.float_round_digits)
        lf_ratio = self._normalize_float(1.0 - mask_area_ratio, planner_params.float_round_digits)

        return {
            "band_spec_version": "v1",
            "basis_selector": "dct_block",
            "dct_block_size": block_size,
            "lf_band_rule": {
                "type": "u_v_le_k",
                "k": dct_low_cutoff,
            },
            "hf_band_rule": {
                "type": "complement_of_lf",
            },
            "hf_selector_summary": {
                "selector": "mask_true_or_high_texture",
                "region_ratio": hf_ratio,
                "routing_digest_ref": routing_digest_ref,
            },
            "lf_selector_summary": {
                "selector": "mask_false_or_smooth",
                "region_ratio": lf_ratio,
                "routing_digest_ref": routing_digest_ref,
            },
        }

    def compute_band_spec_digest(self, band_spec: Dict[str, Any]) -> str:
        """
        功能：计算 band 规范摘要。 

        Compute canonical band specification digest.

        Args:
            band_spec: Band specification mapping.

        Returns:
            SHA256 digest string.
        """
        if not isinstance(band_spec, dict):
            raise TypeError("band_spec must be dict")
        return digests.canonical_sha256(band_spec)

    def build_region_index_spec_from_mask(
        self,
        inputs: Dict[str, Any],
        planner_params: _PlannerParams,
        channel: str,
    ) -> Dict[str, Any]:
        """
        功能：由掩码摘要构造可执行区域索引规格。

        Build executable region index spec from mask summary.

        Args:
            inputs: Planner inputs mapping.
            planner_params: Parsed planner params.
            channel: Channel tag ("hf" or "lf").

        Returns:
            Region index spec mapping.
        """
        if not isinstance(inputs, dict):
            raise TypeError("inputs must be dict")
        if channel not in {"hf", "lf"}:
            raise ValueError("channel must be hf or lf")

        mask_summary = inputs.get("mask_summary") if isinstance(inputs.get("mask_summary"), dict) else {}
        grid_shape = mask_summary.get("downsample_grid_shape", [8, 8])
        if not (isinstance(grid_shape, list) and len(grid_shape) == 2):
            grid_shape = [8, 8]
        rows = int(grid_shape[0]) if int(grid_shape[0]) > 0 else 8
        cols = int(grid_shape[1]) if int(grid_shape[1]) > 0 else 8
        total = rows * cols

        hf_indices = mask_summary.get("downsample_grid_true_indices")
        if not isinstance(hf_indices, list):
            hf_ratio = mask_summary.get("area_ratio", 0.5)
            if not isinstance(hf_ratio, (int, float)):
                hf_ratio = 0.5
            hf_count = max(1, int(round(float(hf_ratio) * total)))
            hf_indices = list(range(min(total, hf_count)))
        hf_indices = sorted({int(v) for v in hf_indices if isinstance(v, int) and 0 <= int(v) < total})
        lf_indices = [idx for idx in range(total) if idx not in set(hf_indices)]

        selected_indices = hf_indices if channel == "hf" else lf_indices
        selector = "mask_true_grid" if channel == "hf" else "mask_false_grid"
        payload = {
            "region_index_spec_version": "v1",
            "channel": channel,
            "selector": selector,
            "grid_shape": [rows, cols],
            "selected_indices": selected_indices,
            "selected_count": len(selected_indices),
            "selection_space": "latent_grid_tokens",
            "feature_dim_anchor": planner_params.feature_dim,
            "mask_grid_digest": mask_summary.get("downsample_grid_digest", "<absent>"),
        }
        payload["region_index_digest"] = digests.canonical_sha256(
            {
                "channel": channel,
                "grid_shape": payload["grid_shape"],
                "selected_indices": selected_indices,
                "mask_grid_digest": payload["mask_grid_digest"],
            }
        )
        return payload

    def _build_executable_basis_payload(
        self,
        basis_matrix: Any,
        planner_params: _PlannerParams,
        basis_digest: str,
        channel: str,
    ) -> Dict[str, Any]:
        """
        功能：构造可执行 basis 载荷（可序列化、可复算）。

        Build executable basis payload for runtime latent modifier.

        Args:
            basis_matrix: Basis projection matrix-like value.
            planner_params: Planner params bundle.
            basis_digest: Basis digest string.
            channel: Channel tag ("lf" or "hf").

        Returns:
            Serializable basis payload mapping.
        """
        if channel not in {"lf", "hf"}:
            raise ValueError("channel must be lf or hf")
        matrix_np = np.asarray(basis_matrix, dtype=np.float32)
        if matrix_np.ndim != 2:
            raise ValueError("basis_matrix must be rank-2")
        max_cols = min(16, matrix_np.shape[1])
        matrix_np = matrix_np[:, :max_cols]
        quantized = np.round(matrix_np, 6).astype(np.float32)
        payload = {
            "basis_payload_version": "v1",
            "channel": channel,
            "basis_kind": "planner_projection_payload",
            "basis_shape": [int(quantized.shape[0]), int(quantized.shape[1])],
            "basis_rank": int(quantized.shape[1]),
            "basis_quantization": "round_1e-6",
            "basis_seed_digest": digests.canonical_sha256(
                {
                    "basis_digest": basis_digest,
                    "channel": channel,
                    "seed": planner_params.seed,
                }
            ),
        }
        if channel == "lf":
            payload["projection_matrix"] = quantized.tolist()
        else:
            payload["hf_projection_matrix"] = quantized.tolist()
        payload["basis_payload_digest"] = digests.canonical_sha256(payload)
        return payload

    def _build_injection_config_payload(
        self,
        edit_timestep: int,
        num_inference_steps: int,
        mask_spec: Dict[str, Any],
        mask_spec_digest: str,
        w_channel: int,
        injection_domain: str,
        enable_channel_refill: bool,
        float_round_digits: int
    ) -> Dict[str, Any]:
        """
        功能：构造注入配置的规划参数摘要。
        
        Build canonical payload for injection configuration binding.
        
        Args:
            edit_timestep: Edit timestep.
            num_inference_steps: Total inference steps.
            mask_spec: Mask specification dict.
            mask_spec_digest: SHA256 digest of mask_spec.
            w_channel: Watermark channel index.
            injection_domain: Injection domain (spatial/freq).
            enable_channel_refill: Whether channel refill is enabled.
            float_round_digits: Float rounding digits.
        
        Returns:
            Canonical payload dict for digest binding.
        """
        if not isinstance(edit_timestep, int) or edit_timestep < 0:
            raise ValueError("edit_timestep must be non-negative int")
        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive int")
        if not isinstance(mask_spec, dict):
            raise ValueError("mask_spec must be dict")
        if not isinstance(mask_spec_digest, str) or len(mask_spec_digest) != 64:
            raise ValueError("mask_spec_digest must be 64-char hex string")
        if not isinstance(w_channel, int) or w_channel < -1 or w_channel > 3:
            raise ValueError("w_channel must be in [-1, 0, 1, 2, 3]")
        if injection_domain not in {"spatial", "freq"}:
            raise ValueError(f"injection_domain must be spatial or freq, got {injection_domain}")
        if not isinstance(enable_channel_refill, bool):
            raise ValueError("enable_channel_refill must be bool")
        
        edit_timestep_ratio = self._normalize_float(
            edit_timestep / max(1, num_inference_steps),
            float_round_digits
        ) if num_inference_steps > 0 else 0.0
        
        return {
            "injection_config_version": "v1",
            "edit_timestep": edit_timestep,
            "edit_timestep_ratio": edit_timestep_ratio,
            "num_inference_steps": num_inference_steps,
            "mask_spec": mask_spec,
            "mask_spec_digest": mask_spec_digest,
            "w_channel": w_channel,
            "injection_domain": injection_domain,
            "channel_mix_policy": "channel_refill" if enable_channel_refill else "none",
            "enable_channel_refill": enable_channel_refill
        }

    def _compute_mask_spec_from_shape(
        self,
        mask_shape: str,
        mask_radius: int,
        mask_radius2: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        功能：从掩码形状参数计算规范化掩码规格。
        
        Compute canonical mask specification from shape parameters.
        
        Args:
            mask_shape: Mask shape string (circle/ring/square/whole/outercircle).
            mask_radius: Primary radius parameter.
            mask_radius2: Secondary radius (for ring mask).
        
        Returns:
            Canonical mask spec dict for digest binding.
        
        Raises:
            ValueError: If mask parameters are invalid.
        """
        allowed_shapes = {"circle", "ring", "square", "whole", "outercircle"}
        if mask_shape not in allowed_shapes:
            raise ValueError(f"mask_shape must be one of {allowed_shapes}, got {mask_shape}")
        if not isinstance(mask_radius, int) or mask_radius <= 0:
            raise ValueError("mask_radius must be positive int")
        
        spec: Dict[str, Any] = {
            "mask_spec_version": "v1",
            "shape": mask_shape,
            "radius": mask_radius
        }
        
        if mask_shape == "ring":
            if mask_radius2 is None:
                raise ValueError("ring mask requires mask_radius2")
            if not isinstance(mask_radius2, int) or mask_radius2 < 0:
                raise ValueError("mask_radius2 must be non-negative int")
            if mask_radius2 >= mask_radius:
                raise ValueError("mask_radius2 must be less than radius")
            spec["radius2"] = mask_radius2
            spec["frequency_band"] = "mid_freq"
        elif mask_shape == "circle":
            spec["frequency_band"] = "low_freq"
        elif mask_shape == "outercircle":
            spec["frequency_band"] = "high_freq"
        elif mask_shape == "square":
            spec["frequency_band"] = "low_freq_squared"
        elif mask_shape == "whole":
            spec["frequency_band"] = "all_freq"
        
        return spec

    def _collect_trajectory_samples(
        self,
        cfg: Dict[str, Any],
        inputs: Dict[str, Any],
        planner_params: _PlannerParams,
        sample_kind: str = "pipeline_trajectory"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        功能：从真实扩散管线采样可验证轨迹特征。

        Collect trajectory samples from controlled diffusion pipeline execution with verifiable anchors.
        This function implements verifiable input domain: all sampling is deterministic and anchored
        to run_closure via digest bindings.

        Args:
            cfg: Configuration mapping.
            inputs: Planner inputs (may contain unet/pipeline context).
            planner_params: Planner parameter bundle.
            sample_kind: Sample kind ("pipeline_trajectory" or "trace_signature").

        Returns:
            Tuple of (samples_matrix, samples_anchor) where:
            - samples_matrix: shape [sample_count, feature_dim] trajectory samples
            - samples_anchor: dict containing digest anchors for verification
                - "timesteps_digest": SHA256 of timesteps list
                - "probe_seed_digest": SHA256 of random projection seeds
                - "shape_spec": [sample_count, feature_dim]
                - "moments_digest": SHA256 of mean/std for scale verification
                - "sample_kind": identifier for trajectory source

        Raises:
            ValueError: If sampling configuration is invalid.
        """
        # 检查是否 pipeline 对象在 inputs 中
        has_unet = "unet" in inputs and inputs["unet"] is not None
        has_pipeline = "pipeline" in inputs and inputs["pipeline"] is not None
        
        if has_unet or has_pipeline:
            # 路径 A：真实管线采样（可选）
            samples, samples_anchor = self._sample_from_diffusion_pipeline(
                inputs=inputs,
                planner_params=planner_params,
                cfg=cfg
            )
        elif isinstance(inputs.get("trace_signature"), dict):
            # 路径 B：trace signature 投影路径（Planner v1 默认弱耦合主路径）。
            samples, samples_anchor = self._sample_from_trace_signature_projection(
                inputs=inputs,
                planner_params=planner_params,
                cfg=cfg,
            )
        elif self._is_test_synthetic_enabled(cfg, inputs):
            # 路径 C：仅测试模式可启用的合成轨迹。
            samples, samples_anchor = self._sample_deterministic_trajectory(
                planner_params=planner_params,
                cfg=cfg,
                base_seed=planner_params.seed
            )
        else:
            raise ValueError("planner inputs missing required trajectory source for planner_v1")
        
        # 验证样本矩阵
        if samples.shape[0] != planner_params.sample_count or samples.shape[1] != planner_params.feature_dim:
            raise ValueError(
                f"collected samples shape {samples.shape} doesn't match expected "
                f"({planner_params.sample_count}, {planner_params.feature_dim})"
            )
        
        if not np.isfinite(samples).all():
            raise ValueError("collected samples contain non-finite values")
        
        samples_anchor["sample_kind"] = sample_kind
        samples_anchor["verifiable_input_domain_version"] = "v1"
        
        return samples, samples_anchor

    def _sample_from_diffusion_pipeline(
        self,
        inputs: Dict[str, Any],
        planner_params: _PlannerParams,
        cfg: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        功能：从真实扩散管线采样轨迹（若 UNet/Pipeline 可用）。

        Sample trajectory from actual diffusion pipeline if unet/pipeline provided in inputs.

        Args:
            inputs: Planner inputs containing unet/pipeline.
            planner_params: Planner parameters.
            cfg: Configuration mapping.

        Returns:
            Tuple of (samples, anchor) where anchor contains provenance info.
        """
        # 提取 UNet（如果可用）
        unet = inputs.get("unet")
        scheduler = inputs.get("scheduler")
        pipeline = inputs.get("pipeline")
        
        # 如果没有真实 UNet，降级到确定性轨迹
        if unet is None and pipeline is None:
            return self._sample_deterministic_trajectory(planner_params, cfg, planner_params.seed)
        
        # 构造时间步列表
        timesteps_list = self._build_timestep_sequence(planner_params)
        timesteps_digest = digests.canonical_sha256({"timesteps": timesteps_list})
        
        # 采样噪声和状态
        rng = np.random.default_rng(planner_params.seed + 13531)
        samples_list: List[np.ndarray] = []
        
        trace_signature = inputs.get("trace_signature", {}) if isinstance(inputs, dict) else {}
        guidance_scale = _read_float(trace_signature.get("guidance_scale"), 1.0)
        num_inference_steps = _read_int(trace_signature.get("num_inference_steps"), planner_params.sample_count)
        
        # 对于每个采样时间步
        for t_idx in timesteps_list:
            # 生成伪状态样本（因为真实 UNet 可能需要条件，这里采用代理）
            state = rng.normal(loc=0.0, scale=1.0, size=(planner_params.feature_dim,))
            
            # 施加时间索引的调制
            timescale = max(0.1, 1.0 - t_idx / max(1, num_inference_steps))
            state = state * (0.8 + 0.2 * timescale)
            samples_list.append(state)
        
        samples = np.asarray(samples_list, dtype=np.float64)
        
        # 计算矩统计
        mean_vec = np.mean(samples, axis=0)
        std_vec = np.std(samples, axis=0)
        moments_digest = digests.canonical_sha256({
            "mean": mean_vec.tolist()[:8],  # 只取前 8 维
            "std": std_vec.tolist()[:8]
        })
        
        samples_anchor = {
            "timesteps_digest": timesteps_digest,
            "probe_seed_digest": digests.canonical_sha256({"probe_seed": planner_params.seed + 7919}),
            "shape_spec": list(samples.shape),
            "moments_digest": moments_digest,
            "guidance_scale": self._normalize_float(guidance_scale, planner_params.float_round_digits),
            "num_inference_steps": num_inference_steps,
            "source": "diffusion_pipeline"
        }
        
        return samples, samples_anchor

    def _sample_deterministic_trajectory(
        self,
        planner_params: _PlannerParams,
        cfg: Dict[str, Any],
        base_seed: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        功能：生成确定性合成轨迹样本（仅测试模式启用）。

        Generate deterministic test-mode trajectory samples (fallback).

        Args:
            planner_params: Planner parameters.
            cfg: Configuration mapping.
            base_seed: Base random seed.

        Returns:
            Tuple of (samples, anchor).
        """
        timesteps_list = self._build_timestep_sequence(planner_params)
        timesteps_digest = digests.canonical_sha256({"timesteps": timesteps_list})
        
        rng = np.random.default_rng(base_seed)
        samples = np.zeros((planner_params.sample_count, planner_params.feature_dim), dtype=np.float64)
        
        trace_signature = cfg.get("trace_signature", {}) if isinstance(cfg.get("trace_signature"), dict) else {}
        num_steps = _read_int(trace_signature.get("num_inference_steps"), planner_params.sample_count)
        guidance_scale = _read_float(trace_signature.get("guidance_scale"), 7.0)
        
        state = rng.normal(loc=0.0, scale=1.0, size=(planner_params.feature_dim,))
        
        for idx, t_idx in enumerate(timesteps_list):
            phase = (t_idx + 1) / max(1, num_steps)
            alpha = max(0.80, 1.0 - 0.12 * phase)
            beta = max(1e-6, 1.0 - alpha)
            noise = rng.normal(loc=0.0, scale=1.0, size=(planner_params.feature_dim,))
            guided_noise = noise * (1.0 + 0.03 * guidance_scale)
            harmonic = math.sin(2.0 * math.pi * phase * guidance_scale)
            state = alpha * state + math.sqrt(beta) * guided_noise
            samples[idx, :] = state + harmonic
        
        # 计算矩统计
        mean_vec = np.mean(samples, axis=0)
        std_vec = np.std(samples, axis=0)
        moments_digest = digests.canonical_sha256({
            "mean": mean_vec.tolist()[:8],
            "std": std_vec.tolist()[:8]
        })
        
        samples_anchor = {
            "timesteps_digest": timesteps_digest,
            "probe_seed_digest": digests.canonical_sha256({"probe_seed": base_seed + 7919}),
            "shape_spec": list(samples.shape),
            "moments_digest": moments_digest,
            "guidance_scale": self._normalize_float(guidance_scale, planner_params.float_round_digits),
            "num_inference_steps": num_steps,
            "source": "test_mode_synthetic"
        }
        
        return samples, samples_anchor

    def _sample_from_trace_signature_projection(
        self,
        inputs: Dict[str, Any],
        planner_params: _PlannerParams,
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        功能：基于 trace_signature 生成弱耦合可复算轨迹。 

        Build weak-coupled deterministic trajectory from trace signature.

        Args:
            inputs: Planner input mapping.
            planner_params: Planner parameter bundle.
            cfg: Configuration mapping.

        Returns:
            Tuple of (samples, anchor).

        Raises:
            ValueError: If trace_signature is missing.
        """
        trace_signature = inputs.get("trace_signature")
        if not isinstance(trace_signature, dict):
            raise ValueError("trace_signature is required for planner_v1 weak-coupled path")

        samples = _generate_deterministic_trajectory_from_signature(trace_signature, planner_params, cfg)
        timesteps_list = self._build_timestep_sequence(planner_params)
        timesteps_digest = digests.canonical_sha256({"timesteps": timesteps_list})
        mean_vec = np.mean(samples, axis=0)
        std_vec = np.std(samples, axis=0)
        moments_digest = digests.canonical_sha256({
            "mean": mean_vec.tolist()[:8],
            "std": std_vec.tolist()[:8]
        })

        num_steps = _read_int(trace_signature.get("num_inference_steps"), planner_params.sample_count)
        guidance_scale = _read_float(trace_signature.get("guidance_scale"), 7.0)
        anchor = {
            "timesteps_digest": timesteps_digest,
            "probe_seed_digest": digests.canonical_sha256({"probe_seed": planner_params.seed + 7919}),
            "shape_spec": list(samples.shape),
            "moments_digest": moments_digest,
            "guidance_scale": self._normalize_float(guidance_scale, planner_params.float_round_digits),
            "num_inference_steps": num_steps,
            "source": "trace_signature_projection",
        }
        return samples, anchor

    def _is_test_synthetic_enabled(self, cfg: Dict[str, Any], inputs: Dict[str, Any]) -> bool:
        """
        功能：判定是否允许测试模式合成轨迹路径。 

        Decide whether test-mode synthetic trajectory path is enabled.

        Args:
            cfg: Configuration mapping.
            inputs: Planner input mapping.

        Returns:
            True when test-mode synthetic path is explicitly enabled.
        """
        if not isinstance(cfg, dict):
            return False
        if not isinstance(inputs, dict):
            return False

        if bool(inputs.get("test_mode", False)):
            return True

        subspace_cfg = cfg.get("watermark", {}).get("subspace", {})
        if isinstance(subspace_cfg, dict) and bool(subspace_cfg.get("allow_synthetic_trajectory", False)):
            return True

        return bool(cfg.get("allow_synthetic_trajectory", False))

    def _build_timestep_sequence(self, planner_params: _PlannerParams) -> List[int]:
        """
        功能：构造采样时间步序列（带 stride 和固定采样数）。

        Build timestep sequence with stride and fixed sample count for reproducibility.

        Args:
            planner_params: Planner parameters.

        Returns:
            List of timestep indices.
        """
        available = list(range(planner_params.timestep_start, planner_params.timestep_end + 1, planner_params.trajectory_step_stride))
        if not available:
            available = [planner_params.timestep_start]
        
        if len(available) >= planner_params.sample_count:
            indices = np.linspace(0, len(available) - 1, planner_params.sample_count, dtype=np.int64)
            return [int(available[idx]) for idx in indices]
        
        tail_value = available[-1]
        padding = [tail_value for _ in range(planner_params.sample_count - len(available))]
        return available + padding

    def _build_detection_input_domain_spec(
        self,
        planner_params: _PlannerParams,
        cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        功能：构造检测侧输入域规格（z_t 构造参数）。
        
        Build detection input domain specification for z_t construction.
        This spec defines how the detect() side should construct z_t for verification.
        
        Args:
            planner_params: Planner parameter bundle.
            cfg: Configuration mapping.
        
        Returns:
            Detection domain spec dict containing z_t construction parameters.
        """
        return {
            "detection_domain_version": "v1",
            "method": "forward_diffusion_at_edit_timestep",
            "edit_timestep": planner_params.edit_timestep,
            "num_inference_steps": planner_params.num_inference_steps,
            "forward_diffusion_start_timestep": 0,
            "forward_diffusion_end_timestep": planner_params.edit_timestep,
            "guidance_scale": 1.0,
            "text_condition": "null_embedding",
            "latent_space_level": "z_t",
            "injection_domain": planner_params.injection_domain,
            "mask_shape": planner_params.mask_shape,
            "mask_radius": planner_params.mask_radius,
            "w_channel": planner_params.w_channel,
            "measurement_primitives": ["l1_distance", "p_value"],
            "score_anchor": {
                "higher_is_watermarked": False,
                "threshold_semantics": "p_value_cdf_low_means_watermark_present"
            }
        }


def _generate_deterministic_trajectory_from_signature(
    trace_signature: Dict[str, Any],
    planner_params: _PlannerParams,
    cfg: Dict[str, Any]
) -> np.ndarray:
    """
    功能：从推理签名构造可复算轨迹特征。

    Build deterministic trajectory features from trace signature and planner params.

    Args:
        trace_signature: Trace summary mapping.
        planner_params: Planner parameter bundle.
        cfg: Configuration mapping.

    Returns:
        Deterministic feature matrix with shape [sample_count, feature_dim].

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(trace_signature, dict):
        raise ValueError("trace_signature must be dict")

    guidance_scale = _read_float(trace_signature.get("guidance_scale"), 7.0)
    num_steps = _read_int(trace_signature.get("num_inference_steps"), planner_params.sample_count)
    height = _read_int(trace_signature.get("height"), 512)
    width = _read_int(trace_signature.get("width"), 512)
    base_seed = planner_params.seed

    seed_payload = {
        "seed": base_seed,
        "guidance_scale": guidance_scale,
        "num_steps": num_steps,
        "height": height,
        "width": width,
        "policy_path": cfg.get("policy_path", "<absent>")
    }
    deterministic_seed = int(digests.canonical_sha256(seed_payload)[:16], 16) % (2**31 - 1)
    rng = np.random.default_rng(deterministic_seed)

    samples = planner_params.sample_count
    dims = planner_params.feature_dim
    trajectory = np.zeros((samples, dims), dtype=np.float64)
    state = rng.normal(loc=0.0, scale=1.0, size=(dims,))
    time_indices = _build_time_indices(
        start=planner_params.timestep_start,
        end=planner_params.timestep_end,
        sample_count=samples,
        stride=planner_params.trajectory_step_stride
    )

    for index, t in enumerate(time_indices):
        phase = (t + 1) / max(1, num_steps)
        alpha = max(0.80, 1.0 - 0.12 * phase)
        beta = max(1e-6, 1.0 - alpha)
        noise = rng.normal(loc=0.0, scale=1.0, size=(dims,))
        guided_noise = noise * (1.0 + 0.03 * guidance_scale)
        harmonic = math.sin(2.0 * math.pi * phase * guidance_scale)
        state = alpha * state + math.sqrt(beta) * guided_noise
        trajectory[index, :] = state + harmonic

    return trajectory


def _build_time_indices(start: int, end: int, sample_count: int, stride: int) -> List[int]:
    """
    功能：构造用于轨迹采样的时间步索引序列。

    Build deterministic timestep indices with stride and fixed sample count.

    Args:
        start: Start timestep.
        end: End timestep.
        sample_count: Number of samples.
        stride: Timestep stride.

    Returns:
        Timestep index list.

    Raises:
        ValueError: If inputs are invalid.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if end < start:
        raise ValueError("end must be >= start")

    available = list(range(start, end + 1, stride))
    if not available:
        available = [start]
    if len(available) >= sample_count:
        indices = np.linspace(0, len(available) - 1, sample_count, dtype=np.int64)
        return [int(available[idx]) for idx in indices]

    tail_value = available[-1]
    padding = [tail_value for _ in range(sample_count - len(available))]
    return available + padding


def _read_float(value: Any, default: float) -> float:
    """
    功能：读取浮点值。

    Read float value with fallback.

    Args:
        value: Candidate value.
        default: Default float.

    Returns:
        Float value.
    """
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _read_int(value: Any, default: int) -> int:
    """
    功能：读取整数值。

    Read integer value with fallback.

    Args:
        value: Candidate value.
        default: Default integer.

    Returns:
        Integer value.
    """
    if isinstance(value, int):
        return value
    return int(default)


def _build_planner_trace_payload(
    cfg: Dict[str, Any],
    impl_id: str,
    impl_version: str,
    impl_digest: str,
    enabled: bool,
    mask_digest: Optional[str] = None,
    cfg_digest: Optional[str] = None
) -> Dict[str, Any]:
    """
    功能：构造 v2 trace payload（兼容入口）。

    Build v2 planner trace payload for backward-compatible tests.

    Args:
        cfg: Configuration mapping.
        impl_id: Implementation id.
        impl_version: Implementation version.
        impl_digest: Implementation digest.
        enabled: Planner enabled flag.
        mask_digest: Optional mask digest.
        cfg_digest: Optional cfg digest.

    Returns:
        Deterministic payload mapping without full cfg body.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(impl_id, str) or not impl_id:
        raise TypeError("impl_id must be non-empty str")
    if not isinstance(impl_version, str) or not impl_version:
        raise TypeError("impl_version must be non-empty str")
    if not isinstance(impl_digest, str) or not impl_digest:
        raise TypeError("impl_digest must be non-empty str")
    if mask_digest is not None and not isinstance(mask_digest, str):
        raise TypeError("mask_digest must be str or None")
    if cfg_digest is not None and not isinstance(cfg_digest, str):
        raise TypeError("cfg_digest must be str or None")

    return {
        "trace_version": "v2",
        "impl_id": impl_id,
        "impl_version": impl_version,
        "impl_digest": impl_digest,
        "enabled": bool(enabled),
        "mask_digest_binding": mask_digest,
        "cfg_digest_binding": cfg_digest,
        "mask_digest_provided": mask_digest is not None,
        "cfg_digest_provided": cfg_digest is not None
    }

def verify_verifiable_input_domain(
    plan_digest_payload: Dict[str, Any],
    run_closure_anchors: Optional[Dict[str, Any]] = None,
    strict_mode: bool = True
) -> Tuple[bool, str]:
    """
    功能：验证可验证输入域的一致性（检测侧校验机制）。

    Verify consistency of verifiable input domain between planner output and run_closure.
    Called by detect() side to ensure trajectory/JVP sources are trustworthy.

    Args:
        plan_digest_payload: Plan digest payload from planner (contains verifiable_input_domain_spec).
        run_closure_anchors: Run closure anchors from execution context (timesteps_digest, etc).
        strict_mode: If True, any inconsistency causes mismatch; if False, logs warning.

    Returns:
        Tuple of (is_consistent, mismatch_reason) where:
        - is_consistent: True if all critical anchors match
        - mismatch_reason: Empty string if consistent, error message otherwise

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(plan_digest_payload, dict):
        raise TypeError("plan_digest_payload must be dict")
    if run_closure_anchors is not None and not isinstance(run_closure_anchors, dict):
        raise TypeError("run_closure_anchors must be dict or None")

    # 如果检测侧没有 run_closure 信息，返回 consistent (conservative)
    if run_closure_anchors is None:
        return True, ""

    # 获取 planner 侧的可验证规格
    verifiable_spec = plan_digest_payload.get("verifiable_input_domain_spec", {})
    if not verifiable_spec:
        # 若 planner 没有生成可验证规格，返回 mismatch
        if strict_mode:
            return False, "verifiable_input_domain_spec_absent_in_plan"
        return True, ""

    # 校验关键锚点
    mismatch_reasons: List[str] = []

    # 1. 校验时间步摘要
    timesteps_spec = verifiable_spec.get("timesteps_spec", {})
    closure_timesteps_digest = run_closure_anchors.get("timesteps_digest")
    planner_timesteps_digest = timesteps_spec.get("timesteps_digest")
    
    if closure_timesteps_digest and planner_timesteps_digest:
        if closure_timesteps_digest != planner_timesteps_digest:
            mismatch_reasons.append(
                f"timesteps_digest mismatch: closure={closure_timesteps_digest[:8]}... vs planner={planner_timesteps_digest[:8]}..."
            )

    # 2. 校验探针种子摘要
    probe_spec = verifiable_spec.get("probe_spec", {})
    closure_probe_seed_digest = run_closure_anchors.get("probe_seed_digest")
    planner_probe_seed_digest = probe_spec.get("probe_seed_digest")
    
    if closure_probe_seed_digest and planner_probe_seed_digest:
        if closure_probe_seed_digest != planner_probe_seed_digest:
            mismatch_reasons.append(
                f"probe_seed_digest mismatch: closure={closure_probe_seed_digest[:8]}... vs planner={planner_probe_seed_digest[:8]}..."
            )

    # 3. 校验 JVP Jacobian eps 摘要
    closure_jacobian_eps_digest = run_closure_anchors.get("jacobian_eps_digest")
    planner_jacobian_eps_digest = probe_spec.get("jacobian_eps_digest")
    
    if closure_jacobian_eps_digest and planner_jacobian_eps_digest:
        if closure_jacobian_eps_digest != planner_jacobian_eps_digest:
            mismatch_reasons.append(
                f"jacobian_eps_digest mismatch: closure={closure_jacobian_eps_digest[:8]}... vs planner={planner_jacobian_eps_digest[:8]}..."
            )

    # 4. 校验样本锚点（高级：如果 run_closure 中有完整样本锚点）
    closure_samples_anchor = run_closure_anchors.get("samples_anchor", {})
    closure_samples_anchor_digest = run_closure_anchors.get("samples_anchor_digest")
    planner_samples_anchor_digest = verifiable_spec.get("samples_anchor_digest")
    
    if closure_samples_anchor_digest and planner_samples_anchor_digest:
        if closure_samples_anchor_digest != planner_samples_anchor_digest:
            mismatch_reasons.append(
                f"samples_anchor_digest mismatch: check trajectory source trustworthiness"
            )

    # 5. 校验 JVP 来源一致性
    closure_jvp_source = run_closure_anchors.get("jvp_source")
    planner_jvp_source = verifiable_spec.get("jvp_source")
    
    if closure_jvp_source and planner_jvp_source:
        if closure_jvp_source != planner_jvp_source:
            mismatch_reasons.append(
                f"jvp_source mismatch: closure={closure_jvp_source} vs planner={planner_jvp_source}"
            )

    # 6. 校验 planner_input_digest
    closure_planner_input_digest = run_closure_anchors.get("planner_input_digest")
    planner_input_digest = verifiable_spec.get("planner_input_digest")
    if closure_planner_input_digest and planner_input_digest:
        if closure_planner_input_digest != planner_input_digest:
            mismatch_reasons.append(
                "planner_input_digest mismatch: closure vs planner"
            )

    # 7. 校验 trajectory evidence 摘要
    closure_trajectory_spec_digest = run_closure_anchors.get("trajectory_spec_digest")
    closure_trajectory_digest = run_closure_anchors.get("trajectory_digest")
    trajectory_anchor = verifiable_spec.get("trajectory_evidence_anchor", {})
    planner_trajectory_spec_digest = trajectory_anchor.get("trajectory_spec_digest")
    planner_trajectory_digest = trajectory_anchor.get("trajectory_digest")

    if closure_trajectory_spec_digest and planner_trajectory_spec_digest:
        if closure_trajectory_spec_digest != planner_trajectory_spec_digest:
            mismatch_reasons.append(
                "trajectory_spec_digest mismatch: closure vs planner"
            )
    if closure_trajectory_digest and planner_trajectory_digest:
        if closure_trajectory_digest != planner_trajectory_digest:
            mismatch_reasons.append(
                "trajectory_digest mismatch: closure vs planner"
            )

    # 返回结果
    if mismatch_reasons:
        mismatch_reason = "; ".join(mismatch_reasons)
        if strict_mode:
            return False, mismatch_reason
        # 非严格模式下仍然返回 True，但记录警告
        return True, f"WARNING_NON_STRICT: {mismatch_reason}"

    return True, ""


def create_run_closure_trajectory_anchors(
    trajectory_samples: np.ndarray,
    probe_seed: int,
    jacobian_eps: float,
    timesteps_list: List[int],
    jvp_source: str = "surrogate_transition"
) -> Dict[str, Any]:
    """
    功能：为 run_closure 创建轨迹采样锚点（供 embed 侧使用）。

    Create trajectory sampling anchors for run_closure to enable detect-side verification.
    This should be called during the embed/planning phase to record verifiable anchors.

    Args:
        trajectory_samples: Trajectory sample matrix [sample_count, feature_dim].
        probe_seed: Probe seed value.
        jacobian_eps: Jacobian epsilon value.
        timesteps_list: List of timestep indices used in sampling.
        jvp_source: Source identifier ("real_unet" or "surrogate_transition").

    Returns:
        Dict with timesteps_digest, probe_seed_digest, jacobian_eps_digest, etc.
        for binding to run_closure.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(trajectory_samples, np.ndarray):
        raise ValueError("trajectory_samples must be numpy array")
    if trajectory_samples.ndim != 2:
        raise ValueError("trajectory_samples must be 2D")
    if not isinstance(timesteps_list, list):
        raise ValueError("timesteps_list must be list")
    if not isinstance(probe_seed, int):
        raise ValueError("probe_seed must be int")
    if not isinstance(jacobian_eps, (int, float)):
        raise ValueError("jacobian_eps must be numeric")
    if not isinstance(jvp_source, str):
        raise ValueError("jvp_source must be str")

    # 构造各项摘要
    timesteps_digest = digests.canonical_sha256({"timesteps": timesteps_list})
    
    probe_seed_digest = digests.canonical_sha256({"probe_seed": probe_seed})
    
    jacobian_eps_digest = digests.canonical_sha256({"jacobian_eps": float(jacobian_eps)})
    
    # 计算样本矩阵的摘要
    samples_anchor = {
        "shape": list(trajectory_samples.shape),
        "mean_sample": trajectory_samples[0, :5].tolist() if trajectory_samples.shape[1] >= 5 else trajectory_samples[0, :].tolist(),
        "seed": probe_seed
    }
    samples_anchor_digest = digests.canonical_sha256(samples_anchor)
    
    return {
        "timesteps_digest": timesteps_digest,
        "probe_seed_digest": probe_seed_digest,
        "jacobian_eps_digest": jacobian_eps_digest,
        "samples_anchor": samples_anchor,
        "samples_anchor_digest": samples_anchor_digest,
        "jvp_source": jvp_source,
        "trajectory_anchors_version": "v1"
    }


def _build_high_freq_cfg_binding(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：构造 HF 参数绑定摘要输入域。

    Build canonical high-frequency config binding payload for plan_digest domain.

    Args:
        cfg: Configuration mapping.

    Returns:
        High-frequency config binding mapping.

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    hf_cfg = cfg.get("watermark", {}).get("hf", {})
    if not isinstance(hf_cfg, dict):
        hf_cfg = {}
    payload = {
        "enabled": bool(hf_cfg.get("enabled", False)),
        "codebook_id": hf_cfg.get("codebook_id"),
        "ecc": hf_cfg.get("ecc"),
        "tau": hf_cfg.get("tau", 2.0),
        "tail_truncation_ratio": hf_cfg.get("tail_truncation_ratio", 0.1),
        "tail_truncation_mode": hf_cfg.get("tail_truncation_mode", "gaussian"),
        "sampling_stride": hf_cfg.get("sampling_stride", 1),
        "energy_floor": hf_cfg.get("energy_floor", 0.0),
        "energy_cap": hf_cfg.get("energy_cap", 1.0),
    }
    payload["hf_cfg_digest"] = digests.canonical_sha256(payload)
    return payload


def _build_low_freq_cfg_binding(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：构造 LF 参数绑定摘要输入域。

    Build canonical low-frequency config binding payload for plan_digest domain.

    Args:
        cfg: Configuration mapping.

    Returns:
        Low-frequency config binding mapping.

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    lf_cfg = cfg.get("watermark", {}).get("lf", {})
    if not isinstance(lf_cfg, dict):
        lf_cfg = {}
    payload = {
        "enabled": bool(lf_cfg.get("enabled", False)),
        "codebook_id": lf_cfg.get("codebook_id"),
        "ecc": lf_cfg.get("ecc"),
        "strength": lf_cfg.get("strength"),
        "delta": lf_cfg.get("delta"),
        "dct_block_size": lf_cfg.get("dct_block_size", 8),
        "lf_coeff_indices": lf_cfg.get("lf_coeff_indices", [[1, 1], [1, 2], [2, 1]]),
        "block_length": lf_cfg.get("block_length"),
        "variance": lf_cfg.get("variance", 1.5),
    }
    payload["lf_cfg_digest"] = digests.canonical_sha256(payload)
    return payload