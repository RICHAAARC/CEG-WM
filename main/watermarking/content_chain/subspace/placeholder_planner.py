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


ALLOWED_PLANNER_FAILURE_REASONS = {
    "planner_disabled_by_policy",
    "mask_absent",
    "planner_input_absent",
    "invalid_subspace_params",
    "decomposition_failed",
    "rank_computation_failed",
    "unknown"
}


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
            plan_payload = self._build_plan_payload_for_digest(
                cfg=cfg,
                cfg_digest=cfg_digest,
                mask_digest=mask_digest,
                planner_params=planner_params,
                basis_summary=basis_summary,
                basis_digest=basis_digest
            )
            plan_digest = self._derive_plan_digest(plan_payload)
            
            # 构造 detection_domain_spec
            detection_domain_spec = self._build_detection_input_domain_spec(
                planner_params=planner_params,
                cfg=cfg
            )

            plan = {
                "plan_version": "v3",
                "subspace_method": "trajectory_jacobian_nullspace_svd",
                "subspace_source": "planner_computed",
                "rank": basis_summary["rank"],
                "energy_ratio": basis_summary["energy_ratio"],
                "null_space_dim": basis_summary["null_space_dim"],
                "subspace_spec": basis_summary["subspace_spec"],
                "basis_digest": basis_digest,
                "planner_impl_identity": {
                    "impl_id": self.impl_id,
                    "impl_version": self.impl_version,
                    "impl_digest": self.impl_digest
                },
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
                "detection_domain_spec": detection_domain_spec
            }

            plan_stats = {
                "rank": basis_summary["rank"],
                "energy_ratio": basis_summary["energy_ratio"],
                "sample_count": basis_summary["sample_count"],
                "feature_dim": basis_summary["feature_dim"],
                "null_space_dim": basis_summary["null_space_dim"],
                "null_space_energy_ratio": basis_summary["null_space_energy_ratio"],
                "spectrum_summary": basis_summary["spectrum_summary"]
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
        功能：估计低维子空间并提取摘要。

        Estimate low-dimensional subspace from trajectory features by SVD.

        Args:
            cfg: Configuration mapping.
            inputs: Planner inputs containing feature trajectory.
            planner_params: Parsed planner parameters.

        Returns:
            Subspace summary mapping for digesting and records.

        Raises:
            ValueError: If feature matrix is invalid.
            RuntimeError: If decomposition fails.
        """
        feature_matrix = self._extract_feature_matrix(cfg, inputs, planner_params)
        feature_matrix = self._align_feature_matrix(feature_matrix, planner_params)
        if feature_matrix.shape[0] < 2 or feature_matrix.shape[1] < 2:
            raise ValueError("feature matrix must be at least 2x2")

        centered = feature_matrix - np.mean(feature_matrix, axis=0, keepdims=True)
        jacobian_samples = self._build_jacobian_probe_samples(
            centered_matrix=centered,
            planner_params=planner_params
        )
        decomposition_matrix = np.concatenate([centered, jacobian_samples], axis=0)
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
        sampling_strategy = {
            "source": self._infer_feature_source(inputs),
            "row_selection": "linspace_resample",
            "feature_projection": "variance_topk_or_zero_pad",
            "trajectory_step_stride": planner_params.trajectory_step_stride,
            "sample_count": planner_params.sample_count,
            "feature_dim": planner_params.feature_dim,
            "jacobian_probe_count": planner_params.jacobian_probe_count,
            "jacobian_eps": self._normalize_float(planner_params.jacobian_eps, planner_params.float_round_digits)
        }
        subspace_spec = {
            "feature_source": self._infer_feature_source(inputs),
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
                "num_inference_steps": _read_int(trace_signature.get("num_inference_steps"), planner_params.sample_count),
                "guidance_scale": self._normalize_float(_read_float(trace_signature.get("guidance_scale"), 7.0), planner_params.float_round_digits)
            }
        }

        basis_digest_payload = {
            "basis_digest_version": "v1",
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
            "subspace_spec": subspace_spec
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
            "basis_digest_payload": basis_digest_payload
        }

    def _build_plan_payload_for_digest(
        self,
        cfg: Dict[str, Any],
        cfg_digest: Optional[str],
        mask_digest: str,
        planner_params: _PlannerParams,
        basis_summary: Dict[str, Any],
        basis_digest: str
    ) -> Dict[str, Any]:
        """
        功能：构造 plan_digest 输入域。

        Build canonical payload for plan digest binding.

        Args:
            cfg: Configuration mapping.
            cfg_digest: Canonical cfg digest.
            mask_digest: Semantic mask digest.
            planner_params: Parsed planner parameters.
            basis_summary: Basis summary mapping.
            basis_digest: Basis digest string.

        Returns:
            Canonical payload mapping for digest.
        """
        model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
        model_id = model_cfg.get("model_id", cfg.get("model_id", "<absent>"))
        model_revision = model_cfg.get("model_revision", cfg.get("model_revision", "<absent>"))
        
        # 构造 mask_spec 摘要
        mask_spec = self._compute_mask_spec_from_shape(
            mask_shape=planner_params.mask_shape,
            mask_radius=planner_params.mask_radius,
            mask_radius2=planner_params.mask_radius2 if planner_params.mask_shape == "ring" else None
        )
        mask_spec_digest = digests.canonical_sha256(mask_spec)
        
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

        return {
            "plan_digest_version": "v3",
            "mask_digest": mask_digest,
            "mask_spec_digest": mask_spec_digest,
            "cfg_digest": cfg_digest,
            "planner_method": "trajectory_jacobian_nullspace_svd",
            "planner_impl_identity": {
                "impl_id": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest
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
                "jacobian_eps": self._normalize_float(planner_params.jacobian_eps, planner_params.float_round_digits)
            },
            "injection_config": injection_config_payload,
            "detection_domain_spec": detection_domain_spec,
            "detection_domain_spec_digest": detection_domain_spec_digest,
            "model_provenance_anchor": {
                "model_id": model_id,
                "model_revision": model_revision
            },
            "basis_digest": basis_digest,
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
        功能：解析并校验规划参数。

        Parse and validate planner parameters from config.

        Args:
            cfg: Configuration mapping.

        Returns:
            Parsed planner parameter bundle.

        Raises:
            ValueError: If parameters are invalid.
        """
        subspace_cfg = cfg.get("watermark", {}).get("subspace", {})
        
        rank = subspace_cfg.get("rank", subspace_cfg.get("k", 8))
        sample_count = subspace_cfg.get("sample_count", 16)
        feature_dim = subspace_cfg.get("feature_dim", 128)
        timestep_start = subspace_cfg.get("timestep_start", 0)
        timestep_end = subspace_cfg.get("timestep_end", 30)
        seed = subspace_cfg.get("seed", cfg.get("seed", 0))
        float_round_digits = subspace_cfg.get("float_round_digits", 8)
        trajectory_step_stride = subspace_cfg.get("trajectory_step_stride", 1)
        spectrum_topk = subspace_cfg.get("spectrum_topk", 8)
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

    def _build_jacobian_probe_samples(
        self,
        centered_matrix: np.ndarray,
        planner_params: _PlannerParams
    ) -> np.ndarray:
        """
        功能：构造扩散轨迹 Jacobian 近似样本。

        Build Jacobian-surrogate samples from adjacent trajectory differences.

        Args:
            centered_matrix: Centered trajectory matrix.
            planner_params: Parsed planner parameters.

        Returns:
            Jacobian-surrogate sample matrix.

        Raises:
            ValueError: If centered matrix is invalid.
        """
        if centered_matrix.ndim != 2:
            raise ValueError("centered_matrix must be 2D")
        if centered_matrix.shape[0] < 2:
            raise ValueError("centered_matrix must have at least two rows")

        x_t = centered_matrix[:-1, :]
        x_tp1 = centered_matrix[1:, :]
        transition_alpha = self._estimate_transition_alpha(x_t, x_tp1)
        transition_bias = np.mean(x_tp1 - transition_alpha * x_t, axis=0, keepdims=True)
        probes = self._build_deterministic_probe_vectors(
            feature_dim=centered_matrix.shape[1],
            probe_count=planner_params.jacobian_probe_count,
            seed=planner_params.seed
        )

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
        return jacobian_like

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

        Build normalized and quantized spectrum summary.

        Args:
            singular_values: Singular values from decomposition.
            effective_rank: Selected rank.
            float_round_digits: Float normalization digits.
            spectrum_topk: Number of normalized singular values to keep.

        Returns:
            Spectrum summary mapping.
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
        return {
            "topk_normalized": normalized_top,
            "total_components": int(singular_values.shape[0]),
            "tail_energy_ratio": self._normalize_float(tail_energy_ratio, float_round_digits),
            "stable_rank": self._normalize_float(stable_rank, float_round_digits)
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
        return "trace_signature"

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
