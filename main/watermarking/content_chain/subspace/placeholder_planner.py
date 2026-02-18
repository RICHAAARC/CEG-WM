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
SUBSPACE_PLANNER_REAL_ID = "subspace_planner_real_v1"
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
            basis_summary = self._estimate_low_dim_subspace(cfg, inputs, planner_params)
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

            plan = {
                "plan_version": "v2",
                "subspace_method": "random_blocks",
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
                }
            }

            plan_stats = {
                "rank": basis_summary["rank"],
                "energy_ratio": basis_summary["energy_ratio"],
                "sample_count": basis_summary["sample_count"],
                "feature_dim": basis_summary["feature_dim"]
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
        planner_params: _PlannerParams
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
        if feature_matrix.shape[0] < 2 or feature_matrix.shape[1] < 2:
            raise ValueError("feature matrix must be at least 2x2")

        centered = feature_matrix - np.mean(feature_matrix, axis=0, keepdims=True)
        try:
            _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
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

        basis = vh[:effective_rank, :]
        basis_importance = np.mean(np.abs(basis), axis=0)
        top_index_count = min(32, basis_importance.shape[0])
        top_indices = np.argsort(-basis_importance)[:top_index_count].tolist()

        singular_preview = [
            self._normalize_float(float(value), planner_params.float_round_digits)
            for value in singular_values[: min(8, singular_values.shape[0])]
        ]

        null_space_dim = int(feature_matrix.shape[1] - effective_rank)
        subspace_spec = {
            "feature_source": self._infer_feature_source(inputs),
            "sample_count": int(feature_matrix.shape[0]),
            "feature_dim": int(feature_matrix.shape[1]),
            "timestep_window": [planner_params.timestep_start, planner_params.timestep_end],
            "top_feature_indices": top_indices,
            "singular_values_preview": singular_preview,
            "null_space_dim": null_space_dim,
            "seed": planner_params.seed
        }

        basis_digest_payload = {
            "basis_digest_version": "v1",
            "estimation_method": "svd_centered",
            "rank": int(effective_rank),
            "energy_ratio": energy_ratio,
            "null_space_dim": null_space_dim,
            "sample_count": int(feature_matrix.shape[0]),
            "feature_dim": int(feature_matrix.shape[1]),
            "subspace_spec": subspace_spec
        }

        return {
            "rank": int(effective_rank),
            "energy_ratio": energy_ratio,
            "null_space_dim": null_space_dim,
            "sample_count": int(feature_matrix.shape[0]),
            "feature_dim": int(feature_matrix.shape[1]),
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

        return {
            "plan_digest_version": "v2",
            "mask_digest": mask_digest,
            "cfg_digest": cfg_digest,
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
                "float_round_digits": planner_params.float_round_digits
            },
            "model_provenance_anchor": {
                "model_id": model_id,
                "model_revision": model_revision
            },
            "basis_digest": basis_digest,
            "basis_summary": {
                "rank": basis_summary["rank"],
                "energy_ratio": basis_summary["energy_ratio"],
                "null_space_dim": basis_summary["null_space_dim"],
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

        Parse and validate planner parameters from cfg.

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

        integer_fields = {
            "rank": rank,
            "sample_count": sample_count,
            "feature_dim": feature_dim,
            "timestep_start": timestep_start,
            "timestep_end": timestep_end,
            "seed": seed,
            "float_round_digits": float_round_digits
        }
        for field_name, field_value in integer_fields.items():
            if not isinstance(field_value, int):
                raise ValueError(f"{field_name} must be int")

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

        return _PlannerParams(
            rank=rank,
            sample_count=sample_count,
            feature_dim=feature_dim,
            timestep_start=timestep_start,
            timestep_end=timestep_end,
            seed=seed,
            float_round_digits=float_round_digits
        )

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
    for index in range(samples):
        t = planner_params.timestep_start + index
        if t > planner_params.timestep_end:
            t = planner_params.timestep_end
        base = rng.normal(loc=0.0, scale=1.0, size=(dims,))
        phase = (t + 1) / max(1, num_steps)
        modulation = math.sin(phase * math.pi * guidance_scale)
        trajectory[index, :] = base + modulation

    return trajectory


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
