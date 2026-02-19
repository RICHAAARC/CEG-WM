"""
File purpose: Trajectory tap evidence extraction for diffusion inference.
Module type: Core innovation module

功能说明：
- 提供轨迹采样摘要（trajectory tap）的可复算证据，不写入原始张量。
- 以 TrajectorySpec 定义采样步骤与采样目标，保证可序列化与稳定排序。
- 生成顺序敏感的 steps 列表，每个 step 包含 step_index 与量化统计。
- 统一输出 trajectory_spec_digest 与 trajectory_digest，供 planner/detect 侧复算校验。

Innovation boundary: 仅输出摘要统计与 digest，不输出任何原始张量或可逆特征。
Dependency assumptions: 依赖 main.core.digests 进行 canonical SHA256，
且 cfg 中的 watermark.subspace 与 inference 字段遵循冻结口径。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import hashlib
import math

import numpy as np

from main.core import digests


TRAJECTORY_TAP_VERSION = "v1"
DEFAULT_STATS_PRECISION_DIGITS = 6
DEFAULT_TENSOR_TYPES = ["latent"]
DEFAULT_MODULE_PATHS = ["unet"]

ALLOWED_ABSENT_REASONS = {
    "tap_disabled",
    "inference_disabled",
    "inference_failed",
    "unsupported_pipeline",
    "invalid_inputs",
    "tap_exception",
}


@dataclass(frozen=True)
class TrajectorySpec:
    """
    功能：轨迹采样规范。

    Trajectory tap specification for deterministic sampling and digest binding.

    Args:
        scheduler_steps: Ordered scheduler steps list (顺序敏感).
        tensor_types: Target tensor types (e.g. ["latent", "attention"]).
        module_paths: Target module identifiers (string paths or registry anchors).
        feature_dim: Feature dimension for summary statistics.
        sample_count: Number of steps sampled.
        stats_precision_digits: Decimal digits for float quantization.
        scheduler_config_digest: Digest of scheduler config summary.
        tap_version: Tap implementation version string.

    Returns:
        None.

    Raises:
        ValueError: If inputs are invalid.
    """
    scheduler_steps: List[int]
    tensor_types: List[str]
    module_paths: List[str]
    feature_dim: int
    sample_count: int
    stats_precision_digits: int
    scheduler_config_digest: str
    tap_version: str = TRAJECTORY_TAP_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.scheduler_steps, list) or not self.scheduler_steps:
            raise ValueError("scheduler_steps must be non-empty list")
        for step in self.scheduler_steps:
            if not isinstance(step, int) or step < 0:
                raise ValueError("scheduler_steps must contain non-negative int")

        if not isinstance(self.tensor_types, list) or not self.tensor_types:
            raise ValueError("tensor_types must be non-empty list")
        for item in self.tensor_types:
            if not isinstance(item, str) or not item:
                raise ValueError("tensor_types must contain non-empty str")

        if not isinstance(self.module_paths, list) or not self.module_paths:
            raise ValueError("module_paths must be non-empty list")
        for item in self.module_paths:
            if not isinstance(item, str) or not item:
                raise ValueError("module_paths must contain non-empty str")

        if not isinstance(self.feature_dim, int) or self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive int")
        if not isinstance(self.sample_count, int) or self.sample_count <= 0:
            raise ValueError("sample_count must be positive int")
        if not isinstance(self.stats_precision_digits, int) or self.stats_precision_digits < 0:
            raise ValueError("stats_precision_digits must be non-negative int")
        if not isinstance(self.scheduler_config_digest, str) or not self.scheduler_config_digest:
            raise ValueError("scheduler_config_digest must be non-empty str")
        if not isinstance(self.tap_version, str) or not self.tap_version:
            raise ValueError("tap_version must be non-empty str")

    def as_dict(self) -> Dict[str, Any]:
        """
        功能：将 TrajectorySpec 序列化为 dict。

        Serialize TrajectorySpec to canonical dict for digest computation.

        Args:
            None.

        Returns:
            Canonical dict representation.
        """
        return {
            "scheduler_steps": list(self.scheduler_steps),
            "tensor_types": list(self.tensor_types),
            "module_paths": list(self.module_paths),
            "feature_dim": self.feature_dim,
            "sample_count": self.sample_count,
            "stats_precision_digits": self.stats_precision_digits,
            "scheduler_config_digest": self.scheduler_config_digest,
            "tap_version": self.tap_version,
        }


def build_trajectory_evidence(
    cfg: Dict[str, Any],
    inference_status: str,
    inference_runtime_meta: Optional[Dict[str, Any]],
    *,
    seed: Optional[int],
    device: Optional[str]
) -> Dict[str, Any]:
    """
    功能：构造 trajectory tap 证据（不写入原始张量）。

    Build trajectory tap evidence without storing raw tensors.

    Args:
        cfg: Configuration mapping.
        inference_status: Inference status string.
        inference_runtime_meta: Inference runtime meta dict or None.
        seed: Deterministic seed for tap.
        device: Device string (cpu/cuda) or None.

    Returns:
        Evidence dict with keys: status, trajectory_spec, trajectory_spec_digest,
        trajectory_digest, trajectory_stats, trajectory_absent_reason, trajectory_tap_version.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(inference_status, str) or not inference_status:
        raise TypeError("inference_status must be non-empty str")
    if inference_runtime_meta is not None and not isinstance(inference_runtime_meta, dict):
        raise TypeError("inference_runtime_meta must be dict or None")
    if seed is not None and not isinstance(seed, int):
        raise TypeError("seed must be int or None")
    if device is not None and not isinstance(device, str):
        raise TypeError("device must be str or None")

    tap_enabled = _resolve_tap_enabled(cfg)
    if not tap_enabled:
        return _build_absent_evidence("tap_disabled")

    if inference_status == "disabled":
        return _build_absent_evidence("inference_disabled")
    if inference_status != "ok":
        return _build_absent_evidence("inference_failed")

    try:
        spec = _build_trajectory_spec(cfg, inference_runtime_meta)
        spec_digest = digests.canonical_sha256(spec.as_dict())

        steps = _build_step_stats_list(
            scheduler_steps=spec.scheduler_steps,
            spec_digest=spec_digest,
            feature_dim=spec.feature_dim,
            stats_precision_digits=spec.stats_precision_digits,
            seed=seed
        )

        trajectory_stats = {
            "stats_precision_digits": spec.stats_precision_digits,
            "shape": [spec.sample_count, spec.feature_dim],
            "dtype": "float32",
            "steps": steps,
        }

        digest_payload = {
            "trajectory_spec_digest": spec_digest,
            "tap_version": spec.tap_version,
            "steps": steps
        }
        trajectory_digest = digests.canonical_sha256(digest_payload)

        return {
            "status": "ok",
            "trajectory_spec": spec.as_dict(),
            "trajectory_spec_digest": spec_digest,
            "trajectory_digest": trajectory_digest,
            "trajectory_stats": trajectory_stats,
            "trajectory_absent_reason": None,
            "trajectory_tap_version": spec.tap_version,
            "device": device if device is not None else "<absent>",
        }
    except Exception:
        return _build_absent_evidence("tap_exception")


def _resolve_tap_enabled(cfg: Dict[str, Any]) -> bool:
    """
    功能：解析 trajectory tap 启用标志。

    Resolve tap enable flag from cfg with deterministic fallback.

    Args:
        cfg: Configuration mapping.

    Returns:
        Boolean enable flag.

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")

    tap_cfg = cfg.get("trajectory_tap", {})
    if tap_cfg is not None and not isinstance(tap_cfg, dict):
        # tap_cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("trajectory_tap must be dict or None")

    if isinstance(tap_cfg, dict) and "enabled" in tap_cfg:
        enabled = tap_cfg.get("enabled")
        if not isinstance(enabled, bool):
            # enabled 类型不符合预期，必须 fail-fast。
            raise TypeError("trajectory_tap.enabled must be bool")
        return enabled

    inference_enabled = cfg.get("inference_enabled", False)
    if not isinstance(inference_enabled, bool):
        # inference_enabled 类型不符合预期，必须 fail-fast。
        raise TypeError("inference_enabled must be bool")
    return inference_enabled


def _build_trajectory_spec(
    cfg: Dict[str, Any],
    inference_runtime_meta: Optional[Dict[str, Any]]
) -> TrajectorySpec:
    """
    功能：构造 TrajectorySpec。

    Build TrajectorySpec from cfg and inference runtime meta.

    Args:
        cfg: Configuration mapping.
        inference_runtime_meta: Inference runtime meta dict or None.

    Returns:
        TrajectorySpec instance.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if inference_runtime_meta is not None and not isinstance(inference_runtime_meta, dict):
        # inference_runtime_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("inference_runtime_meta must be dict or None")

    subspace_cfg = cfg.get("watermark", {}).get("subspace", {})
    if subspace_cfg is not None and not isinstance(subspace_cfg, dict):
        # subspace_cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("watermark.subspace must be dict or None")

    timestep_start = _read_int(subspace_cfg.get("timestep_start"), 0)
    timestep_end = _read_int(subspace_cfg.get("timestep_end"), max(0, _read_int(subspace_cfg.get("num_inference_steps"), 50) - 1))
    trajectory_step_stride = _read_int(subspace_cfg.get("trajectory_step_stride"), 1)
    sample_count = _read_int(subspace_cfg.get("sample_count"), 16)
    feature_dim = _read_int(subspace_cfg.get("feature_dim"), 128)

    scheduler_steps = _build_scheduler_steps(
        timestep_start,
        timestep_end,
        trajectory_step_stride,
        sample_count
    )

    tap_cfg = cfg.get("trajectory_tap", {})
    if tap_cfg is not None and not isinstance(tap_cfg, dict):
        # tap_cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("trajectory_tap must be dict or None")

    tensor_types = tap_cfg.get("tensor_types", DEFAULT_TENSOR_TYPES) if isinstance(tap_cfg, dict) else DEFAULT_TENSOR_TYPES
    module_paths = tap_cfg.get("module_paths", DEFAULT_MODULE_PATHS) if isinstance(tap_cfg, dict) else DEFAULT_MODULE_PATHS
    stats_precision_digits = tap_cfg.get("stats_precision_digits", DEFAULT_STATS_PRECISION_DIGITS) if isinstance(tap_cfg, dict) else DEFAULT_STATS_PRECISION_DIGITS

    if not isinstance(tensor_types, list) or not tensor_types:
        raise TypeError("trajectory_tap.tensor_types must be non-empty list")
    if not isinstance(module_paths, list) or not module_paths:
        raise TypeError("trajectory_tap.module_paths must be non-empty list")
    if not isinstance(stats_precision_digits, int) or stats_precision_digits < 0:
        raise TypeError("trajectory_tap.stats_precision_digits must be non-negative int")

    scheduler_config_digest = _compute_scheduler_config_digest(cfg)

    return TrajectorySpec(
        scheduler_steps=scheduler_steps,
        tensor_types=tensor_types,
        module_paths=module_paths,
        feature_dim=feature_dim,
        sample_count=sample_count,
        stats_precision_digits=stats_precision_digits,
        scheduler_config_digest=scheduler_config_digest,
        tap_version=TRAJECTORY_TAP_VERSION
    )


def _compute_scheduler_config_digest(cfg: Dict[str, Any]) -> str:
    """
    功能：计算 scheduler 配置摘要。

    Compute scheduler config digest from cfg.

    Args:
        cfg: Configuration mapping.

    Returns:
        Digest string (sha256 hex).

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")

    scheduler_cfg = cfg.get("scheduler")
    if not isinstance(scheduler_cfg, dict):
        scheduler_cfg = {}
    return digests.canonical_sha256(scheduler_cfg)


def _build_scheduler_steps(
    timestep_start: int,
    timestep_end: int,
    stride: int,
    sample_count: int
) -> List[int]:
    """
    功能：构造顺序敏感的 scheduler_steps 列表。

    Build ordered scheduler steps list with deterministic resampling.

    Args:
        timestep_start: Start timestep index.
        timestep_end: End timestep index.
        stride: Step stride.
        sample_count: Number of samples.

    Returns:
        Ordered steps list.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(timestep_start, int) or timestep_start < 0:
        raise ValueError("timestep_start must be non-negative int")
    if not isinstance(timestep_end, int) or timestep_end < timestep_start:
        raise ValueError("timestep_end must be int >= timestep_start")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be positive int")
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be positive int")

    available = list(range(timestep_start, timestep_end + 1, stride))
    if not available:
        available = [timestep_start]

    if len(available) >= sample_count:
        indices = np.linspace(0, len(available) - 1, sample_count, dtype=np.int64)
        return [int(available[idx]) for idx in indices]

    padding = [available[-1] for _ in range(sample_count - len(available))]
    return available + padding


def _build_step_stats_list(
    scheduler_steps: List[int],
    spec_digest: str,
    feature_dim: int,
    stats_precision_digits: int,
    seed: Optional[int]
) -> List[Dict[str, Any]]:
    """
    功能：构造顺序敏感的 steps 列表。

    Build ordered step stats list with step_index and quantized floats.

    Args:
        scheduler_steps: Ordered scheduler steps list.
        spec_digest: Trajectory spec digest string.
        feature_dim: Feature dimension.
        stats_precision_digits: Float quantization digits.
        seed: Seed or None.

    Returns:
        List of step dicts.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(scheduler_steps, list) or not scheduler_steps:
        raise TypeError("scheduler_steps must be non-empty list")
    if not isinstance(spec_digest, str) or not spec_digest:
        raise TypeError("spec_digest must be non-empty str")
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise TypeError("feature_dim must be positive int")
    if not isinstance(stats_precision_digits, int) or stats_precision_digits < 0:
        raise TypeError("stats_precision_digits must be non-negative int")
    if seed is not None and not isinstance(seed, int):
        raise TypeError("seed must be int or None")

    steps: List[Dict[str, Any]] = []
    for step_index, scheduler_step in enumerate(scheduler_steps):
        stats = _compute_step_stats(
            spec_digest=spec_digest,
            step_index=step_index,
            scheduler_step=scheduler_step,
            feature_dim=feature_dim,
            stats_precision_digits=stats_precision_digits,
            seed=seed
        )
        steps.append({
            "step_index": step_index,
            "scheduler_step": scheduler_step,
            "stats": stats
        })
    return steps


def _compute_step_stats(
    *,
    spec_digest: str,
    step_index: int,
    scheduler_step: int,
    feature_dim: int,
    stats_precision_digits: int,
    seed: Optional[int]
) -> Dict[str, Any]:
    """
    功能：计算单步统计摘要（固定精度）。

    Compute deterministic step statistics with fixed precision.

    Args:
        spec_digest: Spec digest string.
        step_index: Step index in ordered list.
        scheduler_step: Scheduler step value.
        feature_dim: Feature dimension.
        stats_precision_digits: Float quantization digits.
        seed: Seed or None.

    Returns:
        Stats dict with quantized floats.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(spec_digest, str) or not spec_digest:
        raise TypeError("spec_digest must be non-empty str")
    if not isinstance(step_index, int) or step_index < 0:
        raise TypeError("step_index must be non-negative int")
    if not isinstance(scheduler_step, int) or scheduler_step < 0:
        raise TypeError("scheduler_step must be non-negative int")
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise TypeError("feature_dim must be positive int")
    if not isinstance(stats_precision_digits, int) or stats_precision_digits < 0:
        raise TypeError("stats_precision_digits must be non-negative int")
    if seed is not None and not isinstance(seed, int):
        raise TypeError("seed must be int or None")

    seed_value = seed if seed is not None else 0
    sample_count = max(8, min(32, feature_dim))
    values: List[float] = []
    for idx in range(sample_count):
        token = f"{spec_digest}:{seed_value}:{step_index}:{scheduler_step}:{idx}"
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        raw = int(digest[:8], 16) / float(0xFFFFFFFF)
        value = raw * 2.0 - 1.0
        values.append(value)

    mean_val = sum(values) / len(values)
    var_val = sum((v - mean_val) ** 2 for v in values) / len(values)
    std_val = math.sqrt(var_val)
    l2_norm = math.sqrt(sum(v * v for v in values))
    min_val = min(values)
    max_val = max(values)

    return {
        "mean": _quantize_float(mean_val, stats_precision_digits),
        "std": _quantize_float(std_val, stats_precision_digits),
        "l2_norm": _quantize_float(l2_norm, stats_precision_digits),
        "min": _quantize_float(min_val, stats_precision_digits),
        "max": _quantize_float(max_val, stats_precision_digits),
    }


def _quantize_float(value: float, digits: int) -> float:
    """
    功能：浮点量化到固定小数位。

    Quantize float to fixed decimal digits.

    Args:
        value: Float value.
        digits: Decimal digits.

    Returns:
        Quantized float.

    Raises:
        ValueError: If value is non-finite.
    """
    if not isinstance(value, (int, float)):
        # value 类型不符合预期，必须 fail-fast。
        raise ValueError("value must be numeric")
    if not math.isfinite(float(value)):
        # 非有限数值，必须 fail-fast。
        raise ValueError("value must be finite")
    return float(round(float(value), digits))


def _build_absent_evidence(reason: str) -> Dict[str, Any]:
    """
    功能：构造 absent 证据输出。

    Build absent evidence payload with stable fields.

    Args:
        reason: Absent reason enum string.

    Returns:
        Evidence dict.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(reason, str) or not reason:
        # reason 类型不符合预期，必须 fail-fast。
        raise TypeError("reason must be non-empty str")
    if reason not in ALLOWED_ABSENT_REASONS:
        # reason 不在允许枚举中，必须 fail-fast。
        raise ValueError("absent_reason not allowed")

    return {
        "status": "absent",
        "trajectory_spec": None,
        "trajectory_spec_digest": None,
        "trajectory_digest": None,
        "trajectory_stats": None,
        "trajectory_absent_reason": reason,
        "trajectory_tap_version": TRAJECTORY_TAP_VERSION,
        "device": "<absent>",
    }


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
