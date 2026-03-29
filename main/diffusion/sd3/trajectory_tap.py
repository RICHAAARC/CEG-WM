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
import inspect
import math

import numpy as np

from main.core import digests
from main.diffusion.sd3.callback_composer import compose_step_end_callbacks


TRAJECTORY_TAP_VERSION = "v1"
DEFAULT_STATS_PRECISION_DIGITS = 6
DEFAULT_TENSOR_TYPES = ["latent"]
DEFAULT_MODULE_PATHS = ["transformer"]
MAX_CAPTURE_FAILURE_EXAMPLES = 4

ALLOWED_ABSENT_REASONS = {
    "tap_disabled",
    "inference_disabled",
    "inference_failed",
    "unsupported_pipeline",
    "invalid_inputs",
    "tap_exception",
}


class LatentTrajectoryCache:
    """
    功能：推理过程中各时步 latent 张量的内存缓存（不写入 records）。

    In-memory cache for per-step latent tensors captured during diffusion inference.
    This class is intentionally separate from trajectory evidence and is never written
    to disk or included in any records schema.

    Args:
        None.

    Returns:
        None.
    """

    def __init__(self) -> None:
        # {step_index: numpy_array}，仅内存存储。
        self._cache: Dict[int, Any] = {}
        self._capture_attempt_count = 0
        self._capture_success_count = 0
        self._capture_failure_count = 0
        self._failure_examples: List[Dict[str, Any]] = []

    def _record_capture_failure(self, step_index: int, latent: Any, exc: BaseException) -> None:
        self._capture_failure_count += 1
        if len(self._failure_examples) >= MAX_CAPTURE_FAILURE_EXAMPLES:
            return
        self._failure_examples.append(
            {
                "step_index": int(step_index),
                "latent_type": type(latent).__name__,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        )

    def capture(self, step_index: int, latent: Any) -> None:
        """
        功能：缓存指定时步的 latent 张量副本。

        Store a copy of the latent tensor for the given step index.

        Args:
            step_index: Zero-based denoising step index.
            latent: Latent tensor (torch.Tensor or numpy array).

        Returns:
            None.
        """
        if not isinstance(step_index, int) or step_index < 0:
            return
        if latent is None:
            return
        self._capture_attempt_count += 1
        try:
            # 与 trajectory summary 保持一致的张量转 numpy 路径，避免 callback 已运行但 cache 无声丢失。
            arr = latent
            if hasattr(arr, "detach") and callable(arr.detach):
                arr = arr.detach()
            if hasattr(arr, "float") and callable(arr.float):
                arr = arr.float()
            if hasattr(arr, "cpu") and callable(arr.cpu):
                arr = arr.cpu()
            if hasattr(arr, "numpy") and callable(arr.numpy):
                arr = arr.numpy()
            else:
                arr = np.asarray(arr)
            latent_array = np.asarray(arr, dtype=np.float32)
            if latent_array.size == 0:
                raise ValueError("latent_array_empty")
            self._cache[step_index] = np.array(latent_array, dtype=np.float32, copy=True)
            self._capture_success_count += 1
        except Exception as exc:
            self._record_capture_failure(step_index, latent, exc)

    def capture_diagnostics(self) -> Dict[str, Any]:
        """
        功能：返回 trajectory latent cache 的结构化捕获诊断。 

        Return structured capture diagnostics for the in-memory latent cache.

        Args:
            None.

        Returns:
            Diagnostic mapping with attempt/success/failure counters.
        """
        return {
            "capture_attempt_count": int(self._capture_attempt_count),
            "capture_success_count": int(self._capture_success_count),
            "capture_failure_count": int(self._capture_failure_count),
            "failure_examples": [dict(item) for item in self._failure_examples],
            "available_steps": self.available_steps(),
            "step_count": len(self._cache),
            "is_empty": self.is_empty(),
        }

    def get(self, step_index: int) -> Optional[Any]:
        """
        功能：获取指定时步的 latent 缓存（无则返回 None）。

        Retrieve cached latent for the given step index, or None if absent.

        Args:
            step_index: Zero-based denoising step index.

        Returns:
            Numpy array of cached latent, or None.
        """
        return self._cache.get(step_index)

    def available_steps(self) -> List[int]:
        """
        功能：返回已缓存时步的有序列表。

        Return sorted list of cached step indices.

        Args:
            None.

        Returns:
            Sorted list of integer step indices.
        """
        return sorted(self._cache.keys())

    def is_empty(self) -> bool:
        """
        功能：判断缓存是否为空。

        Return True when no steps have been captured.

        Args:
            None.

        Returns:
            Boolean empty flag.
        """
        return len(self._cache) == 0


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
    device: Optional[str],
    tap_steps: Optional[List[Dict[str, Any]]] = None,
    trajectory_spec: Optional[TrajectorySpec] = None,
    absent_reason_override: Optional[str] = None
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
        trajectory_digest, trajectory_metrics, trajectory_stats, trajectory_absent_reason,
        trajectory_tap_version, and audit trajectory tap status fields.

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
        if absent_reason_override is not None:
            return _build_absent_evidence(absent_reason_override)

        spec = trajectory_spec if isinstance(trajectory_spec, TrajectorySpec) else _build_trajectory_spec(cfg, inference_runtime_meta)
        spec_digest = digests.canonical_sha256(spec.as_dict())

        steps = _normalize_tap_steps(tap_steps, spec)
        if not steps:
            return _build_absent_evidence("unsupported_pipeline")

        trajectory_metrics = {
            "stats_precision_digits": spec.stats_precision_digits,
            "shape": [len(steps), spec.feature_dim],
            "dtype": "float32",
            "steps": steps,
        }

        digest_payload = _build_digest_payload(
            spec_digest,
            spec.tap_version,
            steps
        )
        trajectory_digest = compute_trajectory_digest(digest_payload)

        return {
            "status": "ok",
            "trajectory_spec": spec.as_dict(),
            "trajectory_spec_digest": spec_digest,
            "trajectory_digest": trajectory_digest,
            "trajectory_metrics": trajectory_metrics,
            "trajectory_stats": trajectory_metrics,
            "trajectory_absent_reason": None,
            "trajectory_tap_version": spec.tap_version,
            "audit": {
                "trajectory_tap_status": "ok",
                "trajectory_absent_reason": None,
            },
            "device": device if device is not None else "<absent>",
        }
    except Exception:
        return _build_absent_evidence("tap_exception")


def _build_trajectory_cache_capture_meta(
    latent_capture_cache: Optional["LatentTrajectoryCache"],
    *,
    supports_callback: bool,
    callback_invocation_count: int,
    callback_latent_present_count: int,
    tap_captured_step_count: int,
    required_cache_steps: Optional[List[int]] = None,
) -> Optional[Dict[str, Any]]:
    """
    功能：构造 runtime trajectory cache 的结构化捕获状态。 

    Build structured runtime capture state for the latent cache.

    Args:
        latent_capture_cache: In-memory cache object or None.
        supports_callback: Whether the pipeline supports callback-based tapping.
        callback_invocation_count: Number of callback invocations observed.
        callback_latent_present_count: Number of callback invocations with latents.
        tap_captured_step_count: Number of tap summary steps captured.
        required_cache_steps: Exact planner-required step indices.

    Returns:
        Structured capture metadata mapping, or None when cache was not requested.
    """
    if latent_capture_cache is None:
        return None

    diagnostics = latent_capture_cache.capture_diagnostics()
    available_steps = diagnostics.get("available_steps")
    if not isinstance(available_steps, list):
        available_steps = []
    normalized_available_steps = [int(value) for value in available_steps if isinstance(value, int)]

    normalized_required_steps: List[int] = []
    if isinstance(required_cache_steps, list):
        normalized_required_steps = sorted(
            {
                int(value)
                for value in required_cache_steps
                if isinstance(value, int) and int(value) >= 0
            }
        )

    missing_required_steps = [
        step_index
        for step_index in normalized_required_steps
        if step_index not in normalized_available_steps
    ]
    capture_success_count = diagnostics.get("capture_success_count")
    capture_failure_count = diagnostics.get("capture_failure_count")
    capture_attempt_count = diagnostics.get("capture_attempt_count")
    failure_examples = diagnostics.get("failure_examples")

    if not isinstance(capture_success_count, int):
        capture_success_count = len(normalized_available_steps)
    if not isinstance(capture_failure_count, int):
        capture_failure_count = 0
    if not isinstance(capture_attempt_count, int):
        capture_attempt_count = int(capture_success_count) + int(capture_failure_count)
    if not isinstance(failure_examples, list):
        failure_examples = []

    if not supports_callback:
        capture_status = "unsupported_pipeline"
    elif callback_invocation_count <= 0:
        capture_status = "callback_not_observed"
    elif callback_latent_present_count <= 0:
        capture_status = "callback_invoked_without_latents"
    elif capture_success_count <= 0:
        capture_status = "all_failed"
    elif missing_required_steps:
        capture_status = "partial"
    else:
        capture_status = "complete"

    return {
        "trajectory_cache_capture_status": capture_status,
        "trajectory_cache_step_count": len(normalized_available_steps),
        "trajectory_cache_capture_attempt_count": int(capture_attempt_count),
        "trajectory_cache_capture_success_count": int(capture_success_count),
        "trajectory_cache_capture_failure_count": int(capture_failure_count),
        "trajectory_cache_capture_failure_examples": [dict(item) for item in failure_examples if isinstance(item, dict)],
        "trajectory_cache_available_steps": normalized_available_steps,
        "trajectory_cache_required_step_count": len(normalized_required_steps),
        "trajectory_cache_missing_required_steps": missing_required_steps,
        "trajectory_cache_callback_invocation_count": int(callback_invocation_count),
        "trajectory_cache_callback_latent_present_count": int(callback_latent_present_count),
        "trajectory_cache_tap_captured_step_count": int(tap_captured_step_count),
    }


def tap_from_pipeline(
    cfg: Dict[str, Any],
    pipeline_obj: Any,
    infer_kwargs: Dict[str, Any],
    inference_runtime_meta: Dict[str, Any],
    *,
    seed: Optional[int],
    device: Optional[str],
    latent_capture_cache: Optional["LatentTrajectoryCache"] = None
) -> Dict[str, Any]:
    """
    功能：在 SD3 推理循环中原位采样 trajectory 摘要，并可选地缓存原始 latent 张量。

    Tap real inference tensors from pipeline callback and build digest-only evidence.
    When latent_capture_cache is provided, each step's raw latent tensor is stored
    in memory for downstream planner use (not written to records).

    Args:
        cfg: Runtime configuration mapping.
        pipeline_obj: Diffusion pipeline object.
        infer_kwargs: Inference kwargs for pipeline call.
        inference_runtime_meta: Parsed runtime meta before pipeline execution.
        seed: Deterministic seed for reproducibility.
        device: Runtime device label.
        latent_capture_cache: Optional LatentTrajectoryCache for in-memory tensor storage.

    Returns:
        Mapping with output, trajectory_evidence, and tap_status.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(infer_kwargs, dict):
        raise TypeError("infer_kwargs must be dict")
    if not isinstance(inference_runtime_meta, dict):
        raise TypeError("inference_runtime_meta must be dict")

    spec = _build_trajectory_spec(cfg, inference_runtime_meta)

    signature = inspect.signature(pipeline_obj.__call__)
    supports_callback = (
        "callback_on_step_end" in signature.parameters and
        "callback_on_step_end_tensor_inputs" in signature.parameters
    )

    print(f"[Trajectory-TAP] Pipeline callback support check:")
    print(f"  - Pipeline type: {type(pipeline_obj).__name__}")
    print(f"  - Supports callback_on_step_end: {'callback_on_step_end' in signature.parameters}")
    print(f"  - Supports callback_on_step_end_tensor_inputs: {'callback_on_step_end_tensor_inputs' in signature.parameters}")
    print(f"  - Final supports_callback: {supports_callback}")

    if not supports_callback:
        # pipeline 不支持 callback 接口时，必须移除回调参数以避免报错。
        print(f"[Trajectory-TAP] [WARN] Pipeline does not support callback interface, using absent evidence")
        safe_infer_kwargs = dict(infer_kwargs)
        if "callback_on_step_end" in safe_infer_kwargs:
            safe_infer_kwargs.pop("callback_on_step_end", None)
        if "callback_on_step_end_tensor_inputs" in safe_infer_kwargs:
            safe_infer_kwargs.pop("callback_on_step_end_tensor_inputs", None)
        output = pipeline_obj(**safe_infer_kwargs)
        absent = build_trajectory_evidence(
            cfg,
            "ok",
            inference_runtime_meta,
            seed=seed,
            device=device,
            tap_steps=None,
            trajectory_spec=spec,
            absent_reason_override="unsupported_pipeline"
        )
        return {
            "output": output,
            "trajectory_evidence": absent,
            "tap_status": "unsupported",
            "trajectory_cache_capture_meta": _build_trajectory_cache_capture_meta(
                latent_capture_cache,
                supports_callback=False,
                callback_invocation_count=0,
                callback_latent_present_count=0,
                tap_captured_step_count=0,
                required_cache_steps=sorted(set(spec.scheduler_steps)),
            ),
        }

    captured: Dict[int, Dict[str, Any]] = {}
    callback_invocation_count = 0
    callback_latent_present_count = 0
    required_cache_steps = sorted(set(spec.scheduler_steps))

    def _step_callback(_pipe: Any, step_index: int, timestep: Any, callback_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal callback_invocation_count, callback_latent_present_count
        callback_invocation_count += 1
        if not isinstance(callback_kwargs, dict):
            return callback_kwargs

        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs
        callback_latent_present_count += 1

        if not isinstance(step_index, int) or step_index < 0:
            return callback_kwargs

        if step_index not in captured:
            captured[step_index] = {
                "step_index": step_index,
                "scheduler_step": _normalize_scheduler_step(step_index),
                "scheduler_timestep": _normalize_scheduler_step(timestep),
                "stats": _summarize_tensor(latents, spec.stats_precision_digits)
            }
        # 若调用方提供了缓存对象，则同时保存原始 latent 张量副本（不写入 records）。
        if latent_capture_cache is not None:
            latent_capture_cache.capture(step_index, latents)
        return callback_kwargs

    callback_kwargs = dict(infer_kwargs)
    existing_callback = callback_kwargs.get("callback_on_step_end")
    existing_tensor_inputs = callback_kwargs.get("callback_on_step_end_tensor_inputs")

    if existing_callback is not None:
        if not callable(existing_callback):
            # callback_on_step_end 类型不符合预期，必须 fail-fast。
            raise TypeError("callback_on_step_end must be callable or None")
        # 先执行现有回调，再执行采样回调（观察修改后的张量）。
        callback_kwargs["callback_on_step_end"] = compose_step_end_callbacks(
            existing_callback,
            _step_callback
        )
    else:
        callback_kwargs["callback_on_step_end"] = _step_callback

    if existing_tensor_inputs is None:
        tensor_inputs = []
    elif isinstance(existing_tensor_inputs, list):
        tensor_inputs = list(existing_tensor_inputs)
    elif isinstance(existing_tensor_inputs, tuple):
        tensor_inputs = list(existing_tensor_inputs)
    else:
        # callback_on_step_end_tensor_inputs 类型不符合预期，必须 fail-fast。
        raise TypeError("callback_on_step_end_tensor_inputs must be list, tuple, or None")

    if "latents" not in tensor_inputs:
        tensor_inputs.append("latents")
    callback_kwargs["callback_on_step_end_tensor_inputs"] = tensor_inputs

    output = pipeline_obj(**callback_kwargs)
    ordered_steps = _materialize_tap_steps(spec.scheduler_steps, captured)

    print(f"[Trajectory-TAP] Tap collection results:")
    print(f"  - Captured steps count: {len(captured)}")
    print(f"  - Ordered steps count: {len(ordered_steps) if ordered_steps else 0}")
    print(f"  - Expected scheduler steps: {len(spec.scheduler_steps) if spec.scheduler_steps else 0}")
    if not ordered_steps:
        print(f"  - [WARN] No steps collected! Will return absent evidence")

    evidence = build_trajectory_evidence(
        cfg,
        "ok",
        inference_runtime_meta,
        seed=seed,
        device=device,
        tap_steps=ordered_steps,
        trajectory_spec=spec
    )

    print(f"[Trajectory-TAP] Evidence build result: status={evidence.get('status')}, digest={evidence.get('trajectory_digest', '<absent>')[:16] if evidence.get('trajectory_digest') else '<None>'}")

    return {
        "output": output,
        "trajectory_evidence": evidence,
        "tap_status": "ok" if evidence.get("status") == "ok" else "absent",
        "trajectory_cache_capture_meta": _build_trajectory_cache_capture_meta(
            latent_capture_cache,
            supports_callback=True,
            callback_invocation_count=callback_invocation_count,
            callback_latent_present_count=callback_latent_present_count,
            tap_captured_step_count=len(captured),
            required_cache_steps=required_cache_steps,
        ),
    }


def _resolve_tap_enabled(cfg: Dict[str, Any]) -> bool:
    """
    功能：解析 trajectory tap 启用标志。

    Resolve tap enable flag from cfg with paper_faithfulness mode default.

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

    # (1) 如果 trajectory_tap.enabled 显式设置，则使用该值
    if isinstance(tap_cfg, dict) and "enabled" in tap_cfg:
        enabled = tap_cfg.get("enabled")
        if not isinstance(enabled, bool):
            # enabled 类型不符合预期，必须 fail-fast。
            raise TypeError("trajectory_tap.enabled must be bool")
        return enabled

    # (2) 否则，检查 paper_faithfulness 是否启用
    # 当 paper_faithfulness 启用时，trajectory tap 默认也应启用（paper mode 下必须采样轨迹）
    paper_faithfulness_cfg = cfg.get("paper_faithfulness", {})
    if isinstance(paper_faithfulness_cfg, dict):
        pf_enabled = paper_faithfulness_cfg.get("enabled", False)
        if pf_enabled:
            # paper_faithfulness 启用 → tap 默认启用（除非显式禁用）
            return True

    # (3) 都没有设置时，回退到 inference_enabled（传统逻辑）
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

    runtime_num_steps = _read_int(
        inference_runtime_meta.get("num_inference_steps") if isinstance(inference_runtime_meta, dict) else None,
        50
    )
    timestep_start = _read_int(subspace_cfg.get("timestep_start"), 0)
    timestep_end = _read_int(
        subspace_cfg.get("timestep_end"),
        max(0, _read_int(subspace_cfg.get("num_inference_steps"), runtime_num_steps) - 1)
    )
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


def _normalize_tap_steps(
    tap_steps: Optional[List[Dict[str, Any]]],
    spec: TrajectorySpec
) -> List[Dict[str, Any]]:
    """
    功能：规范化 tap 步摘要列表。

    Normalize externally captured tap steps to canonical structure.

    Args:
        tap_steps: Captured raw tap steps.
        spec: Trajectory specification.

    Returns:
        Canonical step list.
    """
    if tap_steps is None:
        return []
    if not isinstance(tap_steps, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(tap_steps):
        if not isinstance(item, dict):
            continue
        stats = item.get("stats")
        if not isinstance(stats, dict):
            continue
        normalized.append({
            "step_index": _normalize_scheduler_step(item.get("step_index", index)),
            "scheduler_step": _normalize_scheduler_step(item.get("scheduler_step", spec.scheduler_steps[min(index, len(spec.scheduler_steps) - 1)])),
            "stats": _normalize_stats_mapping(stats, spec.stats_precision_digits)
        })
    return normalized


def _normalize_stats_mapping(stats: Dict[str, Any], digits: int) -> Dict[str, float]:
    """
    功能：规范化统计字段为固定精度浮点。

    Normalize statistic mapping into fixed-precision float fields.

    Args:
        stats: Raw statistics mapping.
        digits: Quantization digits.

    Returns:
        Canonical statistics mapping.
    """
    return {
        "mean": _quantize_float(float(stats.get("mean", 0.0)), digits),
        "std": _quantize_float(float(stats.get("std", 0.0)), digits),
        "l2_norm": _quantize_float(float(stats.get("l2_norm", 0.0)), digits),
        "min": _quantize_float(float(stats.get("min", 0.0)), digits),
        "max": _quantize_float(float(stats.get("max", 0.0)), digits),
    }


def _materialize_tap_steps(
    scheduler_steps: List[int],
    captured: Dict[int, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    功能：按 spec 顺序构建 steps 列表（顺序敏感）。

    Build ordered step list according to spec scheduler steps.

    Args:
        scheduler_steps: Ordered scheduler step indices from spec.
        captured: Captured step summaries by step index.

    Returns:
        Ordered canonical step list.
    """
    if not scheduler_steps or not captured:
        return []

    materialized: List[Dict[str, Any]] = []
    last_stats: Optional[Dict[str, Any]] = None
    for output_index, scheduler_step in enumerate(scheduler_steps):
        captured_item = captured.get(scheduler_step)
        if captured_item is not None:
            last_stats = captured_item.get("stats")
            materialized.append({
                "step_index": output_index,
                "scheduler_step": scheduler_step,
                "stats": captured_item.get("stats")
            })
        elif last_stats is not None:
            materialized.append({
                "step_index": output_index,
                "scheduler_step": scheduler_step,
                "stats": last_stats
            })
    return materialized


def _normalize_scheduler_step(value: Any) -> int:
    """
    功能：归一化 scheduler step 为整数。

    Normalize scheduler step value to integer.

    Args:
        value: Step-like value.

    Returns:
        Non-negative integer step.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, np.integer)):
        return max(0, int(value))
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _summarize_tensor(tensor_value: Any, stats_precision_digits: int) -> Dict[str, float]:
    """
    功能：计算张量摘要统计并做固定精度量化。

    Summarize tensor into deterministic scalar statistics with fixed precision.

    Args:
        tensor_value: Tensor-like object (torch.Tensor or numpy-compatible).
        stats_precision_digits: Decimal digits for quantization.

    Returns:
        Summary statistics mapping.

    Raises:
        ValueError: If input tensor cannot be summarized.
    """
    if isinstance(tensor_value, np.ndarray):
        array = tensor_value
    else:
        if hasattr(tensor_value, "detach") and callable(tensor_value.detach):
            tensor_value = tensor_value.detach()
        if hasattr(tensor_value, "float") and callable(tensor_value.float):
            tensor_value = tensor_value.float()
        if hasattr(tensor_value, "cpu") and callable(tensor_value.cpu):
            tensor_value = tensor_value.cpu()
        if hasattr(tensor_value, "numpy") and callable(tensor_value.numpy):
            array = tensor_value.numpy()
        else:
            array = np.asarray(tensor_value)

    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    if array.size == 0:
        raise ValueError("tensor_value must be non-empty")

    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    if not np.isfinite(flat).all():
        raise ValueError("tensor_value contains non-finite values")

    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    l2_norm = float(np.linalg.norm(flat, ord=2))
    min_val = float(np.min(flat))
    max_val = float(np.max(flat))

    return {
        "mean": _quantize_float(mean_val, stats_precision_digits),
        "std": _quantize_float(std_val, stats_precision_digits),
        "l2_norm": _quantize_float(l2_norm, stats_precision_digits),
        "min": _quantize_float(min_val, stats_precision_digits),
        "max": _quantize_float(max_val, stats_precision_digits),
    }


def _build_digest_payload(
    trajectory_spec_digest: str,
    tap_version: str,
    steps: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    功能：构造顺序敏感 trajectory digest 载体。

    Build order-sensitive digest payload from canonical step sequence.

    Args:
        trajectory_spec_digest: Digest of trajectory specification.
        tap_version: Tap implementation version.
        steps: Ordered step list.

    Returns:
        Digest payload mapping.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(trajectory_spec_digest, str) or not trajectory_spec_digest:
        raise TypeError("trajectory_spec_digest must be non-empty str")
    if not isinstance(tap_version, str) or not tap_version:
        raise TypeError("tap_version must be non-empty str")
    if not isinstance(steps, list):
        raise TypeError("steps must be list")

    normalized_steps: List[Dict[str, Any]] = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        stats = item.get("stats", {})
        if not isinstance(stats, dict):
            continue
        normalized_steps.append({
            "step_index": _normalize_scheduler_step(item.get("step_index", 0)),
            "scheduler_step": _normalize_scheduler_step(item.get("scheduler_step", 0)),
            "stats": {
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "l2_norm": stats.get("l2_norm"),
                "min": stats.get("min"),
                "max": stats.get("max")
            }
        })

    return {
        "trajectory_spec_digest": trajectory_spec_digest,
        "tap_version": tap_version,
        "steps": normalized_steps
    }


def compute_trajectory_digest(payload: Dict[str, Any]) -> str:
    """
    功能：计算 trajectory payload 的 canonical sha256。

    Compute trajectory digest using repository canonical JSON + SHA256.

    Args:
        payload: Canonical trajectory payload mapping.

    Returns:
        SHA256 hex digest string.

    Raises:
        TypeError: If payload is invalid.
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be dict")
    return digests.canonical_sha256(payload)


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
        "trajectory_metrics": None,
        "trajectory_stats": None,
        "trajectory_absent_reason": reason,
        "trajectory_tap_version": TRAJECTORY_TAP_VERSION,
        "audit": {
            "trajectory_tap_status": "absent",
            "trajectory_absent_reason": reason,
        },
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
