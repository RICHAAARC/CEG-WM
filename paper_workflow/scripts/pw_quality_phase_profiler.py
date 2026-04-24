"""
File purpose: Optional phase-level profiler for PW quality metric execution.
Module type: Semi-general module
"""

from __future__ import annotations

import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from scripts.notebook_runtime_common import write_json_atomic_compact


PHASE_PROFILE_ARTIFACT_TYPE = "paper_workflow_quality_phase_profile"
PHASE_PROFILE_SCHEMA_VERSION = "pw_quality_phase_profile_v1"
PHASE_PROFILE_SAMPLE_INTERVAL_SECONDS = 0.05
SUPPORTED_PHASE_NAMES = ("psnr_ssim", "lpips", "clip")


def _load_torch_module() -> Any | None:
    """
    功能：按需加载 torch 模块。.

    Load torch lazily for optional runtime profiling.

    Args:
        None.

    Returns:
        Imported torch module when available, otherwise None.
    """
    try:
        import torch
    except Exception:
        return None
    return torch


def _load_pynvml_module() -> Any | None:
    """
    功能：按需加载 pynvml 模块。.

    Load pynvml lazily for optional board-level sampling.

    Args:
        None.

    Returns:
        Imported pynvml module when available, otherwise None.
    """
    try:
        import pynvml  # type: ignore
    except Exception:
        return None
    return pynvml


def _mib_from_bytes(byte_value: int | float | None) -> float | None:
    """
    功能：将字节数转换为 MiB。.

    Convert one byte-sized value into MiB.

    Args:
        byte_value: Byte-sized value.

    Returns:
        MiB value rounded for JSON stability, or None when unavailable.
    """
    if not isinstance(byte_value, (int, float)):
        return None
    if float(byte_value) < 0.0:
        return None
    return round(float(byte_value) / (1024.0 * 1024.0), 6)


def _max_optional_float(lhs: float | None, rhs: float | None) -> float | None:
    """
    功能：返回两个可选浮点数中的较大值。.

    Return the larger value across two optional floats.

    Args:
        lhs: Existing value.
        rhs: Candidate value.

    Returns:
        Maximum available value, or None when both are unavailable.
    """
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return max(lhs, rhs)


def _parse_cuda_device_index(torch_device: str) -> int | None:
    """
    功能：从 torch device 字符串中解析 CUDA device index。.

    Parse the CUDA device index from one torch device string.

    Args:
        torch_device: Torch device string.

    Returns:
        Parsed CUDA device index when explicitly available, otherwise None.
    """
    if not isinstance(torch_device, str) or not torch_device.strip():
        raise TypeError("torch_device must be non-empty str")
    normalized_device = torch_device.strip().lower()
    if not normalized_device.startswith("cuda"):
        return None
    if ":" not in normalized_device:
        return None
    _, _, suffix = normalized_device.partition(":")
    try:
        return int(suffix)
    except ValueError:
        return None


@dataclass
class _BoardPhaseState:
    peak_memory_used_mib: float | None = None
    peak_gpu_utilization_percent: float | None = None
    peak_memory_utilization_percent: float | None = None
    sample_count: int = 0
    monitor_status: str = "not_available"


@dataclass
class _PhaseMetrics:
    phase_name: str
    phase_scope: str
    includes_text_encoding: bool = False
    measurement_mode: str = "full_phase_scope"
    gpu_measurement_mode: str = "full_phase_scope"
    invocation_count: int = 0
    sample_count: int = 0
    batch_count: int = 0
    batch_sizes: List[int] = field(default_factory=list)
    elapsed_seconds_total: float = 0.0
    elapsed_seconds_max: float = 0.0
    torch_peak_memory_allocated_mib: float | None = None
    torch_peak_memory_reserved_mib: float | None = None
    torch_memory_samples_count: int = 0
    board_monitor_status: str = "not_available"
    board_peak_memory_used_mib: float | None = None
    board_peak_gpu_utilization_percent: float | None = None
    board_peak_memory_utilization_percent: float | None = None
    board_samples_count: int = 0

    def to_payload(
        self,
        *,
        overall_elapsed_seconds: float,
        torch_cuda_available: bool,
        torch_device: str,
    ) -> Dict[str, Any]:
        """
        功能：导出单个 phase 的稳定 JSON 结构。.

        Export the stable JSON payload for one aggregated phase.

        Args:
            overall_elapsed_seconds: End-to-end elapsed seconds.
            torch_cuda_available: Whether torch reports CUDA availability.
            torch_device: Requested torch device string.

        Returns:
            JSON-serializable phase payload.
        """
        elapsed_seconds_mean = (
            round(self.elapsed_seconds_total / self.invocation_count, 6)
            if self.invocation_count > 0
            else 0.0
        )
        elapsed_seconds_total = round(self.elapsed_seconds_total, 6)
        elapsed_seconds_share_of_overall = (
            round(self.elapsed_seconds_total / overall_elapsed_seconds, 6)
            if overall_elapsed_seconds > 0.0
            else 0.0
        )
        return {
            "phase_name": self.phase_name,
            "phase_scope": self.phase_scope,
            "includes_text_encoding": self.includes_text_encoding,
            "measurement_mode": self.measurement_mode,
            "gpu_measurement_mode": self.gpu_measurement_mode,
            "invocation_count": int(self.invocation_count),
            "sample_count": int(self.sample_count),
            "batch_count": int(self.batch_count),
            "batch_sizes": [int(value) for value in self.batch_sizes],
            "elapsed_seconds_total": elapsed_seconds_total,
            "elapsed_seconds_mean": elapsed_seconds_mean,
            "elapsed_seconds_max": round(self.elapsed_seconds_max, 6),
            "elapsed_seconds_share_of_overall": elapsed_seconds_share_of_overall,
            "torch_cuda_available": bool(torch_cuda_available),
            "torch_device": str(torch_device),
            "torch_peak_memory_allocated_mib": self.torch_peak_memory_allocated_mib,
            "torch_peak_memory_reserved_mib": self.torch_peak_memory_reserved_mib,
            "torch_memory_samples_count": int(self.torch_memory_samples_count),
            "board_monitor_status": self.board_monitor_status,
            "board_peak_memory_used_mib": self.board_peak_memory_used_mib,
            "board_peak_gpu_utilization_percent": self.board_peak_gpu_utilization_percent,
            "board_peak_memory_utilization_percent": self.board_peak_memory_utilization_percent,
            "board_samples_count": int(self.board_samples_count),
        }


class _BoardMonitorSampler:
    """
    功能：在单个 phase 生命周期内采样板卡级 GPU 指标。.

    Sample board-level GPU metrics during one phase lifetime.

    Args:
        pynvml_module: Imported pynvml module.
        handle: NVML device handle.
        sample_interval_seconds: Sampling interval.

    Returns:
        None.
    """

    def __init__(
        self,
        *,
        pynvml_module: Any,
        handle: Any,
        sample_interval_seconds: float,
    ) -> None:
        if pynvml_module is None:
            raise TypeError("pynvml_module must not be None")
        if sample_interval_seconds <= 0.0:
            raise TypeError("sample_interval_seconds must be positive")
        self._pynvml = pynvml_module
        self._handle = handle
        self._sample_interval_seconds = float(sample_interval_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = _BoardPhaseState(monitor_status="nvml")

    def start(self) -> None:
        """
        功能：启动后台板卡采样线程。.

        Start the background board sampler thread.

        Args:
            None.

        Returns:
            None.
        """
        self._sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="pw_quality_phase_board_sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> _BoardPhaseState:
        """
        功能：停止后台板卡采样并返回聚合结果。.

        Stop the background board sampler and return the aggregate state.

        Args:
            None.

        Returns:
            Aggregated board-level phase state.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._sample_once()
        return self._state

    def _run(self) -> None:
        while not self._stop_event.wait(self._sample_interval_seconds):
            self._sample_once()

    def _sample_once(self) -> None:
        try:
            memory_info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        except Exception:
            if self._state.monitor_status == "nvml":
                self._state.monitor_status = "nvml_sampling_error"
            return

        memory_used_mib = _mib_from_bytes(getattr(memory_info, "used", None))
        memory_total_bytes = getattr(memory_info, "total", None)
        gpu_utilization = getattr(utilization, "gpu", None)
        memory_utilization_percent = None
        if isinstance(memory_total_bytes, (int, float)) and float(memory_total_bytes) > 0.0 and memory_used_mib is not None:
            memory_utilization_percent = round(float(getattr(memory_info, "used", 0)) * 100.0 / float(memory_total_bytes), 6)

        self._state.sample_count += 1
        self._state.peak_memory_used_mib = _max_optional_float(self._state.peak_memory_used_mib, memory_used_mib)
        self._state.peak_gpu_utilization_percent = _max_optional_float(
            self._state.peak_gpu_utilization_percent,
            float(gpu_utilization) if isinstance(gpu_utilization, (int, float)) else None,
        )
        self._state.peak_memory_utilization_percent = _max_optional_float(
            self._state.peak_memory_utilization_percent,
            memory_utilization_percent,
        )


class _PhaseScope(AbstractContextManager[None]):
    def __init__(
        self,
        profiler: "QualityPhaseProfiler",
        *,
        phase_name: str,
        sample_count: int,
        batch_size: int | None,
    ) -> None:
        self._profiler = profiler
        self._phase_name = phase_name
        self._sample_count = sample_count
        self._batch_size = batch_size
        self._started_at = 0.0
        self._board_sampler: _BoardMonitorSampler | None = None

    def __enter__(self) -> None:
        self._profiler._begin_phase(
            phase_name=self._phase_name,
            sample_count=self._sample_count,
            batch_size=self._batch_size,
        )
        self._started_at = time.perf_counter()
        self._board_sampler = self._profiler._start_board_sampler()
        return None

    def __exit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> None:
        elapsed_seconds = time.perf_counter() - self._started_at
        board_state = self._board_sampler.stop() if self._board_sampler is not None else None
        self._profiler._end_phase(
            phase_name=self._phase_name,
            elapsed_seconds=elapsed_seconds,
            board_state=board_state,
        )
        return None


class _NoopPhaseScope(AbstractContextManager[None]):
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> None:
        return None


class QualityPhaseProfiler:
    """
    功能：聚合 PW quality 主流程的阶段级 runtime 与显存信息。.

    Aggregate phase-level runtime and memory diagnostics for PW quality execution.

    Args:
        torch_device: Requested torch device string.
        output_path: Optional phase profile output path.
        label: Optional phase profile label.
        sample_interval_seconds: Board-level sampling interval.

    Returns:
        None.
    """

    def __init__(
        self,
        *,
        torch_device: str,
        output_path: Path | str | None = None,
        label: str | None = None,
        sample_interval_seconds: float = PHASE_PROFILE_SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        if not isinstance(torch_device, str) or not torch_device.strip():
            raise TypeError("torch_device must be non-empty str")
        if output_path is not None and not isinstance(output_path, (Path, str)):
            raise TypeError("output_path must be Path, str, or None")
        if label is not None and not isinstance(label, str):
            raise TypeError("label must be str or None")
        if sample_interval_seconds <= 0.0:
            raise TypeError("sample_interval_seconds must be positive")

        self._torch_device = torch_device.strip()
        self._output_path = Path(output_path).expanduser().resolve() if output_path is not None else None
        self._label = label.strip() if isinstance(label, str) and label.strip() else None
        self._sample_interval_seconds = float(sample_interval_seconds)
        self._torch_module = _load_torch_module()
        self._torch_cuda_available = bool(
            self._torch_module is not None
            and hasattr(self._torch_module, "cuda")
            and callable(getattr(self._torch_module.cuda, "is_available", None))
            and bool(self._torch_module.cuda.is_available())
        )
        self._device_is_cuda = self._torch_device.lower().startswith("cuda")
        self._phases: Dict[str, _PhaseMetrics] = {
            "psnr_ssim": _PhaseMetrics(
                phase_name="psnr_ssim",
                phase_scope="image_load + PSNR + SSIM per pair",
                includes_text_encoding=False,
                measurement_mode="aggregate_only",
                gpu_measurement_mode="not_measured_per_pair",
                board_monitor_status="cpu_only" if not self._device_is_cuda else "not_available",
            ),
            "lpips": _PhaseMetrics(
                phase_name="lpips",
                phase_scope="LPIPS batch flushes and single-item fallback path",
                includes_text_encoding=False,
                measurement_mode="full_phase_scope",
                gpu_measurement_mode="full_phase_scope",
                board_monitor_status="cpu_only" if not self._device_is_cuda else "not_available",
            ),
            "clip": _PhaseMetrics(
                phase_name="clip",
                phase_scope="CLIP image-text similarity batches, including text encoding on cache miss",
                includes_text_encoding=True,
                measurement_mode="full_phase_scope",
                gpu_measurement_mode="full_phase_scope",
                board_monitor_status="cpu_only" if not self._device_is_cuda else "not_available",
            ),
        }
        self._nvml_module: Any | None = None
        self._nvml_handle: Any | None = None
        self._nvml_initialized = False
        self._board_monitor_backend = "cpu_only"
        self._prepare_board_monitor()

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    def phase(
        self,
        phase_name: str,
        *,
        sample_count: int = 0,
        batch_size: int | None = None,
    ) -> AbstractContextManager[None]:
        """
        功能：返回一个阶段级 profiling context manager。.

        Return one phase-level profiling context manager.

        Args:
            phase_name: Stable phase name.
            sample_count: Number of logical samples covered by the invocation.
            batch_size: Batch size when the invocation is batch-based.

        Returns:
            Context manager recording timing and memory metrics.
        """
        if phase_name not in self._phases:
            raise ValueError(f"unsupported phase_name: {phase_name}")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 0:
            raise TypeError("sample_count must be non-negative int")
        if batch_size is not None and (not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0):
            raise TypeError("batch_size must be positive int or None")
        return _PhaseScope(
            self,
            phase_name=phase_name,
            sample_count=sample_count,
            batch_size=batch_size,
        )

    def record_aggregate_phase_timing(
        self,
        phase_name: str,
        *,
        elapsed_seconds: float,
        sample_count: int = 0,
        batch_count: int = 0,
        batch_size: int | None = None,
    ) -> None:
        """
        功能：轻量记录一次 phase 聚合观测而不进入完整 phase scope。

        Record one aggregate-only phase observation without entering the full
        phase scope and without triggering CUDA or board-level sampling.

        Args:
            phase_name: Stable phase name.
            elapsed_seconds: Measured wall-clock duration.
            sample_count: Number of logical samples covered by the observation.
            batch_count: Number of logical batches covered by the observation.
            batch_size: Optional batch size recorded for the observation.

        Returns:
            None.
        """
        if phase_name not in self._phases:
            raise ValueError(f"unsupported phase_name: {phase_name}")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 0:
            raise TypeError("sample_count must be non-negative int")
        if not isinstance(batch_count, int) or isinstance(batch_count, bool) or batch_count < 0:
            raise TypeError("batch_count must be non-negative int")
        if batch_size is not None and (not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0):
            raise TypeError("batch_size must be positive int or None")

        phase_metrics = self._phases[phase_name]
        phase_metrics.invocation_count += 1
        phase_metrics.sample_count += int(sample_count)
        phase_metrics.batch_count += int(batch_count)
        if batch_size is not None:
            append_count = int(batch_count) if batch_count > 0 else 1
            for _ in range(append_count):
                phase_metrics.batch_sizes.append(int(batch_size))
        normalized_elapsed_seconds = max(float(elapsed_seconds), 0.0)
        phase_metrics.elapsed_seconds_total += normalized_elapsed_seconds
        phase_metrics.elapsed_seconds_max = max(phase_metrics.elapsed_seconds_max, normalized_elapsed_seconds)

    def finalize(
        self,
        *,
        pair_spec_count: int,
        successful_pair_count: int,
        missing_count: int,
        error_count: int,
        elapsed_seconds_total: float,
    ) -> Dict[str, Any]:
        """
        功能：导出并可选写出最终 phase profile JSON。.

        Export and optionally persist the final phase profile JSON payload.

        Args:
            pair_spec_count: Total input pair count.
            successful_pair_count: Successfully evaluated pair count.
            missing_count: Missing binding or file count.
            error_count: Runtime error count.
            elapsed_seconds_total: End-to-end function elapsed seconds.

        Returns:
            JSON-serializable phase profile payload.
        """
        if not isinstance(pair_spec_count, int) or isinstance(pair_spec_count, bool) or pair_spec_count < 0:
            raise TypeError("pair_spec_count must be non-negative int")
        if not isinstance(successful_pair_count, int) or isinstance(successful_pair_count, bool) or successful_pair_count < 0:
            raise TypeError("successful_pair_count must be non-negative int")
        if not isinstance(missing_count, int) or isinstance(missing_count, bool) or missing_count < 0:
            raise TypeError("missing_count must be non-negative int")
        if not isinstance(error_count, int) or isinstance(error_count, bool) or error_count < 0:
            raise TypeError("error_count must be non-negative int")

        profile_payload = {
            "artifact_type": PHASE_PROFILE_ARTIFACT_TYPE,
            "schema_version": PHASE_PROFILE_SCHEMA_VERSION,
            "phase_profile_enabled": True,
            "phase_profile_label": self._label,
            "torch_device": self._torch_device,
            "monitor_backend": {
                "torch_cuda": "available" if self._torch_cuda_available else "not_available",
                "board_monitor": self._board_monitor_backend,
            },
            "overall": {
                "pair_spec_count": int(pair_spec_count),
                "successful_pair_count": int(successful_pair_count),
                "missing_count": int(missing_count),
                "error_count": int(error_count),
                "elapsed_seconds_total": round(float(elapsed_seconds_total), 6),
            },
            "phases": {
                phase_name: phase_metrics.to_payload(
                    overall_elapsed_seconds=float(elapsed_seconds_total),
                    torch_cuda_available=self._torch_cuda_available,
                    torch_device=self._torch_device,
                )
                for phase_name, phase_metrics in self._phases.items()
            },
        }
        if self._output_path is not None:
            write_json_atomic_compact(self._output_path, profile_payload)
        self._shutdown_nvml_if_needed()
        return profile_payload

    def _prepare_board_monitor(self) -> None:
        if not self._device_is_cuda:
            self._board_monitor_backend = "cpu_only"
            for phase_metrics in self._phases.values():
                phase_metrics.board_monitor_status = "cpu_only"
            return
        if not self._torch_cuda_available:
            self._board_monitor_backend = "not_available"
            for phase_metrics in self._phases.values():
                phase_metrics.board_monitor_status = "not_available"
            return

        pynvml_module = _load_pynvml_module()
        if pynvml_module is None:
            self._board_monitor_backend = "torch_only"
            for phase_metrics in self._phases.values():
                phase_metrics.board_monitor_status = "nvml_unavailable"
            return

        try:
            pynvml_module.nvmlInit()
            device_index = _parse_cuda_device_index(self._torch_device)
            if device_index is None and hasattr(self._torch_module.cuda, "current_device"):
                device_index = int(self._torch_module.cuda.current_device())
            if device_index is None:
                device_index = 0
            self._nvml_handle = pynvml_module.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            self._board_monitor_backend = "torch_only"
            for phase_metrics in self._phases.values():
                phase_metrics.board_monitor_status = "nvml_unavailable"
            return

        self._nvml_module = pynvml_module
        self._nvml_initialized = True
        self._board_monitor_backend = "nvml"
        for phase_metrics in self._phases.values():
            phase_metrics.board_monitor_status = "nvml"

    def _shutdown_nvml_if_needed(self) -> None:
        if not self._nvml_initialized or self._nvml_module is None:
            return
        try:
            self._nvml_module.nvmlShutdown()
        except Exception:
            return
        self._nvml_initialized = False

    def _start_board_sampler(self) -> _BoardMonitorSampler | None:
        if self._board_monitor_backend != "nvml" or self._nvml_module is None or self._nvml_handle is None:
            return None
        sampler = _BoardMonitorSampler(
            pynvml_module=self._nvml_module,
            handle=self._nvml_handle,
            sample_interval_seconds=self._sample_interval_seconds,
        )
        sampler.start()
        return sampler

    def _begin_phase(
        self,
        *,
        phase_name: str,
        sample_count: int,
        batch_size: int | None,
    ) -> None:
        phase_metrics = self._phases[phase_name]
        phase_metrics.invocation_count += 1
        phase_metrics.sample_count += int(sample_count)
        if batch_size is not None:
            phase_metrics.batch_count += 1
            phase_metrics.batch_sizes.append(int(batch_size))
        self._prepare_torch_phase_memory_tracking(phase_metrics)

    def _end_phase(
        self,
        *,
        phase_name: str,
        elapsed_seconds: float,
        board_state: _BoardPhaseState | None,
    ) -> None:
        phase_metrics = self._phases[phase_name]
        phase_metrics.elapsed_seconds_total += max(float(elapsed_seconds), 0.0)
        phase_metrics.elapsed_seconds_max = max(phase_metrics.elapsed_seconds_max, max(float(elapsed_seconds), 0.0))
        self._finalize_torch_phase_memory_tracking(phase_metrics)
        if board_state is None:
            return
        phase_metrics.board_monitor_status = board_state.monitor_status
        phase_metrics.board_samples_count += int(board_state.sample_count)
        phase_metrics.board_peak_memory_used_mib = _max_optional_float(
            phase_metrics.board_peak_memory_used_mib,
            board_state.peak_memory_used_mib,
        )
        phase_metrics.board_peak_gpu_utilization_percent = _max_optional_float(
            phase_metrics.board_peak_gpu_utilization_percent,
            board_state.peak_gpu_utilization_percent,
        )
        phase_metrics.board_peak_memory_utilization_percent = _max_optional_float(
            phase_metrics.board_peak_memory_utilization_percent,
            board_state.peak_memory_utilization_percent,
        )

    def _prepare_torch_phase_memory_tracking(self, phase_metrics: _PhaseMetrics) -> None:
        if not self._device_is_cuda or not self._torch_cuda_available or self._torch_module is None:
            return
        try:
            if hasattr(self._torch_module.cuda, "synchronize"):
                self._torch_module.cuda.synchronize(self._torch_device)
            if hasattr(self._torch_module.cuda, "reset_peak_memory_stats"):
                self._torch_module.cuda.reset_peak_memory_stats(self._torch_device)
            self._sample_torch_current_memory(phase_metrics)
        except Exception:
            return

    def _finalize_torch_phase_memory_tracking(self, phase_metrics: _PhaseMetrics) -> None:
        if not self._device_is_cuda or not self._torch_cuda_available or self._torch_module is None:
            return
        try:
            if hasattr(self._torch_module.cuda, "synchronize"):
                self._torch_module.cuda.synchronize(self._torch_device)
        except Exception:
            return

        self._sample_torch_current_memory(phase_metrics)
        self._sample_torch_peak_memory(phase_metrics)

    def _sample_torch_current_memory(self, phase_metrics: _PhaseMetrics) -> None:
        try:
            allocated_bytes = self._torch_module.cuda.memory_allocated(self._torch_device)
            reserved_bytes = self._torch_module.cuda.memory_reserved(self._torch_device)
        except Exception:
            return
        phase_metrics.torch_memory_samples_count += 1
        phase_metrics.torch_peak_memory_allocated_mib = _max_optional_float(
            phase_metrics.torch_peak_memory_allocated_mib,
            _mib_from_bytes(allocated_bytes),
        )
        phase_metrics.torch_peak_memory_reserved_mib = _max_optional_float(
            phase_metrics.torch_peak_memory_reserved_mib,
            _mib_from_bytes(reserved_bytes),
        )

    def _sample_torch_peak_memory(self, phase_metrics: _PhaseMetrics) -> None:
        try:
            max_allocated_bytes = self._torch_module.cuda.max_memory_allocated(self._torch_device)
            max_reserved_bytes = self._torch_module.cuda.max_memory_reserved(self._torch_device)
        except Exception:
            return
        phase_metrics.torch_memory_samples_count += 1
        phase_metrics.torch_peak_memory_allocated_mib = _max_optional_float(
            phase_metrics.torch_peak_memory_allocated_mib,
            _mib_from_bytes(max_allocated_bytes),
        )
        phase_metrics.torch_peak_memory_reserved_mib = _max_optional_float(
            phase_metrics.torch_peak_memory_reserved_mib,
            _mib_from_bytes(max_reserved_bytes),
        )


__all__ = [
    "PHASE_PROFILE_ARTIFACT_TYPE",
    "PHASE_PROFILE_SCHEMA_VERSION",
    "PHASE_PROFILE_SAMPLE_INTERVAL_SECONDS",
    "QualityPhaseProfiler",
]