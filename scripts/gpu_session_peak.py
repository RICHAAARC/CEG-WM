"""
文件目的：为 Notebook 顶层命令提供独立 GPU 会话峰值观测包装器。
Module type: General module

职责边界：
1. 仅在被包装命令执行窗口内采样 nvidia-smi 视角下的板卡显存占用，并写出独立 JSON 摘要。
2. 仅作为 Notebook / Colab 观测层工具，不参与 main/ 下正式阈值、评分、融合、attestation 或 records 语义。
3. 保持被包装命令的 stdout、stderr 与返回码语义，不向 stdout 注入任何监控摘要。
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.notebook_runtime_common import utc_now_iso, write_json_atomic


NVIDIA_SMI_QUERY_FIELDS = "index,uuid,name,memory.used,memory.total"
NVIDIA_SMI_QUERY_ARGS = [
    f"--query-gpu={NVIDIA_SMI_QUERY_FIELDS}",
    "--format=csv,noheader,nounits",
]


@dataclass(frozen=True)
class _GpuObservation:
    index: int
    uuid: str
    name: str
    memory_used_mib: int
    memory_total_mib: int


@dataclass(frozen=True)
class _SampleRecord:
    observed_at_utc: str
    board_memory_used_mib: int
    peak_gpu_index: int
    peak_gpu_uuid: str
    peak_gpu_name: str
    gpus: List[_GpuObservation]


def _positive_int(value: str) -> int:
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError("value must be non-empty")
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"value must be an integer: {value}") from exc
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(f"value must be positive: {parsed_value}")
    return parsed_value


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor session-scoped GPU board memory with nvidia-smi while executing one wrapped command.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Destination JSON summary path.",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Stable label for the wrapped notebook or workflow entrypoint.",
    )
    parser.add_argument(
        "--sample-interval-ms",
        type=_positive_int,
        default=200,
        help="Sampling interval in milliseconds.",
    )
    parser.add_argument(
        "wrapped_command",
        nargs=argparse.REMAINDER,
        help="Wrapped command placed after '--'.",
    )
    args = parser.parse_args(argv)
    wrapped_command = list(args.wrapped_command)
    if wrapped_command and wrapped_command[0] == "--":
        wrapped_command = wrapped_command[1:]
    if not wrapped_command:
        parser.error("wrapped command is required after '--'")
    args.wrapped_command = wrapped_command
    return args


def _run_nvidia_smi_query(nvidia_smi_path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [nvidia_smi_path, *NVIDIA_SMI_QUERY_ARGS],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _parse_nvidia_smi_output(stdout_text: str) -> List[_GpuObservation]:
    if not isinstance(stdout_text, str):
        raise TypeError("stdout_text must be str")
    stripped_text = stdout_text.strip()
    if not stripped_text:
        return []

    rows = csv.reader(stripped_text.splitlines())
    observations: List[_GpuObservation] = []
    for row in rows:
        normalized_row = [column.strip() for column in row]
        if len(normalized_row) != 5:
            raise ValueError(f"unexpected nvidia-smi row shape: {normalized_row}")
        observations.append(
            _GpuObservation(
                index=int(normalized_row[0]),
                uuid=normalized_row[1],
                name=normalized_row[2],
                memory_used_mib=int(normalized_row[3]),
                memory_total_mib=int(normalized_row[4]),
            )
        )
    return observations


def _collect_sample(nvidia_smi_path: str) -> _SampleRecord:
    completed_process = _run_nvidia_smi_query(nvidia_smi_path)
    if completed_process.returncode != 0:
        stderr_text = (completed_process.stderr or "").strip()
        raise RuntimeError(
            f"nvidia-smi query failed with exit code {completed_process.returncode}: {stderr_text or '<empty stderr>'}"
        )

    observations = _parse_nvidia_smi_output(completed_process.stdout or "")
    if not observations:
        raise RuntimeError("nvidia-smi query returned no visible GPU rows")

    peak_observation = max(observations, key=lambda item: item.memory_used_mib)
    return _SampleRecord(
        observed_at_utc=utc_now_iso(),
        board_memory_used_mib=peak_observation.memory_used_mib,
        peak_gpu_index=peak_observation.index,
        peak_gpu_uuid=peak_observation.uuid,
        peak_gpu_name=peak_observation.name,
        gpus=observations,
    )


def _update_per_gpu_peaks(
    per_gpu_state: Dict[str, Dict[str, Any]],
    sample_record: _SampleRecord,
) -> None:
    for gpu_observation in sample_record.gpus:
        gpu_key = gpu_observation.uuid or str(gpu_observation.index)
        existing_payload = per_gpu_state.get(gpu_key)
        peak_memory_used_mib = gpu_observation.memory_used_mib
        if existing_payload is not None:
            peak_memory_used_mib = max(
                int(existing_payload["peak_memory_used_mib"]),
                gpu_observation.memory_used_mib,
            )
        per_gpu_state[gpu_key] = {
            "index": gpu_observation.index,
            "uuid": gpu_observation.uuid,
            "name": gpu_observation.name,
            "memory_total_mib": gpu_observation.memory_total_mib,
            "peak_memory_used_mib": peak_memory_used_mib,
        }


def _build_visible_gpu_payload(per_gpu_state: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    visible_gpus = [dict(payload) for payload in per_gpu_state.values()]
    visible_gpus.sort(key=lambda item: (int(item["index"]), str(item["uuid"])))
    return visible_gpus


def _capture_sample_if_available(
    nvidia_smi_path: Optional[str],
    per_gpu_state: Dict[str, Dict[str, Any]],
    first_error_state: Dict[str, Optional[str]],
    sample_error_state: Dict[str, int],
    successful_sample_state: Dict[str, int],
    current_peak: Dict[str, Optional[_SampleRecord]],
) -> Optional[_SampleRecord]:
    if not nvidia_smi_path:
        return None
    try:
        sample_record = _collect_sample(nvidia_smi_path)
    except Exception as exc:
        sample_error_state["count"] += 1
        if first_error_state["message"] is None:
            first_error_state["message"] = f"{type(exc).__name__}: {exc}"
        return None

    successful_sample_state["count"] += 1
    _update_per_gpu_peaks(per_gpu_state, sample_record)
    previous_peak = current_peak["record"]
    if previous_peak is None or sample_record.board_memory_used_mib > previous_peak.board_memory_used_mib:
        current_peak["record"] = sample_record
    return sample_record


def _build_gpu_memory_tier_hint(peak_memory_used_mib: Optional[int]) -> str:
    if peak_memory_used_mib is None:
        return "未取得 nvidia-smi 峰值，暂时无法给出显存档位提示"
    peak_memory_used_gib = peak_memory_used_mib / 1024.0
    if peak_memory_used_gib < 12.0:
        return "12 GB 档可能可用，但仍需留冗余"
    if peak_memory_used_gib <= 20.0:
        return "24 GB 档更稳妥"
    if peak_memory_used_gib <= 32.0:
        return "40 GB 档更稳妥"
    return "建议 40 GB 以上，必要时考虑 48 GB / 80 GB 档"


def build_gpu_peak_display_summary(summary_payload: Mapping[str, Any]) -> Dict[str, Any]:
    """
    功能：构建 Notebook 展示用的 GPU 峰值简短摘要。

    Build a concise notebook-facing summary from one GPU session peak payload.

    Args:
        summary_payload: Parsed JSON payload emitted by the wrapper.

    Returns:
        Mapping containing the monitor status, peak/start/end memory values,
        peak GPU identity, and a coarse memory-tier recommendation.
    """
    if not isinstance(summary_payload, Mapping):
        raise TypeError("summary_payload must be Mapping")

    peak_memory_used_mib_value = summary_payload.get("session_board_peak_memory_used_mib")
    peak_memory_used_mib = int(peak_memory_used_mib_value) if isinstance(peak_memory_used_mib_value, (int, float)) else None
    peak_memory_used_gib = round(peak_memory_used_mib / 1024.0, 2) if peak_memory_used_mib is not None else None

    return {
        "status": summary_payload.get("status"),
        "peak_memory_used_mib": peak_memory_used_mib,
        "peak_memory_used_gib": peak_memory_used_gib,
        "start_memory_used_mib": summary_payload.get("session_board_memory_used_mib_at_start"),
        "end_memory_used_mib": summary_payload.get("session_board_memory_used_mib_at_end"),
        "peak_gpu_name": summary_payload.get("peak_gpu_name"),
        "peak_gpu_index": summary_payload.get("peak_gpu_index"),
        "visible_gpu_count": summary_payload.get("visible_gpu_count"),
        "recommendation": _build_gpu_memory_tier_hint(peak_memory_used_mib),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    功能：执行 GPU 会话峰值包装命令并写出 JSON 摘要。

    Execute the GPU session peak wrapper and persist one JSON summary.

    Args:
        argv: Optional CLI argument sequence. When omitted, sys.argv is used.

    Returns:
        Exit code aligned with the wrapped command when it starts successfully.
    """
    args = _parse_args(argv)
    output_json_path = Path(args.output_json)
    wrapped_command = [str(item) for item in args.wrapped_command]
    nvidia_smi_path = shutil.which("nvidia-smi")
    started_at_utc = utc_now_iso()
    started_at_monotonic = time.monotonic()

    per_gpu_state: Dict[str, Dict[str, Any]] = {}
    first_error_state: Dict[str, Optional[str]] = {"message": None}
    sample_error_state: Dict[str, int] = {"count": 0}
    successful_sample_state: Dict[str, int] = {"count": 0}
    current_peak: Dict[str, Optional[_SampleRecord]] = {"record": None}

    start_sample = _capture_sample_if_available(
        nvidia_smi_path,
        per_gpu_state,
        first_error_state,
        sample_error_state,
        successful_sample_state,
        current_peak,
    )

    wrapped_result_holder: Dict[str, Any] = {"result": None, "exception": None}

    def _run_wrapped_command() -> None:
        try:
            wrapped_result_holder["result"] = subprocess.run(wrapped_command, check=False)
        except Exception as exc:
            wrapped_result_holder["exception"] = exc

    worker_thread = threading.Thread(target=_run_wrapped_command, name="gpu_session_peak_wrapped_command")
    worker_thread.start()
    sample_interval_seconds = args.sample_interval_ms / 1000.0

    while worker_thread.is_alive():
        worker_thread.join(timeout=sample_interval_seconds)
        if worker_thread.is_alive():
            _capture_sample_if_available(
                nvidia_smi_path,
                per_gpu_state,
                first_error_state,
                sample_error_state,
                successful_sample_state,
                current_peak,
            )

    worker_thread.join()
    end_sample = _capture_sample_if_available(
        nvidia_smi_path,
        per_gpu_state,
        first_error_state,
        sample_error_state,
        successful_sample_state,
        current_peak,
    )

    finished_at_utc = utc_now_iso()
    elapsed_seconds = round(time.monotonic() - started_at_monotonic, 6)
    peak_sample = current_peak["record"]
    wrapped_exception = wrapped_result_holder["exception"]
    wrapped_result = wrapped_result_holder["result"]
    wrapped_return_code: Optional[int]
    if wrapped_result is not None:
        wrapped_return_code = int(wrapped_result.returncode)
    else:
        wrapped_return_code = None

    if not nvidia_smi_path:
        status = "absent"
    elif peak_sample is not None:
        status = "ok"
    else:
        status = "failed"

    start_memory_used_mib = start_sample.board_memory_used_mib if start_sample is not None else None
    end_memory_used_mib = end_sample.board_memory_used_mib if end_sample is not None else None
    peak_memory_used_mib = peak_sample.board_memory_used_mib if peak_sample is not None else None
    peak_memory_used_bytes = peak_memory_used_mib * 1024 * 1024 if peak_memory_used_mib is not None else None
    peak_memory_increment_mib = (
        peak_memory_used_mib - start_memory_used_mib
        if peak_memory_used_mib is not None and start_memory_used_mib is not None
        else None
    )

    summary_payload: Dict[str, Any] = {
        "status": status,
        "label": str(args.label),
        "monitor_source": "nvidia-smi",
        "nvidia_smi_available": bool(nvidia_smi_path),
        "nvidia_smi_path": nvidia_smi_path,
        "sample_interval_ms": int(args.sample_interval_ms),
        "sample_count": int(successful_sample_state["count"]),
        "sampling_error_count": int(sample_error_state["count"]),
        "monitor_error": first_error_state["message"],
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "elapsed_seconds": elapsed_seconds,
        "wrapped_command": wrapped_command,
        "wrapped_return_code": wrapped_return_code,
        "visible_gpu_count": len(per_gpu_state),
        "visible_gpus": _build_visible_gpu_payload(per_gpu_state),
        "session_board_memory_used_mib_at_start": start_memory_used_mib,
        "session_board_memory_used_mib_at_end": end_memory_used_mib,
        "session_board_peak_memory_used_mib": peak_memory_used_mib,
        "session_board_peak_memory_used_bytes": peak_memory_used_bytes,
        "session_board_peak_increment_mib": peak_memory_increment_mib,
        "peak_observed_at_utc": peak_sample.observed_at_utc if peak_sample is not None else None,
        "peak_gpu_index": peak_sample.peak_gpu_index if peak_sample is not None else None,
        "peak_gpu_uuid": peak_sample.peak_gpu_uuid if peak_sample is not None else None,
        "peak_gpu_name": peak_sample.peak_gpu_name if peak_sample is not None else None,
        "wrapped_command_error": f"{type(wrapped_exception).__name__}: {wrapped_exception}" if wrapped_exception else None,
    }
    write_json_atomic(output_json_path, summary_payload)

    if wrapped_exception is not None:
        print(f"failed to execute wrapped command: {wrapped_exception}", file=sys.stderr)
        return 1
    if wrapped_return_code is None:
        return 1
    return wrapped_return_code


if __name__ == "__main__":
    raise SystemExit(main())
