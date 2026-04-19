"""
File purpose: Share notebook-only PW04 runtime and result helpers across the three staged notebooks.
Module type: General module

职责边界：
1. 仅承载 PW04 notebook 共用的纯编排逻辑，不承载 notebook 参数常量与路径常量。
2. 不修改 PW04 主脚本语义，不重算 main/ 下的真实方法逻辑。
3. 为 prepare、quality_shard、finalize 三个 notebook 提供统一且可测试的辅助函数。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, cast

from paper_workflow.scripts.pw_quality_metrics import (
    DEFAULT_QUALITY_BATCH_SIZE,
    DEFAULT_QUALITY_TORCH_DEVICE,
    QUALITY_CLIP_BATCH_SIZE_ENV,
    QUALITY_LPIPS_BATCH_SIZE_ENV,
    QUALITY_TORCH_DEVICE_ENV,
)
from scripts.notebook_runtime_common import build_repo_import_subprocess_env, normalize_path_value


VALID_QUALITY_DEVICE_REQUESTS = {"auto", "cuda", "cpu"}
LOW_MEMORY_CUDA_THRESHOLD_GIB = 12.0


def _load_required_json_dict(path_obj: Path, label: str) -> Dict[str, Any]:
    """
    功能：读取必需的 JSON 对象文件。

    Load one required JSON object file.

    Args:
        path_obj: JSON file path.
        label: Human-readable label.

    Returns:
        Parsed JSON mapping.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"{label} not found: {normalize_path_value(path_obj)}")
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be JSON object: {normalize_path_value(path_obj)}")
    return cast(Dict[str, Any], payload)


def _extract_mapping(node: Any) -> Dict[str, Any]:
    """
    功能：将可选映射节点规范化为 dict。

    Normalize one optional mapping node to dict.

    Args:
        node: Candidate mapping node.

    Returns:
        Plain dict payload.
    """
    return dict(cast(Mapping[str, Any], node)) if isinstance(node, Mapping) else {}


def _normalize_quality_device_request(
    raw_value: Any,
    label: str,
    *,
    default_value: str = "auto",
) -> tuple[str | None, str | None]:
    """
    功能：规范化 quality device 请求值。

    Normalize one requested quality-device token.

    Args:
        raw_value: Candidate raw value.
        label: Human-readable label.
        default_value: Fallback token.

    Returns:
        Tuple of (normalized_value_or_none, warning_or_none).
    """
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")
    if not isinstance(default_value, str) or default_value not in VALID_QUALITY_DEVICE_REQUESTS:
        raise TypeError("default_value must be one of auto/cuda/cpu")
    if raw_value is None:
        return None, None
    if not isinstance(raw_value, str):
        return default_value, f"{label} must be one of auto/cuda/cpu; fallback to {default_value}"
    normalized_value = raw_value.strip().lower()
    if not normalized_value:
        return default_value, f"{label} is empty; fallback to {default_value}"
    if normalized_value not in VALID_QUALITY_DEVICE_REQUESTS:
        return default_value, f"{label}={raw_value!r} invalid; fallback to {default_value}"
    return normalized_value, None


def _normalize_positive_batch_size(
    raw_value: Any,
    label: str,
    default_value: int,
) -> tuple[int, str | None, str]:
    """
    功能：规范化 batch size 输入。

    Normalize one quality batch-size override.

    Args:
        raw_value: Candidate raw value.
        label: Human-readable label.
        default_value: Fallback batch size.

    Returns:
        Tuple of (batch_size, warning_or_none, source_label).
    """
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")
    if not isinstance(default_value, int) or isinstance(default_value, bool) or default_value <= 0:
        raise TypeError("default_value must be positive int")
    if raw_value is None:
        return default_value, None, "device_default"
    try:
        normalized_value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        return default_value, f"{label}={raw_value!r} invalid; fallback to {default_value}", "device_default"
    if normalized_value <= 0:
        return default_value, f"{label}={raw_value!r} invalid; fallback to {default_value}", "device_default"
    return normalized_value, None, "environment"


def _resolve_pw04_mode(pw04_mode: str) -> str:
    """
    功能：规范化 PW04 mode。

    Normalize one PW04 mode token.

    Args:
        pw04_mode: Candidate mode token.

    Returns:
        Normalized mode token.
    """
    if not isinstance(pw04_mode, str) or not pw04_mode.strip():
        raise TypeError("pw04_mode must be non-empty str")
    resolved_mode = pw04_mode.strip().lower()
    if resolved_mode not in {"prepare", "quality_shard", "finalize"}:
        raise ValueError("pw04_mode must be one of prepare/quality_shard/finalize")
    return resolved_mode


def _resolve_required_path_from_payload(
    payload: Mapping[str, Any],
    field_name: str,
    label: str,
) -> Path:
    """
    功能：从 JSON payload 中解析必需路径字段。

    Resolve one required path field from a JSON payload.

    Args:
        payload: Source mapping payload.
        field_name: Required field name.
        label: Human-readable label.

    Returns:
        Resolved absolute path.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be Mapping")
    if not isinstance(field_name, str) or not field_name:
        raise TypeError("field_name must be non-empty str")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")
    path_value = payload.get(field_name)
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{label} missing field: {field_name}")
    return Path(path_value).expanduser().resolve()


def _resolve_required_path_from_mapping(
    payload: Mapping[str, Any],
    key_name: str,
    label: str,
) -> Path:
    """
    功能：从路径映射中解析必需路径。

    Resolve one required path from a path-mapping payload.

    Args:
        payload: Path mapping payload.
        key_name: Required key name.
        label: Human-readable label.

    Returns:
        Resolved absolute path.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be Mapping")
    if not isinstance(key_name, str) or not key_name:
        raise TypeError("key_name must be non-empty str")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")
    path_value = payload.get(key_name)
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{label} missing key: {key_name}")
    return Path(path_value).expanduser().resolve()


def _build_expected_quality_shard_status_rows(path_values: Any) -> list[Dict[str, Any]]:
    """
    功能：构造 expected quality shard 路径存在性摘要。

    Build notebook-friendly existence rows for expected quality shard paths.

    Args:
        path_values: Candidate path list.

    Returns:
        Ordered path/existence rows.
    """
    if not isinstance(path_values, list):
        return []
    output_rows: list[Dict[str, Any]] = []
    for path_value in path_values:
        path_text = str(path_value)
        resolved_path = Path(path_text).expanduser().resolve()
        output_rows.append({"path": path_text, "exists": resolved_path.exists()})
    return output_rows


def _build_gpu_memory_tier_hint(peak_memory_used_mib: int | None) -> str:
    """
    功能：根据 GPU 峰值显存给出粗粒度建议。

    Build a coarse GPU memory-tier hint.

    Args:
        peak_memory_used_mib: Peak board memory in MiB.

    Returns:
        Notebook-facing recommendation text.
    """
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


def _build_fallback_gpu_peak_display_summary(
    *,
    raw_summary: Mapping[str, Any] | None,
    monitor_status: str,
) -> Dict[str, Any]:
    """
    功能：在缺少 display helper 时构造兼容展示摘要。

    Build a fallback display summary when the external display helper is unavailable.

    Args:
        raw_summary: Raw GPU peak summary payload.
        monitor_status: Effective monitor status.

    Returns:
        Display-summary-compatible payload.
    """
    if not isinstance(monitor_status, str) or not monitor_status:
        raise TypeError("monitor_status must be non-empty str")
    peak_memory_value = None
    start_memory_value = None
    end_memory_value = None
    peak_gpu_name = None
    peak_gpu_index = None
    visible_gpu_count = 0
    if isinstance(raw_summary, Mapping):
        peak_memory_raw = raw_summary.get("session_board_peak_memory_used_mib")
        start_memory_raw = raw_summary.get("session_board_memory_used_mib_at_start")
        end_memory_raw = raw_summary.get("session_board_memory_used_mib_at_end")
        visible_gpu_raw = raw_summary.get("visible_gpu_count")
        if isinstance(peak_memory_raw, (int, float)) and not isinstance(peak_memory_raw, bool):
            peak_memory_value = int(peak_memory_raw)
        if isinstance(start_memory_raw, (int, float)) and not isinstance(start_memory_raw, bool):
            start_memory_value = int(start_memory_raw)
        if isinstance(end_memory_raw, (int, float)) and not isinstance(end_memory_raw, bool):
            end_memory_value = int(end_memory_raw)
        if isinstance(raw_summary.get("peak_gpu_name"), str):
            peak_gpu_name = str(raw_summary.get("peak_gpu_name"))
        peak_gpu_index_raw = raw_summary.get("peak_gpu_index")
        if isinstance(peak_gpu_index_raw, (int, float)) and not isinstance(peak_gpu_index_raw, bool):
            peak_gpu_index = int(peak_gpu_index_raw)
        if isinstance(visible_gpu_raw, (int, float)) and not isinstance(visible_gpu_raw, bool):
            visible_gpu_count = int(visible_gpu_raw)
    peak_memory_used_gib = round(peak_memory_value / 1024.0, 2) if peak_memory_value is not None else None
    return {
        "status": monitor_status,
        "peak_memory_used_mib": peak_memory_value,
        "peak_memory_used_gib": peak_memory_used_gib,
        "start_memory_used_mib": start_memory_value,
        "end_memory_used_mib": end_memory_value,
        "peak_gpu_name": peak_gpu_name,
        "peak_gpu_index": peak_gpu_index,
        "visible_gpu_count": visible_gpu_count,
        "recommendation": _build_gpu_memory_tier_hint(peak_memory_value),
    }


def resolve_pw04_quality_runtime_summary(
    *,
    quality_device_override: str,
    base_env: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """
    功能：解析 PW04 quality runtime 的设备与 batch 配置摘要。

    Resolve the notebook-facing PW04 quality runtime summary.

    Args:
        quality_device_override: Notebook-level quality device override.
        base_env: Optional source environment mapping.

    Returns:
        Quality runtime summary compatible with the PW04 notebooks.
    """
    if base_env is not None and not isinstance(base_env, Mapping):
        raise TypeError("base_env must be Mapping[str, str] or None")

    env_mapping = os.environ if base_env is None else base_env
    quality_warnings: list[str] = []

    resolved_notebook_request, notebook_warning = _normalize_quality_device_request(
        quality_device_override,
        "QUALITY_DEVICE_OVERRIDE",
        default_value="auto",
    )
    if notebook_warning is not None:
        quality_warnings.append(notebook_warning)

    resolved_env_request = None
    env_requested_device_value = env_mapping.get(QUALITY_TORCH_DEVICE_ENV)
    if env_requested_device_value is not None:
        resolved_env_request, env_warning = _normalize_quality_device_request(
            env_requested_device_value,
            QUALITY_TORCH_DEVICE_ENV,
            default_value="auto",
        )
        if env_warning is not None:
            quality_warnings.append(env_warning)

    if resolved_notebook_request is not None and resolved_notebook_request != "auto":
        requested_device = resolved_notebook_request
        requested_device_source = "notebook_override"
    elif resolved_env_request is not None:
        requested_device = resolved_env_request
        requested_device_source = "environment"
    else:
        requested_device = "auto"
        requested_device_source = "notebook_default_auto"

    detected_cuda_available = False
    detected_cuda_device_count = 0
    detected_cuda_device_name = None
    detected_cuda_total_memory_gib = None
    torch_runtime_status = "not_imported"
    detection_reason = ""
    selection_reason = ""
    fallback_reason = None

    try:
        import torch  # type: ignore
    except Exception as exc:
        torch_runtime_status = f"import_failed: {type(exc).__name__}: {exc}"
        detection_reason = f"torch import failed; treat cuda as unavailable: {type(exc).__name__}: {exc}"
    else:
        torch_runtime_status = "imported"
        try:
            raw_cuda_available = bool(torch.cuda.is_available())
            detected_cuda_device_count = int(torch.cuda.device_count())
            detected_cuda_available = bool(raw_cuda_available and detected_cuda_device_count > 0)
        except Exception as exc:
            detected_cuda_available = False
            detected_cuda_device_count = 0
            detection_reason = f"torch.cuda probe failed; treat cuda as unavailable: {type(exc).__name__}: {exc}"
        else:
            if detected_cuda_available:
                try:
                    detected_cuda_device_name = torch.cuda.get_device_name(0)
                except Exception:
                    detected_cuda_device_name = None
                try:
                    total_memory_bytes = float(torch.cuda.get_device_properties(0).total_memory)
                    detected_cuda_total_memory_gib = round(total_memory_bytes / float(1024 ** 3), 2)
                except Exception:
                    detected_cuda_total_memory_gib = None
                detection_reason = "torch.cuda.is_available() returned True and at least one CUDA device is available"
            elif raw_cuda_available:
                detection_reason = "torch.cuda.is_available() returned True but no CUDA devices were enumerated"
            else:
                detection_reason = "torch.cuda.is_available() returned False"

    if requested_device == "cpu":
        selected_device = "cpu"
        selection_reason = "cpu explicitly requested"
    elif requested_device == "cuda":
        if detected_cuda_available:
            selected_device = "cuda"
            selection_reason = "requested cuda and cuda is available"
        else:
            selected_device = "cpu"
            selection_reason = "requested cuda but unavailable"
            fallback_reason = "requested cuda but unavailable; fallback to cpu"
            quality_warnings.append(fallback_reason)
    else:
        if detected_cuda_available:
            selected_device = "cuda"
            selection_reason = "auto selected cuda because cuda is available"
        else:
            selected_device = "cpu"
            selection_reason = "auto selected cpu because cuda is unavailable"

    if selected_device == "cpu":
        default_lpips_batch_size = DEFAULT_QUALITY_BATCH_SIZE
        default_clip_batch_size = DEFAULT_QUALITY_BATCH_SIZE
        batch_default_reason = "cpu runtime uses single-item batch defaults"
    elif (
        isinstance(detected_cuda_total_memory_gib, float)
        and detected_cuda_total_memory_gib < LOW_MEMORY_CUDA_THRESHOLD_GIB
    ):
        default_lpips_batch_size = 4
        default_clip_batch_size = 8
        batch_default_reason = (
            f"cuda runtime with approximately {detected_cuda_total_memory_gib} GiB memory uses conservative GPU batch defaults"
        )
    else:
        default_lpips_batch_size = 16
        default_clip_batch_size = 32
        batch_default_reason = "cuda runtime uses default GPU batch sizes"

    lpips_batch_size, lpips_warning, lpips_batch_size_source = _normalize_positive_batch_size(
        env_mapping.get(QUALITY_LPIPS_BATCH_SIZE_ENV),
        QUALITY_LPIPS_BATCH_SIZE_ENV,
        default_lpips_batch_size,
    )
    if lpips_warning is not None:
        quality_warnings.append(lpips_warning)

    clip_batch_size, clip_warning, clip_batch_size_source = _normalize_positive_batch_size(
        env_mapping.get(QUALITY_CLIP_BATCH_SIZE_ENV),
        QUALITY_CLIP_BATCH_SIZE_ENV,
        default_clip_batch_size,
    )
    if clip_warning is not None:
        quality_warnings.append(clip_warning)

    return {
        "requested_device": requested_device,
        "requested_device_source": requested_device_source,
        "detected_cuda_available": detected_cuda_available,
        "detected_cuda_device_count": detected_cuda_device_count,
        "detected_cuda_device_name": detected_cuda_device_name,
        "detected_cuda_total_memory_gib": detected_cuda_total_memory_gib,
        "selected_device": selected_device,
        "lpips_batch_size": lpips_batch_size,
        "lpips_batch_size_source": lpips_batch_size_source,
        "clip_batch_size": clip_batch_size,
        "clip_batch_size_source": clip_batch_size_source,
        "torch_runtime_status": torch_runtime_status,
        "detection_reason": detection_reason,
        "selection_reason": selection_reason,
        "fallback_reason": fallback_reason,
        "batch_default_reason": batch_default_reason,
        "warnings": quality_warnings,
    }


def build_pw04_subprocess_env(
    *,
    repo_root: Path,
    base_env: Mapping[str, str],
    quality_runtime_summary: Mapping[str, Any],
) -> Dict[str, str]:
    """
    功能：构造带 PW04 quality runtime 绑定的子进程环境。

    Build the subprocess environment for PW04 notebook execution.

    Args:
        repo_root: Repository root path.
        base_env: Source environment mapping.
        quality_runtime_summary: Resolved quality runtime summary.

    Returns:
        Subprocess environment mapping.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if not isinstance(base_env, Mapping):
        raise TypeError("base_env must be Mapping[str, str]")
    if not isinstance(quality_runtime_summary, Mapping):
        raise TypeError("quality_runtime_summary must be Mapping")

    env_mapping = build_repo_import_subprocess_env(base_env=base_env, repo_root=repo_root)
    selected_device = quality_runtime_summary.get("selected_device", DEFAULT_QUALITY_TORCH_DEVICE)
    if not isinstance(selected_device, str) or selected_device not in {"cpu", "cuda"}:
        selected_device = DEFAULT_QUALITY_TORCH_DEVICE

    lpips_batch_size = quality_runtime_summary.get("lpips_batch_size", DEFAULT_QUALITY_BATCH_SIZE)
    clip_batch_size = quality_runtime_summary.get("clip_batch_size", DEFAULT_QUALITY_BATCH_SIZE)
    if not isinstance(lpips_batch_size, int) or isinstance(lpips_batch_size, bool) or lpips_batch_size <= 0:
        lpips_batch_size = DEFAULT_QUALITY_BATCH_SIZE
    if not isinstance(clip_batch_size, int) or isinstance(clip_batch_size, bool) or clip_batch_size <= 0:
        clip_batch_size = DEFAULT_QUALITY_BATCH_SIZE

    env_mapping[QUALITY_TORCH_DEVICE_ENV] = selected_device
    env_mapping[QUALITY_LPIPS_BATCH_SIZE_ENV] = str(lpips_batch_size)
    env_mapping[QUALITY_CLIP_BATCH_SIZE_ENV] = str(clip_batch_size)
    return env_mapping


def build_pw04_command(
    *,
    script_path: Path,
    drive_project_root: Path,
    family_id: str,
    pw04_mode: str,
    quality_shard_index: int,
    force_rerun: bool,
    enable_tail_estimation: bool,
) -> list[str]:
    """
    功能：构造 PW04 CLI 命令。

    Build the canonical PW04 CLI command.

    Args:
        script_path: PW04 CLI script path.
        drive_project_root: Drive project root.
        family_id: Family identifier.
        pw04_mode: PW04 mode.
        quality_shard_index: Quality shard index.
        force_rerun: Whether rerun is enabled.
        enable_tail_estimation: Whether tail estimation is enabled.

    Returns:
        CLI command token list.
    """
    if not isinstance(script_path, Path):
        raise TypeError("script_path must be Path")
    if not isinstance(drive_project_root, Path):
        raise TypeError("drive_project_root must be Path")
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if not isinstance(quality_shard_index, int) or isinstance(quality_shard_index, bool) or quality_shard_index < 0:
        raise TypeError("quality_shard_index must be non-negative int")
    if not isinstance(force_rerun, bool):
        raise TypeError("force_rerun must be bool")
    if not isinstance(enable_tail_estimation, bool):
        raise TypeError("enable_tail_estimation must be bool")

    resolved_mode = _resolve_pw04_mode(pw04_mode)
    command = [
        sys.executable,
        str(script_path),
        "--drive-project-root",
        str(drive_project_root),
        "--family-id",
        family_id.strip(),
        "--pw04-mode",
        resolved_mode,
    ]
    if resolved_mode == "quality_shard":
        command.extend(["--quality-shard-index", str(quality_shard_index)])
    if force_rerun:
        command.append("--force-rerun")
    if enable_tail_estimation:
        command.append("--enable-tail-estimation")
    return command


def resolve_pw04_expected_output(
    *,
    pw04_mode: str,
    prepare_manifest_path: Path,
    selected_quality_shard_path: Path,
    pw04_summary_path: Path,
) -> tuple[str, Path]:
    """
    功能：根据 PW04 mode 解析期望关键输出。

    Resolve the expected key output label and path for one PW04 mode.

    Args:
        pw04_mode: PW04 mode token.
        prepare_manifest_path: Prepare-manifest path.
        selected_quality_shard_path: Selected quality-shard path.
        pw04_summary_path: Final PW04 summary path.

    Returns:
        Tuple of (output_label, output_path).
    """
    if not isinstance(prepare_manifest_path, Path):
        raise TypeError("prepare_manifest_path must be Path")
    if not isinstance(selected_quality_shard_path, Path):
        raise TypeError("selected_quality_shard_path must be Path")
    if not isinstance(pw04_summary_path, Path):
        raise TypeError("pw04_summary_path must be Path")

    resolved_mode = _resolve_pw04_mode(pw04_mode)
    if resolved_mode == "prepare":
        return "prepare_manifest", prepare_manifest_path
    if resolved_mode == "quality_shard":
        return "quality_shard", selected_quality_shard_path
    return "pw04_summary", pw04_summary_path


def load_gpu_peak_summary(path_obj: Path) -> tuple[Dict[str, Any] | None, str | None]:
    """
    功能：读取 GPU peak summary JSON。

    Load one GPU session peak summary JSON payload.

    Args:
        path_obj: Summary JSON path.

    Returns:
        Tuple of (payload_or_none, error_or_none).
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not path_obj.exists() or not path_obj.is_file():
        return None, f"gpu peak summary missing: {path_obj}"
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"gpu peak summary must be JSON object: {path_obj}"
    return cast(Dict[str, Any], payload), None


def build_gpu_peak_notebook_summary(
    *,
    raw_summary: Mapping[str, Any] | None,
    monitor_status: str,
    monitor_error: str | None,
    fallback_reason: str | None,
    display_helper: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    gpu_peak_summary_path: Path,
) -> Dict[str, Any]:
    """
    功能：把 GPU peak raw summary 规范化为 notebook 展示摘要。

    Normalize the raw GPU peak summary into the notebook-facing summary.

    Args:
        raw_summary: Raw summary payload.
        monitor_status: Effective monitor status.
        monitor_error: Monitor error text.
        fallback_reason: Wrapper fallback reason.
        display_helper: Optional external display helper.
        gpu_peak_summary_path: Summary file path.

    Returns:
        Notebook-facing GPU peak summary.
    """
    if raw_summary is not None and not isinstance(raw_summary, Mapping):
        raise TypeError("raw_summary must be Mapping or None")
    if not isinstance(monitor_status, str) or not monitor_status:
        raise TypeError("monitor_status must be non-empty str")
    if monitor_error is not None and not isinstance(monitor_error, str):
        raise TypeError("monitor_error must be str or None")
    if fallback_reason is not None and not isinstance(fallback_reason, str):
        raise TypeError("fallback_reason must be str or None")
    if display_helper is not None and not callable(display_helper):
        raise TypeError("display_helper must be callable or None")
    if not isinstance(gpu_peak_summary_path, Path):
        raise TypeError("gpu_peak_summary_path must be Path")

    resolved_display_summary: Dict[str, Any]
    resolved_status = monitor_status
    if isinstance(raw_summary, Mapping):
        display_summary_candidate: Mapping[str, Any] | None = None
        if display_helper is not None:
            helper_result = display_helper(raw_summary)
            if isinstance(helper_result, Mapping):
                display_summary_candidate = helper_result
        if display_summary_candidate is None:
            display_summary_candidate = _build_fallback_gpu_peak_display_summary(
                raw_summary=raw_summary,
                monitor_status=str(raw_summary.get("status", monitor_status)),
            )
        resolved_display_summary = dict(display_summary_candidate)
        raw_status = raw_summary.get("status")
        if isinstance(raw_status, str) and raw_status.strip():
            resolved_status = raw_status
        else:
            display_status = resolved_display_summary.get("status")
            if isinstance(display_status, str) and display_status.strip():
                resolved_status = display_status

        gpu_peak_memory_value = raw_summary.get("session_board_peak_memory_used_mib")
        if not isinstance(gpu_peak_memory_value, (int, float)) or isinstance(gpu_peak_memory_value, bool):
            gpu_peak_memory_value = resolved_display_summary.get("peak_memory_used_mib")
        peak_gpu_index = raw_summary.get("peak_gpu_index")
        if not isinstance(peak_gpu_index, (int, float)) or isinstance(peak_gpu_index, bool):
            peak_gpu_index = resolved_display_summary.get("peak_gpu_index")
        peak_gpu_name = raw_summary.get("peak_gpu_name")
        if not isinstance(peak_gpu_name, str):
            peak_gpu_name = resolved_display_summary.get("peak_gpu_name")
        visible_gpu_count = raw_summary.get("visible_gpu_count")
        if not isinstance(visible_gpu_count, (int, float)) or isinstance(visible_gpu_count, bool):
            visible_gpu_count = resolved_display_summary.get("visible_gpu_count")

        return {
            "gpu_session_peak_path": str(gpu_peak_summary_path),
            "gpu_peak_memory_mib": gpu_peak_memory_value,
            "gpu_peak_increment_mib": raw_summary.get("session_board_peak_increment_mib"),
            "peak_gpu_index": peak_gpu_index,
            "peak_gpu_uuid": raw_summary.get("peak_gpu_uuid"),
            "peak_gpu_name": peak_gpu_name,
            "monitor_status": resolved_status,
            "monitor_recommendation": resolved_display_summary.get("recommendation"),
            "monitor_error": monitor_error or raw_summary.get("monitor_error"),
            "visible_gpu_count": visible_gpu_count,
            "wrapped_return_code": raw_summary.get("wrapped_return_code"),
            "monitor_fallback_reason": fallback_reason,
        }

    resolved_display_summary = _build_fallback_gpu_peak_display_summary(
        raw_summary=None,
        monitor_status=monitor_status,
    )
    return {
        "gpu_session_peak_path": str(gpu_peak_summary_path),
        "gpu_peak_memory_mib": resolved_display_summary.get("peak_memory_used_mib"),
        "gpu_peak_increment_mib": None,
        "peak_gpu_index": resolved_display_summary.get("peak_gpu_index"),
        "peak_gpu_uuid": None,
        "peak_gpu_name": resolved_display_summary.get("peak_gpu_name"),
        "monitor_status": monitor_status,
        "monitor_recommendation": resolved_display_summary.get("recommendation"),
        "monitor_error": monitor_error,
        "visible_gpu_count": resolved_display_summary.get("visible_gpu_count"),
        "wrapped_return_code": None,
        "monitor_fallback_reason": fallback_reason,
    }


def read_pw04_result_summary(
    *,
    pw04_mode: str,
    family_root: Path,
    prepare_manifest_path: Path,
    selected_quality_shard_path: Path,
    pw04_summary_path: Path,
    gpu_peak_notebook_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    功能：按 PW04 mode 回读关键产物并构造 notebook 结果摘要。

    Read back the key PW04 artifacts for notebook display.

    Args:
        pw04_mode: PW04 mode token.
        family_root: Family root path.
        prepare_manifest_path: Prepare-manifest path.
        selected_quality_shard_path: Selected quality-shard path.
        pw04_summary_path: Final PW04 summary path.
        gpu_peak_notebook_summary: Notebook-facing GPU peak summary.

    Returns:
        Notebook-facing PW04 result summary.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(prepare_manifest_path, Path):
        raise TypeError("prepare_manifest_path must be Path")
    if not isinstance(selected_quality_shard_path, Path):
        raise TypeError("selected_quality_shard_path must be Path")
    if not isinstance(pw04_summary_path, Path):
        raise TypeError("pw04_summary_path must be Path")
    if not isinstance(gpu_peak_notebook_summary, Mapping):
        raise TypeError("gpu_peak_notebook_summary must be Mapping")

    resolved_mode = _resolve_pw04_mode(pw04_mode)
    gpu_peak_summary_payload = dict(gpu_peak_notebook_summary)

    if resolved_mode == "prepare":
        prepare_manifest = _load_required_json_dict(prepare_manifest_path, "PW04 prepare manifest")
        return {
            "mode": resolved_mode,
            "prepare_manifest": prepare_manifest,
            "attack_merge_manifest": _load_required_json_dict(
                _resolve_required_path_from_payload(
                    prepare_manifest,
                    "attack_merge_manifest_path",
                    "PW04 attack merge manifest",
                ),
                "PW04 attack merge manifest",
            ),
            "attack_positive_pool_manifest": _load_required_json_dict(
                _resolve_required_path_from_payload(
                    prepare_manifest,
                    "attack_positive_pool_manifest_path",
                    "PW04 attack positive pool manifest",
                ),
                "PW04 attack positive pool manifest",
            ),
            "attack_negative_pool_manifest": _load_required_json_dict(
                _resolve_required_path_from_payload(
                    prepare_manifest,
                    "attack_negative_pool_manifest_path",
                    "PW04 attack negative pool manifest",
                ),
                "PW04 attack negative pool manifest",
            ),
            "formal_attack_final_decision_metrics": _load_required_json_dict(
                _resolve_required_path_from_payload(
                    prepare_manifest,
                    "formal_attack_final_decision_metrics_path",
                    "PW04 formal attack final decision metrics",
                ),
                "PW04 formal attack final decision metrics",
            ),
            "formal_attack_attestation_metrics": _load_required_json_dict(
                _resolve_required_path_from_payload(
                    prepare_manifest,
                    "formal_attack_attestation_metrics_path",
                    "PW04 formal attack attestation metrics",
                ),
                "PW04 formal attack attestation metrics",
            ),
            "derived_attack_union_metrics": _load_required_json_dict(
                _resolve_required_path_from_payload(
                    prepare_manifest,
                    "derived_attack_union_metrics_path",
                    "PW04 derived attack union metrics",
                ),
                "PW04 derived attack union metrics",
            ),
            "formal_attack_negative_metrics": _load_required_json_dict(
                _resolve_required_path_from_payload(
                    prepare_manifest,
                    "formal_attack_negative_metrics_path",
                    "PW04 formal attack negative metrics",
                ),
                "PW04 formal attack negative metrics",
            ),
            "quality_pair_plan": _load_required_json_dict(
                _resolve_required_path_from_payload(
                    prepare_manifest,
                    "quality_pair_plan_path",
                    "PW04 quality pair plan",
                ),
                "PW04 quality pair plan",
            ),
            "expected_quality_shard_paths": _build_expected_quality_shard_status_rows(
                prepare_manifest.get("expected_quality_shard_paths")
            ),
            "gpu_session_peak_summary": gpu_peak_summary_payload,
        }

    if resolved_mode == "quality_shard":
        prepare_manifest = _load_required_json_dict(prepare_manifest_path, "PW04 prepare manifest")
        return {
            "mode": resolved_mode,
            "prepare_manifest": prepare_manifest,
            "quality_shard": _load_required_json_dict(selected_quality_shard_path, "PW04 quality shard"),
            "expected_quality_shard_paths": _build_expected_quality_shard_status_rows(
                prepare_manifest.get("expected_quality_shard_paths")
            ),
            "gpu_session_peak_summary": gpu_peak_summary_payload,
        }

    pw04_summary = _load_required_json_dict(pw04_summary_path, "PW04 summary")
    canonical_metrics_paths = _extract_mapping(pw04_summary.get("canonical_metrics_paths"))
    paper_tables_paths = _extract_mapping(pw04_summary.get("paper_tables_paths"))
    paper_figures_paths = _extract_mapping(pw04_summary.get("paper_figures_paths"))
    tail_estimation_paths = _extract_mapping(pw04_summary.get("tail_estimation_paths"))
    paper_metric_registry_path = _resolve_required_path_from_payload(
        pw04_summary,
        "paper_scope_registry_path",
        "PW04 paper metric registry",
    )
    bootstrap_confidence_intervals_path = _resolve_required_path_from_payload(
        pw04_summary,
        "bootstrap_confidence_intervals_path",
        "PW04 bootstrap confidence intervals",
    )
    bootstrap_confidence_intervals_csv_path = _resolve_required_path_from_payload(
        pw04_summary,
        "bootstrap_confidence_intervals_csv_path",
        "PW04 bootstrap confidence intervals CSV",
    )

    formal_attack_final_decision_metrics_path = family_root / "exports" / "pw04" / "formal_attack_final_decision_metrics.json"
    formal_attack_attestation_metrics_path = family_root / "exports" / "pw04" / "formal_attack_attestation_metrics.json"
    derived_attack_union_metrics_path = family_root / "exports" / "pw04" / "derived_attack_union_metrics.json"
    clean_attack_overview_path = family_root / "exports" / "pw04" / "clean_attack_overview.json"

    return {
        "mode": resolved_mode,
        "summary": pw04_summary,
        "formal_attack_final_decision_metrics": _load_required_json_dict(
            formal_attack_final_decision_metrics_path,
            "PW04 formal attack final decision metrics",
        ),
        "formal_attack_attestation_metrics": _load_required_json_dict(
            formal_attack_attestation_metrics_path,
            "PW04 formal attack attestation metrics",
        ),
        "derived_attack_union_metrics": _load_required_json_dict(
            derived_attack_union_metrics_path,
            "PW04 derived attack union metrics",
        ),
        "clean_attack_overview": _load_required_json_dict(
            clean_attack_overview_path,
            "PW04 clean attack overview",
        ),
        "paper_metric_registry": _load_required_json_dict(
            paper_metric_registry_path,
            "PW04 paper metric registry",
        ),
        "canonical_metrics_paths": canonical_metrics_paths,
        "content_chain_metrics": _load_required_json_dict(
            _resolve_required_path_from_mapping(
                canonical_metrics_paths,
                "content_chain",
                "PW04 canonical content-chain metrics",
            ),
            "PW04 canonical content-chain metrics",
        ),
        "event_attestation_metrics": _load_required_json_dict(
            _resolve_required_path_from_mapping(
                canonical_metrics_paths,
                "event_attestation",
                "PW04 canonical event-attestation metrics",
            ),
            "PW04 canonical event-attestation metrics",
        ),
        "system_final_metrics": _load_required_json_dict(
            _resolve_required_path_from_mapping(
                canonical_metrics_paths,
                "system_final",
                "PW04 canonical system-final metrics",
            ),
            "PW04 canonical system-final metrics",
        ),
        "bootstrap_confidence_intervals": _load_required_json_dict(
            bootstrap_confidence_intervals_path,
            "PW04 bootstrap confidence intervals",
        ),
        "bootstrap_confidence_intervals_csv_path": str(bootstrap_confidence_intervals_csv_path),
        "paper_tables_paths": paper_tables_paths,
        "paper_figures_paths": {
            key_name: {
                "path": str(path_value),
                "exists": Path(str(path_value)).expanduser().resolve().exists(),
            }
            for key_name, path_value in paper_figures_paths.items()
        },
        "tail_estimation_paths": tail_estimation_paths,
        "estimated_tail_fpr_1e4": _load_required_json_dict(
            _resolve_required_path_from_mapping(
                tail_estimation_paths,
                "estimated_tail_fpr_1e4_path",
                "PW04 estimated tail FPR 1e-4",
            ),
            "PW04 estimated tail FPR 1e-4",
        ),
        "estimated_tail_fpr_1e5": _load_required_json_dict(
            _resolve_required_path_from_mapping(
                tail_estimation_paths,
                "estimated_tail_fpr_1e5_path",
                "PW04 estimated tail FPR 1e-5",
            ),
            "PW04 estimated tail FPR 1e-5",
        ),
        "tail_fit_diagnostics": _load_required_json_dict(
            _resolve_required_path_from_mapping(
                tail_estimation_paths,
                "tail_fit_diagnostics_path",
                "PW04 tail fit diagnostics",
            ),
            "PW04 tail fit diagnostics",
        ),
        "tail_fit_stability_summary": _load_required_json_dict(
            _resolve_required_path_from_mapping(
                tail_estimation_paths,
                "tail_fit_stability_summary_path",
                "PW04 tail fit stability summary",
            ),
            "PW04 tail fit stability summary",
        ),
        "gpu_session_peak_path": gpu_peak_summary_payload.get("gpu_session_peak_path"),
        "gpu_peak_memory_mib": gpu_peak_summary_payload.get("gpu_peak_memory_mib"),
        "gpu_peak_increment_mib": gpu_peak_summary_payload.get("gpu_peak_increment_mib"),
        "peak_gpu_index": gpu_peak_summary_payload.get("peak_gpu_index"),
        "peak_gpu_uuid": gpu_peak_summary_payload.get("peak_gpu_uuid"),
        "peak_gpu_name": gpu_peak_summary_payload.get("peak_gpu_name"),
        "monitor_status": gpu_peak_summary_payload.get("monitor_status"),
        "monitor_recommendation": gpu_peak_summary_payload.get("monitor_recommendation"),
        "monitor_error": gpu_peak_summary_payload.get("monitor_error"),
        "monitor_execution_mode": gpu_peak_summary_payload.get("monitor_execution_mode"),
        "gpu_session_peak_summary": gpu_peak_summary_payload,
    }