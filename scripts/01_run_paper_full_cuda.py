"""
文件目的：01_run_paper_full_cuda.py 作为 01_Paper_Full_Cuda 的正式主链脚本入口。
Module type: General module

职责边界：
1. 在一次 stage 01 运行中显式消费 prompt pool，生成 direct source records。
2. 显式构建 pooled threshold inputs，再调用 embed、detect、calibrate、evaluate 主链。
3. 不自动执行 parallel attestation statistics 或 experiment matrix。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from scripts.notebook_runtime_common import (
    REPO_ROOT,
    compute_file_sha256,
    copy_file,
    ensure_directory,
    load_yaml_mapping,
    normalize_path_value,
    run_command_with_logs,
    write_json_atomic,
    write_yaml_mapping,
)


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_RUN_ROOT = Path("outputs/colab_run_paper_full_cuda")
SOURCE_CONTRACT_RELATIVE_PATH = "artifacts/parallel_attestation_statistics_input_contract.json"
POOLED_THRESHOLD_BUILD_CONTRACT_RELATIVE_PATH = "artifacts/stage_01_pooled_threshold_build_contract.json"
CANONICAL_SOURCE_POOL_RELATIVE_ROOT = "artifacts/stage_01_canonical_source_pool"
CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH = f"{CANONICAL_SOURCE_POOL_RELATIVE_ROOT}/source_pool_manifest.json"
CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT = f"{CANONICAL_SOURCE_POOL_RELATIVE_ROOT}/entries"
CANONICAL_SOURCE_POOL_ATTESTATION_RELATIVE_ROOT = f"{CANONICAL_SOURCE_POOL_RELATIVE_ROOT}/attestation"
CANONICAL_SOURCE_POOL_SOURCE_IMAGES_RELATIVE_ROOT = f"{CANONICAL_SOURCE_POOL_RELATIVE_ROOT}/source_images"
SOURCE_POOL_DETECT_RECORDS_RELATIVE_ROOT = "artifacts/stage_01_source_pool_detect_records"
SOURCE_POOL_EMBED_RECORDS_RELATIVE_ROOT = "artifacts/stage_01_source_pool_embed_records"
POOLED_THRESHOLD_RECORDS_RELATIVE_ROOT = "artifacts/stage_01_pooled_threshold_records"
SOURCE_POOL_RUNTIME_CONFIG_RELATIVE_ROOT = "artifacts/stage_01_source_pool_runtime_configs"
POOLED_THRESHOLD_RUNTIME_CONFIG_RELATIVE_PATH = "artifacts/stage_01_pooled_threshold_runtime_config.yaml"
EVENT_ATTESTATION_SCORE_NAME = "event_attestation_score"
CONTENT_CHAIN_SCORE_NAME = "content_chain_score"
SOURCE_PLUS_DERIVED_PAIRS_MODE = "source_plus_derived_pairs"
DIRECT_SOURCE_ONLY_MODE = "direct_source_only"


def _resolve_repo_path(path_value: str) -> Path:
    """
    功能：按仓库根目录解析路径。

    Resolve a path against the repository root unless it is already absolute.

    Args:
        path_value: Raw path string.

    Returns:
        Resolved absolute path.
    """
    if not path_value.strip():
        raise TypeError("path_value must be non-empty str")
    candidate = Path(path_value.strip())
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def _format_override_arg(arg_name: str, value: Any) -> str:
    """
    功能：格式化 stage CLI override 参数。

    Format one CLI override argument token.

    Args:
        arg_name: Override name.
        value: Override value.

    Returns:
        Formatted override token.
    """
    if not arg_name:
        raise TypeError("arg_name must be non-empty str")
    return f"{arg_name}={json.dumps(value, ensure_ascii=False)}"


def _build_stage_overrides(stage_name: str, extra_overrides: Optional[Sequence[str]] = None) -> List[str]:
    """
    功能：构造主链 stage 通用 overrides。

    Build the shared CLI overrides for one workflow stage.

    Args:
        stage_name: Workflow stage name.
        extra_overrides: Optional extra overrides.

    Returns:
        Flat CLI override argument list.
    """
    if not stage_name:
        raise TypeError("stage_name must be non-empty str")

    override_items = [
        _format_override_arg("run_root_reuse_allowed", True),
        _format_override_arg("run_root_reuse_reason", f"paper_full_cuda_{stage_name}"),
    ]
    if extra_overrides is not None:
        override_items.extend(str(item) for item in extra_overrides)

    command_args: List[str] = []
    for override_arg in override_items:
        command_args.extend(["--override", override_arg])
    return command_args


def _build_stage_command(
    stage_name: str,
    config_path: Path,
    run_root: Path,
    extra_overrides: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    功能：构造单个主链 stage 命令。

    Build the command for one workflow stage.

    Args:
        stage_name: Workflow stage name.
        config_path: Runtime config path.
        run_root: Workflow run root.
        extra_overrides: Optional extra overrides.

    Returns:
        Command token list.
    """
    if stage_name not in {"embed", "detect", "calibrate", "evaluate"}:
        raise ValueError(f"unsupported stage_name: {stage_name}")
    command = [
        sys.executable,
        "-m",
        f"main.cli.run_{stage_name}",
        "--out",
        str(run_root),
        "--config",
        str(config_path),
    ]
    if stage_name == "detect":
        command.extend(["--input", str(run_root / "records" / "embed_record.json")])
    command.extend(_build_stage_overrides(stage_name, extra_overrides))
    return command


def _required_artifacts(run_root: Path) -> Dict[str, Path]:
    """
    功能：返回 stage 01 必需主输出路径。

    Return the required stage-01 artifact paths.

    Args:
        run_root: Workflow run root.

    Returns:
        Mapping of required output labels to paths.
    """
    return {
        "embed_record": run_root / "records" / "embed_record.json",
        "detect_record": run_root / "records" / "detect_record.json",
        "calibration_record": run_root / "records" / "calibration_record.json",
        "evaluate_record": run_root / "records" / "evaluate_record.json",
        "thresholds_artifact": run_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
        "threshold_metadata_artifact": run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json",
        "evaluation_report": run_root / "artifacts" / "evaluation_report.json",
        "run_closure": run_root / "artifacts" / "run_closure.json",
        "workflow_summary": run_root / "artifacts" / "workflow_summary.json",
        "canonical_source_pool_manifest": run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH,
        "source_contract": run_root / SOURCE_CONTRACT_RELATIVE_PATH,
        "pooled_threshold_build_contract": run_root / POOLED_THRESHOLD_BUILD_CONTRACT_RELATIVE_PATH,
    }


def _artifact_presence(artifact_paths: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    功能：汇总主输出文件存在性。

    Summarize required artifact presence.

    Args:
        artifact_paths: Required artifact path mapping.

    Returns:
        Presence summary mapping.
    """
    return {
        key_name: {
            "path": normalize_path_value(path_obj),
            "exists": bool(path_obj.exists() and path_obj.is_file()),
        }
        for key_name, path_obj in artifact_paths.items()
    }


def _all_required_present(artifact_summary: Dict[str, Dict[str, Any]]) -> bool:
    """
    功能：检查必需输出是否完整。

    Check whether all required artifacts exist.

    Args:
        artifact_summary: Presence summary mapping.

    Returns:
        True when all required artifacts exist.
    """
    return all(bool(item.get("exists", False)) for item in artifact_summary.values())


def _load_json_dict(path_obj: Path, label: str) -> Dict[str, Any]:
    """
    功能：读取 JSON 对象文件。

    Read one JSON file and require a mapping root.

    Args:
        path_obj: JSON file path.
        label: Human-readable label.

    Returns:
        Parsed JSON mapping.
    """
    if not label:
        raise TypeError("label must be non-empty str")
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"{label} not found: {normalize_path_value(path_obj)}")
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be JSON object: {normalize_path_value(path_obj)}")
    return cast(Dict[str, Any], payload)


def _resolve_prompt_lines(prompt_file: str) -> List[str]:
    """
    功能：读取 prompt 文件中的非空行。

    Resolve non-empty prompt lines from a prompt file.

    Args:
        prompt_file: Prompt file path string.

    Returns:
        Non-empty prompt lines.
    """
    if not prompt_file.strip():
        raise TypeError("prompt_file must be non-empty str")
    prompt_path = Path(prompt_file.strip()).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = (REPO_ROOT / prompt_path).resolve()
    else:
        prompt_path = prompt_path.resolve()
    if not prompt_path.exists() or not prompt_path.is_file():
        raise FileNotFoundError(f"prompt file not found: {prompt_path}")
    prompt_lines = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompt_lines:
        raise ValueError(f"prompt file has no non-empty lines: {prompt_path}")
    return prompt_lines


def _resolve_stage_01_source_pool_cfg(cfg_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：解析 stage 01 source pool 配置。

    Resolve the stage-01 direct source pool configuration.

    Args:
        cfg_obj: Runtime config mapping.

    Returns:
        Normalized source-pool config mapping.
    """
    section_node = cfg_obj.get("stage_01_source_pool")
    section = dict(cast(Dict[str, Any], section_node)) if isinstance(section_node, dict) else {}
    target_prompt_count = section.get("target_prompt_count")
    if target_prompt_count is not None and (not isinstance(target_prompt_count, int) or target_prompt_count <= 0):
        raise ValueError("stage_01_source_pool.target_prompt_count must be positive int or null")
    record_usage = section.get("record_usage", "stage_01_direct_source_pool")
    if not isinstance(record_usage, str) or not record_usage:
        raise ValueError("stage_01_source_pool.record_usage must be non-empty str")
    return {
        "enabled": section.get("enabled") is True,
        "use_inference_prompt_file": section.get("use_inference_prompt_file") is True,
        "target_prompt_count": target_prompt_count,
        "record_usage": record_usage,
    }


def _resolve_stage_01_pooled_threshold_build_cfg(cfg_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：解析 stage 01 pooled threshold build 配置。

    Resolve the stage-01 pooled threshold build configuration.

    Args:
        cfg_obj: Runtime config mapping.

    Returns:
        Normalized build config mapping.
    """
    section_node = cfg_obj.get("stage_01_pooled_threshold_build")
    section = dict(cast(Dict[str, Any], section_node)) if isinstance(section_node, dict) else {}
    build_mode = section.get("build_mode", SOURCE_PLUS_DERIVED_PAIRS_MODE)
    if build_mode not in {DIRECT_SOURCE_ONLY_MODE, SOURCE_PLUS_DERIVED_PAIRS_MODE}:
        raise ValueError(
            "stage_01_pooled_threshold_build.build_mode must be one of "
            f"{{'{DIRECT_SOURCE_ONLY_MODE}', '{SOURCE_PLUS_DERIVED_PAIRS_MODE}'}}"
        )
    target_pair_count = section.get("target_pair_count")
    if not isinstance(target_pair_count, int) or target_pair_count <= 0:
        raise ValueError("stage_01_pooled_threshold_build.target_pair_count must be positive int")
    build_usage = section.get("build_usage", "stage_01_pooled_thresholds")
    if not isinstance(build_usage, str) or not build_usage:
        raise ValueError("stage_01_pooled_threshold_build.build_usage must be non-empty str")
    record_derivation_kind = section.get("record_derivation_kind", "prompt_bound_label_balance")
    if not isinstance(record_derivation_kind, str) or not record_derivation_kind:
        raise ValueError("stage_01_pooled_threshold_build.record_derivation_kind must be non-empty str")
    return {
        "enabled": section.get("enabled") is True,
        "build_mode": build_mode,
        "target_pair_count": target_pair_count,
        "build_usage": build_usage,
        "record_derivation_kind": record_derivation_kind,
    }


def _resolve_stage_01_prompt_pool(cfg_obj: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    功能：解析 stage 01 source prompt pool。

    Resolve the prompt pool consumed by stage 01.

    Args:
        cfg_obj: Runtime config mapping.

    Returns:
        Tuple of prompt list and normalized prompt-file path.
    """
    source_pool_cfg = _resolve_stage_01_source_pool_cfg(cfg_obj)
    if source_pool_cfg["enabled"]:
        if source_pool_cfg["use_inference_prompt_file"] is not True:
            raise ValueError("stage_01_source_pool.enabled requires use_inference_prompt_file=true")
        prompt_file = cfg_obj.get("inference_prompt_file")
        if not isinstance(prompt_file, str) or not prompt_file.strip():
            raise ValueError("stage_01_source_pool requires non-empty inference_prompt_file")
        prompt_lines = _resolve_prompt_lines(prompt_file)
        target_prompt_count = source_pool_cfg.get("target_prompt_count")
        if isinstance(target_prompt_count, int) and len(prompt_lines) != target_prompt_count:
            raise ValueError(
                "stage_01_source_pool.target_prompt_count must match prompt file line count: "
                f"expected={target_prompt_count}, actual={len(prompt_lines)}"
            )
        return prompt_lines, normalize_path_value(prompt_file)

    prompt_value = cfg_obj.get("inference_prompt")
    if not isinstance(prompt_value, str) or not prompt_value.strip():
        raise ValueError("inference_prompt must be non-empty when stage_01_source_pool is disabled")
    return [prompt_value.strip()], "<absent>"


def _resolve_content_chain_score(record_payload: Dict[str, Any]) -> float | None:
    """
    功能：从 detect record 解析 content_chain_score。

    Resolve the canonical content-chain score from a detect record.

    Args:
        record_payload: Detect record mapping.

    Returns:
        Finite score when available.
    """
    content_node = record_payload.get("content_evidence_payload")
    if not isinstance(content_node, dict):
        return None
    content_payload = cast(Dict[str, Any], content_node)
    if content_payload.get("status") != "ok":
        return None
    for field_name in [CONTENT_CHAIN_SCORE_NAME, "score", "content_score"]:
        score_value = content_payload.get(field_name)
        if isinstance(score_value, bool):
            continue
        if isinstance(score_value, (int, float)):
            return float(score_value)
    return None


def _resolve_event_attestation_score(record_payload: Dict[str, Any]) -> float | None:
    """
    功能：从 detect record 解析 canonical event_attestation_score。

    Resolve the canonical event-attestation score from a detect record.

    Args:
        record_payload: Detect record mapping.

    Returns:
        Finite score when available.
    """
    attestation_node = record_payload.get("attestation")
    attestation_payload = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    final_decision_node = attestation_payload.get("final_event_attested_decision")
    final_decision = cast(Dict[str, Any], final_decision_node) if isinstance(final_decision_node, dict) else {}
    score_name = final_decision.get("event_attestation_score_name")
    if isinstance(score_name, str) and score_name and score_name != EVENT_ATTESTATION_SCORE_NAME:
        return None
    score_value = final_decision.get("event_attestation_score")
    if isinstance(score_value, bool):
        return None
    if isinstance(score_value, (int, float)):
        return float(score_value)
    return None


def _ensure_event_attestation_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：确保 payload 存在可写入的 event-attestation 容器。

    Ensure the payload contains a mutable final event-attestation node.

    Args:
        payload: Detect record payload.

    Returns:
        Final decision mapping.
    """
    attestation_node = payload.get("attestation")
    attestation_payload = dict(cast(Dict[str, Any], attestation_node)) if isinstance(attestation_node, dict) else {}
    final_decision_node = attestation_payload.get("final_event_attested_decision")
    final_decision = dict(cast(Dict[str, Any], final_decision_node)) if isinstance(final_decision_node, dict) else {}
    attestation_payload["final_event_attested_decision"] = final_decision
    payload["attestation"] = attestation_payload
    return final_decision


def _apply_negative_attestation_semantics(payload: Dict[str, Any]) -> None:
    """
    功能：为派生负样本写入显式 negative attestation 语义。

    Apply explicit negative attestation semantics to a derived negative record.

    Args:
        payload: Mutable detect record payload.

    Returns:
        None.
    """
    final_decision = _ensure_event_attestation_payload(payload)
    final_decision["status"] = "absent"
    final_decision["is_event_attested"] = False
    final_decision["event_attestation_score"] = 0.0
    final_decision["event_attestation_score_name"] = EVENT_ATTESTATION_SCORE_NAME
    final_decision["authenticity_status"] = "statement_only"
    final_decision["image_evidence_status"] = "absent"

    attestation_node = payload.get("attestation")
    if not isinstance(attestation_node, dict):
        return
    attestation_payload = cast(Dict[str, Any], attestation_node)
    attestation_payload["status"] = "absent"
    attestation_payload["verdict"] = "absent"
    if "content_attestation_score" in attestation_payload:
        attestation_payload["content_attestation_score"] = 0.0
    if "fusion_score" in attestation_payload:
        attestation_payload["fusion_score"] = 0.0

    image_evidence_result = attestation_payload.get("image_evidence_result")
    if isinstance(image_evidence_result, dict):
        image_evidence_result["status"] = "absent"
        if "content_attestation_score" in image_evidence_result:
            image_evidence_result["content_attestation_score"] = 0.0
        if "fusion_score" in image_evidence_result:
            image_evidence_result["fusion_score"] = 0.0


def _build_prompt_sha256(prompt_text: str) -> str:
    """
    功能：计算 prompt 文本摘要。

    Compute the SHA256 digest of one prompt text.

    Args:
        prompt_text: Prompt text.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    if not prompt_text:
        raise TypeError("prompt_text must be non-empty str")
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def _normalize_direct_detect_payload(
    payload: Dict[str, Any],
    *,
    prompt_text: str,
    prompt_index: int,
    prompt_file_path: str,
    record_usage: str,
) -> Dict[str, Any]:
    """
    功能：为 stage 01 direct source pool 规范化 detect record。

    Normalize one direct detect record emitted by the source pool.

    Args:
        payload: Original detect record payload.
        prompt_text: Prompt text.
        prompt_index: Prompt index.
        prompt_file_path: Normalized prompt file path.
        record_usage: Direct source usage marker.

    Returns:
        Normalized detect record payload.
    """
    if not prompt_text:
        raise TypeError("prompt_text must be non-empty str")
    if prompt_index < 0:
        raise TypeError("prompt_index must be non-negative int")
    if not prompt_file_path:
        raise TypeError("prompt_file_path must be non-empty str")
    if not record_usage:
        raise TypeError("record_usage must be non-empty str")

    normalized_payload = copy.deepcopy(payload)
    normalized_payload["label"] = True
    normalized_payload["ground_truth"] = True
    normalized_payload["is_watermarked"] = True
    normalized_payload["ground_truth_source"] = "stage_01_direct_source_pool"
    normalized_payload["inference_prompt"] = prompt_text
    normalized_payload["stage_01_source_pool"] = {
        "record_origin": "direct_source_record",
        "record_usage": record_usage,
        "prompt_index": prompt_index,
        "prompt_text": prompt_text,
        "prompt_sha256": _build_prompt_sha256(prompt_text),
        "prompt_file": prompt_file_path,
    }
    return normalized_payload


def _build_prompt_runtime_cfg(
    cfg_obj: Dict[str, Any],
    *,
    prompt_text: str,
    prompt_index: int,
    prompt_file_path: str,
) -> Dict[str, Any]:
    """
    功能：构造单条 prompt 的 stage 01 子运行配置。

    Build the per-prompt runtime config used by the stage-01 source pool.

    Args:
        cfg_obj: Base runtime config mapping.
        prompt_text: Prompt text.
        prompt_index: Prompt index.
        prompt_file_path: Normalized prompt file path.

    Returns:
        Per-prompt runtime config mapping.
    """
    cfg_copy = json.loads(json.dumps(cfg_obj))
    cfg_copy["inference_prompt"] = prompt_text
    cfg_copy["stage_01_prompt_index"] = prompt_index
    cfg_copy["stage_01_prompt_file"] = prompt_file_path
    return cfg_copy


def _run_stage(stage_name: str, command: Sequence[str], run_root: Path) -> Dict[str, Any]:
    """
    功能：执行一个 workflow stage 并写日志。

    Run one workflow stage and persist stdout/stderr logs.

    Args:
        stage_name: Workflow stage name.
        command: Command token list.
        run_root: Stage run root.

    Returns:
        Stage execution summary.
    """
    logs_dir = ensure_directory(run_root / "logs")
    result = run_command_with_logs(
        command=command,
        cwd=REPO_ROOT,
        stdout_log_path=logs_dir / f"{stage_name}_stdout.log",
        stderr_log_path=logs_dir / f"{stage_name}_stderr.log",
    )
    result["status"] = "ok" if result.get("return_code") == 0 else "failed"
    return result


def _read_log_tail(path_value: Any, max_lines: int = 20) -> List[str]:
    """
    功能：读取日志文件尾部若干行用于失败诊断。

    Read the tail lines from one log file for failure diagnostics.

    Args:
        path_value: Log path value.
        max_lines: Maximum number of lines to keep.

    Returns:
        Tail lines in original order.
    """
    if max_lines <= 0:
        raise ValueError("max_lines must be positive int")
    if not isinstance(path_value, (str, Path)):
        return []
    path_obj = Path(path_value)
    if not path_obj.exists() or not path_obj.is_file():
        return []
    lines = path_obj.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def _build_stage_failure_payload(
    stage_name: str,
    result: Dict[str, Any],
    **extra_fields: Any,
) -> Dict[str, Any]:
    """
    功能：构造 stage 失败诊断载荷。

    Build a structured failure payload for one workflow stage.

    Args:
        stage_name: Workflow stage name.
        result: Stage execution result mapping.
        extra_fields: Additional diagnostic fields.

    Returns:
        JSON-serializable failure payload.
    """
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(result, dict):
        raise TypeError("result must be dict")

    stdout_log_path = result.get("stdout_log_path")
    stderr_log_path = result.get("stderr_log_path")
    failure_payload: Dict[str, Any] = {
        "stage_name": stage_name,
        "return_code": int(result.get("return_code", 1)),
        "status": result.get("status"),
        "command": result.get("command"),
        "stdout_log_path": stdout_log_path,
        "stderr_log_path": stderr_log_path,
        "stdout_tail": _read_log_tail(stdout_log_path),
        "stderr_tail": _read_log_tail(stderr_log_path),
    }
    failure_payload.update(extra_fields)
    return failure_payload


def _build_workflow_exception_summary(
    *,
    stage_run_id: Optional[str],
    config_path: Path,
    run_root: Path,
    prompt_pool: Sequence[str],
    source_pool_stage_results: List[Dict[str, Any]],
    pooled_stage_results: Dict[str, Any],
    exc: Exception,
) -> Dict[str, Any]:
    """
    功能：构造 stage 01 主链异常摘要并供 stderr 与 workflow_summary 复用。

    Build the workflow summary payload for uncaught stage-01 mainline exceptions.

    Args:
        stage_run_id: External stage run identifier.
        config_path: Runtime config path.
        run_root: Workflow run root.
        prompt_pool: Prompt pool sequence.
        source_pool_stage_results: Completed source-pool stage results.
        pooled_stage_results: Completed pooled stage results.
        exc: Raised exception.

    Returns:
        JSON-serializable workflow summary payload.
    """
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(prompt_pool, Sequence):
        raise TypeError("prompt_pool must be Sequence")
    if not isinstance(source_pool_stage_results, list):
        raise TypeError("source_pool_stage_results must be list")
    if not isinstance(pooled_stage_results, dict):
        raise TypeError("pooled_stage_results must be dict")
    if not isinstance(exc, Exception):
        raise TypeError("exc must be Exception")

    return {
        "stage_name": "01_Paper_Full_Cuda_mainline",
        "stage_run_id": stage_run_id,
        "config_path": normalize_path_value(config_path),
        "run_root": normalize_path_value(run_root),
        "status": "failed",
        "source_pool_prompt_count": len(prompt_pool),
        "source_pool_stage_results": source_pool_stage_results,
        "pooled_stage_results": pooled_stage_results,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }


def _build_canonical_source_entry_package_relative_path(index: int) -> str:
    """
    功能：构造 canonical source entry 的包内相对路径。

    Build the package-relative path for one canonical source entry file.

    Args:
        index: Prompt index.

    Returns:
        Package-relative JSON path.
    """
    if index < 0:
        raise TypeError("index must be non-negative int")
    return f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/{index:03d}_source_entry.json"


def _build_prompt_scoped_package_relative_path(relative_root: str, index: int, file_name: str) -> str:
    """
    功能：为单条 prompt 构造 canonical source artifact 包内路径。

    Build the package-relative path for one prompt-scoped canonical source
    artifact.

    Args:
        relative_root: Root package-relative directory.
        index: Prompt index.
        file_name: Artifact file name.

    Returns:
        Prompt-scoped package-relative path.
    """
    if not isinstance(relative_root, str) or not relative_root:
        raise TypeError("relative_root must be non-empty str")
    if index < 0:
        raise TypeError("index must be non-negative int")
    if not isinstance(file_name, str) or not file_name:
        raise TypeError("file_name must be non-empty str")
    return f"{relative_root}/prompt_{index:03d}/{file_name}"


def _build_optional_canonical_artifact_view(
    *,
    run_root: Path,
    source_path: Optional[Path],
    package_relative_path: Optional[str],
    missing_reason: str,
) -> Dict[str, Any]:
    """
    功能：构造 optional canonical source artifact 的显式视图。

    Build an explicit artifact view for an optional canonical source artifact.

    Args:
        run_root: Stage-01 run root.
        source_path: Source artifact path when discoverable.
        package_relative_path: Canonical package-relative target path.
        missing_reason: Stable missing-state reason.

    Returns:
        Artifact view carrying existence, path, and package-relative metadata.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if source_path is not None and not isinstance(source_path, Path):
        raise TypeError("source_path must be Path or None")
    if package_relative_path is not None and (
        not isinstance(package_relative_path, str) or not package_relative_path
    ):
        raise TypeError("package_relative_path must be non-empty str or None")
    if not isinstance(missing_reason, str) or not missing_reason:
        raise TypeError("missing_reason must be non-empty str")

    staged_path: Optional[Path] = None
    if isinstance(package_relative_path, str):
        staged_path = run_root / package_relative_path

    if isinstance(source_path, Path) and source_path.exists() and source_path.is_file():
        if staged_path is None:
            raise ValueError("package_relative_path is required when source artifact exists")
        copy_file(source_path, staged_path)
        return {
            "exists": True,
            "path": normalize_path_value(staged_path),
            "package_relative_path": package_relative_path,
            "missing_reason": None,
        }

    return {
        "exists": False,
        "path": normalize_path_value(staged_path) if isinstance(staged_path, Path) else None,
        "package_relative_path": package_relative_path,
        "missing_reason": missing_reason,
    }


def _resolve_source_pool_attestation_views(
    *,
    cfg_obj: Dict[str, Any],
    run_root: Path,
    prompt_run_root: Path,
    prompt_index: int,
) -> Dict[str, Dict[str, Any]]:
    """
    功能：解析 source pool prompt 子运行的 attestation artifact 视图。

    Resolve prompt-scoped attestation artifact views for the stage-01 source
    pool.

    Args:
        cfg_obj: Base runtime config mapping.
        run_root: Stage-01 run root.
        prompt_run_root: Prompt-bound subrun root.
        prompt_index: Prompt index.

    Returns:
        Mapping from canonical attestation artifact name to artifact view.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(prompt_run_root, Path):
        raise TypeError("prompt_run_root must be Path")
    if prompt_index < 0:
        raise TypeError("prompt_index must be non-negative int")

    attestation_cfg = cfg_obj.get("attestation")
    attestation_enabled = (
        isinstance(attestation_cfg, dict)
        and isinstance(attestation_cfg.get("enabled"), bool)
        and attestation_cfg.get("enabled") is True
    )
    attestation_dir = prompt_run_root / "artifacts" / "attestation"
    artifact_specs = {
        "attestation_statement": ("attestation_statement.json", "attestation_statement_not_emitted"),
        "attestation_bundle": ("attestation_bundle.json", "attestation_bundle_not_emitted"),
        "attestation_result": ("attestation_result.json", "attestation_result_not_emitted"),
    }

    artifact_views: Dict[str, Dict[str, Any]] = {}
    for artifact_key, (file_name, missing_reason) in artifact_specs.items():
        package_relative_path = _build_prompt_scoped_package_relative_path(
            CANONICAL_SOURCE_POOL_ATTESTATION_RELATIVE_ROOT,
            prompt_index,
            file_name,
        )
        artifact_views[artifact_key] = _build_optional_canonical_artifact_view(
            run_root=run_root,
            source_path=(attestation_dir / file_name) if attestation_enabled else None,
            package_relative_path=package_relative_path,
            missing_reason="attestation_disabled" if not attestation_enabled else missing_reason,
        )
    return artifact_views


def _resolve_source_pool_source_image_view(
    *,
    cfg_obj: Dict[str, Any],
    run_root: Path,
    prompt_run_root: Path,
    prompt_index: int,
) -> Dict[str, Any]:
    """
    功能：解析 source pool prompt 子运行的 source image 视图。

    Resolve the source-image view for one prompt-bound source-pool subrun.

    Args:
        cfg_obj: Base runtime config mapping.
        run_root: Stage-01 run root.
        prompt_run_root: Prompt-bound subrun root.
        prompt_index: Prompt index.

    Returns:
        Source-image artifact view with explicit missing-state semantics.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(prompt_run_root, Path):
        raise TypeError("prompt_run_root must be Path")
    if prompt_index < 0:
        raise TypeError("prompt_index must be non-negative int")

    embed_cfg = cfg_obj.get("embed")
    if not isinstance(embed_cfg, dict):
        return _build_optional_canonical_artifact_view(
            run_root=run_root,
            source_path=None,
            package_relative_path=None,
            missing_reason="preview_generation_config_missing",
        )
    preview_cfg = embed_cfg.get("preview_generation")
    if not isinstance(preview_cfg, dict):
        return _build_optional_canonical_artifact_view(
            run_root=run_root,
            source_path=None,
            package_relative_path=None,
            missing_reason="preview_generation_config_missing",
        )

    preview_enabled = (
        isinstance(preview_cfg.get("enabled"), bool) and preview_cfg.get("enabled") is True
    )
    if not preview_enabled:
        return _build_optional_canonical_artifact_view(
            run_root=run_root,
            source_path=None,
            package_relative_path=None,
            missing_reason="preview_generation_disabled",
        )

    artifact_rel_path = preview_cfg.get("artifact_rel_path")
    if not isinstance(artifact_rel_path, str) or not artifact_rel_path.strip():
        return _build_optional_canonical_artifact_view(
            run_root=run_root,
            source_path=None,
            package_relative_path=None,
            missing_reason="preview_generation_artifact_rel_path_missing",
        )

    preview_rel_path = Path(artifact_rel_path.strip().replace("\\", "/"))
    package_relative_path = _build_prompt_scoped_package_relative_path(
        CANONICAL_SOURCE_POOL_SOURCE_IMAGES_RELATIVE_ROOT,
        prompt_index,
        preview_rel_path.name,
    )
    return _build_optional_canonical_artifact_view(
        run_root=run_root,
        source_path=prompt_run_root / "artifacts" / preview_rel_path,
        package_relative_path=package_relative_path,
        missing_reason="source_image_not_emitted",
    )


def _run_source_pool_subrun(
    *,
    index: int,
    prompt_text: str,
    prompt_file_path: str,
    cfg_obj: Dict[str, Any],
    run_root: Path,
    record_usage: str,
) -> Dict[str, Any]:
    """
    功能：执行 stage 01 的单条 prompt embed/detect 子运行。

    Run one prompt-bound embed/detect subrun for the stage-01 source pool.

    Args:
        index: Prompt index.
        prompt_text: Prompt text.
        prompt_file_path: Normalized prompt file path.
        cfg_obj: Base runtime config mapping.
        run_root: Stage-01 run root.
        record_usage: Source-pool record usage marker.

    Returns:
        Source-record entry summary with payload and staged paths.
    """
    if index < 0:
        raise TypeError("index must be non-negative int")
    prompt_run_root = ensure_directory(run_root / "source_pool" / f"prompt_{index:03d}")
    runtime_cfg = _build_prompt_runtime_cfg(
        cfg_obj,
        prompt_text=prompt_text,
        prompt_index=index,
        prompt_file_path=prompt_file_path,
    )
    runtime_cfg_path = run_root / SOURCE_POOL_RUNTIME_CONFIG_RELATIVE_ROOT / f"prompt_{index:03d}.yaml"
    write_yaml_mapping(runtime_cfg_path, runtime_cfg)

    stage_results: Dict[str, Any] = {}
    for stage_name in ("embed", "detect"):
        command = _build_stage_command(stage_name, runtime_cfg_path, prompt_run_root)
        result = _run_stage(stage_name, command, prompt_run_root)
        stage_results[stage_name] = result
        if result.get("return_code") != 0:
            failure_payload = _build_stage_failure_payload(
                stage_name,
                result,
                prompt_index=index,
                prompt_text=prompt_text,
                prompt_file=prompt_file_path,
                prompt_run_root=normalize_path_value(prompt_run_root),
            )
            raise RuntimeError(
                "stage 01 source pool stage failed: "
                f"{json.dumps(failure_payload, ensure_ascii=False, sort_keys=True)}"
            )

    source_embed_record_path = prompt_run_root / "records" / "embed_record.json"
    source_detect_record_path = prompt_run_root / "records" / "detect_record.json"
    if not source_embed_record_path.exists() or not source_detect_record_path.exists():
        raise FileNotFoundError(
            "stage 01 source pool subrun missing embed/detect record: "
            f"prompt_index={index}"
        )

    staged_embed_record_path = run_root / SOURCE_POOL_EMBED_RECORDS_RELATIVE_ROOT / f"{index:03d}_embed_record.json"
    copy_file(source_embed_record_path, staged_embed_record_path)

    direct_detect_payload = _normalize_direct_detect_payload(
        _load_json_dict(source_detect_record_path, "source detect record"),
        prompt_text=prompt_text,
        prompt_index=index,
        prompt_file_path=prompt_file_path,
        record_usage=record_usage,
    )
    staged_detect_record_path = run_root / SOURCE_POOL_DETECT_RECORDS_RELATIVE_ROOT / f"{index:03d}_detect_record.json"
    write_json_atomic(staged_detect_record_path, direct_detect_payload)

    content_chain_score = _resolve_content_chain_score(direct_detect_payload)
    event_attestation_score = _resolve_event_attestation_score(direct_detect_payload)
    attestation_views = _resolve_source_pool_attestation_views(
        cfg_obj=cfg_obj,
        run_root=run_root,
        prompt_run_root=prompt_run_root,
        prompt_index=index,
    )
    source_image_view = _resolve_source_pool_source_image_view(
        cfg_obj=cfg_obj,
        run_root=run_root,
        prompt_run_root=prompt_run_root,
        prompt_index=index,
    )
    return {
        "record_kind": "direct",
        "record_usage": record_usage,
        "prompt_index": index,
        "prompt_text": prompt_text,
        "prompt_sha256": _build_prompt_sha256(prompt_text),
        "prompt_file": prompt_file_path,
        "package_relative_path": f"{SOURCE_POOL_DETECT_RECORDS_RELATIVE_ROOT}/{index:03d}_detect_record.json",
        "path": normalize_path_value(staged_detect_record_path),
        "embed_record_path": normalize_path_value(staged_embed_record_path),
        "embed_record_package_relative_path": (
            f"{SOURCE_POOL_EMBED_RECORDS_RELATIVE_ROOT}/{index:03d}_embed_record.json"
        ),
        "runtime_config_path": normalize_path_value(runtime_cfg_path),
        "runtime_config_package_relative_path": (
            f"{SOURCE_POOL_RUNTIME_CONFIG_RELATIVE_ROOT}/prompt_{index:03d}.yaml"
        ),
        "prompt_run_root": normalize_path_value(prompt_run_root),
        "sha256": compute_file_sha256(staged_detect_record_path),
        "label": True,
        "content_chain_score_available": isinstance(content_chain_score, float),
        "event_attestation_score_available": isinstance(event_attestation_score, float),
        "attestation_statement": attestation_views["attestation_statement"],
        "attestation_bundle": attestation_views["attestation_bundle"],
        "attestation_result": attestation_views["attestation_result"],
        "source_image": source_image_view,
        "payload": direct_detect_payload,
        "source_embed_record_path": normalize_path_value(source_embed_record_path),
        "source_detect_record_path": normalize_path_value(source_detect_record_path),
        "stage_results": stage_results,
    }


def _build_stage_01_canonical_source_pool(
    *,
    run_root: Path,
    stage_run_id: str,
    prompt_file_path: str,
    direct_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    功能：写出 stage 01 canonical source pool 根清单与逐 prompt entry 视图。

    Emit the canonical source-pool root manifest and per-prompt source-entry
    views for stage 01.

    Args:
        run_root: Stage-01 run root.
        stage_run_id: Stage run identifier.
        prompt_file_path: Normalized prompt file path.
        direct_entries: Direct source record entries.

    Returns:
        Canonical source-pool manifest payload.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not stage_run_id:
        raise TypeError("stage_run_id must be non-empty str")
    if not prompt_file_path:
        raise TypeError("prompt_file_path must be non-empty str")
    if not direct_entries:
        raise ValueError("direct_entries must be non-empty list")

    canonical_root = ensure_directory(run_root / CANONICAL_SOURCE_POOL_RELATIVE_ROOT)
    entries_root = ensure_directory(run_root / CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT)
    representative_entry = direct_entries[0]
    manifest_entries: List[Dict[str, Any]] = []

    for entry in direct_entries:
        entry_package_relative_path = _build_canonical_source_entry_package_relative_path(entry["prompt_index"])
        entry_path = run_root / entry_package_relative_path
        entry_payload: Dict[str, Any] = {
            "artifact_type": "stage_01_canonical_source_entry",
            "entry_role": "canonical_source_entry",
            "stage_name": "01_Paper_Full_Cuda",
            "stage_run_id": stage_run_id,
            "path": normalize_path_value(entry_path),
            "source_entry_package_relative_path": entry_package_relative_path,
            "prompt_index": entry["prompt_index"],
            "prompt_text": entry["prompt_text"],
            "prompt_sha256": entry["prompt_sha256"],
            "prompt_file": entry["prompt_file"],
            "record_usage": entry["record_usage"],
            "label": entry["label"],
            "content_chain_score_available": entry["content_chain_score_available"],
            "event_attestation_score_available": entry["event_attestation_score_available"],
            "prompt_run_root": entry["prompt_run_root"],
            "runtime_config_path": entry["runtime_config_path"],
            "runtime_config_package_relative_path": entry["runtime_config_package_relative_path"],
            "embed_record_path": entry["embed_record_path"],
            "embed_record_package_relative_path": entry["embed_record_package_relative_path"],
            "detect_record_path": entry["path"],
            "detect_record_package_relative_path": entry["package_relative_path"],
            "detect_record_sha256": entry["sha256"],
            "attestation_statement": entry["attestation_statement"],
            "attestation_bundle": entry["attestation_bundle"],
            "attestation_result": entry["attestation_result"],
            "source_image": entry["source_image"],
            "representative_root_records_alias": entry["prompt_index"] == representative_entry["prompt_index"],
        }
        write_json_atomic(entry_path, entry_payload)
        entry["source_entry_path"] = normalize_path_value(entry_path)
        entry["source_entry_package_relative_path"] = entry_package_relative_path
        manifest_entries.append(
            {
                "prompt_index": entry_payload["prompt_index"],
                "prompt_text": entry_payload["prompt_text"],
                "prompt_sha256": entry_payload["prompt_sha256"],
                "source_entry_package_relative_path": entry_payload["source_entry_package_relative_path"],
                "detect_record_path": entry_payload["detect_record_path"],
                "detect_record_package_relative_path": entry_payload["detect_record_package_relative_path"],
                "embed_record_path": entry_payload["embed_record_path"],
                "embed_record_package_relative_path": entry_payload["embed_record_package_relative_path"],
                "runtime_config_path": entry_payload["runtime_config_path"],
                "runtime_config_package_relative_path": entry_payload["runtime_config_package_relative_path"],
                "attestation_statement": entry_payload["attestation_statement"],
                "attestation_bundle": entry_payload["attestation_bundle"],
                "attestation_result": entry_payload["attestation_result"],
                "source_image": entry_payload["source_image"],
                "representative_root_records_alias": entry_payload["representative_root_records_alias"],
            }
        )

    representative_root_records = {
        "view_role": "representative_summary_view",
        "root_embed_record_package_relative_path": "records/embed_record.json",
        "root_detect_record_package_relative_path": "records/detect_record.json",
        "source_prompt_index": representative_entry["prompt_index"],
        "source_prompt_sha256": representative_entry["prompt_sha256"],
        "source_entry_package_relative_path": representative_entry["source_entry_package_relative_path"],
        "source_embed_record_package_relative_path": representative_entry["embed_record_package_relative_path"],
        "source_detect_record_package_relative_path": representative_entry["package_relative_path"],
    }
    manifest_path = run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH
    manifest_payload: Dict[str, Any] = {
        "artifact_type": "stage_01_canonical_source_pool",
        "artifact_role": "canonical_source_pool_root",
        "artifact_version": "v1",
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": stage_run_id,
        "prompt_file": prompt_file_path,
        "canonical_source_pool_root_path": normalize_path_value(canonical_root),
        "canonical_source_pool_root_package_relative_path": CANONICAL_SOURCE_POOL_RELATIVE_ROOT,
        "manifest_path": normalize_path_value(manifest_path),
        "manifest_package_relative_path": CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH,
        "entries_root_path": normalize_path_value(entries_root),
        "entries_package_relative_root": CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT,
        "prompt_count": len(manifest_entries),
        "entry_count": len(manifest_entries),
        "entries": manifest_entries,
        "representative_root_records": representative_root_records,
    }
    write_json_atomic(manifest_path, manifest_payload)
    return manifest_payload


def _build_stage_01_source_contract(
    *,
    stage_run_id: str,
    direct_entries: List[Dict[str, Any]],
    canonical_source_pool_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    功能：构造 stage 01 direct source contract。

    Build the stage-01 direct source contract consumed by stage 02 and used for
    source-pool auditability.

    Args:
        stage_run_id: Stage run identifier.
        direct_entries: Direct source record entries.
        canonical_source_pool_payload: Canonical source-pool manifest payload.

    Returns:
        Source contract payload.
    """
    if not stage_run_id:
        raise TypeError("stage_run_id must be non-empty str")
    if not direct_entries:
        raise ValueError("direct_entries must be non-empty list")
    if not isinstance(canonical_source_pool_payload, dict):
        raise TypeError("canonical_source_pool_payload must be dict")

    positive_count = sum(1 for entry in direct_entries if entry.get("label") is True)
    negative_count = sum(1 for entry in direct_entries if entry.get("label") is False)
    unknown_count = len(direct_entries) - positive_count - negative_count
    content_chain_available_count = sum(
        1 for entry in direct_entries if entry.get("content_chain_score_available") is True
    )
    event_attestation_available_count = sum(
        1 for entry in direct_entries if entry.get("event_attestation_score_available") is True
    )
    direct_stats_ready = positive_count > 0 and negative_count > 0 and unknown_count == 0
    direct_stats_reason = "ok"
    if not direct_stats_ready:
        if unknown_count > 0:
            direct_stats_reason = "detect_record_label_missing"
        else:
            direct_stats_reason = "parallel_attestation_statistics_requires_label_balanced_detect_records"

    return {
        "artifact_type": "parallel_attestation_statistics_input_contract",
        "contract_role": "source_contract",
        "contract_version": "v1",
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": stage_run_id,
        "status": "ok",
        "reason": "stage_01_direct_source_pool_ready",
        "source_authority": "canonical_source_pool",
        "contract_view_role": "stage_02_compatibility_view",
        "canonical_source_pool_manifest_package_relative_path": canonical_source_pool_payload[
            "manifest_package_relative_path"
        ],
        "canonical_source_pool_entries_package_relative_root": canonical_source_pool_payload[
            "entries_package_relative_root"
        ],
        "representative_root_records": canonical_source_pool_payload["representative_root_records"],
        "score_name": EVENT_ATTESTATION_SCORE_NAME,
        "threshold_score_name": CONTENT_CHAIN_SCORE_NAME,
        "source_records_available": True,
        "record_count": len(direct_entries),
        "label_summary": {
            "positive": positive_count,
            "negative": negative_count,
            "unknown": unknown_count,
            "label_balanced": direct_stats_ready,
        },
        "score_availability": {
            CONTENT_CHAIN_SCORE_NAME: {
                "available_record_count": content_chain_available_count,
                "missing_record_count": len(direct_entries) - content_chain_available_count,
            },
            EVENT_ATTESTATION_SCORE_NAME: {
                "available_record_count": event_attestation_available_count,
                "missing_record_count": len(direct_entries) - event_attestation_available_count,
            },
        },
        "direct_stats_ready": direct_stats_ready,
        "direct_stats_reason": direct_stats_reason,
        "records": [
            {
                "record_role": "direct_source_record",
                "usage": entry["record_usage"],
                "package_relative_path": entry["package_relative_path"],
                "path": entry["path"],
                "sha256": entry["sha256"],
                "label": entry["label"],
                "prompt_index": entry["prompt_index"],
                "prompt_text": entry["prompt_text"],
                "prompt_sha256": entry["prompt_sha256"],
                "prompt_file": entry["prompt_file"],
                "canonical_source_entry_package_relative_path": entry["source_entry_package_relative_path"],
                "embed_record_package_relative_path": entry["embed_record_package_relative_path"],
                "runtime_config_package_relative_path": entry["runtime_config_package_relative_path"],
                "score_name": EVENT_ATTESTATION_SCORE_NAME,
                "score_available": entry["event_attestation_score_available"],
                "threshold_score_name": CONTENT_CHAIN_SCORE_NAME,
                "threshold_score_available": entry["content_chain_score_available"],
            }
            for entry in direct_entries
        ],
    }


def _build_stage_01_pooled_threshold_records(
    *,
    run_root: Path,
    stage_run_id: str,
    prompt_file_path: str,
    direct_entries: List[Dict[str, Any]],
    build_cfg: Dict[str, Any],
    canonical_source_pool_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    功能：构造 stage 01 pooled threshold records 与 build contract 负载。

    Build the pooled threshold detect-record set and the corresponding build
    contract payload.

    Args:
        run_root: Stage-01 run root.
        stage_run_id: Stage run identifier.
        prompt_file_path: Normalized prompt file path.
        direct_entries: Direct source record entries.
        build_cfg: Normalized pooled-threshold build config.
        canonical_source_pool_payload: Canonical source-pool manifest payload.

    Returns:
        Build contract payload.
    """
    if not stage_run_id:
        raise TypeError("stage_run_id must be non-empty str")
    if not prompt_file_path:
        raise TypeError("prompt_file_path must be non-empty str")
    if not direct_entries:
        raise ValueError("direct_entries must be non-empty list")
    if not isinstance(canonical_source_pool_payload, dict):
        raise TypeError("canonical_source_pool_payload must be dict")
    direct_positive_count = sum(1 for entry in direct_entries if entry.get("label") is True)
    direct_negative_count = sum(1 for entry in direct_entries if entry.get("label") is False)
    desired_pair_count = max(build_cfg["target_pair_count"], direct_positive_count, direct_negative_count)
    missing_negative = max(0, desired_pair_count - direct_negative_count)
    needs_derived = missing_negative > 0
    requested_build_mode = build_cfg["build_mode"]
    resolved_build_mode = DIRECT_SOURCE_ONLY_MODE
    if needs_derived:
        if requested_build_mode == DIRECT_SOURCE_ONLY_MODE:
            raise ValueError(
                "stage 01 pooled thresholds direct_source_only requires label-balanced direct source records: "
                f"positive={direct_positive_count}, negative={direct_negative_count}"
            )
        resolved_build_mode = SOURCE_PLUS_DERIVED_PAIRS_MODE

    pooled_records_root = ensure_directory(run_root / POOLED_THRESHOLD_RECORDS_RELATIVE_ROOT)
    direct_records: List[Dict[str, Any]] = []
    direct_record_count = 0
    for entry in direct_entries:
        direct_payload = copy.deepcopy(entry["payload"])
        build_metadata: Dict[str, Any] = {
            "record_origin": "direct_record",
            "build_usage": build_cfg["build_usage"],
            "source_stage_name": "01_Paper_Full_Cuda",
            "source_stage_run_id": stage_run_id,
            "derived_from": None,
            "derivation_kind": None,
            "label": True,
            "prompt_file": prompt_file_path,
            "prompt_index": entry["prompt_index"],
            "prompt_text": entry["prompt_text"],
        }
        direct_payload["stage_01_pooled_threshold_build"] = build_metadata
        content_node = direct_payload.get("content_evidence_payload")
        content_payload: Dict[str, Any] = {}
        if isinstance(content_node, dict):
            content_payload = dict(cast(Dict[str, Any], content_node))
        content_payload["status"] = "ok"
        content_payload["calibration_sample_origin"] = "stage_01_direct_source_pool"
        content_payload["calibration_sample_usage"] = build_cfg["build_usage"]
        direct_payload["content_evidence_payload"] = content_payload

        staged_file_name = f"{direct_record_count:03d}_direct_positive.json"
        staged_path = pooled_records_root / staged_file_name
        write_json_atomic(staged_path, direct_payload)
        direct_records.append(
            {
                "record_kind": "direct",
                "label": True,
                "usage": build_cfg["build_usage"],
                "derived_from": None,
                "derivation_kind": None,
                "source_package_relative_path": entry["package_relative_path"],
                "source_entry_package_relative_path": entry["source_entry_package_relative_path"],
                "staged_path": normalize_path_value(staged_path),
                "package_relative_path": f"{POOLED_THRESHOLD_RECORDS_RELATIVE_ROOT}/{staged_file_name}",
                "sha256": compute_file_sha256(staged_path),
                "prompt_file": prompt_file_path,
                "prompt_index": entry["prompt_index"],
                "prompt_text": entry["prompt_text"],
            }
        )
        direct_record_count += 1

    derived_records: List[Dict[str, Any]] = []
    for derived_index in range(missing_negative):
        source_entry = direct_entries[derived_index % len(direct_entries)]
        source_score = _resolve_content_chain_score(source_entry["payload"])
        if not isinstance(source_score, float):
            raise ValueError(
                "stage 01 pooled threshold build requires content_chain_score on every direct source record: "
                f"prompt_index={source_entry['prompt_index']}"
            )
        derived_payload = copy.deepcopy(source_entry["payload"])
        derived_payload["label"] = False
        derived_payload["ground_truth"] = False
        derived_payload["is_watermarked"] = False
        derived_payload["ground_truth_source"] = "stage_01_pooled_threshold_derived_negative"
        derived_payload["run_id"] = str(uuid.uuid4())
        build_metadata: Dict[str, Any] = {
            "record_origin": "derived_record",
            "build_usage": build_cfg["build_usage"],
            "source_stage_name": "01_Paper_Full_Cuda",
            "source_stage_run_id": stage_run_id,
            "derived_from": source_entry["package_relative_path"],
            "derivation_kind": build_cfg["record_derivation_kind"],
            "label": False,
            "prompt_file": prompt_file_path,
            "prompt_index": source_entry["prompt_index"],
            "prompt_text": source_entry["prompt_text"],
        }
        derived_payload["stage_01_pooled_threshold_build"] = build_metadata
        content_node = derived_payload.get("content_evidence_payload")
        content_payload: Dict[str, Any] = {}
        if isinstance(content_node, dict):
            content_payload = dict(cast(Dict[str, Any], content_node))
        content_payload["status"] = "ok"
        content_payload["calibration_sample_origin"] = "stage_01_pooled_threshold_derived_negative"
        content_payload["calibration_sample_usage"] = build_cfg["build_usage"]
        negative_score = float(source_score - 1.0 - source_entry["prompt_index"] * 1e-6)
        content_payload["score"] = negative_score
        content_payload[CONTENT_CHAIN_SCORE_NAME] = negative_score
        content_payload["content_score"] = negative_score
        derived_payload["content_evidence_payload"] = content_payload
        _apply_negative_attestation_semantics(derived_payload)

        staged_file_name = f"{direct_record_count + derived_index:03d}_derived_negative.json"
        staged_path = pooled_records_root / staged_file_name
        write_json_atomic(staged_path, derived_payload)
        derived_records.append(
            {
                "record_kind": "derived",
                "label": False,
                "usage": build_cfg["build_usage"],
                "derived_from": source_entry["package_relative_path"],
                "derivation_kind": build_cfg["record_derivation_kind"],
                "source_package_relative_path": source_entry["package_relative_path"],
                "source_entry_package_relative_path": source_entry["source_entry_package_relative_path"],
                "staged_path": normalize_path_value(staged_path),
                "package_relative_path": f"{POOLED_THRESHOLD_RECORDS_RELATIVE_ROOT}/{staged_file_name}",
                "sha256": compute_file_sha256(staged_path),
                "prompt_file": prompt_file_path,
                "prompt_index": source_entry["prompt_index"],
                "prompt_text": source_entry["prompt_text"],
            }
        )

    final_records = [*direct_records, *derived_records]
    final_positive_count = sum(1 for entry in final_records if entry["label"] is True)
    final_negative_count = sum(1 for entry in final_records if entry["label"] is False)
    if final_positive_count <= 0 or final_positive_count != final_negative_count:
        raise ValueError(
            "stage 01 pooled threshold build must produce balanced positive/negative inputs: "
            f"positive={final_positive_count}, negative={final_negative_count}"
        )

    thresholds_artifact_path = run_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
    threshold_metadata_artifact_path = run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json"
    return {
        "artifact_type": "stage_01_pooled_threshold_build_contract",
        "contract_role": "pooled_threshold_build_contract",
        "contract_version": "v1",
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": stage_run_id,
        "requested_build_mode": requested_build_mode,
        "build_mode": resolved_build_mode,
        "source_authority": "canonical_source_pool",
        "canonical_source_pool_manifest_package_relative_path": canonical_source_pool_payload[
            "manifest_package_relative_path"
        ],
        "score_name": CONTENT_CHAIN_SCORE_NAME,
        "prompt_file": prompt_file_path,
        "prompt_pool_summary": {
            "prompt_count": len(direct_entries),
            "prompt_indices": [entry["prompt_index"] for entry in direct_entries],
        },
        "staged_records_root": normalize_path_value(pooled_records_root),
        "detect_records_glob": normalize_path_value(pooled_records_root / "*.json"),
        "direct_record_count": len(direct_records),
        "derived_record_count": len(derived_records),
        "final_record_count": len(final_records),
        "direct_summary": {
            "positive": direct_positive_count,
            "negative": direct_negative_count,
        },
        "derived_summary": {
            "positive": 0,
            "negative": len(derived_records),
        },
        "final_positive_count": final_positive_count,
        "final_negative_count": final_negative_count,
        "final_label_balanced": True,
        "build_configuration": {
            "target_pair_count": build_cfg["target_pair_count"],
            "build_usage": build_cfg["build_usage"],
            "record_derivation_kind": build_cfg["record_derivation_kind"],
        },
        "stats_input_set": {
            "score_name": CONTENT_CHAIN_SCORE_NAME,
            "detect_records_glob": normalize_path_value(pooled_records_root / "*.json"),
            "direct_record_count": len(direct_records),
            "derived_record_count": len(derived_records),
            "final_record_count": len(final_records),
            "positive_record_count": final_positive_count,
            "negative_record_count": final_negative_count,
            "thresholds_artifact_path": normalize_path_value(thresholds_artifact_path),
            "threshold_metadata_artifact_path": normalize_path_value(threshold_metadata_artifact_path),
        },
        "records": final_records,
        "direct_records": direct_records,
        "derived_records": derived_records,
        "thresholds_artifact_path": normalize_path_value(thresholds_artifact_path),
        "threshold_metadata_artifact_path": normalize_path_value(threshold_metadata_artifact_path),
    }


def _build_pooled_runtime_config(
    cfg_obj: Dict[str, Any],
    build_contract_payload: Dict[str, Any],
    run_root: Path,
) -> Dict[str, Any]:
    """
    功能：为 pooled calibrate/evaluate 构造运行时配置。

    Build the runtime config used by the pooled calibrate/evaluate stages.

    Args:
        cfg_obj: Base runtime config mapping.
        build_contract_payload: Pooled threshold build contract payload.
        run_root: Stage-01 run root.

    Returns:
        Runtime config mapping.
    """
    config_copy = json.loads(json.dumps(cfg_obj))
    detect_records_glob = build_contract_payload.get("detect_records_glob")
    if not isinstance(detect_records_glob, str) or not detect_records_glob:
        raise ValueError("build_contract_payload.detect_records_glob must be non-empty str")
    calibration_cfg = dict(config_copy.get("calibration")) if isinstance(config_copy.get("calibration"), dict) else {}
    calibration_cfg["detect_records_glob"] = detect_records_glob
    config_copy["calibration"] = calibration_cfg

    evaluate_cfg = dict(config_copy.get("evaluate")) if isinstance(config_copy.get("evaluate"), dict) else {}
    evaluate_cfg["detect_records_glob"] = detect_records_glob
    evaluate_cfg["thresholds_path"] = str((run_root / "artifacts" / "thresholds" / "thresholds_artifact.json").resolve())
    config_copy["evaluate"] = evaluate_cfg
    return config_copy


def run_paper_full_cuda(config_path: Path, run_root: Path, stage_run_id: Optional[str] = None) -> int:
    """
    功能：执行 stage 01 pooled mainline workflow。

    Execute the stage-01 workflow with an explicit source pool and pooled
    threshold build.

    Args:
        config_path: Runtime config path.
        run_root: Workflow run root.
        stage_run_id: Optional external stage run identifier.

    Returns:
        Process-style status code.
    """
    if stage_run_id is not None and not stage_run_id:
        raise TypeError("stage_run_id must be non-empty str or None")

    cfg_obj = load_yaml_mapping(config_path)
    source_pool_cfg = _resolve_stage_01_source_pool_cfg(cfg_obj)
    build_cfg = _resolve_stage_01_pooled_threshold_build_cfg(cfg_obj)
    if source_pool_cfg["enabled"] is not True:
        raise ValueError("stage 01 now requires stage_01_source_pool.enabled=true")
    if build_cfg["enabled"] is not True:
        raise ValueError("stage 01 now requires stage_01_pooled_threshold_build.enabled=true")

    ensure_directory(run_root)
    ensure_directory(run_root / "artifacts")
    ensure_directory(run_root / "records")

    prompt_pool, prompt_file_path = _resolve_stage_01_prompt_pool(cfg_obj)
    direct_entries: List[Dict[str, Any]] = []
    source_pool_stage_results: List[Dict[str, Any]] = []
    representative_embed_record_path: Optional[Path] = None
    representative_detect_record_path: Optional[Path] = None
    pooled_stage_results: Dict[str, Any] = {}

    try:
        for prompt_index, prompt_text in enumerate(prompt_pool):
            direct_entry = _run_source_pool_subrun(
                index=prompt_index,
                prompt_text=prompt_text,
                prompt_file_path=prompt_file_path,
                cfg_obj=cfg_obj,
                run_root=run_root,
                record_usage=source_pool_cfg["record_usage"],
            )
            direct_entries.append(direct_entry)
            source_pool_stage_results.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_text": prompt_text,
                    "stage_results": direct_entry["stage_results"],
                    "package_relative_path": direct_entry["package_relative_path"],
                }
            )
            if representative_embed_record_path is None:
                representative_embed_record_path = run_root / SOURCE_POOL_EMBED_RECORDS_RELATIVE_ROOT / f"{prompt_index:03d}_embed_record.json"
                representative_detect_record_path = Path(str(direct_entry["path"]))

        if representative_embed_record_path is None or representative_detect_record_path is None:
            raise RuntimeError("stage 01 source pool did not emit representative embed/detect records")

        copy_file(representative_embed_record_path, run_root / "records" / "embed_record.json")
        copy_file(representative_detect_record_path, run_root / "records" / "detect_record.json")

        canonical_source_pool_payload = _build_stage_01_canonical_source_pool(
            run_root=run_root,
            stage_run_id=stage_run_id or "stage_01",
            prompt_file_path=prompt_file_path,
            direct_entries=direct_entries,
        )

        source_contract_payload = _build_stage_01_source_contract(
            stage_run_id=stage_run_id or "stage_01",
            direct_entries=direct_entries,
            canonical_source_pool_payload=canonical_source_pool_payload,
        )
        source_contract_path = run_root / SOURCE_CONTRACT_RELATIVE_PATH
        write_json_atomic(source_contract_path, source_contract_payload)

        pooled_threshold_build_contract_payload = _build_stage_01_pooled_threshold_records(
            run_root=run_root,
            stage_run_id=stage_run_id or "stage_01",
            prompt_file_path=prompt_file_path,
            direct_entries=direct_entries,
            build_cfg=build_cfg,
            canonical_source_pool_payload=canonical_source_pool_payload,
        )
        pooled_runtime_cfg = _build_pooled_runtime_config(cfg_obj, pooled_threshold_build_contract_payload, run_root)
        pooled_runtime_cfg_path = run_root / POOLED_THRESHOLD_RUNTIME_CONFIG_RELATIVE_PATH
        write_yaml_mapping(pooled_runtime_cfg_path, pooled_runtime_cfg)

        for stage_name in ("calibrate", "evaluate"):
            command = _build_stage_command(stage_name, pooled_runtime_cfg_path, run_root)
            result = _run_stage(stage_name, command, run_root)
            pooled_stage_results[stage_name] = result
            if result.get("return_code") != 0:
                failed_stage_payload = _build_stage_failure_payload(stage_name, result)
                summary_payload: Dict[str, Any] = {
                    "stage_name": "01_Paper_Full_Cuda_mainline",
                    "stage_run_id": stage_run_id,
                    "config_path": normalize_path_value(config_path),
                    "run_root": normalize_path_value(run_root),
                    "status": "failed",
                    "source_pool_prompt_count": len(prompt_pool),
                    "source_pool_stage_results": source_pool_stage_results,
                    "pooled_stage_results": pooled_stage_results,
                    "failed_stage": failed_stage_payload,
                }
                write_json_atomic(run_root / "artifacts" / "workflow_summary.json", summary_payload)
                print(json.dumps(summary_payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)
                return int(result.get("return_code", 1))
    except Exception as exc:
        summary_payload = _build_workflow_exception_summary(
            stage_run_id=stage_run_id,
            config_path=config_path,
            run_root=run_root,
            prompt_pool=prompt_pool,
            source_pool_stage_results=source_pool_stage_results,
            pooled_stage_results=pooled_stage_results,
            exc=exc,
        )
        write_json_atomic(run_root / "artifacts" / "workflow_summary.json", summary_payload)
        print(json.dumps(summary_payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1

    write_json_atomic(run_root / POOLED_THRESHOLD_BUILD_CONTRACT_RELATIVE_PATH, pooled_threshold_build_contract_payload)

    parallel_attestation_statistics_cfg = cast(
        Dict[str, Any],
        cfg_obj.get("parallel_attestation_statistics")
        if isinstance(cfg_obj.get("parallel_attestation_statistics"), dict)
        else {},
    )

    workflow_summary_path = run_root / "artifacts" / "workflow_summary.json"
    workflow_summary: Dict[str, Any] = {
        "stage_name": "01_Paper_Full_Cuda_mainline",
        "stage_run_id": stage_run_id,
        "config_path": normalize_path_value(config_path),
        "run_root": normalize_path_value(run_root),
        "status": "pending",
        "source_pool_prompt_count": len(prompt_pool),
        "source_pool_prompt_file": prompt_file_path,
        "canonical_source_pool_manifest_path": normalize_path_value(
            run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH
        ),
        "canonical_source_pool_manifest_package_relative_path": CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH,
        "canonical_source_pool_entry_count": canonical_source_pool_payload["entry_count"],
        "representative_root_records": canonical_source_pool_payload["representative_root_records"],
        "direct_source_record_count": source_contract_payload["record_count"],
        "pooled_threshold_record_count": pooled_threshold_build_contract_payload["final_record_count"],
        "pooled_threshold_build_mode": pooled_threshold_build_contract_payload["build_mode"],
        "parallel_attestation_statistics": {
            "enabled_in_default_config": bool(parallel_attestation_statistics_cfg.get("enabled", False)),
            "status": "detached_not_run",
            "execution_mode": "independent_post_flow",
        },
        "experiment_matrix": {
            "status": "detached_not_run",
            "execution_mode": "independent_post_flow",
        },
        "source_pool_stage_results": source_pool_stage_results,
        "pooled_stage_results": pooled_stage_results,
        "required_artifacts": {},
        "required_artifacts_ok": False,
    }
    write_json_atomic(workflow_summary_path, workflow_summary)

    artifact_summary = _artifact_presence(_required_artifacts(run_root))
    workflow_summary["required_artifacts"] = artifact_summary
    workflow_summary["required_artifacts_ok"] = _all_required_present(artifact_summary)
    workflow_summary["status"] = "ok" if workflow_summary["required_artifacts_ok"] else "failed"
    write_json_atomic(workflow_summary_path, workflow_summary)
    return 0 if workflow_summary["status"] == "ok" else 1


def main() -> int:
    """
    功能：stage 01 runner CLI 入口。

    Entry point for the stage-01 runner script.

    Args:
        None.

    Returns:
        Process-style exit code.
    """
    parser = argparse.ArgumentParser(description="Run the stage-01 Paper_Full_Cuda mainline workflow.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH.as_posix()), help="Runtime config path.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT.as_posix()), help="Workflow run root.")
    parser.add_argument("--stage-run-id", default=None, help="Optional external stage run identifier.")
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config)
    run_root = _resolve_repo_path(args.run_root)
    return run_paper_full_cuda(config_path, run_root, stage_run_id=args.stage_run_id)


if __name__ == "__main__":
    sys.exit(main())
