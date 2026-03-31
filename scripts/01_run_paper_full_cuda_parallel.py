"""
文件目的：执行 stage 01 prompt-level 双 worker 并行主链。
Module type: General module

职责边界：
1. 仅将 source pool prompt 子任务拆分到多个 worker，并保持 downstream pooled 阶段串行。
2. 所有下游 contract、threshold、calibrate、evaluate 继续复用既有正式 helper。
3. 不修改既有单路线 runner，仅新增隔离的并行入口与 worker 工件。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts.notebook_runtime_common import (
    REPO_ROOT,
    build_repo_import_subprocess_env,
    ensure_directory,
    load_yaml_mapping,
    normalize_path_value,
    read_json_dict,
    resolve_repo_path,
    write_json_atomic,
    write_yaml_mapping,
)


BASE_RUNNER_SCRIPT_PATH = Path("scripts/01_run_paper_full_cuda.py")
WORKER_SCRIPT_PATH = Path("scripts/01_run_paper_full_cuda_parallel_worker.py")
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_RUN_ROOT = Path("outputs/colab_run_paper_full_cuda_parallel")
DEFAULT_WORKER_COUNT = 2
PARALLEL_STAGE_NAME = "01_Paper_Full_Cuda_Parallel_mainline"
WORKER_RESULT_RELATIVE_ROOT = "artifacts/stage_01_parallel_workers"
WORKER_SHARD_PLAN_RELATIVE_PATH = f"{WORKER_RESULT_RELATIVE_ROOT}/shard_plan.json"


def _load_base_runner_module() -> ModuleType:
    """
    功能：按文件路径加载 baseline stage 01 runner。 

    Load the baseline stage-01 runner module from its file path.

    Args:
        None.

    Returns:
        Loaded module object.

    Raises:
        RuntimeError: If the runner module cannot be loaded.
    """
    runner_path = (REPO_ROOT / BASE_RUNNER_SCRIPT_PATH).resolve()
    spec = importlib.util.spec_from_file_location("stage_01_paper_full_cuda_base_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load stage-01 runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_RUNNER_MODULE = _load_base_runner_module()


def _validate_worker_count(worker_count: int) -> None:
    """
    功能：校验并行 worker 总数。 

    Validate the total worker count.

    Args:
        worker_count: Total worker count.

    Returns:
        None.

    Raises:
        TypeError: If worker_count is invalid.
    """
    if worker_count <= 0:
        raise TypeError("worker_count must be positive int")


def _build_worker_assignments(prompt_count: int, worker_count: int) -> List[Dict[str, Any]]:
    """
    功能：构造稳定的 worker 分片计划。 

    Build the stable worker shard plan for the prompt pool.

    Args:
        prompt_count: Total prompt count.
        worker_count: Total worker count.

    Returns:
        Ordered worker-assignment mappings.
    """
    if prompt_count < 0:
        raise TypeError("prompt_count must be non-negative int")
    _validate_worker_count(worker_count)
    return [
        {
            "worker_index": worker_index,
            "assigned_prompt_indices": [
                prompt_index for prompt_index in range(prompt_count) if prompt_index % worker_count == worker_index
            ],
        }
        for worker_index in range(worker_count)
    ]


def _worker_result_path(run_root: Path, worker_index: int) -> Path:
    """
    功能：构造 worker 结果文件路径。 

    Build the result-file path for one worker.

    Args:
        run_root: Stage run root.
        worker_index: Zero-based worker index.

    Returns:
        Result JSON path.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if worker_index < 0:
        raise TypeError("worker_index must be non-negative int")
    return run_root / WORKER_RESULT_RELATIVE_ROOT / f"worker_{worker_index:02d}_result.json"


def _worker_log_paths(run_root: Path, worker_index: int) -> Tuple[Path, Path]:
    """
    功能：构造 worker stdout / stderr 日志路径。 

    Build the stdout and stderr log paths for one worker.

    Args:
        run_root: Stage run root.
        worker_index: Zero-based worker index.

    Returns:
        Tuple of stdout log path and stderr log path.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if worker_index < 0:
        raise TypeError("worker_index must be non-negative int")
    logs_root = ensure_directory(run_root / "logs" / "stage_01_parallel_workers")
    return (
        logs_root / f"worker_{worker_index:02d}_stdout.log",
        logs_root / f"worker_{worker_index:02d}_stderr.log",
    )


def _build_worker_command(
    *,
    config_path: Path,
    run_root: Path,
    stage_run_id: Optional[str],
    worker_index: int,
    worker_count: int,
) -> List[str]:
    """
    功能：构造 worker 子进程命令。 

    Build the subprocess command for one worker.

    Args:
        config_path: Runtime config path.
        run_root: Workflow run root.
        stage_run_id: External stage run identifier.
        worker_index: Zero-based worker index.
        worker_count: Total worker count.

    Returns:
        Command list suitable for subprocess execution.
    """
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    _validate_worker_count(worker_count)
    if worker_index < 0 or worker_index >= worker_count:
        raise TypeError("worker_index must satisfy 0 <= worker_index < worker_count")
    command = [
        sys.executable,
        str((REPO_ROOT / WORKER_SCRIPT_PATH).resolve()),
        "--config",
        str(config_path),
        "--run-root",
        str(run_root),
        "--worker-index",
        str(worker_index),
        "--worker-count",
        str(worker_count),
    ]
    if stage_run_id:
        command.extend(["--stage-run-id", stage_run_id])
    return command


def _write_worker_shard_plan(
    *,
    run_root: Path,
    stage_run_id: Optional[str],
    prompt_file_path: str,
    prompt_count: int,
    worker_count: int,
    assignments: Sequence[Dict[str, Any]],
) -> None:
    """
    功能：写出 worker 分片计划工件。 

    Write the worker shard-plan artifact for auditability.

    Args:
        run_root: Workflow run root.
        stage_run_id: External stage run identifier.
        prompt_file_path: Normalized prompt file path.
        prompt_count: Total prompt count.
        worker_count: Total worker count.
        assignments: Worker assignments.

    Returns:
        None.
    """
    shard_plan_path = run_root / WORKER_SHARD_PLAN_RELATIVE_PATH
    ensure_directory(shard_plan_path.parent)
    write_json_atomic(
        shard_plan_path,
        {
            "artifact_type": "stage_01_parallel_worker_shard_plan",
            "stage_name": PARALLEL_STAGE_NAME,
            "stage_run_id": stage_run_id,
            "prompt_file": prompt_file_path,
            "prompt_count": prompt_count,
            "worker_count": worker_count,
            "assignments": list(assignments),
        },
    )


def _load_worker_result(result_path: Path) -> Dict[str, Any]:
    """
    功能：读取单个 worker 结果 JSON。 

    Read one worker-result JSON document.

    Args:
        result_path: Worker result path.

    Returns:
        Parsed result mapping.
    """
    if not isinstance(result_path, Path):
        raise TypeError("result_path must be Path")
    return read_json_dict(result_path)


def _merge_worker_results(
    worker_results: Sequence[Dict[str, Any]],
    *,
    prompt_count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    功能：聚合并校验所有 worker 输出。 

    Merge and validate all worker outputs.

    Args:
        worker_results: Worker result mappings.
        prompt_count: Total prompt count.

    Returns:
        Ordered direct entries and ordered source-pool stage results.

    Raises:
        ValueError: If prompt coverage is incomplete or duplicated.
    """
    if prompt_count < 0:
        raise TypeError("prompt_count must be non-negative int")

    direct_entries: List[Dict[str, Any]] = []
    source_pool_stage_results: List[Dict[str, Any]] = []
    direct_indices: set[int] = set()
    stage_result_indices: set[int] = set()

    for worker_result in sorted(worker_results, key=lambda item: int(item["worker_index"])):
        for entry in worker_result.get("direct_entries", []):
            prompt_index = int(entry["prompt_index"])
            if prompt_index in direct_indices:
                raise ValueError(f"duplicate direct entry for prompt_index={prompt_index}")
            direct_indices.add(prompt_index)
            direct_entries.append(entry)

        for stage_result in worker_result.get("source_pool_stage_results", []):
            prompt_index = int(stage_result["prompt_index"])
            if prompt_index in stage_result_indices:
                raise ValueError(f"duplicate stage result for prompt_index={prompt_index}")
            stage_result_indices.add(prompt_index)
            source_pool_stage_results.append(stage_result)

    expected_indices = set(range(prompt_count))
    if direct_indices != expected_indices:
        raise ValueError(
            "parallel source pool direct entry coverage mismatch: "
            f"expected={sorted(expected_indices)}, actual={sorted(direct_indices)}"
        )
    if stage_result_indices != expected_indices:
        raise ValueError(
            "parallel source pool stage-result coverage mismatch: "
            f"expected={sorted(expected_indices)}, actual={sorted(stage_result_indices)}"
        )

    direct_entries.sort(key=lambda item: int(item["prompt_index"]))
    source_pool_stage_results.sort(key=lambda item: int(item["prompt_index"]))
    return direct_entries, source_pool_stage_results


def _collect_partial_source_pool_stage_results(worker_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    功能：收集已完成的 prompt 级 stage 结果。 

    Collect the completed prompt-level stage results from worker outputs.

    Args:
        worker_results: Worker result mappings.

    Returns:
        Sorted partial stage results.
    """
    partial_results: List[Dict[str, Any]] = []
    for worker_result in worker_results:
        partial_results.extend(list(worker_result.get("source_pool_stage_results", [])))
    partial_results.sort(key=lambda item: int(item["prompt_index"]))
    return partial_results


def _build_parallel_worker_failure_payload(worker_executions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    功能：构造并行 source pool worker 失败诊断。 

    Build the structured failure payload for parallel source-pool workers.

    Args:
        worker_executions: Worker execution summaries.

    Returns:
        Failure payload mapping.
    """
    failed_workers = [execution for execution in worker_executions if execution["return_code"] != 0]
    return {
        "stage_name": "source_pool_parallel_workers",
        "status": "failed",
        "worker_count": len(worker_executions),
        "failed_worker_count": len(failed_workers),
        "worker_executions": list(worker_executions),
        "failure_reason": "parallel_source_pool_worker_failed",
    }


def _run_parallel_workers(
    *,
    config_path: Path,
    run_root: Path,
    stage_run_id: Optional[str],
    worker_count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    功能：并发执行所有 source pool workers 并回收结果。 

    Execute all source-pool workers concurrently and collect their results.

    Args:
        config_path: Runtime config path.
        run_root: Workflow run root.
        stage_run_id: External stage run identifier.
        worker_count: Total worker count.

    Returns:
        Worker execution summaries and worker result payloads.
    """
    _validate_worker_count(worker_count)
    env_mapping = build_repo_import_subprocess_env(repo_root=REPO_ROOT)
    processes: List[Dict[str, Any]] = []

    for worker_index in range(worker_count):
        command = _build_worker_command(
            config_path=config_path,
            run_root=run_root,
            stage_run_id=stage_run_id,
            worker_index=worker_index,
            worker_count=worker_count,
        )
        stdout_log_path, stderr_log_path = _worker_log_paths(run_root, worker_index)
        stdout_handle = stdout_log_path.open("w", encoding="utf-8")
        stderr_handle = stderr_log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env_mapping,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        processes.append(
            {
                "worker_index": worker_index,
                "command": command,
                "process": process,
                "stdout_handle": stdout_handle,
                "stderr_handle": stderr_handle,
                "stdout_log_path": stdout_log_path,
                "stderr_log_path": stderr_log_path,
                "result_path": _worker_result_path(run_root, worker_index),
            }
        )

    worker_executions: List[Dict[str, Any]] = []
    worker_results: List[Dict[str, Any]] = []
    for process_info in processes:
        process = process_info["process"]
        return_code = int(process.wait())
        process_info["stdout_handle"].close()
        process_info["stderr_handle"].close()
        execution_summary = {
            "worker_index": process_info["worker_index"],
            "return_code": return_code,
            "command": process_info["command"],
            "stdout_log_path": normalize_path_value(process_info["stdout_log_path"]),
            "stderr_log_path": normalize_path_value(process_info["stderr_log_path"]),
            "result_path": normalize_path_value(process_info["result_path"]),
            "result_exists": process_info["result_path"].exists(),
        }
        worker_executions.append(execution_summary)
        if process_info["result_path"].exists():
            worker_results.append(_load_worker_result(process_info["result_path"]))

    return worker_executions, worker_results


def run_paper_full_cuda_parallel(
    config_path: Path,
    run_root: Path,
    *,
    stage_run_id: Optional[str] = None,
    worker_count: int = DEFAULT_WORKER_COUNT,
) -> int:
    """
    功能：执行 stage 01 prompt-level 并行 workflow。 

    Execute the stage-01 workflow with prompt-level parallel source-pool
    workers and serial downstream pooled stages.

    Args:
        config_path: Runtime config path.
        run_root: Workflow run root.
        stage_run_id: Optional external stage run identifier.
        worker_count: Total worker count.

    Returns:
        Process-style exit code.
    """
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    _validate_worker_count(worker_count)

    cfg_obj = load_yaml_mapping(config_path)
    source_pool_cfg = BASE_RUNNER_MODULE._resolve_stage_01_source_pool_cfg(cfg_obj)
    build_cfg = BASE_RUNNER_MODULE._resolve_stage_01_pooled_threshold_build_cfg(cfg_obj)
    if source_pool_cfg["enabled"] is not True:
        raise ValueError("stage 01 now requires stage_01_source_pool.enabled=true")
    if build_cfg["enabled"] is not True:
        raise ValueError("stage 01 now requires stage_01_pooled_threshold_build.enabled=true")

    ensure_directory(run_root)
    ensure_directory(run_root / "artifacts")
    ensure_directory(run_root / "records")

    prompt_pool, prompt_file_path = BASE_RUNNER_MODULE._resolve_stage_01_prompt_pool(cfg_obj)
    assignments = _build_worker_assignments(len(prompt_pool), worker_count)
    _write_worker_shard_plan(
        run_root=run_root,
        stage_run_id=stage_run_id,
        prompt_file_path=prompt_file_path,
        prompt_count=len(prompt_pool),
        worker_count=worker_count,
        assignments=assignments,
    )

    source_pool_stage_results: List[Dict[str, Any]] = []
    pooled_stage_results: Dict[str, Any] = {}
    worker_executions: List[Dict[str, Any]] = []
    worker_results: List[Dict[str, Any]] = []

    try:
        worker_executions, worker_results = _run_parallel_workers(
            config_path=config_path,
            run_root=run_root,
            stage_run_id=stage_run_id,
            worker_count=worker_count,
        )
        source_pool_stage_results = _collect_partial_source_pool_stage_results(worker_results)

        if len(worker_results) != worker_count or any(result.get("status") != "ok" for result in worker_results):
            summary_payload: Dict[str, Any] = {
                "stage_name": PARALLEL_STAGE_NAME,
                "stage_run_id": stage_run_id,
                "config_path": normalize_path_value(config_path),
                "run_root": normalize_path_value(run_root),
                "status": "failed",
                "source_pool_prompt_count": len(prompt_pool),
                "source_pool_stage_results": source_pool_stage_results,
                "pooled_stage_results": pooled_stage_results,
                "failed_stage": _build_parallel_worker_failure_payload(worker_executions),
            }
            write_json_atomic(run_root / "artifacts" / "workflow_summary.json", summary_payload)
            print(json.dumps(summary_payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)
            return 1

        direct_entries, source_pool_stage_results = _merge_worker_results(worker_results, prompt_count=len(prompt_pool))
        representative_prompt_index = int(direct_entries[0]["prompt_index"])
        representative_embed_record_path = (
            run_root / BASE_RUNNER_MODULE.SOURCE_POOL_EMBED_RECORDS_RELATIVE_ROOT / f"{representative_prompt_index:03d}_embed_record.json"
        )
        representative_detect_record_path = Path(str(direct_entries[0]["path"]))
        shutil.copyfile(representative_embed_record_path, run_root / "records" / "embed_record.json")
        shutil.copyfile(representative_detect_record_path, run_root / "records" / "detect_record.json")

        canonical_source_pool_payload = BASE_RUNNER_MODULE._build_stage_01_canonical_source_pool(
            run_root=run_root,
            stage_run_id=stage_run_id or "stage_01",
            prompt_file_path=prompt_file_path,
            direct_entries=direct_entries,
        )
        source_contract_payload = BASE_RUNNER_MODULE._build_stage_01_source_contract(
            stage_run_id=stage_run_id or "stage_01",
            direct_entries=direct_entries,
            canonical_source_pool_payload=canonical_source_pool_payload,
        )
        write_json_atomic(run_root / BASE_RUNNER_MODULE.SOURCE_CONTRACT_RELATIVE_PATH, source_contract_payload)

        pooled_threshold_build_contract_payload = BASE_RUNNER_MODULE._build_stage_01_pooled_threshold_records(
            run_root=run_root,
            stage_run_id=stage_run_id or "stage_01",
            prompt_file_path=prompt_file_path,
            direct_entries=direct_entries,
            build_cfg=build_cfg,
            canonical_source_pool_payload=canonical_source_pool_payload,
        )
        pooled_runtime_cfg = BASE_RUNNER_MODULE._build_pooled_runtime_config(
            cfg_obj,
            pooled_threshold_build_contract_payload,
            run_root,
        )
        pooled_runtime_cfg_path = run_root / BASE_RUNNER_MODULE.POOLED_THRESHOLD_RUNTIME_CONFIG_RELATIVE_PATH
        write_yaml_mapping(pooled_runtime_cfg_path, pooled_runtime_cfg)

        for stage_name in ("calibrate", "evaluate"):
            command = BASE_RUNNER_MODULE._build_stage_command(stage_name, pooled_runtime_cfg_path, run_root)
            result = BASE_RUNNER_MODULE._run_stage(stage_name, command, run_root)
            pooled_stage_results[stage_name] = result
            if result.get("return_code") != 0:
                failed_stage_payload = BASE_RUNNER_MODULE._build_stage_failure_payload(stage_name, result)
                summary_payload = {
                    "stage_name": PARALLEL_STAGE_NAME,
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
        summary_payload = BASE_RUNNER_MODULE._build_workflow_exception_summary(
            stage_run_id=stage_run_id,
            config_path=config_path,
            run_root=run_root,
            prompt_pool=prompt_pool,
            source_pool_stage_results=source_pool_stage_results,
            pooled_stage_results=pooled_stage_results,
            exc=exc,
        )
        summary_payload["stage_name"] = PARALLEL_STAGE_NAME
        if worker_executions:
            summary_payload["parallel_worker_executions"] = worker_executions
        write_json_atomic(run_root / "artifacts" / "workflow_summary.json", summary_payload)
        print(json.dumps(summary_payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1

    write_json_atomic(run_root / BASE_RUNNER_MODULE.POOLED_THRESHOLD_BUILD_CONTRACT_RELATIVE_PATH, pooled_threshold_build_contract_payload)

    parallel_attestation_statistics_cfg = (
        dict(cfg_obj.get("parallel_attestation_statistics"))
        if isinstance(cfg_obj.get("parallel_attestation_statistics"), dict)
        else {}
    )
    workflow_summary_path = run_root / "artifacts" / "workflow_summary.json"
    workflow_summary: Dict[str, Any] = {
        "stage_name": PARALLEL_STAGE_NAME,
        "stage_run_id": stage_run_id,
        "config_path": normalize_path_value(config_path),
        "run_root": normalize_path_value(run_root),
        "status": "pending",
        "source_pool_prompt_count": len(prompt_pool),
        "source_pool_stage_results": source_pool_stage_results,
        "direct_source_record_count": source_contract_payload["record_count"],
        "canonical_source_pool_manifest": canonical_source_pool_payload["manifest_package_relative_path"],
        "pooled_threshold_build_contract": BASE_RUNNER_MODULE.POOLED_THRESHOLD_BUILD_CONTRACT_RELATIVE_PATH,
        "pooled_stage_results": pooled_stage_results,
        "parallel_attestation_statistics_enabled": parallel_attestation_statistics_cfg.get("enabled") is True,
        "required_artifacts": {},
        "required_artifacts_ok": False,
        "attestation_evidence_resolution": {},
        "attestation_evidence_ok": False,
        "summary_reason": "pending_post_checks",
        "failure_reason": None,
    }
    write_json_atomic(workflow_summary_path, workflow_summary)

    artifact_summary = BASE_RUNNER_MODULE._artifact_presence(BASE_RUNNER_MODULE._required_artifacts(run_root))
    attestation_evidence_resolution = BASE_RUNNER_MODULE._resolve_stage_01_attestation_evidence(
        run_root=run_root,
        canonical_source_pool_payload=canonical_source_pool_payload,
    )
    workflow_summary["required_artifacts"] = artifact_summary
    workflow_summary["required_artifacts_ok"] = BASE_RUNNER_MODULE._all_required_present(artifact_summary)
    workflow_summary["attestation_evidence_resolution"] = attestation_evidence_resolution
    workflow_summary["attestation_evidence_ok"] = attestation_evidence_resolution["overall_status"] == "ok"

    failure_reasons: List[str] = []
    if not workflow_summary["required_artifacts_ok"]:
        failure_reasons.append("required_artifacts_missing")
    if not workflow_summary["attestation_evidence_ok"]:
        failure_reasons.append(str(attestation_evidence_resolution.get("failure_reason") or "failed_attestation_evidence"))

    workflow_summary["summary_reason"] = "ok" if not failure_reasons else "+".join(failure_reasons)
    workflow_summary["failure_reason"] = None if not failure_reasons else "+".join(failure_reasons)
    workflow_summary["status"] = "ok" if not failure_reasons else "failed"
    write_json_atomic(workflow_summary_path, workflow_summary)
    return 0 if workflow_summary["status"] == "ok" else 1


def main() -> int:
    """
    功能：parallel stage 01 runner CLI 入口。 

    Entry point for the prompt-parallel stage-01 runner script.

    Args:
        None.

    Returns:
        Process-style exit code.
    """
    parser = argparse.ArgumentParser(description="Run the stage-01 prompt-parallel Paper_Full_Cuda workflow.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH.as_posix()), help="Runtime config path.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT.as_posix()), help="Workflow run root.")
    parser.add_argument("--stage-run-id", default=None, help="Optional external stage run identifier.")
    parser.add_argument("--worker-count", default=DEFAULT_WORKER_COUNT, type=int, help="Prompt worker count.")
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)
    run_root = resolve_repo_path(args.run_root)
    return run_paper_full_cuda_parallel(
        config_path,
        run_root,
        stage_run_id=args.stage_run_id,
        worker_count=int(args.worker_count),
    )


if __name__ == "__main__":
    sys.exit(main())