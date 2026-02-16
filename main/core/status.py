"""
运行闭包产物生成

功能说明：
- finalize_run() 函数实现了 run_closure.json 的生成逻辑，负责关闭 records bundle、构建 run_closure payload，并将其写入 artifacts 目录。
- build_run_closure_payload() 函数构造符合 schema 的 run_closure 负载，包含对 run_meta 和 manifest 的严格验证，以及状态规范化。
- 包含多个辅助函数用于处理不同的失败场景，确保即使在 records bundle 失败或 manifest anchors 冲突时也能产出可追溯的 run_closure。
- 依赖 core.records_bundle 进行 bundle 关闭，依赖 core.schema 进行 payload 校验，依赖 policy.path_policy 进行输出路径验证。
"""

from __future__ import annotations

import ast
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List

from main.core import records_io
from main.core import digests
from main.core import time_utils
from main.core.records_bundle import close_records_bundle
from main.core.schema import (
    RUN_CLOSURE_SCHEMA_VERSION,
    RECORDS_MANIFEST_NAME,
    RUN_CLOSURE_NAME,
    validate_run_closure
)
from main.core.errors import RunFailureReason, RecordBundleError, FactSourcesNotInitializedError, RecordsWritePolicyError
from main.policy import path_policy


ALLOWED_STATUS_VALUES = ["ok", "failed", "mismatch", "absent"]
ALLOWED_FAIL_REASONS = [
    "missing_required_field",
    "pattern_seed_mismatch",
    "subspace_frame_mismatch",
    "cfg_digest_mismatch",
    "inversion_failed",
    "trace_extraction_failed",
    "evidence_computation_failed",
    "decoder_error",
    "unknown"
]
ALLOWED_MISMATCH_REASONS = ["subspace_frame_mismatch", "cfg_digest_mismatch"]

# pip 子进程超时秒数。
PIP_SUBPROCESS_TIMEOUT_SECONDS = 10

# 进程内缓存：避免在同一进程内多次执行 pip freeze。
_pip_freeze_cache: Optional[List[str]] = None
_pip_version_cache: Optional[str] = None


def validate_status(value: Any, field_path: str = "status") -> str:
    """
    功能：校验 status 枚举值。

    Validate status value against frozen enum list.

    Args:
        value: Status value to validate.
        field_path: Field path for error context.

    Returns:
        Validated status string.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If value is not allowed.
    """
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")
    if not isinstance(value, str) or not value:
        raise TypeError(f"invalid_status: field_path={field_path}, reason=not_str")
    if value not in ALLOWED_STATUS_VALUES:
        raise ValueError(f"invalid_status: field_path={field_path}, reason=not_allowed")
    return value


def validate_fail_reason(value: Any, field_path: str = "fail_reason") -> str:
    """
    功能：校验 fail_reason 枚举值。

    Validate fail_reason value against frozen enum list.

    Args:
        value: Fail reason value to validate.
        field_path: Field path for error context.

    Returns:
        Validated fail_reason string.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If value is not allowed.
    """
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")
    if not isinstance(value, str) or not value:
        raise TypeError(f"invalid_fail_reason: field_path={field_path}, reason=not_str")
    if value not in ALLOWED_FAIL_REASONS:
        raise ValueError(f"invalid_fail_reason: field_path={field_path}, reason=not_allowed")
    return value


def validate_mismatch_reason(value: Any, field_path: str = "mismatch_reason") -> str:
    """
    功能：校验 mismatch_reason 枚举值。

    Validate mismatch_reason value against frozen enum list.

    Args:
        value: Mismatch reason value to validate.
        field_path: Field path for error context.

    Returns:
        Validated mismatch_reason string.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If value is not allowed.
    """
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")
    if not isinstance(value, str) or not value:
        raise TypeError(f"invalid_mismatch_reason: field_path={field_path}, reason=not_str")
    if value not in ALLOWED_MISMATCH_REASONS:
        raise ValueError(f"invalid_mismatch_reason: field_path={field_path}, reason=not_allowed")
    return value


def _build_closure_stage(
    records_present: bool,
    bundle_attempted: bool,
    bundle_succeeded: bool,
    failure_stage: Optional[str] = None,
    upstream_status_reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    构建 closure_stage 审计块。
    
    Build closure stage audit block with precise failure point tracking.
    
    Args:
        records_present: Whether *.json files exist in records_dir.
        bundle_attempted: Whether close_records_bundle() was called.
        bundle_succeeded: Whether bundle succeeded without RecordBundleError.
        failure_stage: Precise failure location or None for success: "bundle"|"anchor_merge"|"unknown".
        upstream_status_reason: Prior failure reason if set (RunFailureReason.value as string or None).
    
    Returns:
        Dict with keys: records_present, bundle_attempted, bundle_succeeded, failure_stage, upstream_status_reason.
    """
    return {
        "records_present": records_present,
        "bundle_attempted": bundle_attempted,
        "bundle_succeeded": bundle_succeeded,
        "failure_stage": failure_stage,
        "upstream_status_reason": upstream_status_reason
    }


def _run_pip_command(args: list[str]) -> str:
    """
    功能：执行 pip 子命令并返回输出。

    Run pip subcommand and return stdout.

    Args:
        args: Command arguments list.

    Returns:
        Stdout string.

    Raises:
        TypeError: If args is invalid.
        RuntimeError: If the command fails or times out.
    """
    if not isinstance(args, list) or not args:
        # args 类型不符合预期，必须 fail-fast。
        raise TypeError("args must be non-empty list")
    for item in args:
        if not isinstance(item, str) or not item:
            # args 成员类型不符合预期，必须 fail-fast。
            raise TypeError("args must contain non-empty str items")

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=PIP_SUBPROCESS_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired as exc:
        # pip 子进程超时，消除无限阻塞风险。
        raise RuntimeError(
            f"pip command timeout after {PIP_SUBPROCESS_TIMEOUT_SECONDS}s"
        ) from exc

    if result.returncode != 0:
        stderr_text = (result.stderr or "").strip()
        stdout_text = (result.stdout or "").strip()
        message = stderr_text or stdout_text or "pip command failed"
        raise RuntimeError(message)
    return (result.stdout or "").strip()


def _build_env_audit_record() -> Dict[str, Any]:
    """
    功能：生成环境与依赖审计记录。

    Build environment and dependency audit record with hardware/CUDA provenance.
    Caches pip freeze result within the same process to avoid repeated executions.

    Args:
        None.

    Returns:
        Audit record mapping.
    """
    global _pip_freeze_cache, _pip_version_cache
    
    generated_at_utc = time_utils.now_utc_iso_z()
    record = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "sys_platform": sys.platform,
        "executable": sys.executable,
        "pip_version": None,
        "pip_version_error": None,
        "pip_freeze": None,
        "pip_freeze_error": None,
        "generated_at_utc": generated_at_utc,
        # 硬件与 CUDA 溯源字段。
        "torch_version": "<absent>",
        "cuda_available": False,
        "torch_cuda_version": "<absent>",
        "cudnn_version": "<absent>",
        "gpu_name": "<absent>",
        "gpu_compute_capability": "<absent>",
        "torch_deterministic_flags": "<absent>"
    }

    try:
        # 检查缓存的 pip_version。
        if _pip_version_cache is None:
            pip_version = _run_pip_command([sys.executable, "-m", "pip", "--version"])
            # 缓存结果。
            _pip_version_cache = pip_version
            record["pip_version"] = pip_version
        else:
            record["pip_version"] = _pip_version_cache
    except Exception as exc:
        # pip 版本获取失败，必须显式记录。
        record["pip_version_error"] = f"{type(exc).__name__}: {exc}"

    try:
        # 检查缓存的 pip_freeze。
        if _pip_freeze_cache is None:
            freeze_output = _run_pip_command([sys.executable, "-m", "pip", "freeze"])
            pip_freeze = [line for line in freeze_output.splitlines() if line.strip()]
            # 缓存结果。
            _pip_freeze_cache = sorted(pip_freeze)
            record["pip_freeze"] = _pip_freeze_cache
        else:
            record["pip_freeze"] = _pip_freeze_cache
    except Exception as exc:
        # pip freeze 失败，必须显式记录。
        record["pip_freeze_error"] = f"{type(exc).__name__}: {exc}"

    # 采集 torch/CUDA/GPU 信息。
    try:
        import torch
        
        record["torch_version"] = torch.__version__
        record["cuda_available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            # CUDA 版本。
            cuda_version = torch.version.cuda
            record["torch_cuda_version"] = str(cuda_version) if cuda_version else "<absent>"
            
            # cuDNN 版本。
            try:
                cudnn_version = torch.backends.cudnn.version()
                record["cudnn_version"] = str(cudnn_version) if cudnn_version else "<absent>"
            except Exception:
                record["cudnn_version"] = "<absent>"
            
            # GPU 名称。
            try:
                gpu_name = torch.cuda.get_device_name(0)
                record["gpu_name"] = str(gpu_name) if gpu_name else "<absent>"
            except Exception:
                record["gpu_name"] = "<absent>"
            
            # GPU 计算能力。
            try:
                compute_cap = torch.cuda.get_device_capability(0)
                if compute_cap:
                    record["gpu_compute_capability"] = f"{compute_cap[0]}.{compute_cap[1]}"
                else:
                    record["gpu_compute_capability"] = "<absent>"
            except Exception:
                record["gpu_compute_capability"] = "<absent>"
            
            # torch deterministic flags。
            try:
                deterministic_flags = {
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic
                }
                record["torch_deterministic_flags"] = str(deterministic_flags)
            except Exception:
                record["torch_deterministic_flags"] = "<absent>"
        
    except ImportError:
        # torch 不可用，保持默认 <absent> 值。
        pass
    except Exception as exc:
        # torch 信息采集失败，记录但不中断。
        record["torch_version"] = f"<error: {type(exc).__name__}>"

    return record


def _extract_env_audit_error(record: Dict[str, Any]) -> Optional[str]:
    """
    功能：提取 env_audit 错误摘要。

    Extract env_audit error summary from audit record.

    Args:
        record: Environment audit record.

    Returns:
        Error summary string or None.

    Raises:
        TypeError: If record is invalid.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")

    errors = []
    pip_version_error = record.get("pip_version_error")
    pip_freeze_error = record.get("pip_freeze_error")
    if isinstance(pip_version_error, str) and pip_version_error:
        errors.append(f"pip_version_error={pip_version_error}")
    if isinstance(pip_freeze_error, str) and pip_freeze_error:
        errors.append(f"pip_freeze_error={pip_freeze_error}")
    if errors:
        return "; ".join(errors)
    return None


def _write_env_audit_record(
    run_root: Path,
    artifacts_dir: Path,
    audit_record: Dict[str, Any],
    canon_sha256: str
) -> Path:
    """
    功能：写入 env_audit 审计记录。

    Write env_audit record to artifacts/env_audits directory.

    Args:
        run_root: Run root directory.
        artifacts_dir: Artifacts output directory.
        audit_record: Environment audit record mapping.
        canon_sha256: Canonical SHA256 of audit_record.

    Returns:
        Path to written audit record.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(run_root, Path):
        # run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("run_root must be Path")
    if not isinstance(artifacts_dir, Path):
        # artifacts_dir 类型不符合预期，必须 fail-fast。
        raise TypeError("artifacts_dir must be Path")
    if not isinstance(audit_record, dict):
        # audit_record 类型不符合预期，必须 fail-fast。
        raise TypeError("audit_record must be dict")
    if not isinstance(canon_sha256, str) or not canon_sha256:
        # canon_sha256 类型不符合预期，必须 fail-fast。
        raise TypeError("canon_sha256 must be non-empty str")

    env_audits_dir = artifacts_dir / "env_audits"
    env_audits_dir.mkdir(parents=True, exist_ok=True)
    audit_filename = f"env_audit_{canon_sha256[:8]}.json"
    audit_path = env_audits_dir / audit_filename

    try:
        records_io.write_artifact_json(str(audit_path), audit_record)
    except FactSourcesNotInitializedError:
        records_io.write_artifact_json_unbound(run_root, artifacts_dir, str(audit_path), audit_record)

    return audit_path


def finalize_run(
    run_root: Path,
    records_dir: Path,
    artifacts_dir: Path,
    run_meta: Dict[str, Any]
) -> Path:
    """
    功能：生成并写入 run_closure.json。

    Close records bundle, build run_closure payload, and write it under artifacts.
    Also generates path_validation_audit record with canon_sha256 anchoring to run_closure.

    Args:
        run_root: Run root directory.
        records_dir: Records output directory.
        artifacts_dir: Artifacts output directory.
        run_meta: Run metadata for closure payload.

    Returns:
        Path to run_closure.json.

    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If inputs are structurally invalid.
    """
    if not isinstance(run_root, Path):
        # run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("run_root must be Path")
    if not isinstance(records_dir, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise TypeError("records_dir must be Path")
    if not isinstance(artifacts_dir, Path):
        # artifacts_dir 类型不符合预期，必须 fail-fast。
        raise TypeError("artifacts_dir must be Path")
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")

    run_meta = dict(run_meta)
    bound_fact_sources, bound_fact_sources_status = _resolve_bound_fact_sources(run_meta)
    if bound_fact_sources is not None:
        run_meta["bound_fact_sources"] = bound_fact_sources
    run_meta["bound_fact_sources_status"] = bound_fact_sources_status
    _backfill_run_meta_contract_from_fact_sources(run_meta, bound_fact_sources)
    _ensure_run_meta_minimal_contract(run_meta)
    _validate_run_meta_timestamps(run_meta)

    def _append_audit_warning(code: str, message: str, exception_name: Optional[str]) -> None:
        """
        功能：追加审计警告到 run_meta。

        Append an audit warning entry to run_meta.

        Args:
            code: Warning code string.
            message: Warning message summary.
            exception_name: Optional exception class name.

        Returns:
            None.
        """
        warnings = run_meta.get("audit_warnings")
        if warnings is None:
            warnings = []
        if not isinstance(warnings, list):
            # audit_warnings 类型不合法，必须 fail-fast。
            raise TypeError("audit_warnings must be list")
        entry = {
            "code": code,
            "message": message,
            "exception": exception_name
        }
        warnings.append(entry)
        run_meta["audit_warnings"] = warnings

    # 绑定事实源与推荐门禁报告。
    bound_fact_sources = run_meta.get("bound_fact_sources")
    if not isinstance(bound_fact_sources, dict):
        bound_fact_sources = None

    try:
        recommended_report = records_io.get_recommended_enforce_report()
        if recommended_report is None:
            # 没有推荐门禁报告时，写入结构化空报告。
            recommended_report = {
                "recommended_enforce_mode": "not_available",
                "items": [],
                "reason": "no_recommended_enforce_report_available"
            }
        run_meta["recommended_enforce_report"] = recommended_report
        items = recommended_report.get("items")
        item_count = len(items) if isinstance(items, list) else None
        warning_detail = {
            "recommended_enforce_warning": {
                "mode": recommended_report.get("recommended_enforce_mode"),
                "item_count": item_count
            }
        }
        run_meta["status_details"] = _merge_status_details(
            run_meta.get("status_details"),
            warning_detail
        )
    except FactSourcesNotInitializedError:
        # fact_sources 未初始化时，也写入结构化空报告。
        run_meta["recommended_enforce_report"] = {
            "recommended_enforce_mode": "not_available",
            "items": [],
            "reason": "fact_sources_not_initialized"
        }

    fact_sources_status = run_meta.get("bound_fact_sources_status")
    if fact_sources_status == "unbound":
        run_meta["fact_sources_binding"] = {
            "status": "unbound",
            "reason": "fact_sources_not_initialized"
        }
        _append_audit_warning(
            "FACT_SOURCES_UNBOUND_ARTIFACTS_ONLY",
            "fact sources not initialized; artifacts written in unbound mode",
            None
        )
    elif fact_sources_status == "bound":
        run_meta["fact_sources_binding"] = {
            "status": "bound",
            "reason": "fact_sources_initialized"
        }

    # 环境与依赖审计：生成记录并锚定到 run_meta。
    env_audit_record = None
    env_audit_canon_sha256 = None
    env_audit_status = "failed"
    env_audit_error = None

    try:
        env_audit_record = _build_env_audit_record()
        # 使用与 artifacts 写盘一致的审计标记，确保 digest 可复算。
        records_io._ensure_artifact_audit_marker(env_audit_record)
        env_audit_record_final = {
            **env_audit_record,
            "pip_freeze": list(env_audit_record.get("pip_freeze") or [])
        }
        env_audit_canon_sha256 = digests.canonical_sha256(env_audit_record_final)
        env_audit_error = _extract_env_audit_error(env_audit_record)
        env_audit_status = "ok" if env_audit_error is None else "failed"

        run_meta["env_audit_canon_sha256"] = env_audit_canon_sha256
        run_meta["env_audit_status"] = env_audit_status
        run_meta["env_audit_error"] = env_audit_error

        try:
            _write_env_audit_record(run_root, artifacts_dir, env_audit_record_final, env_audit_canon_sha256)
        except Exception as exc:
            # env_audit 写盘失败，必须显式记录。
            env_audit_status = "failed"
            env_audit_error = f"{type(exc).__name__}: {exc}"
            run_meta["env_audit_status"] = env_audit_status
            run_meta["env_audit_error"] = env_audit_error
            _append_audit_warning(
                "ENV_AUDIT_WRITE_FAILED",
                "env audit write failed",
                type(exc).__name__
            )

        if env_audit_status == "failed":
            _append_audit_warning(
                "ENV_AUDIT_FAILED",
                "env audit failed",
                None
            )
    except Exception as exc:
        # env_audit 构建失败，必须显式记录。
        env_audit_status = "failed"
        env_audit_error = f"{type(exc).__name__}: {exc}"
        run_meta["env_audit_status"] = env_audit_status
        run_meta["env_audit_error"] = env_audit_error
        _append_audit_warning(
            "ENV_AUDIT_FAILED",
            "env audit build failed",
            type(exc).__name__
        )

    # 依赖锁定文件审计：读取 requirements.txt 并计算 sha256。
    lock_file_path = run_root / "requirements.txt"
    env_lock_kind = None
    env_lock_sha256 = None
    env_lock_error = None

    try:
        if lock_file_path.exists() and lock_file_path.is_file():
            lock_bytes = lock_file_path.read_bytes()
            lock_sha256_value = hashlib.sha256(lock_bytes).hexdigest()
            env_lock_kind = "requirements.txt"
            env_lock_sha256 = lock_sha256_value
            run_meta["env_lock_kind"] = env_lock_kind
            run_meta["env_lock_sha256"] = env_lock_sha256
        else:
            # Lock 文件缺失，显式记录错误。
            env_lock_error = "lock_file_missing"
            run_meta["env_lock_error"] = env_lock_error
            _append_audit_warning(
                "ENV_LOCK_MISSING",
                "requirements.txt not found",
                None
            )
    except Exception as exc:
        # Lock 文件读取或计算失败，显式记录错误。
        env_lock_error = f"{type(exc).__name__}: {exc}"
        run_meta["env_lock_error"] = env_lock_error
        _append_audit_warning(
            "ENV_LOCK_FAILED",
            "env lock computation failed",
            type(exc).__name__
        )

    # 模型来源审计。
    model_provenance = None
    model_provenance_canon_sha256 = None
    model_provenance_error = None

    try:
        model_cfg_path = run_root / "configs" / "model_sd3.yaml"
        if model_cfg_path.exists() and model_cfg_path.is_file():
            from main.core import config_loader as cfg_loader
            model_cfg, model_prov = cfg_loader.load_yaml_with_provenance(model_cfg_path)
            
            # 构造 model_provenance 审计对象。
            model_provenance = {
                "model_id": model_cfg.get("model_id"),
                "source": model_cfg.get("source"),
                "revision": model_cfg.get("revision"),
                "weights_sha256": model_cfg.get("weights_sha256"),
                "config_file_sha256": model_prov.file_sha256,
                "config_canon_sha256": model_prov.canon_sha256
            }
            # 验证必填字段。
            if not model_provenance.get("model_id") or not model_provenance.get("source"):
                model_provenance_error = "model_id or source missing"
                _append_audit_warning(
                    "MODEL_PROVENANCE_INCOMPLETE",
                    "model provenance missing required fields",
                    None
                )
            
            model_provenance_canon_sha256 = digests.canonical_sha256(model_provenance)
            run_meta["model_provenance_canon_sha256"] = model_provenance_canon_sha256
            
            # 写入 artifacts/model_provenance/ 目录。
            model_prov_dir = artifacts_dir / "model_provenance"
            model_prov_dir.mkdir(parents=True, exist_ok=True)
            model_prov_path = model_prov_dir / "model_provenance.json"
            try:
                records_io.write_artifact_json(str(model_prov_path), model_provenance)
            except records_io.FactSourcesNotInitializedError:
                records_io.write_artifact_json_unbound(run_root, artifacts_dir, str(model_prov_path), model_provenance)
        else:
            # 模型配置文件缺失，生成 <absent> 版 model_provenance。
            model_provenance = {
                "model_id": "<absent>",
                "source": "<absent>",
                "revision": "<absent>",
                "weights_sha256": "<absent>",
                "config_file_sha256": "<absent>",
                "config_canon_sha256": "<absent>"
            }
            model_provenance_canon_sha256 = digests.canonical_sha256(model_provenance)
            run_meta["model_provenance_canon_sha256"] = model_provenance_canon_sha256
            model_provenance_error = "model_config_missing"
            run_meta["model_provenance_error"] = model_provenance_error

            # 写入 artifacts，使用无上下文兜底写盘。
            model_prov_dir = artifacts_dir / "model_provenance"
            model_prov_dir.mkdir(parents=True, exist_ok=True)
            model_prov_path = model_prov_dir / "model_provenance.json"
            records_io.write_artifact_json_unbound(run_root, artifacts_dir, str(model_prov_path), model_provenance)

            _append_audit_warning(
                "MODEL_PROVENANCE_MISSING",
                "model_sd3.yaml not found, <absent> provenance generated",
                None
            )
    except Exception as exc:
        # 模型来源审计失败，生成 <absent> 版 model_provenance。
        model_provenance = {
            "model_id": "<absent>",
            "source": "<absent>",
            "revision": "<absent>",
            "weights_sha256": "<absent>",
            "config_file_sha256": "<absent>",
            "config_canon_sha256": "<absent>"
        }
        model_provenance_canon_sha256 = digests.canonical_sha256(model_provenance)
        run_meta["model_provenance_canon_sha256"] = model_provenance_canon_sha256
        model_provenance_error = f"{type(exc).__name__}: {exc}"
        run_meta["model_provenance_error"] = model_provenance_error

        # 尝试写入 artifacts。
        try:
            model_prov_dir = artifacts_dir / "model_provenance"
            model_prov_dir.mkdir(parents=True, exist_ok=True)
            model_prov_path = model_prov_dir / "model_provenance.json"
            records_io.write_artifact_json_unbound(run_root, artifacts_dir, str(model_prov_path), model_provenance)
        except Exception:
            # 写盘失败不阻断 finalize_run。
            pass

        _append_audit_warning(
            "MODEL_PROVENANCE_FAILED",
            "model provenance audit failed, <absent> provenance generated",
            type(exc).__name__
        )

    # RNG 随机性控制审计。
    rng_audit = None
    rng_audit_canon_sha256 = None
    rng_audit_error = None

    try:
        # 从 cfg 或环境获取 RNG 配置。
        cfg_from_meta = run_meta.get("cfg")
        seed_value = None
        rng_backend = None
        torch_deterministic = None
        cudnn_benchmark = None

        if isinstance(cfg_from_meta, dict):
            seed_value = cfg_from_meta.get("seed", "<absent>")
            rng_backend = cfg_from_meta.get("rng_backend", "<absent>")
            torch_deterministic = cfg_from_meta.get("torch_deterministic", "<absent>")
            cudnn_benchmark = cfg_from_meta.get("cudnn_benchmark", "<absent>")
        else:
            # cfg 不可用，显式标记缺失。
            seed_value = "<absent>"
            rng_backend = "<absent>"
            torch_deterministic = "<absent>"
            cudnn_benchmark = "<absent>"

        # 构造 rng_audit 对象。
        rng_audit = {
            "seed_value": seed_value,
            "rng_backend": rng_backend,
            "torch_deterministic": torch_deterministic,
            "cudnn_benchmark": cudnn_benchmark
        }
        
        rng_audit_canon_sha256 = digests.canonical_sha256(rng_audit)
        run_meta["rng_audit_canon_sha256"] = rng_audit_canon_sha256
        
        # 写入 artifacts/rng_audits/ 目录。
        rng_audit_dir = artifacts_dir / "rng_audits"
        rng_audit_dir.mkdir(parents=True, exist_ok=True)
        rng_audit_path = rng_audit_dir / "rng_audit.json"
        try:
            records_io.write_artifact_json(str(rng_audit_path), rng_audit)
        except records_io.FactSourcesNotInitializedError:
            records_io.write_artifact_json_unbound(run_root, artifacts_dir, str(rng_audit_path), rng_audit)
            
    except Exception as exc:
        # RNG 审计失败。
        rng_audit_error = f"{type(exc).__name__}: {exc}"
        run_meta["rng_audit_error"] = rng_audit_error
        _append_audit_warning(
            "RNG_AUDIT_FAILED",
            "rng audit failed",
            type(exc).__name__
        )

    # 输入来源审计。
    input_provenance = None
    input_provenance_digest = None
    input_provenance_error = None

    try:
        from main.core import input_provenance as input_prov_module
        
        cfg_from_meta = run_meta.get("cfg")
        command = run_meta.get("command", "unknown")
        
        if isinstance(cfg_from_meta, dict):
            input_provenance = input_prov_module.build_input_provenance(cfg_from_meta, command)
        else:
            # cfg 不可用，构造全 <absent> 的输入来源。
            input_provenance = input_prov_module.build_input_provenance({}, command)
        
        input_provenance_digest = input_prov_module.compute_input_provenance_digest(input_provenance)
        run_meta["input_provenance_digest"] = input_provenance_digest
        
        # 写入 artifacts/input_provenance/ 目录。
        input_prov_dir = artifacts_dir / "input_provenance"
        input_prov_dir.mkdir(parents=True, exist_ok=True)
        input_prov_path = input_prov_dir / f"input_provenance_{input_provenance_digest[:8]}.json"
        try:
            records_io.write_artifact_json(str(input_prov_path), input_provenance)
        except records_io.FactSourcesNotInitializedError:
            records_io.write_artifact_json_unbound(run_root, artifacts_dir, str(input_prov_path), input_provenance)
            
    except Exception as exc:
        # 输入来源审计失败。
        input_provenance_error = f"{type(exc).__name__}: {exc}"
        run_meta["input_provenance_error"] = input_provenance_error
        _append_audit_warning(
            "INPUT_PROVENANCE_FAILED",
            "input provenance audit failed",
            type(exc).__name__
        )

    # 在 run_root 派生后立即生成路径审计记录。
    from main.policy import path_policy as path_policy_module

    path_audit_record = None
    path_audit_canon_sha256 = None

    try:
        if bound_fact_sources is not None:
            policy_path = run_meta.get("policy_path")
            semantics_version = bound_fact_sources.get("policy_path_semantics_version")
            whitelist_version = bound_fact_sources.get("whitelist_version")
            output_dir_input = run_meta.get("output_dir_input")
            logs_dir = run_root / "logs"
            if isinstance(policy_path, str) and policy_path and policy_path != "<absent>" and isinstance(semantics_version, str) and semantics_version and isinstance(whitelist_version, str) and whitelist_version:
                output_paths_relative = path_policy_module.build_output_paths_relative(
                    run_root,
                    records_dir,
                    artifacts_dir,
                    logs_dir
                )
                path_audit_record = path_policy_module.build_comprehensive_path_validation_audit(
                    run_root=run_root,
                    policy_path=policy_path,
                    policy_path_semantics_version=semantics_version,
                    runtime_whitelist_version=whitelist_version,
                    original_input=output_dir_input,
                    output_paths_relative=output_paths_relative,
                    validation_status="ok",
                    failure_reason=None
                )
                # 手动注入 _artifact_audit 标记，确保 digest 可复算。
                if "_artifact_audit" not in path_audit_record:
                    path_audit_record["_artifact_audit"] = {
                        "schema_version": "v1.0",
                        "writer": "records_io"
                    }
                path_audit_canon_sha256 = path_policy_module.compute_path_audit_canon_sha256(path_audit_record)
                run_meta["path_audit_canon_sha256"] = path_audit_canon_sha256
                run_meta["path_audit_status"] = "ok"
            else:
                run_meta["path_audit_status"] = "failed"
                run_meta["path_audit_error"] = "ValueError: policy_path or bound versions missing"
                _append_audit_warning(
                    "PATH_AUDIT_BUILD_FAILED",
                    "path audit build failed due to missing bound fields",
                    "ValueError"
                )
    except Exception as exc:
        error_summary = f"{type(exc).__name__}: {exc}"
        run_meta["path_audit_status"] = "failed"
        run_meta["path_audit_error"] = error_summary
        _append_audit_warning(
            "PATH_AUDIT_BUILD_FAILED",
            "path audit build raised exception",
            type(exc).__name__
        )

    # 写入路径审计记录到 artifacts/path_audits/ 目录。
    if path_audit_record is not None:
        try:
            path_audits_dir = artifacts_dir / "path_audits"
            path_audits_dir.mkdir(parents=True, exist_ok=True)

            # 使用 canonical_sha256 生成审计文件名。
            if path_audit_canon_sha256:
                audit_filename = f"path_audit_{path_audit_canon_sha256[:8]}.json"
            else:
                from datetime import datetime as dt
                timestamp_str = dt.now(timezone.utc).isoformat().replace(":", "-").replace(".", "-")
                audit_filename = f"path_audit_{timestamp_str}.json"

            audit_path = path_audits_dir / audit_filename

            # 优先使用受控写盘；仅在事实源未初始化或锚点重叠检查失败时回退到 unbound 写法。
            try:
                records_io.write_artifact_json(
                    str(audit_path),
                    path_audit_record
                )
            except (FactSourcesNotInitializedError, RecordsWritePolicyError):
                # 事实源未初始化或锚点重叠检查失败时回退到 unbound 写法。
                records_io.write_artifact_json_unbound(
                    run_root,
                    artifacts_dir,
                    str(audit_path),
                    path_audit_record
                )
                _append_audit_warning(
                    "PATH_AUDIT_WRITE_UNBOUND",
                    "path audit written without fact sources binding or semantic guard rejection",
                    "FactSourcesNotInitializedError or RecordsWritePolicyError"
                )
        except Exception as exc:
            error_summary = f"{type(exc).__name__}: {exc}"
            run_meta["path_audit_status"] = "failed"
            run_meta["path_audit_error"] = error_summary
            _append_audit_warning(
                "PATH_AUDIT_WRITE_FAILED",
                "path audit write failed",
                type(exc).__name__
            )

    def _extract_upstream_reason() -> Optional[str]:
        """从 run_meta 中提取 upstream_status_reason（as string value）。"""
        status_reason = run_meta.get("status_reason")
        if status_reason is None:
            return None
        if isinstance(status_reason, RunFailureReason):
            return status_reason.value
        return str(status_reason) if status_reason else None

    payload = None
    if _has_record_files(records_dir):
        try:
            manifest_path = close_records_bundle(
                records_dir,
                RECORDS_MANIFEST_NAME,
                manifest_dir=artifacts_dir
            )
            manifest = records_io.read_json(str(manifest_path))
            if not isinstance(manifest, dict):
                # manifest 类型不符合预期，必须 fail-fast。
                raise ValueError("records manifest must be dict")

            manifest_rel_path = _derive_manifest_rel_path(run_root, manifest_path)
            run_meta["manifest_rel_path"] = manifest_rel_path
            _cleanup_run_meta_missing_fields(run_meta)
            
            # 在构造 closure payload 前，从 manifest anchors 回填 run_meta。
            try:
                _merge_manifest_anchors_into_run_meta(run_meta, manifest)
                _cleanup_run_meta_missing_fields(run_meta)
                # 成功：records 存在、bundle 成功、anchor merge 成功。
                run_meta["closure_stage"] = _build_closure_stage(
                    records_present=True,
                    bundle_attempted=True,
                    bundle_succeeded=True,
                    failure_stage=None,
                    upstream_status_reason=None
                )
                payload = build_run_closure_payload(run_meta, manifest)
            except ValueError as exc:
                # manifest anchors 冲突错误，降级为失败闭包。
                _apply_manifest_anchor_conflict_failure(run_meta, exc)
                # failure_stage="anchor_merge" 表示 bundle 成功但 merge 失败。
                run_meta["closure_stage"] = _build_closure_stage(
                    records_present=True,
                    bundle_attempted=True,
                    bundle_succeeded=True,
                    failure_stage="anchor_merge",
                    upstream_status_reason=_extract_upstream_reason()
                )
                payload = build_run_closure_payload(run_meta, None)
        except RecordBundleError as exc:
            _apply_records_inconsistent_failure(run_meta, exc)
            # failure_stage="bundle" 表示 bundle 本身失败。
            run_meta["closure_stage"] = _build_closure_stage(
                records_present=True,
                bundle_attempted=True,
                bundle_succeeded=False,
                failure_stage="bundle",
                upstream_status_reason=_extract_upstream_reason()
            )
            payload = build_run_closure_payload(run_meta, None)
        except Exception as exc:
            # 审计增强：任何异常都必须产出 run_closure 以保证闭包可追溯。
            _apply_run_closure_exception(run_meta, exc)
            # failure_stage="unknown" 表示未知异常。
            run_meta["closure_stage"] = _build_closure_stage(
                records_present=True,
                bundle_attempted=True,
                bundle_succeeded=False,
                failure_stage="unknown",
                upstream_status_reason=_extract_upstream_reason()
            )
            payload = build_run_closure_payload(run_meta, None)
    else:
        _apply_records_absent_defaults(run_meta)
        # records 不存在。
        run_meta["closure_stage"] = _build_closure_stage(
            records_present=False,
            bundle_attempted=False,
            bundle_succeeded=False,
            failure_stage=None,
            upstream_status_reason=_extract_upstream_reason()
        )
        payload = build_run_closure_payload(run_meta, None)

    # 检查并锚定 signoff_report.json。
    signoff_report_path = artifacts_dir / "signoff" / "signoff_report.json"
    signoff_report_canon_sha256 = None
    signoff_report_status = "absent"
    
    if signoff_report_path.exists():
        try:
            with open(signoff_report_path, 'r', encoding='utf-8') as f:
                signoff_report = json.load(f)
            if isinstance(signoff_report, dict):
                signoff_report_canon_sha256 = digests.canonical_sha256(signoff_report)
                signoff_report_status = "present"
                run_meta["signoff_report_canon_sha256"] = signoff_report_canon_sha256
                run_meta["signoff_report_path"] = "artifacts/signoff/signoff_report.json"
                run_meta["signoff_report_status"] = signoff_report_status
        except Exception as exc:
            # signoff_report 读取或计算失败，记录但不失败。
            run_meta["signoff_report_status"] = "error"
            run_meta["signoff_report_error"] = f"{type(exc).__name__}: {exc}"
    else:
        run_meta["signoff_report_status"] = "absent"

    run_closure_path = artifacts_dir / RUN_CLOSURE_NAME
    path_policy.validate_output_target(run_closure_path, "artifact", run_root)
    validate_run_closure(payload)
    
    # 尝试使用标准 write_artifact_json。
    try:
        records_io.write_artifact_json(str(run_closure_path), payload)
    except records_io.FactSourcesNotInitializedError:
        # 如果事实源未初始化，降级使用无上下文兜底写盘。
        records_io.write_artifact_json_unbound(run_root, artifacts_dir, str(run_closure_path), payload)
    # 其他异常继续向上抛，不隐藏真实 bug。
    
    
    return run_closure_path


def build_run_closure_payload(
    run_meta: Dict[str, Any],
    manifest: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    功能：构造 run_closure payload。

    Build run closure payload with fixed schema and anchors.

    Args:
        run_meta: Run metadata mapping.
        manifest: Records manifest mapping.

    Returns:
        Run closure payload mapping.

    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If required fields are missing.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")
    if manifest is not None and not isinstance(manifest, dict):
        # manifest 类型不符合预期，必须 fail-fast。
        raise TypeError("manifest must be dict or None")

    required_meta_fields = [
        "run_id",
        "command",
        "created_at_utc",
        "cfg_digest",
        "policy_path",
        "impl_id",
        "impl_version",
        "manifest_rel_path"
    ]
    status_ok = run_meta.get("status_ok", True)
    status_reason = run_meta.get("status_reason")
    status_details = run_meta.get("status_details")
    if isinstance(status_reason, str) and not status_reason:
        status_reason = None
    if not isinstance(status_ok, bool):
        # status_ok 类型不符合预期，必须 fail-fast。
        raise TypeError("status_ok must be bool")
    if status_ok:
        missing_or_absent_fields = _validate_run_meta_for_ok(run_meta)
        if missing_or_absent_fields:
            _apply_run_meta_invalid_for_ok(run_meta, missing_or_absent_fields)
            status_ok = run_meta.get("status_ok", False)
            status_reason = run_meta.get("status_reason")
            status_details = run_meta.get("status_details")
    if status_reason is None:
        if status_ok:
            status_reason = RunFailureReason.OK
        else:
            if manifest is None:
                status_reason = RunFailureReason.RECORDS_ABSENT
            else:
                # status_reason 缺失且 records 存在，必须 fail-fast。
                raise ValueError("status_reason must be provided when records exist")
    if not isinstance(status_reason, RunFailureReason):
        # status_reason 类型不符合预期，必须 fail-fast。
        raise TypeError("status_reason must be RunFailureReason")
    if status_ok and status_reason != RunFailureReason.OK:
        # status_ok=True 时 reason 必须为 ok。
        raise ValueError("status_reason must be RunFailureReason.OK when status_ok is True")
    if not status_ok and status_reason == RunFailureReason.OK:
        # status_ok=False 时 reason 不能为 ok。
        raise ValueError("status_reason must not be RunFailureReason.OK when status_ok is False")

    if status_ok:
        missing_meta = [key for key in required_meta_fields if key not in run_meta]
        if missing_meta:
            # run_meta 字段缺失，必须 fail-fast。
            raise ValueError(f"run_meta missing fields: {missing_meta}")
        if manifest is None:
            # manifest 缺失，必须 fail-fast。
            raise ValueError("manifest must be provided when status_ok is True")

        anchors = manifest.get("anchors")
        if not isinstance(anchors, dict):
            # anchors 类型不符合预期，必须 fail-fast。
            raise ValueError("manifest.anchors must be dict")

        required_anchor_fields = [
            "contract_bound_digest",
            "whitelist_bound_digest",
            "policy_path_semantics_bound_digest"
        ]
        missing_anchor = [key for key in required_anchor_fields if key not in anchors]
        if missing_anchor:
            # anchors 字段缺失，必须 fail-fast。
            raise ValueError(f"manifest.anchors missing fields: {missing_anchor}")

        bundle_canon_sha256 = manifest.get("bundle_canon_sha256")
        if not isinstance(bundle_canon_sha256, str) or not bundle_canon_sha256:
            # bundle_canon_sha256 为空，必须 fail-fast。
            raise ValueError("manifest.bundle_canon_sha256 must be non-empty str")
    else:
        bundle_canon_sha256 = None
        anchors = None
        _ = manifest

    impl_identity, impl_identity_digest = _normalize_impl_identity_meta(run_meta)

    records_bundle_payload = None
    facts_anchor_payload = None
    if status_ok:
        records_bundle_payload = {
            "manifest_rel_path": run_meta["manifest_rel_path"],
            "bundle_canon_sha256": bundle_canon_sha256
        }
        facts_anchor_payload = {
            "contract_bound_digest": anchors["contract_bound_digest"],
            "whitelist_bound_digest": anchors["whitelist_bound_digest"],
            "policy_path_semantics_bound_digest": anchors["policy_path_semantics_bound_digest"]
        }

    # 从 run_meta 获取 path_audit 相关信息。
    path_audit_canon_sha256 = run_meta.get("path_audit_canon_sha256")
    
    # 从 run_meta 获取路径策略信息。
    path_policy_audit = run_meta.get("path_policy")
    
    # 规范化 status_details：确保为 dict 或 None，不允许裸字符串。
    normalized_status_details = status_details
    if isinstance(status_details, str):
        # 裸字符串降级为结构化对象。
        normalized_status_details = {
            "exc_type": "unknown",
            "message": status_details,
            "stack_fingerprint": "<absent>"
        }
    elif status_details is None:
        normalized_status_details = None
    elif not isinstance(status_details, dict):
        # 非法类型强制规范化为 dict。
        normalized_status_details = {
            "exc_type": "unknown",
            "message": str(status_details),
            "stack_fingerprint": "<absent>"
        }
    
    # 校验冻结锚点字段（失败路径也必须校验）。
    freeze_anchor_fields = [
        "contract_version", "contract_digest", "contract_file_sha256",
        "contract_canon_sha256", "contract_bound_digest",
        "whitelist_version", "whitelist_digest", "whitelist_file_sha256",
        "whitelist_canon_sha256", "whitelist_bound_digest",
        "policy_path_semantics_version", "policy_path_semantics_digest",
        "policy_path_semantics_file_sha256", "policy_path_semantics_canon_sha256",
        "policy_path_semantics_bound_digest"
    ]
    missing_freeze_anchors = []
    for field_name in freeze_anchor_fields:
        value = run_meta.get(field_name)
        if value is None or value == "<absent>":
            missing_freeze_anchors.append(field_name)
    
    # 若冻结锚点缺失且当前不是失败状态，记录为次级失败证据。
    if missing_freeze_anchors and not status_ok:
        # 不覆盖主因，作为次级证据记录到 audit_warnings。
        warnings = run_meta.get("audit_warnings")
        if warnings is None:
            warnings = []
        if isinstance(warnings, list):
            warnings.append({
                "code": "FREEZE_ANCHORS_MISSING_IN_FAILURE_PATH",
                "message": f"Freeze anchors missing in failure path: {', '.join(missing_freeze_anchors)}",
                "exception_name": None
            })
            run_meta["audit_warnings"] = warnings
    
    payload = {
        "schema_version": RUN_CLOSURE_SCHEMA_VERSION,
        "run_id": run_meta["run_id"],
        "command": run_meta["command"],
        "created_at_utc": run_meta["created_at_utc"],
        "started_at": run_meta.get("started_at", "<absent>"),
        "ended_at": run_meta.get("ended_at", "<absent>"),
        "cfg_digest": run_meta["cfg_digest"],
        "contract_version": run_meta.get("contract_version", "<absent>"),
        "contract_digest": run_meta.get("contract_digest", "<absent>"),
        "contract_file_sha256": run_meta.get("contract_file_sha256", "<absent>"),
        "contract_canon_sha256": run_meta.get("contract_canon_sha256", "<absent>"),
        "contract_bound_digest": run_meta.get("contract_bound_digest", "<absent>"),
        "whitelist_version": run_meta.get("whitelist_version", "<absent>"),
        "whitelist_digest": run_meta.get("whitelist_digest", "<absent>"),
        "whitelist_file_sha256": run_meta.get("whitelist_file_sha256", "<absent>"),
        "whitelist_canon_sha256": run_meta.get("whitelist_canon_sha256", "<absent>"),
        "whitelist_bound_digest": run_meta.get("whitelist_bound_digest", "<absent>"),
        "policy_path_semantics_version": run_meta.get("policy_path_semantics_version", "<absent>"),
        "policy_path_semantics_digest": run_meta.get("policy_path_semantics_digest", "<absent>"),
        "policy_path_semantics_file_sha256": run_meta.get("policy_path_semantics_file_sha256", "<absent>"),
        "policy_path_semantics_canon_sha256": run_meta.get("policy_path_semantics_canon_sha256", "<absent>"),
        "policy_path_semantics_bound_digest": run_meta.get("policy_path_semantics_bound_digest", "<absent>"),
        "cfg_audit_canon_sha256": run_meta.get("cfg_audit_canon_sha256", "<absent>"),
        "cfg_pruned_for_digest_canon_sha256": run_meta.get("cfg_pruned_for_digest_canon_sha256", "<absent>"),
        "policy_path": run_meta["policy_path"],
        "impl_id": run_meta["impl_id"],
        "impl_version": run_meta["impl_version"],
        "impl_identity": impl_identity,
        "impl_identity_digest": impl_identity_digest,
        "pipeline_provenance_canon_sha256": run_meta.get("pipeline_provenance_canon_sha256", "<absent>"),
        "pipeline_status": run_meta.get("pipeline_status", "unbuilt"),
        "pipeline_error": run_meta.get("pipeline_error", "<absent>"),
        "pipeline_runtime_meta": run_meta.get("pipeline_runtime_meta"),
        "inference_status": run_meta.get("inference_status", "<absent>"),
        "inference_error": run_meta.get("inference_error", "<absent>"),
        "inference_runtime_meta": run_meta.get("inference_runtime_meta"),
        "infer_trace": run_meta.get("infer_trace"),
        "infer_trace_canon_sha256": run_meta.get("infer_trace_canon_sha256", "<absent>"),
        "env_fingerprint_canon_sha256": run_meta.get("env_fingerprint_canon_sha256", "<absent>"),
        "diffusers_version": run_meta.get("diffusers_version", "<absent>"),
        "transformers_version": run_meta.get("transformers_version", "<absent>"),
        "safetensors_version": run_meta.get("safetensors_version", "<absent>"),
        "model_provenance_canon_sha256": run_meta.get("model_provenance_canon_sha256", "<absent>"),
        "facts_anchor": facts_anchor_payload,
        "records_bundle": records_bundle_payload,
        "path_audit_canon_sha256": path_audit_canon_sha256,
        "path_audit_status": run_meta.get("path_audit_status", "<absent>"),
        "path_audit_error": run_meta.get("path_audit_error", "<absent>"),
        "path_policy": path_policy_audit,
        "run_root_reuse_allowed": run_meta.get("run_root_reuse_allowed", False),
        "run_root_reuse_reason": run_meta.get("run_root_reuse_reason", "<absent>"),
        "env_audit_canon_sha256": run_meta.get("env_audit_canon_sha256", "<absent>"),
        "env_audit_status": run_meta.get("env_audit_status", "<absent>"),
        "env_audit_error": run_meta.get("env_audit_error", "<absent>"),
        "env_lock_kind": run_meta.get("env_lock_kind", "<absent>"),
        "env_lock_sha256": run_meta.get("env_lock_sha256", "<absent>"),
        "env_lock_error": run_meta.get("env_lock_error", "<absent>"),
        "thresholds_rule_id": run_meta.get("thresholds_rule_id", "<absent>"),
        "thresholds_rule_version": run_meta.get("thresholds_rule_version", "<absent>"),
        "target_fpr": run_meta.get("target_fpr", "<absent>"),
        "thresholds_digest": run_meta.get("thresholds_digest", "<absent>"),
        "threshold_metadata_digest": run_meta.get("threshold_metadata_digest", "<absent>"),
        "model_provenance_error": run_meta.get("model_provenance_error", "<absent>"),
        "rng_audit_canon_sha256": run_meta.get("rng_audit_canon_sha256", "<absent>"),
        "rng_audit_error": run_meta.get("rng_audit_error", "<absent>"),
        "input_provenance_digest": run_meta.get("input_provenance_digest", "<absent>"),
        "input_provenance_error": run_meta.get("input_provenance_error", "<absent>"),
        "signoff_report_canon_sha256": run_meta.get("signoff_report_canon_sha256", "<absent>"),
        "signoff_report_path": run_meta.get("signoff_report_path", "<absent>"),
        "signoff_report_status": run_meta.get("signoff_report_status", "<absent>"),
        "signoff_report_error": run_meta.get("signoff_report_error", "<absent>"),
        "bound_fact_sources": run_meta.get("bound_fact_sources"),
        "bound_fact_sources_status": run_meta.get("bound_fact_sources_status", "<absent>"),
        "fact_sources_binding": run_meta.get("fact_sources_binding"),
        "recommended_enforce_report": run_meta.get("recommended_enforce_report"),
        "audit_warnings": run_meta.get("audit_warnings"),
        "nondeterminism_notes": run_meta.get("nondeterminism_notes", "<absent>"),
        "determinism_controls": run_meta.get("determinism_controls"),
        "status": {
            "ok": status_ok,
            "reason": status_reason.value,
            "details": normalized_status_details
        },
        "closure_stage": run_meta.get("closure_stage")
    }
    
    # run_closure 类型分层缺失规范化。
    _normalize_run_closure_absence_values(payload)
    # run_closure 类型分层缺失规范化。
    _normalize_run_closure_missing(payload)
    
    return payload


def _normalize_impl_identity_meta(run_meta: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    功能：规范化 impl_identity 元信息。

    Normalize impl_identity metadata for run_closure payload.

    Args:
        run_meta: Run metadata mapping.

    Returns:
        Tuple of (impl_identity, impl_identity_digest).

    Raises:
        TypeError: If impl_identity metadata is invalid.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")

    impl_identity = run_meta.get("impl_identity")
    if impl_identity is None:
        return None, None
    if not isinstance(impl_identity, dict):
        # impl_identity 类型不合法，必须 fail-fast。
        raise TypeError("impl_identity must be dict")

    impl_identity_digest = run_meta.get("impl_identity_digest")
    if impl_identity_digest is None:
        impl_identity_digest = digests.canonical_sha256(impl_identity)
    if not isinstance(impl_identity_digest, str) or not impl_identity_digest:
        # impl_identity_digest 类型不合法，必须 fail-fast。
        raise TypeError("impl_identity_digest must be non-empty str")

    return dict(impl_identity), impl_identity_digest


def _derive_manifest_rel_path(run_root: Path, manifest_path: Path) -> str:
    """
    功能：计算 manifest 相对路径。

    Compute relative path for manifest under run_root.

    Args:
        run_root: Run root directory.
        manifest_path: Manifest file path.

    Returns:
        Relative path string.
    """
    if not isinstance(run_root, Path):
        # run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("run_root must be Path")
    if not isinstance(manifest_path, Path):
        # manifest_path 类型不符合预期，必须 fail-fast。
        raise TypeError("manifest_path must be Path")

    try:
        return manifest_path.resolve().relative_to(run_root.resolve()).as_posix()
    except ValueError:
        return manifest_path.name


def _has_record_files(records_dir: Path) -> bool:
    """
    功能：检查 records_dir 是否包含记录文件。

    Check if records_dir has any records files (*.json or *.jsonl) excluding
    manifest and temporary files.

    Args:
        records_dir: Records directory path.

    Returns:
        True if any record files exist.
    """
    if not isinstance(records_dir, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise TypeError("records_dir must be Path")
    if not records_dir.exists() or not records_dir.is_dir():
        return False

    for path in records_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name == RECORDS_MANIFEST_NAME:
            continue
        if name.startswith(".tmp-") or name.endswith(".writing"):
            continue
        if path.suffix in {".json", ".jsonl"}:
            return True
    return False


def _apply_records_absent_defaults(run_meta: Dict[str, Any]) -> None:
    """
    功能：无 records 时设置默认失败状态，并冻结 reason 口径。

    Apply default failure status when no records are present, with frozen semantics.
    When records_absent occurs:
    - If status_reason is None/empty/"OK", override to RECORDS_ABSENT.
    - If status_reason is already a non-OK failure reason, preserve it but add records_absent=True to details.

    Args:
        run_meta: Run metadata mapping to mutate.

    Returns:
        None.

    Raises:
        TypeError: If run_meta is invalid.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")

    run_meta["status_ok"] = False
    status_reason = run_meta.get("status_reason")
    
    # 根据当前 reason 状态判断覆盖还是补充 details。
    if status_reason is None:
        # reason 为 None，覆盖为 RECORDS_ABSENT。
        run_meta["status_reason"] = RunFailureReason.RECORDS_ABSENT
    elif isinstance(status_reason, str) and not status_reason:
        # reason 为空字符串，覆盖为 RECORDS_ABSENT。
        run_meta["status_reason"] = RunFailureReason.RECORDS_ABSENT
    elif status_reason == RunFailureReason.OK:
        # reason 为 OK，覆盖为 RECORDS_ABSENT。
        run_meta["status_reason"] = RunFailureReason.RECORDS_ABSENT
    else:
        # reason 已是非 OK，保持不变。
        # 但必须在 details 中补充 records_absent 标记。
        existing_details = run_meta.get("status_details")
        new_details = {"records_absent": True}
        merged_details = _merge_status_details(existing_details, new_details)
        run_meta["status_details"] = merged_details


def _apply_records_inconsistent_failure(run_meta: Dict[str, Any], exc: Exception) -> None:
    """
    功能：闭包失败时设置 records_inconsistent 语义。

    Apply records_inconsistent status and attach bundle_error_detail.
    This also adds audit-only upstream failure reason details when present.

    Args:
        run_meta: Run metadata mapping to mutate.
        exc: RecordBundleError instance.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")

    upstream_failure_reason = _extract_upstream_failure_reason(run_meta)
    bundle_error_detail = _extract_bundle_error_detail(exc)
    new_details = {"bundle_error_detail": bundle_error_detail}
    if upstream_failure_reason is not None:
        # 审计增强：保留被 records_inconsistent 覆盖的上层失败原因。
        new_details["upstream_failure_reason"] = upstream_failure_reason
    status_details = _merge_status_details(run_meta.get("status_details"), new_details)
    run_meta["status_ok"] = False
    run_meta["status_reason"] = RunFailureReason.RECORDS_INCONSISTENT
    run_meta["status_details"] = status_details


def _apply_manifest_anchor_conflict_failure(run_meta: Dict[str, Any], exc: Exception) -> None:
    """
    功能：处理 manifest anchors 冲突导致闭包失败。

    Apply config_invalid status when manifest anchors conflict with run_meta.

    Args:
        run_meta: Run metadata mapping to mutate.
        exc: ValueError instance from anchor merge conflict.

    Returns:
        None.

    Raises:
        TypeError: If run_meta is invalid.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")

    conflict_detail = {
        "kind": "manifest_anchor_conflict",
        "error_message": str(exc)
    }
    status_details = _merge_status_details(
        run_meta.get("status_details"),
        conflict_detail
    )
    run_meta["status_ok"] = False
    run_meta["status_reason"] = RunFailureReason.CONFIG_INVALID
    run_meta["status_details"] = status_details


def _merge_status_details(
    existing_details: Any,
    new_details: Dict[str, Any]
) -> Dict[str, Any]:
    """功能：合并 status_details 字段。

    Merge existing status_details with new detail payload.

    Args:
        existing_details: Existing status_details value.
        new_details: New detail mapping to merge.

    Returns:
        Merged status_details mapping.

    Raises:
        TypeError: If new_details is invalid.
    """
    if not isinstance(new_details, dict):
        # new_details 类型不符合预期，必须 fail-fast。
        raise TypeError("new_details must be dict")

    if existing_details is None:
        return dict(new_details)
    if isinstance(existing_details, dict):
        merged = dict(existing_details)
        merged.update(new_details)
        return merged
    return {
        "prior_status_details": existing_details,
        **new_details
    }


def _extract_bundle_error_detail(exc: Exception) -> Dict[str, Any]:
    """
    功能：提取 records bundle 错误细节。

    Extract locatable bundle error details from RecordBundleError.

    Args:
        exc: Exception instance.

    Returns:
        Detail mapping with field_name and files.

    Raises:
        TypeError: If exc is invalid.
    """
    if not isinstance(exc, Exception):
        # exc 类型不符合预期，必须 fail-fast。
        raise TypeError("exc must be Exception")

    message = str(exc)
    field_name = getattr(exc, "field_name", None)
    files = getattr(exc, "files", None)
    if isinstance(field_name, str) and field_name and isinstance(files, list):
        return {
            "field_name": field_name,
            "files": [str(item) for item in files],
            "error_message": message
        }
    # 兜底路径：仅保留 error_message，避免绑定 message 模板语义。
    field_name = "unknown"
    files = []
    return {
        "field_name": field_name,
        "files": files,
        "error_message": message
    }


def _extract_upstream_failure_reason(run_meta: Dict[str, Any]) -> Optional[str]:
    """
    功能：提取被覆盖的上层失败原因。

    Extract upstream failure reason for audit details.

    Args:
        run_meta: Run metadata mapping.

    Returns:
        Upstream failure reason string, or None if not applicable.

    Raises:
        TypeError: If run_meta is invalid.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")

    status_reason = run_meta.get("status_reason")
    if isinstance(status_reason, RunFailureReason):
        if status_reason in {RunFailureReason.OK, RunFailureReason.RECORDS_INCONSISTENT}:
            return None
        return status_reason.value
    if isinstance(status_reason, str):
        normalized = status_reason.strip()
        if not normalized:
            return None
        if normalized in {RunFailureReason.OK.value, RunFailureReason.RECORDS_INCONSISTENT.value}:
            return None
        return normalized
    return None


def _apply_run_closure_exception(run_meta: Dict[str, Any], exc: Exception) -> None:
    """
    功能：处理 run_closure 兜底异常并写入审计细节。

    Apply runtime_error status and attach closure_exception details.

    Args:
        run_meta: Run metadata mapping to mutate.
        exc: Exception instance.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")
    if not isinstance(exc, Exception):
        # exc 类型不符合预期，必须 fail-fast。
        raise TypeError("exc must be Exception")

    closure_exception = {
        "type": type(exc).__name__,
        "message": str(exc)
    }
    status_details = _merge_status_details(
        run_meta.get("status_details"),
        {"closure_exception": closure_exception}
    )
    run_meta["status_ok"] = False
    # 兜底异常固定为 runtime_error，避免扩展枚举。
    run_meta["status_reason"] = RunFailureReason.RUNTIME_ERROR
    run_meta["status_details"] = status_details


def _validate_run_meta_for_ok(run_meta: Dict[str, Any]) -> list[str]:
    """
    功能：校验 ok 闭包所需 run_meta 关键字段。

    Validate critical run_meta fields required for status.ok=True closures.

    Args:
        run_meta: Run metadata mapping.

    Returns:
        List of missing or absent field names.

    Raises:
        TypeError: If run_meta is invalid.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")

    required_fields = [
        "run_id",
        "cfg_digest",
        "policy_path",
        "contract_bound_digest",
        "whitelist_bound_digest",
        "policy_path_semantics_bound_digest",
        "impl_id",
        "impl_version"
    ]
    if run_meta.get("impl_identity") is not None or run_meta.get("impl_identity_digest") is not None:
        required_fields.append("impl_identity_digest")

    missing_or_absent_fields = []
    for field_name in required_fields:
        value = run_meta.get(field_name)
        if value is None:
            missing_or_absent_fields.append(field_name)
            continue
        if not isinstance(value, str):
            missing_or_absent_fields.append(field_name)
            continue
        if not value or value == "<absent>":
            missing_or_absent_fields.append(field_name)

    if not missing_or_absent_fields:
        return []
    return sorted(set(missing_or_absent_fields))


def _apply_run_meta_invalid_for_ok(
    run_meta: Dict[str, Any],
    missing_or_absent_fields: list[str]
) -> None:
    """
    功能：将 ok 闭包降级为失败并记录结构化原因。

    Downgrade ok closure to failure and attach structured status details.

    Args:
        run_meta: Run metadata mapping to mutate.
        missing_or_absent_fields: Missing or absent field names.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")
    if not isinstance(missing_or_absent_fields, list):
        # missing_or_absent_fields 类型不符合预期，必须 fail-fast。
        raise TypeError("missing_or_absent_fields must be list")
    for field_name in missing_or_absent_fields:
        if not isinstance(field_name, str) or not field_name:
            # missing_or_absent_fields 成员不合法，必须 fail-fast。
            raise TypeError("missing_or_absent_fields must contain non-empty str values")

    status_details = _merge_status_details(
        run_meta.get("status_details"),
        {
            "kind": "run_meta_invalid_for_ok",
            "missing_or_absent_fields": sorted(set(missing_or_absent_fields))
        }
    )
    run_meta["status_ok"] = False
    run_meta["status_reason"] = RunFailureReason.CONFIG_INVALID
    run_meta["status_details"] = status_details


def _merge_manifest_anchors_into_run_meta(
    run_meta: Dict[str, Any],
    manifest: Dict[str, Any]
) -> None:
    """
    功能：从 manifest 回填 anchors 到 run_meta。

    Merge manifest anchors into run_meta for ok closure; fail-fast on conflicts.

    Args:
        run_meta: Run metadata mapping to mutate.
        manifest: Records manifest mapping.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If anchor conflicts detected (manifest vs run_meta mismatch).
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")
    if not isinstance(manifest, dict):
        # manifest 类型不符合预期，必须 fail-fast。
        raise TypeError("manifest must be dict")

    anchors = manifest.get("anchors")
    if not isinstance(anchors, dict):
        # anchors 类型不符合预期，必须 fail-fast。
        raise ValueError("manifest.anchors must be dict")

    # 要回填的关键字段列表。
    required_anchor_fields = [
        "contract_bound_digest",
        "whitelist_bound_digest",
        "policy_path_semantics_bound_digest"
    ]

    # 检查冲突：若 run_meta 中字段存在且与 manifest 不一致，则 fail-fast。
    conflicts = []
    for field_name in required_anchor_fields:
        manifest_value = anchors.get(field_name)
        existing_value = run_meta.get(field_name)

        # 如果 run_meta 中已有值且不等于 manifest 值，记录冲突。
        if existing_value is not None and existing_value != "<absent>":
            if manifest_value != existing_value:
                conflicts.append({
                    "field_name": field_name,
                    "run_meta_value": existing_value,
                    "manifest_value": manifest_value
                })

    if conflicts:
        # 冲突必须 fail-fast，附加详情便于审计。
        raise ValueError(
            f"Manifest anchor conflicts with run_meta: {conflicts}"
        )

    # 回填：若 run_meta 中字段缺失或为 "<absent>"，则用 manifest 值覆盖。
    for field_name in required_anchor_fields:
        manifest_value = anchors.get(field_name)
        existing_value = run_meta.get(field_name)

        if existing_value is None or existing_value == "<absent>":
            if manifest_value is not None:
                run_meta[field_name] = manifest_value


def _ensure_run_meta_minimal_contract(run_meta: Dict[str, Any]) -> None:
    """
    功能：确保 run_meta 最小契约字段存在。

    Ensure required run_meta fields exist for run_closure payload construction.
    Normalizes all missing or None fields to "<absent>" sentinel.

    Args:
        run_meta: Run metadata mapping to mutate.

    Returns:
        None.

    Raises:
        TypeError: If run_meta is invalid.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")

    required_fields = [
        "run_id",
        "command",
        "created_at_utc",
        "started_at",
        "cfg_digest",
        "policy_path",
        "impl_id",
        "impl_version",
        "manifest_rel_path",
        "contract_version",
        "contract_digest",
        "contract_file_sha256",
        "contract_canon_sha256",
        "contract_bound_digest",
        "whitelist_version",
        "whitelist_digest",
        "whitelist_file_sha256",
        "whitelist_canon_sha256",
        "whitelist_bound_digest",
        "policy_path_semantics_version",
        "policy_path_semantics_digest",
        "policy_path_semantics_file_sha256",
        "policy_path_semantics_canon_sha256",
        "policy_path_semantics_bound_digest"
    ]

    missing_fields = []
    for field_name in required_fields:
        value = run_meta.get(field_name)
        if isinstance(value, str) and value == "<absent>":
            # 哨兵值禁止复用，必须 fail-fast。
            raise ValueError(
                "run_meta value must not reuse sentinel '<absent>': "
                f"field_name={field_name}"
            )
        # 检查是否为"缺失"：None 或 空字符串 或 "unknown"。
        if value is None or (isinstance(value, str) and not value) or value == "unknown":
            missing_fields.append(field_name)
            continue

    if not missing_fields:
        return

    missing_fields = sorted(set(missing_fields))
    for field_name in missing_fields:
        run_meta[field_name] = "<absent>"

    status_details = _merge_status_details(
        run_meta.get("status_details"),
        {"run_meta_missing_fields": missing_fields}
    )
    run_meta["status_details"] = status_details


def _cleanup_run_meta_missing_fields(run_meta: Dict[str, Any]) -> None:
    """
    功能：清理 run_meta_missing_fields 中已补齐的字段。

    Remove resolved fields from run_meta_missing_fields in status_details.

    Args:
        run_meta: Run metadata mapping to mutate.

    Returns:
        None.

    Raises:
        TypeError: If run_meta is invalid.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")

    status_details = run_meta.get("status_details")
    if not isinstance(status_details, dict):
        return

    missing_fields = status_details.get("run_meta_missing_fields")
    if not isinstance(missing_fields, list):
        return

    unresolved = []
    for field_name in missing_fields:
        value = run_meta.get(field_name)
        if value is None or value == "<absent>" or value == "unknown" or value == "":
            unresolved.append(field_name)

    if unresolved:
        status_details["run_meta_missing_fields"] = unresolved
    else:
        status_details.pop("run_meta_missing_fields", None)

    if status_details:
        run_meta["status_details"] = status_details
    else:
        run_meta["status_details"] = None


def _validate_run_meta_timestamps(run_meta: Dict[str, Any]) -> None:
    """
    功能：校验 run_meta 时间戳字段。

    Validate run_meta timestamp fields with UTC Z format.

    Args:
        run_meta: Run metadata mapping.

    Returns:
        None.

    Raises:
        TypeError: If run_meta is invalid or timestamps are not strings.
        ValueError: If timestamps are not valid UTC ISO Z strings.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")

    time_utils.validate_utc_iso_z(run_meta.get("created_at_utc"), "run_meta.created_at_utc")
    time_utils.validate_utc_iso_z(run_meta.get("started_at"), "run_meta.started_at")
    time_utils.validate_utc_iso_z(run_meta.get("ended_at"), "run_meta.ended_at")


def _resolve_bound_fact_sources(run_meta: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], str]:
    """
    功能：解析 run_meta 或上下文中的 bound_fact_sources。

    Resolve bound fact sources from run_meta or active context.

    Args:
        run_meta: Run metadata mapping.

    Returns:
        Tuple of (bound_fact_sources or None, status string).

    Raises:
        TypeError: If run_meta is invalid.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")

    bound_fact_sources = run_meta.get("bound_fact_sources")
    if isinstance(bound_fact_sources, dict):
        return bound_fact_sources, "bound"

    try:
        return records_io.get_bound_fact_sources(), "bound"
    except FactSourcesNotInitializedError:
        return None, "unbound"


def _backfill_run_meta_contract_from_fact_sources(
    run_meta: Dict[str, Any],
    bound_fact_sources: Optional[Dict[str, Any]]
) -> None:
    """
    功能：从 bound_fact_sources 回填 run_meta 合同绑定字段。

    Backfill contract binding fields in run_meta from bound_fact_sources.

    Args:
        run_meta: Run metadata mapping to mutate.
        bound_fact_sources: Bound fact sources mapping or None.

    Returns:
        None.

    Raises:
        TypeError: If run_meta is invalid.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")
    if bound_fact_sources is None:
        return
    if not isinstance(bound_fact_sources, dict):
        # bound_fact_sources 类型不合法，必须 fail-fast。
        raise TypeError("bound_fact_sources must be dict or None")

    contract_fields = [
        "contract_version",
        "contract_digest",
        "contract_file_sha256",
        "contract_canon_sha256",
        "contract_bound_digest"
    ]
    whitelist_fields = [
        "whitelist_version",
        "whitelist_digest",
        "whitelist_file_sha256",
        "whitelist_canon_sha256",
        "whitelist_bound_digest"
    ]
    semantics_fields = [
        "policy_path_semantics_version",
        "policy_path_semantics_digest",
        "policy_path_semantics_file_sha256",
        "policy_path_semantics_canon_sha256",
        "policy_path_semantics_bound_digest"
    ]
    all_fields = contract_fields + whitelist_fields + semantics_fields

    for field_name in all_fields:
        value = run_meta.get(field_name)
        if value is None or (isinstance(value, str) and (not value or value == "unknown")):
            source_value = bound_fact_sources.get(field_name)
            if source_value is not None:
                run_meta[field_name] = source_value


def bind_freeze_anchors_to_run_meta(
    run_meta: Dict[str, Any],
    contracts: Any,
    whitelist: Any,
    semantics: Any
) -> None:
    """
    功能：将冻结锚点字段一次性写入 run_meta。

    Bind freeze anchor fields from fact sources to run_meta in a single operation.
    Ensures all required contract/whitelist/semantics fields are present with <absent> sentinel on missing.

    Args:
        run_meta: Run metadata mapping to mutate.
        contracts: FrozenContracts instance.
        whitelist: RuntimeWhitelist instance.
        semantics: PolicyPathSemantics instance.

    Returns:
        None.

    Raises:
        TypeError: If run_meta is invalid.
    """
    if not isinstance(run_meta, dict):
        # run_meta 类型不符合预期，必须 fail-fast。
        raise TypeError("run_meta must be dict")

    # 构建快照以获取所有字段。
    snapshot = records_io.build_fact_sources_snapshot(contracts, whitelist, semantics)

    # 定义所有冻结锚点字段。
    freeze_anchor_fields = [
        "contract_version",
        "contract_digest",
        "contract_file_sha256",
        "contract_canon_sha256",
        "contract_bound_digest",
        "whitelist_version",
        "whitelist_digest",
        "whitelist_file_sha256",
        "whitelist_canon_sha256",
        "whitelist_bound_digest",
        "policy_path_semantics_version",
        "policy_path_semantics_digest",
        "policy_path_semantics_file_sha256",
        "policy_path_semantics_canon_sha256",
        "policy_path_semantics_bound_digest"
    ]

    # 一次性写入所有字段，缺失时写入 <absent>。
    for field_name in freeze_anchor_fields:
        value = snapshot.get(field_name)
        if value is None or (isinstance(value, str) and (not value or value == "unknown")):
            run_meta[field_name] = "<absent>"
        else:
            run_meta[field_name] = value


def _normalize_records_absence_values(record: Dict[str, Any]) -> None:
    """
    功能：规范化 records 中所有缺失值为统一的 "<absent>" 哨兵。
    
    Normalize all absence/null/unknown values in records to the frozen sentinel "<absent>".
    Ensures consistent semantics across all records and run_closure outputs.
    
    Args:
        record: Record dict to normalize in-place.
    
    Returns:
        None.
    
    Raises:
        TypeError: If record is not dict.
    """
    if not isinstance(record, dict):
        raise TypeError(f"record must be dict, got {type(record)}")
    
    # 遍历所有记录字段，将 None / "unknown" / "" 规范化为 "<absent>"。
    # 但保留某些特殊字段的 "unknown" 值。
    special_fields_allowing_unknown = {"failure_stage"}
    
    for key, value in list(record.items()):
        # 跳过特殊字段
        if key in special_fields_allowing_unknown:
            continue
        
        # 规范化缺失值
        if value is None or (isinstance(value, str) and (value == "" or value == "unknown")):
            record[key] = "<absent>"


# run_closure 字段族常量：用于类型分层归一化。
# 修改字段族时必须同步更新测试：tests/test_gate_schema_bundle_closure.py::test_run_closure_absence_is_type_layered_in_failure_paths
DICT_OR_NONE_FIELDS = frozenset({
    "impl_identity",
    "facts_anchor",
    "records_bundle",
    "pipeline_runtime_meta",
    "inference_runtime_meta",
    "infer_trace",
    "closure_stage",
    "path_policy",
    "bound_fact_sources",
    "fact_sources_binding",
    "recommended_enforce_report",
    "status"
})

LIST_OR_NONE_FIELDS = frozenset({
    "audit_warnings"
})


def _normalize_run_closure_absence_values(payload: Dict[str, Any]) -> None:
    """
    功能：对 run_closure 缺失值进行类型分层归一化。

    Normalize run_closure absence values with type-aware semantics.
    
    约束：
    - dict|None 字段：禁止归一化为 "<absent>" 字符串，保持 None 或 dict。
    - list|None 字段：禁止归一化为 "<absent>" 字符串，保持 None 或 list。
    - 其他 str 锚点字段：允许归一化为 "<absent>" 哨兵值。

    Args:
        payload: Run closure payload mapping to normalize in-place.

    Returns:
        None.

    Raises:
        TypeError: If payload is invalid.
    """
    if not isinstance(payload, dict):
        # payload 类型不符合预期，必须 fail-fast。
        raise TypeError("payload must be dict")

    for key, value in list(payload.items()):
        if key in DICT_OR_NONE_FIELDS:
            # dict|None 字段：禁止归一化为 "<absent>" 字符串。
            if value is None or (isinstance(value, str) and value in {"", "unknown", "<absent>"}):
                payload[key] = None
            continue
        if key in LIST_OR_NONE_FIELDS:
            # list|None 字段：禁止归一化为 "<absent>" 字符串。
            if value is None or (isinstance(value, str) and value in {"", "unknown", "<absent>"}):
                payload[key] = None
            # 保护 list 内部元素不被改变。
            continue

        # str 哨兵字段：允许归一化为 "<absent>"。
        if value is None or (isinstance(value, str) and (value == "" or value == "unknown")):
            payload[key] = "<absent>"


def _normalize_run_closure_missing(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：对 run_closure 进行类型分层缺失规范化。

    Normalize missing values for run_closure with type-aware semantics.

    Args:
        payload: Run closure payload mapping to normalize.

    Returns:
        Normalized payload mapping.

    Raises:
        TypeError: If payload is invalid.
    """
    if not isinstance(payload, dict):
        # payload 类型不符合预期，必须 fail-fast。
        raise TypeError("payload must be dict")

    required_fields = [
        "impl_identity",
        "facts_anchor",
        "records_bundle"
    ]
    optional_fields = [
        "anchors",
        "records_manifest"
    ]

    for field_name in required_fields:
        if field_name not in payload or payload.get(field_name) == "<absent>":
            payload[field_name] = None

    for field_name in optional_fields:
        if field_name in payload and payload.get(field_name) == "<absent>":
            payload[field_name] = None

    return payload


def _extract_bundle_field_name(message: str) -> str:
    """
    功能：从错误消息提取 field_name。

    Extract field_name from bundle error message.

    Args:
        message: Error message string.

    Returns:
        Field name or "unknown" if not found.

    Raises:
        TypeError: If message is invalid.
    """
    if not isinstance(message, str):
        # message 类型不符合预期，必须 fail-fast。
        raise TypeError("message must be str")

    marker = "field_name="
    start = message.find(marker)
    if start == -1:
        return "unknown"
    start += len(marker)
    end = message.find(",", start)
    if end == -1:
        end = len(message)
    field_name = message[start:end].strip()
    return field_name or "unknown"


def _extract_bundle_files(message: str) -> list[str]:
    """
    功能：从错误消息提取 files 列表。

    Extract file list from bundle error message.

    Args:
        message: Error message string.

    Returns:
        List of file paths (may be empty).

    Raises:
        TypeError: If message is invalid.
    """
    if not isinstance(message, str):
        # message 类型不符合预期，必须 fail-fast。
        raise TypeError("message must be str")

    files_marker = "files="
    start = message.find(files_marker)
    if start != -1:
        raw = message[start + len(files_marker):].strip()
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            pass
        raw = raw.strip("[]")
        return [item.strip().strip("'\"") for item in raw.split(",") if item.strip()]

    file_marker = "file="
    start = message.find(file_marker)
    if start != -1:
        value = message[start + len(file_marker):].strip()
        return [value] if value else []

    return []
