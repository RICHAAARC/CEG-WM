"""
输出路径布局与校验

功能说明：
- 规范化运行输出目录结构，确保 records/artifacts/logs 分层存在。
- 校验输出路径与语义分层一致，禁止越界或符号链接。
- 构建路径校验审计记录，包含 schema_version 和版本化字段。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from main.core import digests, time_utils
from main.core.errors import GateEnforcementError


def _reject_symlink(path: Path, *, label: str) -> None:
    """
    功能：拒绝符号链接路径。

    Reject symbolic link paths to prevent symlink pollution attacks.

    Args:
        path: Path object to check.
        label: Descriptive label for error context (e.g., "run_root", "records_dir").

    Returns:
        None.

    Raises:
        GateEnforcementError: If path exists and is a symbolic link.
    """
    if path.exists() and path.is_symlink():
        raise GateEnforcementError(
            f"SYMLINK_NOT_ALLOWED: {label}={path} must not be a symbolic link"
        )


def derive_run_root(output_dir: Path) -> Path:
    """
    功能：规范化运行输出根目录并拒绝越界。

    Normalize the run root path and reject traversal outside the base directory.

    Args:
        output_dir: Output directory path from CLI.

    Returns:
        Normalized run root path.

    Raises:
        TypeError: If output_dir is not a Path instance.
        ValueError: If output_dir is empty or escapes the base directory.
    """
    if not isinstance(output_dir, Path):
        # output_dir 类型不符合预期，必须 fail-fast。
        raise TypeError("output_dir must be Path")

    if str(output_dir).strip() == "":
        # output_dir 为空，必须 fail-fast。
        raise ValueError("output_dir must be non-empty path")

    if output_dir.is_absolute():
        run_root = output_dir.resolve()
        return run_root

    base_dir = Path.cwd().resolve()
    run_root = (base_dir / output_dir).resolve()
    try:
        run_root.relative_to(base_dir)
    except ValueError as exc:
        # output_dir 越界，必须 fail-fast。
        raise ValueError(f"output_dir escapes base directory: {output_dir}") from exc

    return run_root


def _is_dir_nonempty(path: Path) -> bool:
    """
    功能：判定目录是否非空。

    Check whether a directory is non-empty.

    Args:
        path: Directory path to check.

    Returns:
        True if directory exists and contains files/subdirectories; otherwise False.

    Raises:
        TypeError: If path is not a Path instance.
    """
    if not isinstance(path, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise TypeError("path must be Path")

    if not path.exists() or not path.is_dir():
        return False

    try:
        return any(path.iterdir())
    except OSError:
        # iterdir 错误，判定为非空以 fail-fast。
        return True


def ensure_output_layout(
    run_root: Path,
    allow_nonempty_run_root: bool,
    allow_nonempty_run_root_reason: Optional[str],
    override_applied: Optional[Dict[str, Any]]
) -> Dict[str, Path]:
    """
    功能：确保输出目录布局存在并返回路径集合，附加强约束检查。

    Ensure the run output layout exists and return records/artifacts/logs directories.
    Apply nonempty layout enforcement unless explicitly permitted by cfg.allow_nonempty_run_root.

    Args:
        run_root: Normalized run root directory.
        allow_nonempty_run_root: Explicit reuse flag from CLI/override.
        allow_nonempty_run_root_reason: Explicit reason for reuse; required when allow_nonempty_run_root is True.
        override_applied: override_applied audit mapping; required when allow_nonempty_run_root is True.

    Returns:
        Mapping with keys: records_dir, artifacts_dir, logs_dir.

    Raises:
        TypeError: If run_root is not a Path instance or inputs have invalid types.
        ValueError: If run_root exists but is not a directory.
        GateEnforcementError: If run_root or any layout subdirectory is a symbolic link,
                              or if nonempty layout enforcement fails.
    """
    if not isinstance(run_root, Path):
        # run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("run_root must be Path")

    if not isinstance(allow_nonempty_run_root, bool):
        # allow_nonempty_run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("allow_nonempty_run_root must be bool")
    if allow_nonempty_run_root_reason is not None and not isinstance(allow_nonempty_run_root_reason, str):
        # allow_nonempty_run_root_reason 类型不符合预期，必须 fail-fast。
        raise TypeError("allow_nonempty_run_root_reason must be str or None")
    if override_applied is not None and not isinstance(override_applied, dict):
        # override_applied 类型不符合预期，必须 fail-fast。
        raise TypeError("override_applied must be dict or None")

    # 在任何 mkdir() 之前检查 run_root 本体是否为 symlink。
    _reject_symlink(run_root, label="run_root")

    if run_root.exists() and not run_root.is_dir():
        # run_root 不是目录，必须 fail-fast。
        raise ValueError(f"run_root is not a directory: {run_root}")

    # 检查强约束：run_root 为空或布局目录为空。
    if allow_nonempty_run_root:
        if override_applied is None:
            # 未提供 override_applied 时不得允许复用。
            raise GateEnforcementError(
                "run_root reuse requires override_applied audit block"
            )
        if not isinstance(allow_nonempty_run_root_reason, str) or not allow_nonempty_run_root_reason:
            # 允许复用时必须提供理由。
            raise GateEnforcementError(
                "run_root reuse requires non-empty allow_nonempty_run_root_reason"
            )
    else:
        if allow_nonempty_run_root_reason is not None:
            # 未开启复用时不得提供理由。
            raise GateEnforcementError(
                "allow_nonempty_run_root_reason must be None when reuse is disabled"
            )

    # 如果 run_root 已存在，检查子目录非空性。
    if run_root.exists():
        records_dir_temp = run_root / "records"
        artifacts_dir_temp = run_root / "artifacts"
        logs_dir_temp = run_root / "logs"

        # 检查至少一个子目录非空。
        layout_nonempty = (
            _is_dir_nonempty(records_dir_temp) or
            _is_dir_nonempty(artifacts_dir_temp) or
            _is_dir_nonempty(logs_dir_temp)
        )

        if layout_nonempty and not allow_nonempty_run_root:
            # run_root 存在且任一子目录非空，但未显式开启复用，必须 fail-fast。
            raise GateEnforcementError(
                f"run_root layout nonempty and reuse not permitted: "
                f"run_root={run_root}, "
                f"records_nonempty={_is_dir_nonempty(records_dir_temp)}, "
                f"artifacts_nonempty={_is_dir_nonempty(artifacts_dir_temp)}, "
                f"logs_nonempty={_is_dir_nonempty(logs_dir_temp)}"
            )

    run_root.mkdir(parents=True, exist_ok=True)

    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"

    # 在创建前检查是否已存在且为 symlink。
    _reject_symlink(records_dir, label="records_dir")
    _reject_symlink(artifacts_dir, label="artifacts_dir")
    _reject_symlink(logs_dir, label="logs_dir")

    # 创建或保留目录。
    records_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 创建后再次检查。
    _reject_symlink(records_dir, label="records_dir")
    _reject_symlink(artifacts_dir, label="artifacts_dir")
    _reject_symlink(logs_dir, label="logs_dir")

    return {
        "records_dir": records_dir,
        "artifacts_dir": artifacts_dir,
        "logs_dir": logs_dir
    }


def anchor_requirements(run_root: Path, repo_root: Optional[Path] = None) -> None:
    """
    功能：将仓库根目录的 requirements.txt 复制到 run_root。

    Copy requirements.txt from repository root to run_root for reproducibility anchoring.

    Args:
        run_root: Run root directory.
        repo_root: Repository root directory. If None, uses current working directory.

    Returns:
        None.

    Raises:
        TypeError: If run_root is invalid.
        FileNotFoundError: If requirements.txt not found in repo root.
        OSError: If copy operation fails.
    """
    if not isinstance(run_root, Path):
        # run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("run_root must be Path")
    
    if repo_root is None:
        repo_root = Path.cwd().resolve()
    elif not isinstance(repo_root, Path):
        # repo_root 类型不符合预期，必须 fail-fast。
        raise TypeError("repo_root must be Path or None")
    
    lock_source = repo_root / "requirements.txt"
    if not lock_source.exists() or not lock_source.is_file():
        # requirements.txt 缺失，必须 fail-fast。
        raise FileNotFoundError(
            f"requirements.txt not found at {lock_source}"
        )
    
    lock_target = run_root / "requirements.txt"
    
    # 使用无上下文受控复制接口封装依赖锁文件（启用路径约束验证）。
    from main.core import records_io
    records_io.copy_file_controlled_unbound(
        run_root,
        run_root / "artifacts",
        lock_source,
        lock_target,
        kind="env_lock"
    )


def validate_output_target(path: Path, kind: str, run_root: Path) -> None:
    """
    功能：校验输出目标路径与语义分层一致性。

    Validate that output path is within the correct layout directory.

    Args:
        path: Target output path.
        kind: Output kind: "record", "artifact", or "log".
        run_root: Normalized run root directory.

    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If kind is unsupported or path violates layout constraints.
        GateEnforcementError: If layout directory is a symbolic link.
    """
    if not isinstance(path, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise TypeError("path must be Path")
    if not isinstance(kind, str) or not kind:
        # kind 输入不合法，必须 fail-fast。
        raise ValueError("kind must be non-empty str")
    if not isinstance(run_root, Path):
        # run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("run_root must be Path")

    kind_map = {
        "record": "records",
        "artifact": "artifacts",
        "log": "logs"
    }
    if kind not in kind_map:
        # kind 不支持，必须 fail-fast。
        raise ValueError(f"unsupported output kind: {kind}")

    base_dir = run_root.resolve()
    expected_dir = base_dir / kind_map[kind]
    
    # 在进行 resolve 之前检查 expected_dir 本体是否为 symlink。
    _reject_symlink(expected_dir, label=f"{kind_map[kind]}_dir")
    
    # 仅当 target_path 已存在时才检查其本体。
    if path.exists():
        _reject_symlink(path, label="target_path")
    
    # 进行现有的校验逻辑。
    expected_dir_resolved = expected_dir.resolve()
    target_path = path.resolve()

    try:
        target_path.relative_to(expected_dir_resolved)
    except ValueError as exc:
        # 输出路径越界，必须 fail-fast。
        raise ValueError(
            f"{kind} output must be under {expected_dir_resolved}: {path}"
        ) from exc


def classify_output_path(path: Path, run_root: Path) -> str:
    """
    功能：判定输出路径语义分类。
    
    Classify output path as records, artifacts, logs, or other.
    This function centralizes path classification logic used by records_io and audits.
    
    Args:
        path: Target path to classify.
        run_root: Normalized run root directory.
    
    Returns:
        Output classification string: "records", "artifacts", "logs", or "other".
    
    Raises:
        TypeError: If inputs are of invalid types.
    """
    if not isinstance(path, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise TypeError("path must be Path")
    if not isinstance(run_root, Path):
        # run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("run_root must be Path")
    
    base_dir = run_root.resolve()
    target = path.resolve()
    
    # 检查各目录归属关系，按优先级排列。
    records_dir = base_dir / "records"
    artifacts_dir = base_dir / "artifacts"
    logs_dir = base_dir / "logs"
    
    try:
        target.relative_to(records_dir.resolve())
        return "records"
    except ValueError:
        pass
    
    try:
        target.relative_to(artifacts_dir.resolve())
        return "artifacts"
    except ValueError:
        pass
    
    try:
        target.relative_to(logs_dir.resolve())
        return "logs"
    except ValueError:
        pass
    
    return "other"


def build_path_validation_audit_record(
    path: Path,
    kind: str,
    run_root: Path,
    status: str,
    violation_reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    功能：构造路径校验审计记录。

    Build a path validation audit record for traceability.

    Args:
        path: Target output path.
        kind: Output kind: "record", "artifact", or "log".
        run_root: Normalized run root directory.
        status: Validation status: "ok" or "rejected".
        violation_reason: Optional reason for rejection.

    Returns:
        Audit record dict with record_type="path_validation_audit".

    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If status is invalid.
    """
    if not isinstance(path, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise TypeError("path must be Path")
    if not isinstance(kind, str) or not kind:
        # kind 输入不合法，必须 fail-fast。
        raise ValueError("kind must be non-empty str")
    if not isinstance(run_root, Path):
        # run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("run_root must be Path")
    if status not in {"ok", "rejected"}:
        # status 取值不合法，必须 fail-fast。
        raise ValueError(f"status must be 'ok' or 'rejected', got {status}")

    return {
        "record_type": "path_validation_audit",
        "timestamp": time_utils.now_utc_iso_z(),
        "target_path": path.resolve().as_posix(),
        "kind": kind,
        "run_root": run_root.resolve().as_posix(),
        "status": status,
        "violation_reason": violation_reason
    }


def build_comprehensive_path_validation_audit(
    run_root: Path,
    policy_path: str,
    policy_path_semantics_version: str,
    runtime_whitelist_version: str,
    original_input: Optional[str] = None,
    output_paths_relative: Optional[Dict[str, str]] = None,
    validation_status: str = "ok",
    failure_reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    功能：构造综合路径校验审计记录。
    
    Build comprehensive path validation audit with schema version, 
    versions, and structured metadata for run_closure anchoring.
    
    Args:
        run_root: Resolved run root directory.
        policy_path: Policy path value from record.
        policy_path_semantics_version: Version from policy_path_semantics.yaml.
        runtime_whitelist_version: Version from runtime_whitelist.yaml.
        original_input: Original input parameter (optional).
        output_paths_relative: Optional output path mapping relative to run_root.
        validation_status: Validation result: "ok" or "failed".
        failure_reason: Optional reason for failure.
    
    Returns:
        Comprehensive audit record dict with schema_version and canon_sha256-able structure.
    
    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If validation_status is invalid.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(policy_path, str) or not policy_path:
        raise ValueError("policy_path must be non-empty str")
    if not isinstance(policy_path_semantics_version, str) or not policy_path_semantics_version:
        raise ValueError("policy_path_semantics_version must be non-empty str")
    if not isinstance(runtime_whitelist_version, str) or not runtime_whitelist_version:
        raise ValueError("runtime_whitelist_version must be non-empty str")
    if output_paths_relative is not None:
        if not isinstance(output_paths_relative, dict):
            raise TypeError("output_paths_relative must be dict or None")
        for key, value in output_paths_relative.items():
            if not isinstance(key, str) or not key:
                raise ValueError("output_paths_relative keys must be non-empty str")
            if not isinstance(value, str) or not value:
                raise ValueError("output_paths_relative values must be non-empty str")
    if validation_status not in {"ok", "failed"}:
        raise ValueError(f"validation_status must be 'ok' or 'failed', got {validation_status}")
    
    audit_record = {
        "record_type": "path_validation_audit",
        "schema_version": "v1.0",
        "timestamp": time_utils.now_utc_iso_z(),
        "run_root_resolved": run_root.resolve().as_posix(),
        "original_input": original_input,
        "output_dir_input": original_input,
        "output_paths_relative": output_paths_relative,
        "policy_path": policy_path,
        "policy_path_semantics_version": policy_path_semantics_version,
        "runtime_whitelist_version": runtime_whitelist_version,
        "validation_status": validation_status,
        "failure_reason": failure_reason
    }
    
    return audit_record


def build_output_paths_relative(
    run_root: Path,
    records_dir: Path,
    artifacts_dir: Path,
    logs_dir: Path
) -> Dict[str, str]:
    """
    功能：构造输出路径相对映射。

    Build output directory mapping relative to run_root.

    Args:
        run_root: Run root directory.
        records_dir: Records output directory.
        artifacts_dir: Artifacts output directory.
        logs_dir: Logs output directory.

    Returns:
        Mapping of output paths relative to run_root.

    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If any output path escapes run_root.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(records_dir, Path):
        raise TypeError("records_dir must be Path")
    if not isinstance(artifacts_dir, Path):
        raise TypeError("artifacts_dir must be Path")
    if not isinstance(logs_dir, Path):
        raise TypeError("logs_dir must be Path")

    run_root_resolved = run_root.resolve()
    records_relative = _relative_to_run_root(run_root_resolved, records_dir)
    artifacts_relative = _relative_to_run_root(run_root_resolved, artifacts_dir)
    logs_relative = _relative_to_run_root(run_root_resolved, logs_dir)

    return {
        "records_dir": records_relative,
        "artifacts_dir": artifacts_relative,
        "logs_dir": logs_relative
    }


def _relative_to_run_root(run_root: Path, target_path: Path) -> str:
    """
    功能：计算目标路径相对 run_root 的路径。

    Compute relative path from run_root to target_path.

    Args:
        run_root: Resolved run root directory.
        target_path: Target path to relativize.

    Returns:
        Relative path in posix format.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If target_path escapes run_root.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(target_path, Path):
        raise TypeError("target_path must be Path")

    target_resolved = target_path.resolve()
    try:
        relative = target_resolved.relative_to(run_root)
    except ValueError as exc:
        # target_path 越界，必须 fail-fast。
        raise ValueError(
            f"output path escapes run_root: target_path={target_path}, run_root={run_root}"
        ) from exc

    return relative.as_posix()


def compute_path_audit_canon_sha256(audit_record: Dict[str, Any]) -> str:
    """
    功能：计算路径审计记录的规范 SHA256。
    
    Compute canonical SHA256 hash of path validation audit record.
    
    Args:
        audit_record: Path validation audit record dict.
    
    Returns:
        Canonical SHA256 hash string.
    
    Raises:
        TypeError: If audit_record is not dict.
    """
    if not isinstance(audit_record, dict):
        raise TypeError("audit_record must be dict")
    
    return digests.canonical_sha256(audit_record)
