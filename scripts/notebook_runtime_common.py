"""
文件目的：Notebook 与 scripts 编排共用的纯运行时辅助能力。
Module type: General module

职责边界：
1. 仅承载 SHA256、复制、压缩解压、路径校验、Drive 目录组织、运行元数据收集等纯编排能力。
2. 不承载 main/ 下的 watermarking、detection、evaluation 核心机制。
3. 为独立 Colab 会话下的 stage package、lineage 与审计输出提供稳定工具。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, cast

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_01_NAME = "01_Paper_Full_Cuda"
STAGE_02_NAME = "02_Parallel_Attestation_Statistics"
STAGE_03_NAME = "03_Experiment_Matrix_Full"


def utc_now_iso() -> str:
    """
    功能：返回 UTC ISO 8601 时间戳。

    Return the current UTC timestamp in ISO 8601 format.

    Args:
        None.

    Returns:
        UTC timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def make_stage_run_id() -> str:
    """
    功能：生成阶段运行唯一标识。

    Generate a stable stage run identifier based on UTC time and random suffix.

    Args:
        None.

    Returns:
        Stage run identifier string.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def ensure_directory(path_obj: Path) -> Path:
    """
    功能：确保目录存在。

    Ensure that a directory exists.

    Args:
        path_obj: Directory path.

    Returns:
        The same directory path.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def normalize_path_value(path_value: Any) -> str:
    """
    功能：将路径值规范化为 POSIX 字符串。

    Normalize a path-like value into a stable POSIX string.

    Args:
        path_value: Path-like value.

    Returns:
        Normalized POSIX path string, or "<absent>" when unavailable.
    """
    if isinstance(path_value, Path):
        return path_value.resolve().as_posix()
    if isinstance(path_value, str) and path_value.strip():
        candidate = Path(path_value.strip()).expanduser()
        if candidate.is_absolute():
            return candidate.resolve().as_posix()
        return (REPO_ROOT / candidate).resolve().as_posix()
    return "<absent>"


def relative_path_under_base(base_path: Path, path_value: Any) -> str:
    """
    功能：计算相对基础目录的路径。

    Represent a path-like value relative to a base directory when possible.

    Args:
        base_path: Base directory path.
        path_value: Path-like value.

    Returns:
        Relative POSIX path, "." for the base itself, or "<absent>" when not under base.
    """
    if not isinstance(base_path, Path):
        raise TypeError("base_path must be Path")

    base_text = normalize_path_value(base_path).rstrip("/")
    candidate_text = normalize_path_value(path_value)
    if candidate_text == "<absent>":
        return "<absent>"
    if candidate_text == base_text:
        return "."

    prefix = f"{base_text}/"
    if candidate_text.startswith(prefix):
        return candidate_text[len(prefix):]
    return "<absent>"


def resolve_repo_path(path_value: str, repo_root: Path = REPO_ROOT) -> Path:
    """
    功能：按仓库根目录解析路径。

    Resolve a path against the repository root unless it is already absolute.

    Args:
        path_value: Raw path string.
        repo_root: Repository root path.

    Returns:
        Resolved absolute path.
    """
    if not isinstance(path_value, str) or not path_value.strip():
        raise TypeError("path_value must be non-empty str")
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")

    candidate = Path(path_value.strip()).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def write_json_atomic(path_obj: Path, payload: Mapping[str, Any]) -> None:
    """
    功能：稳定写出 JSON 文件。

    Persist a mapping as formatted JSON with parent-directory creation.

    Args:
        path_obj: Destination JSON path.
        payload: Mapping payload.

    Returns:
        None.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be Mapping")
    ensure_directory(path_obj.parent)
    path_obj.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def read_json_dict(path_obj: Path) -> Dict[str, Any]:
    """
    功能：读取 JSON 对象文件。

    Read a JSON file and require its root node to be a mapping.

    Args:
        path_obj: JSON file path.

    Returns:
        Parsed mapping, or an empty mapping when unavailable or invalid.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not path_obj.exists() or not path_obj.is_file():
        return {}
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return cast(Dict[str, Any], payload) if isinstance(payload, dict) else {}


def load_yaml_mapping(path_obj: Path) -> Dict[str, Any]:
    """
    功能：读取 YAML 映射配置。

    Read a YAML file and require its root node to be a mapping.

    Args:
        path_obj: YAML path.

    Returns:
        Parsed mapping.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"yaml file not found: {path_obj}")
    payload = yaml.safe_load(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be mapping: {path_obj}")
    return cast(Dict[str, Any], payload)


def write_yaml_mapping(path_obj: Path, payload: Mapping[str, Any]) -> None:
    """
    功能：写出 YAML 映射配置。

    Persist a mapping as YAML with stable parent-directory creation.

    Args:
        path_obj: Destination YAML path.
        payload: Mapping payload.

    Returns:
        None.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be Mapping")
    ensure_directory(path_obj.parent)
    path_obj.write_text(
        yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def compute_file_sha256(path_obj: Path) -> str:
    """
    功能：计算单文件 SHA256。

    Compute the SHA256 digest of one file.

    Args:
        path_obj: File path.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"file not found for sha256: {path_obj}")

    hasher = hashlib.sha256()
    with path_obj.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def copy_file(source_path: Path, destination_path: Path) -> Path:
    """
    功能：复制文件并保留元数据。

    Copy one file to the destination path with metadata preservation.

    Args:
        source_path: Existing source file.
        destination_path: Destination file path.

    Returns:
        Destination file path.
    """
    if not isinstance(source_path, Path):
        raise TypeError("source_path must be Path")
    if not isinstance(destination_path, Path):
        raise TypeError("destination_path must be Path")
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"source file not found: {source_path}")
    ensure_directory(destination_path.parent)
    shutil.copy2(source_path, destination_path)
    return destination_path


def validate_path_within_base(base_path: Path, candidate_path: Path, label: str) -> None:
    """
    功能：校验路径位于受控基础目录内。

    Validate that a candidate path is located under a controlled base directory.

    Args:
        base_path: Controlled base directory.
        candidate_path: Candidate path to validate.
        label: Human-readable label for error reporting.

    Returns:
        None.
    """
    if not isinstance(base_path, Path):
        raise TypeError("base_path must be Path")
    if not isinstance(candidate_path, Path):
        raise TypeError("candidate_path must be Path")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")
    try:
        candidate_path.resolve().relative_to(base_path.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be under {base_path}, got {candidate_path}") from exc


def extract_zip_archive(zip_path: Path, destination_root: Path) -> List[str]:
    """
    功能：解压 ZIP 包到目标目录。

    Extract a ZIP archive into a clean destination directory.

    Args:
        zip_path: ZIP archive path.
        destination_root: Extraction directory.

    Returns:
        Relative file names extracted from the archive.
    """
    if not isinstance(zip_path, Path):
        raise TypeError("zip_path must be Path")
    if not isinstance(destination_root, Path):
        raise TypeError("destination_root must be Path")
    if not zip_path.exists() or not zip_path.is_file():
        raise FileNotFoundError(f"zip archive not found: {zip_path}")

    if destination_root.exists():
        shutil.rmtree(destination_root)
    ensure_directory(destination_root)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination_root)
        members = [name for name in archive.namelist() if name and not name.endswith("/")]
    return sorted(members)


def create_zip_archive_from_directory(source_root: Path, zip_path: Path) -> Path:
    """
    功能：将目录压缩为 ZIP 包。

    Create a ZIP archive from one directory using relative paths.

    Args:
        source_root: Directory to archive.
        zip_path: Destination ZIP path.

    Returns:
        Created ZIP path.
    """
    if not isinstance(source_root, Path):
        raise TypeError("source_root must be Path")
    if not isinstance(zip_path, Path):
        raise TypeError("zip_path must be Path")
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"source_root not found: {source_root}")

    ensure_directory(zip_path.parent)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(source_root.rglob("*")):
            if not file_path.is_file():
                continue
            archive.write(file_path, arcname=file_path.relative_to(source_root).as_posix())
    return zip_path


def resolve_stage_roots(drive_project_root: Path, stage_name: str, stage_run_id: str) -> Dict[str, Path]:
    """
    功能：构造阶段隔离的 Drive 目录布局。

    Build the stage-isolated Google Drive directory layout.

    Args:
        drive_project_root: Google Drive project root.
        stage_name: Stable stage name.
        stage_run_id: Unique stage run identifier.

    Returns:
        Mapping containing run_root, log_root, runtime_state_root, and export_root.
    """
    if not isinstance(drive_project_root, Path):
        raise TypeError("drive_project_root must be Path")
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(stage_run_id, str) or not stage_run_id:
        raise TypeError("stage_run_id must be non-empty str")

    return {
        "run_root": drive_project_root / "runs" / stage_name / stage_run_id,
        "log_root": drive_project_root / "logs" / stage_name / stage_run_id,
        "runtime_state_root": drive_project_root / "runtime_state" / stage_name / stage_run_id,
        "export_root": drive_project_root / "exports" / stage_name / stage_run_id,
    }


def build_stage_package_filename(
    stage_name: str,
    stage_run_id: str,
    source_stage_run_id: Optional[str] = None,
) -> str:
    """
    功能：构造带 lineage 的阶段包文件名。

    Build a stage package filename that exposes run lineage in its basename.

    Args:
        stage_name: Stable stage name.
        stage_run_id: Current stage run identifier.
        source_stage_run_id: Optional upstream stage run identifier.

    Returns:
        ZIP file name.
    """
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(stage_run_id, str) or not stage_run_id:
        raise TypeError("stage_run_id must be non-empty str")
    if source_stage_run_id is not None and (not isinstance(source_stage_run_id, str) or not source_stage_run_id):
        raise TypeError("source_stage_run_id must be non-empty str or None")
    if source_stage_run_id:
        return f"{stage_name}__{stage_run_id}__from__{source_stage_run_id}.zip"
    return f"{stage_name}__{stage_run_id}.zip"


def run_command_with_logs(
    command: Sequence[str],
    cwd: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
) -> Dict[str, Any]:
    """
    功能：执行子进程并写出 stdout 与 stderr 日志。

    Execute a subprocess and persist stdout and stderr to explicit log files.

    Args:
        command: Command argument sequence.
        cwd: Working directory.
        stdout_log_path: Stdout log file path.
        stderr_log_path: Stderr log file path.

    Returns:
        Execution result mapping with return code and log paths.
    """
    if not isinstance(cwd, Path):
        raise TypeError("cwd must be Path")
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    command_list = [str(item) for item in command]
    result = subprocess.run(
        command_list,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout_log_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_log_path.write_text(result.stderr or "", encoding="utf-8")
    return {
        "return_code": int(result.returncode),
        "stdout_log_path": stdout_log_path.as_posix(),
        "stderr_log_path": stderr_log_path.as_posix(),
        "command": command_list,
    }


def _run_git_command(repo_root: Path, args: List[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return "<absent>"
    return (result.stdout or "").strip() or "<absent>"


def collect_git_summary(repo_root: Path) -> Dict[str, Any]:
    """
    功能：收集 Git 摘要。

    Collect repository Git metadata for manifest emission.

    Args:
        repo_root: Repository root path.

    Returns:
        Git summary mapping.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    return {
        "remote": _run_git_command(repo_root, ["remote", "get-url", "origin"]),
        "branch": _run_git_command(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run_git_command(repo_root, ["rev-parse", "HEAD"]),
        "commit_short": _run_git_command(repo_root, ["rev-parse", "--short=8", "HEAD"]),
    }


def collect_python_summary() -> Dict[str, Any]:
    """
    功能：收集 Python 运行环境摘要。

    Collect Python runtime metadata.

    Args:
        None.

    Returns:
        Python runtime summary mapping.
    """
    return {
        "python_version": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
    }


def collect_cuda_summary() -> Dict[str, Any]:
    """
    功能：收集 CUDA 与 GPU 摘要。

    Collect CUDA and GPU metadata without failing when hardware is absent.

    Args:
        None.

    Returns:
        CUDA and GPU summary mapping.
    """
    summary: Dict[str, Any] = {
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "nvidia_smi_path": shutil.which("nvidia-smi") or "<absent>",
        "gpu_name": "<absent>",
        "cuda_available": False,
        "cuda_device_count": 0,
    }
    if summary["nvidia_smi_available"]:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            gpu_lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
            summary["gpu_query_lines"] = gpu_lines
            if gpu_lines:
                summary["gpu_name"] = gpu_lines[0]
    try:
        import torch

        summary["cuda_available"] = bool(torch.cuda.is_available())
        summary["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            summary["torch_gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        summary["torch_gpu_name"] = "<unavailable>"
    return summary


def collect_attestation_env_summary(cfg_obj: Mapping[str, Any]) -> Dict[str, Any]:
    """
    功能：收集 attestation 环境变量状态摘要。

    Collect presence-only attestation environment metadata.

    Args:
        cfg_obj: Runtime configuration mapping.

    Returns:
        Attestation environment status summary.
    """
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")
    attestation_node = cfg_obj.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    env_names: List[str] = []
    for key_name in ("k_master_env_var", "k_prompt_env_var", "k_seed_env_var"):
        value = attestation_cfg.get(key_name)
        if isinstance(value, str) and value:
            env_names.append(value)
    return {
        "enabled": bool(attestation_cfg.get("enabled", False)),
        "required_env_vars": env_names,
        "present_env_vars": [item for item in env_names if os.environ.get(item)],
        "missing_env_vars": [item for item in env_names if not os.environ.get(item)],
    }


def collect_model_summary(cfg_obj: Mapping[str, Any]) -> Dict[str, Any]:
    """
    功能：收集模型配置摘要。

    Collect lightweight model metadata without hashing the full cache directory.

    Args:
        cfg_obj: Runtime configuration mapping.

    Returns:
        Model metadata summary.
    """
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")
    summary: Dict[str, Any] = {
        "model_id": cfg_obj.get("model_id"),
        "model_source": cfg_obj.get("model_source"),
        "hf_revision": cfg_obj.get("hf_revision"),
        "hf_home": os.environ.get("HF_HOME", "<absent>"),
        "huggingface_hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE", "<absent>"),
        "cache_scan_status": "not_attempted",
    }
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        repo_id = summary.get("model_id")
        repos = [repo.repo_id for repo in cache_info.repos]
        summary["cache_scan_status"] = "ok"
        summary["cache_repo_present"] = bool(isinstance(repo_id, str) and repo_id in repos)
    except Exception:
        summary["cache_scan_status"] = "unavailable"
        summary["cache_repo_present"] = False
    return summary


def collect_weight_summary(repo_root: Path, cfg_obj: Mapping[str, Any]) -> Dict[str, Any]:
    """
    功能：收集 InSPyReNet 权重摘要。

    Collect a concise summary for the semantic mask weight file.

    Args:
        repo_root: Repository root path.
        cfg_obj: Runtime configuration mapping.

    Returns:
        Weight metadata summary.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")

    mask_cfg = cast(Dict[str, Any], cfg_obj.get("mask")) if isinstance(cfg_obj.get("mask"), dict) else {}
    semantic_model_path = mask_cfg.get("semantic_model_path")
    summary: Dict[str, Any] = {
        "semantic_model_path": semantic_model_path,
        "exists": False,
        "sha256": "<absent>",
        "size_bytes": None,
    }
    if isinstance(semantic_model_path, str) and semantic_model_path:
        path_obj = (repo_root / semantic_model_path).resolve()
        summary["resolved_path"] = path_obj.as_posix()
        if path_obj.exists() and path_obj.is_file():
            summary["exists"] = True
            summary["size_bytes"] = int(path_obj.stat().st_size)
            summary["sha256"] = compute_file_sha256(path_obj)
    return summary


def copy_prompt_snapshot(repo_root: Path, cfg_obj: Mapping[str, Any], destination_root: Path) -> Dict[str, Any]:
    """
    功能：复制 prompt 文件快照。

    Copy the configured prompt file into runtime metadata for packaging and lineage.

    Args:
        repo_root: Repository root path.
        cfg_obj: Runtime configuration mapping.
        destination_root: Prompt snapshot directory.

    Returns:
        Prompt snapshot summary.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")
    if not isinstance(destination_root, Path):
        raise TypeError("destination_root must be Path")

    prompt_value = cfg_obj.get("inference_prompt_file")
    summary: Dict[str, Any] = {
        "source_path": "<absent>",
        "snapshot_path": "<absent>",
        "exists": False,
    }
    if not isinstance(prompt_value, str) or not prompt_value:
        return summary

    source_path = (repo_root / prompt_value).resolve()
    summary["source_path"] = source_path.as_posix()
    if not source_path.exists() or not source_path.is_file():
        return summary

    destination_path = ensure_directory(destination_root) / source_path.name
    copy_file(source_path, destination_path)
    summary["snapshot_path"] = destination_path.as_posix()
    summary["exists"] = True
    summary["sha256"] = compute_file_sha256(destination_path)
    return summary


def build_package_index(file_paths: Iterable[Path], package_root: Path) -> Dict[str, Any]:
    """
    功能：构建 package 内部索引。

    Build a package index with relative paths, sizes, and file digests.

    Args:
        file_paths: Files included in the package staging directory.
        package_root: Package staging root.

    Returns:
        Package index mapping.
    """
    if not isinstance(package_root, Path):
        raise TypeError("package_root must be Path")

    entries: List[Dict[str, Any]] = []
    for file_path in sorted(file_paths):
        if not file_path.exists() or not file_path.is_file():
            continue
        entries.append(
            {
                "relative_path": file_path.relative_to(package_root).as_posix(),
                "size_bytes": int(file_path.stat().st_size),
                "sha256": compute_file_sha256(file_path),
            }
        )
    return {
        "created_at": utc_now_iso(),
        "file_count": len(entries),
        "files": entries,
    }


def collect_file_index(base_root: Path, path_mapping: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    功能：构造文件索引摘要。

    Build a file-index mapping with existence and relative-path information.

    Args:
        base_root: Base path for relative indexing.
        path_mapping: Label to path-like mapping.

    Returns:
        Indexed path summary mapping.
    """
    if not isinstance(base_root, Path):
        raise TypeError("base_root must be Path")
    if not isinstance(path_mapping, Mapping):
        raise TypeError("path_mapping must be Mapping")

    summary: Dict[str, Dict[str, Any]] = {}
    for label, raw_path in path_mapping.items():
        path_obj = raw_path if isinstance(raw_path, Path) else Path(str(raw_path))
        exists = path_obj.exists() and path_obj.is_file()
        summary[str(label)] = {
            "path": normalize_path_value(path_obj),
            "relative_path": relative_path_under_base(base_root, path_obj),
            "exists": exists,
        }
        if exists:
            summary[str(label)]["sha256"] = compute_file_sha256(path_obj)
    return summary


def stage_relative_copy(source_path: Path, package_root: Path, relative_destination: str) -> Path:
    """
    功能：按相对路径复制文件到 package staging 目录。

    Copy one file into the package staging directory using a relative destination.

    Args:
        source_path: Existing source file.
        package_root: Package staging root.
        relative_destination: Relative destination path inside the package.

    Returns:
        Destination file path.
    """
    if not isinstance(source_path, Path):
        raise TypeError("source_path must be Path")
    if not isinstance(package_root, Path):
        raise TypeError("package_root must be Path")
    if not isinstance(relative_destination, str) or not relative_destination.strip():
        raise TypeError("relative_destination must be non-empty str")

    destination_path = package_root / Path(relative_destination)
    copy_file(source_path, destination_path)
    return destination_path


def finalize_stage_package(
    *,
    stage_name: str,
    stage_run_id: str,
    package_root: Path,
    export_root: Path,
    source_stage_run_id: Optional[str],
    source_stage_package_path: Optional[str],
    package_manifest_path: Path,
) -> Dict[str, Any]:
    """
    功能：写出 package index、压缩包与 package_manifest。

    Finalize one stage package by emitting the package index, ZIP archive, and package manifest.

    Args:
        stage_name: Stable stage name.
        stage_run_id: Current stage run identifier.
        package_root: Package staging directory.
        export_root: Stage export directory.
        source_stage_run_id: Optional upstream stage run identifier.
        source_stage_package_path: Optional upstream stage package path.
        package_manifest_path: Destination package manifest path.

    Returns:
        Package manifest mapping.
    """
    package_index_path = package_root / "artifacts" / "package_index.json"
    package_index = build_package_index(package_root.rglob("*"), package_root)
    write_json_atomic(package_index_path, package_index)

    package_filename = build_stage_package_filename(stage_name, stage_run_id, source_stage_run_id)
    package_path = create_zip_archive_from_directory(package_root, export_root / package_filename)
    package_manifest = {
        "package_path": package_path.as_posix(),
        "package_filename": package_filename,
        "package_sha256": compute_file_sha256(package_path),
        "package_created_at": utc_now_iso(),
        "stage_name": stage_name,
        "stage_run_id": stage_run_id,
        "source_stage_run_id": source_stage_run_id,
        "source_stage_package_path": source_stage_package_path,
    }
    write_json_atomic(package_manifest_path, package_manifest)
    return package_manifest


def read_stage_manifest_from_package(extracted_root: Path) -> Dict[str, Any]:
    """
    功能：从解压后的 stage package 中读取 stage_manifest。

    Read the stage_manifest.json file from an extracted stage package.

    Args:
        extracted_root: Extracted package root directory.

    Returns:
        Stage manifest mapping.
    """
    manifest_path = extracted_root / "artifacts" / "stage_manifest.json"
    manifest = read_json_dict(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"stage_manifest.json missing in extracted package: {manifest_path}")
    return manifest


def prepare_source_package(source_package_path: Path, runtime_state_root: Path) -> Dict[str, Any]:
    """
    功能：复制、校验并解压 source package。

    Copy, hash-verify, and extract one source stage package for downstream consumption.

    Args:
        source_package_path: Source ZIP path.
        runtime_state_root: Stage runtime-state root.

    Returns:
        Source package summary mapping.
    """
    package_root = ensure_directory(runtime_state_root / "source_package")
    local_package_path = package_root / source_package_path.name
    copy_file(source_package_path, local_package_path)
    package_sha256 = compute_file_sha256(local_package_path)
    extracted_root = package_root / "extracted"
    members = extract_zip_archive(local_package_path, extracted_root)
    stage_manifest = read_stage_manifest_from_package(extracted_root)
    return {
        "source_package_path": local_package_path.as_posix(),
        "source_package_sha256": package_sha256,
        "extracted_root": extracted_root.as_posix(),
        "members": members,
        "stage_manifest": stage_manifest,
    }


def copy_stage_manifest_snapshot(stage_manifest: Mapping[str, Any], destination_path: Path) -> Path:
    """
    功能：写出 source stage_manifest 快照。

    Persist a copied stage manifest snapshot for lineage auditing.

    Args:
        stage_manifest: Source stage manifest mapping.
        destination_path: Destination JSON path.

    Returns:
        Destination path.
    """
    write_json_atomic(destination_path, dict(stage_manifest))
    return destination_path
