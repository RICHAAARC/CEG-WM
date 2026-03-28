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
import re
import secrets
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
FORMAL_STAGE_PACKAGE_ROLE = "formal_stage_package"
FAILURE_DIAGNOSTICS_PACKAGE_ROLE = "failure_diagnostics_package"
FORMAL_PACKAGE_DISCOVERY_SCOPE = "discoverable_formal_only"
EXCLUDED_PACKAGE_DISCOVERY_SCOPE = "excluded_from_formal_discovery"
ATTESTATION_ENV_FILE_NAME = "attestation_env.json"
ATTESTATION_ENV_INFO_FILE_NAME = "attestation_env_info.json"
ATTESTATION_ENV_VAR_LENGTHS = {
    "k_master_env_var": 64,
    "k_prompt_env_var": 32,
    "k_seed_env_var": 32,
}


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


def _sanitize_identifier(value: str) -> str:
    """
    功能：将标识字符串规范化为安全片段。

    Normalize a free-form identifier into a filesystem-safe token.

    Args:
        value: Raw identifier string.

    Returns:
        Sanitized lowercase identifier token.
    """
    if not isinstance(value, str) or not value.strip():
        raise TypeError("value must be non-empty str")
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return normalized.lower() or "stage"


def make_stage_run_id(stage_name: Optional[str] = None) -> str:
    """
    功能：生成阶段运行唯一标识。

    Generate a stable stage run identifier based on UTC time and random suffix.

    Args:
        stage_name: Optional stable stage name.

    Returns:
        Stage run identifier string.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    random_suffix = uuid.uuid4().hex[:8]
    if isinstance(stage_name, str) and stage_name.strip():
        return f"{_sanitize_identifier(stage_name)}__{timestamp}__{random_suffix}"
    return f"stage__{timestamp}__{random_suffix}"


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


def resolve_attestation_env_var_names(cfg_obj: Mapping[str, Any]) -> Dict[str, str]:
    """
    功能：解析 attestation 配置中的环境变量名。 

    Resolve attestation environment-variable names from the runtime config.

    Args:
        cfg_obj: Runtime configuration mapping.

    Returns:
        Mapping from attestation config keys to environment-variable names.
    """
    attestation_node = cfg_obj.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    env_names: Dict[str, str] = {}
    for config_key in ATTESTATION_ENV_VAR_LENGTHS:
        raw_value = attestation_cfg.get(config_key)
        if isinstance(raw_value, str) and raw_value.strip():
            env_names[config_key] = raw_value.strip()
    return env_names


def _resolve_attestation_env_specs(
    cfg_obj: Mapping[str, Any],
    *,
    require_complete: bool,
) -> List[Dict[str, Any]]:
    """
    功能：解析 attestation secret 的规范集合。 

    Resolve the attestation secret specifications from the runtime config.

    Args:
        cfg_obj: Runtime configuration mapping.
        require_complete: Whether all configured attestation env vars must exist.

    Returns:
        Ordered attestation secret specification list.

    Raises:
        ValueError: If attestation is enabled and required env-var names are absent.
    """
    attestation_node = cfg_obj.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    attestation_enabled = bool(attestation_cfg.get("enabled", False))
    env_var_names = resolve_attestation_env_var_names(cfg_obj)
    specs: List[Dict[str, Any]] = []
    missing_config_keys: List[str] = []
    for config_key, hex_length in ATTESTATION_ENV_VAR_LENGTHS.items():
        env_name = env_var_names.get(config_key)
        if env_name is None:
            missing_config_keys.append(config_key)
            continue
        specs.append(
            {
                "config_key": config_key,
                "env_name": env_name,
                "hex_length": hex_length,
            }
        )
    if attestation_enabled and require_complete and missing_config_keys:
        raise ValueError(
            "attestation.enabled requires env-var bindings for all secret roles: "
            f"missing={missing_config_keys}"
        )
    return specs


def validate_attestation_hex_secret(secret_value: str, expected_hex_length: int, label: str) -> str:
    """
    功能：校验 attestation secret 的十六进制格式与长度。 

    Validate the format and length of one attestation secret.

    Args:
        secret_value: Candidate secret string.
        expected_hex_length: Required hexadecimal length.
        label: Human-readable label used in error reporting.

    Returns:
        Canonical lowercase secret string.

    Raises:
        ValueError: If the secret is not valid lowercase/uppercase hexadecimal with the required length.
    """
    if not secret_value.strip():
        raise ValueError(f"{label} must be non-empty hex string")
    if expected_hex_length <= 0:
        raise TypeError("expected_hex_length must be positive int")
    if not label:
        raise TypeError("label must be non-empty str")

    normalized_secret = secret_value.strip().lower()
    if len(normalized_secret) != expected_hex_length:
        raise ValueError(
            f"{label} must be exactly {expected_hex_length} hex chars: actual={len(normalized_secret)}"
        )
    if re.fullmatch(r"[0-9a-f]+", normalized_secret) is None:
        raise ValueError(f"{label} must contain only hexadecimal characters")
    return normalized_secret


def _mask_attestation_secret(secret_value: str) -> str:
    """
    功能：构造不泄漏真实值的 masked secret。 

    Build a masked representation for one attestation secret.

    Args:
        secret_value: Canonical secret string.

    Returns:
        Masked secret string.
    """
    if not secret_value:
        raise TypeError("secret_value must be non-empty str")
    if len(secret_value) <= 8:
        return "*" * len(secret_value)
    return f"{secret_value[:4]}...{secret_value[-4:]}"


def generate_attestation_session_env(cfg_obj: Mapping[str, Any]) -> Dict[str, str]:
    """
    功能：生成本次会话使用的临时 attestation secrets。 

    Generate session-scoped attestation secrets for the configured env vars.

    Args:
        cfg_obj: Runtime configuration mapping.

    Returns:
        Mapping from env-var name to generated secret value.
    """
    specs = _resolve_attestation_env_specs(cfg_obj, require_complete=True)
    generated_payload: Dict[str, str] = {}
    for spec in specs:
        env_name = str(spec["env_name"])
        hex_length = int(spec["hex_length"])
        generated_payload[env_name] = secrets.token_hex(hex_length // 2)
    return generated_payload


def _resolve_attestation_secret_paths(drive_project_root: Path) -> Dict[str, Path]:
    """
    功能：解析 attestation secret 与 masked info 的固定路径。 

    Resolve the canonical attestation secret and info paths under the Drive project root.

    Args:
        drive_project_root: Google Drive project root.

    Returns:
        Mapping with secrets_root, attestation_env_path, and attestation_env_info_path.
    """
    secrets_root = drive_project_root / "secrets"
    return {
        "secrets_root": secrets_root,
        "attestation_env_path": secrets_root / ATTESTATION_ENV_FILE_NAME,
        "attestation_env_info_path": secrets_root / ATTESTATION_ENV_INFO_FILE_NAME,
    }


def write_attestation_env_file(attestation_env_path: Path, env_payload: Mapping[str, Any]) -> Path:
    """
    功能：写出真实 attestation_env.json。 

    Persist the real attestation secret mapping to the canonical JSON file.

    Args:
        attestation_env_path: Destination JSON path.
        env_payload: Env-var to secret mapping.

    Returns:
        Written JSON path.
    """
    normalized_payload: Dict[str, str] = {}
    for env_name, secret_value in env_payload.items():
        if not isinstance(env_name, str) or not env_name:
            raise ValueError("attestation env payload keys must be non-empty str")
        if not isinstance(secret_value, str) or not secret_value:
            raise ValueError(f"attestation env payload value must be non-empty str: {env_name}")
        normalized_payload[env_name] = secret_value

    write_json_atomic(attestation_env_path, normalized_payload)
    return attestation_env_path


def write_masked_attestation_env_info(
    attestation_env_info_path: Path,
    cfg_obj: Mapping[str, Any],
    env_payload: Mapping[str, Any],
    *,
    status: str,
    reused_existing: bool,
    generated: bool,
    attestation_env_path: Optional[Path] = None,
) -> Path:
    """
    功能：写出不包含真实 secret 的 masked info 文件。 

    Persist the masked attestation info payload without leaking the real secret values.

    Args:
        attestation_env_info_path: Destination JSON path.
        cfg_obj: Runtime configuration mapping.
        env_payload: Valid env-var to secret mapping.
        status: Bootstrap status token.
        reused_existing: Whether the secret file was reused.
        generated: Whether the secret file was generated in the current session.
        attestation_env_path: Optional real secret-file path.

    Returns:
        Written info JSON path.
    """
    if not status:
        raise TypeError("status must be non-empty str")

    required_env_vars = [spec["env_name"] for spec in _resolve_attestation_env_specs(cfg_obj, require_complete=True)]
    masked_values: Dict[str, str] = {}
    for env_name in required_env_vars:
        secret_value = env_payload.get(env_name)
        if not isinstance(secret_value, str) or not secret_value:
            raise ValueError(f"masked attestation info requires non-empty secret for {env_name}")
        masked_values[str(env_name)] = _mask_attestation_secret(secret_value)

    attestation_node = cfg_obj.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    info_payload: Dict[str, Any] = {
        "enabled": bool(attestation_cfg.get("enabled", False)),
        "status": status,
        "generated": generated,
        "reused_existing": reused_existing,
        "attestation_env_path": normalize_path_value(attestation_env_path),
        "required_env_vars": required_env_vars,
        "masked_values": masked_values,
    }
    write_json_atomic(attestation_env_info_path, info_payload)
    return attestation_env_info_path


def restore_attestation_env_from_file(cfg_obj: Mapping[str, Any], attestation_env_path: Path) -> Dict[str, str]:
    """
    功能：从 secrets 文件恢复并注入 attestation 环境变量。 

    Restore attestation secrets from the canonical JSON file and inject them into os.environ.

    Args:
        cfg_obj: Runtime configuration mapping.
        attestation_env_path: Canonical secret-file path.

    Returns:
        Validated env-var to secret mapping.

    Raises:
        FileNotFoundError: If the secret file is missing.
        ValueError: If the secret file is malformed or fails validation.
    """
    if not attestation_env_path.exists() or not attestation_env_path.is_file():
        raise FileNotFoundError(f"attestation secret file not found: {normalize_path_value(attestation_env_path)}")

    payload = json.loads(attestation_env_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("attestation secret file must have a JSON object root")
    payload_obj = cast(Dict[str, Any], payload)

    restored_payload: Dict[str, str] = {}
    for spec in _resolve_attestation_env_specs(cfg_obj, require_complete=True):
        env_name = str(spec["env_name"])
        secret_value = payload_obj.get(env_name)
        validated_secret = validate_attestation_hex_secret(
            str(secret_value) if isinstance(secret_value, str) else "",
            int(spec["hex_length"]),
            env_name,
        )
        restored_payload[env_name] = validated_secret
        os.environ[env_name] = validated_secret
    return restored_payload


def ensure_attestation_env_bootstrap(
    cfg_obj: Mapping[str, Any],
    drive_project_root: Path,
    *,
    allow_generate: bool,
    allow_missing: bool = False,
) -> Dict[str, Any]:
    """
    功能：复用或引导 attestation secret，并返回 masked summary。 

    Reuse or bootstrap attestation secrets, inject them into os.environ, and return a masked summary.

    Args:
        cfg_obj: Runtime configuration mapping.
        drive_project_root: Google Drive project root.
        allow_generate: Whether missing secrets may be generated for the current session.
        allow_missing: Whether a missing secret file should return a non-blocking summary.

    Returns:
        Masked attestation environment summary suitable for notebooks and stage wrappers.

    Raises:
        FileNotFoundError: If secrets are missing and generation is not allowed.
        ValueError: If the existing secret file is malformed.
    """
    attestation_node = cfg_obj.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    attestation_enabled = bool(attestation_cfg.get("enabled", False))
    env_names = resolve_attestation_env_var_names(cfg_obj)
    path_mapping = _resolve_attestation_secret_paths(drive_project_root)
    attestation_env_path = path_mapping["attestation_env_path"]
    attestation_env_info_path = path_mapping["attestation_env_info_path"]

    summary: Dict[str, Any] = {
        "enabled": attestation_enabled,
        "status": "disabled" if not attestation_enabled else "pending",
        "generated": False,
        "reused_existing": False,
        "required_env_vars": list(env_names.values()),
        "present_env_vars": [],
        "missing_env_vars": list(env_names.values()),
        "masked_values": {},
        "attestation_env_path": normalize_path_value(attestation_env_path),
        "attestation_env_info_path": normalize_path_value(attestation_env_info_path),
    }
    if not attestation_enabled:
        return summary

    env_payload: Dict[str, str]
    if attestation_env_path.exists():
        env_payload = restore_attestation_env_from_file(cfg_obj, attestation_env_path)
        summary["status"] = "reused"
        summary["reused_existing"] = True
    elif allow_generate:
        env_payload = generate_attestation_session_env(cfg_obj)
        ensure_directory(path_mapping["secrets_root"])
        write_attestation_env_file(attestation_env_path, env_payload)
        for env_name, secret_value in env_payload.items():
            os.environ[env_name] = secret_value
        summary["status"] = "generated"
        summary["generated"] = True
    elif allow_missing:
        summary["status"] = "missing"
        return summary
    else:
        raise FileNotFoundError(
            f"attestation secret file not found: {normalize_path_value(attestation_env_path)}"
        )

    write_masked_attestation_env_info(
        attestation_env_info_path,
        cfg_obj,
        env_payload,
        status=str(summary["status"]),
        reused_existing=bool(summary["reused_existing"]),
        generated=bool(summary["generated"]),
        attestation_env_path=attestation_env_path,
    )

    final_summary = collect_attestation_env_summary(cfg_obj)
    final_summary.update(
        {
            "status": summary["status"],
            "generated": summary["generated"],
            "reused_existing": summary["reused_existing"],
            "masked_values": {
                env_name: _mask_attestation_secret(secret_value)
                for env_name, secret_value in env_payload.items()
            },
            "attestation_env_path": normalize_path_value(attestation_env_path),
            "attestation_env_info_path": normalize_path_value(attestation_env_info_path),
        }
    )
    return final_summary


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


def compute_mapping_sha256(payload: Mapping[str, Any]) -> str:
    """
    功能：计算 JSON 映射对象的规范 SHA256。

    Compute a canonical SHA256 digest for one JSON-like mapping.

    Args:
        payload: Mapping payload.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be Mapping")
    serialized = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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


def resolve_model_identity(cfg_obj: Mapping[str, Any]) -> Dict[str, Any]:
    """
    功能：解析配置中的模型标识与 revision。

    Resolve the effective model identifier and revision from the runtime config.

    Args:
        cfg_obj: Runtime configuration mapping.

    Returns:
        Mapping with model_id and revision fields.
    """
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")
    model_node = cfg_obj.get("model")
    model_cfg = cast(Dict[str, Any], model_node) if isinstance(model_node, dict) else {}
    model_id = model_cfg.get("model_id") if isinstance(model_cfg.get("model_id"), str) else cfg_obj.get("model_id")
    revision = model_cfg.get("revision") if isinstance(model_cfg.get("revision"), str) else cfg_obj.get("hf_revision")
    return {
        "model_id": model_id if isinstance(model_id, str) and model_id.strip() else "<absent>",
        "revision": revision if isinstance(revision, str) and revision.strip() else "<absent>",
    }


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


def build_failure_diagnostics_filename(stage_name: str, stage_run_id: str) -> str:
    """
    功能：构造 failure diagnostics ZIP 文件名。

    Build the ZIP filename used by one failure-diagnostics package.

    Args:
        stage_name: Stable stage name.
        stage_run_id: Current stage run identifier.

    Returns:
        Failure-diagnostics ZIP file name.
    """
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(stage_run_id, str) or not stage_run_id:
        raise TypeError("stage_run_id must be non-empty str")
    return f"{stage_name}__{stage_run_id}__failure_diagnostics.zip"


def _is_discoverable_formal_package_manifest(package_manifest: Mapping[str, Any]) -> bool:
    """
    功能：判断 manifest 是否属于可发现的正式 stage package。

    Determine whether a manifest belongs to a discoverable formal stage
    package.

    Args:
        package_manifest: Candidate package manifest mapping.

    Returns:
        True when the manifest represents a discoverable formal package.
    """
    if not isinstance(package_manifest, Mapping) or not package_manifest:
        return False

    stage_name = package_manifest.get("stage_name")
    stage_run_id = package_manifest.get("stage_run_id")
    package_filename = package_manifest.get("package_filename")
    if not isinstance(stage_name, str) or not stage_name:
        return False
    if not isinstance(stage_run_id, str) or not stage_run_id:
        return False
    if not isinstance(package_filename, str) or not package_filename.endswith(".zip"):
        return False

    package_role = package_manifest.get("package_role")
    if package_role not in {None, FORMAL_STAGE_PACKAGE_ROLE}:
        return False
    package_discovery_scope = package_manifest.get("package_discovery_scope")
    if package_discovery_scope not in {None, FORMAL_PACKAGE_DISCOVERY_SCOPE}:
        return False
    return True


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
    env_names = list(resolve_attestation_env_var_names(cfg_obj).values())
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
    model_identity = resolve_model_identity(cfg_obj)
    summary: Dict[str, Any] = {
        "model_id": model_identity["model_id"],
        "model_source": cfg_obj.get("model_source"),
        "hf_revision": model_identity["revision"],
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


def build_directory_digest_summary(directory_path: Path, max_entries: int = 24) -> Dict[str, Any]:
    """
    功能：构建目录级清单摘要。

    Build a concise digest summary for a directory tree.

    Args:
        directory_path: Directory path to summarize.
        max_entries: Maximum number of file entries to include.

    Returns:
        Directory digest summary mapping.
    """
    if not isinstance(directory_path, Path):
        raise TypeError("directory_path must be Path")
    if not isinstance(max_entries, int) or max_entries <= 0:
        raise TypeError("max_entries must be positive int")

    summary: Dict[str, Any] = {
        "root_path": normalize_path_value(directory_path),
        "exists": bool(directory_path.exists() and directory_path.is_dir()),
        "file_count": 0,
        "entries": [],
        "manifest_sha256": "<absent>",
    }
    if not directory_path.exists() or not directory_path.is_dir():
        return summary

    entries: List[Dict[str, Any]] = []
    for file_path in sorted(directory_path.rglob("*")):
        if not file_path.is_file():
            continue
        entries.append(
            {
                "relative_path": file_path.relative_to(directory_path).as_posix(),
                "size_bytes": int(file_path.stat().st_size),
                "sha256": compute_file_sha256(file_path),
            }
        )
    summary["file_count"] = len(entries)
    summary["entries"] = entries[:max_entries]
    summary["manifest_sha256"] = hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
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


def resolve_export_package_manifest_path(export_root: Path) -> Path:
    """
    功能：返回 stage export 目录中的外部 package_manifest 路径。

    Return the canonical external package_manifest path for one export directory.

    Args:
        export_root: Stage export directory.

    Returns:
        External package_manifest path.
    """
    if not isinstance(export_root, Path):
        raise TypeError("export_root must be Path")
    return export_root / "package_manifest.json"


def resolve_export_package_index_path(export_root: Path) -> Path:
    """
    功能：返回 stage export 目录中的外部 package_index 路径。

    Return the canonical external package_index path for one export directory.

    Args:
        export_root: Stage export directory.

    Returns:
        External package_index path.
    """
    if not isinstance(export_root, Path):
        raise TypeError("export_root must be Path")
    return export_root / "package_index.json"


def read_json_from_zip(zip_path: Path, member_name: str) -> Dict[str, Any]:
    """
    功能：从 ZIP 中读取 JSON 对象。

    Read one JSON object member from a ZIP archive.

    Args:
        zip_path: ZIP archive path.
        member_name: Member path inside the archive.

    Returns:
        Parsed mapping or an empty mapping when unavailable.
    """
    if not isinstance(zip_path, Path):
        raise TypeError("zip_path must be Path")
    if not isinstance(member_name, str) or not member_name:
        raise TypeError("member_name must be non-empty str")
    if not zip_path.exists() or not zip_path.is_file():
        return {}
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            with archive.open(member_name, "r") as handle:
                payload = json.loads(handle.read().decode("utf-8"))
    except Exception:
        return {}
    return cast(Dict[str, Any], payload) if isinstance(payload, dict) else {}


def find_external_package_manifest(source_package_path: Path) -> Path | None:
    """
    功能：定位 ZIP 包同目录的外部 package_manifest。

    Locate the external package_manifest stored alongside a stage package ZIP.

    Args:
        source_package_path: Stage package ZIP path.

    Returns:
        External package_manifest path when present.
    """
    if not isinstance(source_package_path, Path):
        raise TypeError("source_package_path must be Path")
    candidate_paths = [
        source_package_path.parent / "package_manifest.json",
        source_package_path.with_suffix(f"{source_package_path.suffix}.package_manifest.json"),
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path
    return None


def validate_package_manifest_binding(
    package_path: Path,
    package_manifest: Mapping[str, Any],
    *,
    required_sha_match: bool,
) -> Dict[str, Any]:
    """
    功能：校验 package_manifest 与实际 ZIP 的绑定关系。

    Validate one package manifest against the actual ZIP file.

    Args:
        package_path: Stage package ZIP path.
        package_manifest: Candidate package manifest mapping.
        required_sha_match: Whether package_sha256 must match the ZIP digest.

    Returns:
        Validation summary mapping.
    """
    if not isinstance(package_path, Path):
        raise TypeError("package_path must be Path")
    if not isinstance(package_manifest, Mapping):
        raise TypeError("package_manifest must be Mapping")

    actual_sha256 = compute_file_sha256(package_path)
    manifest_package_sha256 = package_manifest.get("package_sha256")
    manifest_package_filename = package_manifest.get("package_filename")
    result = {
        "actual_sha256": actual_sha256,
        "manifest_package_sha256": manifest_package_sha256,
        "manifest_package_filename": manifest_package_filename,
        "sha256_match": bool(isinstance(manifest_package_sha256, str) and manifest_package_sha256 == actual_sha256),
        "filename_match": bool(isinstance(manifest_package_filename, str) and manifest_package_filename == package_path.name),
        "manifest_digest": compute_mapping_sha256(package_manifest),
    }
    if required_sha_match and not bool(result["sha256_match"]):
        raise ValueError(
            f"package sha256 mismatch: expected={manifest_package_sha256} actual={actual_sha256} path={package_path}"
        )
    if isinstance(manifest_package_filename, str) and manifest_package_filename and manifest_package_filename != package_path.name:
        raise ValueError(
            f"package filename mismatch: manifest={manifest_package_filename} actual={package_path.name}"
        )
    return result


def discover_stage_packages(export_stage_root: Path) -> List[Dict[str, Any]]:
    """
    功能：递归发现 stage export 目录下的候选 package。

    Recursively discover candidate stage packages under one export root.

    Args:
        export_stage_root: Stage export root directory.

    Returns:
        Candidate package summary list.
    """
    if not isinstance(export_stage_root, Path):
        raise TypeError("export_stage_root must be Path")
    if not export_stage_root.exists() or not export_stage_root.is_dir():
        return []

    candidates: List[Dict[str, Any]] = []
    for zip_path in sorted(export_stage_root.rglob("*.zip")):
        if not zip_path.is_file():
            continue
        external_manifest_path = find_external_package_manifest(zip_path)
        external_manifest = read_json_dict(external_manifest_path) if isinstance(external_manifest_path, Path) else {}
        internal_manifest = read_json_from_zip(zip_path, "artifacts/package_manifest.json")
        stage_manifest = read_json_from_zip(zip_path, "artifacts/stage_manifest.json")
        manifest_for_sort = external_manifest if external_manifest else internal_manifest
        if not _is_discoverable_formal_package_manifest(manifest_for_sort):
            continue
        package_created_at = manifest_for_sort.get("package_created_at") if isinstance(manifest_for_sort, dict) else None
        stage_run_id = manifest_for_sort.get("stage_run_id") if isinstance(manifest_for_sort, dict) else None
        validation_error = None
        validation_summary: Dict[str, Any] = {}
        try:
            if external_manifest:
                validation_summary = validate_package_manifest_binding(zip_path, external_manifest, required_sha_match=True)
            elif internal_manifest:
                validation_summary = validate_package_manifest_binding(zip_path, internal_manifest, required_sha_match=False)
        except Exception as exc:
            validation_error = str(exc)
        candidates.append(
            {
                "package_path": zip_path.as_posix(),
                "package_filename": zip_path.name,
                "package_created_at": package_created_at or utc_now_iso(),
                "stage_run_id": stage_run_id or stage_manifest.get("stage_run_id") or "<absent>",
                "external_manifest_path": normalize_path_value(external_manifest_path) if external_manifest_path else "<absent>",
                "external_manifest": external_manifest,
                "internal_manifest": internal_manifest,
                "stage_manifest": stage_manifest,
                "package_role": manifest_for_sort.get("package_role", FORMAL_STAGE_PACKAGE_ROLE),
                "package_discovery_scope": manifest_for_sort.get(
                    "package_discovery_scope",
                    FORMAL_PACKAGE_DISCOVERY_SCOPE,
                ),
                "validation": validation_summary,
                "validation_error": validation_error,
                "mtime": datetime.fromtimestamp(zip_path.stat().st_mtime, timezone.utc).isoformat(),
            }
        )
    candidates.sort(
        key=lambda item: (
            item.get("package_created_at") or "",
            item.get("stage_run_id") or "",
            item.get("mtime") or "",
        ),
        reverse=True,
    )
    return candidates


def select_latest_stage_package(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    功能：从候选 package 列表中选择最新有效项。

    Select the latest valid stage package from discovered candidates.

    Args:
        candidates: Candidate package mappings.

    Returns:
        Selected candidate mapping.
    """
    if not isinstance(candidates, Sequence) or not candidates:
        raise FileNotFoundError("no stage package candidates available")

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("validation_error"):
            continue
        validation = candidate.get("validation")
        if isinstance(validation, Mapping) and validation.get("filename_match"):
            selection = dict(candidate)
            selection["selection_reason"] = (
                "selected the latest candidate by package_created_at/stage_run_id with a valid manifest-to-zip binding"
            )
            return selection
    raise FileNotFoundError("no valid stage package candidate passed manifest binding checks")


def tail_text_file(path_obj: Path, max_lines: int = 20) -> List[str]:
    """
    功能：返回文本文件尾部若干行。

    Return the last N lines of one text file.

    Args:
        path_obj: Text file path.
        max_lines: Maximum number of lines.

    Returns:
        Tail lines list.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not isinstance(max_lines, int) or max_lines <= 0:
        raise TypeError("max_lines must be positive int")
    if not path_obj.exists() or not path_obj.is_file():
        return []
    lines = path_obj.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def collect_missing_file_entries(path_mapping: Mapping[str, Path]) -> List[str]:
    """
    功能：收集缺失文件标签列表。

    Collect labels whose mapped files are missing.

    Args:
        path_mapping: Label to Path mapping.

    Returns:
        Missing label list.
    """
    if not isinstance(path_mapping, Mapping):
        raise TypeError("path_mapping must be Mapping")
    return [str(label) for label, path_obj in path_mapping.items() if not isinstance(path_obj, Path) or not path_obj.exists()]


def summarize_manifest_fields(manifest: Mapping[str, Any], field_names: Sequence[str]) -> Dict[str, Any]:
    """
    功能：提取 manifest 关键字段摘要。

    Extract a small manifest summary using selected field names.

    Args:
        manifest: Manifest mapping.
        field_names: Selected field names.

    Returns:
        Manifest summary mapping.
    """
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be Mapping")
    if not isinstance(field_names, Sequence):
        raise TypeError("field_names must be Sequence")
    return {str(field_name): manifest.get(str(field_name)) for field_name in field_names}


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
    package_role: str = FORMAL_STAGE_PACKAGE_ROLE,
    package_discovery_scope: str = FORMAL_PACKAGE_DISCOVERY_SCOPE,
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
    if not isinstance(package_role, str) or not package_role:
        raise TypeError("package_role must be non-empty str")
    if not isinstance(package_discovery_scope, str) or not package_discovery_scope:
        raise TypeError("package_discovery_scope must be non-empty str")

    package_index_path = package_root / "artifacts" / "package_index.json"
    package_manifest_internal_path = package_root / "artifacts" / "package_manifest.json"
    package_index = build_package_index(package_root.rglob("*"), package_root)
    write_json_atomic(package_index_path, package_index)

    package_filename = build_stage_package_filename(stage_name, stage_run_id, source_stage_run_id)
    internal_manifest = {
        "package_manifest_version": "v2",
        "stage_name": stage_name,
        "stage_run_id": stage_run_id,
        "source_stage_run_id": source_stage_run_id,
        "source_stage_package_path": source_stage_package_path,
        "package_filename": package_filename,
        "package_path": "<external_zip>",
        "package_sha256": "<see_external_manifest>",
        "package_created_at": utc_now_iso(),
        "package_contents_index_path": "artifacts/package_index.json",
        "package_manifest_scope": "internal_copy",
        "package_role": package_role,
        "package_discovery_scope": package_discovery_scope,
    }
    write_json_atomic(package_manifest_internal_path, internal_manifest)

    package_path = create_zip_archive_from_directory(package_root, export_root / package_filename)
    package_manifest = {
        "package_manifest_version": "v2",
        "package_path": package_path.as_posix(),
        "package_filename": package_filename,
        "package_sha256": compute_file_sha256(package_path),
        "package_created_at": utc_now_iso(),
        "package_contents_index_path": resolve_export_package_index_path(export_root).as_posix(),
        "stage_name": stage_name,
        "stage_run_id": stage_run_id,
        "source_stage_run_id": source_stage_run_id,
        "source_stage_package_path": source_stage_package_path,
        "package_role": package_role,
        "package_discovery_scope": package_discovery_scope,
        "package_manifest_digest": "<pending>",
    }
    package_manifest["package_manifest_digest"] = compute_mapping_sha256(package_manifest)
    write_json_atomic(package_manifest_path, package_manifest)
    write_json_atomic(resolve_export_package_manifest_path(export_root), package_manifest)
    write_json_atomic(resolve_export_package_index_path(export_root), package_index)
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
    external_manifest_source_path = find_external_package_manifest(source_package_path)
    local_external_manifest_path = package_root / "package_manifest.external.json"
    external_package_manifest: Dict[str, Any] = {}
    if isinstance(external_manifest_source_path, Path):
        copy_file(external_manifest_source_path, local_external_manifest_path)
        external_package_manifest = read_json_dict(local_external_manifest_path)
        validate_package_manifest_binding(local_package_path, external_package_manifest, required_sha_match=True)
        if not _is_discoverable_formal_package_manifest(external_package_manifest):
            raise ValueError(
                "source package must be a discoverable formal stage package: "
                f"path={normalize_path_value(source_package_path)}"
            )
    extracted_root = package_root / "extracted"
    members = extract_zip_archive(local_package_path, extracted_root)
    stage_manifest = read_stage_manifest_from_package(extracted_root)
    internal_package_manifest = read_json_dict(extracted_root / "artifacts" / "package_manifest.json")
    if not internal_package_manifest:
        raise FileNotFoundError("package_manifest.json missing in extracted package")
    package_index = read_json_dict(extracted_root / "artifacts" / "package_index.json")
    package_manifest_for_lineage = external_package_manifest if external_package_manifest else internal_package_manifest
    if not _is_discoverable_formal_package_manifest(package_manifest_for_lineage):
        raise ValueError(
            "source package must be a discoverable formal stage package: "
            f"path={normalize_path_value(source_package_path)}"
        )
    if package_manifest_for_lineage.get("stage_name") not in {None, stage_manifest.get("stage_name")}:
        raise ValueError("source package manifest stage_name does not match source stage_manifest")
    if package_manifest_for_lineage.get("stage_run_id") not in {None, stage_manifest.get("stage_run_id")}:
        raise ValueError("source package manifest stage_run_id does not match source stage_manifest")
    return {
        "source_package_path": local_package_path.as_posix(),
        "source_package_sha256": package_sha256,
        "extracted_root": extracted_root.as_posix(),
        "members": members,
        "stage_manifest": stage_manifest,
        "external_package_manifest_path": normalize_path_value(local_external_manifest_path) if external_package_manifest else "<absent>",
        "external_package_manifest": external_package_manifest,
        "internal_package_manifest": internal_package_manifest,
        "package_manifest": package_manifest_for_lineage,
        "package_manifest_digest": compute_mapping_sha256(package_manifest_for_lineage),
        "package_index": package_index,
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
