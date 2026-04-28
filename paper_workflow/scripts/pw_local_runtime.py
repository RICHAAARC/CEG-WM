"""
File purpose: Manage notebook-local runtime staging and Google Drive bundle exchange for paper_workflow notebooks.
Module type: General module

职责边界：
1. 仅负责 notebook 级本地 runtime 清理、依赖 bundle 拉取、完整性校验、解压与归档。
2. 不承载 PW00–PW05 的真实方法逻辑，也不修改 main/ 下算法语义。
3. 使用独立 bundle sidecar，避免并发会话写全局索引文件。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Sequence, cast

from scripts.notebook_runtime_common import (
    ensure_directory,
    normalize_path_value,
    utc_now_iso,
    validate_path_within_base,
    write_json_atomic,
)


SCHEMA_VERSION = "pw_bundle_v1"
PW00_STAGE_NAME = "PW00_Paper_Eval_Family_Manifest"
PW01_STAGE_NAME = "PW01_Source_Event_Shards"
PW02_STAGE_NAME = "PW02_Source_Merge_And_Global_Thresholds"
PW03_STAGE_NAME = "PW03_Attack_Event_Shards"
PW04_PREPARE_STAGE_NAME = "PW04_Attack_Merge_And_Metrics_prepare"
PW04_QUALITY_STAGE_NAME = "PW04_Attack_Merge_And_Metrics_quality_shard"
PW04_FINALIZE_STAGE_NAME = "PW04_Attack_Merge_And_Metrics_finalize"
PW05_STAGE_NAME = "PW05_Release_And_Signoff"
POSITIVE_SOURCE_ROLE = "positive_source"
CLEAN_NEGATIVE_ROLE = "clean_negative"
CONTROL_NEGATIVE_ROLE = "planner_conditioned_control_negative"
SOURCE_ROLE_DIRECTORY_NAMES = {
    POSITIVE_SOURCE_ROLE: "positive",
    CLEAN_NEGATIVE_ROLE: "negative",
    CONTROL_NEGATIVE_ROLE: "control_negative",
}
SOURCE_ROLE_BUNDLE_TOKENS = {
    POSITIVE_SOURCE_ROLE: "positive_source",
    CLEAN_NEGATIVE_ROLE: "clean_negative",
    CONTROL_NEGATIVE_ROLE: "control_negative",
}
STAGE_DIRECTORY_NAMES = {
    PW00_STAGE_NAME: "pw00_bootstrap",
    PW01_STAGE_NAME: "pw01_source",
    PW02_STAGE_NAME: "pw02_source_merge",
    PW03_STAGE_NAME: "pw03_attack",
    PW04_PREPARE_STAGE_NAME: "pw04_prepare",
    PW04_QUALITY_STAGE_NAME: "pw04_quality",
    PW04_FINALIZE_STAGE_NAME: "pw04_finalize",
    PW05_STAGE_NAME: "pw05_release",
}
NEXT_STAGE_HINTS = {
    PW00_STAGE_NAME: "PW01_Source_Event_Shards",
    PW01_STAGE_NAME: "PW02_Source_Merge_And_Global_Thresholds",
    PW02_STAGE_NAME: "PW03_Attack_Event_Shards",
    PW03_STAGE_NAME: "PW04_Attack_Merge_And_Metrics_prepare",
    PW04_PREPARE_STAGE_NAME: "PW04_Attack_Merge_And_Metrics_quality_shard",
    PW04_QUALITY_STAGE_NAME: "PW04_Attack_Merge_And_Metrics_finalize",
    PW04_FINALIZE_STAGE_NAME: "PW05_Release_And_Signoff",
    PW05_STAGE_NAME: None,
}
CONTENT_ROOT_POSIX = PurePosixPath("/content")
PROTECTED_LOCAL_RUNTIME_PATHS = [
    PurePosixPath("/content/drive"),
    PurePosixPath("/content/sample_data"),
    PurePosixPath("/content/ceg_wm_workspace"),
    PurePosixPath("/content/ceg_wm_workspace/huggingface_cache"),
]
QUALITY_SHARD_PATTERN = re.compile(r"quality_shard_(?P<index>\d+)\.json$")


def _emit_runtime_summary(title: str, payload: Mapping[str, Any]) -> None:
    """
    功能：打印 notebook 友好的 JSON 摘要。

    Print one notebook-friendly JSON summary.

    Args:
        title: Summary title.
        payload: Summary payload.

    Returns:
        None.
    """
    if not isinstance(title, str) or not title:
        raise TypeError("title must be non-empty str")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be Mapping")
    print(f"\n[{title}]")
    print(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True))


def _as_posix_path(path_obj: Path) -> str:
    """
    功能：保留输入路径的 POSIX 表示。

    Preserve the input path as a POSIX string without resolving drive-specific prefixes.

    Args:
        path_obj: Path object.

    Returns:
        POSIX path string.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    return path_obj.as_posix()


def _normalize_relative_path(relative_path: str) -> str:
    """
    功能：规范化并校验相对路径。

    Normalize and validate one relative path.

    Args:
        relative_path: Candidate relative path.

    Returns:
        Normalized POSIX relative path.
    """
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise TypeError("relative_path must be non-empty str")
    candidate = PurePosixPath(relative_path.replace("\\", "/").strip())
    if candidate.is_absolute():
        raise ValueError(f"relative_path must not be absolute: {relative_path}")
    if not candidate.parts:
        raise ValueError("relative_path must not be empty")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"relative_path must not contain path escape: {relative_path}")
    return candidate.as_posix()


def _relative_path_to_absolute(local_project_root: Path, relative_path: str) -> Path:
    """
    功能：把相对路径映射到 local_project_root 下的绝对路径。

    Resolve one relative path under the local project root.

    Args:
        local_project_root: Local runtime project root.
        relative_path: Normalized relative path.

    Returns:
        Resolved absolute path under the local project root.
    """
    if not isinstance(local_project_root, Path):
        raise TypeError("local_project_root must be Path")
    normalized_relative_path = _normalize_relative_path(relative_path)
    candidate_path = local_project_root / Path(*PurePosixPath(normalized_relative_path).parts)
    validate_path_within_base(local_project_root, candidate_path, "local runtime relative path")
    return candidate_path


def _normalize_relative_paths(relative_paths: Sequence[str]) -> List[str]:
    """
    功能：规范化相对路径列表并去重。

    Normalize and deduplicate one relative-path list.

    Args:
        relative_paths: Candidate relative paths.

    Returns:
        Ordered normalized relative paths.
    """
    if not isinstance(relative_paths, Sequence) or isinstance(relative_paths, (str, bytes)):
        raise TypeError("relative_paths must be Sequence[str]")
    normalized: List[str] = []
    for item in relative_paths:
        normalized_item = _normalize_relative_path(str(item))
        if normalized_item not in normalized:
            normalized.append(normalized_item)
    if not normalized:
        raise ValueError("relative_paths must not be empty")
    return normalized


def _load_json_dict(path_obj: Path, label: str) -> Dict[str, Any]:
    """
    功能：读取一个 JSON 对象文件。

    Load one JSON object file.

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
        raise FileNotFoundError(f"{label} not found: {path_obj}")
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be JSON object: {path_obj}")
    return cast(Dict[str, Any], payload)


def _read_optional_json_dict(path_obj: Path) -> Dict[str, Any] | None:
    """
    功能：读取可选 JSON 对象文件。

    Read one optional JSON object file when available.

    Args:
        path_obj: JSON file path.

    Returns:
        Parsed JSON mapping or None.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not path_obj.exists() or not path_obj.is_file():
        return None
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be object: {path_obj}")
    return cast(Dict[str, Any], payload)


def _extract_int(value: Any) -> int | None:
    """
    功能：从松散标量中提取正整数。

    Extract one positive integer from a loosely typed scalar.

    Args:
        value: Candidate scalar value.

    Returns:
        Positive integer or None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value) if value > 0 else None
    if isinstance(value, float) and float(value).is_integer():
        return int(value) if int(value) > 0 else None
    if isinstance(value, str) and value.strip():
        try:
            parsed_value = int(value.strip())
        except ValueError:
            return None
        return parsed_value if parsed_value > 0 else None
    return None


def _parse_quality_shard_index(path_text: str) -> int | None:
    """
    功能：从 quality shard 文件名中解析 shard index。

    Parse the quality-shard index from one file name.

    Args:
        path_text: Candidate path or file name.

    Returns:
        Parsed shard index or None.
    """
    if not isinstance(path_text, str) or not path_text.strip():
        return None
    match = QUALITY_SHARD_PATTERN.search(Path(path_text.strip()).name)
    if match is None:
        return None
    return int(match.group("index"))


def _family_root(local_project_root: Path, family_id: str) -> Path:
    """
    功能：返回 family 根目录。

    Resolve the family root under the local runtime project root.

    Args:
        local_project_root: Local runtime project root.
        family_id: Family identifier.

    Returns:
        Family root path.
    """
    if not isinstance(local_project_root, Path):
        raise TypeError("local_project_root must be Path")
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    return local_project_root / "paper_workflow" / "families" / family_id.strip()


def _stage_paths(local_project_root: Path, family_id: str) -> Dict[str, Path]:
    """
    功能：构造常用 stage 路径集合。

    Build common stage paths for one family.

    Args:
        local_project_root: Local runtime project root.
        family_id: Family identifier.

    Returns:
        Mapping of commonly used stage paths.
    """
    family_root = _family_root(local_project_root, family_id)
    return {
        "family_root": family_root,
        "manifests_root": family_root / "manifests",
        "snapshots_root": family_root / "snapshots",
        "runtime_state_root": family_root / "runtime_state",
        "exports_pw02_root": family_root / "exports" / "pw02",
        "exports_pw04_root": family_root / "exports" / "pw04",
        "quality_root": family_root / "exports" / "pw04" / "quality",
        "source_shards_root": family_root / "source_shards",
        "attack_shards_root": family_root / "attack_shards",
    }


def _bundle_paths(drive_bundle_root: Path, family_id: str, stage_name: str, bundle_base_name: str) -> Dict[str, Path]:
    """
    功能：构造 bundle 与 sidecar 路径。

    Build the bundle tar and sidecar paths.

    Args:
        drive_bundle_root: Drive bundle root.
        family_id: Family identifier.
        stage_name: Canonical stage name.
        bundle_base_name: Bundle base file name without suffix.

    Returns:
        Mapping with stage_root, tar_path, and sidecar_path.
    """
    if not isinstance(drive_bundle_root, Path):
        raise TypeError("drive_bundle_root must be Path")
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if stage_name not in STAGE_DIRECTORY_NAMES:
        raise ValueError(f"unsupported stage_name: {stage_name}")
    if not isinstance(bundle_base_name, str) or not bundle_base_name:
        raise TypeError("bundle_base_name must be non-empty str")
    stage_root = drive_bundle_root / family_id.strip() / STAGE_DIRECTORY_NAMES[stage_name]
    return {
        "stage_root": stage_root,
        "tar_path": stage_root / f"{bundle_base_name}.tar.gz",
        "sidecar_path": stage_root / f"{bundle_base_name}.bundle.json",
    }


def _bundle_base_name(
    *,
    stage_name: str,
    sample_role: str | None = None,
    shard_index: int | None = None,
) -> str:
    """
    功能：解析某个 stage 的 bundle 基础文件名。

    Resolve the bundle base file name for one stage.

    Args:
        stage_name: Canonical stage name.
        sample_role: Optional sample role for PW01.
        shard_index: Optional shard index.

    Returns:
        Bundle base file name without suffix.
    """
    if stage_name == PW00_STAGE_NAME:
        return "pw00_bootstrap"
    if stage_name == PW01_STAGE_NAME:
        if sample_role not in SOURCE_ROLE_BUNDLE_TOKENS:
            raise ValueError("PW01 bundle requires sample_role")
        if not isinstance(shard_index, int) or shard_index < 0:
            raise ValueError("PW01 bundle requires non-negative shard_index")
        return f"{SOURCE_ROLE_BUNDLE_TOKENS[sample_role]}__shard_{shard_index:04d}"
    if stage_name == PW02_STAGE_NAME:
        return "pw02_source_merge"
    if stage_name == PW03_STAGE_NAME:
        if not isinstance(shard_index, int) or shard_index < 0:
            raise ValueError("PW03 bundle requires non-negative shard_index")
        return f"attack_shard_{shard_index:04d}"
    if stage_name == PW04_PREPARE_STAGE_NAME:
        return "pw04_prepare"
    if stage_name == PW04_QUALITY_STAGE_NAME:
        if not isinstance(shard_index, int) or shard_index < 0:
            raise ValueError("PW04 quality bundle requires non-negative shard_index")
        return f"quality_shard_{shard_index:04d}"
    if stage_name == PW04_FINALIZE_STAGE_NAME:
        return "pw04_finalize"
    if stage_name == PW05_STAGE_NAME:
        return "pw05_release_signoff"
    raise ValueError(f"unsupported stage_name: {stage_name}")


def _bundle_kind(stage_name: str) -> str:
    """
    功能：返回 stage 对应的 bundle kind。

    Return the bundle-kind token for one stage.

    Args:
        stage_name: Canonical stage name.

    Returns:
        Bundle-kind token.
    """
    mapping = {
        PW00_STAGE_NAME: "pw00_bootstrap",
        PW01_STAGE_NAME: "pw01_source_shard",
        PW02_STAGE_NAME: "pw02_source_merge",
        PW03_STAGE_NAME: "pw03_attack_shard",
        PW04_PREPARE_STAGE_NAME: "pw04_prepare",
        PW04_QUALITY_STAGE_NAME: "pw04_quality_shard",
        PW04_FINALIZE_STAGE_NAME: "pw04_finalize",
        PW05_STAGE_NAME: "pw05_release_signoff",
    }
    if stage_name not in mapping:
        raise ValueError(f"unsupported stage_name: {stage_name}")
    return mapping[stage_name]


def _build_bundle_reference(
    *,
    stage_name: str,
    family_id: str,
    drive_bundle_root: Path,
    sample_role: str | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> Dict[str, Any]:
    """
    功能：构造统一的 bundle 引用描述。

    Build a normalized bundle reference descriptor.

    Args:
        stage_name: Canonical stage name.
        family_id: Family identifier.
        drive_bundle_root: Drive bundle root.
        sample_role: Optional sample role.
        shard_index: Optional shard index.
        shard_count: Optional shard count.

    Returns:
        Bundle reference descriptor.
    """
    bundle_base_name = _bundle_base_name(
        stage_name=stage_name,
        sample_role=sample_role,
        shard_index=shard_index,
    )
    bundle_paths = _bundle_paths(drive_bundle_root, family_id, stage_name, bundle_base_name)
    return {
        "stage_name": stage_name,
        "bundle_kind": _bundle_kind(stage_name),
        "bundle_base_name": bundle_base_name,
        "bundle_id": f"{STAGE_DIRECTORY_NAMES[stage_name]}/{bundle_base_name}",
        "stage_directory": STAGE_DIRECTORY_NAMES[stage_name],
        "family_id": family_id,
        "sample_role": sample_role,
        "shard_index": shard_index,
        "shard_count": shard_count,
        **bundle_paths,
    }


def _source_shard_count_from_local_plan(family_root: Path, sample_role: str) -> int | None:
    """
    功能：从本地 source_shard_plan 中读取某个 role 的 shard 数。

    Read one source-role shard count from the local source shard plan.

    Args:
        family_root: Family root path.
        sample_role: Source sample role.

    Returns:
        Shard count or None.
    """
    source_shard_plan_path = family_root / "manifests" / "source_shard_plan.json"
    source_shard_plan = _read_optional_json_dict(source_shard_plan_path)
    if source_shard_plan is None:
        return None
    sample_role_plans = source_shard_plan.get("sample_role_plans")
    if isinstance(sample_role_plans, Mapping):
        role_plan = sample_role_plans.get(sample_role)
        if isinstance(role_plan, Mapping):
            shards_node = role_plan.get("shards")
            if isinstance(shards_node, list) and shards_node:
                return len(shards_node)
    return _extract_int(source_shard_plan.get("source_shard_count"))


def _attack_shard_count_from_local_plan(family_root: Path) -> int | None:
    """
    功能：从本地 attack_shard_plan 中读取 shard 数。

    Read the attack-shard count from the local attack shard plan.

    Args:
        family_root: Family root path.

    Returns:
        Shard count or None.
    """
    attack_shard_plan_path = family_root / "manifests" / "attack_shard_plan.json"
    attack_shard_plan = _read_optional_json_dict(attack_shard_plan_path)
    if attack_shard_plan is None:
        return None
    shard_count = _extract_int(attack_shard_plan.get("attack_shard_count"))
    if shard_count is not None:
        return shard_count
    shards_node = attack_shard_plan.get("shards")
    if isinstance(shards_node, list) and shards_node:
        return len(shards_node)
    return None


def _source_shard_count_from_pw00_summary(family_root: Path) -> int | None:
    """
    功能：从 pw00_summary 读取 source shard 数。

    Read the source-shard count from pw00_summary.json.

    Args:
        family_root: Family root path.

    Returns:
        Shard count or None.
    """
    pw00_summary = _read_optional_json_dict(family_root / "runtime_state" / "pw00_summary.json")
    if pw00_summary is None:
        return None
    return _extract_int(pw00_summary.get("source_shard_count"))


def _attack_shard_count_from_pw00_summary(family_root: Path) -> int | None:
    """
    功能：从 pw00_summary 读取 attack shard 数。

    Read the attack-shard count from pw00_summary.json.

    Args:
        family_root: Family root path.

    Returns:
        Shard count or None.
    """
    pw00_summary = _read_optional_json_dict(family_root / "runtime_state" / "pw00_summary.json")
    if pw00_summary is None:
        return None
    return _extract_int(pw00_summary.get("attack_shard_count"))


def _bootstrap_counts_from_sidecar(drive_bundle_root: Path, family_id: str) -> Dict[str, int | None]:
    """
    功能：从 pw00 bootstrap sidecar 读取冻结的 shard 数。

    Read frozen shard counts from the PW00 bootstrap sidecar.

    Args:
        drive_bundle_root: Drive bundle root.
        family_id: Family identifier.

    Returns:
        Mapping with optional source_shard_count and attack_shard_count.
    """
    bundle_reference = _build_bundle_reference(
        stage_name=PW00_STAGE_NAME,
        family_id=family_id,
        drive_bundle_root=drive_bundle_root,
    )
    sidecar_payload = _read_optional_json_dict(cast(Path, bundle_reference["sidecar_path"]))
    if sidecar_payload is None:
        return {"source_shard_count": None, "attack_shard_count": None}
    return {
        "source_shard_count": _extract_int(sidecar_payload.get("source_shard_count")),
        "attack_shard_count": _extract_int(sidecar_payload.get("attack_shard_count")),
    }


def _scan_shard_sidecars(
    *,
    drive_bundle_root: Path,
    family_id: str,
    stage_name: str,
    sample_role: str | None = None,
) -> Dict[str, Any] | None:
    """
    功能：扫描 Drive sidecar 以发现 shard 计划。

    Scan Drive bundle sidecars to discover shard expectations.

    Args:
        drive_bundle_root: Drive bundle root.
        family_id: Family identifier.
        stage_name: Shard-producing stage name.
        sample_role: Optional source sample role for PW01 bundles.

    Returns:
        Discovery payload or None when unavailable.
    """
    stage_root = drive_bundle_root / family_id / STAGE_DIRECTORY_NAMES[stage_name]
    if not stage_root.exists() or not stage_root.is_dir():
        return None

    payloads: List[Dict[str, Any]] = []
    for sidecar_path in sorted(stage_root.glob("*.bundle.json")):
        sidecar_payload = _read_optional_json_dict(sidecar_path)
        if sidecar_payload is None:
            continue
        if sidecar_payload.get("status") != "complete":
            continue
        if stage_name == PW01_STAGE_NAME and sample_role is not None:
            if sidecar_payload.get("sample_role") != sample_role:
                continue
        payloads.append(sidecar_payload)

    if not payloads:
        return None

    present_indices = sorted(
        {
            int(payload["shard_index"])
            for payload in payloads
            if isinstance(payload.get("shard_index"), int)
            and not isinstance(payload.get("shard_index"), bool)
            and int(payload["shard_index"]) >= 0
        }
    )
    shard_counts = {
        int(payload["shard_count"])
        for payload in payloads
        if isinstance(payload.get("shard_count"), int)
        and not isinstance(payload.get("shard_count"), bool)
        and int(payload["shard_count"]) > 0
    }
    if not present_indices:
        return None
    if len(shard_counts) > 1:
        raise RuntimeError(
            f"inconsistent shard_count discovered from sidecars: stage_name={stage_name} sample_role={sample_role}"
        )
    shard_count = next(iter(shard_counts)) if shard_counts else len(present_indices)
    return {
        "source": "drive_sidecar_scan",
        "shard_count": shard_count,
        "expected_indices": list(range(shard_count)),
        "present_indices": present_indices,
    }


def _quality_shards_from_prepare_manifest(family_root: Path) -> Dict[str, Any] | None:
    """
    功能：从 pw04_prepare manifest 解析 quality shard 计划。

    Resolve expected quality shards from the PW04 prepare manifest.

    Args:
        family_root: Family root path.

    Returns:
        Discovery payload or None.
    """
    prepare_manifest_path = family_root / "exports" / "pw04" / "manifests" / "pw04_prepare_manifest.json"
    prepare_manifest = _read_optional_json_dict(prepare_manifest_path)
    if prepare_manifest is None:
        return None
    expected_paths_node = prepare_manifest.get("expected_quality_shard_paths")
    if not isinstance(expected_paths_node, list) or not expected_paths_node:
        return None
    expected_indices: List[int] = []
    for path_value in expected_paths_node:
        shard_index = _parse_quality_shard_index(str(path_value))
        if shard_index is None:
            raise ValueError(f"prepare manifest contains invalid quality shard path: {path_value}")
        expected_indices.append(shard_index)
    return {
        "source": "prepare_manifest.expected_quality_shard_paths",
        "shard_count": len(expected_indices),
        "expected_indices": sorted(expected_indices),
        "present_indices": sorted(
            [
                shard_index
                for shard_index in expected_indices
                if (family_root / "exports" / "pw04" / "quality" / "shards" / f"quality_shard_{shard_index:04d}.json").exists()
            ]
        ),
    }


def _quality_shards_from_quality_pair_plan(family_root: Path) -> Dict[str, Any] | None:
    """
    功能：从 quality_pair_plan 解析 quality shard 计划。

    Resolve expected quality shards from the local quality pair plan.

    Args:
        family_root: Family root path.

    Returns:
        Discovery payload or None.
    """
    quality_pair_plan_path = family_root / "exports" / "pw04" / "quality" / "quality_pair_plan.json"
    quality_pair_plan = _read_optional_json_dict(quality_pair_plan_path)
    if quality_pair_plan is None:
        return None
    shards_node = quality_pair_plan.get("shards")
    if not isinstance(shards_node, list) or not shards_node:
        return None
    expected_indices: List[int] = []
    for shard_node in shards_node:
        if not isinstance(shard_node, Mapping):
            raise ValueError("quality_pair_plan.shards must contain mappings")
        shard_index = shard_node.get("quality_shard_index")
        if not isinstance(shard_index, int) or isinstance(shard_index, bool) or shard_index < 0:
            raise ValueError("quality_pair_plan shard missing quality_shard_index")
        expected_indices.append(int(shard_index))
    return {
        "source": "quality_pair_plan.shards",
        "shard_count": len(expected_indices),
        "expected_indices": sorted(expected_indices),
        "present_indices": sorted(
            [
                shard_index
                for shard_index in expected_indices
                if (family_root / "exports" / "pw04" / "quality" / "shards" / f"quality_shard_{shard_index:04d}.json").exists()
            ]
        ),
    }


def discover_expected_shards(
    *,
    family_id: str,
    local_project_root: Path,
    drive_bundle_root: Path,
    shard_kind: str,
    sample_role: str | None = None,
) -> Dict[str, Any]:
    """
    功能：动态发现 source / attack / quality shard 计划。

    Dynamically discover source, attack, or quality shard expectations.

    Args:
        family_id: Family identifier.
        local_project_root: Local runtime project root.
        drive_bundle_root: Drive bundle root.
        shard_kind: One of source, attack, or quality.
        sample_role: Optional source sample role when shard_kind=source.

    Returns:
        Discovery payload with shard_count and expected_indices.

    Raises:
        FileNotFoundError: If the shard plan cannot be discovered.
    """
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if not isinstance(local_project_root, Path):
        raise TypeError("local_project_root must be Path")
    if not isinstance(drive_bundle_root, Path):
        raise TypeError("drive_bundle_root must be Path")
    if shard_kind not in {"source", "attack", "quality"}:
        raise ValueError(f"unsupported shard_kind: {shard_kind}")

    family_root = _family_root(local_project_root, family_id)
    if shard_kind == "source":
        if sample_role not in SOURCE_ROLE_DIRECTORY_NAMES:
            raise ValueError("source shard discovery requires valid sample_role")
        shard_count = _source_shard_count_from_local_plan(family_root, sample_role)
        if shard_count is not None:
            return {
                "source": "local_source_shard_plan",
                "shard_count": shard_count,
                "expected_indices": list(range(shard_count)),
                "present_indices": [],
            }

        shard_count = _source_shard_count_from_pw00_summary(family_root)
        if shard_count is not None:
            return {
                "source": "local_pw00_summary",
                "shard_count": shard_count,
                "expected_indices": list(range(shard_count)),
                "present_indices": [],
            }

        bootstrap_counts = _bootstrap_counts_from_sidecar(drive_bundle_root, family_id)
        shard_count = bootstrap_counts["source_shard_count"]
        if isinstance(shard_count, int) and shard_count > 0:
            return {
                "source": "pw00_bootstrap_bundle_sidecar",
                "shard_count": shard_count,
                "expected_indices": list(range(shard_count)),
                "present_indices": [],
            }

        scan_payload = _scan_shard_sidecars(
            drive_bundle_root=drive_bundle_root,
            family_id=family_id,
            stage_name=PW01_STAGE_NAME,
            sample_role=sample_role,
        )
        if scan_payload is not None:
            return scan_payload

        raise FileNotFoundError(
            f"unable to discover source shard count: family_id={family_id} sample_role={sample_role}"
        )

    if shard_kind == "attack":
        shard_count = _attack_shard_count_from_local_plan(family_root)
        if shard_count is not None:
            return {
                "source": "local_attack_shard_plan",
                "shard_count": shard_count,
                "expected_indices": list(range(shard_count)),
                "present_indices": [],
            }

        shard_count = _attack_shard_count_from_pw00_summary(family_root)
        if shard_count is not None:
            return {
                "source": "local_pw00_summary",
                "shard_count": shard_count,
                "expected_indices": list(range(shard_count)),
                "present_indices": [],
            }

        bootstrap_counts = _bootstrap_counts_from_sidecar(drive_bundle_root, family_id)
        shard_count = bootstrap_counts["attack_shard_count"]
        if isinstance(shard_count, int) and shard_count > 0:
            return {
                "source": "pw00_bootstrap_bundle_sidecar",
                "shard_count": shard_count,
                "expected_indices": list(range(shard_count)),
                "present_indices": [],
            }

        scan_payload = _scan_shard_sidecars(
            drive_bundle_root=drive_bundle_root,
            family_id=family_id,
            stage_name=PW03_STAGE_NAME,
        )
        if scan_payload is not None:
            return scan_payload

        raise FileNotFoundError(f"unable to discover attack shard count: family_id={family_id}")

    prepare_manifest_payload = _quality_shards_from_prepare_manifest(family_root)
    if prepare_manifest_payload is not None:
        return prepare_manifest_payload

    quality_pair_plan_payload = _quality_shards_from_quality_pair_plan(family_root)
    if quality_pair_plan_payload is not None:
        return quality_pair_plan_payload

    scan_payload = _scan_shard_sidecars(
        drive_bundle_root=drive_bundle_root,
        family_id=family_id,
        stage_name=PW04_QUALITY_STAGE_NAME,
    )
    if scan_payload is not None:
        return scan_payload

    raise FileNotFoundError(f"unable to discover quality shard count: family_id={family_id}")


def resolve_optional_control_negative_inclusion(
    *,
    stage_name: str,
    family_id: str,
    local_project_root: Path,
    drive_bundle_root: Path,
    include_optional_control_negative: bool | None = None,
) -> bool:
    """
    功能：为 PW02 自动判定是否纳入 optional control-negative source shards。

    Resolve whether PW02 should include optional control-negative source shards.

    Args:
        stage_name: Canonical stage name requesting optional control-negative staging.
        family_id: Family identifier.
        local_project_root: Local runtime project root.
        drive_bundle_root: Drive bundle root.
        include_optional_control_negative: Explicit override. When provided,
            the helper returns this value directly without auto-detection.

    Returns:
        True when PW02 should pull optional control-negative inputs; otherwise False.

    Raises:
        RuntimeError: If complete-consumable control-negative shard bundles are
            only partially available.
    """
    if stage_name not in STAGE_DIRECTORY_NAMES:
        raise ValueError(f"unsupported stage_name: {stage_name}")
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if not isinstance(local_project_root, Path):
        raise TypeError("local_project_root must be Path")
    if not isinstance(drive_bundle_root, Path):
        raise TypeError("drive_bundle_root must be Path")
    if include_optional_control_negative is not None:
        if not isinstance(include_optional_control_negative, bool):
            raise TypeError("include_optional_control_negative must be bool or None")
        return include_optional_control_negative
    if stage_name != PW02_STAGE_NAME:
        return False

    expected_discovery = discover_expected_shards(
        family_id=family_id,
        local_project_root=local_project_root,
        drive_bundle_root=drive_bundle_root,
        shard_kind="source",
        sample_role=CONTROL_NEGATIVE_ROLE,
    )
    scan_payload = _scan_shard_sidecars(
        drive_bundle_root=drive_bundle_root,
        family_id=family_id,
        stage_name=PW01_STAGE_NAME,
        sample_role=CONTROL_NEGATIVE_ROLE,
    )
    if scan_payload is None:
        return False

    expected_indices = sorted(int(index) for index in cast(List[int], expected_discovery["expected_indices"]))
    present_indices = sorted(int(index) for index in cast(List[int], scan_payload["present_indices"]))
    if not present_indices:
        return False

    expected_shard_count = int(expected_discovery["shard_count"])
    sidecar_shard_count = int(scan_payload["shard_count"])
    if sidecar_shard_count != expected_shard_count:
        raise RuntimeError(
            "optional control-negative shard_count mismatch: "
            f"family_id={family_id}, expected={expected_shard_count}, sidecar_shard_count={sidecar_shard_count}"
        )

    missing_indices = sorted(set(expected_indices) - set(present_indices))
    unexpected_indices = sorted(set(present_indices) - set(expected_indices))
    if missing_indices or unexpected_indices:
        raise RuntimeError(
            "partial optional control-negative source shards detected: "
            f"family_id={family_id}, expected_indices={expected_indices}, "
            f"present_indices={present_indices}, missing_indices={missing_indices}, "
            f"unexpected_indices={unexpected_indices}"
        )
    return True


def _initial_dependency_references(
    *,
    stage_name: str,
    family_id: str,
    drive_bundle_root: Path,
) -> List[Dict[str, Any]]:
    """
    功能：返回无需动态 shard 发现的初始依赖。

    Return the initial dependency bundles that do not require dynamic shard discovery.

    Args:
        stage_name: Canonical stage name.
        family_id: Family identifier.
        drive_bundle_root: Drive bundle root.

    Returns:
        Initial dependency descriptors.
    """
    refs: List[Dict[str, Any]] = []
    if stage_name == PW00_STAGE_NAME:
        return refs

    refs.append(
        _build_bundle_reference(
            stage_name=PW00_STAGE_NAME,
            family_id=family_id,
            drive_bundle_root=drive_bundle_root,
        )
    )
    if stage_name in {
        PW03_STAGE_NAME,
        PW04_PREPARE_STAGE_NAME,
        PW04_QUALITY_STAGE_NAME,
        PW04_FINALIZE_STAGE_NAME,
        PW05_STAGE_NAME,
    }:
        refs.append(
            _build_bundle_reference(
                stage_name=PW02_STAGE_NAME,
                family_id=family_id,
                drive_bundle_root=drive_bundle_root,
            )
        )
    if stage_name in {PW04_QUALITY_STAGE_NAME, PW04_FINALIZE_STAGE_NAME, PW05_STAGE_NAME}:
        refs.append(
            _build_bundle_reference(
                stage_name=PW04_PREPARE_STAGE_NAME,
                family_id=family_id,
                drive_bundle_root=drive_bundle_root,
            )
        )
    if stage_name == PW05_STAGE_NAME:
        refs.append(
            _build_bundle_reference(
                stage_name=PW04_FINALIZE_STAGE_NAME,
                family_id=family_id,
                drive_bundle_root=drive_bundle_root,
            )
        )
    return refs


def _append_formal_source_shard_dependencies(
    *,
    required_bundles: List[Dict[str, Any]],
    discovery: Dict[str, Any],
    family_id: str,
    local_project_root: Path,
    drive_bundle_root: Path,
    include_optional_control_negative: bool = False,
) -> None:
    """
    功能：向 stage 依赖中追加 formal source shard bundles。

    Append the formal source-shard dependency bundles for one stage.

    Args:
        required_bundles: Dependency bundle descriptors being accumulated.
        discovery: Discovery payload mapping to update in place.
        family_id: Family identifier.
        local_project_root: Local runtime project root.
        drive_bundle_root: Drive bundle root.
        include_optional_control_negative: Whether optional control-negative shards should be appended.

    Returns:
        None.
    """
    if not isinstance(required_bundles, list):
        raise TypeError("required_bundles must be list")
    if not isinstance(discovery, dict):
        raise TypeError("discovery must be dict")
    if not isinstance(local_project_root, Path):
        raise TypeError("local_project_root must be Path")
    if not isinstance(drive_bundle_root, Path):
        raise TypeError("drive_bundle_root must be Path")

    role_discoveries: List[tuple[str, Dict[str, Any]]] = [
        (
            POSITIVE_SOURCE_ROLE,
            discover_expected_shards(
                family_id=family_id,
                local_project_root=local_project_root,
                drive_bundle_root=drive_bundle_root,
                shard_kind="source",
                sample_role=POSITIVE_SOURCE_ROLE,
            ),
        ),
        (
            CLEAN_NEGATIVE_ROLE,
            discover_expected_shards(
                family_id=family_id,
                local_project_root=local_project_root,
                drive_bundle_root=drive_bundle_root,
                shard_kind="source",
                sample_role=CLEAN_NEGATIVE_ROLE,
            ),
        ),
    ]
    if include_optional_control_negative:
        role_discoveries.append(
            (
                CONTROL_NEGATIVE_ROLE,
                discover_expected_shards(
                    family_id=family_id,
                    local_project_root=local_project_root,
                    drive_bundle_root=drive_bundle_root,
                    shard_kind="source",
                    sample_role=CONTROL_NEGATIVE_ROLE,
                ),
            )
        )

    for source_role, role_discovery in role_discoveries:
        discovery[source_role] = role_discovery
        resolved_shard_count = int(role_discovery["shard_count"])
        for expected_index in cast(List[int], role_discovery["expected_indices"]):
            required_bundles.append(
                _build_bundle_reference(
                    stage_name=PW01_STAGE_NAME,
                    family_id=family_id,
                    drive_bundle_root=drive_bundle_root,
                    sample_role=source_role,
                    shard_index=expected_index,
                    shard_count=resolved_shard_count,
                )
            )


def resolve_stage_dependencies(
    *,
    stage_name: str,
    family_id: str,
    local_project_root: Path,
    drive_bundle_root: Path,
    sample_role: str | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    include_optional_control_negative: bool = False,
) -> Dict[str, Any]:
    """
    功能：按 stage 规则解析所需 bundle 依赖。

    Resolve all required dependency bundles for one stage.

    Args:
        stage_name: Canonical stage name.
        family_id: Family identifier.
        local_project_root: Local runtime project root.
        drive_bundle_root: Drive bundle root.
        sample_role: Optional sample role.
        shard_index: Optional shard index.
        shard_count: Optional shard count.
        include_optional_control_negative: Whether optional control-negative shards should be included.

    Returns:
        Dependency resolution payload.
    """
    if stage_name not in STAGE_DIRECTORY_NAMES:
        raise ValueError(f"unsupported stage_name: {stage_name}")
    if not isinstance(local_project_root, Path):
        raise TypeError("local_project_root must be Path")
    if not isinstance(drive_bundle_root, Path):
        raise TypeError("drive_bundle_root must be Path")

    required_bundles: List[Dict[str, Any]] = _initial_dependency_references(
        stage_name=stage_name,
        family_id=family_id,
        drive_bundle_root=drive_bundle_root,
    )
    discovery: Dict[str, Any] = {}

    if stage_name == PW00_STAGE_NAME:
        return {
            "stage_name": stage_name,
            "family_id": family_id,
            "required_bundles": [],
            "discovery": {},
        }

    if stage_name == PW01_STAGE_NAME:
        if sample_role not in SOURCE_ROLE_DIRECTORY_NAMES:
            raise ValueError("PW01 requires valid sample_role")
        return {
            "stage_name": stage_name,
            "family_id": family_id,
            "required_bundles": required_bundles,
            "discovery": {},
        }

    if stage_name in {PW02_STAGE_NAME, PW03_STAGE_NAME, PW04_QUALITY_STAGE_NAME}:
        _append_formal_source_shard_dependencies(
            required_bundles=required_bundles,
            discovery=discovery,
            family_id=family_id,
            local_project_root=local_project_root,
            drive_bundle_root=drive_bundle_root,
            include_optional_control_negative=include_optional_control_negative,
        )
        if stage_name in {PW02_STAGE_NAME, PW03_STAGE_NAME}:
            return {
                "stage_name": stage_name,
                "family_id": family_id,
                "required_bundles": required_bundles,
                "discovery": discovery,
            }

    attack_discovery = discover_expected_shards(
        family_id=family_id,
        local_project_root=local_project_root,
        drive_bundle_root=drive_bundle_root,
        shard_kind="attack",
    )
    discovery["attack"] = attack_discovery
    for expected_index in cast(List[int], attack_discovery["expected_indices"]):
        required_bundles.append(
            _build_bundle_reference(
                stage_name=PW03_STAGE_NAME,
                family_id=family_id,
                drive_bundle_root=drive_bundle_root,
                shard_index=expected_index,
                shard_count=int(attack_discovery["shard_count"]),
            )
        )

    if stage_name in {PW04_FINALIZE_STAGE_NAME, PW05_STAGE_NAME}:
        quality_discovery = discover_expected_shards(
            family_id=family_id,
            local_project_root=local_project_root,
            drive_bundle_root=drive_bundle_root,
            shard_kind="quality",
        )
        discovery["quality"] = quality_discovery
        for expected_index in cast(List[int], quality_discovery["expected_indices"]):
            required_bundles.append(
                _build_bundle_reference(
                    stage_name=PW04_QUALITY_STAGE_NAME,
                    family_id=family_id,
                    drive_bundle_root=drive_bundle_root,
                    shard_index=expected_index,
                    shard_count=int(quality_discovery["shard_count"]),
                )
            )

    return {
        "stage_name": stage_name,
        "family_id": family_id,
        "required_bundles": required_bundles,
        "discovery": discovery,
    }


def _append_formal_source_shard_required_paths(
    *,
    required_paths: List[str],
    family_id: str,
    local_project_root: Path,
    drive_bundle_root: Path,
    include_optional_control_negative: bool = False,
) -> None:
    """
    功能：向 stage 必需输入列表追加 formal source shard 目录。

    Append the formal source-shard directories to one required-input list.

    Args:
        required_paths: Required relative paths being accumulated.
        family_id: Family identifier.
        local_project_root: Local runtime project root.
        drive_bundle_root: Drive bundle root.
        include_optional_control_negative: Whether optional control-negative shard directories should be appended.

    Returns:
        None.
    """
    if not isinstance(required_paths, list):
        raise TypeError("required_paths must be list")
    if not isinstance(local_project_root, Path):
        raise TypeError("local_project_root must be Path")
    if not isinstance(drive_bundle_root, Path):
        raise TypeError("drive_bundle_root must be Path")

    family_root_relative = f"paper_workflow/families/{family_id}"
    role_discoveries: List[tuple[str, Dict[str, Any]]] = [
        (
            POSITIVE_SOURCE_ROLE,
            discover_expected_shards(
                family_id=family_id,
                local_project_root=local_project_root,
                drive_bundle_root=drive_bundle_root,
                shard_kind="source",
                sample_role=POSITIVE_SOURCE_ROLE,
            ),
        ),
        (
            CLEAN_NEGATIVE_ROLE,
            discover_expected_shards(
                family_id=family_id,
                local_project_root=local_project_root,
                drive_bundle_root=drive_bundle_root,
                shard_kind="source",
                sample_role=CLEAN_NEGATIVE_ROLE,
            ),
        ),
    ]
    if include_optional_control_negative:
        role_discoveries.append(
            (
                CONTROL_NEGATIVE_ROLE,
                discover_expected_shards(
                    family_id=family_id,
                    local_project_root=local_project_root,
                    drive_bundle_root=drive_bundle_root,
                    shard_kind="source",
                    sample_role=CONTROL_NEGATIVE_ROLE,
                ),
            )
        )

    for source_role, role_discovery in role_discoveries:
        role_directory_name = SOURCE_ROLE_DIRECTORY_NAMES[source_role]
        for expected_index in cast(List[int], role_discovery["expected_indices"]):
            required_paths.append(
                f"{family_root_relative}/source_shards/{role_directory_name}/shard_{expected_index:04d}"
            )


def safe_clean_local_runtime(local_project_root: Path) -> Dict[str, Any]:
    """
    功能：按硬约束清理 notebook 本地 runtime 根目录。

    Safely clean the notebook-local runtime root under strict Colab-only guards.

    Args:
        local_project_root: Local runtime project root.

    Returns:
        Cleanup summary payload.

    Raises:
        RuntimeError: If the path violates any protected-path constraint.
    """
    if not isinstance(local_project_root, Path):
        raise TypeError("local_project_root must be Path")

    posix_path = PurePosixPath(_as_posix_path(local_project_root))
    violations: List[str] = []
    if posix_path == CONTENT_ROOT_POSIX:
        violations.append("local_project_root must not equal /content")
    if not str(posix_path).startswith(f"{CONTENT_ROOT_POSIX.as_posix()}/"):
        violations.append("local_project_root must be located under /content/")
    if any(part.lower() == "drive" for part in posix_path.parts):
        violations.append("local_project_root must not contain drive")
    if posix_path == PurePosixPath("/content/ceg_wm_workspace"):
        violations.append("local_project_root must not equal /content/ceg_wm_workspace")

    for protected_path in PROTECTED_LOCAL_RUNTIME_PATHS:
        protected_text = protected_path.as_posix()
        candidate_text = posix_path.as_posix()
        if candidate_text == protected_text or protected_text.startswith(f"{candidate_text}/"):
            violations.append(f"local_project_root must not delete protected path: {protected_text}")

    if violations:
        raise RuntimeError("; ".join(violations))

    if local_project_root.exists() and not local_project_root.is_dir():
        raise RuntimeError(f"local_project_root must be directory when present: {local_project_root}")

    cleaned = False
    if local_project_root.exists():
        shutil.rmtree(local_project_root)
        cleaned = True

    return {
        "status": "cleaned" if cleaned else "already_absent",
        "local_project_root": _as_posix_path(local_project_root),
    }


def compute_file_sha256(file_path: Path) -> str:
    """
    功能：计算文件 SHA256。

    Compute the SHA256 digest for one file.

    Args:
        file_path: File path.

    Returns:
        Lowercase SHA256 digest.
    """
    if not isinstance(file_path, Path):
        raise TypeError("file_path must be Path")
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"file_path not found: {file_path}")

    sha256_obj = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256_obj.update(chunk)
    return sha256_obj.hexdigest()


def create_tar_gz_bundle(
    *,
    local_project_root: Path,
    relative_paths: Sequence[str],
    tar_path: Path,
) -> Dict[str, Any]:
    """
    功能：把指定相对路径打包为 tar.gz bundle。

    Create one tar.gz bundle from a list of relative paths.

    Args:
        local_project_root: Local runtime project root.
        relative_paths: Relative paths to include.
        tar_path: Target tar.gz path.

    Returns:
        Bundle creation summary.
    """
    if not isinstance(local_project_root, Path):
        raise TypeError("local_project_root must be Path")
    if not isinstance(tar_path, Path):
        raise TypeError("tar_path must be Path")

    normalized_relative_paths = _normalize_relative_paths(relative_paths)
    source_paths: List[tuple[str, Path]] = []
    for relative_path in normalized_relative_paths:
        absolute_path = _relative_path_to_absolute(local_project_root, relative_path)
        if not absolute_path.exists():
            raise FileNotFoundError(f"bundle source path not found: {relative_path}")
        source_paths.append((relative_path, absolute_path))

    ensure_directory(tar_path.parent)
    temporary_tar_path = tar_path.with_name(f"{tar_path.name}.tmp")
    if temporary_tar_path.exists():
        temporary_tar_path.unlink()

    added_names: set[str] = set()
    try:
        with tarfile.open(temporary_tar_path, "w:gz") as tar_file:
            for relative_path, absolute_path in source_paths:
                items_to_add = [absolute_path]
                if absolute_path.is_dir():
                    items_to_add.extend(sorted(absolute_path.rglob("*")))
                for item_path in items_to_add:
                    relative_item_path = item_path.relative_to(local_project_root).as_posix()
                    if relative_item_path in added_names:
                        continue
                    tar_file.add(item_path, arcname=relative_item_path, recursive=False)
                    added_names.add(relative_item_path)
        temporary_tar_path.replace(tar_path)
    except Exception:
        if temporary_tar_path.exists():
            temporary_tar_path.unlink()
        raise

    return {
        "status": "ok",
        "tar_path": normalize_path_value(tar_path),
        "relative_paths": normalized_relative_paths,
        "member_count": len(added_names),
    }


def extract_tar_gz_bundle(*, tar_path: Path, local_project_root: Path) -> Dict[str, Any]:
    """
    功能：把 tar.gz bundle 安全解压到本地 runtime 根目录。

    Safely extract one tar.gz bundle into the local runtime root.

    Args:
        tar_path: Bundle tar.gz path.
        local_project_root: Local runtime project root.

    Returns:
        Extraction summary payload.
    """
    if not isinstance(tar_path, Path):
        raise TypeError("tar_path must be Path")
    if not isinstance(local_project_root, Path):
        raise TypeError("local_project_root must be Path")
    if not tar_path.exists() or not tar_path.is_file():
        raise FileNotFoundError(f"tar_path not found: {tar_path}")

    ensure_directory(local_project_root)
    member_names: List[str] = []
    with tarfile.open(tar_path, "r:gz") as tar_file:
        members = tar_file.getmembers()
        for member in members:
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or any(part == ".." for part in member_path.parts):
                raise RuntimeError(f"tar member contains path escape: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"tar bundle must not contain links: {member.name}")
            candidate_path = local_project_root / Path(*member_path.parts)
            validate_path_within_base(local_project_root, candidate_path, "bundle extraction path")
            member_names.append(member.name)
        tar_file.extractall(local_project_root, members=members)

    return {
        "status": "ok",
        "tar_path": normalize_path_value(tar_path),
        "local_project_root": _as_posix_path(local_project_root),
        "member_count": len(member_names),
    }


def write_bundle_sidecar(
    *,
    sidecar_path: Path,
    family_id: str,
    stage_name: str,
    bundle_kind: str,
    tar_path: Path,
    tar_sha256: str,
    local_project_root: Path,
    relative_paths: Sequence[str],
    sample_role: str | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    功能：写入可消费完成态的 bundle sidecar。

    Write the complete-consumable bundle sidecar after the tar is finalized.

    Args:
        sidecar_path: Sidecar path.
        family_id: Family identifier.
        stage_name: Canonical stage name.
        bundle_kind: Bundle-kind token.
        tar_path: Tar.gz path.
        tar_sha256: Tar SHA256 digest.
        local_project_root: Local runtime project root.
        relative_paths: Archived relative paths.
        sample_role: Optional sample role.
        shard_index: Optional shard index.
        shard_count: Optional shard count.
        extra_fields: Optional extra metadata fields.

    Returns:
        Written sidecar payload.
    """
    if not isinstance(sidecar_path, Path):
        raise TypeError("sidecar_path must be Path")
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(bundle_kind, str) or not bundle_kind:
        raise TypeError("bundle_kind must be non-empty str")
    if not isinstance(tar_path, Path):
        raise TypeError("tar_path must be Path")
    if not tar_path.exists() or not tar_path.is_file():
        raise FileNotFoundError(f"tar_path not found: {tar_path}")
    if not isinstance(tar_sha256, str) or not tar_sha256.strip():
        raise TypeError("tar_sha256 must be non-empty str")
    if not isinstance(local_project_root, Path):
        raise TypeError("local_project_root must be Path")
    if extra_fields is not None and not isinstance(extra_fields, Mapping):
        raise TypeError("extra_fields must be Mapping or None")

    tar_name = tar_path.name
    if Path(tar_name).name != tar_name or ".." in tar_name or "/" in tar_name or "\\" in tar_name:
        raise ValueError(f"invalid tar_name: {tar_name}")

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "family_id": family_id.strip(),
        "stage_name": stage_name,
        "bundle_kind": bundle_kind,
        "sample_role": sample_role,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "tar_name": tar_name,
        "tar_sha256": tar_sha256.strip(),
        "created_at_utc": utc_now_iso(),
        "local_project_root": _as_posix_path(local_project_root),
        "relative_paths": _normalize_relative_paths(relative_paths),
        "status": "complete",
    }
    if extra_fields is not None:
        for key_name, value in extra_fields.items():
            payload[str(key_name)] = value

    ensure_directory(sidecar_path.parent)
    write_json_atomic(sidecar_path, payload)
    return payload


def read_bundle_sidecar(sidecar_path: Path) -> Dict[str, Any]:
    """
    功能：读取并校验 bundle sidecar 的基本结构。

    Read one bundle sidecar and validate its required fields.

    Args:
        sidecar_path: Sidecar path.

    Returns:
        Parsed sidecar payload.
    """
    payload = _load_json_dict(sidecar_path, "bundle sidecar")
    required_fields = [
        "schema_version",
        "family_id",
        "stage_name",
        "bundle_kind",
        "sample_role",
        "shard_index",
        "shard_count",
        "tar_name",
        "tar_sha256",
        "created_at_utc",
        "local_project_root",
        "relative_paths",
        "status",
    ]
    missing_fields = [field_name for field_name in required_fields if field_name not in payload]
    if missing_fields:
        raise ValueError(f"bundle sidecar missing fields: {missing_fields}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported bundle schema_version: expected={SCHEMA_VERSION} actual={payload.get('schema_version')}"
        )
    if not isinstance(payload.get("relative_paths"), list):
        raise ValueError("bundle sidecar relative_paths must be list")
    payload["relative_paths"] = _normalize_relative_paths(cast(List[str], payload["relative_paths"]))
    tar_name = payload.get("tar_name")
    if not isinstance(tar_name, str) or not tar_name:
        raise ValueError("bundle sidecar tar_name must be non-empty str")
    if Path(tar_name).name != tar_name or ".." in tar_name or "/" in tar_name or "\\" in tar_name:
        raise ValueError(f"bundle sidecar tar_name contains path escape: {tar_name}")
    return payload


def verify_bundle_integrity(sidecar_path: Path) -> Dict[str, Any]:
    """
    功能：校验 bundle sidecar、tar 存在性与 SHA256 一致性。

    Verify bundle sidecar validity, tar presence, and SHA256 integrity.

    Args:
        sidecar_path: Sidecar path.

    Returns:
        Integrity summary payload.
    """
    payload = read_bundle_sidecar(sidecar_path)
    if payload.get("status") != "complete":
        raise RuntimeError(f"bundle sidecar status must be complete: {sidecar_path}")

    tar_path = sidecar_path.parent / str(payload["tar_name"])
    if not tar_path.exists() or not tar_path.is_file():
        raise FileNotFoundError(f"bundle tar.gz not found: {tar_path}")

    actual_sha256 = compute_file_sha256(tar_path)
    expected_sha256 = str(payload.get("tar_sha256", "")).strip()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"bundle tar_sha256 mismatch: expected={expected_sha256} actual={actual_sha256} path={tar_path}"
        )

    return {
        **payload,
        "sidecar_path": normalize_path_value(sidecar_path),
        "tar_path": normalize_path_value(tar_path),
        "actual_tar_sha256": actual_sha256,
    }


def _required_input_paths_for_stage(
    *,
    stage_name: str,
    family_id: str,
    local_project_root: Path,
    drive_bundle_root: Path,
    include_optional_control_negative: bool = False,
) -> List[str]:
    """
    功能：构造某个 stage 的关键输入路径集合。

    Build the key required-input paths for one stage.

    Args:
        stage_name: Canonical stage name.
        family_id: Family identifier.
        local_project_root: Local runtime project root.
        drive_bundle_root: Drive bundle root.
        include_optional_control_negative: Whether optional control-negative inputs are required.

    Returns:
        Required relative input paths.
    """
    family_root_relative = f"paper_workflow/families/{family_id}"
    required_paths: List[str] = []

    if stage_name == PW00_STAGE_NAME:
        return required_paths

    if stage_name == PW01_STAGE_NAME:
        required_paths.extend(
            [
                f"{family_root_relative}/manifests/paper_eval_family_manifest.json",
                f"{family_root_relative}/manifests/source_event_grid.jsonl",
                f"{family_root_relative}/manifests/source_shard_plan.json",
                f"{family_root_relative}/manifests/source_split_plan.json",
            ]
        )
        return required_paths

    if stage_name in {PW02_STAGE_NAME, PW03_STAGE_NAME}:
        required_paths.extend(
            [
                f"{family_root_relative}/manifests/paper_eval_family_manifest.json",
                f"{family_root_relative}/manifests/source_shard_plan.json",
                f"{family_root_relative}/manifests/source_split_plan.json",
                f"{family_root_relative}/runtime_state/pw00_summary.json",
            ]
        )
        _append_formal_source_shard_required_paths(
            required_paths=required_paths,
            family_id=family_id,
            local_project_root=local_project_root,
            drive_bundle_root=drive_bundle_root,
            include_optional_control_negative=include_optional_control_negative,
        )
        if stage_name == PW03_STAGE_NAME:
            required_paths.extend(
                [
                    f"{family_root_relative}/manifests/attack_event_grid.jsonl",
                    f"{family_root_relative}/manifests/attack_shard_plan.json",
                    f"{family_root_relative}/runtime_state/pw02_summary.json",
                    f"{family_root_relative}/exports/pw02/paper_source_finalize_manifest.json",
                ]
            )
        return required_paths

    attack_discovery = discover_expected_shards(
        family_id=family_id,
        local_project_root=local_project_root,
        drive_bundle_root=drive_bundle_root,
        shard_kind="attack",
    )
    required_paths.extend(
        [
            f"{family_root_relative}/runtime_state/pw00_summary.json",
            f"{family_root_relative}/runtime_state/pw02_summary.json",
            f"{family_root_relative}/exports/pw02/paper_source_finalize_manifest.json",
            f"{family_root_relative}/exports/pw02/thresholds/content/thresholds.json",
            f"{family_root_relative}/exports/pw02/thresholds/attestation/thresholds.json",
            f"{family_root_relative}/manifests/attack_shard_plan.json",
        ]
    )
    for expected_index in cast(List[int], attack_discovery["expected_indices"]):
        required_paths.append(
            f"{family_root_relative}/attack_shards/shard_{expected_index:04d}/shard_manifest.json"
        )

    if stage_name == PW04_PREPARE_STAGE_NAME:
        return required_paths

    required_paths.extend(
        [
            f"{family_root_relative}/exports/pw04/manifests/pw04_prepare_manifest.json",
            f"{family_root_relative}/exports/pw04/quality/quality_pair_plan.json",
        ]
    )
    if stage_name == PW04_QUALITY_STAGE_NAME:
        _append_formal_source_shard_required_paths(
            required_paths=required_paths,
            family_id=family_id,
            local_project_root=local_project_root,
            drive_bundle_root=drive_bundle_root,
            include_optional_control_negative=include_optional_control_negative,
        )
        return required_paths

    quality_discovery = discover_expected_shards(
        family_id=family_id,
        local_project_root=local_project_root,
        drive_bundle_root=drive_bundle_root,
        shard_kind="quality",
    )
    for expected_index in cast(List[int], quality_discovery["expected_indices"]):
        required_paths.append(
            f"{family_root_relative}/exports/pw04/quality/shards/quality_shard_{expected_index:04d}.json"
        )

    if stage_name == PW04_FINALIZE_STAGE_NAME:
        return required_paths

    required_paths.extend(
        [
            f"{family_root_relative}/snapshots/config_snapshot.yaml",
            f"{family_root_relative}/runtime_state/pw04_summary.json",
        ]
    )
    return required_paths


def _verify_required_inputs_for_stage(
    *,
    stage_name: str,
    family_id: str,
    local_project_root: Path,
    drive_bundle_root: Path,
    include_optional_control_negative: bool = False,
) -> List[str]:
    """
    功能：校验 prepare 后关键输入已到位。

    Verify that key required inputs exist after dependency extraction.

    Args:
        stage_name: Canonical stage name.
        family_id: Family identifier.
        local_project_root: Local runtime project root.
        drive_bundle_root: Drive bundle root.
        include_optional_control_negative: Whether optional control-negative inputs are required.

    Returns:
        Verified relative input paths.
    """
    required_paths = _required_input_paths_for_stage(
        stage_name=stage_name,
        family_id=family_id,
        local_project_root=local_project_root,
        drive_bundle_root=drive_bundle_root,
        include_optional_control_negative=include_optional_control_negative,
    )
    missing_paths: List[str] = []
    for relative_path in required_paths:
        absolute_path = _relative_path_to_absolute(local_project_root, relative_path)
        if not absolute_path.exists():
            missing_paths.append(relative_path)
    if missing_paths:
        raise FileNotFoundError(f"required stage inputs missing: {missing_paths}")
    return required_paths


def _stage_output_relative_paths(
    *,
    stage_name: str,
    family_id: str,
    sample_role: str | None = None,
    shard_index: int | None = None,
) -> List[str]:
    """
    功能：返回某个 stage 需要归档的相对路径列表。

    Return the relative paths that must be archived for one stage.

    Args:
        stage_name: Canonical stage name.
        family_id: Family identifier.
        sample_role: Optional sample role.
        shard_index: Optional shard index.

    Returns:
        Relative archive paths.
    """
    family_root_relative = f"paper_workflow/families/{family_id}"
    if stage_name == PW00_STAGE_NAME:
        return [
            f"{family_root_relative}/manifests",
            f"{family_root_relative}/runtime_state/pw00_summary.json",
            f"{family_root_relative}/runtime_state/pw00_runtime_diagnostics.json",
            f"{family_root_relative}/snapshots",
        ]
    if stage_name == PW01_STAGE_NAME:
        if sample_role not in SOURCE_ROLE_DIRECTORY_NAMES:
            raise ValueError("PW01 archive requires valid sample_role")
        if not isinstance(shard_index, int) or shard_index < 0:
            raise ValueError("PW01 archive requires non-negative shard_index")
        return [
            f"{family_root_relative}/source_shards/{SOURCE_ROLE_DIRECTORY_NAMES[sample_role]}/shard_{shard_index:04d}",
            f"{family_root_relative}/runtime_state/pw01_{sample_role}_shard_{shard_index:04d}_runtime_diagnostics.json",
        ]
    if stage_name == PW02_STAGE_NAME:
        return [
            f"{family_root_relative}/manifests",
            f"{family_root_relative}/exports/pw02",
            f"{family_root_relative}/runtime_state/pw02_summary.json",
            f"{family_root_relative}/runtime_state/pw02_runtime_diagnostics.json",
        ]
    if stage_name == PW03_STAGE_NAME:
        if not isinstance(shard_index, int) or shard_index < 0:
            raise ValueError("PW03 archive requires non-negative shard_index")
        return [
            f"{family_root_relative}/attack_shards/shard_{shard_index:04d}",
            f"{family_root_relative}/runtime_state/pw03_attack_shard_{shard_index:04d}_runtime_diagnostics.json",
        ]
    if stage_name == PW04_PREPARE_STAGE_NAME:
        return [
            f"{family_root_relative}/exports/pw04",
            f"{family_root_relative}/runtime_state/pw04_prepare_runtime_diagnostics.json",
            f"{family_root_relative}/runtime_state/pw04_prepare_gpu_session_peak.json",
        ]
    if stage_name == PW04_QUALITY_STAGE_NAME:
        if not isinstance(shard_index, int) or shard_index < 0:
            raise ValueError("PW04 quality archive requires non-negative shard_index")
        return [
            f"{family_root_relative}/exports/pw04/quality/shards/quality_shard_{shard_index:04d}.json",
            f"{family_root_relative}/runtime_state/pw04_quality_shard_{shard_index:04d}_runtime_diagnostics.json",
            f"{family_root_relative}/runtime_state/pw04_quality_shard_{shard_index:04d}_gpu_session_peak.json",
        ]
    if stage_name == PW04_FINALIZE_STAGE_NAME:
        return [
            f"{family_root_relative}/exports/pw04",
            f"{family_root_relative}/runtime_state/pw04_summary.json",
            f"{family_root_relative}/runtime_state/pw04_finalize_runtime_diagnostics.json",
            f"{family_root_relative}/runtime_state/pw04_finalize_gpu_session_peak.json",
        ]
    if stage_name == PW05_STAGE_NAME:
        return [family_root_relative]
    raise ValueError(f"unsupported stage_name: {stage_name}")


def _stage_sidecar_extra_fields(
    *,
    stage_name: str,
    family_id: str,
    local_project_root: Path,
    drive_bundle_root: Path,
    sample_role: str | None,
    shard_index: int | None,
    shard_count: int | None,
) -> Dict[str, Any]:
    """
    功能：构造 sidecar 的 stage 级扩展字段。

    Build stage-specific sidecar extension fields.

    Args:
        stage_name: Canonical stage name.
        family_id: Family identifier.
        local_project_root: Local runtime project root.
        drive_bundle_root: Drive bundle root.
        sample_role: Optional sample role.
        shard_index: Optional shard index.
        shard_count: Optional shard count.

    Returns:
        Extra sidecar fields.
    """
    stage_paths = _stage_paths(local_project_root, family_id)
    extra_fields: Dict[str, Any] = {}

    if stage_name == PW00_STAGE_NAME:
        pw00_summary = _read_optional_json_dict(stage_paths["runtime_state_root"] / "pw00_summary.json") or {}
        extra_fields["source_shard_count"] = _extract_int(pw00_summary.get("source_shard_count"))
        extra_fields["attack_shard_count"] = _extract_int(pw00_summary.get("attack_shard_count"))
        return extra_fields

    if stage_name == PW04_PREPARE_STAGE_NAME:
        quality_discovery = discover_expected_shards(
            family_id=family_id,
            local_project_root=local_project_root,
            drive_bundle_root=drive_bundle_root,
            shard_kind="quality",
        )
        extra_fields["quality_shard_count"] = int(quality_discovery["shard_count"])
        extra_fields["expected_quality_shard_indices"] = list(quality_discovery["expected_indices"])
        extra_fields["quality_root"] = _normalize_relative_path(
            f"paper_workflow/families/{family_id}/exports/pw04/quality"
        )
        extra_fields["quality_pair_plan_path"] = _normalize_relative_path(
            f"paper_workflow/families/{family_id}/exports/pw04/quality/quality_pair_plan.json"
        )
        return extra_fields

    if stage_name == PW04_QUALITY_STAGE_NAME and shard_count is None:
        quality_discovery = discover_expected_shards(
            family_id=family_id,
            local_project_root=local_project_root,
            drive_bundle_root=drive_bundle_root,
            shard_kind="quality",
        )
        extra_fields["quality_shard_count"] = int(quality_discovery["shard_count"])
        return extra_fields

    return extra_fields


def verify_required_outputs_for_stage(
    *,
    stage_name: str,
    family_id: str,
    local_project_root: Path,
    drive_bundle_root: Path,
    sample_role: str | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> Dict[str, Any]:
    """
    功能：校验某个 stage 的归档前关键输出。

    Verify the key stage outputs before archiving.

    Args:
        stage_name: Canonical stage name.
        family_id: Family identifier.
        local_project_root: Local runtime project root.
        drive_bundle_root: Drive bundle root.
        sample_role: Optional sample role.
        shard_index: Optional shard index.
        shard_count: Optional shard count.

    Returns:
        Stage archive spec with validated relative paths.
    """
    if not isinstance(local_project_root, Path):
        raise TypeError("local_project_root must be Path")
    if not isinstance(drive_bundle_root, Path):
        raise TypeError("drive_bundle_root must be Path")

    resolved_shard_count = shard_count
    if stage_name == PW04_QUALITY_STAGE_NAME and resolved_shard_count is None:
        quality_discovery = discover_expected_shards(
            family_id=family_id,
            local_project_root=local_project_root,
            drive_bundle_root=drive_bundle_root,
            shard_kind="quality",
        )
        resolved_shard_count = int(quality_discovery["shard_count"])

    relative_paths = _stage_output_relative_paths(
        stage_name=stage_name,
        family_id=family_id,
        sample_role=sample_role,
        shard_index=shard_index,
    )
    missing_paths: List[str] = []
    for relative_path in relative_paths:
        absolute_path = _relative_path_to_absolute(local_project_root, relative_path)
        if not absolute_path.exists():
            missing_paths.append(relative_path)

    stage_paths = _stage_paths(local_project_root, family_id)
    if stage_name == PW01_STAGE_NAME and sample_role in SOURCE_ROLE_DIRECTORY_NAMES and isinstance(shard_index, int):
        shard_manifest_path = (
            stage_paths["source_shards_root"]
            / SOURCE_ROLE_DIRECTORY_NAMES[sample_role]
            / f"shard_{shard_index:04d}"
            / "shard_manifest.json"
        )
        if not shard_manifest_path.exists():
            missing_paths.append(
                _normalize_relative_path(str(shard_manifest_path.relative_to(local_project_root).as_posix()))
            )
    if stage_name == PW03_STAGE_NAME and isinstance(shard_index, int):
        shard_manifest_path = (
            stage_paths["attack_shards_root"]
            / f"shard_{shard_index:04d}"
            / "shard_manifest.json"
        )
        if not shard_manifest_path.exists():
            missing_paths.append(
                _normalize_relative_path(str(shard_manifest_path.relative_to(local_project_root).as_posix()))
            )

    if missing_paths:
        raise FileNotFoundError(f"required stage outputs missing: {sorted(set(missing_paths))}")

    extra_fields = _stage_sidecar_extra_fields(
        stage_name=stage_name,
        family_id=family_id,
        local_project_root=local_project_root,
        drive_bundle_root=drive_bundle_root,
        sample_role=sample_role,
        shard_index=shard_index,
        shard_count=resolved_shard_count,
    )
    bundle_reference = _build_bundle_reference(
        stage_name=stage_name,
        family_id=family_id,
        drive_bundle_root=drive_bundle_root,
        sample_role=sample_role,
        shard_index=shard_index,
        shard_count=resolved_shard_count,
    )
    return {
        **bundle_reference,
        "relative_paths": _normalize_relative_paths(relative_paths),
        "resolved_shard_count": resolved_shard_count,
        "extra_fields": extra_fields,
    }


def _extract_one_dependency_bundle(
    *,
    bundle_reference: Mapping[str, Any],
    local_project_root: Path,
) -> Dict[str, Any]:
    """
    功能：校验并解压一个依赖 bundle。

    Verify and extract one dependency bundle.

    Args:
        bundle_reference: Bundle reference descriptor.
        local_project_root: Local runtime project root.

    Returns:
        Extraction summary payload.
    """
    if not isinstance(bundle_reference, Mapping):
        raise TypeError("bundle_reference must be Mapping")
    if not isinstance(local_project_root, Path):
        raise TypeError("local_project_root must be Path")

    sidecar_path = cast(Path, bundle_reference["sidecar_path"])
    tar_path = cast(Path, bundle_reference["tar_path"])
    if not sidecar_path.exists():
        return {
            "status": "missing",
            "bundle_id": str(bundle_reference["bundle_id"]),
            "missing_reason": "sidecar_missing",
        }
    if not tar_path.exists():
        return {
            "status": "missing",
            "bundle_id": str(bundle_reference["bundle_id"]),
            "missing_reason": "tar_missing",
        }

    integrity_summary = verify_bundle_integrity(sidecar_path)
    extract_summary = extract_tar_gz_bundle(tar_path=tar_path, local_project_root=local_project_root)
    return {
        "status": "ok",
        "bundle_id": str(bundle_reference["bundle_id"]),
        "sidecar_path": integrity_summary["sidecar_path"],
        "tar_path": integrity_summary["tar_path"],
        "sample_role": bundle_reference.get("sample_role"),
        "shard_index": bundle_reference.get("shard_index"),
        "member_count": extract_summary["member_count"],
    }


def prepare_local_runtime_for_stage(
    *,
    stage_name: str,
    family_id: str,
    local_project_root: Path,
    drive_bundle_root: Path,
    sample_role: str | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    clean_before_run: bool = True,
    include_optional_control_negative: bool = False,
) -> Dict[str, Any]:
    """
    功能：为某个 notebook stage 准备本地 runtime 并解压依赖 bundle。

    Prepare the local runtime for one notebook stage by cleaning, pulling,
    verifying, and extracting dependency bundles.

    Args:
        stage_name: Canonical stage name.
        family_id: Family identifier.
        local_project_root: Local runtime project root.
        drive_bundle_root: Drive bundle root.
        sample_role: Optional sample role for PW01.
        shard_index: Optional shard index for shard-local stages.
        shard_count: Optional shard count for shard-local stages.
        clean_before_run: Whether the local runtime should be cleaned first.
        include_optional_control_negative: Whether optional control-negative inputs should be pulled.

    Returns:
        Preparation summary payload.
    """
    if stage_name not in STAGE_DIRECTORY_NAMES:
        raise ValueError(f"unsupported stage_name: {stage_name}")
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if not isinstance(local_project_root, Path):
        raise TypeError("local_project_root must be Path")
    if not isinstance(drive_bundle_root, Path):
        raise TypeError("drive_bundle_root must be Path")

    clean_summary: Dict[str, Any] | None = None
    if clean_before_run:
        clean_summary = safe_clean_local_runtime(local_project_root)
    ensure_directory(local_project_root)

    pulled_bundles: List[Dict[str, Any]] = []
    missing_bundles: List[str] = []
    extracted_bundle_ids: set[str] = set()

    initial_references = _initial_dependency_references(
        stage_name=stage_name,
        family_id=family_id,
        drive_bundle_root=drive_bundle_root,
    )
    for bundle_reference in initial_references:
        extract_result = _extract_one_dependency_bundle(
            bundle_reference=bundle_reference,
            local_project_root=local_project_root,
        )
        if extract_result["status"] == "missing":
            missing_bundles.append(str(extract_result["bundle_id"]))
            continue
        pulled_bundles.append(extract_result)
        extracted_bundle_ids.add(str(extract_result["bundle_id"]))

    if missing_bundles:
        summary_payload = {
            "当前 stage": stage_name,
            "当前 family_id": family_id,
            "本地 runtime 路径": _as_posix_path(local_project_root),
            "Drive bundle 路径": _as_posix_path(drive_bundle_root),
            "已拉取 bundle": [item["bundle_id"] for item in pulled_bundles],
            "缺失 bundle": missing_bundles,
            "下一步将执行的 stage": NEXT_STAGE_HINTS.get(stage_name),
        }
        _emit_runtime_summary("pw_local_runtime_prepare", summary_payload)
        raise FileNotFoundError(f"missing dependency bundles: {missing_bundles}")

    resolution = resolve_stage_dependencies(
        stage_name=stage_name,
        family_id=family_id,
        local_project_root=local_project_root,
        drive_bundle_root=drive_bundle_root,
        sample_role=sample_role,
        shard_index=shard_index,
        shard_count=shard_count,
        include_optional_control_negative=include_optional_control_negative,
    )
    for bundle_reference in cast(List[Dict[str, Any]], resolution["required_bundles"]):
        bundle_id = str(bundle_reference["bundle_id"])
        if bundle_id in extracted_bundle_ids:
            continue
        extract_result = _extract_one_dependency_bundle(
            bundle_reference=bundle_reference,
            local_project_root=local_project_root,
        )
        if extract_result["status"] == "missing":
            missing_bundles.append(str(extract_result["bundle_id"]))
            continue
        pulled_bundles.append(extract_result)
        extracted_bundle_ids.add(bundle_id)

    if missing_bundles:
        summary_payload = {
            "当前 stage": stage_name,
            "当前 family_id": family_id,
            "本地 runtime 路径": _as_posix_path(local_project_root),
            "Drive bundle 路径": _as_posix_path(drive_bundle_root),
            "已拉取 bundle": [item["bundle_id"] for item in pulled_bundles],
            "缺失 bundle": sorted(set(missing_bundles)),
            "下一步将执行的 stage": NEXT_STAGE_HINTS.get(stage_name),
        }
        _emit_runtime_summary("pw_local_runtime_prepare", summary_payload)
        raise FileNotFoundError(f"missing dependency bundles: {sorted(set(missing_bundles))}")

    verified_inputs = _verify_required_inputs_for_stage(
        stage_name=stage_name,
        family_id=family_id,
        local_project_root=local_project_root,
        drive_bundle_root=drive_bundle_root,
        include_optional_control_negative=include_optional_control_negative,
    )
    summary_payload = {
        "当前 stage": stage_name,
        "当前 family_id": family_id,
        "本地 runtime 路径": _as_posix_path(local_project_root),
        "Drive bundle 路径": _as_posix_path(drive_bundle_root),
        "已拉取 bundle": [item["bundle_id"] for item in pulled_bundles],
        "缺失 bundle": [],
        "下一步将执行的 stage": NEXT_STAGE_HINTS.get(stage_name),
        "已校验关键输入": verified_inputs,
    }
    if clean_summary is not None:
        summary_payload["本地 runtime 清理状态"] = clean_summary["status"]
    _emit_runtime_summary("pw_local_runtime_prepare", summary_payload)
    return {
        "status": "ready",
        "stage_name": stage_name,
        "family_id": family_id,
        "local_project_root": _as_posix_path(local_project_root),
        "drive_bundle_root": _as_posix_path(drive_bundle_root),
        "pulled_bundles": pulled_bundles,
        "missing_bundles": [],
        "dependency_resolution": resolution,
        "verified_input_paths": verified_inputs,
        "next_stage": NEXT_STAGE_HINTS.get(stage_name),
    }


def archive_local_runtime_for_stage(
    *,
    stage_name: str,
    family_id: str,
    local_project_root: Path,
    drive_bundle_root: Path,
    sample_role: str | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    clean_after_archive: bool = False,
) -> Dict[str, Any]:
    """
    功能：把某个 stage 的本地产物归档为可消费 bundle。

    Archive one stage's local outputs into a consumable bundle and sidecar.

    Args:
        stage_name: Canonical stage name.
        family_id: Family identifier.
        local_project_root: Local runtime project root.
        drive_bundle_root: Drive bundle root.
        sample_role: Optional sample role.
        shard_index: Optional shard index.
        shard_count: Optional shard count.
        clean_after_archive: Whether the local runtime should be cleaned after archiving.

    Returns:
        Archive summary payload.
    """
    archive_spec = verify_required_outputs_for_stage(
        stage_name=stage_name,
        family_id=family_id,
        local_project_root=local_project_root,
        drive_bundle_root=drive_bundle_root,
        sample_role=sample_role,
        shard_index=shard_index,
        shard_count=shard_count,
    )

    tar_path = cast(Path, archive_spec["tar_path"])
    sidecar_path = cast(Path, archive_spec["sidecar_path"])
    create_tar_gz_bundle(
        local_project_root=local_project_root,
        relative_paths=cast(List[str], archive_spec["relative_paths"]),
        tar_path=tar_path,
    )
    tar_sha256 = compute_file_sha256(tar_path)
    sidecar_payload = write_bundle_sidecar(
        sidecar_path=sidecar_path,
        family_id=family_id,
        stage_name=stage_name,
        bundle_kind=str(archive_spec["bundle_kind"]),
        tar_path=tar_path,
        tar_sha256=tar_sha256,
        local_project_root=local_project_root,
        relative_paths=cast(List[str], archive_spec["relative_paths"]),
        sample_role=sample_role,
        shard_index=shard_index,
        shard_count=cast(int | None, archive_spec["resolved_shard_count"]),
        extra_fields=cast(Mapping[str, Any], archive_spec["extra_fields"]),
    )

    clean_summary: Dict[str, Any] | None = None
    if clean_after_archive:
        clean_summary = safe_clean_local_runtime(local_project_root)

    summary_payload = {
        "当前 stage": stage_name,
        "当前 family_id": family_id,
        "本地 runtime 路径": _as_posix_path(local_project_root),
        "Drive bundle 路径": _as_posix_path(drive_bundle_root),
        "归档 tar.gz": normalize_path_value(tar_path),
        "bundle sidecar": normalize_path_value(sidecar_path),
        "tar_sha256": tar_sha256,
        "归档 relative_paths": cast(List[str], archive_spec["relative_paths"]),
        "下一步将执行的 stage": NEXT_STAGE_HINTS.get(stage_name),
    }
    if clean_summary is not None:
        summary_payload["归档后清理状态"] = clean_summary["status"]
    _emit_runtime_summary("pw_local_runtime_archive", summary_payload)

    return {
        "status": "archived",
        "stage_name": stage_name,
        "family_id": family_id,
        "local_project_root": _as_posix_path(local_project_root),
        "drive_bundle_root": _as_posix_path(drive_bundle_root),
        "tar_path": normalize_path_value(tar_path),
        "sidecar_path": normalize_path_value(sidecar_path),
        "tar_sha256": tar_sha256,
        "relative_paths": cast(List[str], archive_spec["relative_paths"]),
        "sidecar_payload": sidecar_payload,
        "clean_after_archive": bool(clean_after_archive),
    }
