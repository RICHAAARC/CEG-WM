"""
文件目的：对 stage 01 formal outputs 执行冻结定义审计，并判断是否可供后续阶段消费。
Module type: General module

职责边界：
1. 只读取 stage 01 已落盘的 formal outputs，不重跑 main/ watermark 机制。
2. 显式接受 canonical source pool 为 source truth，同时保留 representative root records 的 strong compatibility contract。
3. 以结构化 JSON 输出 stage 01 定义状态、strong compatibility 状态与 02/03/04 readiness。
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast


_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from scripts.notebook_runtime_common import (  # noqa: E402
    FORMAL_PACKAGE_DISCOVERY_SCOPE,
    FORMAL_STAGE_PACKAGE_ROLE,
    normalize_path_value,
    probe_stage_package_policy,
    write_json_atomic,
)


STAGE_NAME = "01_Paper_Full_Cuda"
SOURCE_TRUTH = "canonical_source_pool"
ROOT_CONTRACT_MODE = "strong_compatibility"
REPRESENTATIVE_ROOT_ROLE = "representative_summary_view"
PASS_STATUS = "passed"
BLOCK_STATUS = "blocked"
SUCCESS_STATUS_TOKENS = {"ok", "success", "passed"}
FORMAL_PACKAGE_SUCCESS_STATUS_TOKENS = {"generated", "ok", "success", "passed"}
CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH = "artifacts/stage_01_canonical_source_pool/source_pool_manifest.json"
STAGE_02_SOURCE_CONTRACT_RELATIVE_PATH = "artifacts/parallel_attestation_statistics_input_contract.json"
ROOT_RECORD_RELATIVE_PATHS = {
    "embed_record": "records/embed_record.json",
    "detect_record": "records/detect_record.json",
}
STAGE_03_REQUIRED_PACKAGE_PATHS = {
    "source_stage_manifest": "artifacts/stage_manifest.json",
    "source_runtime_config_snapshot": "runtime_metadata/runtime_config_snapshot.yaml",
    "source_thresholds_artifact": "artifacts/thresholds/thresholds_artifact.json",
    "canonical_source_pool_manifest": CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH,
}
STAGE_04_REQUIRED_STAGE_01_FIELDS = (
    "stage_status",
    "workflow_summary_status",
    "formal_package_status",
    "attestation_evidence_status",
    "stage_01_canonical_source_pool_manifest_package_relative_path",
    "stage_01_representative_root_records",
)


def _append_unique(items: List[str], item: str) -> None:
    """
    功能：向列表追加唯一字符串。 

    Append one string to a list only when it is not already present.

    Args:
        items: Mutable string list.
        item: Candidate item.

    Returns:
        None.
    """
    if item not in items:
        items.append(item)


def _normalize_status_token(value: Any) -> str:
    """
    功能：规范化状态字符串。 

    Normalize a status-like value into a lowercase token.

    Args:
        value: Candidate status value.

    Returns:
        Normalized lowercase token, or an empty string when unavailable.
    """
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _status_matches(value: Any, allowed_tokens: Sequence[str]) -> bool:
    """
    功能：判断状态是否命中允许集合。 

    Determine whether one status-like value belongs to an allowed token set.

    Args:
        value: Candidate status value.
        allowed_tokens: Allowed normalized tokens.

    Returns:
        True when the value matches one allowed token.
    """
    normalized = _normalize_status_token(value)
    return bool(normalized) and normalized in {token.strip().lower() for token in allowed_tokens}


def _record_checked_path(
    checked_paths: Dict[str, Dict[str, Any]],
    label: str,
    path_value: Any,
    *,
    exists: bool,
    package_relative: bool = False,
) -> None:
    """
    功能：记录一个已检查路径。 

    Record one checked filesystem or package-relative path.

    Args:
        checked_paths: Mutable checked-path mapping.
        label: Stable label.
        path_value: Path-like value.
        exists: Whether the path resolved successfully.
        package_relative: Whether the path is package-relative.

    Returns:
        None.
    """
    payload = {
        "path": normalize_path_value(path_value) if isinstance(path_value, Path) else str(path_value),
        "exists": bool(exists),
    }
    if package_relative:
        payload["path_kind"] = "package_relative"
    checked_paths[label] = payload


def _load_json_mapping(
    path_obj: Path,
    label: str,
    blocking_reasons: List[str],
    checked_paths: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    功能：按容错方式读取 JSON mapping。 

    Load a JSON file and return an empty mapping when the file is missing,
    invalid, or not a dict-root payload.

    Args:
        path_obj: JSON file path.
        label: Stable path label.
        blocking_reasons: Mutable blocking-reason list.
        checked_paths: Mutable checked-path mapping.

    Returns:
        Parsed dict payload, or an empty mapping when unavailable.
    """
    exists = path_obj.exists() and path_obj.is_file()
    _record_checked_path(checked_paths, label, path_obj, exists=exists)
    if not exists:
        _append_unique(blocking_reasons, f"{label}_missing")
        return {}
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except Exception as exc:
        checked_paths[label]["error"] = f"{type(exc).__name__}: {exc}"
        _append_unique(blocking_reasons, f"{label}_invalid_json")
        return {}
    if not isinstance(payload, dict):
        _append_unique(blocking_reasons, f"{label}_not_mapping")
        return {}
    return cast(Dict[str, Any], payload)


def _load_package_members(
    package_path: Optional[Path],
    blocking_reasons: List[str],
    checked_paths: Dict[str, Dict[str, Any]],
) -> List[str]:
    """
    功能：读取 package ZIP 中的成员路径列表。 

    Load the normalized member paths from one stage package ZIP.

    Args:
        package_path: Package ZIP path.
        blocking_reasons: Mutable blocking-reason list.
        checked_paths: Mutable checked-path mapping.

    Returns:
        Normalized ZIP member list.
    """
    if not isinstance(package_path, Path):
        _record_checked_path(checked_paths, "package_path", "<absent>", exists=False)
        _append_unique(blocking_reasons, "package_path_missing")
        return []
    exists = package_path.exists() and package_path.is_file()
    _record_checked_path(checked_paths, "package_path", package_path, exists=exists)
    if not exists:
        _append_unique(blocking_reasons, "package_path_missing")
        return []
    try:
        with zipfile.ZipFile(package_path, "r") as archive:
            return [name.rstrip("/") for name in archive.namelist() if name and not name.endswith("/")]
    except Exception as exc:
        checked_paths["package_path"]["error"] = f"{type(exc).__name__}: {exc}"
        _append_unique(blocking_reasons, "package_zip_invalid")
        return []


def _package_has_path(
    package_members: Iterable[str],
    relative_path: str,
    checked_paths: Dict[str, Dict[str, Any]],
    label: str,
) -> bool:
    """
    功能：判断 package 是否包含给定相对路径。 

    Determine whether one package-relative path exists inside the ZIP.

    Args:
        package_members: ZIP member iterable.
        relative_path: Package-relative path.
        checked_paths: Mutable checked-path mapping.
        label: Stable checked-path label.

    Returns:
        True when the package-relative path exists.
    """
    normalized_relative_path = relative_path.strip().lstrip("/")
    exists = normalized_relative_path in set(package_members)
    _record_checked_path(
        checked_paths,
        label,
        normalized_relative_path,
        exists=exists,
        package_relative=True,
    )
    return exists


def _require_string_field(payload: Mapping[str, Any], field_name: str) -> bool:
    """
    功能：判断字段是否为非空字符串。 

    Determine whether one mapping field is a non-empty string.

    Args:
        payload: Candidate mapping.
        field_name: Field name.

    Returns:
        True when the field is a non-empty string.
    """
    field_value = payload.get(field_name)
    return isinstance(field_value, str) and bool(field_value.strip())


def _resolve_package_path(
    package_manifest: Mapping[str, Any],
    blocking_reasons: List[str],
    checked_paths: Dict[str, Dict[str, Any]],
) -> Optional[Path]:
    """
    功能：从 external package manifest 解析 package ZIP 路径。 

    Resolve the package ZIP path from the external package manifest.

    Args:
        package_manifest: External package manifest mapping.
        blocking_reasons: Mutable blocking-reason list.
        checked_paths: Mutable checked-path mapping.

    Returns:
        Resolved ZIP path, or None when unavailable.
    """
    package_path_value = package_manifest.get("package_path")
    if not isinstance(package_path_value, str) or not package_path_value.strip():
        _record_checked_path(checked_paths, "package_manifest.package_path", "<absent>", exists=False)
        _append_unique(blocking_reasons, "package_manifest_package_path_missing")
        return None
    package_path = Path(package_path_value).expanduser()
    if not package_path.is_absolute():
        package_path = (_BOOTSTRAP_REPO_ROOT / package_path).resolve()
    else:
        package_path = package_path.resolve()
    _record_checked_path(
        checked_paths,
        "package_manifest.package_path",
        package_path,
        exists=package_path.exists() and package_path.is_file(),
    )
    return package_path


def _check_formal_package_policy(
    package_path: Optional[Path],
    blocking_reasons: List[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    功能：校验 formal package policy。 

    Validate the formal package role and discovery-scope policy.

    Args:
        package_path: Package ZIP path.
        blocking_reasons: Mutable blocking-reason list.

    Returns:
        Tuple of formal package policy status and the probe payload.
    """
    if not isinstance(package_path, Path) or not package_path.exists() or not package_path.is_file():
        _append_unique(blocking_reasons, "formal_package_policy.package_missing")
        return BLOCK_STATUS, {}

    package_policy_probe = probe_stage_package_policy(package_path)
    package_role = package_policy_probe.get("package_role")
    package_discovery_scope = package_policy_probe.get("package_discovery_scope")

    if package_role != FORMAL_STAGE_PACKAGE_ROLE:
        _append_unique(blocking_reasons, "formal_package_policy.package_role_invalid")
    if package_discovery_scope != FORMAL_PACKAGE_DISCOVERY_SCOPE:
        _append_unique(blocking_reasons, "formal_package_policy.package_discovery_scope_invalid")
    if package_policy_probe.get("diagnostics_like") is True:
        _append_unique(blocking_reasons, "formal_package_policy.diagnostics_like_input")
    if package_policy_probe.get("formal_package_eligible") is not True:
        _append_unique(blocking_reasons, "formal_package_policy.not_discoverable_formal")

    status = PASS_STATUS
    for reason_code in blocking_reasons:
        if reason_code.startswith("formal_package_policy."):
            status = BLOCK_STATUS
            break
    return status, package_policy_probe


def _check_canonical_source_entries(
    run_root: Path,
    canonical_source_pool_manifest: Mapping[str, Any],
    blocking_reasons: List[str],
    checked_paths: Dict[str, Dict[str, Any]],
) -> bool:
    """
    功能：校验 canonical source entries 的最小字段完整性。 

    Validate the minimum required fields of canonical source entries.

    Args:
        run_root: Stage run root.
        canonical_source_pool_manifest: Canonical source-pool manifest payload.
        blocking_reasons: Mutable blocking-reason list.
        checked_paths: Mutable checked-path mapping.

    Returns:
        True when the canonical source entries satisfy the minimum field set.
    """
    entries = canonical_source_pool_manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        _append_unique(blocking_reasons, "canonical_source_pool.entries_missing")
        return False

    entry_fields_ok = True
    for entry in entries:
        if not isinstance(entry, dict):
            _append_unique(blocking_reasons, "canonical_source_pool.entry_not_mapping")
            entry_fields_ok = False
            continue
        source_entry_relative_path = entry.get("source_entry_package_relative_path")
        if not isinstance(source_entry_relative_path, str) or not source_entry_relative_path:
            _append_unique(blocking_reasons, "canonical_source_pool.source_entry_package_relative_path_missing")
            entry_fields_ok = False
            continue

        source_entry_path = run_root / source_entry_relative_path
        source_entry_payload = _load_json_mapping(
            source_entry_path,
            f"source_entry::{entry.get('prompt_index', 'unknown')}",
            blocking_reasons,
            checked_paths,
        )
        if not source_entry_payload:
            _append_unique(blocking_reasons, "canonical_source_pool.source_entry_unreadable")
            entry_fields_ok = False
            continue

        required_string_fields = (
            "prompt_text",
            "prompt_sha256",
            "embed_record_path",
            "embed_record_package_relative_path",
            "detect_record_path",
            "detect_record_package_relative_path",
            "runtime_config_path",
            "runtime_config_package_relative_path",
        )
        if not isinstance(source_entry_payload.get("prompt_index"), int):
            _append_unique(blocking_reasons, "canonical_source_pool.prompt_index_missing")
            entry_fields_ok = False
        for field_name in required_string_fields:
            if not _require_string_field(source_entry_payload, field_name):
                _append_unique(blocking_reasons, f"canonical_source_pool.{field_name}_missing")
                entry_fields_ok = False
        for node_name in ("attestation_statement", "attestation_bundle", "attestation_result", "source_image"):
            if not isinstance(source_entry_payload.get(node_name), dict):
                _append_unique(blocking_reasons, f"canonical_source_pool.{node_name}_missing")
                entry_fields_ok = False
    return entry_fields_ok


def _check_stage_01_definition(
    run_root: Path,
    stage_manifest: Mapping[str, Any],
    workflow_summary: Mapping[str, Any],
    package_manifest: Mapping[str, Any],
    canonical_source_pool_manifest: Mapping[str, Any],
    package_members: Sequence[str],
    blocking_reasons: List[str],
    warnings: List[str],
    checked_paths: Dict[str, Dict[str, Any]],
) -> Tuple[str, str, bool]:
    """
    功能：校验 stage 01 冻结定义。 

    Validate the frozen stage-01 definition, including stage status, workflow
    status, canonical source pool completeness, and source-entry integrity.

    Args:
        run_root: Stage run root.
        stage_manifest: Stage manifest payload.
        workflow_summary: Workflow summary payload.
        package_manifest: External package manifest payload.
        canonical_source_pool_manifest: Canonical source-pool manifest payload.
        package_members: Package ZIP member list.
        blocking_reasons: Mutable blocking-reason list.
        warnings: Mutable warning list.
        checked_paths: Mutable checked-path mapping.

    Returns:
        Tuple of definition status, canonical source-pool status, and whether
        the representative root records node is present.
    """
    if stage_manifest.get("stage_name") != STAGE_NAME:
        _append_unique(blocking_reasons, "definition.stage_name_invalid")
    if package_manifest.get("stage_name") != STAGE_NAME:
        _append_unique(blocking_reasons, "definition.package_manifest_stage_name_invalid")
    if not _status_matches(stage_manifest.get("stage_status"), sorted(SUCCESS_STATUS_TOKENS)):
        _append_unique(blocking_reasons, "definition.stage_status_not_success")
    if not _status_matches(workflow_summary.get("status"), sorted(SUCCESS_STATUS_TOKENS)):
        _append_unique(blocking_reasons, "definition.workflow_summary_status_not_success")
    if not _status_matches(stage_manifest.get("formal_package_status"), sorted(FORMAL_PACKAGE_SUCCESS_STATUS_TOKENS)):
        _append_unique(blocking_reasons, "definition.formal_package_status_not_success")

    prompt_count = canonical_source_pool_manifest.get("prompt_count")
    entry_count = canonical_source_pool_manifest.get("entry_count")
    if not isinstance(prompt_count, int) or prompt_count <= 0:
        _append_unique(blocking_reasons, "canonical_source_pool.prompt_count_invalid")
    if not isinstance(entry_count, int) or entry_count <= 0:
        _append_unique(blocking_reasons, "canonical_source_pool.entry_count_invalid")
    if isinstance(prompt_count, int) and isinstance(entry_count, int) and prompt_count != entry_count:
        _append_unique(blocking_reasons, "canonical_source_pool.prompt_entry_count_mismatch")

    entries = canonical_source_pool_manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        _append_unique(blocking_reasons, "canonical_source_pool.entries_missing")
    elif isinstance(entry_count, int) and len(entries) != entry_count:
        _append_unique(blocking_reasons, "canonical_source_pool.entries_length_mismatch")

    manifest_relative_path = canonical_source_pool_manifest.get("manifest_package_relative_path")
    if not isinstance(manifest_relative_path, str) or not manifest_relative_path:
        _append_unique(blocking_reasons, "canonical_source_pool.manifest_package_relative_path_missing")
    elif not _package_has_path(
        package_members,
        manifest_relative_path,
        checked_paths,
        "package::canonical_source_pool_manifest",
    ):
        _append_unique(blocking_reasons, "canonical_source_pool.manifest_not_packaged")

    entries_ok = _check_canonical_source_entries(
        run_root,
        canonical_source_pool_manifest,
        blocking_reasons,
        checked_paths,
    )
    if stage_manifest.get("diagnostics_status") == "generated":
        _append_unique(warnings, "diagnostics_package_present_but_not_formal_input")

    representative_root_records = canonical_source_pool_manifest.get("representative_root_records")
    representative_root_present = isinstance(representative_root_records, dict) and bool(representative_root_records)

    canonical_status = PASS_STATUS
    for reason_code in blocking_reasons:
        if reason_code.startswith("canonical_source_pool."):
            canonical_status = BLOCK_STATUS
            break
    definition_status = PASS_STATUS
    for reason_code in blocking_reasons:
        if reason_code.startswith("definition.") or reason_code.startswith("canonical_source_pool."):
            definition_status = BLOCK_STATUS
            break
    if not entries_ok:
        definition_status = BLOCK_STATUS
        canonical_status = BLOCK_STATUS
    return definition_status, canonical_status, representative_root_present


def _check_strong_compatibility(
    run_root: Path,
    stage_manifest: Mapping[str, Any],
    canonical_source_pool_manifest: Mapping[str, Any],
    package_members: Sequence[str],
    blocking_reasons: List[str],
    checked_paths: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    """
    功能：校验 representative root records 的 strong compatibility contract。 

    Validate the current strong-compatibility contract: canonical source pool is
    source truth, while representative root records remain required outputs.

    Args:
        run_root: Stage run root.
        stage_manifest: Stage manifest payload.
        canonical_source_pool_manifest: Canonical source-pool manifest payload.
        package_members: Package ZIP member list.
        blocking_reasons: Mutable blocking-reason list.
        checked_paths: Mutable checked-path mapping.

    Returns:
        Tuple of strong-compatibility status and representative-root status.
    """
    expected_stage_fields = {
        "stage_01_root_contract_mode": ROOT_CONTRACT_MODE,
        "stage_01_root_records_required": True,
        "stage_01_source_truth": SOURCE_TRUTH,
        "stage_01_representative_root_role": REPRESENTATIVE_ROOT_ROLE,
    }
    for field_name, expected_value in expected_stage_fields.items():
        if stage_manifest.get(field_name) != expected_value:
            _append_unique(blocking_reasons, f"strong_compatibility.{field_name}_invalid")

    expected_manifest_fields = {
        "source_truth": SOURCE_TRUTH,
        "root_contract_mode": ROOT_CONTRACT_MODE,
        "root_records_required": True,
        "representative_root_role": REPRESENTATIVE_ROOT_ROLE,
    }
    for field_name, expected_value in expected_manifest_fields.items():
        if canonical_source_pool_manifest.get(field_name) != expected_value:
            _append_unique(blocking_reasons, f"strong_compatibility.canonical_manifest_{field_name}_invalid")

    representative_root_records = canonical_source_pool_manifest.get("representative_root_records")
    representative_root_payload = cast(Dict[str, Any], representative_root_records) if isinstance(representative_root_records, dict) else {}
    expected_representative_fields = {
        "view_role": REPRESENTATIVE_ROOT_ROLE,
        "contract_mode": ROOT_CONTRACT_MODE,
        "source_truth": SOURCE_TRUTH,
        "root_records_required": True,
    }
    for field_name, expected_value in expected_representative_fields.items():
        if representative_root_payload.get(field_name) != expected_value:
            _append_unique(blocking_reasons, f"strong_compatibility.representative_root_{field_name}_invalid")

    for label, relative_path in ROOT_RECORD_RELATIVE_PATHS.items():
        root_path = run_root / relative_path
        exists_on_disk = root_path.exists() and root_path.is_file()
        _record_checked_path(checked_paths, f"root::{label}", root_path, exists=exists_on_disk)
        if not exists_on_disk:
            _append_unique(blocking_reasons, f"strong_compatibility.{label}_missing_on_disk")
        if not _package_has_path(package_members, relative_path, checked_paths, f"package::root::{label}"):
            _append_unique(blocking_reasons, f"strong_compatibility.{label}_not_packaged")

    representative_root_status = PASS_STATUS
    strong_compatibility_status = PASS_STATUS
    for reason_code in blocking_reasons:
        if reason_code.startswith("strong_compatibility."):
            representative_root_status = BLOCK_STATUS
            strong_compatibility_status = BLOCK_STATUS
            break
    return strong_compatibility_status, representative_root_status


def _check_stage_02_readiness(
    run_root: Path,
    stage_manifest: Mapping[str, Any],
    package_members: Sequence[str],
    blocking_reasons: List[str],
    warnings: List[str],
    checked_paths: Dict[str, Dict[str, Any]],
) -> bool:
    """
    功能：校验 stage 02 readiness。 

    Validate that the stage-01 outputs provide the minimum contract required by
    stage 02.

    Args:
        run_root: Stage run root.
        stage_manifest: Stage manifest payload.
        package_members: Package ZIP member list.
        blocking_reasons: Mutable blocking-reason list.
        warnings: Mutable warning list.
        checked_paths: Mutable checked-path mapping.

    Returns:
        True when stage 02 can consume the stage-01 outputs.
    """
    contract_relative_path = stage_manifest.get(
        "parallel_attestation_statistics_input_contract_package_relative_path",
        STAGE_02_SOURCE_CONTRACT_RELATIVE_PATH,
    )
    if not isinstance(contract_relative_path, str) or not contract_relative_path:
        _append_unique(blocking_reasons, "stage_02.source_contract_relative_path_missing")
        return False
    contract_path = run_root / contract_relative_path
    contract_payload = _load_json_mapping(
        contract_path,
        "stage_02_source_contract",
        blocking_reasons,
        checked_paths,
    )
    if not _package_has_path(package_members, contract_relative_path, checked_paths, "package::stage_02_source_contract"):
        _append_unique(blocking_reasons, "stage_02.source_contract_not_packaged")
    if not contract_payload:
        return False
    if contract_payload.get("artifact_type") != "parallel_attestation_statistics_input_contract":
        _append_unique(blocking_reasons, "stage_02.source_contract_artifact_type_invalid")
    if contract_payload.get("contract_role") != "source_contract":
        _append_unique(blocking_reasons, "stage_02.source_contract_role_invalid")
    if contract_payload.get("source_authority") != SOURCE_TRUTH:
        _append_unique(blocking_reasons, "stage_02.source_contract_source_authority_invalid")
    if contract_payload.get("source_records_available") is not True:
        _append_unique(blocking_reasons, "stage_02.source_contract_records_unavailable")

    canonical_manifest_relative_path = contract_payload.get("canonical_source_pool_manifest_package_relative_path")
    if not isinstance(canonical_manifest_relative_path, str) or not canonical_manifest_relative_path:
        _append_unique(blocking_reasons, "stage_02.source_contract_canonical_manifest_relative_path_missing")
    elif not _package_has_path(
        package_members,
        canonical_manifest_relative_path,
        checked_paths,
        "package::stage_02_canonical_source_pool_manifest",
    ):
        _append_unique(blocking_reasons, "stage_02.source_contract_canonical_manifest_not_packaged")

    contract_records = contract_payload.get("records")
    if not isinstance(contract_records, list) or not contract_records:
        _append_unique(blocking_reasons, "stage_02.source_contract_records_missing")
    else:
        for record_entry in contract_records:
            if not isinstance(record_entry, dict):
                _append_unique(blocking_reasons, "stage_02.source_contract_record_not_mapping")
                continue
            for field_name in (
                "package_relative_path",
                "canonical_source_entry_package_relative_path",
                "embed_record_package_relative_path",
                "runtime_config_package_relative_path",
            ):
                field_value = record_entry.get(field_name)
                if not isinstance(field_value, str) or not field_value:
                    _append_unique(blocking_reasons, f"stage_02.source_contract_{field_name}_missing")
                    continue
                if not _package_has_path(
                    package_members,
                    field_value,
                    checked_paths,
                    f"package::stage_02::{field_name}::{record_entry.get('prompt_index', 'unknown')}",
                ):
                    _append_unique(blocking_reasons, f"stage_02.source_contract_{field_name}_not_packaged")
        if contract_payload.get("direct_stats_ready") is not True:
            _append_unique(warnings, "stage_02.source_contract_direct_stats_not_ready")

    for reason_code in blocking_reasons:
        if reason_code.startswith("stage_02."):
            return False
    return True


def _check_stage_03_readiness(
    stage_manifest: Mapping[str, Any],
    canonical_source_pool_manifest: Mapping[str, Any],
    package_members: Sequence[str],
    blocking_reasons: List[str],
    warnings: List[str],
    checked_paths: Dict[str, Dict[str, Any]],
) -> bool:
    """
    功能：校验 stage 03 readiness。 

    Validate that the stage-01 package exposes the minimum lineage and source
    artifacts required by stage 03.

    Args:
        stage_manifest: Stage manifest payload.
        canonical_source_pool_manifest: Canonical source-pool manifest payload.
        package_members: Package ZIP member list.
        blocking_reasons: Mutable blocking-reason list.
        warnings: Mutable warning list.
        checked_paths: Mutable checked-path mapping.

    Returns:
        True when stage 03 can consume the stage-01 package.
    """
    if stage_manifest.get("stage_name") != STAGE_NAME:
        _append_unique(blocking_reasons, "stage_03.source_stage_name_invalid")

    for label, relative_path in STAGE_03_REQUIRED_PACKAGE_PATHS.items():
        if not _package_has_path(package_members, relative_path, checked_paths, f"package::stage_03::{label}"):
            _append_unique(blocking_reasons, f"stage_03.{label}_not_packaged")

    prompt_snapshot_path = stage_manifest.get("prompt_snapshot_path")
    if not isinstance(prompt_snapshot_path, str) or not prompt_snapshot_path.strip() or prompt_snapshot_path == "<absent>":
        _append_unique(warnings, "stage_03.prompt_snapshot_absent")

    if canonical_source_pool_manifest.get("entry_count") in {None, 0}:
        _append_unique(blocking_reasons, "stage_03.canonical_source_pool_unavailable")

    for reason_code in blocking_reasons:
        if reason_code.startswith("stage_03."):
            return False
    return True


def _check_stage_04_readiness(
    stage_manifest: Mapping[str, Any],
    package_policy_probe: Mapping[str, Any],
    attestation_evidence_status: str,
    canonical_source_pool_status: str,
    representative_root_status: str,
    blocking_reasons: List[str],
) -> bool:
    """
    功能：校验 stage 04 readiness。 

    Validate that the stage-01 outputs satisfy the minimum stage-04 readiness
    contract without changing stage-04 signoff semantics.

    Args:
        stage_manifest: Stage manifest payload.
        package_policy_probe: Formal package policy probe payload.
        attestation_evidence_status: Resolved attestation-evidence status token.
        canonical_source_pool_status: Canonical source-pool audit status.
        representative_root_status: Representative-root audit status.
        blocking_reasons: Mutable blocking-reason list.

    Returns:
        True when stage 04 can accept the stage-01 package as formal input.
    """
    if package_policy_probe.get("diagnostics_like") is True:
        _append_unique(blocking_reasons, "stage_04.diagnostics_like_formal_input")
    if not _status_matches(attestation_evidence_status, sorted(SUCCESS_STATUS_TOKENS)):
        _append_unique(blocking_reasons, "stage_04.attestation_evidence_status_not_success")
    if canonical_source_pool_status != PASS_STATUS:
        _append_unique(blocking_reasons, "stage_04.canonical_source_pool_not_ready")
    if representative_root_status != PASS_STATUS:
        _append_unique(blocking_reasons, "stage_04.representative_root_not_ready")
    for field_name in STAGE_04_REQUIRED_STAGE_01_FIELDS:
        if field_name == "stage_01_representative_root_records":
            if not isinstance(stage_manifest.get(field_name), dict) or not stage_manifest.get(field_name):
                _append_unique(blocking_reasons, f"stage_04.{field_name}_missing")
            continue
        if not stage_manifest.get(field_name):
            _append_unique(blocking_reasons, f"stage_04.{field_name}_missing")

    for reason_code in blocking_reasons:
        if reason_code.startswith("stage_04."):
            return False
    return True


def _resolve_attestation_evidence_status(
    stage_manifest: Mapping[str, Any],
    workflow_summary: Mapping[str, Any],
) -> str:
    """
    功能：解析 attestation evidence 状态。 

    Resolve the canonical attestation-evidence status token from stage manifest
    and workflow summary payloads.

    Args:
        stage_manifest: Stage manifest payload.
        workflow_summary: Workflow summary payload.

    Returns:
        Resolved attestation-evidence status token.
    """
    stage_manifest_status = stage_manifest.get("attestation_evidence_status")
    if isinstance(stage_manifest_status, str) and stage_manifest_status.strip():
        return stage_manifest_status.strip()
    resolution = workflow_summary.get("attestation_evidence_resolution")
    if isinstance(resolution, dict):
        resolution_status = resolution.get("overall_status")
        if isinstance(resolution_status, str) and resolution_status.strip():
            return resolution_status.strip()
    return "<absent>"


def run_stage_01_output_audit(
    *,
    run_root: Path,
    stage_manifest_path: Path,
    workflow_summary_path: Path,
    package_manifest_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """
    功能：执行 stage 01 outputs audit 并写出 JSON summary。

    Audit the frozen stage-01 outputs, evaluate strong compatibility and
    downstream readiness, and persist a structured JSON summary.

    Args:
        run_root: Stage run root.
        stage_manifest_path: Stage manifest path.
        workflow_summary_path: Workflow summary path.
        package_manifest_path: External package manifest path.
        output_path: Audit summary output path.

    Returns:
        Structured audit summary mapping.

    Raises:
        TypeError: If any required path argument is not a Path.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(stage_manifest_path, Path):
        raise TypeError("stage_manifest_path must be Path")
    if not isinstance(workflow_summary_path, Path):
        raise TypeError("workflow_summary_path must be Path")
    if not isinstance(package_manifest_path, Path):
        raise TypeError("package_manifest_path must be Path")
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be Path")

    blocking_reasons: List[str] = []
    warnings: List[str] = []
    checked_paths: Dict[str, Dict[str, Any]] = {}

    stage_manifest = _load_json_mapping(stage_manifest_path, "stage_manifest", blocking_reasons, checked_paths)
    workflow_summary = _load_json_mapping(workflow_summary_path, "workflow_summary", blocking_reasons, checked_paths)
    package_manifest = _load_json_mapping(package_manifest_path, "package_manifest", blocking_reasons, checked_paths)

    package_path = _resolve_package_path(package_manifest, blocking_reasons, checked_paths)
    package_members = _load_package_members(package_path, blocking_reasons, checked_paths)
    formal_package_policy_status, package_policy_probe = _check_formal_package_policy(package_path, blocking_reasons)

    canonical_source_pool_manifest_path = run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH
    canonical_source_pool_manifest = _load_json_mapping(
        canonical_source_pool_manifest_path,
        "canonical_source_pool_manifest",
        blocking_reasons,
        checked_paths,
    )
    definition_status, canonical_source_pool_status, _ = _check_stage_01_definition(
        run_root,
        stage_manifest,
        workflow_summary,
        package_manifest,
        canonical_source_pool_manifest,
        package_members,
        blocking_reasons,
        warnings,
        checked_paths,
    )
    strong_compatibility_status, representative_root_status = _check_strong_compatibility(
        run_root,
        stage_manifest,
        canonical_source_pool_manifest,
        package_members,
        blocking_reasons,
        checked_paths,
    )
    stage_02_ready = _check_stage_02_readiness(
        run_root,
        stage_manifest,
        package_members,
        blocking_reasons,
        warnings,
        checked_paths,
    )
    stage_03_ready = _check_stage_03_readiness(
        stage_manifest,
        canonical_source_pool_manifest,
        package_members,
        blocking_reasons,
        warnings,
        checked_paths,
    )
    attestation_evidence_status = _resolve_attestation_evidence_status(stage_manifest, workflow_summary)
    stage_04_ready = _check_stage_04_readiness(
        stage_manifest,
        package_policy_probe,
        attestation_evidence_status,
        canonical_source_pool_status,
        representative_root_status,
        blocking_reasons,
    )

    overall_status = PASS_STATUS
    if (
        formal_package_policy_status != PASS_STATUS
        or definition_status != PASS_STATUS
        or strong_compatibility_status != PASS_STATUS
        or not stage_02_ready
        or not stage_03_ready
        or not stage_04_ready
    ):
        overall_status = BLOCK_STATUS

    summary: Dict[str, Any] = {
        "stage_name": str(stage_manifest.get("stage_name", STAGE_NAME)),
        "stage_run_id": str(stage_manifest.get("stage_run_id", workflow_summary.get("stage_run_id", "<absent>"))),
        "source_truth": SOURCE_TRUTH,
        "root_contract_mode": ROOT_CONTRACT_MODE,
        "root_records_required": True,
        "overall_status": overall_status,
        "definition_status": definition_status,
        "strong_compatibility_status": strong_compatibility_status,
        "stage_02_ready": stage_02_ready,
        "stage_03_ready": stage_03_ready,
        "stage_04_ready": stage_04_ready,
        "canonical_source_pool_status": canonical_source_pool_status,
        "attestation_evidence_status": attestation_evidence_status,
        "representative_root_status": representative_root_status,
        "formal_package_policy_status": formal_package_policy_status,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "checked_paths": checked_paths,
    }
    write_json_atomic(output_path, summary)
    return summary


def main() -> int:
    """
    功能：stage 01 audit CLI 入口。 

    Execute the stage-01 audit from the command line.

    Args:
        None.

    Returns:
        Process-style exit code.
    """
    parser = argparse.ArgumentParser(description="Audit stage-01 outputs and downstream readiness.")
    parser.add_argument("--run-root", required=True, help="Stage-01 run root.")
    parser.add_argument("--stage-manifest", required=True, help="Stage manifest path.")
    parser.add_argument("--workflow-summary", required=True, help="Workflow summary path.")
    parser.add_argument("--package-manifest", required=True, help="External package manifest path.")
    parser.add_argument("--output", required=True, help="Audit summary output path.")
    args = parser.parse_args()

    summary = run_stage_01_output_audit(
        run_root=Path(args.run_root).expanduser().resolve(),
        stage_manifest_path=Path(args.stage_manifest).expanduser().resolve(),
        workflow_summary_path=Path(args.workflow_summary).expanduser().resolve(),
        package_manifest_path=Path(args.package_manifest).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if summary.get("overall_status") == PASS_STATUS else 1


if __name__ == "__main__":
    sys.exit(main())