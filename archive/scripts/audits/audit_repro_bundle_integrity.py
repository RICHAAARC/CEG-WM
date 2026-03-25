"""
文件目的：审计 repro bundle 完整性与指针哈希一致性。
Module type: Core innovation module

审计职责：
1. 校验 artifacts/repro_bundle/manifest.json 与 pointers.json 存在。
2. 校验 manifest 必备锚点字段完整且非空。
3. 校验 pointers 中每个 path 对应文件存在且 sha256 匹配。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REQUIRED_MANIFEST_FIELDS = [
    "cfg_digest",
    "plan_digest",
    "thresholds_digest",
    "threshold_metadata_digest",
    "impl_digest",
    "fusion_rule_version",
    "attack_protocol_version",
    "attack_protocol_digest",
    "policy_path",
]


def _sha256_file(file_path: Path) -> str:
    """
    功能：计算文件 SHA256。 

    Compute SHA256 for a file.

    Args:
        file_path: Path to file.

    Returns:
        SHA256 hex digest.
    """
    hasher = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_json_dict(file_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    功能：加载 JSON dict。 

    Load JSON file and enforce dict root.

    Args:
        file_path: Target JSON file path.

    Returns:
        Tuple of (dict_or_none, error_or_none).
    """
    if not file_path.exists() or not file_path.is_file():
        return None, f"missing_file: {file_path}"
    try:
        loaded_obj = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"json_parse_failed: {file_path}: {type(exc).__name__}: {exc}"
    if not isinstance(loaded_obj, dict):
        return None, f"json_root_not_dict: {file_path}"
    return loaded_obj, None


def _validate_manifest(manifest: Dict[str, Any]) -> List[str]:
    """
    功能：校验 manifest 锚点字段完整性。 

    Validate required anchor fields in repro bundle manifest.

    Args:
        manifest: Manifest dictionary.

    Returns:
        List of validation issues.
    """
    issues: List[str] = []
    for field_name in REQUIRED_MANIFEST_FIELDS:
        value = manifest.get(field_name)
        if not isinstance(value, str) or not value.strip() or value == "<absent>":
            issues.append(f"manifest_missing_or_empty: {field_name}")
    return issues


def _validate_pointers(run_root: Path, pointers: Dict[str, Any]) -> List[str]:
    """
    功能：校验 pointers 中路径存在且哈希一致。 

    Validate pointer entries against file existence and SHA256 digest.

    Args:
        run_root: Run root directory.
        pointers: Pointers dictionary.

    Returns:
        List of validation issues.
    """
    issues: List[str] = []
    files_obj = pointers.get("files")
    if not isinstance(files_obj, list):
        return ["pointers.files must be list"]

    for item in files_obj:
        if not isinstance(item, dict):
            issues.append("pointers.files item must be dict")
            continue
        relative_path = item.get("path")
        recorded_sha256 = item.get("sha256")
        if not isinstance(relative_path, str) or not relative_path:
            issues.append("pointers.files.path must be non-empty str")
            continue
        if not isinstance(recorded_sha256, str) or not recorded_sha256:
            issues.append(f"pointers.files.sha256 missing: {relative_path}")
            continue

        file_path = (run_root / relative_path).resolve()
        try:
            file_path.relative_to(run_root.resolve())
        except ValueError:
            issues.append(f"path_escape_detected: {relative_path}")
            continue

        if not file_path.exists() or not file_path.is_file():
            issues.append(f"pointer_file_missing: {relative_path}")
            continue

        actual_sha256 = _sha256_file(file_path)
        if actual_sha256 != recorded_sha256:
            issues.append(
                "pointer_hash_mismatch: "
                f"path={relative_path}, recorded={recorded_sha256}, actual={actual_sha256}"
            )

    return issues


def main(repo_root_str: Optional[str] = None, run_root_str: Optional[str] = None) -> int:
    """
    功能：执行 repro bundle 完整性审计。 

    Execute repro bundle integrity audit.

    Args:
        repo_root_str: Repository root path.
        run_root_str: Optional run_root path. If missing, return N.A.

    Returns:
        Exit code (0 for PASS/N.A., 1 for FAIL).
    """
    _ = repo_root_str

    if not isinstance(run_root_str, str) or not run_root_str:
        result = {
            "audit_id": "audit_repro_bundle_integrity",
            "gate_name": "gate_repro_bundle_integrity",
            "category": "S",
            "severity": "NON_BLOCK",
            "result": "N.A.",
            "rule": "run_root not provided for repro bundle integrity audit",
            "evidence": {
                "required_arg": "run_root",
                "status": "not_executed",
            },
            "impact": "cannot verify repro bundle integrity without run_root",
            "fix": "provide run_root to audit_repro_bundle_integrity.py",
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    run_root = Path(run_root_str).resolve()
    manifest_path = run_root / "artifacts" / "repro_bundle" / "manifest.json"
    pointers_path = run_root / "artifacts" / "repro_bundle" / "pointers.json"

    manifest_obj, manifest_err = _load_json_dict(manifest_path)
    pointers_obj, pointers_err = _load_json_dict(pointers_path)

    issues: List[str] = []
    if manifest_err is not None:
        issues.append(manifest_err)
    if pointers_err is not None:
        issues.append(pointers_err)

    if manifest_obj is not None:
        issues.extend(_validate_manifest(manifest_obj))
    if pointers_obj is not None:
        issues.extend(_validate_pointers(run_root, pointers_obj))

    missing_bundle_files = [
        issue for issue in issues
        if isinstance(issue, str) and issue.startswith("missing_file: ")
    ]
    missing_bundle_artifacts_only = len(missing_bundle_files) > 0 and len(missing_bundle_files) == len(issues)

    if issues:
        impact_msg = "repro bundle cannot be trusted for paper-level reproducibility"
        fix_msg = "regenerate repro bundle and ensure pointers sha256 match referenced files"
        root_cause = "repro_bundle_integrity_violation"
        if missing_bundle_artifacts_only:
            impact_msg = "bound run_root lacks repro bundle artifacts; integrity cannot be audited"
            fix_msg = "rerun repro pipeline under the same --run-root to generate artifacts/repro_bundle/manifest.json and pointers.json"
            root_cause = "required_artifact_missing_under_bound_run_root"

        result = {
            "audit_id": "audit_repro_bundle_integrity",
            "gate_name": "gate_repro_bundle_integrity",
            "category": "S",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": "repro bundle manifest/pointers integrity check failed",
            "evidence": {
                "run_root": str(run_root),
                "manifest_path": str(manifest_path),
                "pointers_path": str(pointers_path),
                "root_cause": root_cause,
                "issues": issues[:50],
            },
            "impact": impact_msg,
            "fix": fix_msg,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1

    result = {
        "audit_id": "audit_repro_bundle_integrity",
        "gate_name": "gate_repro_bundle_integrity",
        "category": "S",
        "severity": "NON_BLOCK",
        "result": "PASS",
        "rule": "repro bundle integrity ok",
        "evidence": {
            "run_root": str(run_root),
            "manifest_path": str(manifest_path),
            "pointers_path": str(pointers_path),
            "status": "repro bundle integrity ok",
        },
        "impact": "repro bundle is verifiable and auditable",
        "fix": "N.A.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    repo_root_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_root_arg = sys.argv[2] if len(sys.argv) > 2 else None
    exit_code = main(repo_root_arg, run_root_arg)
    sys.exit(exit_code)
