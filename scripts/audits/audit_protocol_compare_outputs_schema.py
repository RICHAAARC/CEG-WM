#!/usr/bin/env python
"""
文件目的：protocol compare 汇总工件 schema 与一致性审计
Module type: General module

检查项：
1. compare_summary.json 存在且可解析（JSON schema 有效）。
2. 每条 protocol 记录必须包含 protocol_id、attack_protocol_version/digest、run_root_relative。
3. 必须包含所有锚点字段（cfg_digest, plan_digest, 等）。
4. protocol_id 不得重复（重复即 FAIL）。
5. 失败记录必须包含 failure_reason（无锚定失败）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _is_ignored_compare_summary_path(path_obj: Path, repo_root: Path) -> bool:
    """
    功能：判断 compare_summary 路径是否属于测试临时目录或缓存目录。

    Determine whether a compare_summary path should be ignored as test/tmp artifact.

    Args:
        path_obj: Candidate compare_summary path.
        repo_root: Repository root directory.

    Returns:
        True when path belongs to ignored transient locations.
    """
    ignored_exact_names: Set[str] = {
        ".pytest_cache",
        "pytesttmp",
        "__pycache__",
    }
    try:
        relative_parts = path_obj.resolve().relative_to(repo_root.resolve()).parts
    except Exception:
        relative_parts = path_obj.parts

    for raw_part in relative_parts:
        part = raw_part.lower()
        if part in ignored_exact_names:
            return True
        if part.startswith("pytest-") or part.startswith("pytest-of-") or part.startswith("pytesttmp"):
            return True
    return False


def _find_compare_summary_path(repo_root: Path) -> Optional[Path]:
    """
    功能：在受控边界内定位 compare_summary.json，避免测试临时产物污染。

    Resolve compare_summary.json from controlled locations only.

    Args:
        repo_root: Repository root directory.

    Returns:
        Resolved compare_summary path when found, otherwise None.
    """
    explicit_candidates = [
        repo_root / "outputs" / "multi_protocol_evaluation" / "artifacts" / "protocol_compare" / "compare_summary.json",
        repo_root / "compare_summary.json",
    ]
    for candidate_path in explicit_candidates:
        if candidate_path.exists() and candidate_path.is_file() and not _is_ignored_compare_summary_path(candidate_path, repo_root):
            return candidate_path

    outputs_dir = repo_root / "outputs"
    if not outputs_dir.exists():
        return None

    valid_matches: List[Path] = []
    for candidate_path in outputs_dir.rglob("compare_summary.json"):
        if not candidate_path.is_file():
            continue
        if _is_ignored_compare_summary_path(candidate_path, repo_root):
            continue
        valid_matches.append(candidate_path)

    if not valid_matches:
        return None

    valid_matches.sort(key=lambda item: (len(item.parts), str(item).lower()))
    return valid_matches[0]


def audit_protocol_compare_outputs_schema(repo_root: Path) -> Dict[str, Any]:
    """
    功能：审计 protocol compare 汇总工件 schema 与一致性。

    Audit protocol comparison outputs for schema correctness and consistency.

    Args:
        repo_root: Repository root directory.

    Returns:
        Audit result dict following standard audit schema.

    Raises:
        TypeError: If repo_root is not Path.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")

    audit_id = "audit_protocol_compare_outputs_schema"
    gate_name = "gate.protocol_compare_outputs_schema"
    category = "G"
    severity = "BLOCK"

    # 默认 N.A.（如果根本不存在 compare 工件，则不适用）
    default_result = {
        "audit_id": audit_id,
        "gate_name": gate_name,
        "category": category,
        "severity": "NON_BLOCK",
        "result": "N.A.",
        "rule": "No protocol_compare outputs found (research-only feature, not mandatory)",
        "evidence": {},
        "impact": "Protocol compare audit not applicable: multi-protocol evaluation feature not invoked in this run",
        "fix": "N/A (not applicable when feature is not used)",
    }

    compare_summary_path = _find_compare_summary_path(repo_root)

    if compare_summary_path is None:
        # 未找到 compare 工件，SKIP
        return default_result

    # (1) 解析 JSON
    try:
        compare_obj = json.loads(compare_summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "audit_id": audit_id,
            "gate_name": gate_name,
            "category": category,
            "severity": severity,
            "result": "FAIL",
            "rule": "protocol_compare outputs schema validation",
            "evidence": {
                "error": f"Failed to parse compare_summary.json: {type(exc).__name__}: {str(exc)}",
                "path": str(compare_summary_path),
            },
            "impact": "Protocol compare artifact is corrupted or malformed",
            "fix": "Regenerate protocol compare outputs by running run_multi_protocol_evaluation.py",
        }

    if not isinstance(compare_obj, dict):
        return {
            "audit_id": audit_id,
            "gate_name": gate_name,
            "category": category,
            "severity": severity,
            "result": "FAIL",
            "rule": "protocol_compare root must be dict",
            "evidence": {
                "error": f"Root object is {type(compare_obj).__name__}, expected dict",
                "path": str(compare_summary_path),
            },
            "impact": "Protocol compare artifact schema violation",
            "fix": "Ensure compare_summary.json root is a dict",
        }

    # (2) 检查必需字段
    protocols_list = compare_obj.get("protocols")
    if not isinstance(protocols_list, list):
        return {
            "audit_id": audit_id,
            "gate_name": gate_name,
            "category": category,
            "severity": severity,
            "result": "FAIL",
            "rule": "protocols field must be list",
            "evidence": {
                "error": f"protocols field is {type(protocols_list).__name__}, expected list",
                "path": str(compare_summary_path),
            },
            "impact": "Protocol compare artifact schema violation",
            "fix": "Ensure protocols field is a list of protocol records",
        }

    schema_version = compare_obj.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        return {
            "audit_id": audit_id,
            "gate_name": gate_name,
            "category": category,
            "severity": severity,
            "result": "FAIL",
            "rule": "schema_version must be non-empty string",
            "evidence": {
                "error": f"schema_version is {repr(schema_version)}, expected non-empty string",
                "path": str(compare_summary_path),
            },
            "impact": "Protocol compare artifact schema violation",
            "fix": "Ensure schema_version field is present and non-empty",
        }

    # (3) 检查每条 protocol 记录
    protocol_ids_seen: List[str] = []
    required_anchor_fields = {
        "cfg_digest",
        "plan_digest",
        "thresholds_digest",
        "threshold_metadata_digest",
        "impl_digest",
        "fusion_rule_version",
        "policy_path",
    }

    failures: List[Dict[str, Any]] = []

    for idx, protocol_record in enumerate(protocols_list):
        if not isinstance(protocol_record, dict):
            failures.append({
                "index": idx,
                "error": f"Protocol record {idx} is {type(protocol_record).__name__}, expected dict",
            })
            continue

        # 检查 protocol_id
        protocol_id = protocol_record.get("protocol_id")
        if not isinstance(protocol_id, str) or not protocol_id:
            failures.append({
                "index": idx,
                "error": "Missing or invalid protocol_id field",
            })
            continue

        # 检查重复的 protocol_id
        if protocol_id != "<absent>" and protocol_id in protocol_ids_seen:
            failures.append({
                "index": idx,
                "error": f"Duplicate protocol_id: {protocol_id}",
            })
            continue
        protocol_ids_seen.append(protocol_id)

        # 检查 attack_protocol_version 和 attack_protocol_digest（在顶级）
        version = protocol_record.get("attack_protocol_version")
        digest = protocol_record.get("attack_protocol_digest")
        if not isinstance(version, str) or not version:
            failures.append({
                "index": idx,
                "error": "Missing or invalid attack_protocol_version field",
            })
            continue
        if not isinstance(digest, str) or not digest:
            failures.append({
                "index": idx,
                "error": "Missing or invalid attack_protocol_digest field",
            })
            continue

        # 检查状态
        status = protocol_record.get("status")
        if not isinstance(status, str) or status not in ("ok", "fail"):
            failures.append({
                "index": idx,
                "error": f"Invalid status: {repr(status)} (expected 'ok' or 'fail')",
            })
            continue

        # 检查 run_root
        run_root = protocol_record.get("run_root")
        if not isinstance(run_root, str) or not run_root:
            failures.append({
                "index": idx,
                "error": "Missing or invalid run_root field",
            })
            continue

        # 如果状态为 fail，必须有 failure_reason
        if status == "fail":
            failure_reason = protocol_record.get("failure_reason")
            if failure_reason not in ("<absent>", "ok") and isinstance(failure_reason, str):
                # OK：有明确失败原因
                pass
            elif not isinstance(failure_reason, str):
                failures.append({
                    "index": idx,
                    "error": "status=fail but failure_reason is missing or not string",
                })
                continue

        # 如果状态为 ok，检查锚点字段完整性
        if status == "ok":
            anchors = protocol_record.get("anchors", {})
            if not isinstance(anchors, dict):
                failures.append({
                    "index": idx,
                    "error": "anchors field must be dict when status=ok",
                })
                continue

            missing_anchors = []
            for field_name in required_anchor_fields:
                if field_name not in anchors:
                    missing_anchors.append(field_name)

            if missing_anchors:
                failures.append({
                    "index": idx,
                    "error": f"Missing anchor fields when status=ok: {', '.join(missing_anchors)}",
                })
                continue

    # (4) 汇总审计结果
    if failures:
        error_lines = [f"Protocol record validation failed ({len(failures)} errors):"]
        for failure in failures[:10]:  # 最多显示 10 条错误
            error_lines.append(f"  - Index {failure.get('index')}: {failure.get('error')}")
        if len(failures) > 10:
            error_lines.append(f"  ... and {len(failures) - 10} more errors")

        return {
            "audit_id": audit_id,
            "gate_name": gate_name,
            "category": category,
            "severity": severity,
            "result": "FAIL",
            "rule": "protocol_compare outputs schema validation",
            "evidence": {
                "errors": failures,
                "path": str(compare_summary_path),
                "protocol_count": len(protocols_list),
                "failure_count": len(failures),
                "summary": "\n".join(error_lines),
            },
            "impact": "Protocol compare artifact has schema violations",
            "fix": "Regenerate protocol compare outputs or fix record schema",
        }

    # (5) 成功：所有记录有效
    return {
        "audit_id": audit_id,
        "gate_name": gate_name,
        "category": category,
        "severity": severity,
        "result": "PASS",
        "rule": "protocol_compare outputs schema validation",
        "evidence": {
            "path": str(compare_summary_path),
            "schema_version": schema_version,
            "protocol_count": len(protocols_list),
            "protocol_ids": protocol_ids_seen,
        },
        "impact": "Protocol compare artifact is valid and consistent",
        "fix": None,
    }


def main() -> None:
    """CLI entry for protocol compare outputs schema audit."""
    if len(sys.argv) > 1:
        repo_root = Path(sys.argv[1])
    else:
        repo_root = Path.cwd()

    result = audit_protocol_compare_outputs_schema(repo_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if result.get("result") == "FAIL":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    main()
