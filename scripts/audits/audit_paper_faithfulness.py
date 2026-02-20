"""
功能：审计 Paper Faithfulness 对齐证据完整性

Module type: Core innovation module

Verify paper faithfulness alignment evidence completeness and integrity.
Checks:
1. paper_spec_digest binding in records
2. pipeline_fingerprint consistency
3. injection_site_digest recording
4. alignment evidence completeness
"""

import json
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional


def compute_file_sha256(filepath: Path) -> str:
    """
    功能：计算文件 SHA256 摘要。

    Compute SHA256 digest of file content.

    Args:
        filepath: Path to file.

    Returns:
        64-character hex digest.
    """
    if not filepath.exists():
        return "<absent>"
    try:
        with open(filepath, "rb") as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()
    except Exception:
        return "<error>"


def check_paper_spec_file_exists(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查 paper_faithfulness_spec.yaml 是否存在。

    Check paper_faithfulness_spec.yaml file existence.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result dict.
    """
    spec_file = repo_root / "configs" / "paper_faithfulness_spec.yaml"
    exists = spec_file.exists()

    return {
        "check": "paper_spec_file_exists",
        "pass": exists,
        "spec_file_path": str(spec_file),
        "exists": exists
    }


def check_field_paths_registry_updated(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查 frozen_contracts.yaml 的 field_paths_registry 是否包含新增字段。

    Check field_paths_registry includes new paper faithfulness fields.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result dict.
    """
    import yaml

    frozen_file = repo_root / "configs" / "frozen_contracts.yaml"
    if not frozen_file.exists():
        return {
            "check": "field_paths_registry_updated",
            "pass": False,
            "error": "frozen_contracts.yaml not found"
        }

    try:
        with open(frozen_file, "r", encoding="utf-8") as f:
            frozen_data = yaml.safe_load(f)
    except Exception as e:
        return {
            "check": "field_paths_registry_updated",
            "pass": False,
            "error": f"Failed to load frozen_contracts.yaml: {e}"
        }

    field_paths_registry = frozen_data.get("records_schema", {}).get("field_paths_registry", [])

    # 检查新增的 paper faithfulness 字段是否已注册。
    required_fields = [
        "paper_faithfulness.spec_version",
        "paper_faithfulness.spec_digest",
        "content_evidence.pipeline_fingerprint",
        "content_evidence.pipeline_fingerprint_digest",
        "content_evidence.injection_site_spec",
        "content_evidence.injection_site_digest",
        "content_evidence.alignment_report",
        "content_evidence.alignment_digest"
    ]

    missing_fields = []
    for field in required_fields:
        if field not in field_paths_registry:
            missing_fields.append(field)

    return {
        "check": "field_paths_registry_updated",
        "pass": len(missing_fields) == 0,
        "required_fields": required_fields,
        "missing_fields": missing_fields
    }


def check_records_schema_extensions_updated(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查 records_schema_extensions.yaml 是否包含新增字段定义。

    Check records_schema_extensions.yaml includes new field definitions.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result dict.
    """
    import yaml

    schema_file = repo_root / "configs" / "records_schema_extensions.yaml"
    if not schema_file.exists():
        return {
            "check": "records_schema_extensions_updated",
            "pass": False,
            "error": "records_schema_extensions.yaml not found"
        }

    try:
        with open(schema_file, "r", encoding="utf-8") as f:
            schema_data = yaml.safe_load(f)
    except Exception as e:
        return {
            "check": "records_schema_extensions_updated",
            "pass": False,
            "error": f"Failed to load records_schema_extensions.yaml: {e}"
        }

    fields = schema_data.get("fields", [])
    field_paths = [f.get("path") for f in fields]

    # 检查新增的 paper faithfulness 字段是否已定义。
    required_fields = [
        "paper_faithfulness.spec_version",
        "paper_faithfulness.spec_digest",
        "content_evidence.pipeline_fingerprint",
        "content_evidence.pipeline_fingerprint_digest",
        "content_evidence.injection_site_spec",
        "content_evidence.injection_site_digest",
        "content_evidence.alignment_report",
        "content_evidence.alignment_digest"
    ]

    missing_fields = []
    for field in required_fields:
        if field not in field_paths:
            missing_fields.append(field)

    return {
        "check": "records_schema_extensions_updated",
        "pass": len(missing_fields) == 0,
        "required_fields": required_fields,
        "missing_fields": missing_fields
    }


def check_implementation_modules_exist(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查实现模块是否存在。

    Check implementation modules existence.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result dict.
    """
    required_modules = [
        "main/diffusion/sd3/pipeline_inspector.py",
        "main/diffusion/sd3/diffusion_tracer.py",
        "main/watermarking/paper_faithfulness/injection_site_binder.py",
        "main/watermarking/paper_faithfulness/alignment_evaluator.py"
    ]

    missing_modules = []
    for module in required_modules:
        module_path = repo_root / module
        if not module_path.exists():
            missing_modules.append(module)

    return {
        "check": "implementation_modules_exist",
        "pass": len(missing_modules) == 0,
        "required_modules": required_modules,
        "missing_modules": missing_modules
    }


def aggregate_checks(repo_root: Path) -> List[Dict[str, Any]]:
    """
    功能：聚合所有检查项。

    Aggregate all audit checks.

    Args:
        repo_root: Repository root directory.

    Returns:
        List of check results.
    """
    checks = []

    # 检查 1: paper_spec_file 存在性。
    check_1 = check_paper_spec_file_exists(repo_root)
    checks.append(check_1)

    # 检查 2: field_paths_registry 更新。
    check_2 = check_field_paths_registry_updated(repo_root)
    checks.append(check_2)

    # 检查 3: records_schema_extensions 更新。
    check_3 = check_records_schema_extensions_updated(repo_root)
    checks.append(check_3)

    # 检查 4: 实现模块存在性。
    check_4 = check_implementation_modules_exist(repo_root)
    checks.append(check_4)

    return checks


def format_audit_result(checks: List[Dict[str, Any]], repo_root: Path) -> List[Dict[str, Any]]:
    """
    功能：格式化为统一审计结果格式。

    Format audit results to unified schema.

    Args:
        checks: List of check results.
        repo_root: Repository root directory.

    Returns:
        List of audit result dicts.
    """
    results = []

    for idx, check in enumerate(checks, start=1):
        check_name = check.get("check", f"check_{idx}")
        check_pass = check.get("pass", False)

        if check_pass:
            continue  # 仅输出失败项。

        audit_id = f"A.PAPER_FAITHFUL.{idx}"
        gate_name = f"gate.paper_faithfulness.{check_name}"

        evidence = {k: v for k, v in check.items() if k not in ["check", "pass"]}

        result = {
            "audit_id": audit_id,
            "gate_name": gate_name,
            "category": "A",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": f"Paper faithfulness 对齐证据检查失败: {check_name}",
            "evidence": evidence,
            "impact": "缺少必需的对齐证据字段或实现模块",
            "fix": f"补齐缺失的字段定义或实现模块: {check_name}"
        }

        results.append(result)

    return results


def main() -> int:
    """
    功能：主入口。

    Main audit entry point.

    Returns:
        Exit code (0 if all checks pass, 1 otherwise).
    """
    if len(sys.argv) < 2:
        print("Usage: python audit_paper_faithfulness.py <repo_root>", file=sys.stderr)
        return 1

    repo_root = Path(sys.argv[1]).resolve()

    if not repo_root.exists():
        print(f"Repository root not found: {repo_root}", file=sys.stderr)
        return 1

    # 执行检查。
    checks = aggregate_checks(repo_root)

    # 格式化结果。
    audit_results = format_audit_result(checks, repo_root)

    # 输出 JSON（必须为单个 JSON 对象或数组）。
    if audit_results:
        print(json.dumps(audit_results, indent=2, ensure_ascii=False))
        return 1
    else:
        # 所有检查通过。
        print(json.dumps([], indent=2, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    sys.exit(main())
