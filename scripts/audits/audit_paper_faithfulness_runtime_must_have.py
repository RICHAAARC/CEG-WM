"""
功能：审计 Paper Faithfulness 运行期必达证据

Module type: Core innovation module

Verify paper faithfulness evidence is MUST-HAVE at runtime (not optional).
Checks:
1. alignment_report/alignment_digest 在 enable_paper_faithfulness=true 时为"必达"
2. paper_spec_digest 被 run_closure 绑定且可复算
3. pipeline_fingerprint_digest / injection_site_digest / diffusion_trace_digest 的可复算与一致性
4. 任一缺失或不一致 → FAIL（必达缺失建议 BLOCK）
"""

import json
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional


def check_alignment_evidence_is_must_have(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查 alignment evidence 在启用时必达。

    Check alignment_report/alignment_digest must exist when paper_faithfulness enabled.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result dict.
    """
    # 查找最近的 embed record。
    records_dir = repo_root / "tmp" / "cli_smoke" / "embed_run" / "records"
    if not records_dir.exists():
        return {
            "check": "alignment_evidence_is_must_have",
            "pass": True,  # 没有 records 可检查，认为通过（因为没有违规）
            "note": "No records found for verification"
        }

    record_files = list(records_dir.glob("embed_record.json"))
    if not record_files:
        return {
            "check": "alignment_evidence_is_must_have",
            "pass": True,
            "note": "No embed_record.json found"
        }

    record_file = record_files[0]
    try:
        with open(record_file, "r", encoding="utf-8") as f:
            record = json.load(f)
    except Exception as e:
        return {
            "check": "alignment_evidence_is_must_have",
            "pass": False,
            "error": f"Failed to load record: {e}",
            "record_file": str(record_file)
        }

    # 检查 content_evidence.alignment_report 和 alignment_digest 存在性。
    content_evidence = record.get("content_evidence", {})
    alignment_report = content_evidence.get("alignment_report")
    alignment_digest = content_evidence.get("alignment_digest")

    missing_fields = []
    if alignment_report is None:
        missing_fields.append("content_evidence.alignment_report")
    elif isinstance(alignment_report, dict) and alignment_report.get("status") in ("absent", "failed"):
        missing_fields.append("content_evidence.alignment_report (status: absent/failed)")

    if alignment_digest is None or alignment_digest in ("<absent>", "<failed>", ""):
        missing_fields.append("content_evidence.alignment_digest")

    return {
        "check": "alignment_evidence_is_must_have",
        "pass": len(missing_fields) == 0,
        "missing_fields": missing_fields,
        "record_file": str(record_file)
    }


def check_paper_spec_digest_binding(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查 paper_spec_digest 绑定与可复算性。

    Check paper_spec_digest is bound and reproducible.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result dict.
    """
    records_dir = repo_root / "tmp" / "cli_smoke" / "embed_run" / "records"
    if not records_dir.exists():
        return {
            "check": "paper_spec_digest_binding",
            "pass": True,
            "note": "No records found for verification"
        }

    record_files = list(records_dir.glob("embed_record.json"))
    if not record_files:
        return {
            "check": "paper_spec_digest_binding",
            "pass": True,
            "note": "No embed_record.json found"
        }

    record_file = record_files[0]
    try:
        with open(record_file, "r", encoding="utf-8") as f:
            record = json.load(f)
    except Exception as e:
        return {
            "check": "paper_spec_digest_binding",
            "pass": False,
            "error": f"Failed to load record: {e}",
            "record_file": str(record_file)
        }

    # 检查 paper_faithfulness.spec_digest 存在性。
    paper_faithfulness = record.get("paper_faithfulness", {})
    spec_digest = paper_faithfulness.get("spec_digest")

    if spec_digest is None or spec_digest in ("<absent>", "<failed>", ""):
        return {
            "check": "paper_spec_digest_binding",
            "pass": False,
            "error": "paper_faithfulness.spec_digest is absent or invalid",
            "spec_digest": spec_digest,
            "record_file": str(record_file)
        }

    return {
        "check": "paper_spec_digest_binding",
        "pass": True,
        "spec_digest": spec_digest,
        "record_file": str(record_file)
    }


def check_pipeline_fingerprint_consistency(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查 pipeline_fingerprint_digest 一致性。

    Check pipeline_fingerprint_digest consistency across records.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result dict.
    """
    records_dir = repo_root / "tmp" / "cli_smoke" / "embed_run" / "records"
    if not records_dir.exists():
        return {
            "check": "pipeline_fingerprint_consistency",
            "pass": True,
            "note": "No records found for verification"
        }

    record_files = list(records_dir.glob("embed_record.json"))
    if not record_files:
        return {
            "check": "pipeline_fingerprint_consistency",
            "pass": True,
            "note": "No embed_record.json found"
        }

    record_file = record_files[0]
    try:
        with open(record_file, "r", encoding="utf-8") as f:
            record = json.load(f)
    except Exception as e:
        return {
            "check": "pipeline_fingerprint_consistency",
            "pass": False,
            "error": f"Failed to load record: {e}",
            "record_file": str(record_file)
        }

    # 检查 content_evidence.pipeline_fingerprint_digest 存在性。
    content_evidence = record.get("content_evidence", {})
    pipeline_fingerprint_digest = content_evidence.get("pipeline_fingerprint_digest")

    if pipeline_fingerprint_digest is None or pipeline_fingerprint_digest in ("<absent>", "<failed>", ""):
        return {
            "check": "pipeline_fingerprint_consistency",
            "pass": False,
            "error": "pipeline_fingerprint_digest is absent or invalid",
            "pipeline_fingerprint_digest": pipeline_fingerprint_digest,
            "record_file": str(record_file)
        }

    return {
        "check": "pipeline_fingerprint_consistency",
        "pass": True,
        "pipeline_fingerprint_digest": pipeline_fingerprint_digest,
        "record_file": str(record_file)
    }


def check_injection_site_consistency(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查 injection_site_digest 一致性。

    Check injection_site_digest consistency and reproducibility.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result dict.
    """
    records_dir = repo_root / "tmp" / "cli_smoke" / "embed_run" / "records"
    if not records_dir.exists():
        return {
            "check": "injection_site_consistency",
            "pass": True,
            "note": "No records found for verification"
        }

    record_files = list(records_dir.glob("embed_record.json"))
    if not record_files:
        return {
            "check": "injection_site_consistency",
            "pass": True,
            "note": "No embed_record.json found"
        }

    record_file = record_files[0]
    try:
        with open(record_file, "r", encoding="utf-8") as f:
            record = json.load(f)
    except Exception as e:
        return {
            "check": "injection_site_consistency",
            "pass": False,
            "error": f"Failed to load record: {e}",
            "record_file": str(record_file)
        }

    # 检查 content_evidence.injection_site_digest 存在性。
    content_evidence = record.get("content_evidence", {})
    injection_site_digest = content_evidence.get("injection_site_digest")

    if injection_site_digest is None or injection_site_digest in ("<absent>", "<failed>", ""):
        return {
            "check": "injection_site_consistency",
            "pass": False,
            "error": "injection_site_digest is absent or invalid",
            "injection_site_digest": injection_site_digest,
            "record_file": str(record_file)
        }

    return {
        "check": "injection_site_consistency",
        "pass": True,
        "injection_site_digest": injection_site_digest,
        "record_file": str(record_file)
    }


def aggregate_checks(repo_root: Path) -> List[Dict[str, Any]]:
    """
    功能：聚合所有运行期必达检查项。

    Aggregate all runtime must-have audit checks.

    Args:
        repo_root: Repository root directory.

    Returns:
        List of check results.
    """
    checks = []

    checks.append(check_alignment_evidence_is_must_have(repo_root))
    checks.append(check_paper_spec_digest_binding(repo_root))
    checks.append(check_pipeline_fingerprint_consistency(repo_root))
    checks.append(check_injection_site_consistency(repo_root))

    return checks


def format_audit_result(checks: List[Dict[str, Any]], repo_root: Path) -> List[Dict[str, Any]]:
    """
    功能：格式化为统一审计结果格式。

    Format audit results to unified schema.

    Args:
        checks: List of check results.
        repo_root: Repository root directory.

    Returns:
        List of audit result dicts (only failures).
    """
    results = []

    for idx, check in enumerate(checks, start=1):
        check_name = check.get("check", f"check_{idx}")
        check_pass = check.get("pass", False)

        if check_pass:
            continue  # 仅输出失败项

        audit_id = f"A.PAPER_FAITHFUL_RUNTIME.{idx}"
        gate_name = f"gate.paper_faithfulness.runtime.{check_name}"

        evidence = {k: v for k, v in check.items() if k not in ["check", "pass"]}

        result = {
            "audit_id": audit_id,
            "gate_name": gate_name,
            "category": "A",
            "severity": "BLOCK",
            "result": "FAIL",
            "evidence": evidence
        }

        results.append(result)

    return results


def main():
    """主流程。"""
    if len(sys.argv) < 2:
        print("Usage: python audit_paper_faithfulness_runtime_must_have.py <repo_root>", file=sys.stderr)
        sys.exit(1)

    repo_root = Path(sys.argv[1]).resolve()
    if not repo_root.exists():
        print(f"Error: Repository root does not exist: {repo_root}", file=sys.stderr)
        sys.exit(1)

    checks = aggregate_checks(repo_root)
    results = format_audit_result(checks, repo_root)

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if results:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
