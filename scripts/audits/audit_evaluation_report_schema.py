"""
静态审计评测报告锚点字段完整性。
Module type: Core innovation module

审计职责：
1. 定位最新生成的 evaluation report JSON（outputs 或 artifacts）
2. 校验是否包含所有必备锚点字段
3. 验证字段非空且为合法值
4. FAIL → 论文级数据落地不完整，无法作为论文发布事实源
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 必备锚点字段（论文级一致性要求）
REQUIRED_ANCHOR_FIELDS = [
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

# 输出格式嵌套路径（支持多种报告格式）
ANCHOR_SEARCH_PATHS = [
    # 直接在 report top level
    ([], REQUIRED_ANCHOR_FIELDS),
    # 在 report.anchors 下
    (["anchors"], REQUIRED_ANCHOR_FIELDS),
]


def find_evaluation_report(repo_root: Path) -> Optional[Path]:
    """
    功能：定位最新的评测报告文件。

    Locate the most recent evaluation report JSON in outputs or artifacts.

    Args:
        repo_root: Repository root directory.

    Returns:
        Path to report file or None if not found.
    """
    # 优先搜索 outputs，再搜索 artifacts
    search_dirs = [
        repo_root / "outputs",
        repo_root / "artifacts",
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        # 查找所有 evaluation_report*.json 和 report*.json
        report_files = list(search_dir.rglob("evaluation_report*.json"))
        report_files.extend(search_dir.rglob("*report*.json"))
        
        if report_files:
            # 按修改时间排序，返回最新的
            return sorted(report_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    
    return None


def validate_anchor_fields(report_obj: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    功能：验证报告中的锚点字段完整性。

    Validate that report contains all required anchor fields.

    Args:
        report_obj: Report dict (parsed from JSON).

    Returns:
        Tuple of (is_valid: bool, missing_fields: List[str]).
    """
    if not isinstance(report_obj, dict):
        return False, ["report_object_must_be_dict"]
    
    # 尝试多个搜索路径
    for path_parts, required_fields in ANCHOR_SEARCH_PATHS:
        # 导航到目标路径
        current = report_obj
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                current = None
                break
        
        if current is None:
            continue
        
        if not isinstance(current, dict):
            continue
        
        # 检查所有必备字段
        missing = []
        for field_name in required_fields:
            value = current.get(field_name)
            # 字段必须存在且非空
            if not value or (isinstance(value, str) and value == "<absent>"):
                missing.append(field_name)
        
        if not missing:
            # 找到了完整的字段集
            return True, []
    
    # 未找到完整的字段集，报告所有缺失字段
    all_missing = []
    for field_name in REQUIRED_ANCHOR_FIELDS:
        if field_name not in report_obj and (
            "anchors" not in report_obj or field_name not in report_obj.get("anchors", {})
        ):
            all_missing.append(field_name)
    
    return False, all_missing if all_missing else REQUIRED_ANCHOR_FIELDS


def main(repo_root_str: Optional[str] = None) -> int:
    """
    功能：执行锚点字段审计并输出 JSON 结果。

    Execute anchor field audit and output structured result.

    Args:
        repo_root_str: Optional repository root path (from command line).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    repo_root = Path(repo_root_str) if repo_root_str else Path.cwd()
    
    # 定位报告文件
    report_path = find_evaluation_report(repo_root)
    
    if report_path is None:
        # 未找到报告（可能尚未生成，非 FAIL）
        result = {
            "audit_id": "audit_evaluation_report_schema",
            "gate_name": "gate_evaluation_report_schema",
            "category": "S",
            "severity": "NON_BLOCK",
            "result": "N.A.",
            "rule": "evaluation report not found in outputs or artifacts",
            "evidence": {
                "search_locations": ["outputs/", "artifacts/"],
                "status": "report_not_yet_generated",
            },
            "impact": "evaluation report is not available for audit",
            "fix": "generate evaluation report via evaluate workflow",
        }
        print(json.dumps(result, indent=2))
        return 0
    
    # 读取报告
    try:
        report_content = report_path.read_text(encoding="utf-8")
        report_obj = json.loads(report_content)
    except (IOError, json.JSONDecodeError) as e:
        result = {
            "audit_id": "audit_evaluation_report_schema",
            "gate_name": "gate_evaluation_report_schema",
            "category": "S",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": f"evaluation report could not be parsed: {type(e).__name__}",
            "evidence": {
                "report_path": str(report_path),
                "error": str(e)[:200],
            },
            "impact": "evaluation report is corrupted or malformed",
            "fix": "regenerate evaluation report, ensure JSON validity",
        }
        print(json.dumps(result, indent=2))
        return 1
    
    # 验证锚点字段
    is_valid, missing_fields = validate_anchor_fields(report_obj)
    
    if is_valid:
        result = {
            "audit_id": "audit_evaluation_report_schema",
            "gate_name": "gate_evaluation_report_schema",
            "category": "S",
            "severity": "NON_BLOCK",
            "result": "PASS",
            "rule": "evaluation report contains all required anchor fields",
            "evidence": {
                "report_path": str(report_path),
                "anchor_fields_found": REQUIRED_ANCHOR_FIELDS,
                "status": "evaluation report anchor fields complete",
            },
            "impact": "evaluation output meets paper-level auditability requirements",
            "fix": "N.A.",
        }
    else:
        result = {
            "audit_id": "audit_evaluation_report_schema",
            "gate_name": "gate_evaluation_report_schema",
            "category": "S",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": "evaluation report missing required anchor fields",
            "evidence": {
                "report_path": str(report_path),
                "required_fields": REQUIRED_ANCHOR_FIELDS,
                "missing_fields": missing_fields,
                "status": "anchor_validation_failed",
            },
            "impact": "evaluation output incomplete for paper reproducibility",
            "fix": "ensure evaluate workflow captures and propagates all anchor fields (cfg_digest, plan_digest, thresholds_digest, threshold_metadata_digest, impl_digest, fusion_rule_version, attack_protocol_version, attack_protocol_digest, policy_path)",
        }
    
    print(json.dumps(result, indent=2))
    return 0 if is_valid else 1


if __name__ == "__main__":
    repo_root = sys.argv[1] if len(sys.argv) > 1 else None
    exit_code = main(repo_root)
    sys.exit(exit_code)
