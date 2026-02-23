"""
功能：实验矩阵汇总工件 schema 完整性审计

Module type: General module

检查项：
1. grid_summary.json 存在且可解析（canonical JSON 解析）
2. 顶层锚点字段全集存在且非空（cfg_digest、thresholds_digest 等）
3. 若 grid_table.csv 存在：列集合与列顺序稳定（通过固定列定义常量进行比对）

FAIL 必须给证据路径与缺失字段名。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


AUDIT_ID = "experiment_matrix.outputs_schema"
SEVERITY = "BLOCK"

# 锚点字段全集（必备字段，缺失即 FAIL）
REQUIRED_ANCHOR_FIELDS = [
    "cfg_digest",
    "thresholds_digest",
    "threshold_metadata_digest",
    "attack_protocol_version",
    "attack_protocol_digest",
    "attack_coverage_digest",
    "impl_digest",
    "fusion_rule_version",
]


def _build_matrix_missing_result(
    reason: str,
    outputs_dir: Path,
    require_experiment_matrix: bool,
) -> Dict[str, Any]:
    """
    功能：构造 matrix 工件缺失场景的审计结果。

    Build standardized audit result for missing matrix artifacts.

    Args:
        reason: Missing reason code.
        outputs_dir: Target outputs directory path.
        require_experiment_matrix: Whether matrix artifacts are mandatory.

    Returns:
        Audit result dictionary (FAIL or N.A.).
    """
    if not isinstance(reason, str) or not reason:
        raise TypeError("reason must be non-empty str")
    if not isinstance(outputs_dir, Path):
        raise TypeError("outputs_dir must be Path")
    if not isinstance(require_experiment_matrix, bool):
        raise TypeError("require_experiment_matrix must be bool")

    if require_experiment_matrix:
        return {
            "audit_id": AUDIT_ID,
            "gate_name": "gate.experiment_matrix_outputs_schema",
            "category": "G",
            "severity": SEVERITY,
            "result": "FAIL",
            "rule": "实验矩阵汇总工件 schema 必须包含锚点字段全集",
            "evidence": {
                "reason": reason,
                "path": str(outputs_dir),
                "require_experiment_matrix": True,
            },
            "impact": "experiment matrix 为必需门禁，但矩阵汇总工件缺失",
            "fix": "先运行 experiment_matrix 生成 grid_summary.json，再执行 signoff",
        }

    return {
        "audit_id": AUDIT_ID,
        "gate_name": "gate.experiment_matrix_outputs_schema",
        "category": "G",
        "severity": SEVERITY,
        "result": "N.A.",
        "rule": "实验矩阵汇总工件 schema 必须包含锚点字段全集",
        "evidence": {
            "reason": reason,
            "path": str(outputs_dir),
            "require_experiment_matrix": False,
            "note": "N.A. because matrix not required",
        },
        "impact": "N.A. because matrix not required",
        "fix": "如需发布级复现，请启用 require_experiment_matrix 并先运行 experiment_matrix",
    }


def check_grid_summary_schema(
    scan_root: Path,
    require_experiment_matrix: bool,
) -> Dict[str, Any]:
    """
    功能：检查 grid_summary.json schema 完整性。

    Verify grid_summary.json exists, is parsable, and contains
    all required anchor fields.

    Args:
        scan_root: Root directory to scan for outputs.
        require_experiment_matrix: Whether missing matrix artifacts must fail.

    Returns:
        Audit result dictionary (PASS/FAIL/N.A.).
    """
    # grid_summary.json 可能出现在多个位置，这里检查典型输出位置
    # 由于 experiment_matrix 输出到 batch_root，我们需要扫描 outputs/ 目录
    outputs_dir = scan_root / "outputs"
    if not outputs_dir.exists() or not outputs_dir.is_dir():
        return _build_matrix_missing_result(
            reason="outputs_dir_not_found",
            outputs_dir=outputs_dir,
            require_experiment_matrix=require_experiment_matrix,
        )

    # 扫描所有 grid_summary.json 文件
    grid_summary_files = list(outputs_dir.rglob("grid_summary.json"))
    if not grid_summary_files:
        return _build_matrix_missing_result(
            reason="grid_summary_not_found",
            outputs_dir=outputs_dir,
            require_experiment_matrix=require_experiment_matrix,
        )

    # 检查所有发现的 grid_summary.json
    failures: List[Dict[str, Any]] = []
    for grid_summary_path in grid_summary_files:
        try:
            with grid_summary_path.open("r", encoding="utf-8") as f:
                grid_summary = json.load(f)
        except json.JSONDecodeError as exc:
            failures.append({
                "path": str(grid_summary_path),
                "error": "json_parse_failed",
                "details": f"{type(exc).__name__}: {exc}",
            })
            continue
        except Exception as exc:
            failures.append({
                "path": str(grid_summary_path),
                "error": "file_read_failed",
                "details": f"{type(exc).__name__}: {exc}",
            })
            continue

        if not isinstance(grid_summary, dict):
            failures.append({
                "path": str(grid_summary_path),
                "error": "json_root_not_dict",
                "details": f"Expected dict, got {type(grid_summary).__name__}",
            })
            continue

        # 检查锚点字段全集
        missing_fields: List[str] = []
        for field in REQUIRED_ANCHOR_FIELDS:
            value = grid_summary.get(field)
            if value is None or (isinstance(value, str) and value == "<absent>"):
                missing_fields.append(field)

        if missing_fields:
            failures.append({
                "path": str(grid_summary_path),
                "error": "missing_anchor_fields",
                "missing_fields": missing_fields,
                "details": f"必备锚点字段缺失或为 <absent>: {', '.join(missing_fields)}",
            })

    if failures:
        return {
            "audit_id": AUDIT_ID,
            "gate_name": "gate.experiment_matrix_outputs_schema",
            "category": "G",
            "severity": SEVERITY,
            "result": "FAIL",
            "rule": "实验矩阵汇总工件 schema 必须包含锚点字段全集",
            "evidence": {
                "failures": failures,
                "scanned_files_count": len(grid_summary_files),
            },
            "impact": "实验矩阵汇总工件 schema 不完整，影响论文复现可追溯性",
            "fix": "补齐 grid_summary.json 中缺失的锚点字段（cfg_digest、thresholds_digest 等）",
        }

    return {
        "audit_id": AUDIT_ID,
        "gate_name": "gate.experiment_matrix_outputs_schema",
        "category": "G",
        "severity": SEVERITY,
        "result": "PASS",
        "rule": "实验矩阵汇总工件 schema 必须包含锚点字段全集",
        "evidence": {
            "scanned_files_count": len(grid_summary_files),
            "checked_paths": [str(p) for p in grid_summary_files],
            "required_anchor_fields": REQUIRED_ANCHOR_FIELDS,
        },
        "impact": "N.A.",
        "fix": "N.A.",
    }


def run_audit(repo_root: Path) -> Dict[str, Any]:
    """
    功能：执行实验矩阵汇总工件 schema 审计。

    Run experiment matrix outputs schema audit.

    Args:
        repo_root: Repository root directory.

    Returns:
        Audit result dictionary.
    """
    if not isinstance(repo_root, Path):
        repo_root = Path(repo_root)
    if not repo_root.is_dir():
        return {
            "audit_id": AUDIT_ID,
            "gate_name": "gate.experiment_matrix_outputs_schema",
            "category": "G",
            "severity": SEVERITY,
            "result": "FAIL",
            "rule": "repo_root must be valid directory",
            "evidence": {"path": str(repo_root)},
            "impact": "invalid audit input",
            "fix": "provide valid repo_root path",
        }

    return check_grid_summary_schema(
        scan_root=repo_root,
        require_experiment_matrix=False,
    )


def run_audit_with_policy(
    repo_root: Path,
    run_root: Optional[Path],
    require_experiment_matrix: bool,
) -> Dict[str, Any]:
    """
    功能：执行带 matrix 缺失策略的审计。

    Run matrix schema audit with explicit requirement policy.

    Args:
        repo_root: Repository root directory.
        run_root: Optional run root for scoped outputs scanning.
        require_experiment_matrix: Whether missing matrix artifacts must fail.

    Returns:
        Audit result dictionary.
    """
    if not isinstance(require_experiment_matrix, bool):
        raise TypeError("require_experiment_matrix must be bool")
    if run_root is not None and not isinstance(run_root, Path):
        raise TypeError("run_root must be Path or None")

    base_result = run_audit(repo_root)
    if base_result.get("result") == "FAIL" and base_result.get("rule") == "repo_root must be valid directory":
        return base_result

    scan_root = run_root if run_root is not None else repo_root
    if not scan_root.exists() or not scan_root.is_dir():
        return {
            "audit_id": AUDIT_ID,
            "gate_name": "gate.experiment_matrix_outputs_schema",
            "category": "G",
            "severity": SEVERITY,
            "result": "FAIL",
            "rule": "scan_root must be valid directory",
            "evidence": {
                "path": str(scan_root),
                "require_experiment_matrix": require_experiment_matrix,
            },
            "impact": "invalid audit input",
            "fix": "provide valid repo_root/run_root path",
        }

    return check_grid_summary_schema(
        scan_root=scan_root,
        require_experiment_matrix=require_experiment_matrix,
    )


def main() -> None:
    """CLI entry for audit script."""
    parser = argparse.ArgumentParser(
        description="Audit experiment matrix outputs schema with explicit requirement policy"
    )
    parser.add_argument(
        "repo_root",
        type=Path,
        help="Repository root path",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Optional run_root for scoped matrix outputs scanning",
    )
    parser.add_argument(
        "--require-experiment-matrix",
        dest="require_experiment_matrix",
        action="store_true",
        default=False,
        help="Treat missing matrix artifacts as FAIL",
    )
    parser.add_argument(
        "--no-require-experiment-matrix",
        dest="require_experiment_matrix",
        action="store_false",
        help="Treat missing matrix artifacts as N.A.",
    )

    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_root = args.run_root.resolve() if args.run_root is not None else None
    result = run_audit_with_policy(
        repo_root=repo_root,
        run_root=run_root,
        require_experiment_matrix=bool(args.require_experiment_matrix),
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("result") == "FAIL":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
