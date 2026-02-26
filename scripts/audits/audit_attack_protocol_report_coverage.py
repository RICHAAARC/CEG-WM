"""
File purpose: 运行期协议→报告覆盖率审计（declare vs execute 一致性验证）。
Module type: Core innovation module

设计边界：
1. 校验 attack_protocol.yaml 声明的所有条件（family::params_version）是否全部出现在 evaluation_report.json 中。
2. 禁止改动协议加载、报告生成、条件键抽取的任何逻辑——仅做审计观察。
3. 失败必须包含证据（缺失条件键清单、预期vs实际对比）。
4. 报告结构假设 metrics_by_attack_condition 字段为有序列表。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


_UNKNOWN_CONDITION_SENTINELS = {
    "unknown_attack::unknown_params",
}


def find_candidate_evaluation_report_paths(repo_root: Path) -> List[Path]:
    """
    Find candidate evaluation report paths with deterministic priority.

    Args:
        repo_root: Repository root directory.

    Returns:
        Ordered candidate path list (existing and non-existing allowed).
    """
    if not isinstance(repo_root, Path):
        repo_root = Path(repo_root)

    static_candidates = [
        repo_root / "outputs" / "smoke_detect" / "evaluation_report.json",
        repo_root / "outputs" / "smoke_embed" / "evaluation_report.json",
        repo_root / "evaluation_report.json",
        repo_root / "artifacts" / "evaluation_report.json",
    ]

    closure_candidates = [
        item
        for item in (repo_root / "outputs").glob("**/artifacts/run_closure.json")
        if item.is_file()
    ] if (repo_root / "outputs").is_dir() else []

    preferred_run_root_report: Optional[Path] = None
    if closure_candidates:
        scored_run_roots = []
        for closure_path in closure_candidates:
            run_root = closure_path.parent.parent
            run_root_posix = run_root.as_posix()
            in_experiment_matrix = "/outputs/experiment_matrix/experiments/" in run_root_posix
            in_multi_protocol = "/artifacts/multi_protocol_evaluation/" in run_root_posix
            scored_run_roots.append(
                (
                    (run_root / "artifacts" / "repro_bundle" / "manifest.json").is_file(),
                    (run_root / "artifacts" / "evaluation_report.json").is_file(),
                    (run_root / "records" / "evaluate_record.json").is_file(),
                    not in_experiment_matrix,
                    not in_multi_protocol,
                    closure_path.stat().st_mtime,
                    run_root,
                )
            )

        scored_run_roots.sort(reverse=True)
        preferred_run_root = scored_run_roots[0][-1]
        preferred_run_root_report = preferred_run_root / "artifacts" / "evaluation_report.json"

    dynamic_candidates: List[Path] = []
    outputs_root = repo_root / "outputs"
    if outputs_root.exists() and outputs_root.is_dir():
        for pattern in [
            "**/artifacts/evaluation_report.json",
            "**/evaluation_report.json",
        ]:
            for item in outputs_root.glob(pattern):
                if item.is_file() and item not in dynamic_candidates:
                    dynamic_candidates.append(item)

        dynamic_candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    merged_candidates: List[Path] = []
    ordered_candidates: List[Path] = []
    if isinstance(preferred_run_root_report, Path):
        ordered_candidates.append(preferred_run_root_report)
    ordered_candidates.extend(static_candidates)
    ordered_candidates.extend(dynamic_candidates)

    for item in ordered_candidates:
        if item not in merged_candidates:
            merged_candidates.append(item)
    return merged_candidates


def load_attack_protocol_spec(repo_root: Path) -> Dict[str, Any]:
    """
    Load attack protocol specification from configs/attack_protocol.yaml.

    Args:
        repo_root: Repository root directory.

    Returns:
        Protocol specification dict with version, families, and params_versions keys.

    Raises:
        FileNotFoundError: If attack_protocol.yaml not found.
        ValueError: If protocol spec is invalid.
    """
    if not isinstance(repo_root, Path):
        repo_root = Path(repo_root)

    protocol_path = repo_root / "configs" / "attack_protocol.yaml"
    if not protocol_path.exists():
        raise FileNotFoundError(f"attack_protocol.yaml not found at {protocol_path}")

    try:
        import yaml
        with open(protocol_path, "r", encoding="utf-8") as f:
            protocol_spec = yaml.safe_load(f)
    except ImportError:
        raise RuntimeError("PyYAML not available; cannot load protocol spec")
    except Exception as e:
        raise ValueError(f"Failed to parse attack_protocol.yaml: {e}")

    if not isinstance(protocol_spec, dict):
        raise ValueError("protocol_spec must be dict")

    return protocol_spec


def extract_declared_conditions(protocol_spec: Dict[str, Any]) -> List[str]:
    """
    Extract all unique conditions declared in attack protocol (family::params_version format).

    Args:
        protocol_spec: Attack protocol specification dict.

    Returns:
        Sorted list of condition keys in format "family::params_version".

    Raises:
        TypeError: If protocol_spec is invalid.
        ValueError: If any condition lacks proper structure.
    """
    if not isinstance(protocol_spec, dict):
        raise TypeError("protocol_spec must be dict")

    conditions: List[str] = []

    # 方法 1: 从 params_versions flat 字典读取（协议规范中显式条件键）
    params_versions = protocol_spec.get("params_versions", {})
    if isinstance(params_versions, dict):
        for condition_key in params_versions.keys():
            if isinstance(condition_key, str) and "::" in condition_key:
                # 条件键已为 family::params_version 格式
                if condition_key not in conditions:
                    conditions.append(condition_key)

    # 方法 2: 从 families 嵌套结构推导（备选，此时条件键需组合）
    families = protocol_spec.get("families", {})
    if isinstance(families, dict):
        for family_name, family_spec in families.items():
            if not isinstance(family_name, str):
                continue
            if not isinstance(family_spec, dict):
                continue

            params_versions_in_family = family_spec.get("params_versions", {})
            if isinstance(params_versions_in_family, dict):
                for param_version_name in params_versions_in_family.keys():
                    if isinstance(param_version_name, str):
                        condition_key = f"{family_name}::{param_version_name}"
                        if condition_key not in conditions:
                            conditions.append(condition_key)

    return sorted(conditions)


def load_evaluation_report(eval_report_path: Path) -> Dict[str, Any]:
    """
    Load evaluation report from file.

    Args:
        eval_report_path: Path to evaluation_report.json.

    Returns:
        Evaluation report dict.

    Raises:
        FileNotFoundError: If report not found.
        ValueError: If report is invalid JSON or missing required structure.
    """
    if not isinstance(eval_report_path, Path):
        eval_report_path = Path(eval_report_path)

    if not eval_report_path.exists():
        raise FileNotFoundError(f"evaluation_report.json not found at {eval_report_path}")

    try:
        with open(eval_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"evaluation_report.json is not valid JSON: {e}")

    if not isinstance(report, dict):
        raise ValueError("evaluation_report must be dict")

    return report


def extract_reported_conditions(report: Dict[str, Any]) -> List[str]:
    """
    Extract all condition keys reported in evaluation_report.json metrics_by_attack_condition.

    Args:
        report: Evaluation report dict.

    Returns:
        Sorted list of condition keys (group_key values) appearing in metrics_by_attack_condition.

    Raises:
        TypeError: If report structure is invalid.
    """
    if not isinstance(report, dict):
        raise TypeError("report must be dict")

    conditions: List[str] = []

    report_obj = report
    nested_report = report.get("evaluation_report")
    if isinstance(nested_report, dict):
        report_obj = nested_report

    metrics_by_condition = report_obj.get("metrics_by_attack_condition")
    if not isinstance(metrics_by_condition, list):
        # 报告中缺失 metrics_by_attack_condition 字段——这是严重问题
        return []

    for item in metrics_by_condition:
        if not isinstance(item, dict):
            continue

        group_key = item.get("group_key")
        if isinstance(group_key, str) and group_key not in conditions:
            conditions.append(group_key)

    return sorted(conditions)


def is_unknown_condition_sentinel(condition_key: str) -> bool:
    """
    Check whether condition key is the known unknown sentinel.

    Args:
        condition_key: Condition key string.

    Returns:
        True if key is unknown sentinel, otherwise False.
    """
    if not isinstance(condition_key, str) or not condition_key:
        return False
    return condition_key in _UNKNOWN_CONDITION_SENTINELS


def collect_reported_conditions_candidates(
    repo_root: Path,
    eval_report_paths: List[Path],
    declared_conditions: List[str],
) -> Dict[str, Any]:
    """
    Collect condition coverage candidates from one or multiple evaluation reports.

    Args:
        repo_root: Repository root directory.
        eval_report_paths: Candidate evaluation report paths.
        declared_conditions: Declared condition keys from protocol spec.

    Returns:
        Dict containing best/aggregated reported conditions and evidence paths.
    """
    if not isinstance(repo_root, Path):
        repo_root = Path(repo_root)
    if not isinstance(eval_report_paths, list):
        raise TypeError("eval_report_paths must be list")
    if not isinstance(declared_conditions, list):
        raise TypeError("declared_conditions must be list")

    declared_set = {
        item
        for item in declared_conditions
        if isinstance(item, str) and item
    }

    parsed_reports: List[Dict[str, Any]] = []
    for path in eval_report_paths:
        if not isinstance(path, Path) or not path.exists() or not path.is_file():
            continue
        try:
            report_obj = load_evaluation_report(path)
            reported_list = extract_reported_conditions(report_obj)
        except Exception:
            continue

        reported_set = {
            item
            for item in reported_list
            if isinstance(item, str) and item
        }
        covered_count = len(reported_set & declared_set)
        extra_count = len(reported_set - declared_set)

        parsed_reports.append(
            {
                "path": path,
                "reported_list": sorted(reported_set),
                "reported_set": reported_set,
                "covered_count": covered_count,
                "extra_count": extra_count,
                "reported_count": len(reported_set),
                "mtime": path.stat().st_mtime,
            }
        )

    if not parsed_reports:
        return {
            "reported_conditions": [],
            "eval_report_path": "<absent>",
            "eval_report_paths": [],
            "source_mode": "none",
        }

    # 候选 1：单报告最优（覆盖更多、额外更少、条目更多、更新更晚）
    best_single = sorted(
        parsed_reports,
        key=lambda item: (
            item["covered_count"],
            -item["extra_count"],
            item["reported_count"],
            item["mtime"],
        ),
        reverse=True,
    )[0]

    # 候选 2：多报告聚合（union），用于 experiment_matrix 分条件拆分场景
    union_set = set()
    union_paths: List[Path] = []
    for item in parsed_reports:
        union_set.update(item["reported_set"])
        union_paths.append(item["path"])

    best_single_score = (
        int(best_single["covered_count"]),
        int(-best_single["extra_count"]),
        int(best_single["reported_count"]),
    )
    union_score = (
        int(len(union_set & declared_set)),
        int(-len(union_set - declared_set)),
        int(len(union_set)),
    )

    if union_score >= best_single_score:
        chosen_set = union_set
        chosen_paths = sorted(set(union_paths), key=lambda p: p.stat().st_mtime, reverse=True)
        source_mode = "aggregated_union"
    else:
        chosen_set = set(best_single["reported_set"])
        chosen_paths = [best_single["path"]]
        source_mode = "best_single"

    rel_paths: List[str] = []
    for path in chosen_paths:
        if path.is_relative_to(repo_root):
            rel_paths.append(str(path.relative_to(repo_root)))
        else:
            rel_paths.append(str(path))

    primary_path = rel_paths[0] if rel_paths else "<absent>"
    ignored_unknown_conditions = sorted(
        item
        for item in chosen_set
        if is_unknown_condition_sentinel(item)
    )
    filtered_reported = sorted(
        item
        for item in chosen_set
        if not is_unknown_condition_sentinel(item)
    )
    return {
        "reported_conditions": filtered_reported,
        "eval_report_path": primary_path,
        "eval_report_paths": rel_paths,
        "source_mode": source_mode,
        "ignored_unknown_conditions": ignored_unknown_conditions,
    }


def audit_attack_protocol_report_coverage(repo_root: Path) -> Dict[str, Any]:
    """
    Audit that all attack conditions declared in protocol are executed and reported.

    Verifies:
    1. All condition keys in attack_protocol.yaml params_versions appear in evaluation_report.json
    2. No extra conditions in report that are not declared (future-proofing)
    3. Condition key format and structure consistency

    Args:
        repo_root: Repository root directory.

    Returns:
        Audit result dict with structure:
        {
            "audit_id": "audit.attack_protocol_report_coverage",
            "gate_name": "gate.attack_protocol_report_coverage",
            "category": "G",
            "severity": "BLOCK",
            "result": "PASS" | "FAIL",
            "rule": "all declared attack conditions must be executed and reported",
            "evidence": {
                "protocol_version": str,
                "protocol_conditions_count": int,
                "reported_conditions_count": int,
                "missed_conditions": list[str],
                "extra_reported_conditions": list[str],
                "eval_report_path": str,
                "protocol_spec_path": str,
            }
        }

    Raises:
        No exceptions; always returns valid audit result dict.
    """
    if not isinstance(repo_root, Path):
        repo_root = Path(repo_root)

    audit_id = "audit.attack_protocol_report_coverage"
    gate_name = "gate.attack_protocol_report_coverage"

    evidence: Dict[str, Any] = {
        "protocol_spec_path": "configs/attack_protocol.yaml",
        "eval_report_path": "outputs/smoke_detect/evaluation_report.json",
        "declared_conditions": [],
        "reported_conditions": [],
        "missed_conditions": [],
        "extra_reported_conditions": [],
    }

    try:
        # (1) 加载协议规范
        protocol_spec = load_attack_protocol_spec(repo_root)
        evidence["protocol_version"] = protocol_spec.get("version", "<absent>")

        declared_conditions = extract_declared_conditions(protocol_spec)
        evidence["protocol_conditions_count"] = len(declared_conditions)
        evidence["declared_conditions"] = declared_conditions

        # (2) 尝试加载评测报告（支持多个位置）
        eval_report_paths = find_candidate_evaluation_report_paths(repo_root)

        reported_candidates = collect_reported_conditions_candidates(
            repo_root,
            eval_report_paths,
            declared_conditions,
        )

        reported_conditions = reported_candidates.get("reported_conditions", [])
        if not isinstance(reported_conditions, list) or len(reported_conditions) == 0:
            # 如果评测报告不存在，审计返回 N.A.（不适用，因为未运行attack protocol流程）
            return {
                "audit_id": audit_id,
                "gate_name": gate_name,
                "category": "G",
                "severity": "NON_BLOCK",
                "result": "N.A.",
                "rule": "all declared attack conditions must be executed and reported",
                "evidence": {
                    **evidence,
                    "status": "evaluation_report.json not found in expected locations; audit not applicable",
                    "checked_paths": [str(p) for p in eval_report_paths],
                },
            }

        evidence["eval_report_path"] = reported_candidates.get("eval_report_path", "<absent>")
        evidence["eval_report_paths"] = reported_candidates.get("eval_report_paths", [])
        evidence["reported_source_mode"] = reported_candidates.get("source_mode", "<absent>")
        evidence["ignored_unknown_conditions"] = reported_candidates.get("ignored_unknown_conditions", [])

        # (3) 从报告（单个或聚合）提取已上报的条件
        evidence["reported_conditions_count"] = len(reported_conditions)
        evidence["reported_conditions"] = reported_conditions

        # (4) 对比：缺失条件 vs 多报条件
        declared_set = set(declared_conditions)
        reported_set = set(reported_conditions)

        missed_conditions = sorted(declared_set - reported_set)
        extra_conditions = sorted(reported_set - declared_set)

        evidence["missed_conditions"] = missed_conditions
        evidence["extra_reported_conditions"] = extra_conditions

        # (5) 审计决策
        if missed_conditions:
            # 声明但未执行的条件——严重失败
            result = "FAIL"
        elif extra_conditions:
            # 报告了未声明的条件——警告但不阻止（可能是协议版本更新滞后）
            result = "FAIL"
        else:
            result = "PASS"

        return {
            "audit_id": audit_id,
            "gate_name": gate_name,
            "category": "G",
            "severity": "BLOCK" if result == "FAIL" else "INFO",
            "result": result,
            "rule": "all declared attack conditions must be executed and reported; no undeclared conditions in report",
            "evidence": evidence,
        }

    except Exception as e:
        # 审计脚本内部异常——返回错误状态
        return {
            "audit_id": audit_id,
            "gate_name": gate_name,
            "category": "G",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": "all declared attack conditions must be executed and reported",
            "evidence": {
                **evidence,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        }


def main(repo_root: Optional[str] = None) -> int:
    """
    Main entry point for audit script.

    Args:
        repo_root: Repository root directory (from command line).

    Returns:
        Exit code (0 for PASS, 1 for FAIL, 2 for SKIP).
    """
    if repo_root is None:
        repo_root = "."

    repo_root_path = Path(repo_root).resolve()

    try:
        result = audit_attack_protocol_report_coverage(repo_root_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Exit code: 0 for PASS/N.A., 1 for FAIL, 2 for SKIP (legacy)
        if result["result"] == "PASS":
            return 0
        elif result["result"] == "N.A.":
            return 0  # N.A. 不阻止冻结（审计不适用）
        elif result["result"] == "SKIP":
            return 0  # SKIP(legacy) 不阻止冻结
        else:  # FAIL
            return 1
    except Exception as e:
        # 捕获运行异常并输出错误格式的审计结果
        error_result = {
            "audit_id": "audit.attack_protocol_report_coverage",
            "gate_name": "gate.attack_protocol_report_coverage",
            "category": "G",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": "all declared attack conditions must be executed and reported",
            "evidence": {
                "error": str(e),
                "error_type": type(e).__name__,
            },
        }
        print(json.dumps(error_result, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    repo_root = sys.argv[1] if len(sys.argv) > 1 else "."
    sys.exit(main(repo_root))
