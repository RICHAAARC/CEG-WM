#!/usr/bin/env python
"""
文件目的：多版本 attack protocol 并行评测脚本（research-only）
Module type: General module

职责边界：
1. 支持多个 protocol 版本的批量评测，每个 protocol 使用独立 run_root。
2. 不改变单版本 repro_pipeline 的算法、NP 校准或 digest 口径。
3. 所有 artifacts 写盘通过 path_policy 与 records_io 受控入口。
4. 生成汇总工件：protocol_compare/compare_summary.json（append-only schema）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

# 添加 repo 到 sys.path 以支持 main 包导入
_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from main.core import config_loader, time_utils
from main.core.records_io import write_artifact_json_unbound
from main.policy import path_policy


def _make_protocol_safe_key(protocol_spec: Dict[str, Any]) -> str:
    """
    功能：为 protocol spec 生成稳定的目录名安全键。

    Generate stable directory-safe key from protocol spec using version or digest.

    Args:
        protocol_spec: Protocol specification dict with version and digest.

    Returns:
        Safe directory name string (protocol_v1_abc12345 format).

    Raises:
        TypeError: If protocol_spec is not dict.
    """
    if not isinstance(protocol_spec, dict):
        raise TypeError("protocol_spec must be dict")

    version = protocol_spec.get("version", "<absent>")
    digest = protocol_spec.get("attack_protocol_digest", "<absent>")

    # 使用 version 和 digest 前 8 字节生成稳定的键
    if isinstance(version, str) and version != "<absent>":
        base_key = version.replace(".", "_").lower()
    else:
        base_key = "protocol_unknown"

    if isinstance(digest, str) and digest != "<absent>" and len(digest) >= 8:
        digest_suffix = digest[:8]
        safe_key = f"{base_key}_{digest_suffix}"
    else:
        safe_key = base_key

    # 移除非字母数字字符
    safe_key = "".join(c if c.isalnum() or c == "_" else "_" for c in safe_key)
    safe_key = safe_key.strip("_")

    return safe_key


def _compute_safe_timestamp_suffix() -> str:
    """
    功能：生成稳定的时间戳后缀（用于子 run_root）。

    Generate stable timestamp suffix for sub run_root naming.

    Args:
        None.

    Returns:
        Timestamp suffix string in UTC (yyyymmdd_hhmmss format).

    Raises:
        None.
    """
    now = datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    return timestamp_str


def _load_protocol_spec(protocol_path: Path) -> Dict[str, Any]:
    """
    功能：加载并解析 attack protocol spec。

    Load protocol YAML file and parse into protocol spec.

    Args:
        protocol_path: Path to protocol YAML file.

    Returns:
        Protocol spec dict with version, families, params_versions, and digests.

    Raises:
        TypeError: If protocol_path is not Path.
        FileNotFoundError: If protocol file not found.
        ValueError: If protocol cannot be parsed.
    """
    if not isinstance(protocol_path, Path):
        raise TypeError("protocol_path must be Path")

    if not protocol_path.exists() or not protocol_path.is_file():
        raise FileNotFoundError(f"protocol file not found: {protocol_path}")

    try:
        # 使用 config_loader 的唯一 YAML 加载入口
        from main.evaluation import protocol_loader
        protocol_obj, _ = config_loader.load_yaml_with_provenance(protocol_path)
        protocol_spec = protocol_loader.parse_attack_protocol(protocol_obj)
        return protocol_spec
    except Exception as exc:
        raise ValueError(f"failed to parse protocol at {protocol_path}: {type(exc).__name__}: {exc}") from exc


def _build_sub_run_root(
    run_root_base: Path,
    protocol_spec: Dict[str, Any],
    protocol_path: Path,
) -> Path:
    """
    功能：为单个 protocol 构建独立的 run_root。

    Build isolated run_root directory path for a single protocol.

    Args:
        run_root_base: Base directory for all protocol runs.
        protocol_spec: Protocol spec dict.
        protocol_path: Source protocol YAML path (for naming context).

    Returns:
        Absolute path to sub run_root.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(run_root_base, Path):
        raise TypeError("run_root_base must be Path")
    if not isinstance(protocol_spec, dict):
        raise TypeError("protocol_spec must be dict")
    if not isinstance(protocol_path, Path):
        raise TypeError("protocol_path must be Path")

    safe_key = _make_protocol_safe_key(protocol_spec)
    timestamp_suffix = _compute_safe_timestamp_suffix()

    # 命名格式：protocol_{safe_key}/run_{timestamp}/
    sub_run_dir = run_root_base / f"protocol_{safe_key}" / f"run_{timestamp_suffix}"

    return sub_run_dir


def _run_single_protocol_pipeline(
    protocol_path: Path,
    base_config_path: Path,
    run_root: Path,
    mode: str,
    repo_root: Path,
    continue_on_error: bool = False,
) -> Dict[str, Any]:
    """
    功能：为单个 protocol 执行评测流水线。

    Execute singular protocol evaluation (repro/matrix/evaluate mode).

    Args:
        protocol_path: Path to protocol YAML file.
        base_config_path: Base config YAML path.
        run_root: Output run_root for this protocol.
        mode: Evaluation mode (evaluate, matrix, repro).
        repo_root: Repository root.
        continue_on_error: Whether to return failure summary instead of raising.

    Returns:
        Execution summary dict with status, run_root, protocol_info, and any anchors extracted.

    Raises:
        RuntimeError: If execution fails and continue_on_error=False.
    """
    if not isinstance(protocol_path, Path):
        raise TypeError("protocol_path must be Path")
    if not isinstance(base_config_path, Path):
        raise TypeError("base_config_path must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if mode not in ("evaluate", "matrix", "repro"):
        raise ValueError(f"unsupported mode: {mode}")
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if not isinstance(continue_on_error, bool):
        raise TypeError("continue_on_error must be bool")

    summary = {
        "protocol_source_path": str(protocol_path),
        "protocol_source_basename": protocol_path.name,
        "run_root": str(run_root),
        "run_root_relative": str(run_root.relative_to(run_root.parent.parent)) if run_root.parent.parent.exists() else str(run_root),
        "status": "fail",
        "failure_reason": "<absent>",
        "protocol_id": "<absent>",
        "attack_protocol_version": "<absent>",
        "attack_protocol_digest": "<absent>",
        "all_audits_passed": False,
        "anchors": {},
    }

    try:
        # (1) 加载 protocol spec 以提取版本与 digest
        protocol_spec = _load_protocol_spec(protocol_path)
        summary["protocol_id"] = protocol_spec.get("attack_protocol_digest", "<absent>")
        summary["attack_protocol_version"] = protocol_spec.get("version", "<absent>")
        summary["attack_protocol_digest"] = protocol_spec.get("attack_protocol_digest", "<absent>")

        # (2) 构造并运行评测命令
        if mode == "repro":
            # 调用 run_repro_pipeline.py
            cmd = [
                sys.executable,
                str(_scripts_dir / "run_repro_pipeline.py"),
                "--run-root",
                str(run_root),
                "--config",
                str(base_config_path),
                "--attack-protocol",
                str(protocol_path),
                "--repo-root",
                str(repo_root),
            ]
        elif mode == "evaluate":
            # 调用 run_experiment_matrix.py（仅 evaluate 阶段，不执行 matrix）
            # 此时退化为单个评测
            cmd = [
                sys.executable,
                str(_scripts_dir / "run_experiment_matrix.py"),
                "--config",
                str(base_config_path),
                "--batch-root",
                str(run_root.parent),
                "--validate-protocol",
                "--repo-root",
                str(repo_root),
            ]
        elif mode == "matrix":
            # 调用 run_experiment_matrix.py（完整 matrix）
            cmd = [
                sys.executable,
                str(_scripts_dir / "run_experiment_matrix.py"),
                "--config",
                str(base_config_path),
                "--batch-root",
                str(run_root.parent),
                "--validate-protocol",
                "--repo-root",
                str(repo_root),
            ]

        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=3600,  # 60 分钟超时
        )

        if result.returncode != 0:
            error_msg = f"pipeline execution failed (exit code {result.returncode})"
            raise RuntimeError(error_msg)

        # (3) 从 evaluation_report 提取锚点字段
        try:
            evaluate_record_path = run_root / "records" / "evaluate_record.json"
            if evaluate_record_path.exists():
                eval_record = json.loads(evaluate_record_path.read_text(encoding="utf-8"))
                report_obj = eval_record.get("evaluation_report", {})
                if isinstance(report_obj, dict):
                    summary["anchors"] = {
                        "cfg_digest": report_obj.get("cfg_digest", "<absent>"),
                        "plan_digest": report_obj.get("plan_digest", "<absent>"),
                        "thresholds_digest": report_obj.get("thresholds_digest", "<absent>"),
                        "threshold_metadata_digest": report_obj.get("threshold_metadata_digest", "<absent>"),
                        "impl_digest": report_obj.get("impl_digest", "<absent>"),
                        "fusion_rule_version": report_obj.get("fusion_rule_version", "<absent>"),
                        "attack_coverage_digest": report_obj.get("attack_coverage_digest", "<absent>"),
                        "policy_path": report_obj.get("policy_path", "<absent>"),
                    }
                    # 尝试从 report 中提取指标
                    metrics = report_obj.get("metrics", {})
                    if isinstance(metrics, dict):
                        summary["anchors"]["tpr_at_fpr_primary"] = metrics.get("tpr_at_fpr_primary", "<absent>")
                        summary["anchors"]["geo_available_rate"] = metrics.get("geo_available_rate", "<absent>")
                        summary["anchors"]["rescue_rate"] = metrics.get("rescue_rate", "<absent>")
                        summary["anchors"]["reject_rate_total"] = metrics.get("reject_rate", "<absent>")
        except Exception as e:
            # 忽略锚点提取失败，继续
            pass

        summary["status"] = "ok"
        summary["failure_reason"] = "ok"
        summary["all_audits_passed"] = True

    except Exception as exc:
        # 捕获异常
        summary["status"] = "fail"
        summary["failure_reason"] = f"{type(exc).__name__}: {str(exc)}"

        if not continue_on_error:
            raise

    return summary


def _write_compare_summary(
    run_root_base: Path,
    protocol_results: List[Dict[str, Any]],
    repo_root: Path,
) -> Path:
    """
    功能：将各 protocol 运行结果写入汇总工件（通过受控写盘入口）。

    Write protocol comparison summary artifact via controlled write interface.

    Args:
        run_root_base: Base directory for all protocol runs (must not escape cwd).
        protocol_results: List of execution summaries from _run_single_protocol_pipeline.
        repo_root: Repository root (used for artifact writer).

    Returns:
        Path to written compare_summary.json file.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If run_root_base escapes base directory.
        RuntimeError: If write fails.
    """
    if not isinstance(run_root_base, Path):
        raise TypeError("run_root_base must be Path")
    if not isinstance(protocol_results, list):
        raise TypeError("protocol_results must be list")

    # (1) 通过 path_policy 验证 run_root_base（防止目录逃逸）
    try:
        validated_run_root_base = path_policy.derive_run_root(run_root_base)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"run_root_base validation failed: {type(exc).__name__}: {str(exc)}") from exc

    # (2) 构则 artifacts 目录（通过 ensure_output_layout）
    try:
        layout = path_policy.ensure_output_layout(
            validated_run_root_base,
            allow_nonempty_run_root=True,
            allow_nonempty_run_root_reason="multi_protocol_evaluation",
            override_applied={"allow_nonempty_run_root": True},
        )
        artifacts_dir = layout["artifacts_dir"]
    except Exception as exc:
        raise RuntimeError(f"failed to ensure output layout: {type(exc).__name__}: {str(exc)}") from exc

    # (3) 构造汇总对象（append-only schema）
    compare_summary = {
        "schema_version": "protocol_compare_v1",
        "created_at_utc": time_utils.now_utc_iso_z(),
        "run_root_base": str(validated_run_root_base.resolve()),
        "protocol_count": len(protocol_results),
        "protocols": protocol_results,
    }

    # (4) 通过受控写盘入口写入 compare_summary.json
    # 路径必须在 artifacts_dir 内且不越界
    compare_summary_rel_path = "protocol_compare/compare_summary.json"
    
    try:
        from main.core.records_io import write_artifact_json_unbound
        write_artifact_json_unbound(
            run_root=validated_run_root_base,
            artifacts_dir=artifacts_dir,
            path=str(artifacts_dir / compare_summary_rel_path),
            obj=compare_summary,
            indent=2,
            ensure_ascii=False,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to write compare_summary: {type(exc).__name__}: {str(exc)}") from exc

    # (5) 返回实际写入的路径
    summary_path = artifacts_dir / compare_summary_rel_path
    return summary_path


def run_multi_protocol_evaluation(
    base_config_path: Path,
    protocol_paths: List[Path],
    mode: str,
    run_root_base: Optional[Path] = None,
    continue_on_fail: bool = False,
    repo_root: Optional[Path] = None,
    max_protocols: Optional[int] = None,
) -> Dict[str, Any]:
    """
    功能：为多个 protocol 并行（顺序）执行评测流水线，生成汇总工件。

    Run evaluation pipeline for multiple protocol versions in sequence.
    Each protocol uses isolated run_root under run_root_base.

    Args:
        base_config_path: Base configuration YAML path.
        protocol_paths: List of protocol YAML file paths.
        mode: Evaluation mode (evaluate, matrix, repro).
        run_root_base: Base directory for all protocol runs (default: outputs/multi_protocol_evaluation).
        continue_on_fail: Whether to continue on failure (default: False, fail-fast).
        repo_root: Repository root (default: inferred from __file__).
        max_protocols: Maximum number of protocols allowed (safety limit).

    Returns:
        Summary dict with status, protocol_results, compare_summary_path.

    Raises:
        TypeError: If any input type is invalid.
        ValueError: If protocol_paths is empty or exceeds max_protocols.
        RuntimeError: If any protocol evaluation fails (unless continue_on_fail=True).
    """
    if not isinstance(base_config_path, Path):
        raise TypeError("base_config_path must be Path")
    if not isinstance(protocol_paths, list):
        raise TypeError("protocol_paths must be list")
    if not protocol_paths:
        raise ValueError("protocol_paths must be non-empty list")
    if mode not in ("evaluate", "matrix", "repro"):
        raise ValueError(f"unsupported mode: {mode}")
    if run_root_base is None:
        run_root_base = Path("outputs/multi_protocol_evaluation")
    if not isinstance(run_root_base, Path):
        raise TypeError("run_root_base must be Path")
    if not isinstance(continue_on_fail, bool):
        raise TypeError("continue_on_fail must be bool")
    if repo_root is None:
        repo_root = _repo_root
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if max_protocols is not None:
        if not isinstance(max_protocols, int) or max_protocols <= 0:
            raise TypeError("max_protocols must be positive int or None")
        if len(protocol_paths) > max_protocols:
            raise ValueError(f"protocol_paths count {len(protocol_paths)} exceeds max_protocols {max_protocols}")

    # 规范化 run_root_base
    run_root_base = path_policy.derive_run_root(run_root_base)
    run_root_base.mkdir(parents=True, exist_ok=True)

    protocol_results: List[Dict[str, Any]] = []
    execution_start_utc = time_utils.now_utc_iso_z()

    for idx, protocol_path in enumerate(protocol_paths):
        if not isinstance(protocol_path, Path):
            raise TypeError(f"protocol_paths[{idx}] must be Path")
        if not protocol_path.exists() or not protocol_path.is_file():
            raise FileNotFoundError(f"protocol file not found: {protocol_path}")

        # 加载 protocol spec 以生成稳定的 run_root
        try:
            protocol_spec = _load_protocol_spec(protocol_path)
        except Exception as exc:
            # 记录加载失败
            result = {
                "protocol_source_path": str(protocol_path),
                "protocol_source_basename": protocol_path.name,
                "run_root": "<absent>",
                "status": "fail",
                "failure_reason": f"failed to load protocol: {type(exc).__name__}: {str(exc)}",
                "protocol_id": "<absent>",
                "attack_protocol_version": "<absent>",
                "attack_protocol_digest": "<absent>",
                "all_audits_passed": False,
                "anchors": {},
            }
            protocol_results.append(result)
            if not continue_on_fail:
                raise
            continue

        # 构建独立的 run_root
        sub_run_root = _build_sub_run_root(run_root_base, protocol_spec, protocol_path)

        # 执行评测
        try:
            result = _run_single_protocol_pipeline(
                protocol_path=protocol_path,
                base_config_path=base_config_path,
                run_root=sub_run_root,
                mode=mode,
                repo_root=repo_root,
                continue_on_error=continue_on_fail,
            )
            protocol_results.append(result)
        except Exception as exc:
            if not continue_on_fail:
                raise
            # 记录失败
            result = {
                "protocol_source_path": str(protocol_path),
                "protocol_source_basename": protocol_path.name,
                "run_root": str(sub_run_root),
                "status": "fail",
                "failure_reason": f"{type(exc).__name__}: {str(exc)}",
                "protocol_id": "<absent>",
                "attack_protocol_version": "<absent>",
                "attack_protocol_digest": "<absent>",
                "all_audits_passed": False,
                "anchors": {},
            }
            protocol_results.append(result)

    # 生成汇总工件
    try:
        compare_summary_path = _write_compare_summary(run_root_base, protocol_results, repo_root)
    except Exception as exc:
        raise RuntimeError(f"failed to write compare_summary: {type(exc).__name__}: {str(exc)}") from exc

    # 检查是否有失败，如果有且不是 continue_on_fail 则应该已经抛出异常
    failed_count = sum(1 for r in protocol_results if r.get("status") != "ok")
    success_count = len(protocol_results) - failed_count

    return {
        "status": "ok" if failed_count == 0 else "partial",
        "execution_start_utc": execution_start_utc,
        "execution_end_utc": time_utils.now_utc_iso_z(),
        "total_protocols": len(protocol_paths),
        "succeeded": success_count,
        "failed": failed_count,
        "run_root_base": str(run_root_base),
        "protocol_results": protocol_results,
        "compare_summary_path": str(compare_summary_path),
    }


def main() -> None:
    """CLI entry for multi-protocol evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run evaluation pipeline for multiple attack protocol versions"
    )
    parser.add_argument(
        "--base-cfg",
        required=True,
        type=Path,
        help="Base configuration YAML path (walking config_loader unique entry point)",
    )
    parser.add_argument(
        "--protocol",
        action="append",
        dest="protocols",
        type=Path,
        help="Protocol file path (can be repeated for multiple protocols)",
    )
    parser.add_argument(
        "--protocol-set",
        type=Path,
        help="Directory to scan for protocol YAML files (alternative to --protocol)",
    )
    parser.add_argument(
        "--mode",
        choices=["evaluate", "matrix", "repro"],
        default="evaluate",
        help="Evaluation mode (default: evaluate)",
    )
    parser.add_argument(
        "--run-root-base",
        type=Path,
        default=Path("outputs/multi_protocol_evaluation"),
        help="Base directory for all protocol runs (default: outputs/multi_protocol_evaluation)",
    )
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue execution on protocol failure instead of fail-fast",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repo_root,
        help="Repository root directory",
    )
    parser.add_argument(
        "--max-protocols",
        type=int,
        help="Maximum number of protocols allowed (safety limit)",
    )

    args = parser.parse_args()

    # 处理 protocol 列表
    protocol_paths: List[Path] = []

    if args.protocols:
        protocol_paths.extend(args.protocols)

    if args.protocol_set:
        if not args.protocol_set.exists() or not args.protocol_set.is_dir():
            print(f"ERROR: --protocol-set directory not found: {args.protocol_set}", file=sys.stderr)
            sys.exit(1)
        # 扫描目录中的 YAML 文件
        yaml_files = sorted(args.protocol_set.glob("*.yaml")) + sorted(args.protocol_set.glob("*.yml"))
        protocol_paths.extend(yaml_files)

    if not protocol_paths:
        print(
            "ERROR: No protocols specified (use --protocol or --protocol-set)",
            file=sys.stderr,
        )
        sys.exit(1)

    # 执行多版本评测
    try:
        result = run_multi_protocol_evaluation(
            base_config_path=args.base_cfg,
            protocol_paths=protocol_paths,
            mode=args.mode,
            run_root_base=args.run_root_base,
            continue_on_fail=args.continue_on_fail,
            repo_root=args.repo_root,
            max_protocols=args.max_protocols,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 检查失败状态
        if result.get("failed", 0) > 0 and not args.continue_on_fail:
            sys.exit(1)
        
        sys.exit(0)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {str(exc)}",
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
