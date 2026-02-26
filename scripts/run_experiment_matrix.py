#!/usr/bin/env python
"""
功能：实验矩阵批量评测脚本级入口（论文级汇总工件生成器）

Module type: General module

提供脚本级 CLI 入口，基于 main.evaluation.experiment_matrix 执行批量评测，
输出可复算汇总工件与表格（用于论文/发布级复现）。
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 添加 repo 到 sys.path 以支持 main 包导入
_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from main.core import config_loader
from main.core.records_io import write_artifact_json_unbound
from main.evaluation import experiment_matrix
from main.evaluation import protocol_loader
from main.evaluation import attack_coverage
from main.evaluation import attack_protocol_guard
from main.policy import path_policy


def _resolve_controlled_summary_output_path(
    output_summary: str,
    batch_root: str,
) -> Dict[str, Path]:
    """
    功能：解析并校验受控 summary 输出路径。

    Resolve and validate controlled summary output path under batch_root/artifacts/experiment_matrix.

    Args:
        output_summary: CLI-provided summary filename.
        batch_root: Batch root path from grid summary.

    Returns:
        Mapping containing validated run_root, artifacts_dir, and target_path.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If output_summary is absolute path, contains traversal, or batch_root is invalid.
        RuntimeError: If output layout cannot be ensured.
    """
    if not isinstance(output_summary, str) or not output_summary.strip():
        # output_summary 输入不合法，必须 fail-fast。
        raise TypeError("output_summary must be non-empty str")
    if not isinstance(batch_root, str) or not batch_root.strip():
        # batch_root 输入不合法，必须 fail-fast。
        raise TypeError("batch_root must be non-empty str")

    output_name = Path(output_summary.strip())
    if output_name.is_absolute():
        # 禁止绝对路径写盘，防止绕过受控目录。
        raise ValueError("--output-summary must be filename only; absolute path is not allowed")
    if output_name.name != output_name.as_posix() or output_name.name in {"", ".", ".."}:
        # 禁止目录分量和逃逸分量，强制仅允许文件名。
        raise ValueError("--output-summary must be filename only; directory components are not allowed")

    try:
        validated_run_root = path_policy.derive_run_root(Path(batch_root))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"batch_root validation failed: {type(exc).__name__}: {exc}") from exc

    try:
        layout = path_policy.ensure_output_layout(
            validated_run_root,
            allow_nonempty_run_root=True,
            allow_nonempty_run_root_reason="experiment_matrix_summary_write",
            override_applied={"allow_nonempty_run_root": True},
        )
    except Exception as exc:
        raise RuntimeError(f"failed to ensure output layout: {type(exc).__name__}: {exc}") from exc

    artifacts_dir = layout["artifacts_dir"]
    target_path = artifacts_dir / "experiment_matrix" / output_name.name

    return {
        "run_root": validated_run_root,
        "artifacts_dir": artifacts_dir,
        "target_path": target_path,
    }


def run_experiment_matrix_batch(
    config_path: str,
    strict: bool = False,
    validate_protocol: bool = True,
    batch_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    功能：执行实验矩阵批量评测并输出汇总工件。

    Build grid from config, optionally validate protocol implementability,
    and run full experiment matrix pipeline with aggregate artifact output.

    Args:
        config_path: YAML config path.
        strict: Whether to fail-fast on first failed sub-run.
        validate_protocol: Whether to execute protocol implementability guard before grid execution.
        batch_root: Optional batch root override for matrix artifacts.

    Returns:
        Grid summary mapping containing paths to all output artifacts.

    Raises:
        RuntimeError: If protocol validation fails or grid execution fails in strict mode.
    """
    if not isinstance(config_path, str) or not config_path:
        raise TypeError("config_path must be non-empty str")
    if not isinstance(strict, bool):
        raise TypeError("strict must be bool")
    if not isinstance(validate_protocol, bool):
        raise TypeError("validate_protocol must be bool")
    if batch_root is not None and (not isinstance(batch_root, str) or not batch_root):
        raise TypeError("batch_root must be non-empty str or None")

    # (1) 加载配置（通过唯一入口 config_loader）
    cfg_obj, _ = config_loader.load_yaml_with_provenance(Path(config_path))
    cfg_dict: Dict[str, Any] = dict(cfg_obj)
    config_loader.normalize_ablation_flags(cfg_dict)

    if batch_root is not None:
        matrix_cfg_obj = cfg_dict.get("experiment_matrix")
        if matrix_cfg_obj is None:
            cfg_dict["experiment_matrix"] = {"batch_root": batch_root}
        elif isinstance(matrix_cfg_obj, dict):
            matrix_cfg_obj["batch_root"] = batch_root
        else:
            raise TypeError("experiment_matrix config must be dict when present")

    # (2) 可选：执行协议可实现性门禁（评测前 fail-fast）
    if validate_protocol:
        protocol_spec = protocol_loader.load_attack_protocol_spec(cfg_dict)
        coverage_manifest = attack_coverage.compute_attack_coverage_manifest()
        attack_protocol_guard.assert_attack_protocol_is_implementable(
            protocol_spec,
            coverage_manifest,
        )

    # (3) 构建并执行实验矩阵
    grid = experiment_matrix.build_experiment_grid(cfg_dict)
    grid_summary = experiment_matrix.run_experiment_grid(grid, strict=strict)

    return grid_summary


def main() -> None:
    """CLI entry for experiment matrix batch runner."""
    parser = argparse.ArgumentParser(
        description="Run experiment matrix batch evaluation and publish aggregate artifacts"
    )
    parser.add_argument(
        "--config",
        default="configs/paper_full_cuda.yaml",
        help="Base config YAML path (default: configs/paper_full_cuda.yaml)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail-fast on first failed grid item (default: False)",
    )
    parser.add_argument(
        "--skip-protocol-validation",
        action="store_true",
        default=False,
        help="Skip protocol implementability validation (default: False, validation enabled)",
    )
    parser.add_argument(
        "--output-summary",
        default=None,
        help="Optional path to write grid summary JSON (default: None, only print to stdout)",
    )
    parser.add_argument(
        "--batch-root",
        default=None,
        help="Optional override for experiment_matrix.batch_root",
    )

    args = parser.parse_args()

    try:
        # 执行批量评测
        grid_summary = run_experiment_matrix_batch(
            config_path=args.config,
            strict=bool(args.strict),
            validate_protocol=not bool(args.skip_protocol_validation),
            batch_root=args.batch_root,
        )

        # 输出摘要到 stdout
        print("[ExperimentMatrixBatch] [OK]", file=sys.stderr)
        print(f"  - Total grid items: {grid_summary.get('total', 0)}", file=sys.stderr)
        print(f"  - Executed: {grid_summary.get('executed', 0)}", file=sys.stderr)
        print(f"  - Succeeded: {grid_summary.get('succeeded', 0)}", file=sys.stderr)
        print(f"  - Failed: {grid_summary.get('failed', 0)}", file=sys.stderr)
        print(f"  - Batch root: {grid_summary.get('batch_root', '<absent>')}", file=sys.stderr)
        print(f"  - Aggregate report: {grid_summary.get('aggregate_report_path', '<absent>')}", file=sys.stderr)
        print(f"  - Grid manifest: {grid_summary.get('grid_manifest_path', '<absent>')}", file=sys.stderr)
        print(f"  - Grid summary: {grid_summary.get('grid_summary_path', '<absent>')}", file=sys.stderr)
        print(f"  - Attack coverage: {grid_summary.get('attack_coverage_manifest_path', '<absent>')}", file=sys.stderr)

        # 可选：写入摘要到指定路径
        if args.output_summary:
            output_plan = _resolve_controlled_summary_output_path(
                output_summary=str(args.output_summary),
                batch_root=str(grid_summary.get("batch_root", "")),
            )
            write_artifact_json_unbound(
                run_root=output_plan["run_root"],
                artifacts_dir=output_plan["artifacts_dir"],
                path=str(output_plan["target_path"]),
                obj=grid_summary,
                indent=2,
                ensure_ascii=False,
            )
            print(f"  - Summary written to: {output_plan['target_path']}", file=sys.stderr)

        # 退出码：根据失败数决定
        if grid_summary.get("failed", 0) > 0 and args.strict:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as exc:
        print(f"[ExperimentMatrixBatch] [ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
