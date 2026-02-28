"""
File purpose: 实验矩阵一键执行 CLI 入口。
Module type: General module
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from main.core import config_loader
from main.evaluation import experiment_matrix
from main.cli.run_common import build_cli_config_migration_hint


def run_experiment_matrix(config_path: str, strict: bool = True) -> Dict[str, Any]:
    """
    功能：执行实验矩阵并输出汇总。

    Build grid from config and run full experiment matrix pipeline.

    Args:
        config_path: YAML config path.
        strict: Whether to fail-fast on first failed sub-run.

    Returns:
        Grid summary mapping.
    """
    if not config_path:
        raise TypeError("config_path must be non-empty str")

    cfg_obj, _ = config_loader.load_yaml_with_provenance(Path(config_path))
    cfg_dict: Dict[str, Any] = dict(cfg_obj)
    config_loader.normalize_ablation_flags(cfg_dict)
    config_loader._validate_paper_lf_ecc_gate(cfg_dict)  # pyright: ignore[reportPrivateUsage]
    grid = experiment_matrix.build_experiment_grid(cfg_dict)
    return experiment_matrix.run_experiment_grid(grid, strict=strict)


def main() -> None:
    """CLI entry for experiment matrix execution."""
    parser = argparse.ArgumentParser(description="Run experiment matrix and publish aggregate artifacts")
    parser.add_argument("--config", default="configs/paper_full_cuda.yaml", help="Config YAML path")
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail-fast when one grid item fails",
    )

    args = parser.parse_args()

    try:
        summary = run_experiment_matrix(args.config, strict=bool(args.strict))
        print("[ExperimentMatrix] [OK]")
        print(f"  - executed: {summary.get('executed')}")
        print(f"  - failed: {summary.get('failed')}")
        print(f"  - aggregate_report_path: {summary.get('aggregate_report_path')}")
        print(f"  - grid_manifest_path: {summary.get('grid_manifest_path')}")
        sys.exit(0)
    except Exception as exc:
        print(f"[ExperimentMatrix] [ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        hint = build_cli_config_migration_hint(exc)
        if isinstance(hint, str) and hint:
            print(f"[ExperimentMatrix] [HINT] {hint}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
