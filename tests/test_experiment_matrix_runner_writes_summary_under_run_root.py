"""
功能：实验矩阵 runner 在 run_root 下写入摘要工件

Module type: General module

Regression test: experiment matrix runner writes summary artifacts under controlled paths.
"""

import json
import tempfile
import shutil
from pathlib import Path
import pytest

# 单元级导入（不依赖脚本执行）
from main.core import config_loader
from main.evaluation import experiment_matrix


def test_experiment_matrix_runner_writes_summary_under_run_root(tmp_path: Path):
    """
    功能：实验矩阵 runner 在 run_root 下写入摘要工件。

    Verify that experiment_matrix builds and executes grid,
    writing aggregate_report.json, grid_summary.json,
    and grid_manifest.json under {batch_root}/artifacts/.

    GIVEN: Valid config with minimal grid specification
    WHEN: run_experiment_grid executes
    THEN: Grid summary contains paths to all aggregate artifacts.
    """
    # (1) 构造最小配置（避免实际嵌入/检测，仅验证路径写入）
    minimal_cfg = {
        "output_root": str(tmp_path / "output"),
        "run_name": "test_grid",
        "watermark_embed": {"num_seeds": 1},
        "watermark_detect": {"detection_threshold": 0.5},
        "grid": {
            "embed_grid": {
                "num_seeds": [1],
            },
            "detect_grid": {},
        },
    }
    
    # (2) 构建网格（不调用 normalize_ablation_flags，build_experiment_grid 会内部处理）
    grid = experiment_matrix.build_experiment_grid(minimal_cfg)
    
    # (3) 执行网格（strict=False 允许子任务失败，仅验证写入行为）
    grid_summary = experiment_matrix.run_experiment_grid(grid, strict=False)
    
    # (4) 验证摘要字段存在
    assert "batch_root" in grid_summary, "grid_summary 必须包含 batch_root"
    assert "aggregate_report_path" in grid_summary, "grid_summary 必须包含 aggregate_report_path"
    assert "grid_summary_path" in grid_summary, "grid_summary 必须包含 grid_summary_path"
    assert "grid_manifest_path" in grid_summary, "grid_summary 必须包含 grid_manifest_path"
    assert "attack_coverage_manifest_path" in grid_summary, "grid_summary 必须包含 attack_coverage_manifest_path"
    assert "hf_template_comparison_table_path" in grid_summary, "grid_summary 必须包含 hf_template_comparison_table_path"
    assert "hf_template_comparison_table_csv_path" in grid_summary, "grid_summary 必须包含 hf_template_comparison_table_csv_path"
    
    # (5) 验证路径结构：所有工件路径必须在 batch_root/artifacts 下
    batch_root = Path(grid_summary["batch_root"])
    artifacts_dir = batch_root / "artifacts"
    
    aggregate_report_path = Path(grid_summary["aggregate_report_path"])
    grid_summary_path = Path(grid_summary["grid_summary_path"])
    grid_manifest_path = Path(grid_summary["grid_manifest_path"])
    attack_coverage_path = Path(grid_summary["attack_coverage_manifest_path"])
    comparison_table_path = Path(grid_summary["hf_template_comparison_table_path"])
    comparison_csv_path = Path(grid_summary["hf_template_comparison_table_csv_path"])
    
    # 所有路径都应该在 artifacts_dir 下
    assert aggregate_report_path.parent == artifacts_dir, \
        f"aggregate_report 必须在 {artifacts_dir} 下，实际为: {aggregate_report_path.parent}"
    assert grid_summary_path.parent == artifacts_dir, \
        f"grid_summary 必须在 {artifacts_dir} 下，实际为: {grid_summary_path.parent}"
    assert grid_manifest_path.parent == artifacts_dir, \
        f"grid_manifest 必须在 {artifacts_dir} 下，实际为: {grid_manifest_path.parent}"
    assert attack_coverage_path.parent == artifacts_dir, \
        f"attack_coverage_manifest 必须在 {artifacts_dir} 下，实际为: {attack_coverage_path.parent}"
    assert comparison_table_path.parent == artifacts_dir, \
        f"hf_template_comparison_table 必须在 {artifacts_dir} 下，实际为: {comparison_table_path.parent}"
    assert comparison_csv_path.parent == artifacts_dir, \
        f"hf_template_comparison_table_csv 必须在 {artifacts_dir} 下，实际为: {comparison_csv_path.parent}"
    
    # (6) 验证工件文件实际存在（即使内容可能为空或错误，文件必须被写入）
    # 注意：此测试不关心内容正确性，仅关心路径策略和写入行为
    assert aggregate_report_path.exists(), f"aggregate_report 文件必须存在: {aggregate_report_path}"
    assert grid_summary_path.exists(), f"grid_summary 文件必须存在: {grid_summary_path}"
    assert grid_manifest_path.exists(), f"grid_manifest 文件必须存在: {grid_manifest_path}"
    assert attack_coverage_path.exists(), f"attack_coverage_manifest 文件必须存在: {attack_coverage_path}"
    assert comparison_table_path.exists(), f"hf_template_comparison_table 文件必须存在: {comparison_table_path}"
    assert comparison_csv_path.exists(), f"hf_template_comparison_table_csv 文件必须存在: {comparison_csv_path}"
    
    # (7) 验证 JSON 格式可解析（基本健康性检查）
    with aggregate_report_path.open("r", encoding="utf-8") as f:
        aggregate_data = json.load(f)
    with grid_summary_path.open("r", encoding="utf-8") as f:
        summary_data = json.load(f)
    with grid_manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    with attack_coverage_path.open("r", encoding="utf-8") as f:
        coverage_data = json.load(f)
    with comparison_table_path.open("r", encoding="utf-8") as f:
        comparison_data = json.load(f)

    comparison_csv_content = comparison_csv_path.read_text(encoding="utf-8")
    assert "grid_item_digest" in comparison_csv_content
    
    # 清理
    if batch_root.exists():
        shutil.rmtree(batch_root, ignore_errors=True)
