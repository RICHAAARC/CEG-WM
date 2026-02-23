"""
文件目的：发布级工作流编排脚本
Module type: General module

发布级完整工作流，从嵌入到冻结签署：
  1. embed: 生成水印嵌入计划与指标
  2. detect: 生成并执行检测方案
  3. calibrate: 校准阈值与融合规则（仅当 attack_seed 已知时）
  4. evaluate: 以只读阈值进行完整性能评测，输出 evaluation_report.json
  5. table_export: 导出评测指标为 CSV
  6. signoff: 执行最小审计集并生成冻结签署决策

本脚本是 run_repro_pipeline.py 的高层包装，增加对 signoff 的调用。
signoff 要求输入必须包含 evaluation_report.json（来自完整工作流）。

smoke_detect（快速冒烟测试） 仅运行 embed/detect，不运行 evaluate，
因此不适用于 signoff 输入。
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> int:
    """
    功能：运行子进程并返回退出码。
    
    Run a subprocess and return exit code.
    
    Args:
        cmd: Command as list of strings.
        description: Human-readable description of the step.
    
    Returns:
        Exit code from subprocess.
    """
    print(f"\n{'='*70}")
    print(f"[Publish Workflow] {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with exit code {result.returncode}: {description}")
        return result.returncode
    
    print(f"\n[OK] Step completed: {description}")
    return 0


def main() -> None:
    """
    功能：发布级工作流主入口。
    
    Main entry for publish-grade workflow.
    """
    parser = argparse.ArgumentParser(
        description="发布级工作流（embed → detect → calibrate → evaluate → signoff）"
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="运行输出根目录",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="仓库根目录",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="experiment matrix 与 repro 流程共享的配置文件路径",
    )
    parser.add_argument(
        "--skip-repro",
        action="store_true",
        help="跳过 run_repro_pipeline（若工作流已完成），仅运行 signoff",
    )
    parser.add_argument(
        "--signoff-profile",
        type=str,
        default="paper",
        choices=["baseline", "paper", "publish"],
        help="Signoff 审计集合 profile（baseline: 最小集合; paper/publish: 论文复现级，默认: paper）",
    )
    
    args = parser.parse_args()
    
    run_root = args.run_root.resolve()
    repo_root = args.repo_root.resolve()
    scripts_dir = repo_root / "scripts"
    signoff_profile = args.signoff_profile
    config_path = args.config.resolve()
    
    if not repo_root.is_dir():
        print(f"错误：repo_root 不存在: {repo_root}", file=sys.stderr)
        sys.exit(2)
    
    print(f"\n{'='*70}")
    print(" 发布级工作流（Publish-grade Workflow）")
    print(f"{'='*70}")
    print(f"Run Root: {run_root}")
    print(f"Repo Root: {repo_root}")
    print(f"Config: {config_path}")
    print(f"Signoff Profile: {signoff_profile}")
    
    # 第一阶段：embed → detect → calibrate → evaluate（可选跳过）
    if not args.skip_repro:
        print(f"\n[Publish Workflow] 执行再现流程 (embed → detect → calibrate → evaluate)...")
        result = run_command(
            [sys.executable, str(scripts_dir / "run_repro_pipeline.py"), 
             "--run-root", str(run_root), "--repo-root", str(repo_root)],
            "Run Repro Pipeline (embed → detect → calibrate → evaluate)"
        )
        if result != 0:
            print(f"\n[FATAL] Repro pipeline failed. Cannot proceed to signoff.")
            sys.exit(1)
    else:
        print(f"\n[Publish Workflow] 跳过 repro 流程（--skip-repro 指定）")

    # 第二阶段：experiment matrix（发布级门禁工件，必须在同一 run_root 下产出）
    matrix_batch_root = run_root / "outputs" / "experiment_matrix"
    print(f"\n[Publish Workflow] 执行 experiment matrix（batch_root={matrix_batch_root}）...")
    result = run_command(
        [
            sys.executable,
            str(scripts_dir / "run_experiment_matrix.py"),
            "--config",
            str(config_path),
            "--batch-root",
            str(matrix_batch_root),
        ],
        "Run Experiment Matrix (publish gate artifacts)"
    )
    if result != 0:
        print(f"\n[FATAL] Experiment matrix failed. Cannot proceed to signoff.")
        sys.exit(1)
    
    # 第三阶段：冻结签署（强制要求 evaluation_report.json 与 matrix 工件）
    print(f"\n[Publish Workflow] 执行冻结签署...")
    result = run_command(
        [sys.executable, str(scripts_dir / "run_freeze_signoff.py"),
         "--run-root", str(run_root), "--repo-root", str(repo_root),
         "--signoff-profile", signoff_profile,
         "--require-experiment-matrix"],
        f"Freeze Signoff (profile={signoff_profile}, 基于完整工作流输出)"
    )
    
    if result == 0:
        print(f"\n{'='*70}")
        print(" ✅ 发布级工作流完成 (ALLOW_FREEZE or further investigation required)")
        print(f"{'='*70}")
        signoff_report = run_root / "artifacts" / "signoff" / "signoff_report.json"
        print(f"Signoff Report: {signoff_report}")
        sys.exit(0)
    else:
        print(f"\n{'='*70}")
        print(" ❌ 冻结签署失败 (BLOCK_FREEZE)")
        print(f"{'='*70}")
        signoff_report = run_root / "artifacts" / "signoff" / "signoff_report.json"
        print(f"Signoff Report: {signoff_report}")
        sys.exit(1)


if __name__ == "__main__":
    main()
