#!/usr/bin/env python
"""
快速验证脚本：验证 P1/P2 alignment checks 是否通过

使用方法：
    python scripts/verify_p1_p2_alignment.py [--config CONFIG_PATH] [--output OUTPUT_DIR]

示例：
    # 使用默认配置（预期 NA）
    python scripts/verify_p1_p2_alignment.py --config configs/default.yaml --output output/verify_default
    
    # 使用 paper_faithful 配置（预期 PASS）
    python scripts/verify_p1_p2_alignment.py --config configs/paper_faithful_runtime.yaml --output output/verify_pf

"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_embed_record(output_dir: str) -> Dict[str, Any] | None:
    """
    功能：加载 embed_record.json 文件。

    Load embed record from output directory.

    Args:
        output_dir: Output directory path.

    Returns:
        Parsed record dict or None if not found.
    """
    record_path = Path(output_dir) / "embed_record.json"
    if not record_path.exists():
        print(f"❌ 找不到 embed_record.json: {record_path}")
        return None
    
    try:
        with open(record_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载 embed_record.json 失败: {e}")
        return None


def verify_alignment_report(record: Dict[str, Any]) -> bool:
    """
    功能：检查 alignment_report 中 P1/P2 checks 的结果。

    Verify P1/P2 checks in alignment_report.

    Args:
        record: Embed record dict.

    Returns:
        True if both P1 and P2 are PASS, False otherwise.
    """
    try:
        alignment_report = record.get("content_evidence", {}).get("alignment_report", {})
        
        if not isinstance(alignment_report, dict):
            print("❌ alignment_report 为空或无效")
            return False
        
        checks = alignment_report.get("checks", [])
        if not isinstance(checks, list):
            print("❌ checks 字段无效")
            return False
        
        check_results = {
            c.get("check_name"): c.get("result") 
            for c in checks
        }
        
        # P1: Pipeline Fingerprint Presence
        p1_result = check_results.get("pipeline_fingerprint_presence")
        # P2: Trajectory Digest Reproducibility
        p2_result = check_results.get("trajectory_digest_reproducibility")
        
        print("\n=== 对齐检查结果 ===")
        print(f"✅ P1 (Pipeline Fingerprint Presence): {p1_result}" if p1_result == "PASS" else f"⚠️ P1 (Pipeline Fingerprint Presence): {p1_result}")
        print(f"✅ P2 (Trajectory Digest Reproducibility): {p2_result}" if p2_result == "PASS" else f"⚠️ P2 (Trajectory Digest Reproducibility): {p2_result}")
        
        # 详细信息
        print("\n=== 完整检查清单 ===")
        for check in checks:
            check_name = check.get("check_name", "?")
            result = check.get("result", "?")
            status_icon = "✅" if result == "PASS" else "⚠️"
            print(f"{status_icon} {check_name}: {result}")
            if result != "PASS":
                reason = check.get("na_reason") or check.get("failure_message") or ""
                if reason:
                    print(f"   原因: {reason}")
        
        overall_status = alignment_report.get("overall_status")
        print(f"\n=== 总体状态 ===")
        print(f"overall_status: {overall_status}")
        
        return p1_result == "PASS" and p2_result == "PASS"
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="验证 P1/P2 alignment checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 验证默认配置（预期 NA）
  python scripts/verify_p1_p2_alignment.py --config configs/default.yaml --output output/test_default
  
  # 验证 paper_faithful 配置（预期 PASS）
  python scripts/verify_p1_p2_alignment.py --config configs/paper_faithful_runtime.yaml --output output/test_pf
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paper_faithful_runtime.yaml",
        help="配置文件路径（默认: configs/paper_faithful_runtime.yaml）"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="输出目录（如果不指定，将运行 embed 流程）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）"
    )
    
    args = parser.parse_args()
    
    # 如果未指定 output 目录，运行 embed 并等待完成
    if not args.output:
        import tempfile
        import subprocess
        from datetime import datetime
        
        temp_dir = tempfile.mkdtemp(prefix="verify_p1_p2_")
        print(f"[INFO] 未指定输出目录，运行 embed 流程: {temp_dir}")
        
        cmd = [
            sys.executable, "-m", "main.cli.run_embed",
            "--output", temp_dir,
            "--config", args.config,
            "--override", f"seed={args.seed}"
        ]
        
        print(f"[INFO] 执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=Path.cwd())
        
        if result.returncode != 0:
            print(f"❌ embed 流程失败 (exit code: {result.returncode})")
            return 1
        
        args.output = temp_dir
    
    # 加载并检查结果
    record = load_embed_record(args.output)
    if record is None:
        return 1
    
    passed = verify_alignment_report(record)
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
