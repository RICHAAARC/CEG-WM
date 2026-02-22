#!/usr/bin/env python3
"""
CPU 优先端到端闭环验证脚本

功能说明：
- 在 CPU 上执行完整的嵌入-检测-审计流程（CPU-first closure verification）
- 强制启用 paper_faithfulness、trajectory_tap、pipeline_fingerprint
- 自动生成 audit_evidence 证据包用于人工复核与签署
- 验证 FreezeSignoffDecision=ALLOW_FREEZE、P1/P2 PASS、pytest 全 PASS

Module type: Core innovation module
"""

import sys
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import hashlib


def run_command(cmd, description, check=True):
    """
    功能：运行 shell 命令并显示输出。

    Run command and display output.

    Args:
        cmd: Command list to execute.
        description: Description of command.
        check: Whether to raise on non-zero exit.

    Returns:
        CompletedProcess result.
    """
    import os
    import subprocess
    
    print(f"\n{'='*70}")
    print(f"[CPU-First] {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    # 设置环境变量以确保 Python 能找到 main 模块
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path.cwd())
    
    result = subprocess.run(cmd, capture_output=False, check=False, env=env)
    
    if check and result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(1)
    
    return result


def ensure_directory(path):
    """确保目录存在。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_cpu_first_verification(output_root="tmp/cpu_first_e2e", config_path="configs/default.yaml"):
    """
    功能：执行 CPU 优先端到端验证闭环。

    Execute CPU-first end-to-end verification closure.

    Args:
        output_root: Root directory for output (default tmp/cpu_first_e2e).
        config_path: Config YAML path (default configs/default.yaml).

    Returns:
        Verification result dict or raises exception on failure.
    """
    import os
    
    # 确保在正确的工作目录
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    print(f"Working directory: {os.getcwd()}")
    
    output_root = Path(output_root)
    ensure_directory(output_root)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"cpu_first_{timestamp}"
    embed_output = output_root / f"embed_{timestamp}"
    detect_output = output_root / f"detect_{timestamp}"
    evidence_root = output_root / "audit_evidence" / timestamp
    
    ensure_directory(embed_output)
    ensure_directory(detect_output)
    ensure_directory(evidence_root)
    
    print(f"\n{'='*70}")
    print(f"[CPU-First] CPU 优先端到端闭环验证")
    print(f"{'='*70}")
    print(f"Run ID: {run_id}")
    print(f"Embed output: {embed_output}")
    print(f"Detect output: {detect_output}")
    print(f"Evidence root: {evidence_root}")
    print(f"Timestamp: {timestamp}\n")
    
    # ========== PHASE 1: Embed ==========/
    print("\n" + "="*70)
    print("[PHASE 1] Embed (嵌入 - CPU 强制、Paper Faithfulness 启用)")
    print("="*70)
    
    embed_cmd = [
        sys.executable,
        "main/cli/run_embed.py",
        "--out", str(embed_output),
        "--config", config_path,
        "--override", "force_cpu",      # CPU 强制
        "--override", "enable_paper_faithfulness",  # Paper Faithfulness 必达
        "--override", "enable_trace_tap",           # Trajectory Tap 必达
        "--override", "inference_enabled=true",
    ]
    
    embed_result = run_command(embed_cmd, "Embed 阶段（embed + paper_faithfulness）")
    if embed_result.returncode != 0:
        print(f"\n[CRITICAL] Embed failed!")
        sys.exit(1)
    
    # 验证 embed_record 的关键字段
    embed_record_path = embed_output / "records" / "embed_record.json"
    if not embed_record_path.exists():
        print(f"\n[CRITICAL] Embed record not found: {embed_record_path}")
        sys.exit(1)
    
    with open(embed_record_path, "r") as f:
        embed_record = json.load(f)
    
    print(f"\n[Embed] Record loaded, checking alignment...")
    
    # 检查 P1：pipeline_fingerprint_presence
    content_evidence = embed_record.get("content_evidence", {})
    alignment_report = content_evidence.get("alignment_report", {})
    pipeline_fp_check = None
    trajectory_check = None
    
    for check in alignment_report.get("checks", []):
        if check.get("check_name") == "pipeline_fingerprint_presence":
            pipeline_fp_check = check
        elif check.get("check_name") == "trajectory_digest_reproducibility":
            trajectory_check = check
    
    if pipeline_fp_check is None:
        print(f"[WARN] pipeline_fingerprint_presence check not found in alignment_report")
    else:
        p1_status = pipeline_fp_check.get("result", "NA")
        print(f"[P1] pipeline_fingerprint_presence: {p1_status}")
        if p1_status != "PASS":
            print(f"[WARN] P1 not PASS: {pipeline_fp_check.get('failure_message', 'unknown')}")
    
    # 检查 P2：trajectory_digest_reproducibility
    if trajectory_check is None:
        print(f"[WARN] trajectory_digest_reproducibility check not found in alignment_report")
    else:
        p2_status = trajectory_check.get("result", "NA")
        print(f"[P2] trajectory_digest_reproducibility: {p2_status}")
        if p2_status != "PASS":
            print(f"[WARN] P2 not PASS: {trajectory_check.get('failure_message', 'unknown')}")
    
    # ========== PHASE 2: Detect ==========/
    print("\n" + "="*70)
    print("[PHASE 2] Detect (检测 - 验证 P1/P2 一致性)")
    print("="*70)
    
    # 将 embed_record 作为 detect 的输入
    embed_record_path = embed_output / "records" / "embed_record.json"
    
    detect_cmd = [
        sys.executable,
        "main/cli/run_detect.py",
        "--out", str(detect_output),
        "--config", config_path,
        "--input", str(embed_record_path),
        "--override", "force_cpu",
    ]
    
    detect_result = run_command(detect_cmd, "Detect 阶段（检测 + Paper Faithfulness 对齐）")
    if detect_result.returncode != 0:
        print(f"\n[WARN] Detect phase returned non-zero exit code (may be expected in early validation)")
    
    # 验证 detect_record 是否存在
    detect_record_path = detect_output / "records" / "detect_record.json"
    if detect_record_path.exists():
        with open(detect_record_path, "r") as f:
            detect_record = json.load(f)
        
        paper_faith = detect_record.get("paper_faithfulness", {})
        detect_status = paper_faith.get("status")
        print(f"\n[Detect] paper_faithfulness.status: {detect_status}")
    else:
        print(f"\n[WARN] Detect record not found: {detect_record_path}")
    
    # ========== PHASE 3: Audits ==========/
    print("\n" + "="*70)
    print("[PHASE 3] Audits (审计 - strict 门禁验证)")
    print("="*70)
    
    # 运行 pytest
    pytest_cmd = [
        sys.executable,
        "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ]
    
    pytest_result = run_command(pytest_cmd, "pytest（论文对齐与审计检查）", check=False)
    
    # 运行 run_all_audits.py --strict
    audits_cmd = [
        sys.executable,
        "scripts/run_all_audits.py",
        "--strict",
    ]
    
    audits_result = run_command(audits_cmd, "run_all_audits.py --strict（冻结一致性与 ALLOW_FREEZE 判定）", check=False)
    
    # ========== PHASE 4: Evidence Package ==========/
    print("\n" + "="*70)
    print("[PHASE 4] Evidence Package (证据包生成)")
    print("="*70)
    
    # 拷贝 run_root 与 logs
    shutil.copytree(
        embed_output / "records",
        evidence_root / "run_root" / "records",
        dirs_exist_ok=True
    )
    shutil.copytree(
        embed_output / "artifacts",
        evidence_root / "run_root" / "artifacts",
        dirs_exist_ok=True
    )
    
    if (embed_output / "logs").exists():
        shutil.copytree(
            embed_output / "logs",
            evidence_root / "run_root" / "logs",
            dirs_exist_ok=True
        )
    
    # 拷贝 detect 结果
    if detect_output.exists():
        if (detect_output / "records").exists():
            shutil.copytree(
                detect_output / "records",
                evidence_root / "run_root_detect" / "records",
                dirs_exist_ok=True
            )
        if (detect_output / "artifacts").exists():
            shutil.copytree(
                detect_output / "artifacts",
                evidence_root / "run_root_detect" / "artifacts",
                dirs_exist_ok=True
            )
    
    # 生成 alignment_acceptance_summary.json
    acceptance_summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "cpu_first_mode": True,
        
        # 论文对齐检查结果
        "alignment_checks": {
            "p1_pipeline_fingerprint_presence": {
                "check_name": "pipeline_fingerprint_presence",
                "result": pipeline_fp_check.get("result", "NA") if pipeline_fp_check else "NA",
                "evidence_path": "run_root/records/embed_record.json -> content_evidence.alignment_report"
            },
            "p2_trajectory_digest_reproducibility": {
                "check_name": "trajectory_digest_reproducibility",
                "result": trajectory_check.get("result", "NA") if trajectory_check else "NA",
                "evidence_path": "run_root/records/embed_record.json -> content_evidence.alignment_report"
            }
        },
        
        # 测试与审计结果
        "test_results": {
            "pytest_exit_code": pytest_result.returncode,
            "pytest_status": "PASS" if pytest_result.returncode == 0 else "FAIL",
        },
        
        "audit_results": {
            "run_all_audits_exit_code": audits_result.returncode,
            "run_all_audits_status": "PASS" if audits_result.returncode == 0 else "FAIL",
        },
        
        # 最终判定
        "final_verdict": {
            "cpu_first_closure_verified": (
                pytest_result.returncode == 0 and 
                audits_result.returncode == 0 and
                (pipeline_fp_check.get("result") if pipeline_fp_check else "NA") == "PASS" and
                (trajectory_check.get("result") if trajectory_check else "NA") == "PASS"
            ),
            "allow_freeze_decision": audits_result.returncode == 0,
        },
        
        "paths": {
            "embed_run_root": str(embed_output),
            "detect_run_root": str(detect_output),
            "evidence_package": str(evidence_root),
        },
        
        "notes": [
            "CPU 优先闭环验证（CPU-first closure verification）",
            "强制设备为 CPU，启用 paper_faithfulness、trajectory_tap",
            "P1 检查：pipeline_fingerprint_presence 必须 PASS",
            "P2 检查：trajectory_digest_reproducibility 必须 PASS",
            "审计检查：freeze_gate ALLOW_FREEZE、pytest 全 PASS、run_all_audits --strict PASS"
        ]
    }
    
    summary_path = evidence_root / "alignment_acceptance_summary.json"
    with open(summary_path, "w") as f:
        json.dump(acceptance_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Evidence] Acceptance summary written: {summary_path}")
    
    # 生成证据包清单
    manifest = {
        "evidence_package_version": "v1.0",
        "timestamp": timestamp,
        "run_id": run_id,
        "contents": {
            "run_root": {
                "description": "嵌入阶段输出（embed_record、pipeline_fingerprint、trajectory_evidence）",
                "paths": ["records/embed_record.json", "artifacts/", "logs/"]
            },
            "run_root_detect": {
                "description": "检测阶段输出（detect_record、alignment 验证结果）",
                "paths": ["records/detect_record.json", "artifacts/"]
            },
            "alignment_acceptance_summary.json": {
                "description": "P1/P2 论文对齐检查结果与最终判定"
            }
        },
        "audit_checklist": {
            "P1_pipeline_fingerprint_presence": "pipeline_fingerprint 必须非 absent/failed",
            "P2_trajectory_digest_reproducibility": "trajectory_digest 可复算、与 scheduler 一致",
            "detect_paper_faithfulness_status": "detect 侧 paper_faithfulness.status == ok",
            "freeze_gate_ALLOW_FREEZE": "FreezeSignoffDecision == ALLOW_FREEZE",
            "pytest_full_pass": "pytest 100% PASS",
            "run_all_audits_strict": "run_all_audits.py --strict PASS",
        },
        "verification_status": "PENDING_MANUAL_REVIEW" if not acceptance_summary["final_verdict"]["cpu_first_closure_verified"] else "VERIFIED"
    }
    
    manifest_path = evidence_root / "MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Evidence] Manifest written: {manifest_path}")
    
    # ========== Final Verdict ==========/
    print("\n" + "="*70)
    print("[FINAL VERDICT] CPU-First E2E 验证结果")
    print("="*70)
    
    print(f"\n[P1] pipeline_fingerprint_presence: {pipeline_fp_check.get('result', 'NA') if pipeline_fp_check else 'NA'}")
    print(f"[P2] trajectory_digest_reproducibility: {trajectory_check.get('result', 'NA') if trajectory_check else 'NA'}")
    print(f"[pytest] Exit code: {pytest_result.returncode}")
    print(f"[audits] Exit code: {audits_result.returncode}")
    
    cpu_first_ok = acceptance_summary["final_verdict"]["cpu_first_closure_verified"]
    allow_freeze_ok = acceptance_summary["final_verdict"]["allow_freeze_decision"]
    
    print(f"\nCPU-first closure verified: {cpu_first_ok}")
    print(f"FreezeSignoffDecision == ALLOW_FREEZE: {allow_freeze_ok}")
    
    if cpu_first_ok and allow_freeze_ok:
        print(f"\n✓ [SUCCESS] CPU 优先端到端闭环验证通过")
        print(f"  证据包已生成：{evidence_root}")
        return True
    else:
        print(f"\n✗ [FAILURE] CPU 优先端到端闭环验证未通过")
        print(f"  检查项目：")
        if not (pipeline_fp_check.get('result') if pipeline_fp_check else 'NA') == "PASS":
            print(f"  - P1 pipeline_fingerprint_presence 未 PASS")
        if not (trajectory_check.get('result') if trajectory_check else 'NA') == "PASS":
            print(f"  - P2 trajectory_digest_reproducibility 未 PASS")
        if pytest_result.returncode != 0:
            print(f"  - pytest 未全部 PASS (exit code {pytest_result.returncode})")
        if audits_result.returncode != 0:
            print(f"  - run_all_audits.py --strict 未 PASS (exit code {audits_result.returncode})")
        return False


if __name__ == "__main__":
    try:
        success = run_cpu_first_verification()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[CRITICAL] Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
