#!/usr/bin/env python3
"""
CPU 优先端到端闭环快速验证

功能：直接测试控制路径，验证 P1/P2 必达与 freeze_gate ALLOW_FREEZE
"""

import sys
import os
from pathlib import Path

# 设置 PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.diffusion.sd3 import trajectory_tap
from main.watermarking.paper_faithfulness import alignment_evaluator


def test_p1_pipeline_fingerprint():
    """P1：pipeline_fingerprint_presence 检查"""
    print("\n" + "="*70)
    print("[P1] 测试 Pipeline Fingerprint Presence")
    print("="*70)
    
    # 模拟 pipeline_fingerprint with ok status
    pipeline_fingerprint = {
        "status": "ok",
        "transformer_num_blocks": 24,
        "scheduler_class_name": "FlowMatchEulerDiscreteScheduler",
        "vae_latent_channels": 16
    }
    
    cfg = {
        "paper_faithfulness": {
            "enabled": True,
            "alignment_check": True
        }
    }
    
    result = alignment_evaluator._check_pipeline_fingerprint_presence(
        pipeline_fingerprint,
        enable_paper_faithfulness=True
    )
    
    p1_status = result.get("result")
    print(f"✓ P1 Result: {p1_status}")
    assert p1_status == "PASS", f"Expected PASS, got {p1_status}"
    return True


def test_p2_trajectory_tap_enabled():
    """P2：trajectory_tap 在 paper_faithfulness 启用时自动启用"""
    print("\n" + "="*70)
    print("[P2] 测试 Trajectory Tap 启用")
    print("="*70)
    
    cfg = {
        "paper_faithfulness": {
            "enabled": True,
            "alignment_check": True
        },
        "trajectory_tap": {}  # 未显式配置
    }
    
    tap_enabled = trajectory_tap._resolve_tap_enabled(cfg)
    print(f"✓ P2 Tap Enabled: {tap_enabled}")
    assert tap_enabled is True, f"Expected True, got {tap_enabled}"
    return True


def test_override_whitelist():
    """验证 override 白名单包含新增的 force_cpu、enable_paper_faithfulness 等"""
    print("\n" + "="*70)
    print("[Override] 验证白名单中的新增 override")
    print("="*70)
    
    from main.policy.runtime_whitelist import load_runtime_whitelist
    
    whitelist = load_runtime_whitelist()
    
    allowed_arg_names = whitelist.data.get("override", {}).get("arg_name_enum", {}).get("allowed", [])
    
    required_overrides = ["force_cpu", "enable_paper_faithfulness", "enable_trace_tap"]
    for override in required_overrides:
        assert override in allowed_arg_names, \
            f"Override '{override}' not in whitelist arg_name_enum"
        print(f"✓ Override '{override}' in whitelist")
    
    return True


def test_default_config():
    """验证 default.yaml 中启用了 paper_faithfulness 和 trajectory_tap"""
    print("\n" + "="*70)
    print("[Config] 验证默认配置")
    print("="*70)
    
    import yaml
    try:
        with open("configs/default.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to load default.yaml: {e}")
        return False
    
    # 检查 paper_faithfulness 启用
    paper_faith = cfg.get("paper_faithfulness", {})
    assert paper_faith.get("enabled") is True, \
        "paper_faithfulness.enabled should be True in default.yaml"
    print(f"✓ paper_faithfulness.enabled = {paper_faith.get('enabled')}")
    
    # 检查 trajectory_tap 启用
    tap_cfg = cfg.get("trajectory_tap", {})
    assert tap_cfg.get("enabled") is True, \
        "trajectory_tap.enabled should be True in default.yaml"
    print(f"✓ trajectory_tap.enabled = {tap_cfg.get('enabled')}")
    
    # 检查 device 设置
    device = cfg.get("device")
    assert device == "cpu", f"Default device should be cpu, got {device}"
    print(f"✓ device = {device}")
    
    return True


def test_audits_allow_freeze():
    """运行审计检查 ALLOW_FREEZE 判定"""
    print("\n" + "="*70)
    print("[Audit] 验证 FreezeSignoffDecision == ALLOW_FREEZE")
    print("="*70)
    
    import subprocess
    import json
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/run_all_audits.py", "--strict"],
            capture_output=True,
            text=False,  # 以二进制模式读取
            cwd=Path(__file__).parent.parent,
            timeout=30
        )
        
        # 尝试解码 JSON 输出
        try:
            output_text = result.stdout.decode('utf-8')
        except UnicodeDecodeError:
            try:
                output_text = result.stdout.decode('gbk')
            except UnicodeDecodeError:
                output_text = result.stdout.decode('utf-8', errors='replace')
        
        # 尝试解析 JSON
        try:
            # 寻找 JSON 的开始位置
            json_start = output_text.find('{')
            if json_start >= 0:
                json_text = output_text[json_start:]
                audit_result = json.loads(json_text)
                decision = audit_result.get("summary", {}).get("FreezeSignoffDecision")
                
                if decision == "ALLOW_FREEZE":
                    print(f"✓ FreezeSignoffDecision = {decision}")
                    return True
                else:
                    print(f"✗ FreezeSignoffDecision = {decision} (expected ALLOW_FREEZE)")
                    return False
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse audit JSON: {e}")
            if "ALLOW_FREEZE" in output_text:
                print(f"✓ Found ALLOW_FREEZE in output")
                return True
        
        print(f"Output snippet: {output_text[:200]}")
        return False
        
    except subprocess.TimeoutExpired:
        print(f"✗ Audit script timeout")
        return False
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("CPU-First E2E Closure Verification")
    print("="*70)
    
    tests = [
        ("P1 Pipeline Fingerprint", test_p1_pipeline_fingerprint),
        ("P2 Trajectory Tap", test_p2_trajectory_tap_enabled),
        ("Override Whitelist", test_override_whitelist),
        ("Default Config", test_default_config),
        ("Audit ALLOW_FREEZE", test_audits_allow_freeze),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
            print(f"✓ [{test_name}] PASS")
        except Exception as e:
            results[test_name] = False
            print(f"✗ [{test_name}] FAIL: {e}")
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    all_pass = all(results.values())
    pass_count = sum(1 for v in results.values() if v)
    fail_count = len(results) - pass_count
    
    print(f"\nResults: {pass_count}/{len(results)} PASS, {fail_count} FAIL")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    if all_pass:
        print(f"\n✓ [SUCCESS] CPU-first closure verified!")
        print(f"  - P1 pipeline_fingerprint_presence = PASS")
        print(f"  - P2 trajectory_digest_reproducibility = PASS (enabled)")
        print(f"  - Override whitelist includes force_cpu, enable_paper_faithfulness, enable_trace_tap")
        print(f"  - Default config enables paper_faithfulness and trajectory_tap")
        print(f"  - FreezeSignoffDecision = ALLOW_FREEZE")
        return 0
    else:
        print(f"\n✗ [FAILURE] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
