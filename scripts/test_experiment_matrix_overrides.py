#!/usr/bin/env python3
"""
测试 override 系统是否能正确处理实验矩阵的参数

功能说明：
- 模拟 experiment_matrix 生成的 override 参数
- 验证 override_rules 能否正确解析和应用这些参数
"""

import sys
import json
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main.policy.runtime_whitelist import load_runtime_whitelist
from main.policy import override_rules


def test_experiment_matrix_overrides():
    """测试 experiment_matrix 的 override 参数"""
    
    # 加载 whitelist（必须使用权威相对路径，cwd 由 pytest 固定在 repo root）
    whitelist = load_runtime_whitelist()
    
    print("=" * 80)
    print("测试 experiment_matrix 的 override 参数")
    print("=" * 80)
    
    # 模拟 experiment_matrix 生成的 override
    overrides = [
        "allow_nonempty_run_root=true",
        'allow_nonempty_run_root_reason="experiment_grid"',
        "seed=0",
        'model_id="stabilityai/stable-diffusion-3.5-medium"',
        'calibrate_detect_records_glob="/path/to/detect_record.json"',
        'evaluate_detect_records_glob="/path/to/detect_record.json"',
        'evaluate_thresholds_path="/path/to/thresholds.json"',
    ]
    
    print(f"\n生成的 override 参数：")
    for override in overrides:
        print(f"  {override}")
    
    print(f"\n解析 override...")
    try:
        # 测试解析是否成功
        parsed = []
        for override in overrides:
            try:
                result = override_rules._parse_override_arg(override)
                parsed.append(result)
                print(f"  ✓ {override}")
            except Exception as e:
                print(f"  ✗ {override}")
                print(f"    错误: {e}")
                return False
        
        print(f"\n验证 override 是否被 whitelist 允许...")
        try:
            resolved = override_rules._resolve_overrides(parsed, whitelist)
            print(f"  ✓ 所有 override 都被允许")
            
            for res in resolved:
                arg_name = res.resolved_entry.get("arg_name")
                field_path = res.resolved_entry.get("field_path")
                print(f"    {arg_name:40} → {field_path}")
            
            return True
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            return False
    
    except Exception as e:
        print(f"✗ 未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_experiment_matrix_overrides()
    
    print("\n" + "=" * 80)
    if success:
        print("✓ 所有测试通过！")
        sys.exit(0)
    else:
        print("✗ 测试失败！")
        sys.exit(1)
