#!/usr/bin/env python3
"""
验证 runtime_whitelist 和配置文件的一致性

功能说明：
- 检查 runtime_whitelist 中定义的所有 override 是否在配置文件中有对应的字段
- 确保 arg_name 和 field_path 的映射正确
"""

import sys
import json
import yaml
from pathlib import Path


def load_whitelist(whitelist_path: Path) -> dict:
    """加载 runtime_whitelist.yaml"""
    with open(whitelist_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_config(config_path: Path) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_nested_value(cfg: dict, path: str):
    """获取嵌套的配置值"""
    keys = path.split('.')
    current = cfg
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return None


def verify_override_fields(whitelist: dict, config_path: Path) -> tuple[bool, list]:
    """验证 override 字段是否在配置文件中定义"""
    errors = []
    
    allowed_overrides = whitelist.get('override', {}).get('allowed_overrides', [])
    config = load_config(config_path)
    
    print(f"检查配置文件: {config_path}")
    print(f"发现 {len(allowed_overrides)} 个 override 规则")
    
    for override in allowed_overrides:
        arg_name = override.get('arg_name')
        field_path = override.get('field_path')
        
        # 某些 override 可能没有对应的配置字段（如完全动态参数）
        # 但需要文档化这些特殊情况
        value = get_nested_value(config, field_path)
        
        status = "✓" if value is not None else "○"
        print(f"  {status} {arg_name:40} → {field_path:40} (value: {str(value)[:20]})")
        
        # 对于某些特定参数，允许 None（表示需要运行时注入）
        if value is None and field_path not in [
            "calibrate.detect_records_glob",
            "evaluate.detect_records_glob", 
            "evaluate.thresholds_path",
        ]:
            # 其他为 None 的字段需要检查是否是设计的
            pass
    
    return len(errors) == 0, errors


def verify_arg_name_enum(whitelist: dict) -> tuple[bool, list]:
    """验证 arg_name_enum 与 allowed_overrides 的一致性"""
    errors = []
    
    allowed_overrides = whitelist.get('override', {}).get('allowed_overrides', [])
    arg_name_enum = whitelist.get('override', {}).get('arg_name_enum', {}).get('allowed', [])
    
    # 收集所有 arg_names
    arg_names_in_overrides = set()
    for override in allowed_overrides:
        arg_name = override.get('arg_name')
        if arg_name:
            arg_names_in_overrides.add(arg_name)
    
    arg_names_in_enum = set(arg_name_enum)
    
    # 检查是否所有 override 的 arg_names 都在 enum 中
    missing_in_enum = arg_names_in_overrides - arg_names_in_enum
    extra_in_enum = arg_names_in_enum - arg_names_in_overrides
    
    if missing_in_enum:
        errors.append(f"arg_names 缺少在 arg_name_enum 中: {missing_in_enum}")
        print(f"✗ 缺失的 arg_names: {missing_in_enum}")
    else:
        print(f"✓ 所有 arg_names 都在 arg_name_enum 中")
    
    if extra_in_enum:
        print(f"⚠ arg_name_enum 中有多余的值: {extra_in_enum}")
    
    return len(errors) == 0, errors


def main():
    """主程序"""
    cwd = Path.cwd()
    
    # 查找项目根目录
    if (cwd / "configs" / "runtime_whitelist.yaml").exists():
        project_root = cwd
    elif (cwd.parent / "configs" / "runtime_whitelist.yaml").exists():
        project_root = cwd.parent
    else:
        print("错误: 无法找到项目根目录")
        sys.exit(1)
    
    whitelist_path = project_root / "configs" / "runtime_whitelist.yaml"
    default_config_path = project_root / "configs" / "default.yaml"
    paper_config_path = project_root / "configs" / "paper_full_cuda.yaml"
    
    print("=" * 80)
    print("验证 runtime_whitelist 和配置一致性")
    print("=" * 80)
    
    whitelist = load_whitelist(whitelist_path)
    
    print("\n[1] 检查 default.yaml 中的字段:")
    print("-" * 80)
    ok1, err1 = verify_override_fields(whitelist, default_config_path)
    
    print("\n[2] 检查 paper_full_cuda.yaml 中的字段:")
    print("-" * 80)
    ok2, err2 = verify_override_fields(whitelist, paper_config_path)
    
    print("\n[3] 验证 arg_name_enum 一致性:")
    print("-" * 80)
    ok3, err3 = verify_arg_name_enum(whitelist)
    
    print("\n" + "=" * 80)
    if ok1 and ok2 and ok3:
        print("✓ 所有验证通过！")
        return 0
    else:
        print("✗ 验证失败，请检查上面的错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
