"""
File purpose: 审计脚本 audit_no_empty_py_modules 的回归测试
Module type: General module

Test that audit_no_empty_py_modules correctly detects and blocks 
zero-byte .py modules in the release surface.
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest


# 导入审计模块
_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
_audits_dir = _scripts_dir / "audits"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from audits.audit_no_empty_py_modules import audit_no_empty_py_modules


def test_audit_no_empty_py_modules_passes_on_repo():
    """
    功能：验证当前仓库删除空文件后，审计脚本返回 PASS。

    Test that audit passes when no empty .py modules exist in main/.
    """
    repo_root = Path(__file__).resolve().parent.parent
    result = audit_no_empty_py_modules(repo_root)
    
    # 断言：结果字典结构正确
    assert isinstance(result, dict)
    assert "audit_id" in result
    assert "result" in result
    assert "severity" in result
    assert "evidence" in result
    
    # 断言：审计通过（当前仓库已删除空文件）
    assert result["audit_id"] == "empty_py_modules"
    assert result["result"] == "PASS"
    assert result["severity"] == "NON_BLOCK"
    
    # 断言：证据显示没有空模块
    evidence = result.get("evidence", {})
    assert evidence.get("count") == 0
    assert evidence.get("empty_modules") == []


def test_audit_no_empty_py_modules_fails_on_injected_empty_module(tmp_path):
    """
    功能：验证审计脚本能检测到注入的空文件并返回 FAIL。

    Test that audit fails when an empty .py module is injected.
    """
    # (1) 构造最小目录树：tmp_path/main/foo.py (0 bytes)
    repo_root = tmp_path
    main_dir = repo_root / "main"
    main_dir.mkdir(parents=True, exist_ok=True)
    
    # (2) 创建一个 0 字节的 .py 文件（非 __init__.py）
    empty_module = main_dir / "foo.py"
    empty_module.touch()  # 创建 0 字节文件
    
    assert empty_module.stat().st_size == 0
    assert empty_module.name == "foo.py"
    
    # (3) 运行审计
    result = audit_no_empty_py_modules(repo_root)
    
    # (4) 断言：审计失败
    assert result["audit_id"] == "empty_py_modules"
    assert result["result"] == "FAIL"
    assert result["severity"] == "BLOCK"
    
    # (5) 断言：输出中包含 main/foo.py
    evidence = result.get("evidence", {})
    assert evidence.get("count") == 1
    assert "main/foo.py" in evidence.get("empty_modules", [])


def test_audit_no_empty_py_modules_ignores_init_files(tmp_path):
    """
    功能：验证审计脚本忽略 __init__.py（即使为 0 字节）。

    Test that __init__.py files are never flagged as violations.
    """
    # 构造目录树：tmp_path/main/__init__.py (0 bytes) + content.py (0 bytes)
    repo_root = tmp_path
    main_dir = repo_root / "main"
    main_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建 __init__.py（0 字节，应被忽略）
    init_file = main_dir / "__init__.py"
    init_file.touch()
    
    # 创建其他 0 字节文件（应被检测）
    content_file = main_dir / "content.py"
    content_file.touch()
    
    result = audit_no_empty_py_modules(repo_root)
    
    # 断言：只检测到 content.py，不检测 __init__.py
    assert result["result"] == "FAIL"
    evidence = result.get("evidence", {})
    empty_modules = evidence.get("empty_modules", [])
    
    assert "main/content.py" in empty_modules
    assert "main/__init__.py" not in empty_modules


def test_audit_no_empty_py_modules_handles_nested_directories(tmp_path):
    """
    功能：验证审计脚本正确处理嵌套目录中的空文件。

    Test that audit finds empty modules in nested main/ subdirectories.
    """
    # 构造嵌套目录：tmp_path/main/sub1/sub2/empty.py
    repo_root = tmp_path
    nested_dir = repo_root / "main" / "sub1" / "sub2"
    nested_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建 __init__.py（应被忽略）
    (nested_dir.parent / "__init__.py").touch()
    (nested_dir / "__init__.py").touch()
    
    # 创建空模块文件
    empty_file = nested_dir / "empty.py"
    empty_file.touch()
    
    result = audit_no_empty_py_modules(repo_root)
    
    # 断言：检测到嵌套的空文件
    assert result["result"] == "FAIL"
    evidence = result.get("evidence", {})
    empty_modules = evidence.get("empty_modules", [])
    
    assert "main/sub1/sub2/empty.py" in empty_modules


def test_audit_no_empty_py_modules_ignores_non_py_files(tmp_path):
    """
    功能：验证审计脚本仅检查 .py 文件（忽略其他类型）。

    Test that audit ignores non-.py files.
    """
    repo_root = tmp_path
    main_dir = repo_root / "main"
    main_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建各种 0 字节文件
    (main_dir / "empty.txt").touch()  # 不是 .py，应被忽略
    (main_dir / "empty.json").touch()  # 不是 .py，应被忽略
    (main_dir / "empty.py").touch()  # 是 .py，应被检测
    
    result = audit_no_empty_py_modules(repo_root)
    
    # 断言：只检测到 .py 文件
    assert result["result"] == "FAIL"
    evidence = result.get("evidence", {})
    empty_modules = evidence.get("empty_modules", [])
    
    assert len(empty_modules) == 1
    assert "main/empty.py" in empty_modules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
