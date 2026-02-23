"""
功能：publish workflow 默认使用 paper profile

Module type: General module

Regression test: publish workflow must use paper profile by default.
"""

import argparse
import sys
from pathlib import Path


def test_publish_workflow_uses_paper_profile_by_default():
    """
    功能：publish workflow 默认使用 paper profile。

    Verify that run_publish_workflow.py uses paper profile as default
    for signoff execution.

    GIVEN: run_publish_workflow.py argparse configuration
    WHEN: No --signoff-profile argument is provided
    THEN: Default value is "paper".
    """
    # 读取 run_publish_workflow.py 的 argparse 配置
    repo_root = Path(__file__).resolve().parent.parent
    publish_workflow_path = repo_root / "scripts" / "run_publish_workflow.py"

    assert publish_workflow_path.exists(), \
        f"publish workflow 脚本必须存在: {publish_workflow_path}"

    # 解析 argparse 默认值
    # 由于直接执行会启动流程，我们通过代码检查方式验证
    with publish_workflow_path.open("r", encoding="utf-8") as f:
        content = f.read()

    # 验证 --signoff-profile 参数存在且默认为 paper
    assert "--signoff-profile" in content, \
        "publish workflow 必须支持 --signoff-profile 参数"
    assert 'default="paper"' in content, \
        "publish workflow 的 --signoff-profile 默认值必须为 'paper'"
    assert 'choices=["baseline", "paper", "publish"]' in content, \
        "publish workflow 必须支持 baseline/paper/publish 三个 profile 选项"

    # 验证 signoff 调用传递了 profile 参数
    assert "--signoff-profile" in content and "signoff_profile" in content, \
        "publish workflow 必须将 profile 参数传递给 signoff 调用"

    # 验证代码中显式使用了 paper 作为发布级默认
    # 通过正则或字符串匹配确认 signoff 调用包含 --signoff-profile
    signoff_call_pattern = 'str(scripts_dir / "run_freeze_signoff.py")'
    assert signoff_call_pattern in content, \
        "publish workflow 必须调用 run_freeze_signoff.py"

    # 验证 signoff_profile 变量存在且被传递
    assert "signoff_profile = args.signoff_profile" in content or \
           'args.signoff_profile' in content, \
        "publish workflow 必须使用 args.signoff_profile 变量"


def test_publish_workflow_passes_require_experiment_matrix_to_signoff():
    """
    功能：publish workflow 调用 signoff 时必须传递 matrix 强制参数。

    Verify run_publish_workflow forwards --require-experiment-matrix to signoff.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    publish_workflow_path = repo_root / "scripts" / "run_publish_workflow.py"

    assert publish_workflow_path.exists(), \
        f"publish workflow 脚本必须存在: {publish_workflow_path}"

    content = publish_workflow_path.read_text(encoding="utf-8")
    assert "--require-experiment-matrix" in content, \
        "publish workflow 调用 signoff 时必须显式传 --require-experiment-matrix"
