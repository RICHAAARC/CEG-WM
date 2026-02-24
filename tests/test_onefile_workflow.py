"""
文件目的：onefile workflow 编排回归测试。
Module type: General module
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, List
import sys

import pytest


def _load_onefile_module(repo_root: Path):
    """
    功能：动态加载 onefile workflow 脚本模块。 

    Dynamically load scripts/run_onefile_workflow.py as module.

    Args:
        repo_root: Repository root path.

    Returns:
        Imported module object.

    Raises:
        RuntimeError: If module spec cannot be loaded.
    """
    module_path = repo_root / "scripts" / "run_onefile_workflow.py"
    spec = importlib.util.spec_from_file_location("run_onefile_workflow", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_onefile_workflow_builds_commands_and_fail_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    功能：验证 onefile 命令顺序与 fail-fast。 

    Validate command sequence and fail-fast behavior for onefile workflow.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    cfg_path = repo_root / "configs" / "default.yaml"

    calls: List[List[str]] = []

    class _Completed:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def _fake_run_ok(cmd: List[str], cwd: str, check: bool, env: dict) -> Any:
        _ = cwd
        _ = check
        _ = env
        calls.append(list(cmd))
        return _Completed(0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run_ok)

    return_code = module.run_onefile_workflow(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile="cpu_smoke",
        signoff_profile="baseline",
        dry_run=False,
    )
    assert return_code == 0

    expected_order = [
        "main.cli.run_embed",
        "main.cli.run_detect",
        "main.cli.run_calibrate",
        "main.cli.run_evaluate",
        "run_all_audits.py",
        "run_all_audits.py",
        "run_freeze_signoff.py",
    ]

    observed_order = []
    for command in calls:
        command_text = " ".join(command)
        if "-m main.cli.run_embed" in command_text:
            observed_order.append("main.cli.run_embed")
        elif "-m main.cli.run_detect" in command_text:
            observed_order.append("main.cli.run_detect")
        elif "-m main.cli.run_calibrate" in command_text:
            observed_order.append("main.cli.run_calibrate")
        elif "-m main.cli.run_evaluate" in command_text:
            observed_order.append("main.cli.run_evaluate")
        elif "run_all_audits.py" in command_text and "--strict" not in command:
            observed_order.append("run_all_audits.py")
        elif "run_all_audits.py" in command_text and "--strict" in command:
            observed_order.append("run_all_audits.py")
        elif "run_freeze_signoff.py" in command_text:
            observed_order.append("run_freeze_signoff.py")

    assert observed_order == expected_order
    assert any("--strict" in command for command in calls), "strict 审计步骤必须存在"

    calls_fail_fast: List[List[str]] = []

    def _fake_run_fail_on_calibrate(cmd: List[str], cwd: str, check: bool, env: dict) -> Any:
        _ = cwd
        _ = check
        _ = env
        calls_fail_fast.append(list(cmd))
        command_text = " ".join(cmd)
        if "-m main.cli.run_calibrate" in command_text:
            return _Completed(9)
        return _Completed(0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run_fail_on_calibrate)
    return_code_fail = module.run_onefile_workflow(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile="cpu_smoke",
        signoff_profile="baseline",
        dry_run=False,
    )
    assert return_code_fail == 9
    joined_fail_fast = [" ".join(command) for command in calls_fail_fast]
    assert any("main.cli.run_calibrate" in item for item in joined_fail_fast)
    assert not any("main.cli.run_evaluate" in item for item in joined_fail_fast)
    assert not any("run_all_audits.py" in item for item in joined_fail_fast)
    assert not any("run_freeze_signoff.py" in item for item in joined_fail_fast)


def test_onefile_workflow_requires_run_root_or_generates_one(tmp_path: Path) -> None:
    """
    功能：验证 run_root 缺省自动生成与路径约束。 

    Validate generated run_root format and ensure it stays under repo outputs.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    module_path = Path(__file__).resolve().parent.parent / "scripts" / "run_onefile_workflow.py"
    spec = importlib.util.spec_from_file_location("run_onefile_workflow_for_path", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load onefile module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    generated = module._build_run_root(repo_root, None, "cpu_smoke")
    generated.resolve().relative_to((repo_root / "outputs").resolve())
    assert generated.name.startswith("onefile_cpu_smoke_")

    provided_relative = module._build_run_root(repo_root, "outputs/custom_run", "cpu_smoke")
    assert provided_relative == (repo_root / "outputs" / "custom_run").resolve()


def test_onefile_workflow_paper_full_profile_generates_real_sd3_config(tmp_path: Path) -> None:
    """
    功能：验证 paper_full_cuda profile 的真实 SD3 配置与命令覆盖。 

    Validate paper_full_cuda profile writes real SD3 config and command overrides.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "paper_full_run"
    cfg_path = repo_root / "configs" / "default.yaml"

    profile_cfg_path = module._prepare_profile_cfg_path("paper_full_cuda", run_root, cfg_path)
    profile_cfg_text = profile_cfg_path.read_text(encoding="utf-8")

    assert "pipeline_impl_id: sd3_diffusers_real_v1" in profile_cfg_text
    assert "pipeline_build_enabled: true" in profile_cfg_text
    assert "device: cuda" in profile_cfg_text
    assert "enabled: true" in profile_cfg_text
    assert "alignment_check: true" in profile_cfg_text
    assert "tail_truncation_mode: top_k_per_latent" in profile_cfg_text
    assert "coding_mode: latent_space_sign_flipping" in profile_cfg_text

    steps = module.build_workflow_steps(
        run_root=run_root,
        cfg_path=profile_cfg_path,
        repo_root=repo_root,
        profile="paper_full_cuda",
        signoff_profile="baseline",
    )
    embed_command_text = " ".join(steps[0].command)
    assert "enable_paper_faithfulness=true" in embed_command_text
    assert "enable_trace_tap=true" in embed_command_text
    assert "force_cpu=\"cpu\"" not in embed_command_text
    step_names = [item.name for item in steps]
    assert "multi_protocol_evaluation" in step_names
    assert "assert_paper_mechanisms" in step_names
