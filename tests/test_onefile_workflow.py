"""
文件目的：onefile workflow 编排回归测试。
Module type: General module
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, List
import sys
import json

import pytest
import yaml


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
    audit_commands = [command for command in calls if "run_all_audits.py" in " ".join(command)]
    assert len(audit_commands) == 2, "must execute audits and audits_strict"
    for audit_command in audit_commands:
        assert "--run-root" in audit_command, "audits command must bind current run_root"
        run_root_arg_idx = audit_command.index("--run-root")
        assert audit_command[run_root_arg_idx + 1] == str(run_root)
    detect_commands = [command for command in calls if "-m" in command and "main.cli.run_detect" in command]
    assert detect_commands, "detect command must be present"
    assert "allow_threshold_fallback_for_tests=true" in detect_commands[0]

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
    cfg_path = repo_root / "configs" / "paper_full_cuda.yaml"

    profile_cfg_path = module._prepare_profile_cfg_path("paper_full_cuda", run_root, cfg_path)
    profile_cfg_text = profile_cfg_path.read_text(encoding="utf-8")
    profile_cfg_obj = yaml.safe_load(profile_cfg_text)

    assert profile_cfg_obj["impl"]["sync_module_id"] == "geometry_latent_sync_sd3_v2"
    assert profile_cfg_obj["impl"]["geometry_extractor_id"] == "attention_anchor_map_relation_v1"
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


def test_onefile_workflow_paper_full_profile_fails_fast_on_mismatched_impl(tmp_path: Path) -> None:
    """
    功能：验证 paper_full_cuda 对关键 impl 错配执行 fail-fast。 

    Verify paper_full_cuda rejects mismatched frozen impl bindings.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "paper_full_bad_impl"
    cfg_obj = {
        "impl": {
            "sync_module_id": "geometry_latent_sync_sd3_v1",
            "geometry_extractor_id": "geometry_align_invariance_sd3_v1",
        }
    }
    cfg_path = tmp_path / "bad_impl_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="paper_full_cuda requires impl.sync_module_id"):
        module._prepare_profile_cfg_path("paper_full_cuda", run_root, cfg_path)


def test_resolve_default_signoff_profile_for_profile() -> None:
    """
    功能：验证 signoff profile 默认解析策略。

    Validate default signoff profile resolution by workflow profile.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    assert module._resolve_default_signoff_profile_for_profile("cpu_smoke", None) == "baseline"
    assert module._resolve_default_signoff_profile_for_profile("paper_full_cuda", None) == "paper"
    assert module._resolve_default_signoff_profile_for_profile("paper_full_cuda", "publish") == "publish"


def test_build_stage_overrides_sets_embed_detect_content_switch() -> None:
    """
    功能：验证 embed/detect 阶段 content detect 开关显式写入 override。

    Validate explicit content detect switch overrides for embed and detect stages.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    embed_overrides = module._build_stage_overrides("embed", "paper_full_cuda")
    detect_overrides = module._build_stage_overrides("detect", "paper_full_cuda")

    assert "disable_content_detect=true" in embed_overrides
    assert "enable_content_detect=true" in detect_overrides


def test_validate_multi_protocol_compare_summary_rejects_failed_protocol(tmp_path: Path) -> None:
    """
    功能：验证 compare summary 含失败协议时触发阻断。

    Validate compare summary validation fails when protocol status is not ok.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    compare_summary_path = tmp_path / "compare_summary.json"
    compare_summary_path.write_text(
        json.dumps(
            {
                "schema_version": "protocol_compare_v1",
                "protocols": [{"status": "ok"}, {"status": "fail"}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="failed protocols"):
        module._validate_multi_protocol_compare_summary(compare_summary_path)


def test_validate_multi_protocol_compare_summary_accepts_all_ok(tmp_path: Path) -> None:
    """
    功能：验证 compare summary 全部成功时通过校验。

    Validate compare summary validation passes when all protocols are ok.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    compare_summary_path = tmp_path / "compare_summary.json"
    compare_summary_path.write_text(
        json.dumps(
            {
                "schema_version": "protocol_compare_v1",
                "protocols": [{"status": "ok"}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    module._validate_multi_protocol_compare_summary(compare_summary_path)
