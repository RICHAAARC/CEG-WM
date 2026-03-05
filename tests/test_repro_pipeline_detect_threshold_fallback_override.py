"""
文件用途：回归测试 - repro pipeline detect 阶段必须携带 allow_threshold_fallback_for_tests=true。
模块类型：General module

背景：repro pipeline 的执行顺序为 embed → detect → calibrate → evaluate。
detect 阶段执行时尚无 NP 阈值工件（calibrate 尚未运行），
因此必须通过 CLI override 显式启用 fallback，与 onefile 主工作流行为一致。

回归来源：GPU 运行 compare_summary 显示 detect 崩溃（exit 1），
错误为 "np threshold artifact is required; set __thresholds_artifact__ or
explicitly enable allow_threshold_fallback_for_tests"。
"""

import sys
from pathlib import Path

import pytest

# 将 scripts 目录加入 sys.path 以便直接导入。
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run_repro_pipeline import _build_stage_command  # noqa: E402


@pytest.fixture()
def dummy_paths(tmp_path):
    """返回 _build_stage_command 所需的虚拟路径参数。"""
    run_root = tmp_path / "run"
    run_root.mkdir()
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("", encoding="utf-8")
    attack_path = tmp_path / "attack.yaml"
    attack_path.write_text("", encoding="utf-8")
    return run_root, config_path, attack_path


def test_detect_stage_includes_fallback_override(dummy_paths):
    """detect 阶段命令必须包含 allow_threshold_fallback_for_tests=true override。"""
    run_root, config_path, attack_path = dummy_paths
    cmd = _build_stage_command(
        stage_name="detect",
        run_root=run_root,
        config_path=config_path,
        attack_protocol_path=attack_path,
        seeds=None,
        max_samples=None,
    )
    cmd_str = " ".join(cmd)
    assert "allow_threshold_fallback_for_tests=true" in cmd_str, (
        "detect 阶段命令缺少 allow_threshold_fallback_for_tests=true，"
        "将导致无阈值工件时 get_np_threshold 抛出 ValueError。"
    )


def test_detect_stage_override_is_properly_structured(dummy_paths):
    """allow_threshold_fallback_for_tests=true 必须以 --override 参数对的形式出现。"""
    run_root, config_path, attack_path = dummy_paths
    cmd = _build_stage_command(
        stage_name="detect",
        run_root=run_root,
        config_path=config_path,
        attack_protocol_path=attack_path,
        seeds=None,
        max_samples=None,
    )
    # 确认 --override 和值作为相邻两项出现（非单字符串合并）。
    for i, token in enumerate(cmd):
        if token == "--override" and i + 1 < len(cmd):
            if cmd[i + 1] == "allow_threshold_fallback_for_tests=true":
                return
    pytest.fail(
        "detect 命令中未找到以 --override allow_threshold_fallback_for_tests=true 形式出现的参数对。"
    )


@pytest.mark.parametrize("stage_name", ["embed", "calibrate", "evaluate"])
def test_non_detect_stages_do_not_include_fallback_override(dummy_paths, stage_name):
    """非 detect 阶段不应携带 allow_threshold_fallback_for_tests override（语义隔离）。"""
    run_root, config_path, attack_path = dummy_paths

    # calibrate / evaluate 阶段需要 embed_record（路径隐含依赖，此处仅测试命令构建，不执行）。
    cmd = _build_stage_command(
        stage_name=stage_name,
        run_root=run_root,
        config_path=config_path,
        attack_protocol_path=attack_path,
        seeds=None,
        max_samples=None,
    )
    cmd_str = " ".join(cmd)
    assert "allow_threshold_fallback_for_tests" not in cmd_str, (
        f"{stage_name} 阶段不应携带 allow_threshold_fallback_for_tests override，"
        "该参数仅限 detect 阶段使用以对齐 onefile 主工作流。"
    )
