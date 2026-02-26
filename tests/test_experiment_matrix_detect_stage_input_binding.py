"""
功能：验证 experiment_matrix 的 detect 阶段命令绑定 input 记录。

Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from main.evaluation import experiment_matrix


def test_experiment_matrix_detect_stage_passes_default_input_record(monkeypatch) -> None:
    """
    功能：detect 阶段必须自动传入 embed_record 作为 input。

    Verify detect stage command always includes --input with
    {run_root}/records/embed_record.json when no explicit input path is provided.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    captured = {"command": None}

    def _fake_run(command, capture_output, text):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(experiment_matrix.subprocess, "run", _fake_run)

    run_root = Path("outputs/experiment_matrix/experiments/item_0000")
    config_path = Path("configs/paper_full_cuda.yaml")

    experiment_matrix._run_stage_command(
        stage_name="detect",
        run_root=run_root,
        config_path=config_path,
        stage_overrides=["allow_nonempty_run_root=true"],
    )

    command = captured["command"]
    assert isinstance(command, list), "subprocess command must be captured as list"
    assert "--input" in command, "detect stage must include --input argument"

    input_idx = command.index("--input")
    assert input_idx + 1 < len(command), "--input must have a path value"
    assert command[input_idx + 1] == str(run_root / "records" / "embed_record.json")


def test_experiment_matrix_detect_stage_sets_content_enabled_override(monkeypatch) -> None:
    """
    功能：detect 阶段必须显式设置 detect.content.enabled=true。

    Verify _run_stage_sequence includes detect.content.enabled=true in detect stage overrides.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    captured_overrides = {}

    def _fake_layout(*args, **kwargs):
        return {
            "run_root": args[0],
            "artifacts_dir": args[0] / "artifacts",
            "records_dir": args[0] / "records",
        }

    def _fake_run_stage_command(stage_name, run_root, config_path, stage_overrides, input_record_path=None):
        captured_overrides[stage_name] = list(stage_overrides)

    monkeypatch.setattr(experiment_matrix.path_policy, "ensure_output_layout", _fake_layout)
    monkeypatch.setattr(experiment_matrix, "_run_stage_command", _fake_run_stage_command)

    grid_item_cfg = {
        "config_path": "configs/paper_full_cuda.yaml",
        "attack_protocol_path": "configs/attack_protocol.yaml",
        "cfg_snapshot": {
            "seed": 0,
            "model_id": "stabilityai/stable-diffusion-3.5-medium",
        },
        "ablation_flags": {},
        "max_samples": None,
    }

    experiment_matrix._run_stage_sequence(grid_item_cfg, Path("outputs/experiment_matrix/experiments/item_0000"))

    detect_overrides = captured_overrides.get("detect", [])
    assert "enable_content_detect=true" in detect_overrides


def test_prepare_detect_record_for_attack_grouping_writes_attack_fields(tmp_path: Path) -> None:
    """
    功能：为 calibrate/evaluate 生成带 attack 条件字段的 detect 副本。

    Verify helper writes an artifact copy containing both top-level and nested attack fields.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        None.
    """
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    source_path = records_dir / "detect_record.json"
    source_path.write_text(json.dumps({"operation": "detect"}), encoding="utf-8")

    grid_item_cfg = {
        "attack_protocol_family": "rotate",
        "attack_protocol_path": "configs/attack_protocol.yaml",
    }

    enriched_path = experiment_matrix._prepare_detect_record_for_attack_grouping(run_root, grid_item_cfg)

    assert enriched_path.exists()
    assert enriched_path != source_path

    enriched_obj = json.loads(enriched_path.read_text(encoding="utf-8"))
    assert enriched_obj["attack_family"] == "rotate"
    assert enriched_obj["attack_params_version"] == "v1"
    assert isinstance(enriched_obj.get("attack"), dict)
    assert enriched_obj["attack"]["family"] == "rotate"
    assert enriched_obj["attack"]["params_version"] == "v1"
