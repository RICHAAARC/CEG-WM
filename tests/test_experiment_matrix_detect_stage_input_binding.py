"""
功能：验证 experiment_matrix 的 detect 阶段命令绑定 input 记录。

Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        if stage_name == "detect":
            detect_record_path = run_root / "records" / "detect_record.json"
            detect_record_path.parent.mkdir(parents=True, exist_ok=True)
            detect_record_path.write_text(
                json.dumps({"content_evidence_payload": {"status": "ok", "score": 0.1}}),
                encoding="utf-8",
            )

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
    assert "allow_threshold_fallback_for_tests=true" in detect_overrides


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
    source_path.write_text(
        json.dumps(
            {
                "operation": "detect",
                "contract_bound_digest": "x",
                "whitelist_bound_digest": "y",
                "policy_path_semantics_bound_digest": "z",
            }
        ),
        encoding="utf-8",
    )

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
    assert enriched_obj.get("label") is None
    assert enriched_obj.get("ground_truth") is None
    assert enriched_obj.get("is_watermarked") is None
    assert enriched_obj.get("calibration_label_resolution") == "missing"
    assert enriched_obj.get("calibration_excluded_from_labelled_sampling") is True
    assert isinstance(enriched_obj.get("attack"), dict)
    assert enriched_obj["attack"]["family"] == "rotate"
    assert enriched_obj["attack"]["params_version"] == "v1"
    assert "contract_bound_digest" not in enriched_obj
    assert "whitelist_bound_digest" not in enriched_obj
    assert "policy_path_semantics_bound_digest" not in enriched_obj


def test_prepare_detect_record_for_attack_grouping_preserves_false_label(tmp_path: Path) -> None:
    """
    功能：若源记录含有效 bool 标签，enrich 结果必须保留该标签语义。

    Verify enrich helper preserves boolean negative label and marks resolution as resolved.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        None.
    """
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    source_path = records_dir / "detect_record.json"
    source_path.write_text(
        json.dumps(
            {
                "operation": "detect",
                "ground_truth": False,
            }
        ),
        encoding="utf-8",
    )

    grid_item_cfg = {
        "attack_protocol_family": "rotate",
        "attack_protocol_path": "configs/attack_protocol.yaml",
    }

    enriched_path = experiment_matrix._prepare_detect_record_for_attack_grouping(run_root, grid_item_cfg)
    enriched_obj = json.loads(enriched_path.read_text(encoding="utf-8"))

    assert enriched_obj.get("label") is False
    assert enriched_obj.get("ground_truth") is False
    assert enriched_obj.get("is_watermarked") is False
    assert enriched_obj.get("calibration_label_resolution") == "resolved"
    assert enriched_obj.get("calibration_excluded_from_labelled_sampling") is None


def test_detect_gate_blocks_calibrate_when_no_valid_content_score(tmp_path: Path, monkeypatch) -> None:
    """detect 后若没有有效 content_score，必须 fail-fast 且不进入 calibrate。"""
    called_stages = []

    def _fake_layout(*args, **kwargs):
        run_root = args[0]
        (run_root / "records").mkdir(parents=True, exist_ok=True)
        (run_root / "artifacts").mkdir(parents=True, exist_ok=True)
        return {
            "run_root": run_root,
            "artifacts_dir": run_root / "artifacts",
            "records_dir": run_root / "records",
        }

    def _fake_prepare_detect_record(run_root: Path, _grid_item_cfg: dict) -> Path:
        path = run_root / "artifacts" / "evaluate_inputs" / "detect_record_with_attack.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    def _fake_run_stage(stage_name, run_root, config_path, stage_overrides, input_record_path=None):
        called_stages.append(stage_name)
        if stage_name == "detect":
            detect_record_path = run_root / "records" / "detect_record.json"
            detect_record_path.parent.mkdir(parents=True, exist_ok=True)
            detect_record_path.write_text(
                json.dumps({"content_evidence_payload": {"status": "mismatch", "score": None}}),
                encoding="utf-8",
            )

    monkeypatch.setattr(experiment_matrix.path_policy, "ensure_output_layout", _fake_layout)
    monkeypatch.setattr(experiment_matrix, "_prepare_detect_record_for_attack_grouping", _fake_prepare_detect_record)
    monkeypatch.setattr(experiment_matrix, "_run_stage_command", _fake_run_stage)

    grid_item_cfg = {
        "config_path": "configs/paper_full_cuda.yaml",
        "attack_protocol_path": "configs/attack_protocol.yaml",
        "cfg_snapshot": {"seed": 0, "model_id": "stabilityai/stable-diffusion-3.5-medium"},
        "ablation_flags": {},
        "max_samples": None,
    }

    with pytest.raises(RuntimeError, match="insufficient valid content_score samples"):
        experiment_matrix._run_stage_sequence(grid_item_cfg, tmp_path / "run")

    assert called_stages == ["embed", "detect"]


def test_detect_gate_allows_progress_when_content_score_valid(tmp_path: Path, monkeypatch) -> None:
    """detect 后若有至少 1 条有效 content_score，允许进入 calibrate/evaluate。"""
    called_stages = []

    def _fake_layout(*args, **kwargs):
        run_root = args[0]
        (run_root / "records").mkdir(parents=True, exist_ok=True)
        (run_root / "artifacts").mkdir(parents=True, exist_ok=True)
        return {
            "run_root": run_root,
            "artifacts_dir": run_root / "artifacts",
            "records_dir": run_root / "records",
        }

    def _fake_prepare_detect_record(run_root: Path, _grid_item_cfg: dict) -> Path:
        path = run_root / "artifacts" / "evaluate_inputs" / "detect_record_with_attack.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    def _fake_run_stage(stage_name, run_root, config_path, stage_overrides, input_record_path=None):
        called_stages.append(stage_name)
        if stage_name == "detect":
            detect_record_path = run_root / "records" / "detect_record.json"
            detect_record_path.parent.mkdir(parents=True, exist_ok=True)
            detect_record_path.write_text(
                json.dumps({"content_evidence_payload": {"status": "ok", "score": 0.123}}),
                encoding="utf-8",
            )

    monkeypatch.setattr(experiment_matrix.path_policy, "ensure_output_layout", _fake_layout)
    monkeypatch.setattr(experiment_matrix, "_prepare_detect_record_for_attack_grouping", _fake_prepare_detect_record)
    monkeypatch.setattr(experiment_matrix, "_run_stage_command", _fake_run_stage)

    grid_item_cfg = {
        "config_path": "configs/paper_full_cuda.yaml",
        "attack_protocol_path": "configs/attack_protocol.yaml",
        "cfg_snapshot": {"seed": 0, "model_id": "stabilityai/stable-diffusion-3.5-medium"},
        "ablation_flags": {},
        "max_samples": None,
    }

    experiment_matrix._run_stage_sequence(grid_item_cfg, tmp_path / "run")
    assert called_stages == ["embed", "detect", "calibrate", "evaluate"]


def test_run_stage_sequence_only_binds_attacked_detect_glob_for_evaluate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    功能：矩阵子实验中仅 evaluate 绑定 attacked detect 记录，calibrate 不应绑定单条记录。

    Verify experiment matrix binds attacked detect record only for evaluate stage,
    while calibrate stage keeps default sampling path.

    Args:
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    captured_overrides = {}

    def _fake_layout(*args, **kwargs):
        run_root = args[0]
        (run_root / "records").mkdir(parents=True, exist_ok=True)
        (run_root / "artifacts").mkdir(parents=True, exist_ok=True)
        return {
            "run_root": run_root,
            "artifacts_dir": run_root / "artifacts",
            "records_dir": run_root / "records",
        }

    def _fake_prepare_detect_record(run_root: Path, _grid_item_cfg: dict) -> Path:
        path = run_root / "artifacts" / "evaluate_inputs" / "detect_record_with_attack.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    def _fake_run_stage_command(stage_name, run_root, config_path, stage_overrides, input_record_path=None):
        captured_overrides[stage_name] = list(stage_overrides)
        if stage_name == "detect":
            detect_record_path = run_root / "records" / "detect_record.json"
            detect_record_path.parent.mkdir(parents=True, exist_ok=True)
            detect_record_path.write_text(
                json.dumps({"content_evidence_payload": {"status": "ok", "score": 0.2}}),
                encoding="utf-8",
            )

    monkeypatch.setattr(experiment_matrix.path_policy, "ensure_output_layout", _fake_layout)
    monkeypatch.setattr(experiment_matrix, "_prepare_detect_record_for_attack_grouping", _fake_prepare_detect_record)
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

    experiment_matrix._run_stage_sequence(grid_item_cfg, tmp_path / "run")

    calibrate_overrides = captured_overrides.get("calibrate", [])
    evaluate_overrides = captured_overrides.get("evaluate", [])

    assert all(not item.startswith("calibrate_detect_records_glob=") for item in calibrate_overrides)
    assert any(item.startswith("evaluate_detect_records_glob=") for item in evaluate_overrides)


def test_run_stage_sequence_skips_ablation_overrides_when_cfg_snapshot_has_no_ablation(
    monkeypatch,
) -> None:
    """
    功能：当 cfg_snapshot 不含 ablation 段时，禁止注入 ablation_enable_* 覆盖项。

    Verify _run_stage_sequence does not append ablation_enable_* overrides
    when cfg_snapshot has no ablation section.

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
        if stage_name == "detect":
            detect_record_path = run_root / "records" / "detect_record.json"
            detect_record_path.parent.mkdir(parents=True, exist_ok=True)
            detect_record_path.write_text(
                json.dumps({"content_evidence_payload": {"status": "ok", "score": 0.2}}),
                encoding="utf-8",
            )

    def _fake_prepare_detect_record(run_root: Path, _grid_item_cfg: dict) -> Path:
        path = run_root / "artifacts" / "evaluate_inputs" / "detect_record_with_attack.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(experiment_matrix.path_policy, "ensure_output_layout", _fake_layout)
    monkeypatch.setattr(experiment_matrix, "_run_stage_command", _fake_run_stage_command)
    monkeypatch.setattr(experiment_matrix, "_prepare_detect_record_for_attack_grouping", _fake_prepare_detect_record)

    grid_item_cfg = {
        "config_path": "configs/paper_full_cuda.yaml",
        "attack_protocol_path": "configs/attack_protocol.yaml",
        "cfg_snapshot": {
            "seed": 0,
            "model_id": "stabilityai/stable-diffusion-3.5-medium",
        },
        "ablation_flags": {
            "enable_content": True,
            "enable_geometry": False,
        },
        "max_samples": None,
    }

    experiment_matrix._run_stage_sequence(
        grid_item_cfg,
        Path("outputs/experiment_matrix/experiments/item_0000"),
    )

    for stage_name in ["embed", "detect", "calibrate", "evaluate"]:
        overrides = captured_overrides.get(stage_name, [])
        assert all(not item.startswith("ablation_enable_") for item in overrides)
        assert all(not item.startswith("enable_paper_faithfulness=") for item in overrides)


def test_run_stage_sequence_disables_paper_faithfulness_for_all_matrix_items(monkeypatch) -> None:
    """
    功能：矩阵子实验不应强制注入 paper faithfulness 覆盖（含 baseline 与 ablation）。

    Verify _run_stage_sequence does not append any enable_paper_faithfulness override
    for all stages regardless of ablation flags.

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
        if stage_name == "detect":
            detect_record_path = run_root / "records" / "detect_record.json"
            detect_record_path.parent.mkdir(parents=True, exist_ok=True)
            detect_record_path.write_text(
                json.dumps({"content_evidence_payload": {"status": "ok", "score": 0.3}}),
                encoding="utf-8",
            )

    def _fake_prepare_detect_record(run_root: Path, _grid_item_cfg: dict) -> Path:
        path = run_root / "artifacts" / "evaluate_inputs" / "detect_record_with_attack.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(experiment_matrix.path_policy, "ensure_output_layout", _fake_layout)
    monkeypatch.setattr(experiment_matrix, "_run_stage_command", _fake_run_stage_command)
    monkeypatch.setattr(experiment_matrix, "_prepare_detect_record_for_attack_grouping", _fake_prepare_detect_record)

    grid_item_cfg = {
        "config_path": "configs/paper_full_cuda.yaml",
        "attack_protocol_path": "configs/attack_protocol.yaml",
        "cfg_snapshot": {
            "seed": 0,
            "model_id": "stabilityai/stable-diffusion-3.5-medium",
            "ablation": {
                "enable_content": None,
            },
        },
        "ablation_flags": {
            "enable_content": False,
        },
        "max_samples": None,
    }

    experiment_matrix._run_stage_sequence(
        grid_item_cfg,
        Path("outputs/experiment_matrix/experiments/item_0001"),
    )

    for stage_name in ["embed", "detect", "calibrate", "evaluate"]:
        overrides = captured_overrides.get(stage_name, [])
        assert all(not item.startswith("enable_paper_faithfulness=") for item in overrides)


def test_detect_gate_research_collection_mode_relaxes_and_records_metadata(tmp_path: Path, monkeypatch) -> None:
    """开启研究采集模式后，detect 硬门禁可受控放行并返回 gate_relaxed 元数据。"""
    called_stages = []

    def _fake_layout(*args, **kwargs):
        run_root = args[0]
        (run_root / "records").mkdir(parents=True, exist_ok=True)
        (run_root / "artifacts").mkdir(parents=True, exist_ok=True)
        return {
            "run_root": run_root,
            "artifacts_dir": run_root / "artifacts",
            "records_dir": run_root / "records",
        }

    def _fake_prepare_detect_record(run_root: Path, _grid_item_cfg: dict) -> Path:
        path = run_root / "artifacts" / "evaluate_inputs" / "detect_record_with_attack.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    def _fake_run_stage(stage_name, run_root, config_path, stage_overrides, input_record_path=None):
        called_stages.append(stage_name)
        if stage_name == "detect":
            detect_record_path = run_root / "records" / "detect_record.json"
            detect_record_path.parent.mkdir(parents=True, exist_ok=True)
            detect_record_path.write_text(
                json.dumps({"content_evidence_payload": {"status": "mismatch", "score": 0}}),
                encoding="utf-8",
            )

    monkeypatch.setattr(experiment_matrix.path_policy, "ensure_output_layout", _fake_layout)
    monkeypatch.setattr(experiment_matrix, "_prepare_detect_record_for_attack_grouping", _fake_prepare_detect_record)
    monkeypatch.setattr(experiment_matrix, "_run_stage_command", _fake_run_stage)

    grid_item_cfg = {
        "config_path": "configs/paper_full_cuda.yaml",
        "attack_protocol_path": "configs/attack_protocol.yaml",
        "cfg_snapshot": {"seed": 0, "model_id": "stabilityai/stable-diffusion-3.5-medium"},
        "ablation_flags": {},
        "max_samples": None,
        "allow_failed_semantics_collection": True,
    }

    gate_info = experiment_matrix._run_stage_sequence(grid_item_cfg, tmp_path / "run")
    assert called_stages == ["embed", "detect", "calibrate", "evaluate"]
    assert gate_info.get("gate_relaxed") is True
    assert gate_info.get("reason") == "insufficient_valid_content_score_samples_research_collection_mode"
    assert isinstance(gate_info.get("sample_counts"), dict)
    assert gate_info["sample_counts"].get("valid_content_score_samples") == 0


def test_run_single_experiment_uses_failed_status_token_on_error(tmp_path: Path, monkeypatch) -> None:
    """
    功能：单实验异常时状态口径必须为 failed（禁止 legacy fail）。

    Verify run_single_experiment returns status="failed" when execution raises.

    Args:
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr(experiment_matrix, "_derive_run_root", lambda _cfg: tmp_path / "run")

    def _raise_stage_error(_grid_item_cfg, _run_root):
        raise RuntimeError("synthetic_failure")

    monkeypatch.setattr(experiment_matrix, "_run_stage_sequence", _raise_stage_error)

    grid_item_cfg = {
        "grid_index": 0,
        "grid_item_digest": "d" * 64,
        "cfg_digest": "c" * 64,
        "ablation_digest": "a" * 64,
        "attack_protocol_digest": "p" * 64,
        "attack_protocol_version": "attack_protocol_v1",
        "attack_protocol_family": "rotate",
        "model_id": "model_a",
        "seed": 1,
    }

    summary = experiment_matrix.run_single_experiment(grid_item_cfg)
    assert summary.get("status") == "failed"
    assert "synthetic_failure" in str(summary.get("failure_reason"))


def test_build_experiment_grid_propagates_allow_failed_semantics_collection() -> None:
    """
    功能：experiment_matrix 配置中的 allow_failed_semantics_collection 必须透传到每个网格项。

    Verify allow_failed_semantics_collection from experiment_matrix config
    is propagated to each grid item.

    Args:
        None.

    Returns:
        None.
    """
    base_cfg = {
        "output_root": "outputs/test_matrix",
        "run_name": "test_matrix",
        "watermark_embed": {"num_seeds": 1},
        "grid": {
            "embed_grid": {"num_seeds": [1]},
            "detect_grid": {},
        },
        "experiment_matrix": {
            "allow_failed_semantics_collection": True,
        },
    }

    grid = experiment_matrix.build_experiment_grid(base_cfg)
    assert isinstance(grid, list) and len(grid) > 0
    assert all(item.get("allow_failed_semantics_collection") is True for item in grid)
