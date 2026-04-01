"""
File purpose: 验证 stage 03 experiment_matrix 对 shared thresholds 注入与 clean negative detect 输入的合同。
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from main.evaluation import experiment_matrix


def _fake_layout(run_root: Path, **_kwargs: Any) -> Dict[str, Path]:
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"
    records_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_root": run_root,
        "records_dir": records_dir,
        "artifacts_dir": artifacts_dir,
        "logs_dir": logs_dir,
    }


def test_run_stage_command_passes_detect_thresholds_path(monkeypatch, tmp_path: Path) -> None:
    """
    功能：验证 detect CLI 在显式提供 thresholds_path 时必须注入 --thresholds-path。

    Verify detect stage command appends --thresholds-path when a canonical
    thresholds artifact path is provided.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    captured: Dict[str, Any] = {"command": None}

    def _fake_run(command: List[str], capture_output: bool, text: bool) -> SimpleNamespace:
        captured["command"] = list(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    thresholds_path = tmp_path / "thresholds_artifact.json"
    thresholds_path.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "default.yaml"
    config_path.write_text("experiment_matrix: {}\n", encoding="utf-8")
    run_root = tmp_path / "run"

    monkeypatch.setattr(experiment_matrix.subprocess, "run", _fake_run)

    experiment_matrix._run_stage_command(
        stage_name="detect",
        run_root=run_root,
        config_path=config_path,
        stage_overrides=["enable_content_detect=true"],
        thresholds_path=thresholds_path,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert "--thresholds-path" in command
    thresholds_arg_index = command.index("--thresholds-path")
    assert command[thresholds_arg_index + 1] == str(thresholds_path.resolve())


def test_run_stage_sequence_injects_shared_thresholds_without_detect_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 formal stage 03 detect 主路径必须显式消费 shared thresholds，且不得再保留 fallback override。

    Verify the formal detect stage passes thresholds_path into _run_stage_command
    and removes the test-only fallback override when shared thresholds are available.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    captured_calls: List[Dict[str, Any]] = []
    thresholds_path = tmp_path / "global_calibrate" / "artifacts" / "thresholds" / "thresholds_artifact.json"
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text("{}", encoding="utf-8")

    def _fake_run_stage_command(
        stage_name: str,
        run_root: Path,
        config_path: Path,
        stage_overrides: List[str],
        input_record_path: Optional[Path] = None,
        thresholds_path: Optional[Path] = None,
    ) -> None:
        captured_calls.append(
            {
                "stage_name": stage_name,
                "run_root": run_root,
                "config_path": config_path,
                "stage_overrides": list(stage_overrides),
                "input_record_path": input_record_path,
                "thresholds_path": thresholds_path,
            }
        )
        if stage_name == "detect":
            detect_record_path = run_root / "records" / "detect_record.json"
            detect_record_path.parent.mkdir(parents=True, exist_ok=True)
            detect_record_path.write_text(
                json.dumps({"content_evidence_payload": {"status": "ok", "score": 0.2}}),
                encoding="utf-8",
            )

    monkeypatch.setattr(experiment_matrix.path_policy, "ensure_output_layout", _fake_layout)
    monkeypatch.setattr(experiment_matrix, "_run_stage_command", _fake_run_stage_command)
    monkeypatch.setattr(
        experiment_matrix,
        "_assert_min_valid_content_scores_after_detect",
        lambda *args, **kwargs: {"gate_relaxed": False, "reason": "ok", "sample_counts": {"valid": 1}},
    )
    monkeypatch.setattr(
        experiment_matrix,
        "_prepare_system_final_labelled_detect_records_glob_for_matrix",
        lambda *args, **kwargs: str(tmp_path / "labelled_detect_records" / "*.json"),
    )

    grid_item_cfg = {
        "config_path": "configs/default.yaml",
        "attack_protocol_path": "configs/attack_protocol.yaml",
        "cfg_snapshot": {"seed": 0, "model_id": "stabilityai/stable-diffusion-3.5-medium"},
        "ablation_flags": {},
        "max_samples": None,
        "shared_thresholds_path": str(thresholds_path),
        "require_shared_thresholds": True,
        "require_real_negative_cache": True,
        "disallow_forced_pair_fallback": True,
        "enable_auxiliary_analysis_runtime": False,
    }

    stage_result = experiment_matrix._run_stage_sequence(grid_item_cfg, tmp_path / "run")

    detect_call = next(item for item in captured_calls if item["stage_name"] == "detect")
    assert detect_call["thresholds_path"] == thresholds_path
    assert "allow_threshold_fallback_for_tests=true" not in detect_call["stage_overrides"]
    assert stage_result["auxiliary_analysis"]["status"] == "skipped"
    assert stage_result["auxiliary_analysis"]["auxiliary_analysis_runtime_executed"] is False


def test_run_stage_sequence_fails_fast_when_shared_thresholds_required_but_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 formal stage 03 在缺失 shared thresholds 时必须 fail-fast，禁止静默回退到 detect fallback。

    Verify the formal detect path raises immediately when shared thresholds are
    required but unavailable, without invoking detect fallback execution.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    called_stages: List[str] = []

    def _fake_run_stage_command(
        stage_name: str,
        run_root: Path,
        config_path: Path,
        stage_overrides: List[str],
        input_record_path: Optional[Path] = None,
        thresholds_path: Optional[Path] = None,
    ) -> None:
        _ = run_root, config_path, stage_overrides, input_record_path, thresholds_path
        called_stages.append(stage_name)

    monkeypatch.setattr(experiment_matrix.path_policy, "ensure_output_layout", _fake_layout)
    monkeypatch.setattr(experiment_matrix, "_run_stage_command", _fake_run_stage_command)

    grid_item_cfg = {
        "config_path": "configs/default.yaml",
        "attack_protocol_path": "configs/attack_protocol.yaml",
        "cfg_snapshot": {"seed": 0, "model_id": "stabilityai/stable-diffusion-3.5-medium"},
        "ablation_flags": {},
        "max_samples": None,
        "shared_thresholds_path": str(tmp_path / "missing_thresholds_artifact.json"),
        "require_shared_thresholds": True,
        "require_real_negative_cache": True,
        "disallow_forced_pair_fallback": True,
        "enable_auxiliary_analysis_runtime": False,
    }

    with pytest.raises(RuntimeError, match="requires shared thresholds"):
        experiment_matrix._run_stage_sequence(grid_item_cfg, tmp_path / "run")

    assert called_stages == ["embed"]


def test_write_neg_preview_input_record_omits_plan_bound_fields(tmp_path: Path) -> None:
    """
    功能：验证 neg_cache detect_input_record 只保留 clean negative 的最小图像输入语义。

    Verify the neg-cache detect input record omits plan-bound fields and keeps
    only the clean preview image bindings.

    Args:
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    preview_image_path = tmp_path / "preview.png"
    preview_image_path.write_bytes(b"preview-bytes")
    embed_record_path = tmp_path / "embed_record.json"
    embed_record_path.write_text(
        json.dumps(
            {
                "plan_digest": "plan_digest_01",
                "basis_digest": "basis_digest_01",
                "subspace_planner_impl_identity": {"impl_id": "planner_impl"},
                "subspace_plan": {"rank": 128},
                "content_evidence": {
                    "trajectory_evidence": {"trajectory_digest": "trajectory_digest_01"},
                },
            }
        ),
        encoding="utf-8",
    )

    input_record_path = experiment_matrix._write_neg_preview_input_record(
        run_root=run_root,
        preview_image_path=preview_image_path,
        embed_record_path=embed_record_path,
    )

    input_record_payload = json.loads(input_record_path.read_text(encoding="utf-8"))

    assert input_record_payload["operation"] == "embed_preview_input"
    assert input_record_payload["image_path"] == str(preview_image_path)
    assert input_record_payload["watermarked_path"] == str(preview_image_path)
    assert input_record_payload["inputs"]["input_image_path"] == str(preview_image_path)
    assert "plan_digest" not in input_record_payload
    assert "basis_digest" not in input_record_payload
    assert "subspace_planner_impl_identity" not in input_record_payload
    assert "subspace_plan" not in input_record_payload
    assert "content_evidence" not in input_record_payload