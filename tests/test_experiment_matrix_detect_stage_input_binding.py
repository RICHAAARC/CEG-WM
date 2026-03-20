"""
功能：验证 experiment_matrix 的 detect 阶段命令绑定 input 记录。

Module type: General module
"""

from __future__ import annotations

import copy
import glob
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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


def test_experiment_matrix_embed_stage_aligns_content_extractor_override(monkeypatch) -> None:
    """
    功能：embed 阶段必须显式对齐 onefile 的 content extractor 模式 override。

    Verify _run_stage_sequence adds disable_content_detect=false for the
    matrix embed stage so embed uses the same content extractor mode as the
    onefile embed path.

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

    embed_overrides = captured_overrides.get("embed", [])
    assert "disable_content_detect=false" in embed_overrides


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


def test_prepare_detect_record_for_attack_grouping_builds_join_key_from_infer_trace_prompt(
    tmp_path: Path,
) -> None:
    """
    功能：当 top-level prompt 缺失时，attack metadata join key 必须绑定 infer_trace prompt。 

    Verify detect_record_with_attack binds attack metadata join key from
    infer_trace.inference_prompt when the top-level inference_prompt is absent.

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
                "infer_trace": {
                    "inference_prompt": "matrix canonical prompt",
                },
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

    assert enriched_obj.get("inference_prompt") is None
    assert enriched_obj["infer_trace"]["inference_prompt"] == "matrix canonical prompt"
    assert enriched_obj["attack_metadata_source_prompt"] == "matrix canonical prompt"
    assert enriched_obj["attack_metadata_source_prompt_field"] == "infer_trace.inference_prompt"
    assert enriched_obj["attack_metadata_join_key"] == experiment_matrix._build_attack_metadata_join_key(
        "matrix canonical prompt",
        "rotate",
        "v1",
    )


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


def test_prepare_labelled_detect_records_uses_real_negative_payload_when_available(tmp_path: Path) -> None:
    """real neg payload 可用时，negative record 不得再克隆正样本 payload。"""
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    detect_record_path = records_dir / "detect_record.json"
    detect_record_path.write_text(
        json.dumps(
            {
                "operation": "detect",
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.75,
                },
                "positive_only_marker": "attacked_positive",
            }
        ),
        encoding="utf-8",
    )

    neg_detect_record_path = tmp_path / "neg_detect_record.json"
    neg_detect_record_path.write_text(
        json.dumps(
            {
                "operation": "detect",
                "contract_bound_digest": "contract-anchor",
                "whitelist_bound_digest": "whitelist-anchor",
                "policy_path_semantics_bound_digest": "semantics-anchor",
                "injection_scope_manifest_bound_digest": "injection-anchor",
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.12,
                },
                "neg_only_marker": "real_negative",
            }
        ),
        encoding="utf-8",
    )

    grid_item_cfg = {
        "attack_protocol_family": "rotate",
        "attack_protocol_path": "configs/attack_protocol.yaml",
        "disallow_forced_pair_fallback": True,
    }

    records_glob = experiment_matrix._prepare_labelled_detect_records_glob_for_matrix(
        run_root,
        grid_item_cfg,
        neg_detect_record_path=neg_detect_record_path,
    )
    payloads = [json.loads(Path(path_str).read_text(encoding="utf-8")) for path_str in sorted(glob.glob(records_glob))]

    negative_payload = next(item for item in payloads if item.get("label") is False)
    assert negative_payload["ground_truth_source"] == "real_neg_embed_detect"
    assert negative_payload["calibration_sample_usage"] == "real_negative_for_experiment_matrix_label_balance"
    assert negative_payload["calibration_label_resolution"] == "real_negative_payload"
    assert negative_payload["neg_only_marker"] == "real_negative"
    assert negative_payload.get("positive_only_marker") is None
    assert negative_payload["attack_family"] == "rotate"
    assert float(negative_payload["content_evidence_payload"]["score"]) == 0.12
    assert negative_payload.get("contract_bound_digest") is None
    assert negative_payload.get("whitelist_bound_digest") is None
    assert negative_payload.get("policy_path_semantics_bound_digest") is None
    assert negative_payload.get("injection_scope_manifest_bound_digest") is None


def test_prepare_labelled_detect_records_blocks_synthetic_negative_fallback_in_formal_mode(tmp_path: Path) -> None:
    """formal guard 开启时，若 real neg 无效，helper 必须阻断 synthetic fallback。"""
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    detect_record_path = records_dir / "detect_record.json"
    detect_record_path.write_text(
        json.dumps(
            {
                "operation": "detect",
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )

    invalid_neg_detect_record_path = tmp_path / "neg_detect_record_invalid.json"
    invalid_neg_detect_record_path.write_text(
        json.dumps(
            {
                "operation": "detect",
                "content_evidence_payload": {
                    "status": "failed",
                    "score": None,
                },
            }
        ),
        encoding="utf-8",
    )

    grid_item_cfg = {
        "attack_protocol_family": "rotate",
        "attack_protocol_path": "configs/attack_protocol.yaml",
        "disallow_forced_pair_fallback": True,
    }

    with pytest.raises(RuntimeError, match="disallows synthetic negative fallback"):
        experiment_matrix._prepare_labelled_detect_records_glob_for_matrix(
            run_root,
            grid_item_cfg,
            neg_detect_record_path=invalid_neg_detect_record_path,
        )


def test_prepare_labelled_detect_records_recovers_real_negative_detect_hf_score(tmp_path: Path) -> None:
    """real neg 仅有 detect_hf_score 时，helper 也必须恢复可校准分数。"""
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    detect_record_path = records_dir / "detect_record.json"
    detect_record_path.write_text(
        json.dumps(
            {
                "operation": "detect",
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )

    neg_detect_record_path = tmp_path / "neg_detect_record.json"
    neg_detect_record_path.write_text(
        json.dumps(
            {
                "operation": "detect",
                "content_evidence_payload": {
                    "status": "absent",
                    "score": None,
                    "detect_hf_score": 0.12,
                    "content_failure_reason": "formal_profile_sidecar_disabled",
                },
            }
        ),
        encoding="utf-8",
    )

    grid_item_cfg = {
        "attack_protocol_family": "rotate",
        "attack_protocol_path": "configs/attack_protocol.yaml",
        "disallow_forced_pair_fallback": True,
    }

    records_glob = experiment_matrix._prepare_labelled_detect_records_glob_for_matrix(
        run_root,
        grid_item_cfg,
        neg_detect_record_path=neg_detect_record_path,
    )
    payloads = [json.loads(Path(path_str).read_text(encoding="utf-8")) for path_str in sorted(glob.glob(records_glob))]

    negative_payload = next(item for item in payloads if item.get("label") is False)
    negative_content = negative_payload.get("content_evidence_payload")
    assert isinstance(negative_content, dict)
    assert negative_content.get("status") == "ok"
    assert float(negative_content["score"]) == 0.12
    assert negative_content.get("calibration_score_recovery_reason") == "content_evidence_payload.detect_hf_score"
    assert negative_content.get("calibration_sample_origin") == "real_negative_payload_recovery"
    assert negative_payload.get("calibration_sample_usage") == "real_negative_for_experiment_matrix_label_balance"


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


def test_run_stage_sequence_binds_labelled_detect_glob_for_calibrate_and_evaluate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    功能：矩阵子实验中 calibrate/evaluate 都应绑定带标签的 detect 记录 glob。

    Verify experiment matrix binds labelled detect records glob for both calibrate
    and evaluate stages.

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

    def _fake_prepare_labelled_glob(run_root: Path, _grid_item_cfg: dict, neg_detect_record_path=None) -> str:
        labelled_dir = run_root / "artifacts" / "evaluate_inputs" / "labelled_detect_records"
        labelled_dir.mkdir(parents=True, exist_ok=True)
        (labelled_dir / "detect_record_label_pos.json").write_text("{}", encoding="utf-8")
        (labelled_dir / "detect_record_label_neg.json").write_text("{}", encoding="utf-8")
        return str(labelled_dir / "*.json")

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
    monkeypatch.setattr(experiment_matrix, "_prepare_labelled_detect_records_glob_for_matrix", _fake_prepare_labelled_glob)
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

    assert any(item.startswith("calibrate_detect_records_glob=") for item in calibrate_overrides)
    assert any(item.startswith("evaluate_detect_records_glob=") for item in evaluate_overrides)


def test_prepare_labelled_detect_records_glob_contains_pos_and_neg_labels(tmp_path: Path) -> None:
    """
    功能：matrix 标注样本生成必须产出正负两类 bool 标签记录。

    Verify helper creates a labelled detect-record glob with both positive and
    negative boolean labels for calibration/evaluate label-balance gates.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        None.
    """
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    detect_record_path = records_dir / "detect_record.json"
    detect_record_path.write_text(
        json.dumps(
            {
                "operation": "detect",
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )

    grid_item_cfg = {
        "attack_protocol_family": "rotate",
        "attack_protocol_path": "configs/attack_protocol.yaml",
    }

    records_glob = experiment_matrix._prepare_labelled_detect_records_glob_for_matrix(run_root, grid_item_cfg)
    matched_paths = sorted(glob.glob(records_glob))

    assert len(matched_paths) == 2
    payloads = [json.loads(Path(path_str).read_text(encoding="utf-8")) for path_str in matched_paths]

    labels = sorted(payload.get("label") for payload in payloads)
    assert labels == [False, True]
    assert all(payload.get("ground_truth") in {False, True} for payload in payloads)
    assert all(payload.get("is_watermarked") in {False, True} for payload in payloads)
    assert all(payload.get("calibration_excluded_from_labelled_sampling") is None for payload in payloads)

    for payload in payloads:
        content_node = payload.get("content_evidence_payload")
        assert isinstance(content_node, dict)
        assert content_node.get("status") == "ok"
        assert isinstance(content_node.get("score"), (int, float))

    positive_payload = next(item for item in payloads if item.get("label") is True)
    negative_payload = next(item for item in payloads if item.get("label") is False)
    assert float(negative_payload["content_evidence_payload"]["score"]) < float(positive_payload["content_evidence_payload"]["score"])


def test_prepare_formal_evaluate_detect_records_glob_uses_shared_real_negatives(tmp_path: Path) -> None:
    """formal evaluate 输入必须使用 shared neg_staged，而不是本地 forced negative helper。"""
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    detect_record_path = records_dir / "detect_record.json"
    detect_record_path.write_text(
        json.dumps(
            {
                "operation": "detect",
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.75,
                },
                "infer_trace": {"inference_prompt": "matrix canonical prompt"},
            }
        ),
        encoding="utf-8",
    )

    thresholds_path = tmp_path / "global_calibrate" / "artifacts" / "thresholds" / "thresholds_artifact.json"
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text("{}", encoding="utf-8")

    neg_staged_dir = thresholds_path.parent.parent / "neg_staged"
    neg_staged_dir.mkdir(parents=True, exist_ok=True)
    for idx, score_value in enumerate([0.12, 0.23]):
        (neg_staged_dir / f"neg_record_{idx:04d}.json").write_text(
            json.dumps(
                {
                    "operation": "detect",
                    "label": False,
                    "ground_truth": False,
                    "is_watermarked": False,
                    "calibration_label_resolution": "global_calibrate_real_neg",
                    "ground_truth_source": "real_neg_embed_detect",
                    "calibration_sample_usage": "real_negative_global_calibrate_null_distribution",
                    "content_evidence_payload": {
                        "status": "ok",
                        "score": score_value,
                    },
                }
            ),
            encoding="utf-8",
        )

    grid_item_cfg = {
        "attack_protocol_family": "rotate",
        "attack_protocol_path": "configs/attack_protocol.yaml",
    }

    records_glob = experiment_matrix._prepare_formal_evaluate_detect_records_glob_for_matrix(
        run_root,
        grid_item_cfg,
        shared_thresholds_path=thresholds_path,
    )
    matched_paths = sorted(glob.glob(records_glob))

    assert Path(records_glob).parent.name == "formal_evaluate_records"
    assert len(matched_paths) == 3

    payloads = [json.loads(Path(path_str).read_text(encoding="utf-8")) for path_str in matched_paths]
    positive_payload = next(item for item in payloads if item.get("label") is True)
    negative_payloads = [item for item in payloads if item.get("label") is False]

    assert positive_payload["calibration_label_resolution"] == "matrix_forced_positive"
    assert positive_payload["attack_family"] == "rotate"
    assert positive_payload["attack"]["family"] == "rotate"
    assert len(negative_payloads) == 2
    assert all(item.get("calibration_label_resolution") == "global_calibrate_real_neg" for item in negative_payloads)
    assert all(item.get("calibration_sample_usage") == "real_negative_global_calibrate_null_distribution" for item in negative_payloads)
    assert all(item.get("attack_family") == "rotate" for item in negative_payloads)
    assert all(item.get("attack", {}).get("family") == "rotate" for item in negative_payloads)


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


def test_build_experiment_grid_propagates_formal_validation_guards() -> None:
    """
    功能：experiment_matrix formal guard 必须透传到每个网格项。

    Verify explicit formal-validation guard flags are propagated to each grid item.

    Args:
        None.

    Returns:
        None.
    """
    base_cfg = {
        "model_id": "model_a",
        "seed": 0,
        "experiment_matrix": {
            "require_real_negative_cache": True,
            "require_shared_thresholds": True,
            "disallow_forced_pair_fallback": True,
        },
    }

    grid = experiment_matrix.build_experiment_grid(base_cfg)
    assert isinstance(grid, list) and len(grid) > 0
    assert all(item.get("require_real_negative_cache") is True for item in grid)
    assert all(item.get("require_shared_thresholds") is True for item in grid)
    assert all(item.get("disallow_forced_pair_fallback") is True for item in grid)


def test_run_experiment_grid_fails_fast_when_formal_real_negative_cache_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    功能：formal matrix 若缺少 real negative cache，必须在网格级直接失败。

    Verify run_experiment_grid aborts before per-item execution when guarded items
    require real negative cache and cache generation fails.

    Args:
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    called = {"run_single_experiment": 0}

    def _fake_neg_cache(**kwargs):
        raise RuntimeError("neg_cache_failed")

    def _fake_run_single(_grid_item_cfg):
        called["run_single_experiment"] += 1
        return {"status": "ok"}

    monkeypatch.setattr(experiment_matrix, "_run_neg_embed_detect_for_cache", _fake_neg_cache)
    monkeypatch.setattr(experiment_matrix, "_run_global_calibrate", lambda **kwargs: tmp_path / "thresholds.json")
    monkeypatch.setattr(experiment_matrix, "run_single_experiment", _fake_run_single)

    grid = [
        {
            "grid_index": 0,
            "grid_item_digest": "a" * 64,
            "cfg_snapshot": {"seed": 1, "model_id": "model_a"},
            "ablation_flags": {},
            "attack_protocol_version": "attack_protocol_v1",
            "batch_root": str(tmp_path / "batch_root"),
            "config_path": "configs/default.yaml",
            "attack_protocol_path": "configs/attack_protocol.yaml",
            "max_samples": 2,
            "model_id": "model_a",
            "seed": 1,
            "require_real_negative_cache": True,
        },
    ]

    with pytest.raises(RuntimeError, match="requires real negative cache"):
        experiment_matrix.run_experiment_grid(grid, strict=True)

    assert called["run_single_experiment"] == 0


def test_run_experiment_grid_fails_fast_when_formal_shared_thresholds_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    功能：formal matrix 若缺少 shared thresholds，必须在网格级直接失败。

    Verify run_experiment_grid aborts before per-item execution when guarded items
    require shared thresholds and global calibrate produces no artifact.

    Args:
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    called = {"run_single_experiment": 0}
    neg_path = tmp_path / "neg_detect_record.json"
    neg_path.write_text(json.dumps({"content_evidence_payload": {"status": "ok", "score": 0.1}}), encoding="utf-8")

    def _fake_run_single(_grid_item_cfg):
        called["run_single_experiment"] += 1
        return {"status": "ok"}

    monkeypatch.setattr(experiment_matrix, "_run_neg_embed_detect_for_cache", lambda **kwargs: neg_path)
    monkeypatch.setattr(experiment_matrix, "_run_global_calibrate", lambda **kwargs: None)
    monkeypatch.setattr(experiment_matrix, "run_single_experiment", _fake_run_single)

    grid = [
        {
            "grid_index": 0,
            "grid_item_digest": "a" * 64,
            "cfg_snapshot": {"seed": 1, "model_id": "model_a"},
            "ablation_flags": {},
            "attack_protocol_version": "attack_protocol_v1",
            "batch_root": str(tmp_path / "batch_root"),
            "config_path": "configs/default.yaml",
            "attack_protocol_path": "configs/attack_protocol.yaml",
            "max_samples": 2,
            "model_id": "model_a",
            "seed": 1,
            "require_shared_thresholds": True,
        },
    ]

    with pytest.raises(RuntimeError, match="requires shared thresholds"):
        experiment_matrix.run_experiment_grid(grid, strict=True)

    assert called["run_single_experiment"] == 0


def test_run_global_calibrate_writes_shared_thresholds_from_neg_only_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    功能：global_calibrate 在仅有真实负样本 cache 时，也必须写出 shared thresholds 工件。

    Verify _run_global_calibrate bypasses the CLI label-balance gate and writes
    thresholds artifacts directly from the negative-only null distribution.

    Args:
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    neg_path = tmp_path / "neg_detect_record.json"
    neg_path.write_text(
        json.dumps(
            {
                "operation": "detect",
                "content_evidence_payload": {
                    "status": "absent",
                    "score": None,
                    "detect_hf_score": 0.12,
                    "content_failure_reason": "formal_profile_sidecar_disabled",
                },
                "label": False,
            }
        ),
        encoding="utf-8",
    )

    captured = {"cfg": None, "impl_set": None}

    def _fake_load_yaml_with_provenance(_config_path):
        return {
            "target_fpr": 0.01,
            "calibration": {
                "score_name": "content_score",
                "exclude_formal_sidecar_disabled_marker": True,
                "exclude_synthetic_negative_closure_marker": True,
            },
        }, {"file_sha256": "unused"}

    def _fake_run_calibrate_orchestrator(cfg, impl_set):
        captured["cfg"] = copy.deepcopy(cfg)
        captured["impl_set"] = impl_set
        return {
            "thresholds_artifact": {
                "threshold_id": "content_score_np_1e-02",
                "score_name": "content_score",
                "target_fpr": 0.01,
                "threshold_value": 0.12,
                "threshold_key_used": "1e-02",
            },
            "threshold_metadata_artifact": {
                "method": "neyman_pearson_v1",
                "null_source": "label_false_from_detect_records",
                "n_null": 1,
                "calibration_date": "1970-01-01",
                "quantile_method": "higher",
                "target_fprs": [0.01],
            },
        }

    monkeypatch.setattr(experiment_matrix.config_loader, "load_yaml_with_provenance", _fake_load_yaml_with_provenance)
    monkeypatch.setattr(experiment_matrix.detect_orchestrator, "run_calibrate_orchestrator", _fake_run_calibrate_orchestrator)

    thresholds_path = experiment_matrix._run_global_calibrate(
        batch_root=str(tmp_path / "batch_root"),
        config_path="configs/paper_full_cuda.yaml",
        neg_detect_record_cache={(
            "stabilityai/stable-diffusion-3.5-medium",
            0,
        ): neg_path},
    )

    assert thresholds_path is not None
    assert thresholds_path.exists()
    assert thresholds_path.name == "thresholds_artifact.json"
    staged_glob = Path(captured["cfg"]["calibration"]["detect_records_glob"])
    assert staged_glob.parent.name == "neg_staged"
    assert staged_glob.name == "*.json"
    assert captured["impl_set"] is not None

    staged_paths = sorted(staged_glob.parent.glob("*.json"))
    assert len(staged_paths) == 1
    staged_payload = json.loads(staged_paths[0].read_text(encoding="utf-8"))
    staged_content = staged_payload.get("content_evidence_payload")
    assert isinstance(staged_content, dict)
    assert staged_content.get("status") == "ok"
    assert float(staged_content["score"]) == 0.12
    assert staged_content.get("calibration_score_recovery_reason") == "content_evidence_payload.detect_hf_score"
    assert staged_content.get("calibration_sample_origin") == "global_calibrate_real_negative_recovery"

    threshold_metadata_path = thresholds_path.parent / "threshold_metadata_artifact.json"
    assert threshold_metadata_path.exists()


def test_run_stage_sequence_uses_pair_free_formal_evaluate_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    功能：formal matrix 在 shared thresholds 下必须切换到 pair-free evaluate 输入路径。

    Verify _run_stage_sequence bypasses the local labelled-pair helper and binds
    the pair-free formal evaluate input glob when shared thresholds are active.

    Args:
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    called_stages = []
    captured_overrides = {}
    called_labelled_helper = {"called": 0}
    called_formal_helper = {"called": 0}
    thresholds_path = tmp_path / "shared_thresholds.json"
    thresholds_path.write_text("{}", encoding="utf-8")
    neg_path = tmp_path / "neg_detect_record.json"
    neg_path.write_text(json.dumps({"content_evidence_payload": {"status": "ok", "score": 0.1}}), encoding="utf-8")

    def _fake_layout(*args, **kwargs):
        run_root = args[0]
        (run_root / "records").mkdir(parents=True, exist_ok=True)
        (run_root / "artifacts").mkdir(parents=True, exist_ok=True)
        return {
            "run_root": run_root,
            "artifacts_dir": run_root / "artifacts",
            "records_dir": run_root / "records",
        }

    def _fake_run_stage(stage_name, run_root, config_path, stage_overrides, input_record_path=None):
        called_stages.append(stage_name)
        captured_overrides[stage_name] = list(stage_overrides)
        if stage_name == "detect":
            detect_record_path = run_root / "records" / "detect_record.json"
            detect_record_path.parent.mkdir(parents=True, exist_ok=True)
            detect_record_path.write_text(
                json.dumps({"content_evidence_payload": {"status": "ok", "score": 0.2}}),
                encoding="utf-8",
            )

    def _fake_prepare_labelled_glob(*args, **kwargs):
        called_labelled_helper["called"] += 1
        labelled_dir = tmp_path / "run" / "artifacts" / "evaluate_inputs" / "labelled_detect_records"
        labelled_dir.mkdir(parents=True, exist_ok=True)
        (labelled_dir / "detect_record_label_pos.json").write_text("{}", encoding="utf-8")
        (labelled_dir / "detect_record_label_neg.json").write_text("{}", encoding="utf-8")
        return str(labelled_dir / "*.json")

    def _fake_prepare_formal_glob(*args, **kwargs):
        called_formal_helper["called"] += 1
        formal_dir = tmp_path / "run" / "artifacts" / "evaluate_inputs" / "formal_evaluate_records"
        formal_dir.mkdir(parents=True, exist_ok=True)
        (formal_dir / "detect_record_positive.json").write_text("{}", encoding="utf-8")
        (formal_dir / "neg_record_0000.json").write_text("{}", encoding="utf-8")
        return str(formal_dir / "*.json")

    monkeypatch.setattr(experiment_matrix.path_policy, "ensure_output_layout", _fake_layout)
    monkeypatch.setattr(experiment_matrix, "_run_stage_command", _fake_run_stage)
    monkeypatch.setattr(experiment_matrix, "_prepare_labelled_detect_records_glob_for_matrix", _fake_prepare_labelled_glob)
    monkeypatch.setattr(
        experiment_matrix,
        "_prepare_formal_evaluate_detect_records_glob_for_matrix",
        _fake_prepare_formal_glob,
    )

    grid_item_cfg = {
        "config_path": "configs/paper_full_cuda.yaml",
        "attack_protocol_path": "configs/attack_protocol.yaml",
        "cfg_snapshot": {"seed": 0, "model_id": "stabilityai/stable-diffusion-3.5-medium"},
        "ablation_flags": {},
        "max_samples": None,
        "neg_detect_record_path": str(neg_path),
        "shared_thresholds_path": str(thresholds_path),
        "require_real_negative_cache": True,
        "require_shared_thresholds": True,
        "disallow_forced_pair_fallback": True,
    }

    experiment_matrix._run_stage_sequence(grid_item_cfg, tmp_path / "run")

    assert called_stages == ["embed", "detect", "evaluate"]
    assert called_labelled_helper["called"] == 0
    assert called_formal_helper["called"] == 1
    evaluate_overrides = captured_overrides.get("evaluate", [])
    assert any("formal_evaluate_records" in item for item in evaluate_overrides)


def test_run_stage_sequence_preserves_debug_forced_pair_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    功能：非 formal guard 路径仍应保留 forced pair fallback，供 debug/test 使用。

    Verify the legacy forced-pair labelled fallback remains available when no
    formal-validation guard is enabled.

    Args:
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    called_stages = []
    called_labelled_helper = {"called": 0}

    def _fake_layout(*args, **kwargs):
        run_root = args[0]
        (run_root / "records").mkdir(parents=True, exist_ok=True)
        (run_root / "artifacts").mkdir(parents=True, exist_ok=True)
        return {
            "run_root": run_root,
            "artifacts_dir": run_root / "artifacts",
            "records_dir": run_root / "records",
        }

    def _fake_run_stage(stage_name, run_root, config_path, stage_overrides, input_record_path=None):
        called_stages.append(stage_name)
        if stage_name == "detect":
            detect_record_path = run_root / "records" / "detect_record.json"
            detect_record_path.parent.mkdir(parents=True, exist_ok=True)
            detect_record_path.write_text(
                json.dumps({"content_evidence_payload": {"status": "ok", "score": 0.2}}),
                encoding="utf-8",
            )

    def _fake_prepare_labelled_glob(run_root: Path, _grid_item_cfg: dict, neg_detect_record_path=None) -> str:
        called_labelled_helper["called"] += 1
        labelled_dir = run_root / "artifacts" / "evaluate_inputs" / "labelled_detect_records"
        labelled_dir.mkdir(parents=True, exist_ok=True)
        (labelled_dir / "detect_record_label_pos.json").write_text("{}", encoding="utf-8")
        (labelled_dir / "detect_record_label_neg.json").write_text("{}", encoding="utf-8")
        return str(labelled_dir / "*.json")

    monkeypatch.setattr(experiment_matrix.path_policy, "ensure_output_layout", _fake_layout)
    monkeypatch.setattr(experiment_matrix, "_run_stage_command", _fake_run_stage)
    monkeypatch.setattr(experiment_matrix, "_prepare_labelled_detect_records_glob_for_matrix", _fake_prepare_labelled_glob)

    grid_item_cfg = {
        "config_path": "configs/paper_full_cuda.yaml",
        "attack_protocol_path": "configs/attack_protocol.yaml",
        "cfg_snapshot": {"seed": 0, "model_id": "stabilityai/stable-diffusion-3.5-medium"},
        "ablation_flags": {},
        "max_samples": None,
    }

    experiment_matrix._run_stage_sequence(grid_item_cfg, tmp_path / "run")

    assert called_stages == ["embed", "detect", "calibrate", "evaluate"]
    assert called_labelled_helper["called"] == 1


def test_paper_acceptance_gate_prefers_canonical_runtime_mode(tmp_path: Path) -> None:
    """
    功能：验证 paper acceptance gate 优先读取 canonical runtime mode。 

    Verify paper acceptance uses detect_runtime_mode_canonical before the legacy
    detect_runtime_mode field.
    """
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    (records_dir / "detect_record.json").write_text(
        json.dumps(
            {
                "detect_runtime_mode": "fallback_identity",
                "detect_runtime_mode_canonical": "real",
                "pipeline_runtime_meta": {"synthetic_pipeline": False},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "status": "ok",
        "failure_reason": "ok",
        "metrics": {"geo_available_rate": 0.5},
        "hf_truncation_baseline_comparison": {"comparison_ready": True},
    }
    grid_item_cfg = {
        "paper_faithfulness": {"enabled": True},
    }

    experiment_matrix._enforce_paper_acceptance_gate(summary, grid_item_cfg, run_root)  # pyright: ignore[reportPrivateUsage]

    assert summary["status"] == "ok"
    assert summary["failure_reason"] == "ok"


def test_run_neg_embed_detect_for_cache_uses_preview_image_as_detect_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    功能：neg cache 必须基于 preview_generation 的 clean 图驱动 detect。 

    Verify neg cache runs embed without embed_identity_mode and passes a preview-based
    input record into detect.

    Args:
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    batch_root = tmp_path / "batch_root"
    preview_image_path = tmp_path / "preview_clean.png"
    preview_image_path.write_bytes(b"preview-bytes")

    captured_embed_command = {}
    captured_detect_call = {}

    def _fake_subprocess_run(command, capture_output, text):
        captured_embed_command["command"] = list(command)
        assert capture_output is True
        assert text is True
        return SimpleNamespace(
            returncode=0,
            stdout=f"[Preview Generation] 预览图已生成，路径：{preview_image_path}\n",
            stderr="",
        )

    def _fake_run_stage_command(stage_name, run_root, config_path, stage_overrides, input_record_path=None):
        captured_detect_call["stage_name"] = stage_name
        captured_detect_call["run_root"] = run_root
        captured_detect_call["config_path"] = config_path
        captured_detect_call["stage_overrides"] = list(stage_overrides)
        captured_detect_call["input_record_path"] = input_record_path
        detect_record_path = run_root / "records" / "detect_record.json"
        detect_record_path.parent.mkdir(parents=True, exist_ok=True)
        detect_record_path.write_text(
            json.dumps({"content_evidence_payload": {"status": "ok", "score": 0.11}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(experiment_matrix.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(experiment_matrix, "_run_stage_command", _fake_run_stage_command)

    detect_record_path = experiment_matrix._run_neg_embed_detect_for_cache(
        model_id="stabilityai/stable-diffusion-3.5-medium",
        seed=7,
        config_path="configs/paper_full_cuda.yaml",
        batch_root=str(batch_root),
        max_samples=None,
    )

    assert isinstance(detect_record_path, Path)
    assert detect_record_path.exists()

    embed_command = captured_embed_command.get("command")
    assert isinstance(embed_command, list)
    assert "main.cli.run_embed" in embed_command
    assert not any("embed_identity_mode=true" == item for item in embed_command)
    assert "--config" in embed_command

    embed_config_index = embed_command.index("--config")
    embed_config_path = Path(embed_command[embed_config_index + 1])
    embed_cfg_obj = yaml.safe_load(embed_config_path.read_text(encoding="utf-8"))
    assert isinstance(embed_cfg_obj, dict)
    assert embed_cfg_obj["detect"]["content"]["enabled"] is False
    assert embed_cfg_obj["attestation"]["enabled"] is False
    assert embed_cfg_obj["attestation"]["require_signed_bundle_verification"] is False

    assert captured_detect_call.get("stage_name") == "detect"
    assert "allow_threshold_fallback_for_tests=true" in captured_detect_call.get("stage_overrides", [])
    detect_config_path = captured_detect_call.get("config_path")
    assert detect_config_path == embed_config_path
    input_record_path = captured_detect_call.get("input_record_path")
    assert isinstance(input_record_path, Path)
    input_record_obj = json.loads(input_record_path.read_text(encoding="utf-8"))
    assert input_record_obj["image_path"] == str(preview_image_path)
    assert input_record_obj["watermarked_path"] == str(preview_image_path)
    assert input_record_obj["inputs"]["input_image_path"] == str(preview_image_path)
