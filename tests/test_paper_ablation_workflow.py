"""
文件目的：验证 paper 单对象单变量消融工作流。
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, cast

import pytest
import yaml

from scripts import run_paper_ablation_workflow as ablation_workflow


def test_paper_ablation_config_exposes_workflow_section() -> None:
    """
    功能：验证 paper_ablation 配置同时保留 experiment_matrix 与 notebook workflow 配置段。

    Verify the ablation config preserves experiment_matrix semantics while also
    exposing the dedicated notebook workflow section.

    Args:
        None.

    Returns:
        None.
    """
    config_path = Path("configs/ablation/paper_ablation_cuda.yaml")
    config_obj = cast(Dict[str, Any], yaml.safe_load(config_path.read_text(encoding="utf-8")))

    assert isinstance(config_obj, dict)
    experiment_matrix_cfg = cast(Dict[str, Any], config_obj["experiment_matrix"])
    workflow_cfg = cast(Dict[str, Any], config_obj["paper_ablation_workflow"])
    base_embed_cfg = cast(Dict[str, Any], workflow_cfg["base_embed"])
    compare_cfg = cast(Dict[str, Any], workflow_cfg["compare"])
    notebook_runtime_cfg = cast(Dict[str, Any], workflow_cfg["notebook_runtime"])
    embed_cfg = cast(Dict[str, Any], config_obj["embed"])
    detect_cfg = cast(Dict[str, Any], config_obj["detect"])
    attestation_cfg = cast(Dict[str, Any], config_obj["attestation"])
    variants = cast(List[Dict[str, Any]], workflow_cfg["detect_rerun"]["variants"])
    variant_by_name = {cast(str, variant["name"]): variant for variant in variants}

    assert isinstance(experiment_matrix_cfg.get("ablation_variants"), list)
    assert cast(Dict[str, Any], embed_cfg["preview_generation"])["enabled"] is True
    assert cast(Dict[str, Any], detect_cfg["content"])["lf_exact_repair"]["enabled"] is True
    assert cast(Dict[str, Any], detect_cfg["geometry"])["geo_score_repair"]["enabled"] is True
    assert attestation_cfg["enabled"] is True
    assert attestation_cfg["use_trajectory_mix"] is False
    assert base_embed_cfg["allow_resume"] is True
    assert base_embed_cfg["allow_reuse_existing_record"] is True
    assert notebook_runtime_cfg["base_embed_reuse_mode"] == "fresh_run"
    assert "active_geo_score_source" in cast(List[str], compare_cfg["summary_fields"])
    assert "geo_repair_enabled" in cast(List[str], compare_cfg["table_fields"])
    assert variant_by_name["GEO-on"]["overrides"] == {"detect.geometry.geo_score_repair.enabled": True}
    assert variant_by_name["GEO-off"]["overrides"] == {"detect.geometry.geo_score_repair.enabled": False}
    assert variant_by_name["LF-repair-on"]["overrides"] == {"detect.content.lf_exact_repair.enabled": True}
    assert variant_by_name["LF-repair-off"]["overrides"] == {"detect.content.lf_exact_repair.enabled": False}
    assert "ablation.enable_geometry" not in variant_by_name["GEO-off"]["overrides"]


def test_run_paper_ablation_workflow_reuses_base_embed_for_detect_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 ablation workflow 只执行一次 embed，并让多个 detect variant 复用同一 input record。

    Verify the workflow executes one embed stage and reuses the same base
    embed record for all detect variants.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    config_path = tmp_path / "paper_ablation.yaml"
    config_obj: Dict[str, Any] = {
        "policy_path": "content_np_geo_rescue",
        "detect": {
            "content": {
                "enabled": True,
                "lf_exact_repair": {
                    "enabled": True,
                    "mode": "host_template_recenter",
                },
            },
            "geometry": {
                "enabled": True,
                "geo_score_repair": {
                    "enabled": True,
                    "mode": "template_confidence",
                },
            },
        },
        "ablation": {
            "enable_geometry": None,
            "enable_lf": None,
        },
        "experiment_matrix": {
            "config_path": "configs/ablation/paper_ablation_cuda.yaml",
            "ablation_variants": [{"enable_geometry": False}],
        },
        "paper_ablation_workflow": {
            "output_root": str(tmp_path / "outputs"),
            "config_snapshot_dir": "compare/config_snapshots",
            "base_embed": {
                "run_subdir": "base_embed",
                "embed_record_rel_path": "records/embed_record.json",
                "allow_resume": True,
                "allow_reuse_existing_record": True,
                "overrides": {},
            },
            "detect_rerun": {
                "variants_dir": "variants",
                "compare_dir": "compare",
                "input_record_rel_path": "records/embed_record.json",
                "variant_dir_pattern": "{suffix}",
                "strict_single_variable": True,
                "allow_detect_only": True,
                "reuse_existing_detect_results": True,
                "enable_calibration": False,
                "enable_evaluate": False,
                "reuse_thresholds_artifact": None,
                "variants": [
                    {
                        "name": "GEO-on",
                        "suffix": "GEO-on",
                        "enabled": True,
                        "group": "geo_repair_toggle",
                        "category": "detect_single_variable",
                        "description": "baseline",
                        "overrides": {"detect.geometry.geo_score_repair.enabled": True},
                    },
                    {
                        "name": "GEO-off",
                        "suffix": "GEO-off",
                        "enabled": True,
                        "group": "geo_repair_toggle",
                        "category": "detect_single_variable",
                        "description": "disable geo",
                        "overrides": {"detect.geometry.geo_score_repair.enabled": False},
                    },
                ],
            },
            "compare": {
                "summary_fields": [
                    "variant_name",
                    "variant_suffix",
                    "attestation_status",
                    "content_attestation_score",
                    "event_attestation_score",
                    "channel_scores_lf",
                    "channel_scores_hf",
                    "channel_scores_geo",
                    "active_geo_score_source",
                    "geo_repair_enabled",
                    "geo_repair_active",
                    "geo_repair_mode",
                    "formal_exact_evidence_source",
                    "protocol_root_cause_classification",
                    "detect_record_path",
                ],
                "table_fields": [
                    "variant_name",
                    "variant_suffix",
                    "channel_scores_geo",
                    "active_geo_score_source",
                    "geo_repair_enabled",
                    "lf_exact_repair_enabled",
                    "detect_record_path",
                ],
            },
            "notebook_runtime": {
                "selected_variants": ["GEO-on", "GEO-off"],
                "base_embed_reuse_mode": "fresh_run",
                "reuse_base_embed_record": None,
                "package_zip": True,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config_obj, sort_keys=False, allow_unicode=True), encoding="utf-8")

    captured_commands = []

    def _fake_run(
        command: List[str],
        cwd: str,
        capture_output: bool,
        text: bool,
        encoding: str,
        errors: str,
    ) -> SimpleNamespace:
        captured_commands.append(command)
        run_root = Path(command[command.index("--out") + 1])
        cfg_path = Path(command[command.index("--config") + 1])
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "records").mkdir(parents=True, exist_ok=True)
        cfg_obj_local = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        detect_cfg_local = cfg_obj_local.get("detect", {}) if isinstance(cfg_obj_local, dict) else {}
        geometry_cfg = detect_cfg_local.get("geometry", {}) if isinstance(detect_cfg_local, dict) else {}
        geo_repair_cfg = geometry_cfg.get("geo_score_repair", {}) if isinstance(geometry_cfg, dict) else {}
        geo_enabled = geo_repair_cfg.get("enabled") is not False

        if "main.cli.run_embed" in command:
            embed_payload = {
                "run_stage": "embed",
                "watermarked_image_path": str(run_root / "artifacts" / "embed" / "watermarked.png"),
            }
            (run_root / "records" / "embed_record.json").write_text(
                json.dumps(embed_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif "main.cli.run_detect" in command:
            score_value = 0.92 if geo_enabled else 0.11
            detect_payload = {
                "attestation": {
                    "status": "attested" if geo_enabled else "unattested",
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": score_value,
                        "content_attestation_score_name": "content_attestation_score",
                        "channel_scores": {
                            "lf": 0.8,
                            "hf": 0.7,
                            "geo": 1.0 if geo_enabled else 0.0,
                        },
                    },
                    "final_event_attested_decision": {
                        "status": "attested" if geo_enabled else "unattested",
                        "event_attestation_score": score_value,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
                "content_evidence_payload": {
                    "status": "ok",
                    "content_score": score_value,
                    "formal_exact_evidence_source": "input_image_conditioned_reconstruction",
                    "lf_exact_repair_enabled": True,
                    "lf_exact_repair_applied": True,
                    "lf_exact_repair_mode": "host_template_recenter",
                    "protocol_root_cause_classification": "minor_same_seed_residual_not_primary",
                    "lf_score": 0.8,
                    "hf_score": 0.7,
                    "score_parts": {
                        "content_score_rule_id": "lf_hf_weighted_sum",
                    },
                },
                "geometry_evidence_payload": {
                    "status": "ok" if geo_enabled else "disabled",
                    "geo_score": 1.0 if geo_enabled else 0.0,
                    "geo_repair_enabled": geo_enabled,
                    "geo_repair_active": geo_enabled,
                    "geo_repair_mode": "template_confidence",
                },
                "geo_rescue_diagnostics_artifact": {
                    "active_geo_score_source": "template_confidence",
                    "geo_repair_enabled": geo_enabled,
                    "geo_repair_active": geo_enabled,
                    "geo_repair_mode": "template_confidence",
                    "geo_score_repair_enabled": geo_enabled,
                    "geo_score_repair_active": geo_enabled,
                    "geo_score_repair_mode": "template_confidence",
                    "geo_repair_direction_classification": (
                        "template_confidence_rebinding_active" if geo_enabled else "template_confidence_disabled_by_config"
                    ),
                },
            }
            (run_root / "records" / "detect_record.json").write_text(
                json.dumps(detect_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ablation_workflow.subprocess, "run", _fake_run)

    run_root = tmp_path / "ablation_run"
    result = ablation_workflow.run_paper_ablation_workflow(
        config_path=config_path,
        run_root=run_root,
        selected_variant_names=["GEO-on", "GEO-off"],
        dry_run=False,
    )

    embed_commands = [command for command in captured_commands if "main.cli.run_embed" in command]
    detect_commands = [command for command in captured_commands if "main.cli.run_detect" in command]

    assert len(embed_commands) == 1
    assert len(detect_commands) == 2

    detect_input_paths = [command[command.index("--input") + 1] for command in detect_commands]
    assert len(set(detect_input_paths)) == 1
    assert detect_input_paths[0] == str(run_root / "base_embed" / "records" / "embed_record.json")

    manifest_path = Path(result["manifest_path"])
    summary_path = Path(result["compare_summary_path"])
    compare_csv_path = Path(result["compare_table_path"])

    assert manifest_path.exists()
    assert summary_path.exists()
    assert compare_csv_path.exists()
    assert Path(result["archive_path"]).exists()
    assert (run_root / "variants" / "GEO-on" / "records" / "detect_record.json").exists()
    assert (run_root / "variants" / "GEO-off" / "records" / "detect_record.json").exists()

    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))

    assert manifest_obj["base_embed"]["embed_record_path"] == str(run_root / "base_embed" / "records" / "embed_record.json")
    assert manifest_obj["base_embed"]["stage_executed"] is True
    assert manifest_obj["base_embed"]["source_mode"] == "new_embed"
    assert [variant["suffix"] for variant in manifest_obj["variants"]] == ["GEO-on", "GEO-off"]
    assert summary_obj["variant_count"] == 2
    assert [variant["variant_suffix"] for variant in summary_obj["variants"]] == ["GEO-on", "GEO-off"]
    assert summary_obj["variants"][0]["active_geo_score_source"] == "template_confidence"
    assert summary_obj["variants"][0]["formal_exact_evidence_source"] == "input_image_conditioned_reconstruction"


def test_run_paper_ablation_workflow_resume_reuses_existing_base_embed_and_detect_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 resume 模式会复用已有 base embed 和 detect record，仅补跑缺失 variant。

    Verify resume mode reuses existing base embed and detect records and only
    executes missing detect variants.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    config_path = tmp_path / "paper_ablation_resume.yaml"
    config_obj: Dict[str, Any] = {
        "policy_path": "content_np_geo_rescue",
        "detect": {
            "content": {"enabled": True, "lf_exact_repair": {"enabled": True, "mode": "host_template_recenter"}},
            "geometry": {"enabled": True, "geo_score_repair": {"enabled": True, "mode": "template_confidence"}},
        },
        "paper_ablation_workflow": {
            "output_root": str(tmp_path / "outputs"),
            "config_snapshot_dir": "compare/config_snapshots",
            "base_embed": {
                "run_subdir": "base_embed",
                "embed_record_rel_path": "records/embed_record.json",
                "allow_resume": True,
                "allow_reuse_existing_record": True,
                "overrides": {},
            },
            "detect_rerun": {
                "variants_dir": "variants",
                "compare_dir": "compare",
                "input_record_rel_path": "records/embed_record.json",
                "variant_dir_pattern": "{suffix}",
                "strict_single_variable": True,
                "allow_detect_only": True,
                "reuse_existing_detect_results": True,
                "enable_calibration": False,
                "enable_evaluate": False,
                "reuse_thresholds_artifact": None,
                "variants": [
                    {
                        "name": "GEO-on",
                        "suffix": "GEO-on",
                        "enabled": True,
                        "group": "geo_repair_toggle",
                        "category": "detect_single_variable",
                        "description": "baseline",
                        "overrides": {"detect.geometry.geo_score_repair.enabled": True},
                    },
                    {
                        "name": "GEO-off",
                        "suffix": "GEO-off",
                        "enabled": True,
                        "group": "geo_repair_toggle",
                        "category": "detect_single_variable",
                        "description": "disable geo",
                        "overrides": {"detect.geometry.geo_score_repair.enabled": False},
                    },
                ],
            },
            "compare": {
                "summary_fields": ["variant_name", "variant_suffix", "detect_record_path"],
                "table_fields": ["variant_name", "variant_suffix", "detect_record_path"],
            },
            "notebook_runtime": {
                "selected_variants": ["GEO-on", "GEO-off"],
                "base_embed_reuse_mode": "resume",
                "reuse_base_embed_record": None,
                "package_zip": False,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config_obj, sort_keys=False, allow_unicode=True), encoding="utf-8")

    run_root = tmp_path / "resume_run"
    (run_root / "base_embed" / "records").mkdir(parents=True, exist_ok=True)
    (run_root / "base_embed" / "records" / "embed_record.json").write_text("{}", encoding="utf-8")
    (run_root / "variants" / "GEO-on" / "records").mkdir(parents=True, exist_ok=True)
    (run_root / "variants" / "GEO-on" / "records" / "detect_record.json").write_text("{}", encoding="utf-8")

    captured_commands: List[List[str]] = []

    def _fake_run_resume(
        command: List[str],
        cwd: str,
        capture_output: bool,
        text: bool,
        encoding: str,
        errors: str,
    ) -> SimpleNamespace:
        captured_commands.append(command)
        if "main.cli.run_detect" in command:
            detect_root = Path(command[command.index("--out") + 1])
            (detect_root / "records").mkdir(parents=True, exist_ok=True)
            (detect_root / "records" / "detect_record.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ablation_workflow.subprocess, "run", _fake_run_resume)

    result = ablation_workflow.run_paper_ablation_workflow(
        config_path=config_path,
        run_root=run_root,
        selected_variant_names=["GEO-on", "GEO-off"],
        resume=True,
        dry_run=False,
    )

    assert not any("main.cli.run_embed" in command for command in captured_commands)
    detect_commands = [command for command in captured_commands if "main.cli.run_detect" in command]
    assert len(detect_commands) == 1
    assert str(run_root / "variants" / "GEO-off") in detect_commands[0]

    manifest_obj = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest_obj["base_embed"]["stage_executed"] is False
    assert manifest_obj["base_embed"]["source_mode"] == "resume_existing_base_embed"
    assert manifest_obj["variants"][0]["reused_existing_detect_record"] is True
    assert manifest_obj["variants"][1]["reused_existing_detect_record"] is False


def test_run_paper_ablation_workflow_supports_external_embed_record_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 external embed_record 可直接作为 detect 复跑输入。

    Verify an explicit external embed_record can be reused as the shared detect input.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    config_path = tmp_path / "paper_ablation_reuse.yaml"
    external_embed_record = tmp_path / "external" / "records" / "embed_record.json"
    external_embed_record.parent.mkdir(parents=True, exist_ok=True)
    external_embed_record.write_text("{}", encoding="utf-8")

    config_obj: Dict[str, Any] = {
        "policy_path": "content_np_geo_rescue",
        "detect": {
            "content": {"enabled": True, "lf_exact_repair": {"enabled": True, "mode": "host_template_recenter"}},
            "geometry": {"enabled": True, "geo_score_repair": {"enabled": True, "mode": "template_confidence"}},
        },
        "paper_ablation_workflow": {
            "output_root": str(tmp_path / "outputs"),
            "config_snapshot_dir": "compare/config_snapshots",
            "base_embed": {
                "run_subdir": "base_embed",
                "embed_record_rel_path": "records/embed_record.json",
                "allow_resume": True,
                "allow_reuse_existing_record": True,
                "overrides": {},
            },
            "detect_rerun": {
                "variants_dir": "variants",
                "compare_dir": "compare",
                "input_record_rel_path": "records/embed_record.json",
                "variant_dir_pattern": "{suffix}",
                "strict_single_variable": True,
                "allow_detect_only": True,
                "reuse_existing_detect_results": False,
                "enable_calibration": False,
                "enable_evaluate": False,
                "reuse_thresholds_artifact": None,
                "variants": [
                    {
                        "name": "GEO-on",
                        "suffix": "GEO-on",
                        "enabled": True,
                        "group": "geo_repair_toggle",
                        "category": "detect_single_variable",
                        "description": "baseline",
                        "overrides": {"detect.geometry.geo_score_repair.enabled": True},
                    }
                ],
            },
            "compare": {
                "summary_fields": ["variant_name", "detect_record_path"],
                "table_fields": ["variant_name", "detect_record_path"],
            },
            "notebook_runtime": {
                "selected_variants": ["GEO-on"],
                "base_embed_reuse_mode": "reuse_existing_record",
                "reuse_base_embed_record": str(external_embed_record),
                "package_zip": False,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config_obj, sort_keys=False, allow_unicode=True), encoding="utf-8")

    captured_commands: List[List[str]] = []

    def _fake_run_reuse(
        command: List[str],
        cwd: str,
        capture_output: bool,
        text: bool,
        encoding: str,
        errors: str,
    ) -> SimpleNamespace:
        captured_commands.append(command)
        detect_root = Path(command[command.index("--out") + 1])
        (detect_root / "records").mkdir(parents=True, exist_ok=True)
        (detect_root / "records" / "detect_record.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ablation_workflow.subprocess, "run", _fake_run_reuse)

    result = ablation_workflow.run_paper_ablation_workflow(
        config_path=config_path,
        run_root=tmp_path / "reuse_run",
        selected_variant_names=["GEO-on"],
        reuse_base_embed_record=external_embed_record,
        dry_run=False,
    )

    assert not any("main.cli.run_embed" in command for command in captured_commands)
    detect_commands = [command for command in captured_commands if "main.cli.run_detect" in command]
    assert len(detect_commands) == 1
    assert detect_commands[0][detect_commands[0].index("--input") + 1] == str(external_embed_record)

    manifest_obj = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest_obj["base_embed"]["source_mode"] == "external_embed_record"
    assert manifest_obj["base_embed"]["reuse_source_path"] == str(external_embed_record)


def test_paper_ablation_notebook_is_parseable_and_has_key_cells() -> None:
    """
    功能：验证 ablation notebook 可解析，并包含参数 cell、variant cell、运行 cell 与摘要 cell。

    Verify the ablation notebook is valid JSON and contains the expected user
    editable, run, and summary cells.

    Args:
        None.

    Returns:
        None.
    """
    notebook_path = Path("notebook/Paper_Ablation_Cuda.ipynb")
    notebook_obj = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert isinstance(notebook_obj, dict)
    cells = notebook_obj.get("cells")
    assert isinstance(cells, list) and cells

    joined_cell_sources = ["\n".join(cell.get("source", [])) for cell in cells if isinstance(cell, dict)]

    assert any("REPO_SOURCE_MODE" in cell_source and "git_clone_refresh" in cell_source for cell_source in joined_cell_sources)
    assert any("RUN_TAG" in cell_source and "BASE_EMBED_REUSE_MODE" in cell_source for cell_source in joined_cell_sources)
    assert any("ABLATION_SWITCH_NAME" in cell_source and "SINGLE_VARIABLE_REASON" in cell_source for cell_source in joined_cell_sources)
    assert any("run_paper_ablation_workflow.py" in cell_source for cell_source in joined_cell_sources)
    assert any("ablation_compare_summary.json" in cell_source for cell_source in joined_cell_sources)
