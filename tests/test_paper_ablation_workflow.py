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
    variants = cast(List[Dict[str, Any]], workflow_cfg["detect_rerun"]["variants"])
    variant_names = [cast(str, variant["name"]) for variant in variants]

    assert isinstance(experiment_matrix_cfg.get("ablation_variants"), list)
    assert "GEO-on" in variant_names
    assert "GEO-off" in variant_names
    assert "LF-off" in variant_names


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
                "overrides": {},
            },
            "detect_rerun": {
                "variants_dir": "variants",
                "compare_dir": "compare",
                "input_record_rel_path": "records/embed_record.json",
                "variant_dir_pattern": "{suffix}",
                "strict_single_variable": True,
                "enable_calibration": False,
                "enable_evaluate": False,
                "reuse_thresholds_artifact": None,
                "variants": [
                    {
                        "name": "GEO-on",
                        "suffix": "GEO-on",
                        "enabled": True,
                        "description": "baseline",
                        "overrides": {},
                    },
                    {
                        "name": "GEO-off",
                        "suffix": "GEO-off",
                        "enabled": True,
                        "description": "disable geo",
                        "overrides": {"ablation.enable_geometry": False},
                    },
                ],
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
        ablation_cfg = cfg_obj_local.get("ablation", {}) if isinstance(cfg_obj_local, dict) else {}
        geo_enabled = ablation_cfg.get("enable_geometry") is not False

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
                    "lf_score": 0.8,
                    "hf_score": 0.7,
                    "score_parts": {
                        "content_score_rule_id": "lf_hf_weighted_sum",
                    },
                },
                "geometry_evidence_payload": {
                    "status": "ok" if geo_enabled else "disabled",
                    "geo_score": 1.0 if geo_enabled else 0.0,
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
    assert (run_root / "variants" / "GEO-on" / "records" / "detect_record.json").exists()
    assert (run_root / "variants" / "GEO-off" / "records" / "detect_record.json").exists()

    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))

    assert manifest_obj["base_embed"]["embed_record_path"] == str(run_root / "base_embed" / "records" / "embed_record.json")
    assert [variant["suffix"] for variant in manifest_obj["variants"]] == ["GEO-on", "GEO-off"]
    assert summary_obj["variant_count"] == 2
    assert [variant["variant_suffix"] for variant in summary_obj["variants"]] == ["GEO-on", "GEO-off"]


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

    assert any("RUN_TAG" in cell_source and "OUTPUT_ROOT" in cell_source for cell_source in joined_cell_sources)
    assert any("SELECTED_VARIANTS" in cell_source for cell_source in joined_cell_sources)
    assert any("run_paper_ablation_workflow.py" in cell_source for cell_source in joined_cell_sources)
    assert any("ablation_compare_summary.json" in cell_source for cell_source in joined_cell_sources)
