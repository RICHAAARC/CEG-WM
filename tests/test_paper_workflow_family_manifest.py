"""
File purpose: Unit tests for PW00 family manifest generation stability.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from main.evaluation import attack_plan, protocol_loader
from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest
from paper_workflow.scripts.pw_common import (
    build_attack_condition_catalog,
    build_source_event_grid,
    build_source_split_plan,
    load_pw_matrix_config,
    read_jsonl,
    resolve_pw_matrix_settings,
)
from scripts.notebook_runtime_common import REPO_ROOT, build_repo_import_subprocess_env, load_yaml_mapping, normalize_path_value


DEFAULT_PW_BASE_CONFIG_PATH = (REPO_ROOT / "paper_workflow" / "configs" / "pw_base.yaml").resolve()
DEFAULT_PW_MATRIX_CONFIG_PATH = (REPO_ROOT / "paper_workflow" / "configs" / "pw_matrix.yaml").resolve()
PILOT_PW_BASE_CONFIG_PATH = (REPO_ROOT / "paper_workflow" / "configs" / "pw_base_pilot.yaml").resolve()
PILOT_PW_MATRIX_CONFIG_PATH = (REPO_ROOT / "paper_workflow" / "configs" / "pw_matrix_pilot.yaml").resolve()
RESCUE_PW_BASE_CONFIG_PATH = (REPO_ROOT / "paper_workflow" / "configs" / "pw_base_geometry_rescue_v1.yaml").resolve()
RESCUE_PW_MATRIX_CONFIG_PATH = (REPO_ROOT / "paper_workflow" / "configs" / "pw_matrix_geometry_rescue_v1.yaml").resolve()
SHARED_BENCHMARK_PW_BASE_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_base_geometry_shared_benchmark_v1.yaml"
).resolve()
SHARED_BENCHMARK_PW_MATRIX_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_matrix_geometry_shared_benchmark_v1.yaml"
).resolve()
SHARED_BENCHMARK_PROTOCOL_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_protocol_shared_hardneg_benchmark_v1.yaml"
).resolve()
INTERVAL_DISCOVERY_PW_BASE_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_base_geometry_interval_discovery_v1.yaml"
).resolve()
INTERVAL_DISCOVERY_PW_MATRIX_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_matrix_geometry_interval_discovery_v1.yaml"
).resolve()
INTERVAL_DISCOVERY_V2_PW_BASE_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_base_geometry_interval_discovery_v2.yaml"
).resolve()
INTERVAL_DISCOVERY_V2_PW_MATRIX_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_matrix_geometry_interval_discovery_v2.yaml"
).resolve()
INTERVAL_DISCOVERY_V2_PROTOCOL_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_protocol_geometry_interval_discovery_v2.yaml"
).resolve()
GEOMETRY_MIX_PW_BASE_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_base_geometry_mix.yaml"
).resolve()
GEOMETRY_MIX_PW_MATRIX_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_matrix_geometry_mix.yaml"
).resolve()
GEOMETRY_MIX_PROTOCOL_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_protocol_geometry_mix.yaml"
).resolve()
GEOMETRY_MIX_V2_PW_BASE_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_base_geometry_mix_v2.yaml"
).resolve()
GEOMETRY_MIX_V2_PW_MATRIX_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_matrix_geometry_mix_v2.yaml"
).resolve()
GEOMETRY_MIX_V2_PROTOCOL_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_protocol_geometry_mix_v2.yaml"
).resolve()
GEOMETRY_MIX_V3_PW_BASE_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_base_geometry_mix_v3.yaml"
).resolve()
GEOMETRY_MIX_V3_PW_MATRIX_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_matrix_geometry_mix_v3.yaml"
).resolve()
GEOMETRY_MIX_V3_PROTOCOL_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_protocol_geometry_mix_v3.yaml"
).resolve()
GEOMETRY_MIX_V4_PW_BASE_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_base_geometry_mix_v4.yaml"
).resolve()
GEOMETRY_MIX_V4_PW_MATRIX_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_matrix_geometry_mix_v4.yaml"
).resolve()
GEOMETRY_MIX_V4_PROTOCOL_CONFIG_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_protocol_geometry_mix_v4.yaml"
).resolve()


def test_pw00_builds_stable_event_grid_and_shard_plan(tmp_path: Path) -> None:
    """
    Build PW00 twice and verify deterministic outputs.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text("prompt alpha\nprompt beta\n", encoding="utf-8")

    first_summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="family_pw00_demo",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
    )
    second_summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="family_pw00_demo",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
    )

    first_event_grid_path = Path(str(first_summary["source_event_grid_path"]))
    first_shard_plan_path = Path(str(first_summary["source_shard_plan_path"]))
    first_split_plan_path = Path(str(first_summary["source_split_plan_path"]))
    first_wrong_event_challenge_plan_path = Path(str(first_summary["wrong_event_attestation_challenge_plan_path"]))
    second_event_grid_path = Path(str(second_summary["source_event_grid_path"]))
    second_shard_plan_path = Path(str(second_summary["source_shard_plan_path"]))
    second_split_plan_path = Path(str(second_summary["source_split_plan_path"]))
    second_wrong_event_challenge_plan_path = Path(str(second_summary["wrong_event_attestation_challenge_plan_path"]))

    assert first_event_grid_path.read_text(encoding="utf-8") == second_event_grid_path.read_text(encoding="utf-8")
    assert json.loads(first_shard_plan_path.read_text(encoding="utf-8")) == json.loads(
        second_shard_plan_path.read_text(encoding="utf-8")
    )
    assert json.loads(first_split_plan_path.read_text(encoding="utf-8")) == json.loads(
        second_split_plan_path.read_text(encoding="utf-8")
    )
    assert first_wrong_event_challenge_plan_path.exists()
    assert second_wrong_event_challenge_plan_path.exists()

    event_rows = read_jsonl(first_event_grid_path)
    assert len(event_rows) == 12
    assert [row["event_index"] for row in event_rows] == list(range(12))
    assert [row["sample_role"] for row in event_rows] == [
        "positive_source",
        "positive_source",
        "positive_source",
        "positive_source",
        "clean_negative",
        "clean_negative",
        "clean_negative",
        "clean_negative",
        "planner_conditioned_control_negative",
        "planner_conditioned_control_negative",
        "planner_conditioned_control_negative",
        "planner_conditioned_control_negative",
    ]
    assert len({row["event_id"] for row in event_rows}) == len(event_rows)

    family_manifest_path = Path(str(first_summary["paper_eval_family_manifest_path"]))
    family_manifest = json.loads(family_manifest_path.read_text(encoding="utf-8"))
    wrong_event_challenge_plan = json.loads(first_wrong_event_challenge_plan_path.read_text(encoding="utf-8"))
    method_identity_snapshot_path = Path(str(family_manifest["paths"]["method_identity_snapshot"]))
    method_identity_snapshot = json.loads(method_identity_snapshot_path.read_text(encoding="utf-8"))
    assert family_manifest["sample_roles"]["active"] == [
        "positive_source",
        "clean_negative",
        "planner_conditioned_control_negative",
    ]
    assert family_manifest["sample_roles"]["reserved"] == ["attacked_positive", "attacked_negative"]
    assert family_manifest["stage_boundary"]["implemented"] == ["PW00", "PW01", "PW02", "PW03", "PW04", "PW05"]
    assert family_manifest["stage_boundary"]["excluded"] == []
    assert family_manifest["source_parameters"]["seed_list"] == [0, 7]
    assert family_manifest["source_parameters"]["calibration_fraction"] == 0.5
    assert family_manifest["source_parameters"]["source_shard_count"] == 3
    assert family_manifest["attack_parameters"]["attack_shard_count"] == 3
    assert family_manifest["attack_parameters"]["materialization_profile"] == "matrix_defined_concrete_conditions"
    assert family_manifest["attack_parameters"]["matrix_profile"] == "family_x_severity_v1"
    assert family_manifest["attack_parameters"]["system_event_count_sweep"]["repeat_count"] == 64
    assert family_manifest["attack_parameters"]["wrong_event_attestation_challenge_plan_frozen"] is True
    assert family_manifest["attack_parameters"]["wrong_event_challenge_parent_event_count"] == 4
    assert family_manifest["attack_parameters"]["wrong_event_challenge_available_assignment_count"] == 4
    assert family_manifest["attack_parameters"]["severity_metadata_frozen"] is True
    assert family_manifest["attack_parameters"]["severity_axis_kind"] == "family_local"
    assert family_manifest["attack_parameters"]["severity_available_family_count"] > 0
    assert family_manifest["attack_plan"]["attack_shard_count"] == 3
    assert family_manifest["attack_plan"]["wrong_event_attestation_challenge_plan_frozen"] is True
    assert family_manifest["attack_plan"]["wrong_event_attestation_challenge_plan_policy"] == "cyclic_next_positive_parent_event_by_event_index"
    assert family_manifest["attack_plan"]["wrong_event_challenge_parent_event_count"] == 4
    assert family_manifest["attack_plan"]["wrong_event_challenge_available_assignment_count"] == 4
    assert family_manifest["attack_plan"]["severity_metadata_frozen"] is True
    assert family_manifest["attack_plan"]["severity_status_counts"]["ok"] > 0
    assert family_manifest["pw_base_config_path"] == normalize_path_value(DEFAULT_PW_BASE_CONFIG_PATH)
    assert family_manifest["pw_matrix_config_path"] == normalize_path_value(DEFAULT_PW_MATRIX_CONFIG_PATH)
    assert family_manifest["counts"]["positive_source_event_count"] == 4
    assert family_manifest["counts"]["clean_negative_event_count"] == 4
    assert family_manifest["counts"]["planner_conditioned_control_negative_event_count"] == 4
    assert family_manifest["counts"]["wrong_event_challenge_parent_event_count"] == 4
    assert family_manifest["counts"]["calibration_event_count"] == 4
    assert family_manifest["counts"]["evaluate_event_count"] == 4
    assert family_manifest["counts"]["control_calibration_event_count"] == 2
    assert family_manifest["counts"]["control_evaluate_event_count"] == 2
    assert family_manifest["paths"]["wrong_event_attestation_challenge_plan"] == first_summary[
        "wrong_event_attestation_challenge_plan_path"
    ]
    assert family_manifest["source_truth_stage"] == "PW01_Source_Event_Shards"
    assert wrong_event_challenge_plan["plan_policy"] == "cyclic_next_positive_parent_event_by_event_index"
    assert wrong_event_challenge_plan["positive_parent_event_count"] == 4
    assert wrong_event_challenge_plan["available_assignment_count"] == 4
    assert len(wrong_event_challenge_plan["rows"]) == 4
    assert all(row["status"] == "ready" for row in wrong_event_challenge_plan["rows"])
    assert all(
        row["parent_event_id"] != row["challenge_parent_event_id"]
        for row in wrong_event_challenge_plan["rows"]
    )
    assert method_identity_snapshot["source_truth_stage"] == "PW01_Source_Event_Shards"
    assert method_identity_snapshot["source_alignment_reference_files"] == [
        "paper_workflow/configs/pw_base.yaml",
        "paper_workflow/configs/pw_matrix.yaml",
        "paper_workflow/scripts/pw_common.py",
        "paper_workflow/scripts/pw00_build_family_manifest.py",
        "paper_workflow/scripts/pw01_stage_runtime_helpers.py",
        "paper_workflow/scripts/pw01_run_source_event_shard.py",
        "paper_workflow/notebook/PW00_Paper_Eval_Family_Manifest.ipynb",
        "paper_workflow/notebook/PW01_Source_Event_Shards.ipynb",
        "scripts/notebook_runtime_common.py",
        "configs/default.yaml",
    ]
    assert first_summary["source_shard_count"] == 3
    assert first_summary["attack_shard_count"] == 3
    assert first_summary["materialization_profile"] == "matrix_defined_concrete_conditions"
    assert first_summary["matrix_profile"] == "family_x_severity_v1"
    assert first_summary["pw_base_config_path"] == normalize_path_value(DEFAULT_PW_BASE_CONFIG_PATH)
    assert first_summary["pw_matrix_config_path"] == normalize_path_value(DEFAULT_PW_MATRIX_CONFIG_PATH)
    assert first_summary["wrong_event_challenge_parent_event_count"] == 4
    assert first_summary["wrong_event_challenge_available_assignment_count"] == 4
    assert Path(str(first_summary["wrong_event_attestation_challenge_plan_path"])) == first_wrong_event_challenge_plan_path
    assert first_summary["severity_metadata_frozen"] is True
    assert first_summary["severity_axis_kind"] == "family_local"
    assert first_summary["severity_status_counts"]["ok"] > 0
    assert first_summary["severity_available_family_count"] > 0

    attack_condition_catalog = build_attack_condition_catalog()
    rotate_condition_keys = [
        row["attack_condition_key"]
        for row in attack_condition_catalog
        if row["attack_condition_base_key"] == "rotate::v1"
    ]
    assert rotate_condition_keys == [
        "rotate::v1::sev00",
        "rotate::v1::sev01",
        "rotate::v1::sev02",
    ]


def test_pw00_can_freeze_independent_attack_shard_count(tmp_path: Path) -> None:
    """
    Verify PW00 can freeze an attack shard count independent from source shards.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text("prompt alpha\nprompt beta\n", encoding="utf-8")

    summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="family_pw00_attack_decoupled",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
        attack_shard_count=5,
    )

    family_manifest_path = Path(str(summary["paper_eval_family_manifest_path"]))
    attack_shard_plan_path = Path(str(summary["attack_shard_plan_path"]))
    source_shard_plan_path = Path(str(summary["source_shard_plan_path"]))

    family_manifest = json.loads(family_manifest_path.read_text(encoding="utf-8"))
    attack_shard_plan = json.loads(attack_shard_plan_path.read_text(encoding="utf-8"))
    source_shard_plan = json.loads(source_shard_plan_path.read_text(encoding="utf-8"))

    assert family_manifest["source_parameters"]["source_shard_count"] == 3
    assert family_manifest["attack_parameters"]["attack_shard_count"] == 5
    assert family_manifest["attack_parameters"]["wrong_event_attestation_challenge_plan_frozen"] is True
    assert family_manifest["attack_plan"]["attack_shard_count"] == 5
    assert family_manifest["attack_plan"]["wrong_event_attestation_challenge_plan_policy"] == "cyclic_next_positive_parent_event_by_event_index"
    assert attack_shard_plan["attack_shard_count"] == 5
    assert len(attack_shard_plan["shards"]) == 5
    assert source_shard_plan["source_shard_count"] == 3
    assert summary["source_shard_count"] == 3
    assert summary["attack_shard_count"] == 5


def test_pw00_cli_wrapper_passes_explicit_attack_shard_count(tmp_path: Path) -> None:
    """
    Verify the PW00 CLI wrapper exposes and forwards attack_shard_count.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text("prompt alpha\nprompt beta\n", encoding="utf-8")

    script_path = REPO_ROOT / "paper_workflow" / "scripts" / "PW00_Paper_Eval_Family_Manifest.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--drive-project-root",
            str(drive_project_root),
            "--family-id",
            "family_pw00_cli_attack_decoupled",
            "--prompt-file",
            str(prompt_file),
            "--seed-list",
            "[0, 7]",
            "--source-shard-count",
            "3",
            "--attack-shard-count",
            "5",
        ],
        cwd=REPO_ROOT,
        env=build_repo_import_subprocess_env(repo_root=REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, result.stderr

    summary = json.loads(result.stdout)
    family_manifest = json.loads(Path(str(summary["paper_eval_family_manifest_path"])).read_text(encoding="utf-8"))
    attack_shard_plan = json.loads(Path(str(summary["attack_shard_plan_path"])).read_text(encoding="utf-8"))

    assert summary["source_shard_count"] == 3
    assert summary["attack_shard_count"] == 5
    assert family_manifest["source_parameters"]["source_shard_count"] == 3
    assert family_manifest["attack_parameters"]["attack_shard_count"] == 5
    assert family_manifest["attack_plan"]["attack_shard_count"] == 5
    assert attack_shard_plan["attack_shard_count"] == 5


def test_pw00_cli_defaults_to_formal_pw_base_config_when_base_arg_is_omitted(tmp_path: Path) -> None:
    """
    Verify the PW00 CLI defaults to the formal pw_base config when the
    optional base-config argument is omitted.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text("prompt alpha\nprompt beta\n", encoding="utf-8")

    script_path = REPO_ROOT / "paper_workflow" / "scripts" / "PW00_Paper_Eval_Family_Manifest.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--drive-project-root",
            str(drive_project_root),
            "--family-id",
            "family_pw00_cli_default_base",
            "--prompt-file",
            str(prompt_file),
            "--seed-list",
            "[0, 7]",
            "--source-shard-count",
            "3",
        ],
        cwd=REPO_ROOT,
        env=build_repo_import_subprocess_env(repo_root=REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, result.stderr

    summary = json.loads(result.stdout)
    family_manifest = json.loads(Path(str(summary["paper_eval_family_manifest_path"])).read_text(encoding="utf-8"))

    assert summary["pw_base_config_path"] == normalize_path_value(DEFAULT_PW_BASE_CONFIG_PATH)
    assert summary["pw_matrix_config_path"] == normalize_path_value(DEFAULT_PW_MATRIX_CONFIG_PATH)
    assert family_manifest["pw_base_config_path"] == normalize_path_value(DEFAULT_PW_BASE_CONFIG_PATH)
    assert family_manifest["pw_matrix_config_path"] == normalize_path_value(DEFAULT_PW_MATRIX_CONFIG_PATH)


def test_pw00_records_pilot_base_and_matrix_paths_and_freezes_pilot_geometry_policy(tmp_path: Path) -> None:
    """
    Verify PW00 can bind the pilot pw_base config and freeze the pilot matrix
    paths and geometry policy into summary artifacts.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text("prompt alpha\nprompt beta\n", encoding="utf-8")

    summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="family_pw00_pilot_bound",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
        pw_base_config_path=PILOT_PW_BASE_CONFIG_PATH,
    )

    family_manifest = json.loads(Path(str(summary["paper_eval_family_manifest_path"])).read_text(encoding="utf-8"))
    method_identity_snapshot = json.loads(
        Path(str(family_manifest["paths"]["method_identity_snapshot"])).read_text(encoding="utf-8")
    )
    geometry_optional_claim_plan = json.loads(
        Path(str(summary["geometry_optional_claim_plan_path"])).read_text(encoding="utf-8")
    )
    attack_event_rows = read_jsonl(Path(str(summary["attack_event_grid_path"])))
    pilot_matrix_settings = resolve_pw_matrix_settings(
        load_pw_matrix_config(matrix_config_path=PILOT_PW_MATRIX_CONFIG_PATH)
    )

    assert summary["pw_base_config_path"] == normalize_path_value(PILOT_PW_BASE_CONFIG_PATH)
    assert summary["pw_matrix_config_path"] == normalize_path_value(PILOT_PW_MATRIX_CONFIG_PATH)
    assert family_manifest["pw_base_config_path"] == normalize_path_value(PILOT_PW_BASE_CONFIG_PATH)
    assert family_manifest["pw_matrix_config_path"] == normalize_path_value(PILOT_PW_MATRIX_CONFIG_PATH)
    assert method_identity_snapshot["source_alignment_reference_files"] == [
        "paper_workflow/configs/pw_base_pilot.yaml",
        "paper_workflow/configs/pw_matrix_pilot.yaml",
        "paper_workflow/scripts/pw_common.py",
        "paper_workflow/scripts/pw00_build_family_manifest.py",
        "paper_workflow/scripts/pw01_stage_runtime_helpers.py",
        "paper_workflow/scripts/pw01_run_source_event_shard.py",
        "paper_workflow/notebook/PW00_Paper_Eval_Family_Manifest.ipynb",
        "paper_workflow/notebook/PW01_Source_Event_Shards.ipynb",
        "scripts/notebook_runtime_common.py",
        "configs/default.yaml",
    ]
    assert summary["matrix_profile"] == "family_x_severity_pilot_v1"
    assert family_manifest["attack_parameters"]["matrix_profile"] == "family_x_severity_pilot_v1"
    assert summary["system_event_count_sweep"]["repeat_count"] == 48
    assert family_manifest["attack_parameters"]["system_event_count_sweep"]["repeat_count"] == 48
    assert summary["geometry_optional_claim_boundary_abs_margin_min"] == pytest.approx(0.01)
    assert summary["geometry_optional_claim_boundary_abs_margin_max"] == pytest.approx(0.25)
    assert family_manifest["attack_parameters"]["geometry_optional_claim_boundary_abs_margin_min"] == pytest.approx(0.01)
    assert family_manifest["attack_parameters"]["geometry_optional_claim_boundary_abs_margin_max"] == pytest.approx(0.25)
    assert pilot_matrix_settings["geometry_optional_claim"]["boundary_abs_margin_min"] == pytest.approx(0.01)
    assert pilot_matrix_settings["geometry_optional_claim"]["boundary_abs_margin_max"] == pytest.approx(0.25)
    assert set(pilot_matrix_settings["geometry_optional_claim"]["candidate_attack_families"]) == {
        "rotate",
        "resize",
        "crop",
        "composite",
    }
    assert "translate" not in geometry_optional_claim_plan["geometry_rescue_candidate_attack_families"]
    assert geometry_optional_claim_plan["boundary_abs_margin_min"] == pytest.approx(0.01)
    assert geometry_optional_claim_plan["boundary_abs_margin_max"] == pytest.approx(0.25)
    assert all(
        row["geometry_rescue_candidate"] is False
        for row in attack_event_rows
        if row["attack_family"] == "translate"
    )


def test_attack_matrix_validation_rejects_unknown_geometry_candidate_set() -> None:
    """
    Verify invalid matrix config fails fast before PW00 attack catalog materialization.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(ValueError, match="candidate_attack_set"):
        build_attack_condition_catalog(
            matrix_cfg={
                "matrix_profile": "family_x_severity_v1",
                "matrix_version": "pw_attack_matrix_v1",
                "materialization_profile": "protocol_list_cartesian_per_condition",
                "attack_sets": {
                    "general_attacks": {
                        "families": ["rotate"],
                    }
                },
                "geometry_optional_claim": {
                    "candidate_attack_set": "missing_geometry_set",
                    "boundary_rule_version": "geometry_optional_claim_boundary_band_v2",
                    "boundary_metric": "abs_content_margin",
                    "boundary_abs_margin_min": 0.005,
                    "boundary_abs_margin_max": 0.05,
                },
                "system_event_count_sweep": {
                    "event_counts": [1, 2],
                    "repeat_count": 2,
                    "random_seed": 1,
                },
            }
        )


def test_attack_protocol_geometry_rescue_versions_are_loadable() -> None:
    """
    Verify append-only geometry-rescue protocol versions are loadable.

    Args:
        None.

    Returns:
        None.
    """
    protocol_spec = protocol_loader.load_attack_protocol_spec({})
    params_versions = protocol_spec.get("params_versions", {})
    generated_plan = attack_plan.generate_attack_plan(protocol_spec)

    expected_condition_keys = {
        "rotate::v2",
        "resize::v2",
        "crop::v2",
        "composite::rotate_resize_jpeg_v1",
        "composite::rotate_resize_jpeg_v2",
        "composite::crop_resize_v1",
    }

    assert expected_condition_keys.issubset(set(params_versions))
    assert expected_condition_keys.issubset(set(generated_plan.conditions))


def test_geometry_rescue_matrix_materializes_expected_condition_subset() -> None:
    """
    Verify the geometry-rescue matrix parses and materializes the expected concrete conditions.

    Args:
        None.

    Returns:
        None.
    """
    matrix_cfg = load_pw_matrix_config(matrix_config_path=RESCUE_PW_MATRIX_CONFIG_PATH)
    matrix_settings = resolve_pw_matrix_settings(matrix_cfg)
    attack_condition_catalog = build_attack_condition_catalog(matrix_cfg=matrix_cfg)

    expected_families = ["rotate", "resize", "crop", "composite"]
    expected_condition_keys = [
        "composite::crop_resize_v1::sev00",
        "composite::rotate_resize_jpeg_v1::sev00",
        "composite::rotate_resize_jpeg_v2::sev00",
        "crop::v2::sev00",
        "crop::v2::sev01",
        "crop::v2::sev02",
        "resize::v2::sev00",
        "resize::v2::sev01",
        "resize::v2::sev02",
        "rotate::v2::sev00",
        "rotate::v2::sev01",
        "rotate::v2::sev02",
    ]

    assert matrix_settings["matrix_profile"] == "geometry_rescue_slice_v1"
    assert matrix_settings["matrix_version"] == "pw_attack_matrix_geometry_rescue_v1"
    assert matrix_settings["materialization_profile"] == "matrix_defined_concrete_conditions"
    assert matrix_settings["attack_sets"]["general_attacks"] == expected_families
    assert matrix_settings["attack_sets"]["geometry_rescue_candidates"] == expected_families
    assert matrix_settings["geometry_optional_claim"]["candidate_attack_set"] == "geometry_rescue_candidates"
    assert matrix_settings["geometry_optional_claim"]["candidate_attack_families"] == expected_families
    assert matrix_settings["geometry_optional_claim"]["boundary_rule_version"] == "geometry_optional_claim_boundary_band_v3"
    assert matrix_settings["geometry_optional_claim"]["boundary_metric"] == "abs_content_margin"
    assert matrix_settings["geometry_optional_claim"]["boundary_abs_margin_min"] == pytest.approx(0.01)
    assert matrix_settings["geometry_optional_claim"]["boundary_abs_margin_max"] == pytest.approx(0.35)
    assert [row["attack_condition_key"] for row in attack_condition_catalog] == expected_condition_keys


def test_pw00_binds_geometry_rescue_base_and_freezes_new_reference_pair(tmp_path: Path) -> None:
    """
    Verify PW00 can bind the geometry-rescue base config and freeze the rescue reference pair.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text("prompt alpha\nprompt beta\n", encoding="utf-8")

    summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="family_pw00_geometry_rescue_v1",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
        pw_base_config_path=RESCUE_PW_BASE_CONFIG_PATH,
    )

    family_manifest = json.loads(Path(str(summary["paper_eval_family_manifest_path"])).read_text(encoding="utf-8"))
    method_identity_snapshot = json.loads(
        Path(str(family_manifest["paths"]["method_identity_snapshot"])).read_text(encoding="utf-8")
    )
    geometry_optional_claim_plan = json.loads(
        Path(str(summary["geometry_optional_claim_plan_path"])).read_text(encoding="utf-8")
    )

    assert summary["pw_base_config_path"] == normalize_path_value(RESCUE_PW_BASE_CONFIG_PATH)
    assert summary["pw_matrix_config_path"] == normalize_path_value(RESCUE_PW_MATRIX_CONFIG_PATH)
    assert family_manifest["pw_base_config_path"] == normalize_path_value(RESCUE_PW_BASE_CONFIG_PATH)
    assert family_manifest["pw_matrix_config_path"] == normalize_path_value(RESCUE_PW_MATRIX_CONFIG_PATH)
    assert summary["matrix_profile"] == "geometry_rescue_slice_v1"
    assert summary["attack_condition_count"] == 12
    assert summary["system_event_count_sweep"]["repeat_count"] == 48
    assert summary["geometry_optional_claim_boundary_abs_margin_min"] == pytest.approx(0.01)
    assert summary["geometry_optional_claim_boundary_abs_margin_max"] == pytest.approx(0.35)
    assert method_identity_snapshot["source_alignment_reference_files"] == [
        "paper_workflow/configs/pw_base_geometry_rescue_v1.yaml",
        "paper_workflow/configs/pw_matrix_geometry_rescue_v1.yaml",
        "paper_workflow/scripts/pw_common.py",
        "paper_workflow/scripts/pw00_build_family_manifest.py",
        "paper_workflow/scripts/pw01_stage_runtime_helpers.py",
        "paper_workflow/scripts/pw01_run_source_event_shard.py",
        "paper_workflow/notebook/PW00_Paper_Eval_Family_Manifest.ipynb",
        "paper_workflow/notebook/PW01_Source_Event_Shards.ipynb",
        "scripts/notebook_runtime_common.py",
        "configs/default.yaml",
    ]
    assert geometry_optional_claim_plan["geometry_rescue_candidate_attack_families"] == [
        "rotate",
        "resize",
        "crop",
        "composite",
    ]
    assert geometry_optional_claim_plan["boundary_abs_margin_min"] == pytest.approx(0.01)
    assert geometry_optional_claim_plan["boundary_abs_margin_max"] == pytest.approx(0.35)


def test_shared_benchmark_matrix_uses_stronger_whitelisted_geometry_ladder_than_rescue() -> None:
    """
    Verify the shared benchmark matrix stays whitelist-compliant and is stricter than rescue.

    Args:
        None.

    Returns:
        None.
    """
    rescue_matrix_cfg = load_pw_matrix_config(matrix_config_path=RESCUE_PW_MATRIX_CONFIG_PATH)
    rescue_matrix_settings = resolve_pw_matrix_settings(rescue_matrix_cfg)
    shared_matrix_cfg = load_pw_matrix_config(matrix_config_path=SHARED_BENCHMARK_PW_MATRIX_CONFIG_PATH)
    shared_matrix_settings = resolve_pw_matrix_settings(shared_matrix_cfg)
    shared_catalog = build_attack_condition_catalog(matrix_cfg=shared_matrix_cfg)
    shared_params_by_key = {
        row["attack_condition_key"]: row["attack_params"]
        for row in shared_catalog
    }

    assert [row["attack_condition_key"] for row in shared_catalog] == [
        "composite::crop_resize_v1::sev00",
        "composite::rotate_resize_jpeg_v2::sev00",
        "composite::rotate_resize_jpeg_v2::sev01",
        "crop::v2::sev00",
        "crop::v2::sev01",
        "crop::v2::sev02",
        "resize::v2::sev00",
        "resize::v2::sev01",
        "resize::v2::sev02",
        "rotate::v2::sev00",
        "rotate::v2::sev01",
        "rotate::v2::sev02",
    ]
    assert shared_params_by_key["crop::v2::sev00"] == {"crop_ratios": 0.75}
    assert shared_params_by_key["crop::v2::sev01"] == {"crop_ratios": 0.65}
    assert shared_params_by_key["crop::v2::sev02"] == {"crop_ratios": 0.65}
    assert shared_params_by_key["resize::v2::sev00"] == {"scale_factors": 0.7}
    assert shared_params_by_key["resize::v2::sev01"] == {"scale_factors": 0.6}
    assert shared_params_by_key["resize::v2::sev02"] == {"scale_factors": 0.6}
    assert shared_params_by_key["rotate::v2::sev00"] == {"degrees": 18}
    assert shared_params_by_key["rotate::v2::sev01"] == {"degrees": 24}
    assert shared_params_by_key["rotate::v2::sev02"] == {"degrees": 24}
    assert shared_matrix_settings["geometry_optional_claim"]["boundary_abs_margin_max"] == pytest.approx(0.3)
    assert shared_matrix_settings["geometry_optional_claim"]["boundary_abs_margin_max"] < rescue_matrix_settings[
        "geometry_optional_claim"
    ]["boundary_abs_margin_max"]


def test_attack_protocol_geometry_interval_discovery_versions_are_loadable() -> None:
    """
    Verify append-only geometry-interval-discovery protocol versions are loadable.

    Args:
        None.

    Returns:
        None.
    """
    protocol_spec = protocol_loader.load_attack_protocol_spec({})
    params_versions = protocol_spec.get("params_versions", {})
    generated_plan = attack_plan.generate_attack_plan(protocol_spec)

    expected_condition_keys = {
        "rotate::v3",
        "resize::v3",
        "crop::v3",
        "composite::rotate_resize_jpeg_v3",
        "composite::crop_resize_v2",
    }

    assert expected_condition_keys.issubset(set(params_versions))
    assert expected_condition_keys.issubset(set(generated_plan.conditions))


def test_geometry_interval_discovery_matrix_materializes_expected_condition_subset() -> None:
    """
    Verify the geometry-interval-discovery matrix parses and materializes the expected dense conditions.

    Args:
        None.

    Returns:
        None.
    """
    discovery_base_cfg = load_yaml_mapping(INTERVAL_DISCOVERY_PW_BASE_CONFIG_PATH)
    matrix_cfg = load_pw_matrix_config(matrix_config_path=INTERVAL_DISCOVERY_PW_MATRIX_CONFIG_PATH)
    matrix_settings = resolve_pw_matrix_settings(matrix_cfg)
    attack_condition_catalog = build_attack_condition_catalog(matrix_cfg=matrix_cfg)
    catalog_params = {
        row["attack_condition_key"]: row["attack_params"]
        for row in attack_condition_catalog
    }

    expected_families = ["rotate", "resize", "crop", "composite"]
    expected_condition_keys = [
        "composite::crop_resize_v2::sev00",
        "composite::rotate_resize_jpeg_v2::sev00",
        "composite::rotate_resize_jpeg_v3::sev00",
        "crop::v3::sev00",
        "crop::v3::sev01",
        "crop::v3::sev02",
        "crop::v3::sev03",
        "resize::v3::sev00",
        "resize::v3::sev01",
        "resize::v3::sev02",
        "resize::v3::sev03",
        "rotate::v3::sev00",
        "rotate::v3::sev01",
        "rotate::v3::sev02",
        "rotate::v3::sev03",
        "rotate::v3::sev04",
    ]

    assert discovery_base_cfg["benchmark_mode"] == "geometry_interval_discovery"
    assert discovery_base_cfg["benchmark_mode_version"] == "geometry_interval_discovery_v1"
    assert discovery_base_cfg["matrix_config_path"] == "paper_workflow/configs/pw_matrix_geometry_interval_discovery_v1.yaml"
    assert discovery_base_cfg["benchmark_protocol_config_path"] == "paper_workflow/configs/pw_protocol_shared_hardneg_benchmark_v1.yaml"
    assert discovery_base_cfg["sample_roles"]["active"] == [
        "positive_source",
        "clean_negative",
        "planner_conditioned_control_negative",
    ]
    assert discovery_base_cfg["sample_roles"]["reserved"] == ["attacked_positive", "attacked_negative"]
    assert matrix_settings["matrix_profile"] == "geometry_interval_discovery_v1"
    assert matrix_settings["matrix_version"] == "pw_attack_matrix_geometry_interval_discovery_v1"
    assert matrix_settings["materialization_profile"] == "matrix_defined_concrete_conditions"
    assert matrix_settings["attack_sets"]["general_attacks"] == expected_families
    assert matrix_settings["attack_sets"]["geometry_rescue_candidates"] == expected_families
    assert matrix_settings["geometry_optional_claim"]["candidate_attack_set"] == "geometry_rescue_candidates"
    assert matrix_settings["geometry_optional_claim"]["candidate_attack_families"] == expected_families
    assert matrix_settings["geometry_optional_claim"]["boundary_abs_margin_min"] == pytest.approx(0.01)
    assert matrix_settings["geometry_optional_claim"]["boundary_abs_margin_max"] == pytest.approx(0.3)
    assert [row["attack_condition_key"] for row in attack_condition_catalog] == expected_condition_keys
    assert catalog_params["crop::v3::sev00"] == {"crop_ratios": 0.8}
    assert catalog_params["crop::v3::sev01"] == {"crop_ratios": 0.75}
    assert catalog_params["crop::v3::sev02"] == {"crop_ratios": 0.7}
    assert catalog_params["crop::v3::sev03"] == {"crop_ratios": 0.65}
    assert catalog_params["resize::v3::sev00"] == {"scale_factors": 0.75}
    assert catalog_params["resize::v3::sev01"] == {"scale_factors": 0.7}
    assert catalog_params["resize::v3::sev02"] == {"scale_factors": 0.65}
    assert catalog_params["resize::v3::sev03"] == {"scale_factors": 0.6}
    assert catalog_params["rotate::v3::sev00"] == {"degrees": 18}
    assert catalog_params["rotate::v3::sev01"] == {"degrees": 20}
    assert catalog_params["rotate::v3::sev02"] == {"degrees": 22}
    assert catalog_params["rotate::v3::sev03"] == {"degrees": 24}
    assert catalog_params["rotate::v3::sev04"] == {"degrees": 27}


def test_pw00_binds_geometry_interval_discovery_base_and_shared_protocol(tmp_path: Path) -> None:
    """
    Verify PW00 can bind the geometry-interval-discovery base config and append shared benchmark provenance.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text("prompt alpha\nprompt beta\n", encoding="utf-8")

    summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="paper_eval_family_geometry_interval_discovery_v1",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
        pw_base_config_path=INTERVAL_DISCOVERY_PW_BASE_CONFIG_PATH,
    )

    family_manifest = json.loads(Path(str(summary["paper_eval_family_manifest_path"])).read_text(encoding="utf-8"))
    method_identity_snapshot = json.loads(
        Path(str(family_manifest["paths"]["method_identity_snapshot"])).read_text(encoding="utf-8")
    )

    assert summary["pw_base_config_path"] == normalize_path_value(INTERVAL_DISCOVERY_PW_BASE_CONFIG_PATH)
    assert summary["pw_matrix_config_path"] == normalize_path_value(INTERVAL_DISCOVERY_PW_MATRIX_CONFIG_PATH)
    assert summary["benchmark_protocol_config_path"] == normalize_path_value(SHARED_BENCHMARK_PROTOCOL_CONFIG_PATH)
    assert summary["benchmark_protocol"]["protocol_id"] == "shared_hardneg_geometry_benchmark_v1"
    assert summary["matrix_profile"] == "geometry_interval_discovery_v1"
    assert summary["matrix_version"] == "pw_attack_matrix_geometry_interval_discovery_v1"
    assert summary["attack_condition_count"] == 16
    assert summary["source_shard_count"] == 3
    assert summary["attack_shard_count"] == 3
    assert summary["geometry_optional_claim_boundary_abs_margin_max"] == pytest.approx(0.3)
    assert family_manifest["family_id"] == "paper_eval_family_geometry_interval_discovery_v1"
    assert family_manifest["pw_base_config_path"] == normalize_path_value(INTERVAL_DISCOVERY_PW_BASE_CONFIG_PATH)
    assert family_manifest["pw_matrix_config_path"] == normalize_path_value(INTERVAL_DISCOVERY_PW_MATRIX_CONFIG_PATH)
    assert family_manifest["benchmark_protocol_config_path"] == normalize_path_value(
        SHARED_BENCHMARK_PROTOCOL_CONFIG_PATH
    )
    assert family_manifest["benchmark_protocol"]["protocol_id"] == "shared_hardneg_geometry_benchmark_v1"
    assert family_manifest["benchmark_protocol"]["score_pools"]["content_chain_score"]["evaluate_role_order"] == [
        "positive_source",
        "clean_negative",
    ]
    assert family_manifest["attack_parameters"]["matrix_profile"] == "geometry_interval_discovery_v1"
    assert family_manifest["attack_parameters"]["matrix_version"] == "pw_attack_matrix_geometry_interval_discovery_v1"
    assert family_manifest["attack_parameters"]["attack_condition_count"] == 16
    assert family_manifest["attack_parameters"]["geometry_optional_claim_boundary_abs_margin_max"] == pytest.approx(0.3)
    assert method_identity_snapshot["source_alignment_reference_files"] == [
        "paper_workflow/configs/pw_base_geometry_interval_discovery_v1.yaml",
        "paper_workflow/configs/pw_matrix_geometry_interval_discovery_v1.yaml",
        "paper_workflow/configs/pw_protocol_shared_hardneg_benchmark_v1.yaml",
        "paper_workflow/scripts/pw_common.py",
        "paper_workflow/scripts/pw00_build_family_manifest.py",
        "paper_workflow/scripts/pw01_stage_runtime_helpers.py",
        "paper_workflow/scripts/pw01_run_source_event_shard.py",
        "paper_workflow/scripts/pw02_merge_source_event_shards.py",
        "paper_workflow/notebook/PW00_Paper_Eval_Family_Manifest.ipynb",
        "paper_workflow/notebook/PW01_Source_Event_Shards.ipynb",
        "paper_workflow/notebook/PW02_Source_Merge_And_Global_Thresholds.ipynb",
        "scripts/notebook_runtime_common.py",
        "configs/default.yaml",
    ]


def test_attack_protocol_geometry_interval_discovery_v2_versions_are_loadable() -> None:
    """
    Verify append-only geometry-interval-discovery-v2 protocol versions are loadable.

    Args:
        None.

    Returns:
        None.
    """
    protocol_spec = protocol_loader.load_attack_protocol_spec({})
    params_versions = protocol_spec.get("params_versions", {})
    generated_plan = attack_plan.generate_attack_plan(protocol_spec)

    expected_condition_keys = {
        "rotate::v4",
        "resize::v4",
        "crop::v4",
        "composite::crop_resize_v3",
        "composite::rotate_resize_jpeg_v4",
        "composite::rotate_resize_jpeg_v5",
    }

    assert expected_condition_keys.issubset(set(params_versions))
    assert expected_condition_keys.issubset(set(generated_plan.conditions))


def test_geometry_interval_discovery_v2_matrix_materializes_expected_condition_subset() -> None:
    """
    Verify the geometry-interval-discovery-v2 matrix parses and materializes the denser conditions.

    Args:
        None.

    Returns:
        None.
    """
    discovery_base_cfg = load_yaml_mapping(INTERVAL_DISCOVERY_V2_PW_BASE_CONFIG_PATH)
    matrix_cfg = load_pw_matrix_config(matrix_config_path=INTERVAL_DISCOVERY_V2_PW_MATRIX_CONFIG_PATH)
    matrix_settings = resolve_pw_matrix_settings(matrix_cfg)
    attack_condition_catalog = build_attack_condition_catalog(matrix_cfg=matrix_cfg)
    catalog_params = {
        row["attack_condition_key"]: row["attack_params"]
        for row in attack_condition_catalog
    }

    expected_families = ["rotate", "resize", "crop", "composite"]
    expected_condition_keys = [
        "composite::crop_resize_v3::sev00",
        "composite::rotate_resize_jpeg_v4::sev00",
        "composite::rotate_resize_jpeg_v5::sev00",
        "crop::v4::sev00",
        "crop::v4::sev01",
        "crop::v4::sev02",
        "crop::v4::sev03",
        "crop::v4::sev04",
        "resize::v4::sev00",
        "resize::v4::sev01",
        "resize::v4::sev02",
        "resize::v4::sev03",
        "resize::v4::sev04",
        "rotate::v4::sev00",
        "rotate::v4::sev01",
        "rotate::v4::sev02",
        "rotate::v4::sev03",
        "rotate::v4::sev04",
    ]

    assert discovery_base_cfg["benchmark_mode"] == "geometry_interval_discovery"
    assert discovery_base_cfg["benchmark_mode_version"] == "geometry_interval_discovery_v2"
    assert discovery_base_cfg["matrix_config_path"] == "paper_workflow/configs/pw_matrix_geometry_interval_discovery_v2.yaml"
    assert discovery_base_cfg["benchmark_protocol_config_path"] == "paper_workflow/configs/pw_protocol_geometry_interval_discovery_v2.yaml"
    assert discovery_base_cfg["sample_roles"]["active"] == [
        "positive_source",
        "clean_negative",
        "planner_conditioned_control_negative",
    ]
    assert discovery_base_cfg["sample_roles"]["reserved"] == ["attacked_positive", "attacked_negative"]
    assert matrix_settings["matrix_profile"] == "geometry_interval_discovery_v2"
    assert matrix_settings["matrix_version"] == "pw_attack_matrix_geometry_interval_discovery_v2"
    assert matrix_settings["materialization_profile"] == "matrix_defined_concrete_conditions"
    assert matrix_settings["attack_sets"]["general_attacks"] == expected_families
    assert matrix_settings["attack_sets"]["geometry_rescue_candidates"] == expected_families
    assert matrix_settings["geometry_optional_claim"]["candidate_attack_set"] == "geometry_rescue_candidates"
    assert matrix_settings["geometry_optional_claim"]["candidate_attack_families"] == expected_families
    assert matrix_settings["geometry_optional_claim"]["boundary_abs_margin_min"] == pytest.approx(0.01)
    assert matrix_settings["geometry_optional_claim"]["boundary_abs_margin_max"] == pytest.approx(0.3)
    assert [row["attack_condition_key"] for row in attack_condition_catalog] == expected_condition_keys
    assert len(attack_condition_catalog) == 18
    assert catalog_params["crop::v4::sev00"] == {"crop_ratios": 0.8}
    assert catalog_params["crop::v4::sev01"] == {"crop_ratios": 0.75}
    assert catalog_params["crop::v4::sev02"] == {"crop_ratios": 0.7}
    assert catalog_params["crop::v4::sev03"] == {"crop_ratios": 0.675}
    assert catalog_params["crop::v4::sev04"] == {"crop_ratios": 0.65}
    assert catalog_params["resize::v4::sev00"] == {"scale_factors": 0.75}
    assert catalog_params["resize::v4::sev01"] == {"scale_factors": 0.7}
    assert catalog_params["resize::v4::sev02"] == {"scale_factors": 0.675}
    assert catalog_params["resize::v4::sev03"] == {"scale_factors": 0.65}
    assert catalog_params["resize::v4::sev04"] == {"scale_factors": 0.6}
    assert catalog_params["rotate::v4::sev00"] == {"degrees": 18}
    assert catalog_params["rotate::v4::sev01"] == {"degrees": 20}
    assert catalog_params["rotate::v4::sev02"] == {"degrees": 22}
    assert catalog_params["rotate::v4::sev03"] == {"degrees": 24}
    assert catalog_params["rotate::v4::sev04"] == {"degrees": 27}


def test_pw00_binds_geometry_interval_discovery_v2_base_and_shared_protocol(tmp_path: Path) -> None:
    """
    Verify PW00 can bind the geometry-interval-discovery-v2 base config and append discovery-v2 protocol provenance.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text("prompt alpha\nprompt beta\n", encoding="utf-8")

    summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="paper_eval_family_geometry_interval_discovery_v2",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
        pw_base_config_path=INTERVAL_DISCOVERY_V2_PW_BASE_CONFIG_PATH,
    )

    family_manifest = json.loads(Path(str(summary["paper_eval_family_manifest_path"])).read_text(encoding="utf-8"))
    method_identity_snapshot = json.loads(
        Path(str(family_manifest["paths"]["method_identity_snapshot"])).read_text(encoding="utf-8")
    )

    assert summary["pw_base_config_path"] == normalize_path_value(INTERVAL_DISCOVERY_V2_PW_BASE_CONFIG_PATH)
    assert summary["pw_matrix_config_path"] == normalize_path_value(INTERVAL_DISCOVERY_V2_PW_MATRIX_CONFIG_PATH)
    assert summary["benchmark_protocol_config_path"] == normalize_path_value(INTERVAL_DISCOVERY_V2_PROTOCOL_CONFIG_PATH)
    assert summary["benchmark_protocol"]["protocol_id"] == "geometry_interval_discovery_v2"
    assert summary["benchmark_protocol"]["protocol_family_id"] == "paper_eval_family_geometry_interval_discovery_v2"
    assert summary["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_profile"] == "geometry_interval_discovery_v2"
    assert summary["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_version"] == "pw_attack_matrix_geometry_interval_discovery_v2"
    assert summary["benchmark_provenance"]["protocol_id"] == "geometry_interval_discovery_v2"
    assert summary["benchmark_provenance"]["benchmark_protocol_config_path"] == normalize_path_value(
        INTERVAL_DISCOVERY_V2_PROTOCOL_CONFIG_PATH
    )
    assert summary["matrix_profile"] == "geometry_interval_discovery_v2"
    assert summary["matrix_version"] == "pw_attack_matrix_geometry_interval_discovery_v2"
    assert summary["attack_condition_count"] == 18
    assert summary["source_shard_count"] == 3
    assert summary["attack_shard_count"] == 3
    assert summary["geometry_optional_claim_boundary_abs_margin_max"] == pytest.approx(0.3)
    assert family_manifest["family_id"] == "paper_eval_family_geometry_interval_discovery_v2"
    assert family_manifest["pw_base_config_path"] == normalize_path_value(INTERVAL_DISCOVERY_V2_PW_BASE_CONFIG_PATH)
    assert family_manifest["pw_matrix_config_path"] == normalize_path_value(INTERVAL_DISCOVERY_V2_PW_MATRIX_CONFIG_PATH)
    assert family_manifest["benchmark_protocol_config_path"] == normalize_path_value(
        INTERVAL_DISCOVERY_V2_PROTOCOL_CONFIG_PATH
    )
    assert family_manifest["benchmark_protocol"]["protocol_id"] == "geometry_interval_discovery_v2"
    assert family_manifest["benchmark_protocol"]["protocol_family_id"] == "paper_eval_family_geometry_interval_discovery_v2"
    assert family_manifest["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_profile"] == "geometry_interval_discovery_v2"
    assert family_manifest["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_version"] == "pw_attack_matrix_geometry_interval_discovery_v2"
    assert family_manifest["benchmark_provenance"]["protocol_id"] == "geometry_interval_discovery_v2"
    assert family_manifest["benchmark_provenance"]["benchmark_protocol_config_path"] == normalize_path_value(
        INTERVAL_DISCOVERY_V2_PROTOCOL_CONFIG_PATH
    )
    assert family_manifest["benchmark_protocol"]["score_pools"]["content_chain_score"]["evaluate_role_order"] == [
        "positive_source",
        "clean_negative",
    ]
    assert family_manifest["attack_parameters"]["matrix_profile"] == "geometry_interval_discovery_v2"
    assert family_manifest["attack_parameters"]["matrix_version"] == "pw_attack_matrix_geometry_interval_discovery_v2"
    assert family_manifest["attack_parameters"]["attack_condition_count"] == 18
    assert family_manifest["attack_parameters"]["geometry_optional_claim_boundary_abs_margin_max"] == pytest.approx(0.3)
    assert method_identity_snapshot["source_alignment_reference_files"] == [
        "paper_workflow/configs/pw_base_geometry_interval_discovery_v2.yaml",
        "paper_workflow/configs/pw_matrix_geometry_interval_discovery_v2.yaml",
        "paper_workflow/configs/pw_protocol_geometry_interval_discovery_v2.yaml",
        "paper_workflow/scripts/pw_common.py",
        "paper_workflow/scripts/pw00_build_family_manifest.py",
        "paper_workflow/scripts/pw01_stage_runtime_helpers.py",
        "paper_workflow/scripts/pw01_run_source_event_shard.py",
        "paper_workflow/scripts/pw02_merge_source_event_shards.py",
        "paper_workflow/notebook/PW00_Paper_Eval_Family_Manifest.ipynb",
        "paper_workflow/notebook/PW01_Source_Event_Shards.ipynb",
        "paper_workflow/notebook/PW02_Source_Merge_And_Global_Thresholds.ipynb",
        "scripts/notebook_runtime_common.py",
        "configs/default.yaml",
    ]


def test_attack_protocol_geometry_mix_versions_are_loadable() -> None:
    """
    Verify append-only geometry-mix protocol versions are loadable.

    Args:
        None.

    Returns:
        None.
    """
    protocol_spec = protocol_loader.load_attack_protocol_spec({})
    params_versions = protocol_spec.get("params_versions", {})
    generated_plan = attack_plan.generate_attack_plan(protocol_spec)

    expected_condition_keys = {
        "rotate::v5",
        "crop::v5",
        "translate::v2",
        "composite::rotate_crop_v1",
        "composite::rotate_crop_v2",
        "composite::rotate_resize_jpeg_v6",
        "composite::crop_resize_translate_v1",
        "composite::rotate_translate_resize_v1",
    }

    assert expected_condition_keys.issubset(set(params_versions))
    assert expected_condition_keys.issubset(set(generated_plan.conditions))


def test_geometry_mix_matrix_materializes_expected_condition_subset() -> None:
    """
    Verify the geometry-mix matrix parses and materializes the new geometry-dominant conditions.

    Args:
        None.

    Returns:
        None.
    """
    geometry_mix_base_cfg = load_yaml_mapping(GEOMETRY_MIX_PW_BASE_CONFIG_PATH)
    matrix_cfg = load_pw_matrix_config(matrix_config_path=GEOMETRY_MIX_PW_MATRIX_CONFIG_PATH)
    matrix_settings = resolve_pw_matrix_settings(matrix_cfg)
    attack_condition_catalog = build_attack_condition_catalog(matrix_cfg=matrix_cfg)
    catalog_params = {
        row["attack_condition_key"]: row["attack_params"]
        for row in attack_condition_catalog
    }

    expected_families = ["rotate", "crop", "translate", "composite"]
    expected_condition_keys = [
        "composite::crop_resize_translate_v1::sev00",
        "composite::rotate_crop_v1::sev00",
        "composite::rotate_crop_v2::sev00",
        "composite::rotate_resize_jpeg_v6::sev00",
        "composite::rotate_translate_resize_v1::sev00",
        "crop::v5::sev00",
        "crop::v5::sev01",
        "crop::v5::sev02",
        "crop::v5::sev03",
        "crop::v5::sev04",
        "rotate::v5::sev00",
        "rotate::v5::sev01",
        "rotate::v5::sev02",
        "rotate::v5::sev03",
        "rotate::v5::sev04",
        "translate::v2::sev00",
        "translate::v2::sev01",
        "translate::v2::sev02",
    ]

    assert geometry_mix_base_cfg["benchmark_mode"] == "geometry_mix"
    assert geometry_mix_base_cfg["benchmark_mode_version"] == "geometry_mix_v1"
    assert geometry_mix_base_cfg["matrix_config_path"] == "paper_workflow/configs/pw_matrix_geometry_mix.yaml"
    assert geometry_mix_base_cfg["benchmark_protocol_config_path"] == "paper_workflow/configs/pw_protocol_geometry_mix.yaml"
    assert geometry_mix_base_cfg["sample_roles"]["active"] == [
        "positive_source",
        "clean_negative",
        "planner_conditioned_control_negative",
    ]
    assert geometry_mix_base_cfg["sample_roles"]["reserved"] == ["attacked_positive", "attacked_negative"]
    assert matrix_settings["matrix_profile"] == "geometry_mix_v1"
    assert matrix_settings["matrix_version"] == "pw_attack_matrix_geometry_mix_v1"
    assert matrix_settings["materialization_profile"] == "matrix_defined_concrete_conditions"
    assert matrix_settings["attack_sets"]["general_attacks"] == expected_families
    assert matrix_settings["attack_sets"]["geometry_rescue_candidates"] == expected_families
    assert matrix_settings["geometry_optional_claim"]["candidate_attack_set"] == "geometry_rescue_candidates"
    assert matrix_settings["geometry_optional_claim"]["candidate_attack_families"] == expected_families
    assert matrix_settings["geometry_optional_claim"]["boundary_abs_margin_min"] == pytest.approx(0.005)
    assert matrix_settings["geometry_optional_claim"]["boundary_abs_margin_max"] == pytest.approx(0.18)
    assert [row["attack_condition_key"] for row in attack_condition_catalog] == expected_condition_keys
    assert len(attack_condition_catalog) == 18
    assert catalog_params["rotate::v5::sev00"] == {"degrees": 19}
    assert catalog_params["rotate::v5::sev04"] == {"degrees": 30}
    assert catalog_params["crop::v5::sev00"] == {"crop_ratios": 0.78}
    assert catalog_params["crop::v5::sev04"] == {"crop_ratios": 0.58}
    assert catalog_params["translate::v2::sev00"] == {"x_shift": 8, "y_shift": 0}
    assert catalog_params["translate::v2::sev02"] == {"x_shift": 16, "y_shift": 8}
    assert catalog_params["composite::rotate_crop_v1::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 19}},
            {"family": "crop", "params": {"crop_ratio": 0.78}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::rotate_crop_v2::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 23}},
            {"family": "crop", "params": {"crop_ratio": 0.72}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::rotate_resize_jpeg_v6::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 26}},
            {"family": "resize", "params": {"scale_factor": 0.66}},
            {"family": "jpeg", "params": {"quality": 70}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::crop_resize_translate_v1::sev00"] == {
        "steps": [
            {"family": "crop", "params": {"crop_ratio": 0.68}},
            {"family": "resize", "params": {"scale_factor": 0.62}},
            {"family": "translate", "params": {"x_shift": 8, "y_shift": 0}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::rotate_translate_resize_v1::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 21}},
            {"family": "translate", "params": {"x_shift": 12, "y_shift": 4}},
            {"family": "resize", "params": {"scale_factor": 0.72}},
        ],
        "seed_policy": "shared",
    }


def test_pw00_binds_geometry_mix_base_and_shared_protocol(tmp_path: Path) -> None:
    """
    Verify PW00 can bind the geometry-mix base config and append geometry-mix protocol provenance.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text("prompt alpha\nprompt beta\n", encoding="utf-8")

    summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="paper_eval_family_geometry_mix",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
        pw_base_config_path=GEOMETRY_MIX_PW_BASE_CONFIG_PATH,
    )

    family_manifest = json.loads(Path(str(summary["paper_eval_family_manifest_path"])).read_text(encoding="utf-8"))
    method_identity_snapshot = json.loads(
        Path(str(family_manifest["paths"]["method_identity_snapshot"])).read_text(encoding="utf-8")
    )

    assert summary["pw_base_config_path"] == normalize_path_value(GEOMETRY_MIX_PW_BASE_CONFIG_PATH)
    assert summary["pw_matrix_config_path"] == normalize_path_value(GEOMETRY_MIX_PW_MATRIX_CONFIG_PATH)
    assert summary["benchmark_protocol_config_path"] == normalize_path_value(GEOMETRY_MIX_PROTOCOL_CONFIG_PATH)
    assert summary["benchmark_protocol"]["protocol_id"] == "geometry_mix_v1"
    assert summary["benchmark_protocol"]["protocol_family_id"] == "paper_eval_family_geometry_mix"
    assert summary["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_profile"] == "geometry_mix_v1"
    assert summary["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_version"] == "pw_attack_matrix_geometry_mix_v1"
    assert summary["benchmark_provenance"]["protocol_id"] == "geometry_mix_v1"
    assert summary["benchmark_provenance"]["benchmark_protocol_config_path"] == normalize_path_value(
        GEOMETRY_MIX_PROTOCOL_CONFIG_PATH
    )
    assert summary["matrix_profile"] == "geometry_mix_v1"
    assert summary["matrix_version"] == "pw_attack_matrix_geometry_mix_v1"
    assert summary["attack_condition_count"] == 18
    assert summary["source_shard_count"] == 3
    assert summary["attack_shard_count"] == 3
    assert summary["geometry_optional_claim_boundary_abs_margin_min"] == pytest.approx(0.005)
    assert summary["geometry_optional_claim_boundary_abs_margin_max"] == pytest.approx(0.18)
    assert family_manifest["family_id"] == "paper_eval_family_geometry_mix"
    assert family_manifest["pw_base_config_path"] == normalize_path_value(GEOMETRY_MIX_PW_BASE_CONFIG_PATH)
    assert family_manifest["pw_matrix_config_path"] == normalize_path_value(GEOMETRY_MIX_PW_MATRIX_CONFIG_PATH)
    assert family_manifest["benchmark_protocol_config_path"] == normalize_path_value(
        GEOMETRY_MIX_PROTOCOL_CONFIG_PATH
    )
    assert family_manifest["benchmark_protocol"]["protocol_id"] == "geometry_mix_v1"
    assert family_manifest["benchmark_protocol"]["protocol_family_id"] == "paper_eval_family_geometry_mix"
    assert family_manifest["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_profile"] == "geometry_mix_v1"
    assert family_manifest["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_version"] == "pw_attack_matrix_geometry_mix_v1"
    assert family_manifest["benchmark_provenance"]["protocol_id"] == "geometry_mix_v1"
    assert family_manifest["benchmark_provenance"]["benchmark_protocol_config_path"] == normalize_path_value(
        GEOMETRY_MIX_PROTOCOL_CONFIG_PATH
    )
    assert family_manifest["benchmark_protocol"]["score_pools"]["content_chain_score"]["evaluate_role_order"] == [
        "positive_source",
        "clean_negative",
    ]
    assert family_manifest["attack_parameters"]["matrix_profile"] == "geometry_mix_v1"
    assert family_manifest["attack_parameters"]["matrix_version"] == "pw_attack_matrix_geometry_mix_v1"
    assert family_manifest["attack_parameters"]["attack_condition_count"] == 18
    assert family_manifest["attack_parameters"]["geometry_optional_claim_boundary_abs_margin_min"] == pytest.approx(0.005)
    assert family_manifest["attack_parameters"]["geometry_optional_claim_boundary_abs_margin_max"] == pytest.approx(0.18)
    assert method_identity_snapshot["source_alignment_reference_files"] == [
        "paper_workflow/configs/pw_base_geometry_mix.yaml",
        "paper_workflow/configs/pw_matrix_geometry_mix.yaml",
        "paper_workflow/configs/pw_protocol_geometry_mix.yaml",
        "paper_workflow/scripts/pw_common.py",
        "paper_workflow/scripts/pw00_build_family_manifest.py",
        "paper_workflow/scripts/pw01_stage_runtime_helpers.py",
        "paper_workflow/scripts/pw01_run_source_event_shard.py",
        "paper_workflow/scripts/pw02_merge_source_event_shards.py",
        "paper_workflow/notebook/PW00_Paper_Eval_Family_Manifest.ipynb",
        "paper_workflow/notebook/PW01_Source_Event_Shards.ipynb",
        "paper_workflow/notebook/PW02_Source_Merge_And_Global_Thresholds.ipynb",
        "scripts/notebook_runtime_common.py",
        "configs/default.yaml",
    ]


def test_attack_protocol_geometry_mix_v2_versions_are_loadable() -> None:
    """
    Verify append-only geometry-mix-v2 protocol versions are loadable.

    Args:
        None.

    Returns:
        None.
    """
    protocol_spec = protocol_loader.load_attack_protocol_spec({})
    params_versions = protocol_spec.get("params_versions", {})
    generated_plan = attack_plan.generate_attack_plan(protocol_spec)

    expected_condition_keys = {
        "rotate::v6",
        "crop::v6",
        "translate::v3",
        "composite::rotate_crop_v3",
        "composite::rotate_crop_v4",
        "composite::rotate_resize_jpeg_v7",
        "composite::rotate_resize_jpeg_v8",
        "composite::crop_resize_translate_v2",
        "composite::crop_resize_translate_v3",
    }

    assert expected_condition_keys.issubset(set(params_versions))
    assert expected_condition_keys.issubset(set(generated_plan.conditions))


def test_attack_protocol_geometry_mix_v3_versions_are_loadable() -> None:
    """
    Verify append-only geometry-mix-v3 protocol versions are loadable.

    Args:
        None.

    Returns:
        None.
    """
    protocol_spec = protocol_loader.load_attack_protocol_spec({})
    params_versions = protocol_spec.get("params_versions", {})
    generated_plan = attack_plan.generate_attack_plan(protocol_spec)

    expected_condition_keys = {
        "rotate::v7",
        "crop::v7",
        "translate::v4",
        "composite::rotate_crop_v5",
        "composite::rotate_crop_v6",
        "composite::rotate_crop_v7",
        "composite::rotate_resize_jpeg_v9",
        "composite::rotate_resize_jpeg_v10",
        "composite::crop_resize_translate_v4",
        "composite::crop_resize_translate_v5",
        "composite::rotate_translate_resize_v2",
    }

    assert expected_condition_keys.issubset(set(params_versions))
    assert expected_condition_keys.issubset(set(generated_plan.conditions))


def test_attack_protocol_geometry_mix_v4_versions_are_loadable() -> None:
    """
    Verify append-only geometry-mix-v4 protocol versions are loadable.

    Args:
        None.

    Returns:
        None.
    """
    protocol_spec = protocol_loader.load_attack_protocol_spec({})
    params_versions = protocol_spec.get("params_versions", {})
    generated_plan = attack_plan.generate_attack_plan(protocol_spec)

    expected_condition_keys = {
        "rotate::v8",
        "crop::v8",
        "translate::v5",
        "composite::rotate_crop_v8",
        "composite::rotate_crop_v9",
        "composite::rotate_resize_jpeg_v11",
        "composite::rotate_resize_jpeg_v12",
        "composite::crop_resize_translate_v6",
        "composite::crop_resize_translate_v7",
    }

    assert expected_condition_keys.issubset(set(params_versions))
    assert expected_condition_keys.issubset(set(generated_plan.conditions))


def test_geometry_mix_v2_matrix_materializes_expected_condition_subset() -> None:
    """
    Verify the geometry-mix-v2 matrix parses and materializes the denser mild-to-strong conditions.

    Args:
        None.

    Returns:
        None.
    """
    geometry_mix_v2_base_cfg = load_yaml_mapping(GEOMETRY_MIX_V2_PW_BASE_CONFIG_PATH)
    matrix_cfg = load_pw_matrix_config(matrix_config_path=GEOMETRY_MIX_V2_PW_MATRIX_CONFIG_PATH)
    matrix_settings = resolve_pw_matrix_settings(matrix_cfg)
    attack_condition_catalog = build_attack_condition_catalog(matrix_cfg=matrix_cfg)
    catalog_params = {
        row["attack_condition_key"]: row["attack_params"]
        for row in attack_condition_catalog
    }

    expected_families = ["rotate", "crop", "translate", "composite"]
    expected_rescue_families = ["rotate", "crop", "composite"]
    expected_condition_keys = [
        "composite::crop_resize_translate_v2::sev00",
        "composite::crop_resize_translate_v3::sev00",
        "composite::rotate_crop_v3::sev00",
        "composite::rotate_crop_v4::sev00",
        "composite::rotate_resize_jpeg_v7::sev00",
        "composite::rotate_resize_jpeg_v8::sev00",
        "crop::v6::sev00",
        "crop::v6::sev01",
        "crop::v6::sev02",
        "crop::v6::sev03",
        "crop::v6::sev04",
        "crop::v6::sev05",
        "crop::v6::sev06",
        "rotate::v6::sev00",
        "rotate::v6::sev01",
        "rotate::v6::sev02",
        "rotate::v6::sev03",
        "rotate::v6::sev04",
        "rotate::v6::sev05",
        "rotate::v6::sev06",
        "translate::v3::sev00",
        "translate::v3::sev01",
        "translate::v3::sev02",
        "translate::v3::sev03",
    ]

    assert geometry_mix_v2_base_cfg["benchmark_mode"] == "geometry_mix"
    assert geometry_mix_v2_base_cfg["benchmark_mode_version"] == "geometry_mix_v2"
    assert geometry_mix_v2_base_cfg["matrix_config_path"] == "paper_workflow/configs/pw_matrix_geometry_mix_v2.yaml"
    assert geometry_mix_v2_base_cfg["benchmark_protocol_config_path"] == "paper_workflow/configs/pw_protocol_geometry_mix_v2.yaml"
    assert geometry_mix_v2_base_cfg["sample_roles"]["active"] == [
        "positive_source",
        "clean_negative",
        "planner_conditioned_control_negative",
    ]
    assert geometry_mix_v2_base_cfg["sample_roles"]["reserved"] == ["attacked_positive", "attacked_negative"]
    assert matrix_settings["matrix_profile"] == "geometry_mix_v2"
    assert matrix_settings["matrix_version"] == "pw_attack_matrix_geometry_mix_v2"
    assert matrix_settings["materialization_profile"] == "matrix_defined_concrete_conditions"
    assert matrix_settings["attack_sets"]["general_attacks"] == expected_families
    assert matrix_settings["attack_sets"]["geometry_rescue_candidates"] == expected_rescue_families
    assert matrix_settings["geometry_optional_claim"]["candidate_attack_set"] == "geometry_rescue_candidates"
    assert matrix_settings["geometry_optional_claim"]["candidate_attack_families"] == expected_rescue_families
    assert matrix_settings["geometry_optional_claim"]["boundary_abs_margin_min"] == pytest.approx(0.005)
    assert matrix_settings["geometry_optional_claim"]["boundary_abs_margin_max"] == pytest.approx(0.12)
    assert [row["attack_condition_key"] for row in attack_condition_catalog] == expected_condition_keys
    assert len(attack_condition_catalog) == 24
    assert catalog_params["rotate::v6::sev00"] == {"degrees": 15}
    assert catalog_params["rotate::v6::sev06"] == {"degrees": 27}
    assert catalog_params["crop::v6::sev00"] == {"crop_ratios": 0.84}
    assert catalog_params["crop::v6::sev06"] == {"crop_ratios": 0.6}
    assert catalog_params["translate::v3::sev00"] == {"x_shift": 4, "y_shift": 0}
    assert catalog_params["translate::v3::sev03"] == {"x_shift": 10, "y_shift": 4}
    assert catalog_params["composite::rotate_crop_v3::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 17}},
            {"family": "crop", "params": {"crop_ratio": 0.8}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::rotate_crop_v4::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 21}},
            {"family": "crop", "params": {"crop_ratio": 0.72}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::rotate_resize_jpeg_v7::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 19}},
            {"family": "resize", "params": {"scale_factor": 0.78}},
            {"family": "jpeg", "params": {"quality": 80}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::rotate_resize_jpeg_v8::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 23}},
            {"family": "resize", "params": {"scale_factor": 0.72}},
            {"family": "jpeg", "params": {"quality": 75}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::crop_resize_translate_v2::sev00"] == {
        "steps": [
            {"family": "crop", "params": {"crop_ratio": 0.76}},
            {"family": "resize", "params": {"scale_factor": 0.72}},
            {"family": "translate", "params": {"x_shift": 6, "y_shift": 0}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::crop_resize_translate_v3::sev00"] == {
        "steps": [
            {"family": "crop", "params": {"crop_ratio": 0.7}},
            {"family": "resize", "params": {"scale_factor": 0.68}},
            {"family": "translate", "params": {"x_shift": 8, "y_shift": 2}},
        ],
        "seed_policy": "shared",
    }


def test_geometry_mix_v3_matrix_materializes_expected_condition_subset() -> None:
    """
    Verify the geometry-mix-v3 matrix parses and materializes the lighter interval-search conditions.

    Args:
        None.

    Returns:
        None.
    """
    geometry_mix_v3_base_cfg = load_yaml_mapping(GEOMETRY_MIX_V3_PW_BASE_CONFIG_PATH)
    geometry_mix_v3_protocol_cfg = load_yaml_mapping(GEOMETRY_MIX_V3_PROTOCOL_CONFIG_PATH)
    matrix_cfg = load_pw_matrix_config(matrix_config_path=GEOMETRY_MIX_V3_PW_MATRIX_CONFIG_PATH)
    matrix_settings = resolve_pw_matrix_settings(matrix_cfg)
    attack_condition_catalog = build_attack_condition_catalog(matrix_cfg=matrix_cfg)
    catalog_params = {
        row["attack_condition_key"]: row["attack_params"]
        for row in attack_condition_catalog
    }

    expected_families = ["rotate", "crop", "translate", "composite"]
    expected_rescue_families = ["rotate", "crop", "composite"]
    expected_condition_keys = sorted(
        [
            "composite::crop_resize_translate_v4::sev00",
            "composite::crop_resize_translate_v5::sev00",
            "composite::rotate_crop_v5::sev00",
            "composite::rotate_crop_v6::sev00",
            "composite::rotate_crop_v7::sev00",
            "composite::rotate_resize_jpeg_v10::sev00",
            "composite::rotate_resize_jpeg_v9::sev00",
            "composite::rotate_translate_resize_v2::sev00",
            "crop::v7::sev00",
            "crop::v7::sev01",
            "crop::v7::sev02",
            "crop::v7::sev03",
            "crop::v7::sev04",
            "crop::v7::sev05",
            "crop::v7::sev06",
            "rotate::v7::sev00",
            "rotate::v7::sev01",
            "rotate::v7::sev02",
            "rotate::v7::sev03",
            "rotate::v7::sev04",
            "rotate::v7::sev05",
            "rotate::v7::sev06",
            "translate::v4::sev00",
            "translate::v4::sev01",
            "translate::v4::sev02",
        ]
    )

    assert geometry_mix_v3_base_cfg["benchmark_mode"] == "geometry_mix"
    assert geometry_mix_v3_base_cfg["benchmark_mode_version"] == "geometry_mix_v3"
    assert geometry_mix_v3_base_cfg["matrix_config_path"] == "paper_workflow/configs/pw_matrix_geometry_mix_v3.yaml"
    assert geometry_mix_v3_base_cfg["benchmark_protocol_config_path"] == "paper_workflow/configs/pw_protocol_geometry_mix_v3.yaml"
    assert geometry_mix_v3_base_cfg["source_alignment_reference_files"][:3] == [
        "paper_workflow/configs/pw_base_geometry_mix_v3.yaml",
        "paper_workflow/configs/pw_matrix_geometry_mix_v3.yaml",
        "paper_workflow/configs/pw_protocol_geometry_mix_v3.yaml",
    ]
    assert geometry_mix_v3_protocol_cfg["protocol_id"] == "geometry_mix_v3"
    assert geometry_mix_v3_protocol_cfg["benchmark_name"] == "geometry_mix_v3"
    assert geometry_mix_v3_protocol_cfg["protocol_family_id"] == "paper_eval_family_geometry_mix"
    assert geometry_mix_v3_protocol_cfg["geometry_dominant_severity_ladder"]["matrix_profile"] == "geometry_mix_v3"
    assert geometry_mix_v3_protocol_cfg["geometry_dominant_severity_ladder"]["matrix_version"] == "pw_attack_matrix_geometry_mix_v3"
    assert matrix_settings["matrix_profile"] == "geometry_mix_v3"
    assert matrix_settings["matrix_version"] == "pw_attack_matrix_geometry_mix_v3"
    assert matrix_settings["materialization_profile"] == "matrix_defined_concrete_conditions"
    assert matrix_settings["attack_sets"]["general_attacks"] == expected_families
    assert matrix_settings["attack_sets"]["geometry_rescue_candidates"] == expected_rescue_families
    assert "translate" not in matrix_settings["geometry_optional_claim"]["candidate_attack_families"]
    assert matrix_settings["geometry_optional_claim"]["candidate_attack_set"] == "geometry_rescue_candidates"
    assert matrix_settings["geometry_optional_claim"]["boundary_abs_margin_min"] == pytest.approx(0.005)
    assert matrix_settings["geometry_optional_claim"]["boundary_abs_margin_max"] == pytest.approx(0.10)
    assert matrix_settings["system_event_count_sweep"]["event_counts"] == [1, 2, 4, 8, 16, 32, 64]
    assert matrix_settings["system_event_count_sweep"]["repeat_count"] == 48
    assert matrix_settings["system_event_count_sweep"]["random_seed"] == 20260415
    assert [row["attack_condition_key"] for row in attack_condition_catalog] == expected_condition_keys
    assert len(attack_condition_catalog) == 25
    assert catalog_params["rotate::v7::sev00"] == {"degrees": 9}
    assert catalog_params["rotate::v7::sev06"] == {"degrees": 21}
    assert catalog_params["crop::v7::sev00"] == {"crop_ratios": 0.9}
    assert catalog_params["crop::v7::sev06"] == {"crop_ratios": 0.66}
    assert catalog_params["translate::v4::sev00"] == {"x_shift": 2, "y_shift": 0}
    assert catalog_params["translate::v4::sev02"] == {"x_shift": 6, "y_shift": 2}
    assert catalog_params["composite::rotate_crop_v5::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 11}},
            {"family": "crop", "params": {"crop_ratio": 0.86}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::rotate_crop_v6::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 15}},
            {"family": "crop", "params": {"crop_ratio": 0.78}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::rotate_crop_v7::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 19}},
            {"family": "crop", "params": {"crop_ratio": 0.7}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::rotate_resize_jpeg_v9::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 13}},
            {"family": "resize", "params": {"scale_factor": 0.86}},
            {"family": "jpeg", "params": {"quality": 85}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::rotate_resize_jpeg_v10::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 17}},
            {"family": "resize", "params": {"scale_factor": 0.8}},
            {"family": "jpeg", "params": {"quality": 80}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::crop_resize_translate_v4::sev00"] == {
        "steps": [
            {"family": "crop", "params": {"crop_ratio": 0.82}},
            {"family": "resize", "params": {"scale_factor": 0.78}},
            {"family": "translate", "params": {"x_shift": 4, "y_shift": 0}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::crop_resize_translate_v5::sev00"] == {
        "steps": [
            {"family": "crop", "params": {"crop_ratio": 0.74}},
            {"family": "resize", "params": {"scale_factor": 0.72}},
            {"family": "translate", "params": {"x_shift": 6, "y_shift": 0}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::rotate_translate_resize_v2::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 17}},
            {"family": "translate", "params": {"x_shift": 4, "y_shift": 0}},
            {"family": "resize", "params": {"scale_factor": 0.82}},
        ],
        "seed_policy": "shared",
    }


def test_geometry_mix_v4_matrix_materializes_expected_condition_subset() -> None:
    """
    Verify the geometry-mix-v4 matrix parses and materializes the intentionally weakened conditions.

    Args:
        None.

    Returns:
        None.
    """
    geometry_mix_v4_base_cfg = load_yaml_mapping(GEOMETRY_MIX_V4_PW_BASE_CONFIG_PATH)
    geometry_mix_v4_protocol_cfg = load_yaml_mapping(GEOMETRY_MIX_V4_PROTOCOL_CONFIG_PATH)
    matrix_cfg = load_pw_matrix_config(matrix_config_path=GEOMETRY_MIX_V4_PW_MATRIX_CONFIG_PATH)
    matrix_settings = resolve_pw_matrix_settings(matrix_cfg)
    attack_condition_catalog = build_attack_condition_catalog(matrix_cfg=matrix_cfg)
    catalog_params = {
        row["attack_condition_key"]: row["attack_params"]
        for row in attack_condition_catalog
    }

    rotate_condition_keys = sorted(
        row["attack_condition_key"]
        for row in attack_condition_catalog
        if row["attack_condition_base_key"] == "rotate::v8"
    )
    crop_condition_keys = sorted(
        row["attack_condition_key"]
        for row in attack_condition_catalog
        if row["attack_condition_base_key"] == "crop::v8"
    )
    translate_condition_keys = sorted(
        row["attack_condition_key"]
        for row in attack_condition_catalog
        if row["attack_condition_base_key"] == "translate::v5"
    )
    composite_condition_keys = sorted(
        row["attack_condition_key"]
        for row in attack_condition_catalog
        if row["attack_family"] == "composite"
    )

    assert geometry_mix_v4_base_cfg["benchmark_mode"] == "geometry_mix"
    assert geometry_mix_v4_base_cfg["benchmark_mode_version"] == "geometry_mix_v4"
    assert geometry_mix_v4_base_cfg["matrix_config_path"] == "paper_workflow/configs/pw_matrix_geometry_mix_v4.yaml"
    assert geometry_mix_v4_base_cfg["benchmark_protocol_config_path"] == "paper_workflow/configs/pw_protocol_geometry_mix_v4.yaml"
    assert geometry_mix_v4_base_cfg["calibration_fraction"] == pytest.approx(0.5)
    assert geometry_mix_v4_base_cfg["calibration_fraction_by_role"] == {
        "positive_source": 0.5,
        "clean_negative": 0.5,
        "planner_conditioned_control_negative": 0.15,
    }
    assert geometry_mix_v4_protocol_cfg["protocol_id"] == "geometry_mix_v4"
    assert geometry_mix_v4_protocol_cfg["benchmark_name"] == "geometry_mix_v4"
    assert geometry_mix_v4_protocol_cfg["protocol_family_id"] == "paper_eval_family_geometry_mix"
    assert geometry_mix_v4_protocol_cfg["geometry_dominant_severity_ladder"]["matrix_profile"] == "geometry_mix_v4"
    assert geometry_mix_v4_protocol_cfg["geometry_dominant_severity_ladder"]["matrix_version"] == "pw_attack_matrix_geometry_mix_v4"
    assert matrix_settings["matrix_profile"] == "geometry_mix_v4"
    assert matrix_settings["matrix_version"] == "pw_attack_matrix_geometry_mix_v4"
    assert matrix_settings["materialization_profile"] == "matrix_defined_concrete_conditions"
    assert matrix_settings["attack_sets"]["general_attacks"] == ["rotate", "crop", "translate", "composite"]
    assert matrix_settings["attack_sets"]["geometry_rescue_candidates"] == ["rotate", "crop", "composite"]
    assert "translate" not in matrix_settings["geometry_optional_claim"]["candidate_attack_families"]
    assert matrix_settings["geometry_optional_claim"]["boundary_abs_margin_min"] == pytest.approx(0.005)
    assert matrix_settings["geometry_optional_claim"]["boundary_abs_margin_max"] == pytest.approx(0.08)
    assert matrix_settings["system_event_count_sweep"]["event_counts"] == [1, 2, 4, 8, 16, 32, 64]
    assert matrix_settings["system_event_count_sweep"]["repeat_count"] == 48
    assert matrix_settings["system_event_count_sweep"]["random_seed"] == 20260415
    assert len(attack_condition_catalog) == 21
    assert rotate_condition_keys == [
        "rotate::v8::sev00",
        "rotate::v8::sev01",
        "rotate::v8::sev02",
        "rotate::v8::sev03",
        "rotate::v8::sev04",
        "rotate::v8::sev05",
    ]
    assert crop_condition_keys == [
        "crop::v8::sev00",
        "crop::v8::sev01",
        "crop::v8::sev02",
        "crop::v8::sev03",
        "crop::v8::sev04",
        "crop::v8::sev05",
    ]
    assert translate_condition_keys == [
        "translate::v5::sev00",
        "translate::v5::sev01",
        "translate::v5::sev02",
    ]
    assert composite_condition_keys == [
        "composite::crop_resize_translate_v6::sev00",
        "composite::crop_resize_translate_v7::sev00",
        "composite::rotate_crop_v8::sev00",
        "composite::rotate_crop_v9::sev00",
        "composite::rotate_resize_jpeg_v11::sev00",
        "composite::rotate_resize_jpeg_v12::sev00",
    ]
    assert catalog_params["rotate::v8::sev00"] == {"degrees": 7}
    assert catalog_params["rotate::v8::sev05"] == {"degrees": 17}
    assert catalog_params["crop::v8::sev00"] == {"crop_ratios": 0.94}
    assert catalog_params["crop::v8::sev05"] == {"crop_ratios": 0.74}
    assert catalog_params["translate::v5::sev02"] == {"x_shift": 6, "y_shift": 0}
    assert catalog_params["composite::rotate_crop_v8::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 9}},
            {"family": "crop", "params": {"crop_ratio": 0.9}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::rotate_crop_v9::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 13}},
            {"family": "crop", "params": {"crop_ratio": 0.82}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::rotate_resize_jpeg_v11::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 11}},
            {"family": "resize", "params": {"scale_factor": 0.9}},
            {"family": "jpeg", "params": {"quality": 90}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::rotate_resize_jpeg_v12::sev00"] == {
        "steps": [
            {"family": "rotate", "params": {"degrees": 15}},
            {"family": "resize", "params": {"scale_factor": 0.84}},
            {"family": "jpeg", "params": {"quality": 85}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::crop_resize_translate_v6::sev00"] == {
        "steps": [
            {"family": "crop", "params": {"crop_ratio": 0.86}},
            {"family": "resize", "params": {"scale_factor": 0.82}},
            {"family": "translate", "params": {"x_shift": 2, "y_shift": 0}},
        ],
        "seed_policy": "shared",
    }
    assert catalog_params["composite::crop_resize_translate_v7::sev00"] == {
        "steps": [
            {"family": "crop", "params": {"crop_ratio": 0.78}},
            {"family": "resize", "params": {"scale_factor": 0.76}},
            {"family": "translate", "params": {"x_shift": 4, "y_shift": 0}},
        ],
        "seed_policy": "shared",
    }


def test_build_source_split_plan_supports_role_level_calibration_fraction() -> None:
    """
    Verify source split can apply a weaker calibration fraction to control-negative role.

    Args:
        None.

    Returns:
        None.
    """
    prompt_lines = [f"prompt {index}" for index in range(20)]
    event_grid = build_source_event_grid(
        family_id="paper_eval_family_geometry_mix",
        prompt_lines=prompt_lines,
        seeds=[0],
        prompt_file="prompts/paper_pilot_10.txt",
        sample_roles=[
            "positive_source",
            "clean_negative",
            "planner_conditioned_control_negative",
        ],
    )

    split_plan = build_source_split_plan(
        family_id="paper_eval_family_geometry_mix",
        events=event_grid,
        calibration_fraction=0.5,
        calibration_fraction_by_role={
            "positive_source": 0.5,
            "clean_negative": 0.5,
            "planner_conditioned_control_negative": 0.15,
        },
    )

    assert split_plan["calibration_fraction"] == pytest.approx(0.5)
    assert split_plan["calibration_fraction_by_role"] == {
        "positive_source": 0.5,
        "clean_negative": 0.5,
        "planner_conditioned_control_negative": 0.15,
    }
    assert split_plan["role_level_calibration_counts"]["positive_source"]["calibration_event_count"] == 10
    assert split_plan["role_level_calibration_counts"]["clean_negative"]["calibration_event_count"] == 10
    assert split_plan["role_level_calibration_counts"]["planner_conditioned_control_negative"]["calibration_event_count"] == 3
    assert split_plan["roles"]["planner_conditioned_control_negative"]["evaluate_event_count"] == 17


def test_pw00_binds_geometry_mix_v2_base_and_protocol(tmp_path: Path) -> None:
    """
    Verify PW00 can bind the geometry-mix-v2 base config and append geometry-mix-v2 protocol provenance.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text("prompt alpha\nprompt beta\n", encoding="utf-8")

    summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="paper_eval_family_geometry_mix_v2",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
        pw_base_config_path=GEOMETRY_MIX_V2_PW_BASE_CONFIG_PATH,
    )

    family_manifest = json.loads(Path(str(summary["paper_eval_family_manifest_path"])).read_text(encoding="utf-8"))
    method_identity_snapshot = json.loads(
        Path(str(family_manifest["paths"]["method_identity_snapshot"])).read_text(encoding="utf-8")
    )
    geometry_optional_claim_plan = json.loads(
        Path(str(summary["geometry_optional_claim_plan_path"])).read_text(encoding="utf-8")
    )

    assert summary["pw_base_config_path"] == normalize_path_value(GEOMETRY_MIX_V2_PW_BASE_CONFIG_PATH)
    assert summary["pw_matrix_config_path"] == normalize_path_value(GEOMETRY_MIX_V2_PW_MATRIX_CONFIG_PATH)
    assert summary["benchmark_protocol_config_path"] == normalize_path_value(GEOMETRY_MIX_V2_PROTOCOL_CONFIG_PATH)
    assert summary["benchmark_protocol"]["protocol_id"] == "geometry_mix_v2"
    assert summary["benchmark_protocol"]["protocol_family_id"] == "paper_eval_family_geometry_mix_v2"
    assert summary["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_profile"] == "geometry_mix_v2"
    assert summary["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_version"] == "pw_attack_matrix_geometry_mix_v2"
    assert summary["benchmark_provenance"]["protocol_id"] == "geometry_mix_v2"
    assert summary["benchmark_provenance"]["benchmark_protocol_config_path"] == normalize_path_value(
        GEOMETRY_MIX_V2_PROTOCOL_CONFIG_PATH
    )
    assert summary["matrix_profile"] == "geometry_mix_v2"
    assert summary["matrix_version"] == "pw_attack_matrix_geometry_mix_v2"
    assert summary["attack_condition_count"] == 24
    assert summary["source_shard_count"] == 3
    assert summary["attack_shard_count"] == 3
    assert summary["geometry_optional_claim_boundary_abs_margin_min"] == pytest.approx(0.005)
    assert summary["geometry_optional_claim_boundary_abs_margin_max"] == pytest.approx(0.12)
    assert family_manifest["family_id"] == "paper_eval_family_geometry_mix_v2"
    assert family_manifest["pw_base_config_path"] == normalize_path_value(GEOMETRY_MIX_V2_PW_BASE_CONFIG_PATH)
    assert family_manifest["pw_matrix_config_path"] == normalize_path_value(GEOMETRY_MIX_V2_PW_MATRIX_CONFIG_PATH)
    assert family_manifest["benchmark_protocol_config_path"] == normalize_path_value(
        GEOMETRY_MIX_V2_PROTOCOL_CONFIG_PATH
    )
    assert family_manifest["benchmark_protocol"]["protocol_id"] == "geometry_mix_v2"
    assert family_manifest["benchmark_protocol"]["protocol_family_id"] == "paper_eval_family_geometry_mix_v2"
    assert family_manifest["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_profile"] == "geometry_mix_v2"
    assert family_manifest["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_version"] == "pw_attack_matrix_geometry_mix_v2"
    assert family_manifest["benchmark_provenance"]["protocol_id"] == "geometry_mix_v2"
    assert family_manifest["benchmark_provenance"]["benchmark_protocol_config_path"] == normalize_path_value(
        GEOMETRY_MIX_V2_PROTOCOL_CONFIG_PATH
    )
    assert family_manifest["benchmark_protocol"]["score_pools"]["content_chain_score"]["evaluate_role_order"] == [
        "positive_source",
        "clean_negative",
    ]
    assert family_manifest["attack_parameters"]["matrix_profile"] == "geometry_mix_v2"
    assert family_manifest["attack_parameters"]["matrix_version"] == "pw_attack_matrix_geometry_mix_v2"
    assert family_manifest["attack_parameters"]["attack_condition_count"] == 24
    assert family_manifest["attack_parameters"]["geometry_optional_claim_boundary_abs_margin_min"] == pytest.approx(0.005)
    assert family_manifest["attack_parameters"]["geometry_optional_claim_boundary_abs_margin_max"] == pytest.approx(0.12)
    assert geometry_optional_claim_plan["geometry_rescue_candidate_attack_families"] == ["rotate", "crop", "composite"]
    assert geometry_optional_claim_plan["boundary_abs_margin_min"] == pytest.approx(0.005)
    assert geometry_optional_claim_plan["boundary_abs_margin_max"] == pytest.approx(0.12)
    assert method_identity_snapshot["source_alignment_reference_files"] == [
        "paper_workflow/configs/pw_base_geometry_mix_v2.yaml",
        "paper_workflow/configs/pw_matrix_geometry_mix_v2.yaml",
        "paper_workflow/configs/pw_protocol_geometry_mix_v2.yaml",
        "paper_workflow/scripts/pw_common.py",
        "paper_workflow/scripts/pw00_build_family_manifest.py",
        "paper_workflow/scripts/pw01_stage_runtime_helpers.py",
        "paper_workflow/scripts/pw01_run_source_event_shard.py",
        "paper_workflow/scripts/pw02_merge_source_event_shards.py",
        "paper_workflow/notebook/PW00_Paper_Eval_Family_Manifest.ipynb",
        "paper_workflow/notebook/PW01_Source_Event_Shards.ipynb",
        "paper_workflow/notebook/PW02_Source_Merge_And_Global_Thresholds.ipynb",
        "scripts/notebook_runtime_common.py",
        "configs/default.yaml",
    ]


def test_pw00_binds_geometry_mix_family_to_v3_base_and_protocol(tmp_path: Path) -> None:
    """
    Verify PW00 keeps the geometry-mix family id while binding the v3 base config and protocol provenance.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text("prompt alpha\nprompt beta\n", encoding="utf-8")

    summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="paper_eval_family_geometry_mix",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
        pw_base_config_path=GEOMETRY_MIX_V3_PW_BASE_CONFIG_PATH,
    )

    family_manifest = json.loads(Path(str(summary["paper_eval_family_manifest_path"])).read_text(encoding="utf-8"))

    assert summary["family_id"] == "paper_eval_family_geometry_mix"
    assert summary["pw_base_config_path"] == normalize_path_value(GEOMETRY_MIX_V3_PW_BASE_CONFIG_PATH)
    assert summary["pw_matrix_config_path"] == normalize_path_value(GEOMETRY_MIX_V3_PW_MATRIX_CONFIG_PATH)
    assert summary["benchmark_protocol_config_path"] == normalize_path_value(GEOMETRY_MIX_V3_PROTOCOL_CONFIG_PATH)
    assert summary["benchmark_protocol"]["protocol_id"] == "geometry_mix_v3"
    assert summary["benchmark_protocol"]["protocol_family_id"] == "paper_eval_family_geometry_mix"
    assert summary["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_profile"] == "geometry_mix_v3"
    assert summary["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_version"] == "pw_attack_matrix_geometry_mix_v3"
    assert summary["benchmark_provenance"]["protocol_id"] == "geometry_mix_v3"
    assert summary["benchmark_provenance"]["benchmark_protocol_config_path"] == normalize_path_value(
        GEOMETRY_MIX_V3_PROTOCOL_CONFIG_PATH
    )
    assert summary["matrix_profile"] == "geometry_mix_v3"
    assert summary["matrix_version"] == "pw_attack_matrix_geometry_mix_v3"

    assert family_manifest["family_id"] == "paper_eval_family_geometry_mix"
    assert family_manifest["pw_base_config_path"] == normalize_path_value(GEOMETRY_MIX_V3_PW_BASE_CONFIG_PATH)
    assert family_manifest["pw_matrix_config_path"] == normalize_path_value(GEOMETRY_MIX_V3_PW_MATRIX_CONFIG_PATH)
    assert family_manifest["benchmark_protocol_config_path"] == normalize_path_value(
        GEOMETRY_MIX_V3_PROTOCOL_CONFIG_PATH
    )
    assert family_manifest["benchmark_protocol"]["protocol_id"] == "geometry_mix_v3"
    assert family_manifest["benchmark_protocol"]["protocol_family_id"] == "paper_eval_family_geometry_mix"
    assert family_manifest["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_profile"] == "geometry_mix_v3"
    assert family_manifest["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_version"] == "pw_attack_matrix_geometry_mix_v3"
    assert family_manifest["benchmark_provenance"]["protocol_id"] == "geometry_mix_v3"
    assert family_manifest["benchmark_provenance"]["benchmark_protocol_config_path"] == normalize_path_value(
        GEOMETRY_MIX_V3_PROTOCOL_CONFIG_PATH
    )
    assert family_manifest["attack_parameters"]["matrix_profile"] == "geometry_mix_v3"
    assert family_manifest["attack_parameters"]["matrix_version"] == "pw_attack_matrix_geometry_mix_v3"


def test_pw00_binds_geometry_mix_family_to_v4_base_and_role_calibration(tmp_path: Path) -> None:
    """
    Verify PW00 keeps the geometry-mix family id while binding the v4 calibration and matrix provenance.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text(
        "prompt alpha\nprompt beta\nprompt gamma\nprompt delta\n",
        encoding="utf-8",
    )

    summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="paper_eval_family_geometry_mix",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
        pw_base_config_path=GEOMETRY_MIX_V4_PW_BASE_CONFIG_PATH,
    )

    family_manifest = json.loads(Path(str(summary["paper_eval_family_manifest_path"])).read_text(encoding="utf-8"))
    source_split_plan = json.loads(Path(str(summary["source_split_plan_path"])).read_text(encoding="utf-8"))

    assert summary["family_id"] == "paper_eval_family_geometry_mix"
    assert summary["pw_base_config_path"] == normalize_path_value(GEOMETRY_MIX_V4_PW_BASE_CONFIG_PATH)
    assert summary["pw_matrix_config_path"] == normalize_path_value(GEOMETRY_MIX_V4_PW_MATRIX_CONFIG_PATH)
    assert summary["benchmark_protocol_config_path"] == normalize_path_value(GEOMETRY_MIX_V4_PROTOCOL_CONFIG_PATH)
    assert summary["benchmark_protocol"]["protocol_id"] == "geometry_mix_v4"
    assert summary["benchmark_protocol"]["protocol_family_id"] == "paper_eval_family_geometry_mix"
    assert summary["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_profile"] == "geometry_mix_v4"
    assert summary["benchmark_protocol"]["geometry_dominant_severity_ladder"]["matrix_version"] == "pw_attack_matrix_geometry_mix_v4"
    assert summary["matrix_profile"] == "geometry_mix_v4"
    assert summary["matrix_version"] == "pw_attack_matrix_geometry_mix_v4"
    assert summary["attack_condition_count"] == 21
    assert summary["geometry_optional_claim_boundary_abs_margin_min"] == pytest.approx(0.005)
    assert summary["geometry_optional_claim_boundary_abs_margin_max"] == pytest.approx(0.08)
    assert summary["calibration_fraction"] == pytest.approx(0.5)
    assert summary["calibration_fraction_by_role"]["planner_conditioned_control_negative"] == pytest.approx(0.15)
    assert summary["control_negative_calibration_fraction_effective"] == pytest.approx(0.15)
    assert summary["role_level_calibration_counts"]["planner_conditioned_control_negative"]["calibration_event_count"] == 1

    assert family_manifest["family_id"] == "paper_eval_family_geometry_mix"
    assert family_manifest["pw_base_config_path"] == normalize_path_value(GEOMETRY_MIX_V4_PW_BASE_CONFIG_PATH)
    assert family_manifest["pw_matrix_config_path"] == normalize_path_value(GEOMETRY_MIX_V4_PW_MATRIX_CONFIG_PATH)
    assert family_manifest["benchmark_protocol_config_path"] == normalize_path_value(
        GEOMETRY_MIX_V4_PROTOCOL_CONFIG_PATH
    )
    assert family_manifest["benchmark_protocol"]["protocol_id"] == "geometry_mix_v4"
    assert family_manifest["benchmark_protocol"]["protocol_family_id"] == "paper_eval_family_geometry_mix"
    assert family_manifest["attack_parameters"]["matrix_profile"] == "geometry_mix_v4"
    assert family_manifest["attack_parameters"]["matrix_version"] == "pw_attack_matrix_geometry_mix_v4"
    assert family_manifest["source_parameters"]["calibration_fraction"] == pytest.approx(0.5)
    assert family_manifest["source_parameters"]["calibration_fraction_by_role"][
        "planner_conditioned_control_negative"
    ] == pytest.approx(0.15)
    assert family_manifest["source_parameters"]["control_negative_calibration_fraction_effective"] == pytest.approx(0.15)

    assert source_split_plan["calibration_fraction"] == pytest.approx(0.5)
    assert source_split_plan["calibration_fraction_by_role"]["positive_source"] == pytest.approx(0.5)
    assert source_split_plan["calibration_fraction_by_role"]["clean_negative"] == pytest.approx(0.5)
    assert source_split_plan["calibration_fraction_by_role"]["planner_conditioned_control_negative"] == pytest.approx(0.15)
    assert source_split_plan["role_level_calibration_counts"]["planner_conditioned_control_negative"]["calibration_event_count"] == 1
    assert source_split_plan["role_level_calibration_counts"]["positive_source"]["calibration_event_count"] == 4


def test_pw00_binds_shared_benchmark_protocol_and_provenance(tmp_path: Path) -> None:
    """
    Verify PW00 appends shared benchmark protocol config and provenance to summary artifacts.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text("prompt alpha\nprompt beta\n", encoding="utf-8")

    summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="paper_eval_family_geometry_shared_benchmark_v1",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
        pw_base_config_path=SHARED_BENCHMARK_PW_BASE_CONFIG_PATH,
    )

    family_manifest = json.loads(Path(str(summary["paper_eval_family_manifest_path"])).read_text(encoding="utf-8"))
    method_identity_snapshot = json.loads(
        Path(str(family_manifest["paths"]["method_identity_snapshot"])).read_text(encoding="utf-8")
    )

    assert summary["pw_base_config_path"] == normalize_path_value(SHARED_BENCHMARK_PW_BASE_CONFIG_PATH)
    assert summary["pw_matrix_config_path"] == normalize_path_value(SHARED_BENCHMARK_PW_MATRIX_CONFIG_PATH)
    assert summary["benchmark_protocol_config_path"] == normalize_path_value(SHARED_BENCHMARK_PROTOCOL_CONFIG_PATH)
    assert summary["benchmark_protocol"]["protocol_id"] == "shared_hardneg_geometry_benchmark_v1"
    assert summary["benchmark_protocol"]["protocol_family_id"] == "paper_eval_family_geometry_shared_benchmark_v1"
    assert summary["benchmark_provenance"]["benchmark_protocol_config_path"] == normalize_path_value(
        SHARED_BENCHMARK_PROTOCOL_CONFIG_PATH
    )
    assert family_manifest["pw_base_config_path"] == normalize_path_value(SHARED_BENCHMARK_PW_BASE_CONFIG_PATH)
    assert family_manifest["pw_matrix_config_path"] == normalize_path_value(SHARED_BENCHMARK_PW_MATRIX_CONFIG_PATH)
    assert family_manifest["benchmark_protocol_config_path"] == normalize_path_value(
        SHARED_BENCHMARK_PROTOCOL_CONFIG_PATH
    )
    assert family_manifest["benchmark_protocol"]["protocol_id"] == "shared_hardneg_geometry_benchmark_v1"
    assert family_manifest["benchmark_provenance"]["protocol_id"] == "shared_hardneg_geometry_benchmark_v1"
    assert family_manifest["benchmark_provenance"]["schema_version"] == "pw_shared_benchmark_protocol_v1"
    assert family_manifest["sample_roles"]["active"] == [
        "positive_source",
        "clean_negative",
        "planner_conditioned_control_negative",
    ]
    assert family_manifest["sample_roles"]["reserved"] == ["attacked_positive", "attacked_negative"]
    assert family_manifest["benchmark_protocol"]["score_pools"]["content_chain_score"]["calibration_role_order"] == [
        "positive_source",
        "clean_negative",
        "planner_conditioned_control_negative",
    ]
    assert family_manifest["benchmark_protocol"]["score_pools"]["content_chain_score"]["evaluate_role_order"] == [
        "positive_source",
        "clean_negative",
    ]
    assert family_manifest["paths"]["benchmark_protocol_config"] == normalize_path_value(
        SHARED_BENCHMARK_PROTOCOL_CONFIG_PATH
    )
    assert summary["matrix_profile"] == "geometry_shared_benchmark_v1"
    assert summary["attack_condition_count"] == 12
    assert summary["geometry_optional_claim_boundary_abs_margin_max"] == pytest.approx(0.3)
    assert family_manifest["attack_parameters"]["matrix_profile"] == "geometry_shared_benchmark_v1"
    assert method_identity_snapshot["source_alignment_reference_files"] == [
        "paper_workflow/configs/pw_base_geometry_shared_benchmark_v1.yaml",
        "paper_workflow/configs/pw_matrix_geometry_shared_benchmark_v1.yaml",
        "paper_workflow/configs/pw_protocol_shared_hardneg_benchmark_v1.yaml",
        "paper_workflow/scripts/pw_common.py",
        "paper_workflow/scripts/pw00_build_family_manifest.py",
        "paper_workflow/scripts/pw01_stage_runtime_helpers.py",
        "paper_workflow/scripts/pw01_run_source_event_shard.py",
        "paper_workflow/scripts/pw02_merge_source_event_shards.py",
        "paper_workflow/notebook/PW00_Paper_Eval_Family_Manifest.ipynb",
        "paper_workflow/notebook/PW01_Source_Event_Shards.ipynb",
        "paper_workflow/notebook/PW02_Source_Merge_And_Global_Thresholds.ipynb",
        "scripts/notebook_runtime_common.py",
        "configs/default.yaml",
    ]


def test_pilot_base_and_matrix_remain_unchanged_after_geometry_rescue_append_only_addition() -> None:
    """
    Verify the default pilot binding and matrix content remain unchanged.

    Args:
        None.

    Returns:
        None.
    """
    pilot_base_cfg = load_yaml_mapping(PILOT_PW_BASE_CONFIG_PATH)
    pilot_matrix_cfg = load_yaml_mapping(PILOT_PW_MATRIX_CONFIG_PATH)
    pilot_matrix_settings = resolve_pw_matrix_settings(pilot_matrix_cfg)
    pilot_catalog = build_attack_condition_catalog(matrix_cfg=pilot_matrix_cfg)

    assert pilot_base_cfg["matrix_config_path"] == "paper_workflow/configs/pw_matrix_pilot.yaml"
    assert pilot_matrix_cfg["matrix_profile"] == "family_x_severity_pilot_v1"
    assert pilot_matrix_settings["attack_sets"]["general_attacks"] == [
        "rotate",
        "resize",
        "crop",
        "translate",
        "jpeg",
        "gaussian_noise",
        "gaussian_blur",
        "composite",
    ]
    assert pilot_matrix_settings["attack_sets"]["geometry_rescue_candidates"] == [
        "rotate",
        "resize",
        "crop",
        "composite",
    ]
    assert pilot_matrix_settings["geometry_optional_claim"]["boundary_abs_margin_min"] == pytest.approx(0.01)
    assert pilot_matrix_settings["geometry_optional_claim"]["boundary_abs_margin_max"] == pytest.approx(0.25)
    assert len(pilot_catalog) == 19
    assert all("::v2" not in str(row["attack_condition_base_key"]) for row in pilot_catalog)
