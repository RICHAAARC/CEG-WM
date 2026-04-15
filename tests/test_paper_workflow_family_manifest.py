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

from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest
from paper_workflow.scripts.pw_common import build_attack_condition_catalog, read_jsonl
from scripts.notebook_runtime_common import REPO_ROOT, build_repo_import_subprocess_env


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
    assert family_manifest["attack_parameters"]["materialization_profile"] == "protocol_list_cartesian_per_condition"
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
    assert first_summary["materialization_profile"] == "protocol_list_cartesian_per_condition"
    assert first_summary["matrix_profile"] == "family_x_severity_v1"
    assert first_summary["wrong_event_challenge_parent_event_count"] == 4
    assert first_summary["wrong_event_challenge_available_assignment_count"] == 4
    assert Path(str(first_summary["wrong_event_attestation_challenge_plan_path"])) == first_wrong_event_challenge_plan_path
    assert first_summary["severity_metadata_frozen"] is True
    assert first_summary["severity_axis_kind"] == "family_local"
    assert first_summary["severity_status_counts"]["ok"] > 0
    assert first_summary["severity_available_family_count"] > 0


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
