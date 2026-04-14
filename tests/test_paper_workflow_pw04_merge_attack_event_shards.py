"""
File purpose: Validate PW04 attack merge, formal materialization, and export contracts.
Module type: General module
"""

from __future__ import annotations

import builtins
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, cast

import pytest
from PIL import Image

import paper_workflow.scripts.pw03_run_attack_event_shard as pw03_module
import paper_workflow.scripts.pw04_merge_attack_event_shards as pw04_module
import paper_workflow.scripts.pw04_metrics_extensions as pw04_metrics_extensions_module
import paper_workflow.scripts.pw04_run_quality_shard as pw04_quality_shard_module
import paper_workflow.scripts.pw_quality_metrics as pw_quality_metrics_module
from main.watermarking.provenance.attestation_statement import (
    ATTESTATION_SCHEMA,
    AttestationStatement,
    build_signed_attestation_bundle,
    compute_attestation_digest,
)
from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest
from paper_workflow.scripts.pw_common import ACTIVE_SAMPLE_ROLE, read_jsonl
from scripts.notebook_runtime_common import compute_file_sha256, ensure_directory, normalize_path_value, write_json_atomic


@pytest.fixture(autouse=True)
def _force_pw04_png_fallback(monkeypatch: Any) -> None:
    """
    Force PW04 figure generation tests through the builtin PNG fallback path.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    original_import = builtins.__import__

    def patched_import(
        name: str,
        globals_arg: Any = None,
        locals_arg: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("forced matplotlib fallback for PW04 tests")
        return original_import(name, globals_arg, locals_arg, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", patched_import)


def _build_pw00_family(tmp_path: Path, family_id: str) -> Dict[str, Any]:
    """
    Build a minimal PW00 family fixture for PW04 tests.

    Args:
        tmp_path: Pytest temporary directory.
        family_id: Fixture family identifier.

    Returns:
        PW00 summary payload.
    """
    prompt_file = tmp_path / f"{family_id}_prompts.txt"
    prompt_file.write_text("prompt one\nprompt two\n", encoding="utf-8")
    return run_pw00_build_family_manifest(
        drive_project_root=tmp_path / "drive",
        family_id=family_id,
        prompt_file=str(prompt_file),
        seed_list=[7],
        source_shard_count=2,
        attack_shard_count=2,
    )


def _fake_clip_text_similarity(candidate_image: Any, prompt_text: str) -> float:
    """
    Build deterministic fake CLIP similarity for paper-workflow tests.

    Args:
        candidate_image: Unused candidate image payload.
        prompt_text: Prompt text.

    Returns:
        Deterministic fake similarity.
    """
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("prompt_text must be non-empty str")
    return 0.81 if prompt_text == "prompt one" else 0.69


def test_build_quality_metrics_from_pairs_supports_env_gpu_batch_runtime(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """
    Verify quality helpers honor env-driven GPU and batch runtime options.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    pair_specs: List[Dict[str, Any]] = []
    for pair_index in range(3):
        reference_path = tmp_path / f"reference_{pair_index}.png"
        candidate_path = tmp_path / f"candidate_{pair_index}.png"
        Image.new("RGB", (8, 8), color=(32 + pair_index, 64, 96)).save(reference_path)
        Image.new("RGB", (8, 8), color=(36 + pair_index, 64, 96)).save(candidate_path)
        pair_specs.append(
            {
                "pair_id": f"pair_{pair_index}",
                "reference_image_path": normalize_path_value(reference_path),
                "candidate_image_path": normalize_path_value(candidate_path),
                "prompt_text": "prompt one" if pair_index < 2 else "prompt two",
            }
        )

    lpips_batch_calls: List[Dict[str, Any]] = []
    clip_batch_calls: List[Dict[str, Any]] = []

    def fake_lpips_values_batch(
        reference_images: Any,
        candidate_images: Any,
        torch_device: str = "cpu",
    ) -> List[float]:
        lpips_batch_calls.append(
            {
                "batch_size": len(reference_images),
                "torch_device": torch_device,
            }
        )
        return [0.1 + 0.01 * batch_index for batch_index in range(len(reference_images))]

    def fake_clip_text_similarity_batch(
        candidate_images: Any,
        prompt_texts: Any,
        torch_device: str = "cpu",
    ) -> List[float]:
        clip_batch_calls.append(
            {
                "batch_size": len(candidate_images),
                "torch_device": torch_device,
                "prompt_texts": list(prompt_texts),
            }
        )
        return [0.8 - 0.01 * batch_index for batch_index in range(len(candidate_images))]

    def fail_single_lpips(*args: Any, **kwargs: Any) -> float:
        raise AssertionError("single-item LPIPS path should not run when batch mode is enabled")

    def fail_single_clip(*args: Any, **kwargs: Any) -> float:
        raise AssertionError("single-item CLIP path should not run when batch mode is enabled")

    monkeypatch.setenv(pw_quality_metrics_module.QUALITY_TORCH_DEVICE_ENV, "cuda:0")
    monkeypatch.setenv(pw_quality_metrics_module.QUALITY_LPIPS_BATCH_SIZE_ENV, "2")
    monkeypatch.setenv(pw_quality_metrics_module.QUALITY_CLIP_BATCH_SIZE_ENV, "2")
    monkeypatch.setattr(pw_quality_metrics_module, "_compute_lpips_values_batch", fake_lpips_values_batch)
    monkeypatch.setattr(
        pw_quality_metrics_module,
        "_compute_clip_text_similarity_batch",
        fake_clip_text_similarity_batch,
    )
    monkeypatch.setattr(pw_quality_metrics_module, "_compute_lpips_value", fail_single_lpips)
    monkeypatch.setattr(pw_quality_metrics_module, "_compute_clip_text_similarity", fail_single_clip)

    quality_summary = pw_quality_metrics_module.build_quality_metrics_from_pairs(
        pair_specs=pair_specs,
        reference_path_key="reference_image_path",
        candidate_path_key="candidate_image_path",
        pair_id_key="pair_id",
        text_key="prompt_text",
    )

    assert lpips_batch_calls == [
        {"batch_size": 2, "torch_device": "cuda:0"},
        {"batch_size": 1, "torch_device": "cuda:0"},
    ]
    assert [call["batch_size"] for call in clip_batch_calls] == [2, 1]
    assert all(call["torch_device"] == "cuda:0" for call in clip_batch_calls)
    assert quality_summary["quality_runtime"] == {
        "torch_device": "cuda:0",
        "lpips_batch_size": 2,
        "clip_batch_size": 2,
    }
    assert quality_summary["lpips_status"] == "ok"
    assert quality_summary["clip_status"] == "ok"
    assert quality_summary["prompt_text_coverage_status"] == "ok"
    assert quality_summary["quality_readiness_status"] == "ready"
    assert quality_summary["quality_readiness_blocking"] is False
    assert quality_summary["clip_model_name"] == pw_quality_metrics_module.CLIP_MODEL_NAME
    assert all(row["lpips"] is not None for row in cast(List[Dict[str, Any]], quality_summary["pair_rows"]))
    assert all(
        row["clip_text_similarity"] is not None
        for row in cast(List[Dict[str, Any]], quality_summary["pair_rows"])
    )


def _load_json_dict(path_obj: Path) -> Dict[str, Any]:
    """
    Load one JSON object file.

    Args:
        path_obj: JSON file path.

    Returns:
        Parsed JSON object.
    """
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"JSON root must be object: {path_obj}")
    return cast(Dict[str, Any], payload)


def _load_csv_rows(path_obj: Path) -> List[Dict[str, Any]]:
    """
    Load one CSV file into ordered rows.

    Args:
        path_obj: CSV file path.

    Returns:
        Parsed CSV rows.
    """
    with path_obj.open("r", encoding="utf-8", newline="") as handle:
        return [dict(cast(Mapping[str, Any], row)) for row in csv.DictReader(handle)]


def _score_pair_for_index(event_index: int) -> Tuple[float, float]:
    """
    Build deterministic content and attestation scores for one attack event.

    Args:
        event_index: Attack event index.

    Returns:
        Tuple of (content_score, attestation_score).
    """
    pattern_index = event_index % 4
    if pattern_index == 0:
        return 0.91, 0.12
    if pattern_index == 1:
        return 0.21, 0.74
    if pattern_index == 2:
        return 0.19, 0.18
    return 0.83, 0.79


def _image_evidence_payload_for_index(event_index: int) -> Dict[str, Any]:
    """
    Build deterministic image-evidence rescue fields for one attack event.

    Args:
        event_index: Attack event index.

    Returns:
        Image evidence payload.
    """
    pattern_index = event_index % 4
    if pattern_index == 1:
        return {
            "status": "ok",
            "geo_rescue_eligible": True,
            "geo_rescue_applied": True,
            "geo_not_used_reason": None,
        }
    if pattern_index == 2:
        return {
            "status": "ok",
            "geo_rescue_eligible": True,
            "geo_rescue_applied": False,
            "geo_not_used_reason": "geometry_score_below_rescue_min",
        }
    if pattern_index == 0:
        return {
            "status": "ok",
            "geo_rescue_eligible": False,
            "geo_rescue_applied": False,
            "geo_not_used_reason": "content_chain_already_positive",
        }
    return {
        "status": "ok",
        "geo_rescue_eligible": False,
        "geo_rescue_applied": False,
        "geo_not_used_reason": "attestation_already_positive",
    }


def _build_pw02_fixture(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Materialize the minimal PW02 outputs consumed by PW04.

    Args:
        summary: PW00 summary payload.

    Returns:
        PW02 fixture metadata.
    """
    family_root = Path(str(summary["family_root"]))
    family_id = str(summary["family_id"])
    pw02_root = ensure_directory(family_root / "exports" / "pw02")
    thresholds_root = ensure_directory(pw02_root / "thresholds")
    content_thresholds_artifact_path = thresholds_root / "content" / "thresholds_artifact.json"
    attestation_thresholds_artifact_path = thresholds_root / "attestation" / "thresholds_artifact.json"
    ensure_directory(content_thresholds_artifact_path.parent)
    ensure_directory(attestation_thresholds_artifact_path.parent)

    content_thresholds_artifact = {
        "threshold_id": "content_np_0p01",
        "score_name": "content_chain_score",
        "target_fpr": 0.01,
        "threshold_value": 0.5,
        "threshold_key_used": "0p01",
        "decision_operator": "score_greater_equal_threshold_value",
        "selected_order_stat_score": 0.5,
    }
    attestation_thresholds_artifact = {
        "threshold_id": "attestation_np_0p01",
        "score_name": "event_attestation_score",
        "target_fpr": 0.01,
        "threshold_value": 0.6,
        "threshold_key_used": "0p01",
        "decision_operator": "score_greater_equal_threshold_value",
        "selected_order_stat_score": 0.6,
    }
    write_json_atomic(content_thresholds_artifact_path, content_thresholds_artifact)
    write_json_atomic(attestation_thresholds_artifact_path, attestation_thresholds_artifact)

    content_threshold_export_path = thresholds_root / "content" / "thresholds.json"
    attestation_threshold_export_path = thresholds_root / "attestation" / "thresholds.json"
    write_json_atomic(
        content_threshold_export_path,
        {
            "artifact_type": "paper_workflow_pw02_threshold_export",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "score_name": "content_chain_score",
            "source_thresholds_artifact_path": normalize_path_value(content_thresholds_artifact_path),
            "thresholds_artifact": content_thresholds_artifact,
            "calibration_record_status": "ok",
        },
    )
    write_json_atomic(
        attestation_threshold_export_path,
        {
            "artifact_type": "paper_workflow_pw02_threshold_export",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "score_name": "event_attestation_score",
            "source_thresholds_artifact_path": normalize_path_value(attestation_thresholds_artifact_path),
            "thresholds_artifact": attestation_thresholds_artifact,
            "calibration_record_status": "ok",
        },
    )

    formal_clean_metrics_path = pw02_root / "formal_final_decision_metrics.json"
    derived_clean_metrics_path = pw02_root / "derived_system_union_metrics.json"
    quality_root = ensure_directory(pw02_root / "quality")
    clean_quality_pair_manifest_path = quality_root / "clean_quality_pair_manifest.json"
    payload_root = ensure_directory(pw02_root / "payload")
    payload_clean_summary_path = payload_root / "payload_clean_summary.json"
    clean_evaluate_root = ensure_directory(pw02_root / "evaluate" / "clean")
    content_clean_evaluate_export_path = clean_evaluate_root / "content" / "evaluate_record.json"
    attestation_clean_evaluate_export_path = clean_evaluate_root / "attestation" / "evaluate_record.json"
    ensure_directory(content_clean_evaluate_export_path.parent)
    ensure_directory(attestation_clean_evaluate_export_path.parent)
    write_json_atomic(
        formal_clean_metrics_path,
        {
            "artifact_type": "paper_workflow_pw02_formal_final_decision_metrics",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "metrics": {
                "scope": "formal_final_decision",
                "n_total": 8,
                "n_positive": 4,
                "n_negative": 4,
                "final_decision_available_rate": 1.0,
                "content_chain_available_rate": 1.0,
                "final_decision_tpr": 1.0,
                "final_decision_fpr": 0.0,
                "final_decision_status_counts": {
                    "accept": 4,
                    "reject": 4,
                },
            },
        },
    )
    write_json_atomic(
        derived_clean_metrics_path,
        {
            "artifact_type": "paper_workflow_pw02_derived_system_union_metrics",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "metrics": {
                "scope": "system_final",
                "n_total": 8,
                "n_positive": 4,
                "n_negative": 4,
                "final_decision_available_rate": 1.0,
                "content_chain_available_rate": 1.0,
                "image_evidence_ok_rate": 1.0,
                "event_attestation_available_rate": 1.0,
                "geo_rescue_eligible_rate": 0.5,
                "system_tpr": 1.0,
                "system_fpr": 0.0,
                "final_decision_tpr": 1.0,
                "final_decision_fpr": 0.0,
                "event_attestation_tpr": 0.75,
                "event_attestation_fpr": 0.25,
                "geo_rescue_applied_rate": 0.25,
                "final_decision_status_counts": {
                    "accept": 4,
                    "reject": 4,
                },
                "geo_not_used_reason_counts": {
                    "geometry_score_below_rescue_min": 1,
                },
            },
        },
    )
    write_json_atomic(
        content_clean_evaluate_export_path,
        {
            "artifact_type": "paper_workflow_pw02_clean_evaluate_export",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "score_name": "content_chain_score",
            "evaluate_record": {
                "status": "completed",
                "metrics": {
                    "n_total": 8,
                    "n_pos": 4,
                    "n_neg": 4,
                    "tpr_at_fpr_primary": 1.0,
                    "fpr_empirical": 0.0,
                },
                "breakdown": {
                    "confusion": {
                        "tp": 4,
                        "fp": 0,
                        "fn": 0,
                        "tn": 4,
                    }
                },
            },
        },
    )
    write_json_atomic(
        attestation_clean_evaluate_export_path,
        {
            "artifact_type": "paper_workflow_pw02_clean_evaluate_export",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "score_name": "event_attestation_score",
            "evaluate_record": {
                "status": "completed",
                "metrics": {
                    "n_total": 8,
                    "n_pos": 4,
                    "n_neg": 4,
                    "tpr_at_fpr_primary": 0.75,
                    "fpr_empirical": 0.25,
                },
                "breakdown": {
                    "confusion": {
                        "tp": 3,
                        "fp": 1,
                        "fn": 1,
                        "tn": 3,
                    }
                },
            },
        },
    )
    clean_pair_rows: List[Dict[str, Any]] = []
    for pair_index, prompt_text in enumerate(["prompt one", "prompt two"], start=1):
        reference_path = quality_root / f"plain_preview_{pair_index:06d}.png"
        candidate_path = quality_root / f"watermarked_output_{pair_index:06d}.png"
        Image.new("RGB", (8, 8), color=(80 + pair_index, 110, 140)).save(reference_path)
        Image.new("RGB", (8, 8), color=(82 + pair_index, 110, 140)).save(candidate_path)
        clean_pair_rows.append(
            {
                "event_id": f"source_event_{pair_index:06d}",
                "reference_image_path": normalize_path_value(reference_path),
                "candidate_image_path": normalize_path_value(candidate_path),
                "prompt_text": prompt_text,
                "sample_role": ACTIVE_SAMPLE_ROLE,
                "plain_preview_image_path": normalize_path_value(reference_path),
                "watermarked_output_image_path": normalize_path_value(candidate_path),
            }
        )
    write_json_atomic(
        clean_quality_pair_manifest_path,
        {
            "artifact_type": "paper_workflow_pw02_clean_quality_pair_manifest",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "score_name": "content_chain_score",
            "scope": "content_chain",
            "pair_id_key": "event_id",
            "reference_path_key": "reference_image_path",
            "candidate_path_key": "candidate_image_path",
            "text_key": "prompt_text",
            "reference_artifact_name": "plain_preview_image",
            "candidate_artifact_name": "watermarked_output_image",
            "reference_semantics": "preview_generation_persisted_artifact_vs_watermarked_output_image",
            "pair_rows": clean_pair_rows,
        },
    )
    write_json_atomic(
        payload_clean_summary_path,
        {
            "artifact_type": "paper_workflow_pw02_payload_clean_summary",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "status": "not_available",
            "reason": "missing upstream decoded bits / reference bits / bit error sidecar",
            "future_upstream_sidecar_required": True,
        },
    )

    finalize_manifest_path = pw02_root / "paper_source_finalize_manifest.json"
    write_json_atomic(
        finalize_manifest_path,
        {
            "artifact_type": "paper_workflow_pw02_finalize_manifest",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "threshold_exports": {
                "content": {"path": normalize_path_value(content_threshold_export_path)},
                "attestation": {"path": normalize_path_value(attestation_threshold_export_path)},
            },
            "source_pools": {},
        },
    )
    finalize_manifest_digest = compute_file_sha256(finalize_manifest_path)

    pw02_summary_path = family_root / "runtime_state" / "pw02_summary.json"
    write_json_atomic(
        pw02_summary_path,
        {
            "status": "completed",
            "family_id": family_id,
            "paper_source_finalize_manifest_path": normalize_path_value(finalize_manifest_path),
            "formal_final_decision_metrics_artifact_path": normalize_path_value(formal_clean_metrics_path),
            "derived_system_union_metrics_artifact_path": normalize_path_value(derived_clean_metrics_path),
            "clean_pair_artifacts_dir": normalize_path_value(quality_root),
            "clean_quality_pair_manifest_path": normalize_path_value(clean_quality_pair_manifest_path),
            "payload_metrics_dir": normalize_path_value(payload_root),
            "payload_clean_summary_path": normalize_path_value(payload_clean_summary_path),
            "analysis_only_artifact_paths": {
                "pw02_clean_quality_pair_manifest": normalize_path_value(clean_quality_pair_manifest_path),
            },
            "analysis_only_artifact_annotations": {
                "pw02_clean_quality_pair_manifest": {"canonical": False, "analysis_only": True},
            },
            "clean_evaluate_exports": {
                "content": normalize_path_value(content_clean_evaluate_export_path),
                "attestation": normalize_path_value(attestation_clean_evaluate_export_path),
            },
        },
    )

    return {
        "family_root": family_root,
        "family_id": family_id,
        "pw02_summary_path": pw02_summary_path,
        "finalize_manifest_path": finalize_manifest_path,
        "finalize_manifest_digest": finalize_manifest_digest,
        "content_threshold_export_path": content_threshold_export_path,
        "attestation_threshold_export_path": attestation_threshold_export_path,
        "formal_clean_metrics_path": formal_clean_metrics_path,
        "derived_clean_metrics_path": derived_clean_metrics_path,
        "clean_quality_pair_manifest_path": clean_quality_pair_manifest_path,
        "payload_clean_summary_path": payload_clean_summary_path,
        "content_clean_evaluate_export_path": content_clean_evaluate_export_path,
        "attestation_clean_evaluate_export_path": attestation_clean_evaluate_export_path,
    }


def _build_pw03_fixture(summary: Dict[str, Any], pw02_fixture: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Materialize completed PW03 shard manifests, event manifests, and staged detect records.

    Args:
        summary: PW00 summary payload.
        pw02_fixture: PW02 fixture metadata.

    Returns:
        PW03 fixture metadata.
    """
    family_root = Path(str(summary["family_root"]))
    attack_event_grid = read_jsonl(Path(str(summary["attack_event_grid_path"])))
    attack_shard_plan = _load_json_dict(Path(str(summary["attack_shard_plan_path"])))
    attack_event_lookup = {
        str(row.get("attack_event_id", row.get("event_id"))): row
        for row in attack_event_grid
    }
    wrong_event_challenge_plan = _load_json_dict(
        Path(str(summary["wrong_event_attestation_challenge_plan_path"]))
    )
    wrong_event_challenge_lookup = {
        str(row["parent_event_id"]): row
        for row in cast(List[Dict[str, Any]], wrong_event_challenge_plan["rows"])
    }
    threshold_artifact_paths = {
        "content": normalize_path_value(Path(str(pw02_fixture["content_threshold_export_path"]))),
        "attestation": normalize_path_value(Path(str(pw02_fixture["attestation_threshold_export_path"]))),
    }

    expected_positive_attack_event_count = 0
    expected_negative_attack_event_count = 0
    expected_formal_final_positive_count = 0
    expected_formal_attestation_positive_count = 0
    expected_derived_union_positive_count = 0
    expected_formal_final_negative_false_accept_count = 0
    expected_formal_attestation_negative_false_accept_count = 0
    expected_derived_union_negative_false_accept_count = 0
    expected_payload_codeword_agreement_values: List[float] = []
    expected_payload_attestation_score_values: List[float] = []
    attestation_master_key = "5" * 64
    parent_attestation_materials: Dict[str, Dict[str, Any]] = {}
    detect_record_paths: List[Path] = []
    event_manifest_paths: List[Path] = []

    parent_event_index_lookup: Dict[str, int] = {}
    for attack_event in cast(List[Dict[str, Any]], attack_event_grid):
        parent_event_id = str(attack_event["parent_event_id"])
        parent_event_index_lookup.setdefault(parent_event_id, int(attack_event["parent_event_index"]))
    for parent_event_id, parent_event_index in sorted(parent_event_index_lookup.items(), key=lambda item: item[1]):
        statement = AttestationStatement(
            schema=ATTESTATION_SCHEMA,
            model_id="sd3-test",
            prompt_commit=f"prompt_commit_{parent_event_id}",
            seed_commit=f"seed_commit_{parent_event_id}",
            plan_digest=f"plan_digest_{parent_event_id}",
            event_nonce=f"nonce_{parent_event_id}",
            time_bucket="2026-03-01",
        )
        attestation_digest = compute_attestation_digest(statement)
        signed_bundle = build_signed_attestation_bundle(
            statement,
            attestation_digest,
            attestation_master_key,
            lf_payload_hex="ab" * 16,
            trace_commit="cd" * 32,
            geo_anchor_seed=parent_event_index,
        )
        parent_embed_record_path = family_root / "source_attestation" / f"{parent_event_id}_embed_record.json"
        ensure_directory(parent_embed_record_path.parent)
        write_json_atomic(
            parent_embed_record_path,
            {
                "attestation": {
                    "status": "ok",
                    "statement": statement.as_dict(),
                    "attestation_digest": attestation_digest,
                    "signed_bundle": signed_bundle,
                }
            },
        )
        parent_attestation_materials[parent_event_id] = {
            "parent_embed_record_path": normalize_path_value(parent_embed_record_path),
            "statement": statement.as_dict(),
            "attestation_digest": attestation_digest,
            "signed_bundle": signed_bundle,
        }

    for shard_row in cast(List[Dict[str, Any]], attack_shard_plan["shards"]):
        attack_shard_index = int(shard_row["attack_shard_index"])
        shard_root = ensure_directory(family_root / "attack_shards" / f"shard_{attack_shard_index:04d}")
        ensure_directory(shard_root / "records")
        events_payload: List[Dict[str, Any]] = []
        for attack_event_id in cast(List[str], shard_row["assigned_attack_event_ids"]):
            attack_event = cast(Dict[str, Any], attack_event_lookup[attack_event_id])
            attack_event_index = int(attack_event["attack_event_index"])
            event_root = ensure_directory(shard_root / "events" / f"event_{attack_event_index:06d}")
            artifacts_root = ensure_directory(event_root / "artifacts")
            parent_source_image_path = artifacts_root / f"parent_{attack_event_index:06d}.png"
            attacked_image_path = artifacts_root / f"event_{attack_event_index:06d}.png"
            parent_source_image = Image.new("RGB", (8, 8), color=(60 + attack_event_index, 90, 120))
            attacked_image = Image.new("RGB", (8, 8), color=(62 + attack_event_index, 90, 120))
            parent_source_image.save(parent_source_image_path)
            attacked_image.save(attacked_image_path)

            content_score, attestation_score = _score_pair_for_index(attack_event_index)
            attack_sample_role = str(attack_event["sample_role"])
            codeword_agreement = 0.6 + 0.05 * float(attack_event_index % 4)
            if attack_sample_role == pw04_module.ATTACKED_POSITIVE_SAMPLE_ROLE:
                expected_positive_attack_event_count += 1
                expected_payload_codeword_agreement_values.append(codeword_agreement)
                expected_payload_attestation_score_values.append(attestation_score)
                if content_score >= 0.5:
                    expected_formal_final_positive_count += 1
                if attestation_score >= 0.6:
                    expected_formal_attestation_positive_count += 1
                if content_score >= 0.5 or attestation_score >= 0.6:
                    expected_derived_union_positive_count += 1
            else:
                expected_negative_attack_event_count += 1
                if content_score >= 0.5:
                    expected_formal_final_negative_false_accept_count += 1
                if attestation_score >= 0.6:
                    expected_formal_attestation_negative_false_accept_count += 1
                if content_score >= 0.5 or attestation_score >= 0.6:
                    expected_derived_union_negative_false_accept_count += 1

            severity_metadata = {
                "severity_status": attack_event.get("severity_status"),
                "severity_reason": attack_event.get("severity_reason"),
                "severity_rule_version": attack_event.get("severity_rule_version"),
                "severity_axis_kind": attack_event.get("severity_axis_kind"),
                "severity_directionality": attack_event.get("severity_directionality"),
                "severity_source_param": attack_event.get("severity_source_param"),
                "severity_scalarization": attack_event.get("severity_scalarization"),
                "severity_value": attack_event.get("severity_value"),
                "severity_sort_value": attack_event.get("severity_sort_value"),
                "severity_label": attack_event.get("severity_label"),
                "severity_level_index": attack_event.get("severity_level_index"),
            }
            geometry_diagnostics = {
                "sync_status": "ok" if attack_event_index % 2 == 0 else "degraded",
                "sync_success": attack_event_index % 2 == 0,
                "sync_success_status": "ok",
                "sync_success_reason": None,
                "sync_digest": f"sync_digest_{attack_event_index:06d}",
                "geometry_failure_reason": None if attack_event_index % 2 == 0 else "template_match_below_threshold",
                "geometry_failure_reason_status": "not_available" if attack_event_index % 2 == 0 else "ok",
                "geometry_failure_reason_reason": None if attack_event_index % 2 != 0 else "geometry chain did not report failure reason",
                "relation_digest_bound": attack_event_index % 2 == 0,
                "template_match_metrics": {
                    "peak_value": 0.8 - 0.01 * (attack_event_index % 3),
                },
                "sync_quality_metrics": {
                    "match_score": 0.9 - 0.02 * (attack_event_index % 3),
                },
                "sync_quality_metrics_status": "ok",
                "sync_quality_metrics_reason": None,
                "inverse_transform_success": attack_event_index % 3 != 2,
                "inverse_transform_success_status": "ok",
                "inverse_transform_success_reason": None,
                "attention_anchor_available": attack_event_index % 4 != 3,
                "attention_anchor_available_status": "ok",
                "attention_anchor_available_reason": None,
                "anchor_digest": f"anchor_digest_{attack_event_index:06d}" if attack_event_index % 4 != 3 else None,
            }
            parent_prompt_text = str(attack_event.get("prompt_text") or ("prompt one" if attack_event_index % 2 == 0 else "prompt two"))
            parent_event_id = str(attack_event["parent_event_id"])
            parent_material = parent_attestation_materials[parent_event_id]

            runtime_config_snapshot_path = event_root / "runtime_config_snapshot.json"
            write_json_atomic(
                runtime_config_snapshot_path,
                {
                    "__attestation_verify_k_master__": attestation_master_key,
                    "attestation": {
                        "use_trajectory_mix": True,
                    },
                },
            )

            detect_record_path = shard_root / "records" / f"event_{attack_event_index:06d}_detect_record.json"
            detect_payload = {
                "sample_role": attack_sample_role,
                "parent_event_id": None,
                "paper_workflow_parent_event_id": attack_event["parent_event_id"],
                "paper_workflow_attack_event_id": attack_event_id,
                "paper_workflow_attack_family": attack_event["attack_family"],
                "paper_workflow_attack_config_name": attack_event["attack_config_name"],
                "paper_workflow_attack_condition_key": attack_event["attack_condition_key"],
                "paper_workflow_attack_params_digest": attack_event["attack_params_digest"],
                "paper_workflow_parent_source_image_path": normalize_path_value(parent_source_image_path),
                "paper_workflow_severity_metadata": severity_metadata,
                "paper_workflow_geometry_diagnostics": geometry_diagnostics,
                "content_evidence_payload": {
                    "status": "ok",
                    "content_chain_score": content_score,
                    "score_parts": {
                        "lf_trajectory_detect_trace": {
                            "codeword_agreement": codeword_agreement,
                            "n_bits_compared": 96,
                            "detect_variant": "correlation_v2",
                            "message_source": "attestation_event_digest",
                        }
                    },
                },
                "attestation": {
                    "final_event_attested_decision": {
                        "status": "ok",
                        "is_event_attested": attestation_score >= 0.6,
                        "event_attestation_score_name": "event_attestation_score",
                        "event_attestation_score": attestation_score,
                    },
                    "image_evidence_result": _image_evidence_payload_for_index(attack_event_index),
                },
                "final_decision": {
                    "decision_status": "accept" if content_score >= 0.5 else "reject",
                    "is_watermarked": content_score >= 0.5,
                },
                "fusion_result": {
                    "decision_status": "accept" if content_score >= 0.5 else "reject",
                },
                "geometry_evidence_payload": {
                    "status": "ok",
                    "anchor_digest": geometry_diagnostics["anchor_digest"],
                    "align_metrics": {
                        "inverse_recovery_success": geometry_diagnostics["inverse_transform_success"],
                    },
                    "sync_quality_metrics": geometry_diagnostics["sync_quality_metrics"],
                    "template_match_metrics": geometry_diagnostics["template_match_metrics"],
                },
                "geometry_result": {
                    "sync_status": geometry_diagnostics["sync_status"],
                    "sync_digest": geometry_diagnostics["sync_digest"],
                    "relation_digest_bound": geometry_diagnostics["relation_digest_bound"],
                    "sync_result": {
                        "sync_success": geometry_diagnostics["sync_success"],
                        "failure_reason": geometry_diagnostics["geometry_failure_reason"],
                        "sync_quality_metrics": geometry_diagnostics["sync_quality_metrics"],
                        "template_match_metrics": geometry_diagnostics["template_match_metrics"],
                    },
                },
                "lf_detect_variant": "lf_v1",
            }
            write_json_atomic(detect_record_path, detect_payload)
            detect_record_paths.append(detect_record_path)

            wrong_event_challenge_record_path: Path | None = None
            wrong_event_challenge_record_payload: Dict[str, Any] | None = None
            if attack_sample_role == pw04_module.ATTACKED_POSITIVE_SAMPLE_ROLE:
                challenge_assignment = cast(Dict[str, Any], wrong_event_challenge_lookup[parent_event_id])
                challenge_parent_event_id = challenge_assignment.get("challenge_parent_event_id")
                wrong_event_challenge_record_path = artifacts_root / "wrong_event_attestation_challenge_record.json"
                if isinstance(challenge_parent_event_id, str) and challenge_parent_event_id:
                    challenge_parent_material = parent_attestation_materials[challenge_parent_event_id]
                    wrong_event_challenge_record_payload = {
                        "artifact_type": "paper_workflow_wrong_event_attestation_challenge_record",
                        "schema_version": "pw_stage_03_v1",
                        "stage_name": "PW03_Attack_Event_Shards",
                        "family_id": summary["family_id"],
                        "attack_event_id": attack_event_id,
                        "parent_event_id": parent_event_id,
                        "challenge_parent_event_id": challenge_parent_event_id,
                        "challenge_parent_event_index": challenge_assignment.get("challenge_parent_event_index"),
                        "plan_status": challenge_assignment.get("status"),
                        "plan_reason": challenge_assignment.get("reason"),
                        "plan_policy": challenge_assignment.get("assignment_policy"),
                        "challenge_plan_path": summary["wrong_event_attestation_challenge_plan_path"],
                        "status": "ok",
                        "reason": None,
                        "bundle_verification_status": "ok",
                        "bundle_verification_mismatch_reasons": [],
                        "wrong_statement_digest": challenge_parent_material["attestation_digest"],
                        "bundle_attestation_digest": cast(Dict[str, Any], parent_material["signed_bundle"])["attestation_digest"],
                        "wrong_event_rejected": True,
                    }
                else:
                    wrong_event_challenge_record_payload = {
                        "artifact_type": "paper_workflow_wrong_event_attestation_challenge_record",
                        "schema_version": "pw_stage_03_v1",
                        "stage_name": "PW03_Attack_Event_Shards",
                        "family_id": summary["family_id"],
                        "attack_event_id": attack_event_id,
                        "parent_event_id": parent_event_id,
                        "challenge_parent_event_id": None,
                        "challenge_parent_event_index": None,
                        "plan_status": challenge_assignment.get("status"),
                        "plan_reason": challenge_assignment.get("reason"),
                        "plan_policy": challenge_assignment.get("assignment_policy"),
                        "challenge_plan_path": summary["wrong_event_attestation_challenge_plan_path"],
                        "status": "not_available",
                        "reason": challenge_assignment.get("reason"),
                        "bundle_verification_status": None,
                        "bundle_verification_mismatch_reasons": [],
                        "wrong_statement_digest": None,
                        "bundle_attestation_digest": None,
                        "wrong_event_rejected": None,
                    }
                write_json_atomic(wrong_event_challenge_record_path, wrong_event_challenge_record_payload)

            event_manifest_path = event_root / "event_manifest.json"
            event_manifest_payload = {
                "artifact_type": "paper_workflow_attack_event",
                "schema_version": "pw_stage_03_v1",
                "stage_name": "PW03_Attack_Event_Shards",
                "status": "completed",
                "event_id": attack_event_id,
                "attack_event_id": attack_event_id,
                "attack_event_index": attack_event_index,
                "sample_role": attack_sample_role,
                "parent_event_id": attack_event["parent_event_id"],
                "parent_event_reference": {
                    "prompt_text": parent_prompt_text,
                    "parent_embed_record_path": parent_material["parent_embed_record_path"],
                },
                "parent_source_image_path": normalize_path_value(parent_source_image_path),
                "attack_family": attack_event["attack_family"],
                "attack_config_name": attack_event["attack_config_name"],
                "attack_condition_key": attack_event["attack_condition_key"],
                "attack_params_digest": attack_event["attack_params_digest"],
                "severity_metadata": severity_metadata,
                "geometry_diagnostics": geometry_diagnostics,
                "runtime_config_snapshot_path": normalize_path_value(runtime_config_snapshot_path),
                "source_finalize_manifest_digest": str(pw02_fixture["finalize_manifest_digest"]),
                "threshold_artifact_paths": threshold_artifact_paths,
                "attacked_image_path": normalize_path_value(attacked_image_path),
                "detect_record_path": normalize_path_value(detect_record_path),
                "wrong_event_attestation_challenge_record_path": normalize_path_value(
                    wrong_event_challenge_record_path
                )
                if isinstance(wrong_event_challenge_record_path, Path)
                else None,
                "wrong_event_attestation_challenge_record_package_relative_path": wrong_event_challenge_record_path.relative_to(
                    shard_root
                ).as_posix()
                if isinstance(wrong_event_challenge_record_path, Path)
                else None,
                "wrong_event_attestation_challenge_record": wrong_event_challenge_record_payload,
            }
            write_json_atomic(event_manifest_path, event_manifest_payload)
            event_manifest_paths.append(event_manifest_path)
            events_payload.append(event_manifest_payload)

        shard_manifest_path = shard_root / "shard_manifest.json"
        write_json_atomic(
            shard_manifest_path,
            {
                "artifact_type": "paper_workflow_attack_shard_manifest",
                "schema_version": "pw_stage_03_v1",
                "family_id": summary["family_id"],
                "sample_role": pw04_module.MIXED_ATTACK_SAMPLE_ROLE,
                "attack_shard_index": attack_shard_index,
                "status": "completed",
                "event_count": len(events_payload),
                "completed_event_count": len(events_payload),
                "failed_event_count": 0,
                "event_ids": [str(event_payload["event_id"]) for event_payload in events_payload],
                "assigned_attack_event_ids": list(cast(List[str], shard_row["assigned_attack_event_ids"])),
                "events": events_payload,
            },
        )

    expected_attack_event_count = int(attack_shard_plan["attack_event_count"])
    unique_positive_parent_count = len(
        {
            str(row["parent_event_id"])
            for row in cast(List[Dict[str, Any]], attack_event_grid)
            if str(row["sample_role"]) == pw04_module.ATTACKED_POSITIVE_SAMPLE_ROLE
        }
    )
    return {
        "family_root": family_root,
        "attack_event_grid": attack_event_grid,
        "attack_shard_plan": attack_shard_plan,
        "attack_event_lookup": attack_event_lookup,
        "event_manifest_paths": event_manifest_paths,
        "detect_record_paths": detect_record_paths,
        "expected_attack_event_count": expected_attack_event_count,
        "expected_positive_attack_event_count": expected_positive_attack_event_count,
        "expected_negative_attack_event_count": expected_negative_attack_event_count,
        "expected_formal_final_positive_count": expected_formal_final_positive_count,
        "expected_formal_attestation_positive_count": expected_formal_attestation_positive_count,
        "expected_derived_union_positive_count": expected_derived_union_positive_count,
        "expected_formal_final_negative_false_accept_count": expected_formal_final_negative_false_accept_count,
        "expected_formal_attestation_negative_false_accept_count": expected_formal_attestation_negative_false_accept_count,
        "expected_derived_union_negative_false_accept_count": expected_derived_union_negative_false_accept_count,
        "expected_payload_mean_codeword_agreement": sum(expected_payload_codeword_agreement_values) / len(expected_payload_codeword_agreement_values),
        "expected_payload_min_codeword_agreement": min(expected_payload_codeword_agreement_values),
        "expected_payload_max_codeword_agreement": max(expected_payload_codeword_agreement_values),
        "expected_payload_mean_attestation_score": sum(expected_payload_attestation_score_values) / len(expected_payload_attestation_score_values),
        "expected_wrong_event_challenge_attempt_count": expected_positive_attack_event_count,
        "expected_wrong_event_rejected_count": expected_positive_attack_event_count if unique_positive_parent_count >= 2 else 0,
    }


def _build_pw04_fixture(tmp_path: Path, family_id: str) -> Dict[str, Any]:
    """
    Build the full PW00 + PW02 + PW03 fixture consumed by PW04.

    Args:
        tmp_path: Pytest temporary directory.
        family_id: Fixture family identifier.

    Returns:
        Full PW04 fixture metadata.
    """
    summary = _build_pw00_family(tmp_path, family_id)
    pw02_fixture = _build_pw02_fixture(summary)
    pw03_fixture = _build_pw03_fixture(summary, pw02_fixture)
    return {
        "summary": summary,
        "pw02": pw02_fixture,
        "pw03": pw03_fixture,
    }


def test_pw04_merge_attack_event_shards_success_path(tmp_path: Path, monkeypatch: Any) -> None:
    """
    Verify PW04 merges all completed PW03 shards and exports the required artifacts.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    monkeypatch.setattr(
        pw_quality_metrics_module,
        "_compute_clip_text_similarity",
        _fake_clip_text_similarity,
    )
    fixture = _build_pw04_fixture(tmp_path, "family_pw04_success")
    summary = cast(Dict[str, Any], fixture["summary"])
    pw03_fixture = cast(Dict[str, Any], fixture["pw03"])

    prepare_summary = pw04_module.run_pw04_merge_attack_event_shards(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw04_success",
        pw04_mode=pw04_module.PW04_MODE_PREPARE,
    )

    prepare_manifest_path = Path(str(prepare_summary["prepare_manifest_path"]))
    prepared_quality_pair_plan_path = Path(str(prepare_summary["quality_pair_plan_path"]))
    prepared_quality_shard_paths = [
        Path(str(path_value)) for path_value in cast(List[str], prepare_summary["expected_quality_shard_paths"])
    ]
    prepare_manifest = _load_json_dict(prepare_manifest_path)
    assert prepare_summary["pw04_mode"] == pw04_module.PW04_MODE_PREPARE
    assert prepare_summary["status"] == "completed"
    assert prepare_manifest_path.exists()
    assert prepared_quality_pair_plan_path.exists()
    assert prepare_manifest["quality_shard_count"] == len(prepared_quality_shard_paths)
    assert not (Path(str(summary["family_root"])) / "runtime_state" / "pw04_summary.json").exists()
    assert not (Path(str(summary["family_root"])) / "exports" / "pw04" / "metrics" / "clean_quality_metrics.json").exists()
    assert not (Path(str(summary["family_root"])) / "exports" / "pw04" / "metrics" / "attack_quality_metrics.json").exists()

    for shard_index in range(len(prepared_quality_shard_paths)):
        shard_summary = pw04_module.run_pw04_merge_attack_event_shards(
            drive_project_root=tmp_path / "drive",
            family_id="family_pw04_success",
            pw04_mode=pw04_module.PW04_MODE_QUALITY_SHARD,
            quality_shard_index=shard_index,
        )
        assert shard_summary["pw04_mode"] == pw04_module.PW04_MODE_QUALITY_SHARD
        assert shard_summary["quality_shard_index"] == shard_index
        assert Path(str(shard_summary["quality_shard_path"])).exists()

    pw04_summary = pw04_module.run_pw04_merge_attack_event_shards(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw04_success",
        pw04_mode=pw04_module.PW04_MODE_FINALIZE,
    )

    canonical_metrics_paths = cast(Dict[str, str], pw04_summary["canonical_metrics_paths"])
    paper_tables_paths = cast(Dict[str, str], pw04_summary["paper_tables_paths"])
    paper_figures_paths = cast(Dict[str, str], pw04_summary["paper_figures_paths"])
    tail_estimation_paths = cast(Dict[str, str], pw04_summary["tail_estimation_paths"])
    clean_quality_metrics_path = Path(str(pw04_summary["clean_quality_metrics_path"]))
    attack_quality_metrics_path = Path(str(pw04_summary["attack_quality_metrics_path"]))
    quality_pair_plan_path = Path(str(pw04_summary["quality_pair_plan_path"]))
    quality_finalize_manifest_path = Path(str(pw04_summary["quality_finalize_manifest_path"]))
    quality_shard_paths = [Path(str(path_value)) for path_value in cast(List[str], pw04_summary["quality_shard_paths"])]
    robustness_curve_by_family_path = Path(str(pw04_summary["robustness_curve_by_family_path"]))
    robustness_macro_summary_path = Path(str(pw04_summary["robustness_macro_summary_path"]))
    worst_case_attack_summary_path = Path(str(pw04_summary["worst_case_attack_summary_path"]))
    geo_chain_usage_by_family_path = Path(str(pw04_summary["geo_chain_usage_by_family_path"]))
    geo_diagnostics_summary_path = Path(str(pw04_summary["geo_diagnostics_summary_path"]))
    geo_diagnostics_conditional_metrics_path = Path(str(pw04_summary["geo_diagnostics_conditional_metrics_path"]))
    conditional_rescue_metrics_path = Path(str(pw04_summary["conditional_rescue_metrics_path"]))
    geometry_optional_claim_summary_path = Path(str(pw04_summary["geometry_optional_claim_summary_path"]))
    payload_attack_summary_path = Path(str(pw04_summary["payload_attack_summary_path"]))
    wrong_event_attestation_challenge_summary_path = Path(str(pw04_summary["wrong_event_attestation_challenge_summary_path"]))
    quality_robustness_tradeoff_path = Path(str(pw04_summary["quality_robustness_tradeoff_path"]))
    quality_robustness_frontier_path = Path(str(pw04_summary["quality_robustness_frontier_path"]))
    system_final_auxiliary_attack_summary_path = Path(str(pw04_summary["system_final_auxiliary_attack_summary_path"]))
    system_final_auxiliary_attack_by_family_path = Path(str(pw04_summary["system_final_auxiliary_attack_by_family_path"]))
    system_final_auxiliary_attack_by_condition_path = Path(str(pw04_summary["system_final_auxiliary_attack_by_condition_path"]))
    attack_negative_pool_manifest_path = Path(str(pw04_summary["attack_negative_pool_manifest_path"]))
    formal_attack_negative_metrics_path = Path(str(pw04_summary["formal_attack_negative_metrics_path"]))

    required_paths = [
        Path(str(pw04_summary["summary_path"])),
        Path(str(pw04_summary["attack_merge_manifest_path"])),
        Path(str(pw04_summary["attack_positive_pool_manifest_path"])),
        attack_negative_pool_manifest_path,
        Path(str(pw04_summary["formal_attack_final_decision_metrics_path"])),
        Path(str(pw04_summary["formal_attack_attestation_metrics_path"])),
        Path(str(pw04_summary["derived_attack_union_metrics_path"])),
        formal_attack_negative_metrics_path,
        Path(str(pw04_summary["per_attack_family_metrics_path"])),
        Path(str(pw04_summary["per_attack_condition_metrics_path"])),
        clean_quality_metrics_path,
        attack_quality_metrics_path,
        quality_pair_plan_path,
        quality_finalize_manifest_path,
        *quality_shard_paths,
        robustness_curve_by_family_path,
        robustness_macro_summary_path,
        worst_case_attack_summary_path,
        geo_chain_usage_by_family_path,
        geo_diagnostics_summary_path,
        geo_diagnostics_conditional_metrics_path,
        conditional_rescue_metrics_path,
        geometry_optional_claim_summary_path,
        payload_attack_summary_path,
        wrong_event_attestation_challenge_summary_path,
        quality_robustness_tradeoff_path,
        quality_robustness_frontier_path,
        system_final_auxiliary_attack_summary_path,
        system_final_auxiliary_attack_by_family_path,
        system_final_auxiliary_attack_by_condition_path,
        Path(str(pw04_summary["attack_event_table_path"])),
        Path(str(pw04_summary["attack_family_summary_csv_path"])),
        Path(str(pw04_summary["attack_condition_summary_csv_path"])),
        Path(str(pw04_summary["clean_attack_overview_path"])),
        Path(str(pw04_summary["paper_scope_registry_path"])),
        Path(str(pw04_summary["bootstrap_confidence_intervals_path"])),
        Path(str(pw04_summary["bootstrap_confidence_intervals_csv_path"])),
        *[Path(path_value) for path_value in canonical_metrics_paths.values()],
        *[Path(path_value) for path_value in paper_tables_paths.values()],
        *[Path(path_value) for path_value in paper_figures_paths.values()],
        *[Path(path_value) for path_value in tail_estimation_paths.values()],
    ]
    for path_obj in required_paths:
        assert path_obj.exists(), path_obj
    assert Path(str(pw04_summary["robustness_dir"])).is_dir()
    assert Path(str(pw04_summary["geometry_diagnostics_dir"])).is_dir()
    assert Path(str(pw04_summary["payload_robustness_dir"])).is_dir()
    assert Path(str(pw04_summary["tradeoff_dir"])).is_dir()

    merge_manifest = _load_json_dict(Path(str(pw04_summary["attack_merge_manifest_path"])))
    pool_manifest = _load_json_dict(Path(str(pw04_summary["attack_positive_pool_manifest_path"])))
    attack_negative_pool_manifest = _load_json_dict(attack_negative_pool_manifest_path)
    formal_final_metrics = _load_json_dict(Path(str(pw04_summary["formal_attack_final_decision_metrics_path"])))
    formal_attestation_metrics = _load_json_dict(Path(str(pw04_summary["formal_attack_attestation_metrics_path"])))
    derived_union_metrics = _load_json_dict(Path(str(pw04_summary["derived_attack_union_metrics_path"])))
    formal_attack_negative_metrics = _load_json_dict(formal_attack_negative_metrics_path)
    clean_quality_metrics = _load_json_dict(clean_quality_metrics_path)
    attack_quality_metrics = _load_json_dict(attack_quality_metrics_path)
    quality_pair_plan = _load_json_dict(quality_pair_plan_path)
    quality_finalize_manifest = _load_json_dict(quality_finalize_manifest_path)
    clean_attack_overview = _load_json_dict(Path(str(pw04_summary["clean_attack_overview_path"])))
    paper_metric_registry = _load_json_dict(Path(str(pw04_summary["paper_scope_registry_path"])))
    content_chain_metrics = _load_json_dict(Path(str(canonical_metrics_paths["content_chain"])))
    event_attestation_metrics = _load_json_dict(Path(str(canonical_metrics_paths["event_attestation"])))
    system_final_metrics = _load_json_dict(Path(str(canonical_metrics_paths["system_final"])))
    bootstrap_payload = _load_json_dict(Path(str(pw04_summary["bootstrap_confidence_intervals_path"])))
    bootstrap_csv_rows = _load_csv_rows(Path(str(pw04_summary["bootstrap_confidence_intervals_csv_path"])))
    main_metrics_rows = _load_csv_rows(Path(str(paper_tables_paths["main_metrics_summary_csv_path"])))
    family_paper_rows = _load_csv_rows(Path(str(paper_tables_paths["attack_family_summary_paper_csv_path"])))
    condition_paper_rows = _load_csv_rows(Path(str(paper_tables_paths["attack_condition_summary_paper_csv_path"])))
    rescue_rows = _load_csv_rows(Path(str(paper_tables_paths["rescue_metrics_summary_csv_path"])))
    tail_fpr_1e4 = _load_json_dict(Path(str(tail_estimation_paths["estimated_tail_fpr_1e4_path"])))
    tail_fpr_1e5 = _load_json_dict(Path(str(tail_estimation_paths["estimated_tail_fpr_1e5_path"])))
    tail_fit_diagnostics = _load_json_dict(Path(str(tail_estimation_paths["tail_fit_diagnostics_path"])))
    tail_fit_stability = _load_json_dict(Path(str(tail_estimation_paths["tail_fit_stability_summary_path"])))
    robustness_curve_rows = _load_csv_rows(robustness_curve_by_family_path)
    robustness_macro_rows = _load_csv_rows(robustness_macro_summary_path)
    worst_case_rows = _load_csv_rows(worst_case_attack_summary_path)
    geometry_family_rows = _load_csv_rows(geo_chain_usage_by_family_path)
    geometry_summary_rows = _load_csv_rows(geo_diagnostics_summary_path)
    geometry_conditional_rows = _load_csv_rows(geo_diagnostics_conditional_metrics_path)
    conditional_rescue_metrics = _load_json_dict(conditional_rescue_metrics_path)
    geometry_optional_claim_summary = _load_json_dict(geometry_optional_claim_summary_path)
    payload_attack_summary = _load_json_dict(payload_attack_summary_path)
    wrong_event_attestation_challenge_summary = _load_json_dict(wrong_event_attestation_challenge_summary_path)
    tradeoff_rows = _load_csv_rows(quality_robustness_tradeoff_path)
    system_final_auxiliary_attack_summary = _load_json_dict(system_final_auxiliary_attack_summary_path)
    system_final_auxiliary_family_rows = _load_csv_rows(system_final_auxiliary_attack_by_family_path)
    system_final_auxiliary_condition_rows = _load_csv_rows(system_final_auxiliary_attack_by_condition_path)
    attack_event_rows = read_jsonl(Path(str(pw04_summary["attack_event_table_path"])))

    expected_attack_event_count = int(pw03_fixture["expected_attack_event_count"])
    expected_positive_attack_event_count = int(pw03_fixture["expected_positive_attack_event_count"])
    expected_negative_attack_event_count = int(pw03_fixture["expected_negative_attack_event_count"])
    assert pw04_summary["status"] == "completed"
    assert pw04_summary["pw04_mode"] == pw04_module.PW04_MODE_FINALIZE
    assert pw04_summary["prepare_manifest_path"] == normalize_path_value(prepare_manifest_path)
    assert pw04_summary["paper_exports_completed"] is True
    assert pw04_summary["tail_estimation_enabled"] is False
    assert pw04_summary["quality_shard_count"] == len(quality_shard_paths)
    assert [normalize_path_value(path_obj) for path_obj in prepared_quality_shard_paths] == [
        normalize_path_value(path_obj) for path_obj in quality_shard_paths
    ]
    assert pw04_summary["completed_attack_event_count"] == expected_attack_event_count
    assert merge_manifest["expected_attack_event_count"] == expected_attack_event_count
    assert merge_manifest["completed_attack_event_count"] == expected_attack_event_count
    assert merge_manifest["attack_family_count"] == len({row["attack_family"] for row in cast(List[Dict[str, Any]], pw03_fixture["attack_event_grid"])})
    assert merge_manifest["parent_event_count"] == len({row["parent_event_id"] for row in cast(List[Dict[str, Any]], pw03_fixture["attack_event_grid"])})
    assert pw04_summary["attacked_positive_event_count"] == expected_positive_attack_event_count
    assert pw04_summary["attacked_negative_event_count"] == expected_negative_attack_event_count
    assert pool_manifest["event_count"] == expected_positive_attack_event_count
    assert attack_negative_pool_manifest["event_count"] == expected_negative_attack_event_count

    assert formal_final_metrics["metrics"]["accepted_count"] == pw03_fixture["expected_formal_final_positive_count"]
    assert formal_final_metrics["metrics"]["attack_tpr"] == pytest.approx(
        pw03_fixture["expected_formal_final_positive_count"] / expected_positive_attack_event_count
    )
    assert formal_attestation_metrics["metrics"]["accepted_count"] == pw03_fixture["expected_formal_attestation_positive_count"]
    assert formal_attestation_metrics["metrics"]["attack_tpr"] == pytest.approx(
        pw03_fixture["expected_formal_attestation_positive_count"] / expected_positive_attack_event_count
    )
    assert derived_union_metrics["metrics"]["accepted_count"] == pw03_fixture["expected_derived_union_positive_count"]
    assert derived_union_metrics["metrics"]["attack_tpr"] == pytest.approx(
        pw03_fixture["expected_derived_union_positive_count"] / expected_positive_attack_event_count
    )
    assert formal_attack_negative_metrics["metrics"]["attack_negative_count"] == expected_negative_attack_event_count
    assert formal_attack_negative_metrics["metrics"]["formal_final_false_accept_count"] == pw03_fixture["expected_formal_final_negative_false_accept_count"]
    assert formal_attack_negative_metrics["metrics"]["formal_attestation_false_accept_count"] == pw03_fixture["expected_formal_attestation_negative_false_accept_count"]
    assert formal_attack_negative_metrics["metrics"]["derived_attack_union_false_accept_count"] == pw03_fixture["expected_derived_union_negative_false_accept_count"]
    assert clean_quality_metrics["overall"]["count"] == 2
    assert clean_quality_metrics["overall"]["clip_status"] == "ok"
    assert clean_quality_metrics["overall"]["prompt_text_coverage_status"] == "ok"
    assert attack_quality_metrics["overall"]["count"] == expected_positive_attack_event_count
    assert attack_quality_metrics["overall"]["mean_psnr"] is not None
    assert attack_quality_metrics["overall"]["mean_ssim"] is not None
    assert attack_quality_metrics["overall"]["mean_clip_text_similarity"] is not None
    assert attack_quality_metrics["overall"]["clip_model_name"] == pw_quality_metrics_module.CLIP_MODEL_NAME
    assert attack_quality_metrics["overall"]["clip_sample_count"] == expected_positive_attack_event_count
    assert attack_quality_metrics["overall"]["clip_status"] == "ok"
    assert attack_quality_metrics["overall"]["prompt_text_coverage_status"] == "ok"
    if attack_quality_metrics["overall"]["lpips_status"] == "ok":
        assert attack_quality_metrics["overall"]["quality_readiness_status"] == "ready"
        assert attack_quality_metrics["overall"]["quality_readiness_blocking"] is False
    else:
        assert attack_quality_metrics["overall"]["quality_readiness_status"] == "partial"
        assert attack_quality_metrics["overall"]["quality_readiness_blocking"] is True
        assert attack_quality_metrics["overall"]["quality_readiness_reason"]
    assert quality_pair_plan["clean_quality_pair_manifest_path"] == normalize_path_value(
        Path(str(cast(Dict[str, Any], fixture["pw02"])["clean_quality_pair_manifest_path"]))
    )
    assert quality_pair_plan["clean_expected_pair_count"] == clean_quality_metrics["overall"]["expected_count"]
    assert quality_pair_plan["attack_expected_pair_count"] == attack_quality_metrics["overall"]["expected_count"]
    assert quality_finalize_manifest["clean_quality_metrics_path"] == normalize_path_value(clean_quality_metrics_path)
    assert quality_finalize_manifest["attack_quality_metrics_path"] == normalize_path_value(attack_quality_metrics_path)
    assert quality_finalize_manifest["quality_pair_plan_path"] == normalize_path_value(quality_pair_plan_path)
    assert clean_attack_overview["attack_quality_mean_psnr"] == attack_quality_metrics["overall"]["mean_psnr"]
    assert clean_attack_overview["attack_quality_mean_ssim"] == attack_quality_metrics["overall"]["mean_ssim"]
    assert clean_attack_overview["attack_quality_mean_clip_text_similarity"] == attack_quality_metrics["overall"]["mean_clip_text_similarity"]
    assert clean_attack_overview["attack_quality_clip_model_name"] == pw_quality_metrics_module.CLIP_MODEL_NAME
    assert clean_attack_overview["attack_quality_clip_status"] == "ok"
    assert clean_attack_overview["attack_negative_formal_fpr"] == formal_attack_negative_metrics["metrics"]["formal_final_attack_fpr"]
    assert clean_attack_overview["attack_negative_formal_attestation_fpr"] == formal_attack_negative_metrics["metrics"]["formal_attestation_attack_fpr"]
    assert clean_attack_overview["attack_negative_derived_union_fpr"] == formal_attack_negative_metrics["metrics"]["derived_attack_union_attack_fpr"]

    assert paper_metric_registry["canonical_scopes"] == ["content_chain", "event_attestation", "system_final"]
    assert paper_metric_registry["legacy_scope_mapping"]["content_chain"]["attack"]["legacy_scope_name"] == "formal_attack_final_decision"
    assert paper_metric_registry["legacy_scope_mapping"]["event_attestation"]["clean"]["legacy_scope_name"] == "clean_attestation_evaluate_export"
    assert paper_metric_registry["legacy_scope_mapping"]["system_final"]["attack"]["legacy_scope_name"] == "derived_attack_union"
    assert paper_metric_registry["artifact_paths"]["supplemental_metrics"]["clean_quality_metrics_path"] == normalize_path_value(clean_quality_metrics_path)
    assert paper_metric_registry["artifact_paths"]["supplemental_metrics"]["attack_quality_metrics_path"] == normalize_path_value(attack_quality_metrics_path)
    assert paper_metric_registry["artifact_paths"]["supplemental_metrics"]["robustness_curve_by_family_path"] == normalize_path_value(robustness_curve_by_family_path)
    assert paper_metric_registry["artifact_paths"]["supplemental_metrics"]["quality_robustness_tradeoff_path"] == normalize_path_value(quality_robustness_tradeoff_path)
    assert paper_metric_registry["artifact_paths"]["supplemental_metrics"]["system_final_auxiliary_attack_summary_path"] == normalize_path_value(system_final_auxiliary_attack_summary_path)

    assert content_chain_metrics["scope"] == "content_chain"
    assert content_chain_metrics["clean_metrics"]["clean_positive_count"] == 4
    assert content_chain_metrics["clean_metrics"]["clean_negative_count"] == 4
    assert event_attestation_metrics["scope"] == "event_attestation"
    assert event_attestation_metrics["clean_metrics"]["clean_tpr"] == pytest.approx(0.75)
    assert event_attestation_metrics["clean_metrics"]["clean_fpr"] == pytest.approx(0.25)
    assert event_attestation_metrics["clean_metrics"]["accepted_count_clean_positive"] == 3
    assert event_attestation_metrics["clean_metrics"]["accepted_count_clean_negative"] == 1
    assert system_final_metrics["scope"] == "system_final"
    assert system_final_metrics["compatibility"]["attack_legacy_scope_name"] == "derived_attack_union"

    assert [row["scope"] for row in main_metrics_rows] == ["content_chain", "event_attestation", "system_final"]
    event_attestation_row = next(row for row in main_metrics_rows if row["scope"] == "event_attestation")
    assert event_attestation_row["clean_tpr"] == "0.75"
    assert event_attestation_row["clean_fpr"] == "0.25"
    assert event_attestation_row["accepted_count_clean_positive"] == "3"
    assert event_attestation_row["accepted_count_clean_negative"] == "1"
    assert event_attestation_row["bootstrap_ci_clean_tpr_lower"] != ""
    assert event_attestation_row["metric_source_clean"].endswith("/exports/pw02/evaluate/clean/attestation/evaluate_record.json")

    assert family_paper_rows
    assert "content_chain_attack_tpr" in family_paper_rows[0]
    assert "formal_final_decision_attack_tpr" not in family_paper_rows[0]
    assert "attack_mean_psnr" in family_paper_rows[0]
    assert family_paper_rows[0]["attack_mean_psnr"] != ""
    assert "attack_mean_lpips" in family_paper_rows[0]
    assert "attack_mean_clip_text_similarity" in family_paper_rows[0]
    assert condition_paper_rows
    assert "system_final_attack_tpr" in condition_paper_rows[0]
    assert "attack_family" in condition_paper_rows[0]
    assert "attack_mean_ssim" in condition_paper_rows[0]
    assert condition_paper_rows[0]["attack_mean_ssim"] != ""
    assert "attack_mean_lpips" in condition_paper_rows[0]
    assert "attack_mean_clip_text_similarity" in condition_paper_rows[0]

    assert len(rescue_rows) == 1
    rescue_row = rescue_rows[0]
    assert rescue_row["clean_false_accept_count"] == "0"
    assert rescue_row["attack_true_accept_count"] == str(pw03_fixture["expected_derived_union_positive_count"])
    attack_true_accept_count_by_family = json.loads(rescue_row["attack_true_accept_count_by_family"])
    assert sum(int(value) for value in attack_true_accept_count_by_family.values()) == pw03_fixture["expected_derived_union_positive_count"]
    geo_not_used_reason_counts = json.loads(rescue_row["geo_not_used_reason_counts"])
    assert geo_not_used_reason_counts

    assert set(bootstrap_payload["scopes"].keys()) == {"content_chain", "event_attestation", "system_final"}
    assert bootstrap_payload["scopes"]["content_chain"]["clean_tpr"]["status"] == "ok"
    assert bootstrap_payload["scopes"]["event_attestation"]["clean_fpr"]["status"] == "ok"
    assert bootstrap_payload["scopes"]["system_final"]["attack_tpr"]["lower_bound"] is not None
    assert len(bootstrap_csv_rows) == 9

    assert tail_fpr_1e4["tail_estimation_enabled"] is False
    assert tail_fpr_1e4["readiness"]["status"] == "disabled"
    assert tail_fpr_1e4["scope_estimates"]["content_chain"]["status"] == "disabled"
    assert tail_fpr_1e5["scope_estimates"]["event_attestation"]["status"] == "disabled"
    assert tail_fit_diagnostics["scope_diagnostics"]["system_final"]["status"] == "not_applicable"
    assert tail_fit_diagnostics["readiness"]["status"] == "disabled"
    assert tail_fit_stability["scopes"]["system_final"]["reason"] == "system_final_is_decision_union_without_scalar_score"

    attack_family_count = len({row["attack_family"] for row in cast(List[Dict[str, Any]], pw03_fixture["attack_event_grid"])})
    assert len(robustness_curve_rows) == attack_family_count * 3
    assert {row["scope"] for row in robustness_curve_rows} == {"content_chain", "event_attestation", "system_final"}
    assert all(row["severity_level_status"] in {"ok", "partial", "not_available"} for row in robustness_curve_rows)
    assert any(row["severity_level_status"] in {"ok", "partial"} for row in robustness_curve_rows)
    assert len(robustness_macro_rows) == 3
    assert {row["scope"] for row in robustness_macro_rows} == {"content_chain", "event_attestation", "system_final"}
    assert all(row["severity_level_status"] in {"ok", "partial"} for row in robustness_macro_rows)
    assert len(worst_case_rows) == 3
    assert {row["scope"] for row in worst_case_rows} == {"content_chain", "event_attestation", "system_final"}
    assert all("severity_label" in row for row in worst_case_rows)

    assert len(geometry_family_rows) == attack_family_count
    assert all(row["sync_success_status"] == "ok" for row in geometry_family_rows)
    assert all(row["inverse_transform_success_status"] == "ok" for row in geometry_family_rows)
    assert all(row["attention_anchor_available_status"] == "ok" for row in geometry_family_rows)
    assert len(geometry_summary_rows) == 1
    assert geometry_summary_rows[0]["event_count"] == str(expected_positive_attack_event_count)
    assert geometry_summary_rows[0]["future_upstream_sidecar_required"] == "False"
    assert len(geometry_conditional_rows) == 15
    assert {row["geometry_condition_name"] for row in geometry_conditional_rows} == {
        "sync_success",
        "inverse_transform_success",
        "attention_anchor_available",
        "geo_rescue_eligible",
        "geo_rescue_applied",
    }
    assert {row["geometry_condition_value"] for row in geometry_conditional_rows} == {"true", "false", "missing"}
    for condition_name in {
        "sync_success",
        "inverse_transform_success",
        "attention_anchor_available",
        "geo_rescue_eligible",
        "geo_rescue_applied",
    }:
        condition_rows = [row for row in geometry_conditional_rows if row["geometry_condition_name"] == condition_name]
        assert sum(int(row["event_count"]) for row in condition_rows) == expected_positive_attack_event_count
    assert conditional_rescue_metrics["readiness"]["status"] == "ready"
    assert conditional_rescue_metrics["readiness"]["blocking"] is False
    assert geometry_optional_claim_summary["readiness"]["status"] == "ready"
    assert geometry_optional_claim_summary["readiness"]["blocking"] is False
    assert geometry_optional_claim_summary["overall"]["event_count"] == expected_positive_attack_event_count
    assert geometry_optional_claim_summary["overall"]["eligible_event_count"] == expected_positive_attack_event_count
    assert geometry_optional_claim_summary["overall"]["evidence_event_count"] == expected_positive_attack_event_count
    assert geometry_optional_claim_summary["overall"]["supporting_evidence_event_count"] > 0

    assert payload_attack_summary["status"] == "ok"
    assert payload_attack_summary["reason"] is None
    assert payload_attack_summary["future_upstream_sidecar_required"] is False
    assert payload_attack_summary["readiness"]["status"] == "ready"
    assert payload_attack_summary["readiness"]["blocking"] is False
    assert payload_attack_summary["probe_overall"]["status"] == "ready"
    assert payload_attack_summary["probe_overall"]["available_probe_event_count"] == expected_positive_attack_event_count
    assert payload_attack_summary["probe_overall"]["mean_consistency_score"] == pytest.approx(
        pw03_fixture["expected_payload_mean_codeword_agreement"]
    )
    assert payload_attack_summary["overall"]["event_count"] == expected_positive_attack_event_count
    assert payload_attack_summary["overall"]["available_payload_event_count"] == expected_positive_attack_event_count
    assert payload_attack_summary["overall"]["missing_payload_event_count"] == 0
    assert payload_attack_summary["overall"]["mean_codeword_agreement"] == pytest.approx(
        pw03_fixture["expected_payload_mean_codeword_agreement"]
    )
    assert payload_attack_summary["overall"]["min_codeword_agreement"] == pytest.approx(
        pw03_fixture["expected_payload_min_codeword_agreement"]
    )
    assert payload_attack_summary["overall"]["max_codeword_agreement"] == pytest.approx(
        pw03_fixture["expected_payload_max_codeword_agreement"]
    )
    assert payload_attack_summary["overall"]["mean_n_bits_compared"] == pytest.approx(96.0)
    assert payload_attack_summary["overall"]["mean_bit_accuracy"] == pytest.approx(
        pw03_fixture["expected_payload_mean_codeword_agreement"]
    )
    assert payload_attack_summary["overall"]["weighted_bit_accuracy"] == pytest.approx(
        pw03_fixture["expected_payload_mean_codeword_agreement"]
    )
    assert payload_attack_summary["overall"]["mean_bit_error_rate"] == pytest.approx(
        1.0 - pw03_fixture["expected_payload_mean_codeword_agreement"]
    )
    assert payload_attack_summary["overall"]["weighted_bit_error_rate"] == pytest.approx(
        1.0 - pw03_fixture["expected_payload_mean_codeword_agreement"]
    )
    assert payload_attack_summary["overall"]["message_success_count"] == 0
    assert payload_attack_summary["overall"]["message_success_rate"] == pytest.approx(0.0)
    assert payload_attack_summary["overall"]["payload_primary_metric_sources"] == ["codeword_agreement_and_n_bits_compared"]
    assert payload_attack_summary["overall"]["attested_event_count"] == pw03_fixture["expected_formal_attestation_positive_count"]
    assert payload_attack_summary["overall"]["mean_event_attestation_score"] == pytest.approx(
        pw03_fixture["expected_payload_mean_attestation_score"]
    )
    assert payload_attack_summary["overall"]["lf_detect_variants"] == ["correlation_v2"]
    assert payload_attack_summary["overall"]["message_sources"] == ["attestation_event_digest"]
    assert len(payload_attack_summary["by_attack_family"]) == attack_family_count
    assert len(payload_attack_summary["probe_by_attack_family"]) == attack_family_count
    assert wrong_event_attestation_challenge_summary["status"] == "ok"
    assert wrong_event_attestation_challenge_summary["reason"] is None
    assert wrong_event_attestation_challenge_summary["future_upstream_sidecar_required"] is False
    assert wrong_event_attestation_challenge_summary["readiness"]["status"] == "ready"
    assert wrong_event_attestation_challenge_summary["readiness"]["blocking"] is False
    assert wrong_event_attestation_challenge_summary["overall"]["event_count"] == expected_positive_attack_event_count
    assert wrong_event_attestation_challenge_summary["overall"]["attempted_event_count"] == pw03_fixture["expected_wrong_event_challenge_attempt_count"]
    assert wrong_event_attestation_challenge_summary["overall"]["bundle_verified_count"] == pw03_fixture["expected_wrong_event_challenge_attempt_count"]
    assert wrong_event_attestation_challenge_summary["overall"]["wrong_event_rejected_count"] == pw03_fixture["expected_wrong_event_rejected_count"]
    assert wrong_event_attestation_challenge_summary["overall"]["wrong_event_false_accept_count"] == 0
    assert wrong_event_attestation_challenge_summary["overall"]["wrong_event_rejection_rate"] == pytest.approx(1.0)
    assert len(wrong_event_attestation_challenge_summary["by_attack_family"]) == attack_family_count
    assert all(row["status"] == "ok" for row in wrong_event_attestation_challenge_summary["rows"])
    assert cast(Dict[str, Any], pw04_summary["analysis_only_artifact_paths"])["pw04_payload_attack_summary"] == normalize_path_value(
        payload_attack_summary_path
    )
    assert cast(Dict[str, Any], pw04_summary["analysis_only_artifact_paths"])["pw04_geometry_optional_claim_summary"] == normalize_path_value(
        geometry_optional_claim_summary_path
    )
    assert cast(Dict[str, Any], pw04_summary["analysis_only_artifact_paths"])["pw04_wrong_event_attestation_challenge_summary"] == normalize_path_value(
        wrong_event_attestation_challenge_summary_path
    )
    assert cast(Dict[str, Any], pw04_summary["analysis_only_artifact_paths"])["pw04_quality_pair_plan"] == normalize_path_value(
        quality_pair_plan_path
    )
    assert cast(Dict[str, Any], pw04_summary["analysis_only_artifact_paths"])["pw04_quality_finalize_manifest"] == normalize_path_value(
        quality_finalize_manifest_path
    )
    assert cast(Dict[str, Any], pw04_summary["analysis_only_artifact_annotations"])["pw04_wrong_event_attestation_challenge_summary"] == {
        "canonical": False,
        "analysis_only": True,
    }
    assert cast(Dict[str, Any], pw04_summary["analysis_only_artifact_annotations"])["pw04_payload_attack_summary"] == {
        "canonical": False,
        "analysis_only": True,
    }
    assert cast(Dict[str, Any], pw04_summary["analysis_only_artifact_annotations"])["pw04_geometry_optional_claim_summary"] == {
        "canonical": False,
        "analysis_only": True,
    }

    assert system_final_auxiliary_attack_summary["scope"] == "system_final_auxiliary"
    assert system_final_auxiliary_attack_summary["canonical"] is False
    assert system_final_auxiliary_attack_summary["analysis_only"] is True
    assert system_final_auxiliary_attack_summary["overall"]["consistency_status"] == "exact_match"
    assert system_final_auxiliary_attack_summary["overall"]["consistency_mismatch_count"] == 0
    assert system_final_auxiliary_attack_summary["overall"]["system_final_auxiliary_attack_tpr"] == pytest.approx(
        pw03_fixture["expected_derived_union_positive_count"] / expected_positive_attack_event_count
    )
    assert system_final_auxiliary_family_rows
    assert system_final_auxiliary_condition_rows
    assert cast(Dict[str, Any], pw04_summary["analysis_only_artifact_paths"])["pw04_system_final_auxiliary_attack_summary"] == normalize_path_value(system_final_auxiliary_attack_summary_path)
    assert cast(Dict[str, Any], pw04_summary["analysis_only_artifact_annotations"])["pw04_system_final_auxiliary_attack_summary"] == {
        "canonical": False,
        "analysis_only": True,
    }
    pw04_analysis_only_annotations = cast(Dict[str, Any], pw04_summary["analysis_only_artifact_annotations"])
    if "pw02_system_final_auxiliary_operating_semantics" in pw04_analysis_only_annotations:
        assert pw04_analysis_only_annotations["pw02_system_final_auxiliary_operating_semantics"] == {
            "canonical": False,
            "analysis_only": True,
        }

    assert len(tradeoff_rows) == 3
    assert {row["scope"] for row in tradeoff_rows} == {"content_chain", "event_attestation", "system_final"}
    assert all(row["clean_quality_scope"] == "content_chain" for row in tradeoff_rows)
    assert all(row["clean_quality_status"] == str(clean_quality_metrics["overall"]["status"]) for row in tradeoff_rows)
    assert all(row["lpips_status"] == str(clean_quality_metrics["overall"]["lpips_status"]) for row in tradeoff_rows)
    assert all(row["clip_status"] == str(clean_quality_metrics["overall"]["clip_status"]) for row in tradeoff_rows)
    assert all("attack_mean_lpips" in row for row in tradeoff_rows)
    assert all("attack_lpips_status" in row for row in tradeoff_rows)
    assert all("attack_mean_clip_text_similarity" in row for row in tradeoff_rows)
    assert all(row["attack_clip_status"] == str(attack_quality_metrics["overall"]["clip_status"]) for row in tradeoff_rows)
    assert all(row["attack_clip_model_name"] == pw_quality_metrics_module.CLIP_MODEL_NAME for row in tradeoff_rows)
    assert all(Path(str(row["clean_quality_metrics_path"])).exists() for row in tradeoff_rows)
    assert all(Path(str(row["attack_quality_metrics_path"])).exists() for row in tradeoff_rows)
    assert all(Path(str(row["robustness_macro_summary_path"])).exists() for row in tradeoff_rows)
    assert quality_robustness_frontier_path.stat().st_size > 0

    attack_event_lookup = cast(Dict[str, Dict[str, Any]], pw03_fixture["attack_event_lookup"])
    for row in attack_event_rows:
        expected_attack_event = attack_event_lookup[row["attack_event_id"]]
        assert row["attack_family"] == expected_attack_event["attack_family"]
        assert row["parent_event_id"] == expected_attack_event["parent_event_id"]
        assert row["sample_role"] == expected_attack_event["sample_role"]
        assert "geo_rescue_eligible" in row
        assert "geo_rescue_applied" in row
        assert "geo_not_used_reason" in row
        assert row["severity_status"] in {"ok", "not_available"}
        assert row["sync_status"] in {"ok", "degraded"}
        assert row["sync_success"] in {True, False}
        assert row["sync_success_status"] == "ok"
        assert row["inverse_transform_success"] in {True, False}
        assert row["inverse_transform_success_status"] == "ok"
        assert row["attention_anchor_available"] in {True, False}
        assert row["attention_anchor_available_status"] == "ok"
        if row["sample_role"] == pw03_module.ATTACKED_POSITIVE_SAMPLE_ROLE:
            assert row["attack_quality_status"] == "ok"
            assert row["attack_quality_psnr"] is not None
            assert row["attack_quality_ssim"] is not None
            assert "attack_quality_lpips" in row
            assert row["attack_quality_clip_text_similarity"] is not None
        else:
            assert row["sample_role"] == pw03_module.ATTACKED_NEGATIVE_SAMPLE_ROLE
            assert row["attack_quality_status"] == "not_applicable"
            assert row["attack_quality_psnr"] is None
            assert row["attack_quality_ssim"] is None
            assert row["attack_quality_lpips"] is None
            assert row["attack_quality_clip_text_similarity"] is None

    assert any(row["geo_rescue_applied"] is True for row in attack_event_rows)
    assert any(isinstance(row["geo_not_used_reason"], str) and row["geo_not_used_reason"] for row in attack_event_rows)

    first_pool_event = cast(List[Dict[str, Any]], pool_manifest["events"])[0]
    assert first_pool_event["formal_record_path"]
    assert Path(str(first_pool_event["formal_record_path"])).exists()
    assert summary["family_id"] == pw04_summary["family_id"]


def test_pw04_quality_shard_worker_only_writes_shard_payload(tmp_path: Path, monkeypatch: Any) -> None:
    """
    Verify the PW04 quality shard worker writes only shard-local payloads.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr(
        pw_quality_metrics_module,
        "_compute_clip_text_similarity",
        _fake_clip_text_similarity,
    )
    family_root = tmp_path / "paper_workflow" / "families" / "family_pw04_quality_shard"
    quality_root = family_root / "exports" / "pw04" / "quality"
    ensure_directory(quality_root)

    clean_reference_path = quality_root / "clean_reference.png"
    clean_candidate_path = quality_root / "clean_candidate.png"
    attack_reference_path = quality_root / "attack_reference.png"
    attack_candidate_path = quality_root / "attack_candidate.png"
    Image.new("RGB", (8, 8), color=(120, 80, 60)).save(clean_reference_path)
    Image.new("RGB", (8, 8), color=(121, 80, 60)).save(clean_candidate_path)
    Image.new("RGB", (8, 8), color=(90, 100, 110)).save(attack_reference_path)
    Image.new("RGB", (8, 8), color=(93, 100, 110)).save(attack_candidate_path)

    quality_pair_plan_path = quality_root / "quality_pair_plan.json"
    write_json_atomic(
        quality_pair_plan_path,
        {
            "artifact_type": "paper_workflow_pw04_quality_pair_plan",
            "schema_version": "pw_stage_04_v1",
            "family_id": "family_pw04_quality_shard",
            "clean_pairs": [
                {
                    "event_id": "source_event_000001",
                    "reference_image_path": normalize_path_value(clean_reference_path),
                    "candidate_image_path": normalize_path_value(clean_candidate_path),
                    "prompt_text": "prompt one",
                }
            ],
            "attack_pairs": [
                {
                    "attack_event_id": "attack_event_000001",
                    "parent_event_id": "source_event_000001",
                    "attack_family": "resize",
                    "attack_condition_key": "resize::0.8",
                    "attack_config_name": "resize_cfg",
                    "reference_image_path": normalize_path_value(attack_reference_path),
                    "candidate_image_path": normalize_path_value(attack_candidate_path),
                    "prompt_text": "prompt two",
                }
            ],
            "shards": [
                {
                    "quality_shard_index": 0,
                    "clean_pair_ids": ["source_event_000001"],
                    "attack_pair_ids": ["attack_event_000001"],
                    "clean_pair_count": 1,
                    "attack_pair_count": 1,
                    "total_pair_count": 2,
                }
            ],
        },
    )

    shard_export = pw04_quality_shard_module.run_pw04_quality_shard(
        family_id="family_pw04_quality_shard",
        quality_pair_plan_path=quality_pair_plan_path,
        quality_shard_index=0,
    )

    shard_path = Path(str(shard_export["path"]))
    shard_payload = _load_json_dict(shard_path)
    assert shard_path.exists()
    assert shard_payload["artifact_type"] == "paper_workflow_pw04_quality_shard"
    assert shard_payload["clean_pair_count"] == 1
    assert shard_payload["attack_pair_count"] == 1
    assert shard_payload["clean_quality_summary"]["count"] == 1
    assert shard_payload["attack_quality_summary"]["count"] == 1
    assert not (family_root / "exports" / "pw04" / "metrics" / "clean_quality_metrics.json").exists()
    assert not (family_root / "exports" / "pw04" / "metrics" / "attack_quality_metrics.json").exists()
    assert not (family_root / "exports" / "pw04" / "tables").exists()
    assert not (family_root / "exports" / "pw04" / "figures").exists()


def test_pw04_quality_shard_mode_requires_prepare_manifest(tmp_path: Path) -> None:
    """
    Verify PW04 worker mode refuses to run before prepare manifest exists.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    with pytest.raises(FileNotFoundError, match="PW04 prepare manifest"):
        pw04_module.run_pw04_merge_attack_event_shards(
            drive_project_root=tmp_path / "drive",
            family_id="family_pw04_missing_prepare",
            pw04_mode=pw04_module.PW04_MODE_QUALITY_SHARD,
            quality_shard_index=0,
        )


def test_pw04_finalize_requires_all_planned_quality_shards(tmp_path: Path, monkeypatch: Any) -> None:
    """
    Verify PW04 finalize fails until all planned quality shards are present.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr(
        pw_quality_metrics_module,
        "_compute_clip_text_similarity",
        _fake_clip_text_similarity,
    )
    _build_pw04_fixture(tmp_path, "family_pw04_missing_quality_shards")
    prepare_summary = pw04_module.run_pw04_merge_attack_event_shards(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw04_missing_quality_shards",
        pw04_mode=pw04_module.PW04_MODE_PREPARE,
    )

    assert prepare_summary["quality_shard_count"] >= 1
    with pytest.raises(RuntimeError, match="requires all prepared quality shard outputs"):
        pw04_module.run_pw04_merge_attack_event_shards(
            drive_project_root=tmp_path / "drive",
            family_id="family_pw04_missing_quality_shards",
            pw04_mode=pw04_module.PW04_MODE_FINALIZE,
        )


def test_pw04_payload_attack_summary_prefers_decode_sidecar_metrics(tmp_path: Path) -> None:
    """
    Verify PW04 payload attack summary prefers decode sidecar metrics over legacy LF trace values.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    payload_decode_sidecar_path = tmp_path / "attack_payload_decode_sidecar.json"
    write_json_atomic(
        payload_decode_sidecar_path,
        {
            "artifact_type": "paper_workflow_payload_decode_sidecar",
            "schema_version": "pw_payload_sidecar_v1",
            "event_id": "attack_event_000001",
            "sample_role": "attacked_positive",
            "reference_event_id": "source_event_000001",
            "message_source": "sidecar_message_source",
            "lf_detect_variant": "sidecar_attack_variant",
            "n_bits_compared": 96,
            "bit_error_count": 48,
            "codeword_agreement": 0.5,
            "message_decode_success": False,
        },
    )

    payload_attack_summary = pw04_metrics_extensions_module._build_payload_attack_summary_payload(
        family_id="family_payload_sidecar_pw04",
        attack_event_rows=[
            {
                "attack_event_id": "attack_event_000001",
                "attack_family": "resize",
                "attack_condition_key": "resize::0.8",
                "payload_decode_sidecar_path": normalize_path_value(payload_decode_sidecar_path),
                "formal_record": {
                    "content_evidence_payload": {
                        "score_parts": {
                            "lf_trajectory_detect_trace": {
                                "codeword_agreement": 0.88,
                                "n_bits_compared": 64,
                                "detect_variant": "trace_variant",
                                "message_source": "trace_source",
                            }
                        }
                    },
                    "attestation": {
                        "final_event_attested_decision": {
                            "event_attestation_score": 0.7,
                            "is_event_attested": True,
                        }
                    },
                },
            }
        ],
    )

    assert payload_attack_summary["status"] == "ok"
    assert payload_attack_summary["probe_overall"]["status"] == "ready"
    assert payload_attack_summary["probe_overall"]["available_probe_event_count"] == 1
    assert payload_attack_summary["probe_overall"]["mean_consistency_score"] == pytest.approx(0.5)
    assert payload_attack_summary["overall"]["mean_codeword_agreement"] == pytest.approx(0.5)
    assert payload_attack_summary["overall"]["mean_n_bits_compared"] == pytest.approx(96.0)
    assert payload_attack_summary["overall"]["payload_primary_metric_sources"] == ["codeword_agreement_and_n_bits_compared"]
    assert payload_attack_summary["overall"]["lf_detect_variants"] == ["sidecar_attack_variant"]
    assert payload_attack_summary["overall"]["message_sources"] == ["sidecar_message_source"]
    assert payload_attack_summary["by_attack_family"][0]["attack_family"] == "resize"
    assert payload_attack_summary["by_attack_family"][0]["mean_codeword_agreement"] == pytest.approx(0.5)
    assert payload_attack_summary["probe_by_attack_family"][0]["attack_family"] == "resize"
    assert payload_attack_summary["probe_by_attack_family"][0]["mean_consistency_score"] == pytest.approx(0.5)


def test_pw03_geometry_sidecar_stabilizes_missing_fields() -> None:
    """
    Verify PW03 geometry sidecar emits stable status and reason fields when values are absent.

    Args:
        None.

    Returns:
        None.
    """
    diagnostics = pw03_module._extract_attack_geometry_diagnostics(
        {
            "geometry_result": {
                "sync_result": {},
            },
            "geometry_evidence_payload": {},
        }
    )

    assert diagnostics["sync_success"] is None
    assert diagnostics["sync_success_status"] == "not_available"
    assert diagnostics["sync_quality_metrics"] is None
    assert diagnostics["sync_quality_metrics_status"] == "not_available"
    assert diagnostics["inverse_transform_success"] is None
    assert diagnostics["inverse_transform_success_status"] == "not_available"
    assert diagnostics["attention_anchor_available"] is None
    assert diagnostics["attention_anchor_available_status"] == "not_available"
    assert diagnostics["geometry_failure_reason"] is None
    assert diagnostics["geometry_failure_reason_status"] == "not_available"


def test_pw04_fails_fast_when_one_planned_shard_manifest_is_missing(tmp_path: Path) -> None:
    """
    Verify PW04 fails fast when one planned PW03 shard manifest is missing.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw04_fixture(tmp_path, "family_pw04_missing_shard")
    pw03_fixture = cast(Dict[str, Any], fixture["pw03"])
    attack_shard_plan = cast(Dict[str, Any], pw03_fixture["attack_shard_plan"])
    missing_shard_index = int(cast(List[Dict[str, Any]], attack_shard_plan["shards"])[0]["attack_shard_index"])
    missing_path = Path(str(cast(Dict[str, Any], fixture["summary"])["family_root"])) / "attack_shards" / f"shard_{missing_shard_index:04d}" / "shard_manifest.json"
    missing_path.unlink()

    with pytest.raises(FileNotFoundError):
        pw04_module.run_pw04_merge_attack_event_shards(
            drive_project_root=tmp_path / "drive",
            family_id="family_pw04_missing_shard",
            pw04_mode=pw04_module.PW04_MODE_PREPARE,
        )


def test_pw04_fails_fast_when_planned_shard_is_not_completed(tmp_path: Path) -> None:
    """
    Verify PW04 fails fast when one planned PW03 shard manifest is not completed.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw04_fixture(tmp_path, "family_pw04_non_completed_shard")
    shard_manifest_path = Path(str(cast(Dict[str, Any], fixture["summary"])["family_root"])) / "attack_shards" / "shard_0000" / "shard_manifest.json"
    shard_manifest = _load_json_dict(shard_manifest_path)
    shard_manifest["status"] = "failed"
    write_json_atomic(shard_manifest_path, shard_manifest)

    with pytest.raises(RuntimeError, match="completed"):
        pw04_module.run_pw04_merge_attack_event_shards(
            drive_project_root=tmp_path / "drive",
            family_id="family_pw04_non_completed_shard",
            pw04_mode=pw04_module.PW04_MODE_PREPARE,
        )


def test_pw04_fails_fast_on_duplicate_attack_event_id(tmp_path: Path) -> None:
    """
    Verify PW04 fails fast when two event manifests expose the same attack_event_id.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw04_fixture(tmp_path, "family_pw04_duplicate_attack_event_id")
    pw03_fixture = cast(Dict[str, Any], fixture["pw03"])
    event_manifest_paths = cast(List[Path], pw03_fixture["event_manifest_paths"])
    assert len(event_manifest_paths) >= 2

    first_event_manifest = _load_json_dict(event_manifest_paths[0])
    second_event_manifest = _load_json_dict(event_manifest_paths[1])
    second_event_manifest["attack_event_id"] = first_event_manifest["attack_event_id"]
    write_json_atomic(event_manifest_paths[1], second_event_manifest)

    with pytest.raises(ValueError, match="duplicate attack_event_id"):
        pw04_module.run_pw04_merge_attack_event_shards(
            drive_project_root=tmp_path / "drive",
            family_id="family_pw04_duplicate_attack_event_id",
            pw04_mode=pw04_module.PW04_MODE_PREPARE,
        )


def test_pw04_fails_fast_on_threshold_binding_inconsistency(tmp_path: Path) -> None:
    """
    Verify PW04 fails fast when one event carries a mismatched finalize digest binding.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw04_fixture(tmp_path, "family_pw04_threshold_binding")
    pw03_fixture = cast(Dict[str, Any], fixture["pw03"])
    event_manifest_path = cast(List[Path], pw03_fixture["event_manifest_paths"])[0]
    event_manifest = _load_json_dict(event_manifest_path)
    event_manifest["source_finalize_manifest_digest"] = "broken_finalize_digest"
    write_json_atomic(event_manifest_path, event_manifest)

    with pytest.raises(ValueError, match="source_finalize_manifest_digest mismatch"):
        pw04_module.run_pw04_merge_attack_event_shards(
            drive_project_root=tmp_path / "drive",
            family_id="family_pw04_threshold_binding",
            pw04_mode=pw04_module.PW04_MODE_PREPARE,
        )


def test_pw04_uses_event_manifest_parent_event_id_when_detect_parent_is_absent(tmp_path: Path) -> None:
    """
    Verify PW04 does not depend on detect-record top-level parent_event_id.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw04_fixture(tmp_path, "family_pw04_parent_authority")
    pw03_fixture = cast(Dict[str, Any], fixture["pw03"])
    detect_record_path = cast(List[Path], pw03_fixture["detect_record_paths"])[0]
    detect_payload = _load_json_dict(detect_record_path)
    detect_payload["parent_event_id"] = None
    detect_payload["paper_workflow_parent_event_id"] = None
    write_json_atomic(detect_record_path, detect_payload)

    pw04_summary = pw04_module.run_pw04_merge_attack_event_shards(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw04_parent_authority",
        pw04_mode=pw04_module.PW04_MODE_PREPARE,
    )
    pool_manifest = _load_json_dict(Path(str(pw04_summary["attack_positive_pool_manifest_path"])))
    first_event = cast(List[Dict[str, Any]], pool_manifest["events"])[0]
    expected_attack_event = cast(Dict[str, Any], pw03_fixture["attack_event_lookup"])[first_event["attack_event_id"]]
    assert first_event["parent_event_id"] == expected_attack_event["parent_event_id"]
