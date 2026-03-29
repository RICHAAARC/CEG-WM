"""
文件目的：验证 01→02 parallel attestation statistics 的合同化收口。
Module type: General module
"""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import yaml

from main.cli import run_calibrate as run_calibrate_cli
from main.cli import run_evaluate as run_evaluate_cli
from main.evaluation import metrics as eval_metrics
from main.watermarking.detect import orchestrator as detect_orchestrator
from scripts.notebook_runtime_common import (
    build_failure_diagnostics_filename,
    compute_file_sha256,
    discover_stage_packages,
    finalize_stage_package,
    persist_source_package_lineage,
    probe_stage_package_policy,
    prepare_source_package,
    resolve_export_package_manifest_path,
    resolve_source_lineage_paths,
    resolve_source_prompt_snapshot_path,
)


CANONICAL_SOURCE_POOL_RELATIVE_ROOT = "artifacts/stage_01_canonical_source_pool"
CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH = f"{CANONICAL_SOURCE_POOL_RELATIVE_ROOT}/source_pool_manifest.json"
CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT = f"{CANONICAL_SOURCE_POOL_RELATIVE_ROOT}/entries"
CANONICAL_SOURCE_POOL_ATTESTATION_RELATIVE_ROOT = f"{CANONICAL_SOURCE_POOL_RELATIVE_ROOT}/attestation"
CANONICAL_SOURCE_POOL_SOURCE_IMAGES_RELATIVE_ROOT = f"{CANONICAL_SOURCE_POOL_RELATIVE_ROOT}/source_images"


def _load_script_module(relative_path: str, module_name: str) -> object:
    """
    功能：按路径加载 stage 脚本模块。

    Load a stage script module from a repository-relative path.

    Args:
        relative_path: Repository-relative script path.
        module_name: Temporary module name for importlib.

    Returns:
        Loaded module object.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path_obj: Path, payload: Dict[str, Any]) -> None:
    """
    功能：写入 JSON 测试工件。

    Write a JSON payload for tests.

    Args:
        path_obj: Destination path.
        payload: JSON-serializable mapping.

    Returns:
        None.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_bytes(path_obj: Path, payload: bytes) -> None:
    """
    功能：写入二进制测试工件。

    Write a binary payload for tests.

    Args:
        path_obj: Destination path.
        payload: Binary payload.

    Returns:
        None.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_bytes(payload)


def _strip_discovery_metadata_from_formal_package(package_path: Path, external_manifest_path: Path) -> None:
    """
    功能：把 formal package 改写为 legacy manifest 形态。 

    Rewrite a formal package so its manifests omit the new discovery metadata,
    simulating a legacy package.

    Args:
        package_path: Formal package ZIP path.
        external_manifest_path: External manifest path discovered beside the ZIP.

    Returns:
        None.
    """
    external_manifest = json.loads(external_manifest_path.read_text(encoding="utf-8"))
    external_manifest.pop("package_role", None)
    external_manifest.pop("package_discovery_scope", None)
    _write_json(external_manifest_path, external_manifest)

    with zipfile.ZipFile(package_path, "r") as archive:
        archived_entries = {name: archive.read(name) for name in archive.namelist()}

    internal_manifest = json.loads(archived_entries["artifacts/package_manifest.json"].decode("utf-8"))
    internal_manifest.pop("package_role", None)
    internal_manifest.pop("package_discovery_scope", None)
    archived_entries["artifacts/package_manifest.json"] = json.dumps(
        internal_manifest,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in archived_entries.items():
            archive.writestr(name, content)


def _create_failure_diagnostics_package(export_stage_root: Path, stage_run_id: str) -> Path:
    """
    功能：构造一个最小 failure diagnostics ZIP。 

    Create a minimal failure-diagnostics ZIP with an excluded external manifest.

    Args:
        export_stage_root: Stage export root used by discover_stage_packages.
        stage_run_id: Diagnostics stage run identifier.

    Returns:
        Diagnostics package ZIP path.
    """
    diagnostics_dir = export_stage_root / stage_run_id
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    package_path = diagnostics_dir / build_failure_diagnostics_filename("01_Paper_Full_Cuda", stage_run_id)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "failure_diagnostics_summary.json",
            json.dumps(
                {
                    "stage_name": "01_Paper_Full_Cuda",
                    "stage_run_id": stage_run_id,
                    "stage_status": "failed",
                    "failure_reason": "stage_01_mainline_failed",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        archive.writestr(
            "failure_diagnostics_index.json",
            json.dumps({"files": []}, ensure_ascii=False, indent=2),
        )
        archive.writestr(
            "failure_diagnostics_manifest.json",
            json.dumps(
                {
                    "artifact_type": "failure_diagnostics_manifest",
                    "stage_name": "01_Paper_Full_Cuda",
                    "stage_run_id": stage_run_id,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        archive.writestr(
            "artifacts/stage_manifest.json",
            json.dumps(
                {
                    "stage_name": "01_Paper_Full_Cuda",
                    "stage_run_id": stage_run_id,
                    "stage_status": "failed",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    external_manifest_path = resolve_export_package_manifest_path(package_path.parent)
    _write_json(
        external_manifest_path,
        {
            "stage_name": "01_Paper_Full_Cuda",
            "stage_run_id": stage_run_id,
            "package_filename": package_path.name,
            "package_path": package_path.as_posix(),
            "package_sha256": compute_file_sha256(package_path),
            "package_role": "failure_diagnostics_package",
            "package_discovery_scope": "excluded_from_formal_discovery",
            "package_created_at": "2025-01-01T00:00:00+00:00",
        },
    )
    return package_path


def _normalize_path_string(path_obj: Path) -> str:
    """
    功能：统一测试中的路径字符串格式。

    Normalize a path string for cross-platform assertions in tests.

    Args:
        path_obj: Path to normalize.

    Returns:
        Forward-slash path string.
    """
    return path_obj.as_posix()


def _make_detect_record(label: bool, score: float) -> Dict[str, Any]:
    """
    功能：构造最小 event_attestation detect record。

    Build a minimal detect record carrying canonical event-attestation fields.

    Args:
        label: Ground-truth label.
        score: Event attestation score.

    Returns:
        Detect record mapping.
    """
    return {
        "label": label,
        "ground_truth": label,
        "is_watermarked": label,
        "attestation": {
            "final_event_attested_decision": {
                "event_attestation_score": score,
                "event_attestation_score_name": "event_attestation_score",
                "is_event_attested": label,
            }
        },
    }


def _make_source_contract(records: list[Dict[str, Any]], *, direct_stats_ready: bool, direct_stats_reason: str) -> Dict[str, Any]:
    """
    功能：构造最小 source contract 测试载荷。

    Build a minimal stage-01 source contract payload for stage-02 tests.

    Args:
        records: Source record entries.
        direct_stats_ready: Whether the source is directly stats-ready.
        direct_stats_reason: Direct stats readiness reason.

    Returns:
        Source contract mapping.
    """
    positive = sum(1 for entry in records if entry["label"] is True)
    negative = sum(1 for entry in records if entry["label"] is False)
    return {
        "artifact_type": "parallel_attestation_statistics_input_contract",
        "contract_role": "source_contract",
        "contract_version": "v1",
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": "source_stage",
        "status": "ok",
        "reason": "parallel_attestation_statistics_source_records_available",
        "score_name": "event_attestation_score",
        "source_records_available": True,
        "record_count": len(records),
        "label_summary": {
            "positive": positive,
            "negative": negative,
            "unknown": 0,
            "label_balanced": positive > 0 and negative > 0,
        },
        "direct_stats_ready": direct_stats_ready,
        "direct_stats_reason": direct_stats_reason,
        "records": records,
    }


def _make_stage_01_source_pool_contract(run_root: Path, prompt_count: int) -> Dict[str, Any]:
    """
    功能：构造 stage 01 direct source pool contract 测试载荷。

    Build a stage-01 direct source pool contract payload for shell tests.

    Args:
        run_root: Stage-01 run root.
        prompt_count: Number of source prompts.

    Returns:
        Source contract mapping.
    """
    records: list[Dict[str, Any]] = []
    canonical_entries: list[Dict[str, Any]] = []
    for prompt_index in range(prompt_count):
        record_path = run_root / "artifacts" / "stage_01_source_pool_detect_records" / f"{prompt_index:03d}_detect_record.json"
        embed_record_path = run_root / "artifacts" / "stage_01_source_pool_embed_records" / f"{prompt_index:03d}_embed_record.json"
        runtime_config_path = run_root / "artifacts" / "stage_01_source_pool_runtime_configs" / f"prompt_{prompt_index:03d}.yaml"
        entry_package_relative_path = f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/{prompt_index:03d}_source_entry.json"
        entry_path = run_root / entry_package_relative_path
        attestation_statement_package_relative_path = (
            f"{CANONICAL_SOURCE_POOL_ATTESTATION_RELATIVE_ROOT}/prompt_{prompt_index:03d}/attestation_statement.json"
        )
        attestation_bundle_package_relative_path = (
            f"{CANONICAL_SOURCE_POOL_ATTESTATION_RELATIVE_ROOT}/prompt_{prompt_index:03d}/attestation_bundle.json"
        )
        attestation_result_package_relative_path = (
            f"{CANONICAL_SOURCE_POOL_ATTESTATION_RELATIVE_ROOT}/prompt_{prompt_index:03d}/attestation_result.json"
        )
        source_image_package_relative_path = (
            f"{CANONICAL_SOURCE_POOL_SOURCE_IMAGES_RELATIVE_ROOT}/prompt_{prompt_index:03d}/preview.png"
        )
        attestation_statement_path = run_root / attestation_statement_package_relative_path
        attestation_bundle_path = run_root / attestation_bundle_package_relative_path
        attestation_result_path = run_root / attestation_result_package_relative_path
        source_image_path = run_root / source_image_package_relative_path
        _write_json(
            record_path,
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "content_evidence_payload": {
                    "status": "ok",
                    "content_chain_score": 0.9 - prompt_index * 1e-3,
                },
                "attestation": {
                    "final_event_attested_decision": {
                        "event_attestation_score": 0.95 - prompt_index * 1e-3,
                        "event_attestation_score_name": "event_attestation_score",
                        "is_event_attested": True,
                    }
                },
            },
        )
        _write_json(embed_record_path, {"status": "ok", "prompt_index": prompt_index})
        runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_config_path.write_text(f"prompt_index: {prompt_index}\n", encoding="utf-8")
        _write_json(attestation_statement_path, {"status": "ok", "prompt_index": prompt_index})
        _write_json(attestation_bundle_path, {"status": "ok", "prompt_index": prompt_index})
        _write_json(
            attestation_result_path,
            {
                "status": "ok",
                "prompt_index": prompt_index,
                "final_event_attested_decision": {
                    "event_attestation_score": 0.95 - prompt_index * 1e-3,
                },
            },
        )
        _write_bytes(source_image_path, b"preview")
        _write_json(
            entry_path,
            {
                "artifact_type": "stage_01_canonical_source_entry",
                "entry_role": "canonical_source_entry",
                "path": _normalize_path_string(entry_path),
                "source_entry_package_relative_path": entry_package_relative_path,
                "prompt_index": prompt_index,
                "prompt_text": f"prompt {prompt_index}",
                "prompt_sha256": f"prompt_sha_{prompt_index:03d}",
                "prompt_file": "prompts/paper_small.txt",
                "detect_record_path": _normalize_path_string(record_path),
                "detect_record_package_relative_path": (
                    f"artifacts/stage_01_source_pool_detect_records/{prompt_index:03d}_detect_record.json"
                ),
                "embed_record_path": _normalize_path_string(embed_record_path),
                "embed_record_package_relative_path": (
                    f"artifacts/stage_01_source_pool_embed_records/{prompt_index:03d}_embed_record.json"
                ),
                "runtime_config_path": _normalize_path_string(runtime_config_path),
                "runtime_config_package_relative_path": (
                    f"artifacts/stage_01_source_pool_runtime_configs/prompt_{prompt_index:03d}.yaml"
                ),
                "attestation_statement": {
                    "exists": True,
                    "path": _normalize_path_string(attestation_statement_path),
                    "package_relative_path": attestation_statement_package_relative_path,
                    "missing_reason": None,
                },
                "attestation_bundle": {
                    "exists": True,
                    "path": _normalize_path_string(attestation_bundle_path),
                    "package_relative_path": attestation_bundle_package_relative_path,
                    "missing_reason": None,
                },
                "attestation_result": {
                    "exists": True,
                    "path": _normalize_path_string(attestation_result_path),
                    "package_relative_path": attestation_result_package_relative_path,
                    "missing_reason": None,
                },
                "source_image": {
                    "exists": True,
                    "path": _normalize_path_string(source_image_path),
                    "package_relative_path": source_image_package_relative_path,
                    "missing_reason": None,
                },
                "representative_root_records_alias": prompt_index == 0,
            },
        )
        canonical_entries.append(
            {
                "prompt_index": prompt_index,
                "prompt_text": f"prompt {prompt_index}",
                "prompt_sha256": f"prompt_sha_{prompt_index:03d}",
                "source_entry_package_relative_path": entry_package_relative_path,
                "detect_record_path": _normalize_path_string(record_path),
                "detect_record_package_relative_path": (
                    f"artifacts/stage_01_source_pool_detect_records/{prompt_index:03d}_detect_record.json"
                ),
                "embed_record_path": _normalize_path_string(embed_record_path),
                "embed_record_package_relative_path": (
                    f"artifacts/stage_01_source_pool_embed_records/{prompt_index:03d}_embed_record.json"
                ),
                "runtime_config_path": _normalize_path_string(runtime_config_path),
                "runtime_config_package_relative_path": (
                    f"artifacts/stage_01_source_pool_runtime_configs/prompt_{prompt_index:03d}.yaml"
                ),
                "attestation_statement": {
                    "exists": True,
                    "path": _normalize_path_string(attestation_statement_path),
                    "package_relative_path": attestation_statement_package_relative_path,
                    "missing_reason": None,
                },
                "attestation_bundle": {
                    "exists": True,
                    "path": _normalize_path_string(attestation_bundle_path),
                    "package_relative_path": attestation_bundle_package_relative_path,
                    "missing_reason": None,
                },
                "attestation_result": {
                    "exists": True,
                    "path": _normalize_path_string(attestation_result_path),
                    "package_relative_path": attestation_result_package_relative_path,
                    "missing_reason": None,
                },
                "source_image": {
                    "exists": True,
                    "path": _normalize_path_string(source_image_path),
                    "package_relative_path": source_image_package_relative_path,
                    "missing_reason": None,
                },
                "representative_root_records_alias": prompt_index == 0,
            }
        )
        records.append(
            {
                "record_role": "direct_source_record",
                "usage": "stage_01_direct_source_pool",
                "package_relative_path": (
                    f"artifacts/stage_01_source_pool_detect_records/{prompt_index:03d}_detect_record.json"
                ),
                "path": str(record_path),
                "sha256": f"detect_sha_{prompt_index:03d}",
                "label": True,
                "prompt_index": prompt_index,
                "prompt_text": f"prompt {prompt_index}",
                "prompt_sha256": f"prompt_sha_{prompt_index:03d}",
                "prompt_file": "prompts/paper_small.txt",
                "canonical_source_entry_package_relative_path": entry_package_relative_path,
                "embed_record_package_relative_path": (
                    f"artifacts/stage_01_source_pool_embed_records/{prompt_index:03d}_embed_record.json"
                ),
                "runtime_config_package_relative_path": (
                    f"artifacts/stage_01_source_pool_runtime_configs/prompt_{prompt_index:03d}.yaml"
                ),
                "score_name": "event_attestation_score",
                "score_available": True,
                "event_attestation_score_available": True,
                "threshold_score_name": "content_chain_score",
                "threshold_score_available": True,
                "content_chain_score_available": True,
            }
        )

    manifest_path = run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH
    _write_json(
        manifest_path,
        {
            "artifact_type": "stage_01_canonical_source_pool",
            "artifact_role": "canonical_source_pool_root",
            "stage_name": "01_Paper_Full_Cuda",
            "stage_run_id": "stage01_test",
            "source_truth": "canonical_source_pool",
            "root_contract_mode": "strong_compatibility",
            "root_records_required": True,
            "representative_root_role": "representative_summary_view",
            "prompt_file": "prompts/paper_small.txt",
            "canonical_source_pool_root_path": _normalize_path_string(run_root / CANONICAL_SOURCE_POOL_RELATIVE_ROOT),
            "canonical_source_pool_root_package_relative_path": CANONICAL_SOURCE_POOL_RELATIVE_ROOT,
            "manifest_path": _normalize_path_string(manifest_path),
            "manifest_package_relative_path": CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH,
            "entries_root_path": _normalize_path_string(run_root / CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT),
            "entries_package_relative_root": CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT,
            "prompt_count": prompt_count,
            "entry_count": prompt_count,
            "entries": canonical_entries,
            "representative_root_records": {
                "view_role": "representative_summary_view",
                "contract_mode": "strong_compatibility",
                "source_truth": "canonical_source_pool",
                "root_records_required": True,
                "root_embed_record_package_relative_path": "records/embed_record.json",
                "root_detect_record_package_relative_path": "records/detect_record.json",
                "source_prompt_index": 0,
                "source_prompt_sha256": "prompt_sha_000",
                "source_entry_package_relative_path": f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/000_source_entry.json",
                "source_embed_record_package_relative_path": "artifacts/stage_01_source_pool_embed_records/000_embed_record.json",
                "source_detect_record_package_relative_path": "artifacts/stage_01_source_pool_detect_records/000_detect_record.json",
            },
        },
    )

    return {
        "artifact_type": "parallel_attestation_statistics_input_contract",
        "contract_role": "source_contract",
        "contract_version": "v1",
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": "stage01_test",
        "status": "ok",
        "reason": "stage_01_direct_source_pool_ready",
        "source_authority": "canonical_source_pool",
        "contract_view_role": "stage_02_compatibility_view",
        "canonical_source_pool_manifest_package_relative_path": CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH,
        "canonical_source_pool_entries_package_relative_root": CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT,
        "representative_root_records": {
            "view_role": "representative_summary_view",
            "root_embed_record_package_relative_path": "records/embed_record.json",
            "root_detect_record_package_relative_path": "records/detect_record.json",
            "source_prompt_index": 0,
            "source_prompt_sha256": "prompt_sha_000",
            "source_entry_package_relative_path": f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/000_source_entry.json",
            "source_embed_record_package_relative_path": "artifacts/stage_01_source_pool_embed_records/000_embed_record.json",
            "source_detect_record_package_relative_path": "artifacts/stage_01_source_pool_detect_records/000_detect_record.json",
        },
        "score_name": "event_attestation_score",
        "threshold_score_name": "content_chain_score",
        "source_records_available": True,
        "record_count": prompt_count,
        "label_summary": {
            "positive": prompt_count,
            "negative": 0,
            "unknown": 0,
            "label_balanced": False,
        },
        "score_availability": {
            "content_chain_score": {
                "available_record_count": prompt_count,
                "missing_record_count": 0,
            },
            "event_attestation_score": {
                "available_record_count": prompt_count,
                "missing_record_count": 0,
            },
        },
        "direct_stats_ready": False,
        "direct_stats_reason": "parallel_attestation_statistics_requires_label_balanced_detect_records",
        "records": records,
    }


def _make_stage_01_pooled_threshold_contract(run_root: Path, prompt_count: int) -> Dict[str, Any]:
    """
    功能：构造 stage 01 pooled threshold build contract 测试载荷。

    Build a stage-01 pooled threshold build contract payload for shell tests.

    Args:
        run_root: Stage-01 run root.
        prompt_count: Number of direct source prompts.

    Returns:
        Build contract mapping.
    """
    records: list[Dict[str, Any]] = []
    direct_records: list[Dict[str, Any]] = []
    derived_records: list[Dict[str, Any]] = []
    pooled_root = run_root / "artifacts" / "stage_01_pooled_threshold_records"
    for prompt_index in range(prompt_count):
        direct_path = pooled_root / f"{prompt_index:03d}_direct_positive.json"
        _write_json(direct_path, {"label": True, "content_evidence_payload": {"status": "ok", "content_chain_score": 0.8}})
        direct_record = {
            "record_kind": "direct",
            "label": True,
            "usage": "stage_01_pooled_thresholds",
            "derived_from": None,
            "derivation_kind": None,
            "source_package_relative_path": (
                f"artifacts/stage_01_source_pool_detect_records/{prompt_index:03d}_detect_record.json"
            ),
            "source_entry_package_relative_path": (
                f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/{prompt_index:03d}_source_entry.json"
            ),
            "staged_path": str(direct_path),
            "package_relative_path": f"artifacts/stage_01_pooled_threshold_records/{prompt_index:03d}_direct_positive.json",
            "sha256": f"pooled_direct_sha_{prompt_index:03d}",
            "prompt_file": "prompts/paper_small.txt",
            "prompt_index": prompt_index,
            "prompt_text": f"prompt {prompt_index}",
        }
        direct_records.append(direct_record)
        records.append(direct_record)

    for prompt_index in range(prompt_count):
        derived_path = pooled_root / f"{prompt_count + prompt_index:03d}_derived_negative.json"
        _write_json(derived_path, {"label": False, "content_evidence_payload": {"status": "ok", "content_chain_score": -0.2}})
        derived_record = {
            "record_kind": "derived",
            "label": False,
            "usage": "stage_01_pooled_thresholds",
            "derived_from": f"artifacts/stage_01_source_pool_detect_records/{prompt_index:03d}_detect_record.json",
            "derivation_kind": "prompt_bound_label_balance",
            "source_package_relative_path": (
                f"artifacts/stage_01_source_pool_detect_records/{prompt_index:03d}_detect_record.json"
            ),
            "source_entry_package_relative_path": (
                f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/{prompt_index:03d}_source_entry.json"
            ),
            "staged_path": str(derived_path),
            "package_relative_path": (
                f"artifacts/stage_01_pooled_threshold_records/{prompt_count + prompt_index:03d}_derived_negative.json"
            ),
            "sha256": f"pooled_derived_sha_{prompt_index:03d}",
            "prompt_file": "prompts/paper_small.txt",
            "prompt_index": prompt_index,
            "prompt_text": f"prompt {prompt_index}",
        }
        derived_records.append(derived_record)
        records.append(derived_record)

    thresholds_path = run_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
    threshold_metadata_path = run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json"
    return {
        "artifact_type": "stage_01_pooled_threshold_build_contract",
        "contract_role": "pooled_threshold_build_contract",
        "contract_version": "v1",
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": "stage01_test",
        "requested_build_mode": "source_plus_derived_pairs",
        "build_mode": "source_plus_derived_pairs",
        "source_authority": "canonical_source_pool",
        "canonical_source_pool_manifest_package_relative_path": CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH,
        "score_name": "content_chain_score",
        "prompt_file": "prompts/paper_small.txt",
        "prompt_pool_summary": {
            "prompt_count": prompt_count,
            "prompt_indices": list(range(prompt_count)),
        },
        "staged_records_root": str(pooled_root),
        "detect_records_glob": str(pooled_root / "*.json"),
        "direct_record_count": prompt_count,
        "derived_record_count": prompt_count,
        "final_record_count": prompt_count * 2,
        "direct_summary": {"positive": prompt_count, "negative": 0},
        "derived_summary": {"positive": 0, "negative": prompt_count},
        "final_positive_count": prompt_count,
        "final_negative_count": prompt_count,
        "final_label_balanced": True,
        "build_configuration": {
            "target_pair_count": prompt_count,
            "build_usage": "stage_01_pooled_thresholds",
            "record_derivation_kind": "prompt_bound_label_balance",
        },
        "stats_input_set": {
            "score_name": "content_chain_score",
            "detect_records_glob": str(pooled_root / "*.json"),
            "direct_record_count": prompt_count,
            "derived_record_count": prompt_count,
            "final_record_count": prompt_count * 2,
            "positive_record_count": prompt_count,
            "negative_record_count": prompt_count,
            "thresholds_artifact_path": str(thresholds_path),
            "threshold_metadata_artifact_path": str(threshold_metadata_path),
        },
        "records": records,
        "direct_records": direct_records,
        "derived_records": derived_records,
        "thresholds_artifact_path": str(thresholds_path),
        "threshold_metadata_artifact_path": str(threshold_metadata_path),
    }


def _prepare_stage_01_mainline_config_monkeypatches(
    monkeypatch: pytest.MonkeyPatch,
    runner: Any,
    *,
    prompt_count: int,
) -> None:
    """
    功能：统一设置 stage 01 mainline 测试配置 monkeypatch。

    Apply the common stage-01 mainline configuration monkeypatches used by the
    attestation post-check tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        runner: Loaded stage-01 mainline module.
        prompt_count: Number of prompt-pool entries.

    Returns:
        None.
    """
    prompt_pool = [f"prompt {prompt_index}" for prompt_index in range(prompt_count)]
    monkeypatch.setattr(
        runner,
        "load_yaml_mapping",
        lambda _path: {
            "policy_path": "content_np_geo_rescue",
            "attestation": {"enabled": True},
            "embed": {
                "preview_generation": {
                    "enabled": True,
                    "artifact_rel_path": "preview/preview.png",
                }
            },
        },
    )
    monkeypatch.setattr(
        runner,
        "_resolve_stage_01_source_pool_cfg",
        lambda _cfg: {
            "enabled": True,
            "use_inference_prompt_file": True,
            "target_prompt_count": prompt_count,
            "record_usage": "stage_01_direct_source_pool",
        },
    )
    monkeypatch.setattr(
        runner,
        "_resolve_stage_01_pooled_threshold_build_cfg",
        lambda _cfg: {
            "enabled": True,
            "build_mode": "source_plus_derived_pairs",
            "target_pair_count": prompt_count,
            "build_usage": "stage_01_pooled_thresholds",
            "record_derivation_kind": "prompt_bound_label_balance",
        },
    )
    monkeypatch.setattr(
        runner,
        "_resolve_stage_01_prompt_pool",
        lambda _cfg: (prompt_pool, "prompts/paper_small.txt"),
    )


def _make_stage_01_mainline_run_stage(run_root: Path, attestation_prompt_indices: list[int]) -> Any:
    """
    功能：构造 stage 01 mainline 测试用的假 stage 执行器。

    Build the fake stage executor used by stage-01 mainline tests.

    Args:
        run_root: Stage-01 run root.
        attestation_prompt_indices: Prompt indices that emit attestation files.

    Returns:
        Callable-compatible fake _run_stage implementation.
    """
    attestation_index_set = set(attestation_prompt_indices)

    def _fake_run_stage(stage_name: str, _command: Any, stage_run_root: Path) -> Dict[str, Any]:
        if stage_name == "embed":
            prompt_index = int(stage_run_root.name.split("_")[-1])
            _write_json(stage_run_root / "records" / "embed_record.json", {"status": "ok"})
            if prompt_index == 0:
                _write_bytes(stage_run_root / "artifacts" / "preview" / "preview.png", b"preview")
        elif stage_name == "detect":
            prompt_index = int(stage_run_root.name.split("_")[-1])
            _write_json(
                stage_run_root / "records" / "detect_record.json",
                {
                    "label": True,
                    "ground_truth": True,
                    "is_watermarked": True,
                    "content_evidence_payload": {
                        "status": "ok",
                        "content_chain_score": 0.9 - prompt_index * 1e-3,
                    },
                    "attestation": {
                        "final_event_attested_decision": {
                            "event_attestation_score": 0.95 - prompt_index * 1e-3,
                            "event_attestation_score_name": "event_attestation_score",
                            "is_event_attested": True,
                        }
                    },
                },
            )
            if prompt_index in attestation_index_set:
                _write_json(stage_run_root / "artifacts" / "attestation" / "attestation_statement.json", {"status": "ok"})
                _write_json(stage_run_root / "artifacts" / "attestation" / "attestation_bundle.json", {"status": "ok"})
                _write_json(
                    stage_run_root / "artifacts" / "attestation" / "attestation_result.json",
                    {
                        "status": "ok",
                        "final_event_attested_decision": {
                            "event_attestation_score": 0.95 - prompt_index * 1e-3,
                        },
                    },
                )
        elif stage_name == "calibrate":
            _write_json(run_root / "records" / "calibration_record.json", {"status": "ok"})
            _write_json(run_root / "artifacts" / "thresholds" / "thresholds_artifact.json", {"threshold": 0.5})
            _write_json(
                run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json",
                {"meta": True},
            )
        elif stage_name == "evaluate":
            _write_json(run_root / "records" / "evaluate_record.json", {"status": "ok"})
            _write_json(run_root / "artifacts" / "evaluation_report.json", {"status": "ok"})
            _write_json(run_root / "artifacts" / "run_closure.json", {"status": "ok"})
        else:
            raise AssertionError(f"unexpected stage_name: {stage_name}")
        return {"return_code": 0, "stage_name": stage_name}

    return _fake_run_stage


def _prepare_stage_02_monkeypatches(
    monkeypatch: pytest.MonkeyPatch,
    stage_02: Any,
    drive_root: Path,
    run_root: Path,
    log_root: Path,
    runtime_state_root: Path,
    export_root: Path,
    extracted_root: Path,
    source_stage_manifest: Dict[str, Any],
    source_package_path: Path,
    config_payload: Dict[str, Any],
) -> None:
    """
    功能：统一设置 stage 02 测试所需 monkeypatch。

    Apply the common monkeypatch setup used by stage-02 contract tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        stage_02: Loaded stage-02 script module.
        drive_root: Temporary drive root.
        run_root: Stage run root.
        log_root: Stage log root.
        runtime_state_root: Stage runtime-state root.
        export_root: Stage export root.
        extracted_root: Prepared extracted source package root.
        source_stage_manifest: Source stage manifest payload.
        source_package_path: Placeholder ZIP path.
        config_payload: Runtime config payload returned by load_yaml_mapping.

    Returns:
        None.
    """
    monkeypatch.setattr(
        stage_02,
        "resolve_stage_roots",
        lambda *_args, **_kwargs: {
            "run_root": run_root,
            "log_root": log_root,
            "runtime_state_root": runtime_state_root,
            "export_root": export_root,
        },
    )
    monkeypatch.setattr(stage_02, "load_yaml_mapping", lambda _path: config_payload)
    monkeypatch.setattr(stage_02, "detect_stage_02_preflight", lambda *_args: {"ok": True})
    monkeypatch.setattr(
        stage_02,
        "prepare_source_package",
        lambda _source_package_path, _runtime_state_root: {
            "stage_manifest": source_stage_manifest,
            "package_manifest": {"status": "ok"},
            "extracted_root": str(extracted_root),
            "source_package_path": str(source_package_path),
            "source_package_sha256": "source_sha256",
            "package_manifest_digest": "manifest_digest",
        },
    )
    monkeypatch.setattr(stage_02, "collect_git_summary", lambda _root: {"commit": "test"})
    monkeypatch.setattr(stage_02, "collect_python_summary", lambda: {"version": "3.11"})
    monkeypatch.setattr(stage_02, "collect_cuda_summary", lambda: {"available": False})
    monkeypatch.setattr(stage_02, "collect_attestation_env_summary", lambda _cfg: {"enabled": True})
    monkeypatch.setattr(stage_02, "collect_model_summary", lambda _cfg: {"model": "test"})
    monkeypatch.setattr(stage_02, "collect_weight_summary", lambda _root, _cfg: {"weights": []})

    def _fake_stage_run(**kwargs: Any) -> Dict[str, Any]:
        stdout_path = kwargs["stdout_log_path"]
        stage_name = stdout_path.stem.replace("_stdout", "")
        _write_json(run_root / "records" / "calibration_record.json", {"status": "ok", "stage": stage_name})
        _write_json(run_root / "artifacts" / "thresholds" / "thresholds_artifact.json", {"threshold": 0.5})
        _write_json(run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json", {"meta": True})
        _write_json(run_root / "records" / "evaluate_record.json", {"status": "ok", "stage": stage_name})
        _write_json(run_root / "artifacts" / "evaluation_report.json", {"status": "ok"})
        _write_json(run_root / "artifacts" / "run_closure.json", {"status": "ok"})
        return {"return_code": 0}

    monkeypatch.setattr(stage_02, "run_command_with_logs", _fake_stage_run)
    monkeypatch.setattr(
        stage_02,
        "finalize_stage_package",
        lambda **_kwargs: {"package_path": str(export_root / "stage_02.zip"), "package_sha256": "sha256"},
    )


def test_stage_01_writes_source_contract_even_when_direct_stats_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：验证 stage 01 会写出 source contract，即使 direct stats 尚未 ready。

    Validate that stage 01 emits a source contract artifact even when the source
    detect output is not directly stats-ready.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    stage_01 = _load_script_module("scripts/01_Paper_Full_Cuda.py", "stage_01_contract_closure")
    drive_root = tmp_path / "drive"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")

    run_root = drive_root / "runs" / "stage_01"
    log_root = drive_root / "logs" / "stage_01"
    runtime_state_root = drive_root / "runtime_state" / "stage_01"
    export_root = drive_root / "exports" / "stage_01"

    monkeypatch.setattr(
        stage_01,
        "resolve_stage_roots",
        lambda *_args, **_kwargs: {
            "run_root": run_root,
            "log_root": log_root,
            "runtime_state_root": runtime_state_root,
            "export_root": export_root,
        },
    )
    monkeypatch.setattr(stage_01, "load_yaml_mapping", lambda _path: {"attestation": {}})
    monkeypatch.setattr(stage_01, "detect_stage_01_preflight", lambda _path: {"ok": True})
    monkeypatch.setattr(
        stage_01,
        "copy_prompt_snapshot",
        lambda *_args, **_kwargs: {
            "snapshot_path": str(runtime_state_root / "runtime_metadata" / "prompt_snapshot" / "prompt.txt"),
            "source_path": "prompts/paper_small.txt",
        },
    )
    monkeypatch.setattr(stage_01, "collect_git_summary", lambda _root: {"commit": "test"})
    monkeypatch.setattr(stage_01, "collect_python_summary", lambda: {"version": "3.11"})
    monkeypatch.setattr(stage_01, "collect_cuda_summary", lambda: {"available": False})
    monkeypatch.setattr(stage_01, "collect_attestation_env_summary", lambda _cfg: {"enabled": True})
    monkeypatch.setattr(stage_01, "collect_model_summary", lambda _cfg: {"model": "test"})
    monkeypatch.setattr(stage_01, "collect_weight_summary", lambda _root, _cfg: {"weights": []})

    def _fake_run_command_with_logs(**_kwargs: Any) -> Dict[str, Any]:
        prompt_count = 16
        _write_json(run_root / "records" / "embed_record.json", {"status": "ok"})
        _write_json(run_root / "records" / "detect_record.json", _make_detect_record(True, 0.91))
        _write_json(run_root / "records" / "calibration_record.json", {"status": "ok"})
        _write_json(run_root / "records" / "evaluate_record.json", {"status": "ok"})
        _write_json(run_root / "artifacts" / "thresholds" / "thresholds_artifact.json", {"threshold": 0.5})
        _write_json(run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json", {"meta": True})
        _write_json(run_root / "artifacts" / "evaluation_report.json", {"status": "ok"})
        _write_json(run_root / "artifacts" / "run_closure.json", {"status": "ok"})
        _write_json(run_root / "artifacts" / "workflow_summary.json", {"status": "ok"})
        _write_json(
            run_root / "artifacts" / "parallel_attestation_statistics_input_contract.json",
            _make_stage_01_source_pool_contract(run_root, prompt_count),
        )
        _write_json(
            run_root / "artifacts" / "stage_01_pooled_threshold_build_contract.json",
            _make_stage_01_pooled_threshold_contract(run_root, prompt_count),
        )
        return {"return_code": 0}

    monkeypatch.setattr(stage_01, "run_command_with_logs", _fake_run_command_with_logs)
    monkeypatch.setattr(
        stage_01,
        "finalize_stage_package",
        lambda **_kwargs: {"package_path": str(export_root / "stage_01.zip"), "package_sha256": "sha256"},
    )
    monkeypatch.setattr(
        stage_01,
        "_run_stage_01_output_audit",
        lambda **_kwargs: {
            "audit_summary_path": run_root / "artifacts" / "stage_01_audit_summary.json",
            "audit_summary": {
                "overall_status": "passed",
                "blocking_reasons": [],
                "warnings": [],
                "stage_02_ready": True,
                "stage_03_ready": True,
                "stage_04_ready": True,
            },
        },
    )

    stage_01.run_stage_01(
        drive_project_root=drive_root,
        config_path=config_path,
        notebook_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_test",
    )

    contract_path = run_root / "artifacts" / "parallel_attestation_statistics_input_contract.json"
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract_payload["artifact_type"] == "parallel_attestation_statistics_input_contract"
    assert contract_payload["contract_role"] == "source_contract"
    assert contract_payload["status"] == "ok"
    assert contract_payload["source_authority"] == "canonical_source_pool"
    assert contract_payload["contract_view_role"] == "stage_02_compatibility_view"
    assert contract_payload["canonical_source_pool_manifest_package_relative_path"] == (
        CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH
    )
    assert contract_payload["source_records_available"] is True
    assert contract_payload["direct_stats_ready"] is False
    assert contract_payload["record_count"] == 16
    assert contract_payload["direct_stats_reason"] == "parallel_attestation_statistics_requires_label_balanced_detect_records"
    assert contract_payload["records"][0]["canonical_source_entry_package_relative_path"] == (
        f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/000_source_entry.json"
    )

    canonical_manifest = json.loads(
        (run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    assert canonical_manifest["artifact_role"] == "canonical_source_pool_root"
    assert canonical_manifest["prompt_count"] == 16
    assert canonical_manifest["entry_count"] == 16
    assert canonical_manifest["source_truth"] == "canonical_source_pool"
    assert canonical_manifest["root_contract_mode"] == "strong_compatibility"
    assert canonical_manifest["root_records_required"] is True
    assert canonical_manifest["representative_root_role"] == "representative_summary_view"
    assert canonical_manifest["entries"][0]["attestation_statement"]["exists"] is True
    assert canonical_manifest["entries"][0]["source_image"]["exists"] is True
    assert canonical_manifest["representative_root_records"]["view_role"] == "representative_summary_view"
    assert canonical_manifest["representative_root_records"]["contract_mode"] == "strong_compatibility"
    assert canonical_manifest["representative_root_records"]["source_truth"] == "canonical_source_pool"
    assert canonical_manifest["representative_root_records"]["root_records_required"] is True

    pooled_build_contract = json.loads(
        (run_root / "artifacts" / "stage_01_pooled_threshold_build_contract.json").read_text(encoding="utf-8")
    )
    assert pooled_build_contract["build_mode"] == "source_plus_derived_pairs"
    assert pooled_build_contract["direct_record_count"] == 16
    assert pooled_build_contract["derived_record_count"] == 16
    assert pooled_build_contract["final_record_count"] == 32
    assert pooled_build_contract["final_label_balanced"] is True

    stage_manifest = json.loads((run_root / "artifacts" / "stage_manifest.json").read_text(encoding="utf-8"))
    assert stage_manifest["parallel_attestation_statistics_input_contract_status"] == "ok"
    assert stage_manifest["parallel_attestation_statistics_input_contract_source_records_available"] is True
    assert stage_manifest["parallel_attestation_statistics_input_contract_direct_stats_ready"] is False
    assert stage_manifest["parallel_attestation_statistics_input_contract_record_count"] == 16
    assert stage_manifest["parallel_attestation_statistics_input_contract_package_relative_path"] == (
        "artifacts/parallel_attestation_statistics_input_contract.json"
    )
    assert stage_manifest["stage_01_canonical_source_pool_manifest_package_relative_path"] == (
        CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH
    )
    assert stage_manifest["stage_01_canonical_source_pool_root_path"] == (
        run_root / CANONICAL_SOURCE_POOL_RELATIVE_ROOT
    ).as_posix()
    assert stage_manifest["stage_01_canonical_source_pool_prompt_count"] == 16
    assert stage_manifest["stage_01_canonical_source_pool_entry_count"] == 16
    assert stage_manifest["stage_01_root_contract_mode"] == "strong_compatibility"
    assert stage_manifest["stage_01_root_records_required"] is True
    assert stage_manifest["stage_01_source_truth"] == "canonical_source_pool"
    assert stage_manifest["stage_01_representative_root_role"] == "representative_summary_view"
    assert stage_manifest["stage_01_representative_root_records"]["view_role"] == "representative_summary_view"
    assert stage_manifest["stage_01_canonical_source_pool_prompt_file_path"] == "prompts/paper_small.txt"
    assert stage_manifest["stage_01_representative_root_prompt_index"] == 0
    assert stage_manifest["stage_01_representative_root_source_entry_package_relative_path"] == (
        f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/000_source_entry.json"
    )
    assert stage_manifest["stage_01_pooled_threshold_build_mode"] == "source_plus_derived_pairs"
    assert stage_manifest["stage_01_pooled_threshold_direct_record_count"] == 16

    stage_summary = json.loads((runtime_state_root / "stage_summary.json").read_text(encoding="utf-8"))
    assert stage_summary["audit_status"] == "passed"
    assert stage_summary["audit_blocking_reasons"] == []
    assert stage_manifest["stage_01_pooled_threshold_derived_record_count"] == 16
    assert stage_manifest["stage_01_pooled_threshold_final_record_count"] == 32
    assert stage_manifest["stage_01_pooled_threshold_final_label_balanced"] is True
    assert stage_manifest["attestation_evidence_status"] == "legacy_unavailable"
    assert stage_manifest["attestation_evidence_required_entry_count"] == 16
    assert stage_manifest["attestation_evidence_checked_entry_count"] == 0
    assert stage_manifest["attestation_evidence_missing_count"] == 0
    assert stage_manifest["attestation_evidence_summary_reason"] == "workflow_summary_attestation_resolution_missing"
    assert stage_manifest["attestation_evidence_representative_root_summary"]["resolution_role"] == (
        "representative_summary_view_only"
    )
    assert stage_manifest["formal_package_status"] == "generated"
    assert stage_manifest["formal_package_role"] == "formal_stage_package"
    assert stage_manifest["formal_package_discovery_scope"] == "discoverable_formal_only"
    assert stage_manifest["diagnostics_status"] == "not_generated"
    assert stage_manifest["diagnostics_generation_reason"] == "stage_completed_without_failure"
    assert stage_manifest["diagnostics_summary_path"] == "<absent>"
    assert stage_manifest["diagnostics_package_path"] == "<absent>"
    assert (runtime_state_root / "package_staging" / "artifacts" / "parallel_attestation_statistics_input_contract.json").exists()
    assert (runtime_state_root / "package_staging" / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH).exists()

    stage_summary = json.loads((runtime_state_root / "stage_summary.json").read_text(encoding="utf-8"))
    assert stage_summary["status"] == "ok"
    assert stage_summary["formal_package_status"] == "generated"
    assert stage_summary["diagnostics_status"] == "not_generated"
    assert stage_summary["diagnostics_generation_reason"] == "stage_completed_without_failure"
    assert stage_summary["diagnostics_summary_path"] == "<absent>"
    assert stage_summary["diagnostics_package_path"] == "<absent>"
    assert stage_summary["diagnostics_manifest_path"] == "<absent>"
    assert (
        runtime_state_root / "package_staging" / CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT / "015_source_entry.json"
    ).exists()
    assert (
        runtime_state_root / "package_staging" / "artifacts" / "stage_01_source_pool_detect_records" / "015_detect_record.json"
    ).exists()
    assert (
        runtime_state_root / "package_staging" / "artifacts" / "stage_01_source_pool_embed_records" / "015_embed_record.json"
    ).exists()
    assert (
        runtime_state_root / "package_staging" / "artifacts" / "stage_01_source_pool_runtime_configs" / "prompt_015.yaml"
    ).exists()
    assert (
        runtime_state_root / "package_staging" / CANONICAL_SOURCE_POOL_ATTESTATION_RELATIVE_ROOT / "prompt_015" / "attestation_result.json"
    ).exists()
    assert (
        runtime_state_root / "package_staging" / CANONICAL_SOURCE_POOL_SOURCE_IMAGES_RELATIVE_ROOT / "prompt_015" / "preview.png"
    ).exists()
    assert (
        runtime_state_root / "package_staging" / "artifacts" / "stage_01_pooled_threshold_records" / "031_derived_negative.json"
    ).exists()
    assert (
        runtime_state_root / "package_staging" / "artifacts" / "stage_01_pooled_threshold_build_contract.json"
    ).exists()


def test_default_config_enables_stage_01_pool_and_stage_02_target_pair_count() -> None:
    """
    功能：验证默认配置显式启用 stage 01 prompt pool 和 stage 02 的 16 对统计目标。

    Validate that the default config enables the stage-01 prompt pool and sets
    the stage-02 target pair count to 16.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parents[1]
    config_payload = yaml.safe_load((repo_root / "configs" / "default.yaml").read_text(encoding="utf-8"))

    assert config_payload["inference_prompt_file"] == "prompts/paper_small.txt"
    assert config_payload["stage_01_source_pool"]["enabled"] is True
    assert config_payload["stage_01_source_pool"]["use_inference_prompt_file"] is True
    assert config_payload["stage_01_source_pool"]["target_prompt_count"] == 16
    assert config_payload["stage_01_pooled_threshold_build"]["enabled"] is True
    assert config_payload["stage_01_pooled_threshold_build"]["build_mode"] == "source_plus_derived_pairs"
    assert config_payload["stage_01_pooled_threshold_build"]["target_pair_count"] == 16
    assert config_payload["parallel_attestation_statistics"]["target_pair_count"] == 16


def test_stage_01_runner_failure_exposes_log_tails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 stage 01 外层脚本在主链失败时抛出日志路径与日志尾部。

    Verify the stage-01 outer runner exposes log paths and tail lines when the
    mainline subprocess fails.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    stage_01 = _load_script_module("scripts/01_Paper_Full_Cuda.py", "stage_01_failure_diagnostics")
    drive_root = tmp_path / "drive"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")
    secret_plaintext = "TOP_SECRET_PROMPT_PAYLOAD"

    run_root = drive_root / "runs" / "stage_01"
    log_root = drive_root / "logs" / "stage_01"
    runtime_state_root = drive_root / "runtime_state" / "stage_01"
    export_root = drive_root / "exports" / "stage_01"
    prompt_snapshot_path = runtime_state_root / "runtime_metadata" / "prompt_snapshot" / "prompt.txt"
    prompt_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_snapshot_path.write_text(secret_plaintext, encoding="utf-8")

    monkeypatch.setattr(
        stage_01,
        "resolve_stage_roots",
        lambda *_args, **_kwargs: {
            "run_root": run_root,
            "log_root": log_root,
            "runtime_state_root": runtime_state_root,
            "export_root": export_root,
        },
    )
    monkeypatch.setattr(stage_01, "load_yaml_mapping", lambda _path: {"attestation": {}})
    monkeypatch.setattr(stage_01, "detect_stage_01_preflight", lambda _path: {"ok": True})
    monkeypatch.setattr(
        stage_01,
        "copy_prompt_snapshot",
        lambda *_args, **_kwargs: {
            "snapshot_path": str(prompt_snapshot_path),
            "source_path": "prompts/paper_small.txt",
        },
    )
    monkeypatch.setattr(stage_01, "collect_git_summary", lambda _root: {"commit": "test"})
    monkeypatch.setattr(stage_01, "collect_python_summary", lambda: {"version": "3.11"})
    monkeypatch.setattr(stage_01, "collect_cuda_summary", lambda: {"available": False})
    monkeypatch.setattr(stage_01, "collect_attestation_env_summary", lambda _cfg: {"enabled": True})
    monkeypatch.setattr(stage_01, "collect_model_summary", lambda _cfg: {"model": "test"})
    monkeypatch.setattr(stage_01, "collect_weight_summary", lambda _root, _cfg: {"weights": []})

    def _fake_run_command_with_logs(**kwargs: Any) -> Dict[str, Any]:
        stdout_log_path = Path(kwargs["stdout_log_path"])
        stderr_log_path = Path(kwargs["stderr_log_path"])
        stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_log_path.write_text("mainline stdout\n", encoding="utf-8")
        stderr_log_path.write_text("nested failure detail\n", encoding="utf-8")
        _write_json(
            run_root / "artifacts" / "workflow_summary.json",
            {
                "status": "failed",
                "summary_reason": "attestation_evidence_missing_required_artifacts",
                "failure_reason": "attestation_evidence_missing_required_artifacts",
                "required_artifacts": {"workflow_summary": True},
                "attestation_evidence_resolution": {
                    "overall_status": "failed",
                    "required_entry_count": 1,
                    "checked_entry_count": 0,
                    "missing_evidence_count": 1,
                    "failing_prompt_indices": [0],
                    "failing_source_entry_paths": [
                        f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/000_source_entry.json"
                    ],
                },
            },
        )
        _write_json(
            run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH,
            {
                "entry_count": 1,
                "prompt_count": 1,
                "entries_package_relative_root": CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT,
                "representative_root_records": {
                    "view_role": "representative_summary_view",
                    "source_prompt_index": 0,
                    "source_entry_package_relative_path": (
                        f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/000_source_entry.json"
                    ),
                },
            },
        )
        nested_log_path = run_root / "logs" / "source_pool_failure.log"
        nested_log_path.parent.mkdir(parents=True, exist_ok=True)
        nested_log_path.write_text("source pool failed before artifacts closed\n", encoding="utf-8")
        return {
            "return_code": 1,
            "stdout_log_path": str(stdout_log_path),
            "stderr_log_path": str(stderr_log_path),
            "command": ["python", "scripts/01_run_paper_full_cuda.py"],
        }

    monkeypatch.setattr(stage_01, "run_command_with_logs", _fake_run_command_with_logs)

    with pytest.raises(RuntimeError) as exc_info:
        stage_01.run_stage_01(
            drive_project_root=drive_root,
            config_path=config_path,
            notebook_name="01_Paper_Full_Cuda",
            stage_run_id="stage01_failure",
        )

    error_text = str(exc_info.value)
    assert "stage 01 mainline failed" in error_text
    assert "nested failure detail" in error_text
    assert "01_mainline_stderr.log" in error_text

    stage_summary = json.loads((runtime_state_root / "stage_summary.json").read_text(encoding="utf-8"))
    stage_manifest = json.loads((run_root / "artifacts" / "stage_manifest.json").read_text(encoding="utf-8"))
    diagnostics_root = runtime_state_root / "failure_diagnostics"
    diagnostics_summary_path = diagnostics_root / "failure_diagnostics_summary.json"
    diagnostics_manifest_path = diagnostics_root / "failure_diagnostics_manifest.json"
    diagnostics_index_path = diagnostics_root / "failure_diagnostics_index.json"
    diagnostics_package_path = diagnostics_root / build_failure_diagnostics_filename(
        "01_Paper_Full_Cuda",
        "stage01_failure",
    )

    assert stage_summary["status"] == "failed"
    assert stage_summary["formal_package_status"] == "not_generated"
    assert stage_summary["diagnostics_status"] == "generated"
    assert stage_summary["diagnostics_generation_reason"] == "attestation_evidence_missing_required_artifacts"
    assert stage_summary["diagnostics_summary_path"] == diagnostics_summary_path.as_posix()
    assert stage_summary["diagnostics_package_path"] == diagnostics_package_path.as_posix()
    assert stage_summary["diagnostics_manifest_path"] == diagnostics_manifest_path.as_posix()
    assert stage_manifest["stage_status"] == "failed"
    assert stage_manifest["formal_package_role"] == "formal_stage_package"
    assert stage_manifest["diagnostics_status"] == "generated"
    assert stage_manifest["diagnostics_package_path"] == diagnostics_package_path.as_posix()

    diagnostics_summary = json.loads(diagnostics_summary_path.read_text(encoding="utf-8"))
    diagnostics_manifest = json.loads(diagnostics_manifest_path.read_text(encoding="utf-8"))
    assert diagnostics_index_path.exists()
    assert diagnostics_package_path.exists()
    assert diagnostics_summary["stage_name"] == "01_Paper_Full_Cuda"
    assert diagnostics_summary["stage_run_id"] == "stage01_failure"
    assert diagnostics_summary["stage_status"] == "failed"
    assert diagnostics_summary["failure_reason"] == "attestation_evidence_missing_required_artifacts"
    assert diagnostics_summary["workflow_summary_path"] == (run_root / "artifacts" / "workflow_summary.json").as_posix()
    assert diagnostics_summary["stage_manifest_path"] == (run_root / "artifacts" / "stage_manifest.json").as_posix()
    assert diagnostics_summary["log_root"] == log_root.as_posix()
    assert diagnostics_summary["command_stderr_tail"][-1] == "nested failure detail"
    assert diagnostics_summary["preflight_ok"] is True
    assert diagnostics_summary["attestation_evidence_resolution"]["overall_status"] == "failed"
    assert "embed_record" in diagnostics_summary["missing_required_artifacts"]
    assert diagnostics_summary["canonical_source_pool_summary"]["entry_count"] == 1
    assert diagnostics_manifest["package_role"] == "failure_diagnostics_package"
    assert diagnostics_manifest["package_discovery_scope"] == "excluded_from_formal_discovery"
    assert diagnostics_manifest["diagnostics_package_path"] == diagnostics_package_path.as_posix()
    assert diagnostics_manifest["diagnostics_package_filename"].endswith("failure_diagnostics.zip")

    diagnostics_summary_text = diagnostics_summary_path.read_text(encoding="utf-8")
    diagnostics_manifest_text = diagnostics_manifest_path.read_text(encoding="utf-8")
    assert secret_plaintext not in diagnostics_summary_text
    assert secret_plaintext not in diagnostics_manifest_text

    with zipfile.ZipFile(diagnostics_package_path, "r") as archive:
        archived_names = set(archive.namelist())
        assert "failure_diagnostics_summary.json" in archived_names
        assert "failure_diagnostics_manifest.json" in archived_names
        assert "failure_diagnostics_index.json" in archived_names
        assert "artifacts/stage_manifest.json" in archived_names
        assert "artifacts/workflow_summary.json" in archived_names
        assert "artifacts/package_manifest.json" not in archived_names
        assert "artifacts/package_index.json" not in archived_names
        for archived_name in archived_names:
            if archived_name.endswith("/"):
                continue
            assert secret_plaintext not in archive.read(archived_name).decode("utf-8", errors="ignore")


def test_discover_stage_packages_excludes_failure_diagnostics_and_accepts_legacy_formal_packages(
    tmp_path: Path,
) -> None:
    """
    功能：验证 diagnostics package 不会进入 formal discovery，且 legacy formal package 仍可发现。 

    Verify failure-diagnostics ZIPs are excluded from formal discovery while
    legacy formal packages without the new metadata remain discoverable.

    Args:
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    export_stage_root = tmp_path / "exports" / "stage_01"
    run_root = tmp_path / "runs" / "stage_01"
    package_root = tmp_path / "package_root" / "stage_01"
    package_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        package_root / "artifacts" / "stage_manifest.json",
        {
            "stage_name": "01_Paper_Full_Cuda",
            "stage_run_id": "legacy_formal_run",
            "stage_status": "ok",
            "workflow_summary_path": "/drive/runs/01/artifacts/workflow_summary.json",
        },
    )
    _write_json(package_root / "artifacts" / "workflow_summary.json", {"status": "ok"})

    formal_package_manifest_path = run_root / "artifacts" / "package_manifest.json"
    formal_package = finalize_stage_package(
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="legacy_formal_run",
        package_root=package_root,
        export_root=export_stage_root,
        source_stage_run_id=None,
        source_stage_package_path=None,
        package_manifest_path=formal_package_manifest_path,
    )
    formal_package_path = Path(str(formal_package["package_path"]))
    external_manifest_path = resolve_export_package_manifest_path(formal_package_path.parent)
    _strip_discovery_metadata_from_formal_package(formal_package_path, external_manifest_path)

    diagnostics_package_path = _create_failure_diagnostics_package(export_stage_root, "failed_run")
    formal_policy_probe = probe_stage_package_policy(formal_package_path)
    diagnostics_policy_probe = probe_stage_package_policy(diagnostics_package_path)
    discovered_packages = discover_stage_packages(export_stage_root)

    assert formal_policy_probe["formal_package_eligible"] is True
    assert formal_policy_probe["explicit_non_formal"] is False
    assert diagnostics_policy_probe["formal_package_eligible"] is False
    assert diagnostics_policy_probe["diagnostics_like"] is True
    assert diagnostics_policy_probe["explicit_non_formal"] is True
    assert len(discovered_packages) == 1
    assert discovered_packages[0]["package_path"] == formal_package_path.as_posix()
    assert discovered_packages[0]["package_role"] == "formal_stage_package"
    assert discovered_packages[0]["package_discovery_scope"] == "discoverable_formal_only"

    with pytest.raises(ValueError) as exc_info:
        prepare_source_package(diagnostics_package_path, tmp_path / "runtime_state")

    assert "discoverable formal stage package" in str(exc_info.value)


def test_shared_source_lineage_helpers_preserve_contract_paths_and_snapshots(tmp_path: Path) -> None:
    """
    功能：验证共享 lineage helper 保持既有路径合同与快照写出语义。

    Verify the shared source-lineage helpers preserve the established path
    contract and lineage snapshot persistence semantics.

    Args:
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    extracted_root = tmp_path / "source_extracted"
    _write_json(
        extracted_root / "artifacts" / "stage_manifest.json",
        {"stage_name": "01_Paper_Full_Cuda", "stage_run_id": "stage01_run"},
    )
    _write_json(
        extracted_root / "artifacts" / "package_manifest.json",
        {
            "stage_name": "01_Paper_Full_Cuda",
            "stage_run_id": "stage01_run",
            "package_filename": "source_stage01.zip",
        },
    )
    _write_json(
        extracted_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
        {"thresholds_digest": "thr01"},
    )
    runtime_config_snapshot_path = extracted_root / "runtime_metadata" / "runtime_config_snapshot.yaml"
    runtime_config_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_config_snapshot_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")
    prompt_snapshot_path = extracted_root / "runtime_metadata" / "prompt_snapshot" / "prompt.txt"
    prompt_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_snapshot_path.write_text("prompt text\n", encoding="utf-8")

    lineage_paths = resolve_source_lineage_paths(extracted_root)
    assert lineage_paths["source_stage_manifest_path"] == extracted_root / "artifacts" / "stage_manifest.json"
    assert lineage_paths["source_package_manifest_path"] == extracted_root / "artifacts" / "package_manifest.json"
    assert lineage_paths["source_runtime_config_snapshot_path"] == runtime_config_snapshot_path
    assert lineage_paths["source_thresholds_artifact_path"] == extracted_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
    assert resolve_source_prompt_snapshot_path(extracted_root) == prompt_snapshot_path.as_posix()

    runtime_state_root = tmp_path / "runtime_state"
    persisted_paths = persist_source_package_lineage(
        runtime_state_root,
        {
            "stage_manifest": {"stage_name": "01_Paper_Full_Cuda", "stage_run_id": "stage01_run"},
            "package_manifest": {
                "stage_name": "01_Paper_Full_Cuda",
                "stage_run_id": "stage01_run",
                "package_filename": "source_stage01.zip",
            },
        },
    )

    assert persisted_paths["source_stage_manifest_copy_path"] == runtime_state_root / "lineage" / "source_stage_manifest.json"
    assert persisted_paths["source_package_manifest_copy_path"] == runtime_state_root / "lineage" / "source_package_manifest.json"
    assert json.loads(persisted_paths["source_stage_manifest_copy_path"].read_text(encoding="utf-8"))["stage_run_id"] == "stage01_run"
    assert json.loads(persisted_paths["source_package_manifest_copy_path"].read_text(encoding="utf-8"))["package_filename"] == "source_stage01.zip"


def test_stage_01_source_pool_failure_exposes_nested_log_tails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 stage 01 source pool 子运行失败时抛出嵌套日志诊断。

    Verify source-pool subrun failures expose nested stage log paths and tail
    lines.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    runner = _load_script_module("scripts/01_run_paper_full_cuda.py", "stage_01_source_pool_failure_diagnostics")
    run_root = tmp_path / "stage_01_run"

    def _fake_run_stage(stage_name: str, command: Any, prompt_run_root: Path) -> Dict[str, Any]:
        _ = command
        logs_dir = prompt_run_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_log_path = logs_dir / f"{stage_name}_stdout.log"
        stderr_log_path = logs_dir / f"{stage_name}_stderr.log"
        stdout_log_path.write_text("source pool stdout\n", encoding="utf-8")
        stderr_log_path.write_text("embed exploded in source pool\n", encoding="utf-8")
        return {
            "return_code": 1,
            "status": "failed",
            "command": ["python", stage_name],
            "stdout_log_path": str(stdout_log_path),
            "stderr_log_path": str(stderr_log_path),
        }

    monkeypatch.setattr(runner, "_run_stage", _fake_run_stage)

    with pytest.raises(RuntimeError) as exc_info:
        runner._run_source_pool_subrun(
            index=0,
            prompt_text="prompt text",
            prompt_file_path="prompts/paper_small.txt",
            cfg_obj={"policy_path": "content_np_geo_rescue"},
            run_root=run_root,
            record_usage="stage_01_direct_source_pool",
        )

    error_text = str(exc_info.value)
    assert "stage 01 source pool stage failed" in error_text
    assert "embed exploded in source pool" in error_text
    assert "embed_stderr.log" in error_text


def test_stage_01_mainline_writes_workflow_summary_for_source_pool_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 source pool/build 异常会写出 workflow_summary 并返回非零。

    Verify source-pool or build exceptions are summarized into workflow_summary
    and converted into a non-zero process-style return code.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    runner = _load_script_module("scripts/01_run_paper_full_cuda.py", "stage_01_mainline_exception_summary")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")
    run_root = tmp_path / "run_root"

    monkeypatch.setattr(runner, "load_yaml_mapping", lambda _path: {"policy_path": "content_np_geo_rescue"})
    monkeypatch.setattr(
        runner,
        "_resolve_stage_01_source_pool_cfg",
        lambda _cfg: {
            "enabled": True,
            "use_inference_prompt_file": True,
            "target_prompt_count": 1,
            "record_usage": "stage_01_direct_source_pool",
        },
    )
    monkeypatch.setattr(
        runner,
        "_resolve_stage_01_pooled_threshold_build_cfg",
        lambda _cfg: {
            "enabled": True,
            "build_mode": "source_plus_derived_pairs",
            "target_pair_count": 1,
            "build_usage": "stage_01_pooled_thresholds",
            "record_derivation_kind": "prompt_bound_label_balance",
        },
    )
    monkeypatch.setattr(runner, "_resolve_stage_01_prompt_pool", lambda _cfg: (["prompt 0"], "prompts/paper_small.txt"))
    monkeypatch.setattr(
        runner,
        "_run_source_pool_subrun",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("source pool exploded")),
    )

    exit_code = runner.run_paper_full_cuda(config_path, run_root, stage_run_id="stage01_failure")
    workflow_summary_path = run_root / "artifacts" / "workflow_summary.json"
    workflow_summary = json.loads(workflow_summary_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert workflow_summary["status"] == "failed"
    assert workflow_summary["exception_type"] == "RuntimeError"
    assert workflow_summary["exception_message"] == "source pool exploded"
    assert workflow_summary["source_pool_prompt_count"] == 1


def test_stage_01_mainline_promotes_canonical_source_pool_and_keeps_compatibility_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 stage 01 mainline 以 canonical source pool 为权威，并保留兼容 contract。

    Validate that the stage-01 mainline emits a canonical source-pool root
    while preserving the compatibility contract consumed by later stages.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    runner = _load_script_module("scripts/01_run_paper_full_cuda.py", "stage_01_canonical_source_pool_success")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")
    run_root = tmp_path / "run_root"

    _prepare_stage_01_mainline_config_monkeypatches(monkeypatch, runner, prompt_count=2)
    monkeypatch.setattr(runner, "_run_stage", _make_stage_01_mainline_run_stage(run_root, [0, 1]))

    exit_code = runner.run_paper_full_cuda(config_path, run_root, stage_run_id="stage01_success")

    canonical_manifest = json.loads(
        (run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    source_contract = json.loads(
        (run_root / "artifacts" / "parallel_attestation_statistics_input_contract.json").read_text(encoding="utf-8")
    )
    workflow_summary = json.loads((run_root / "artifacts" / "workflow_summary.json").read_text(encoding="utf-8"))
    canonical_entry_0 = json.loads(
        (run_root / CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT / "000_source_entry.json").read_text(encoding="utf-8")
    )
    canonical_entry_1 = json.loads(
        (run_root / CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT / "001_source_entry.json").read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert canonical_manifest["artifact_role"] == "canonical_source_pool_root"
    assert canonical_manifest["prompt_count"] == 2
    assert canonical_manifest["entry_count"] == 2
    assert canonical_manifest["representative_root_records"]["view_role"] == "representative_summary_view"
    assert source_contract["source_authority"] == "canonical_source_pool"
    assert source_contract["contract_view_role"] == "stage_02_compatibility_view"
    assert source_contract["canonical_source_pool_manifest_package_relative_path"] == CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH
    assert source_contract["records"][0]["canonical_source_entry_package_relative_path"] == (
        f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/000_source_entry.json"
    )
    assert canonical_entry_0["attestation_statement"]["exists"] is True
    assert canonical_entry_0["attestation_bundle"]["exists"] is True
    assert canonical_entry_0["attestation_result"]["exists"] is True
    assert canonical_entry_0["source_image"]["exists"] is True
    assert canonical_entry_1["attestation_statement"]["exists"] is True
    assert canonical_entry_1["attestation_bundle"]["exists"] is True
    assert canonical_entry_1["attestation_result"]["exists"] is True
    assert canonical_entry_1["source_image"]["exists"] is False
    assert canonical_entry_1["source_image"]["missing_reason"] == "source_image_not_emitted"
    assert workflow_summary["status"] == "ok"
    assert workflow_summary["required_artifacts_ok"] is True
    assert workflow_summary["attestation_evidence_ok"] is True
    assert workflow_summary["summary_reason"] == "ok"
    assert workflow_summary["failure_reason"] is None
    assert workflow_summary["canonical_source_pool_entry_count"] == 2
    assert workflow_summary["representative_root_records"]["view_role"] == "representative_summary_view"
    assert workflow_summary["attestation_evidence_resolution"]["overall_status"] == "ok"
    assert workflow_summary["attestation_evidence_resolution"]["required_entry_count"] == 2
    assert workflow_summary["attestation_evidence_resolution"]["checked_entry_count"] == 2
    assert workflow_summary["attestation_evidence_resolution"]["missing_evidence_count"] == 0
    assert workflow_summary["attestation_evidence_resolution"]["failing_prompt_indices"] == []
    assert workflow_summary["attestation_evidence_resolution"]["failing_source_entry_paths"] == []
    assert workflow_summary["attestation_evidence_resolution"]["representative_root_summary"]["resolution_role"] == (
        "representative_summary_view_only"
    )


def test_stage_01_mainline_fails_when_required_canonical_attestation_exists_flag_is_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 canonical source entry 若将必需证据标成 exists=false，stage 01 必须失败。

    Verify stage-01 success is blocked when a canonical source entry marks one
    required attestation artifact as exists=false.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    runner = _load_script_module("scripts/01_run_paper_full_cuda.py", "stage_01_attestation_exists_false")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")
    run_root = tmp_path / "run_root"

    _prepare_stage_01_mainline_config_monkeypatches(monkeypatch, runner, prompt_count=2)
    monkeypatch.setattr(runner, "_run_stage", _make_stage_01_mainline_run_stage(run_root, [0, 1]))

    original_build = runner._build_stage_01_canonical_source_pool

    def _tamper_canonical_source_pool(**kwargs: Any) -> Dict[str, Any]:
        manifest_payload = original_build(**kwargs)
        entry_path = run_root / CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT / "001_source_entry.json"
        entry_payload = json.loads(entry_path.read_text(encoding="utf-8"))
        tampered_bundle = dict(entry_payload["attestation_bundle"])
        tampered_bundle["exists"] = False
        entry_payload["attestation_bundle"] = tampered_bundle
        _write_json(entry_path, entry_payload)
        manifest_payload["entries"][1]["attestation_bundle"] = tampered_bundle
        _write_json(run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH, manifest_payload)
        return manifest_payload

    monkeypatch.setattr(runner, "_build_stage_01_canonical_source_pool", _tamper_canonical_source_pool)

    exit_code = runner.run_paper_full_cuda(config_path, run_root, stage_run_id="stage01_exists_false")
    workflow_summary = json.loads((run_root / "artifacts" / "workflow_summary.json").read_text(encoding="utf-8"))
    attestation_resolution = workflow_summary["attestation_evidence_resolution"]
    bundle_check = next(
        artifact_check
        for artifact_check in attestation_resolution["failing_entries"][0]["artifact_checks"]
        if artifact_check["artifact_key"] == "attestation_bundle"
    )

    assert exit_code == 1
    assert workflow_summary["status"] == "failed"
    assert workflow_summary["required_artifacts_ok"] is True
    assert workflow_summary["attestation_evidence_ok"] is False
    assert workflow_summary["summary_reason"] == "failed_attestation_evidence"
    assert workflow_summary["failure_reason"] == "failed_attestation_evidence"
    assert attestation_resolution["overall_status"] == "missing_evidence"
    assert attestation_resolution["required_entry_count"] == 2
    assert attestation_resolution["checked_entry_count"] == 2
    assert attestation_resolution["missing_evidence_count"] == 1
    assert attestation_resolution["failing_prompt_indices"] == [1]
    assert attestation_resolution["failing_entries"][0]["missing_artifact_keys"] == ["attestation_bundle"]
    assert "attestation_bundle.exists_not_true" in bundle_check["missing_reasons"]


def test_stage_01_mainline_representative_root_cannot_mask_non_root_attestation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 representative root 仅是摘要视图，不能掩盖其他 canonical entry 的证据缺失。

    Verify representative-root records remain a summary view and cannot mask a
    missing attestation artifact in another canonical source entry.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    runner = _load_script_module("scripts/01_run_paper_full_cuda.py", "stage_01_representative_root_masking")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")
    run_root = tmp_path / "run_root"

    _prepare_stage_01_mainline_config_monkeypatches(monkeypatch, runner, prompt_count=2)
    monkeypatch.setattr(runner, "_run_stage", _make_stage_01_mainline_run_stage(run_root, [0, 1]))

    original_build = runner._build_stage_01_canonical_source_pool
    missing_package_relative_path = (
        f"{CANONICAL_SOURCE_POOL_ATTESTATION_RELATIVE_ROOT}/prompt_001/attestation_result_missing.json"
    )

    def _tamper_canonical_source_pool(**kwargs: Any) -> Dict[str, Any]:
        manifest_payload = original_build(**kwargs)
        entry_path = run_root / CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT / "001_source_entry.json"
        entry_payload = json.loads(entry_path.read_text(encoding="utf-8"))
        tampered_result = dict(entry_payload["attestation_result"])
        tampered_result["package_relative_path"] = missing_package_relative_path
        entry_payload["attestation_result"] = tampered_result
        _write_json(entry_path, entry_payload)
        manifest_payload["entries"][1]["attestation_result"] = tampered_result
        _write_json(run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH, manifest_payload)
        return manifest_payload

    monkeypatch.setattr(runner, "_build_stage_01_canonical_source_pool", _tamper_canonical_source_pool)

    exit_code = runner.run_paper_full_cuda(config_path, run_root, stage_run_id="stage01_rep_root_mask")
    workflow_summary = json.loads((run_root / "artifacts" / "workflow_summary.json").read_text(encoding="utf-8"))
    attestation_resolution = workflow_summary["attestation_evidence_resolution"]
    result_check = next(
        artifact_check
        for artifact_check in attestation_resolution["failing_entries"][0]["artifact_checks"]
        if artifact_check["artifact_key"] == "attestation_result"
    )

    assert exit_code == 1
    assert workflow_summary["status"] == "failed"
    assert workflow_summary["required_artifacts_ok"] is True
    assert workflow_summary["failure_reason"] == "failed_attestation_evidence"
    assert workflow_summary["representative_root_records"]["source_prompt_index"] == 0
    assert attestation_resolution["representative_root_summary"]["source_prompt_index"] == 0
    assert attestation_resolution["representative_root_summary"]["resolution_role"] == (
        "representative_summary_view_only"
    )
    assert attestation_resolution["failing_prompt_indices"] == [1]
    assert attestation_resolution["failing_source_entry_paths"] == [
        (run_root / CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT / "001_source_entry.json").resolve().as_posix()
    ]
    assert attestation_resolution["missing_evidence_count"] == 1
    assert attestation_resolution["failing_entries"][0]["missing_artifact_keys"] == ["attestation_result"]
    assert "attestation_result.package_relative_path_not_found" in result_check["missing_reasons"]


def test_stage_02_direct_only_build_uses_source_records_and_writes_build_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证当 source 已平衡时，stage 02 直接消费 source records 并写出 direct-only build contract。

    Validate that stage 02 consumes balanced source records directly and emits a
    direct-only build contract without derived records.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    stage_02 = _load_script_module("scripts/02_Parallel_Attestation_Statistics.py", "stage_02_direct_only")
    drive_root = tmp_path / "drive"
    source_package_path = tmp_path / "stage_01_package.zip"
    source_package_path.write_text("placeholder", encoding="utf-8")

    run_root = drive_root / "runs" / "stage_02"
    log_root = drive_root / "logs" / "stage_02"
    runtime_state_root = drive_root / "runtime_state" / "stage_02"
    export_root = drive_root / "exports" / "stage_02"
    extracted_root = tmp_path / "source_package_extract_direct"

    positive_path = extracted_root / "records" / "detect_positive.json"
    negative_path = extracted_root / "records" / "detect_negative.json"
    _write_json(positive_path, _make_detect_record(True, 0.92))
    _write_json(negative_path, _make_detect_record(False, 0.0))

    source_contract = _make_source_contract(
        [
            {
                "package_relative_path": "records/detect_positive.json",
                "sha256": stage_02.compute_file_sha256(positive_path),
                "label": True,
            },
            {
                "package_relative_path": "records/detect_negative.json",
                "sha256": stage_02.compute_file_sha256(negative_path),
                "label": False,
            },
        ],
        direct_stats_ready=True,
        direct_stats_reason="ok",
    )
    _write_json(extracted_root / "artifacts" / "parallel_attestation_statistics_input_contract.json", source_contract)
    source_stage_manifest = {
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": "source_stage",
        "parallel_attestation_statistics_input_contract_package_relative_path": (
            "artifacts/parallel_attestation_statistics_input_contract.json"
        ),
    }
    _write_json(extracted_root / "artifacts" / "stage_manifest.json", source_stage_manifest)
    _write_json(extracted_root / "artifacts" / "thresholds" / "thresholds_artifact.json", {"threshold": 0.5})
    (extracted_root / "runtime_metadata").mkdir(parents=True, exist_ok=True)
    (extracted_root / "runtime_metadata" / "runtime_config_snapshot.yaml").write_text("calibration: {}\n", encoding="utf-8")

    _prepare_stage_02_monkeypatches(
        monkeypatch,
        stage_02,
        drive_root,
        run_root,
        log_root,
        runtime_state_root,
        export_root,
        extracted_root,
        source_stage_manifest,
        source_package_path,
        {
            "parallel_attestation_statistics": {
                "enabled": True,
                "build_mode": "direct_source_only",
                "allow_derived_input_build": False,
                "target_pair_count": 1,
                "calibration_score_name": "event_attestation_score",
                "evaluate_score_name": "event_attestation_score",
            }
        },
    )

    stage_02.run_stage_02(
        drive_project_root=drive_root,
        config_path=tmp_path / "config.yaml",
        source_package_path=source_package_path,
        notebook_name="02_Parallel_Attestation_Statistics",
        stage_run_id="stage02_direct_only",
    )

    runtime_cfg = yaml.safe_load((runtime_state_root / "runtime_metadata" / "runtime_config_snapshot.yaml").read_text(encoding="utf-8"))
    detect_glob = runtime_cfg["calibration"]["detect_records_glob"]
    assert "parallel_attestation_statistics_detect_records" in detect_glob
    assert "*detect*.json" not in detect_glob

    build_contract_path = run_root / "artifacts" / "parallel_attestation_statistics_build_contract.json"
    build_contract = json.loads(build_contract_path.read_text(encoding="utf-8"))
    assert build_contract["build_mode"] == "direct_source_only"
    assert build_contract["direct_record_count"] == 2
    assert build_contract["derived_record_count"] == 0
    assert build_contract["final_label_balanced"] is True

    stage_manifest = json.loads((run_root / "artifacts" / "stage_manifest.json").read_text(encoding="utf-8"))
    assert stage_manifest["parallel_attestation_statistics_direct_only"] is True
    assert stage_manifest["parallel_attestation_statistics_build_mode"] == "direct_source_only"
    assert stage_manifest["parallel_attestation_statistics_derived_record_count"] == 0
    assert (runtime_state_root / "package_staging" / "artifacts" / "parallel_attestation_statistics_build_contract.json").exists()


def test_stage_02_derived_build_records_direct_and_derived_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证当 source 不平衡但允许 derived build 时，stage 02 会显式构建 derived records 并写出 build contract。

    Validate that stage 02 explicitly builds derived records when the source is
    unbalanced and the build config allows source-plus-derived construction.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    stage_02 = _load_script_module("scripts/02_Parallel_Attestation_Statistics.py", "stage_02_derived_build")
    drive_root = tmp_path / "drive"
    source_package_path = tmp_path / "stage_01_package.zip"
    source_package_path.write_text("placeholder", encoding="utf-8")

    run_root = drive_root / "runs" / "stage_02"
    log_root = drive_root / "logs" / "stage_02"
    runtime_state_root = drive_root / "runtime_state" / "stage_02"
    export_root = drive_root / "exports" / "stage_02"
    extracted_root = tmp_path / "source_package_extract_derived"
    prompt_file = tmp_path / "paper_small_prompts.txt"
    prompt_file.write_text("a cat on a sofa\n", encoding="utf-8")

    positive_path = extracted_root / "records" / "detect_positive.json"
    _write_json(positive_path, _make_detect_record(True, 0.92))

    source_contract = _make_source_contract(
        [
            {
                "package_relative_path": "records/detect_positive.json",
                "sha256": stage_02.compute_file_sha256(positive_path),
                "label": True,
            }
        ],
        direct_stats_ready=False,
        direct_stats_reason="parallel_attestation_statistics_requires_label_balanced_detect_records",
    )
    _write_json(extracted_root / "artifacts" / "parallel_attestation_statistics_input_contract.json", source_contract)
    source_stage_manifest = {
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": "source_stage",
        "parallel_attestation_statistics_input_contract_package_relative_path": (
            "artifacts/parallel_attestation_statistics_input_contract.json"
        ),
    }
    _write_json(extracted_root / "artifacts" / "stage_manifest.json", source_stage_manifest)
    _write_json(extracted_root / "artifacts" / "thresholds" / "thresholds_artifact.json", {"threshold": 0.5})
    (extracted_root / "runtime_metadata").mkdir(parents=True, exist_ok=True)
    (extracted_root / "runtime_metadata" / "runtime_config_snapshot.yaml").write_text("calibration: {}\n", encoding="utf-8")

    _prepare_stage_02_monkeypatches(
        monkeypatch,
        stage_02,
        drive_root,
        run_root,
        log_root,
        runtime_state_root,
        export_root,
        extracted_root,
        source_stage_manifest,
        source_package_path,
        {
            "parallel_attestation_statistics": {
                "enabled": True,
                "build_mode": "source_plus_derived_pairs",
                "allow_derived_input_build": True,
                "derived_input_prompts_file": str(prompt_file),
                "target_pair_count": 1,
                "build_usage": "parallel_attestation_statistics",
                "record_derivation_kind": "prompt_bound_label_balance",
                "calibration_score_name": "event_attestation_score",
                "evaluate_score_name": "event_attestation_score",
            }
        },
    )

    stage_02.run_stage_02(
        drive_project_root=drive_root,
        config_path=tmp_path / "config.yaml",
        source_package_path=source_package_path,
        notebook_name="02_Parallel_Attestation_Statistics",
        stage_run_id="stage02_derived_build",
    )

    build_contract_path = run_root / "artifacts" / "parallel_attestation_statistics_build_contract.json"
    build_contract = json.loads(build_contract_path.read_text(encoding="utf-8"))
    assert build_contract["build_mode"] == "source_plus_derived_pairs"
    assert build_contract["direct_record_count"] == 1
    assert build_contract["derived_record_count"] == 1
    assert build_contract["final_label_balanced"] is True
    assert build_contract["direct_only"] is False

    derived_record = build_contract["derived_records"][0]
    assert derived_record["derived_from"] == "records/detect_positive.json"
    assert derived_record["derivation_kind"] == "prompt_bound_label_balance"
    assert derived_record["label"] is False
    assert derived_record["usage"] == "parallel_attestation_statistics"
    assert derived_record["prompt_file"] == str(prompt_file.resolve()).replace("\\", "/")

    derived_payload_path = Path(derived_record["staged_path"])
    derived_payload = json.loads(derived_payload_path.read_text(encoding="utf-8"))
    assert derived_payload["label"] is False
    assert derived_payload["parallel_attestation_statistics_build"]["derivation_kind"] == "prompt_bound_label_balance"
    assert derived_payload["attestation"]["final_event_attested_decision"]["event_attestation_score"] == 0.0

    runtime_cfg = yaml.safe_load((runtime_state_root / "runtime_metadata" / "runtime_config_snapshot.yaml").read_text(encoding="utf-8"))
    assert "parallel_attestation_statistics_detect_records" in runtime_cfg["calibration"]["detect_records_glob"]
    assert runtime_cfg["calibration"]["detect_records_glob"] == runtime_cfg["evaluate"]["detect_records_glob"]

    stage_manifest = json.loads((run_root / "artifacts" / "stage_manifest.json").read_text(encoding="utf-8"))
    assert stage_manifest["parallel_attestation_statistics_build_mode"] == "source_plus_derived_pairs"
    assert stage_manifest["parallel_attestation_statistics_derived_record_count"] == 1
    assert stage_manifest["parallel_attestation_statistics_direct_only"] is False


def test_formal_calibrate_and_evaluate_do_not_call_synthetic_helper_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：验证 formal calibrate/evaluate 默认不再调用 synthetic minimal ground-truth helper。

    Validate that formal calibrate/evaluate do not call the synthetic minimal
    ground-truth helper by default.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    def _fail_if_called(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("synthetic helper must not be called")

    monkeypatch.setattr(run_calibrate_cli, "ensure_minimal_ground_truth_records", _fail_if_called)
    monkeypatch.setattr(run_evaluate_cli, "ensure_minimal_ground_truth_records", _fail_if_called)

    calibration_result = run_calibrate_cli._ensure_calibration_detect_records_ready(
        {"calibration": {"detect_records_glob": "records/*.json", "allow_synthetic_minimal_ground_truth": False}},
        Path("run_root"),
    )
    evaluate_result = run_evaluate_cli._ensure_evaluate_detect_records_ready(
        {"evaluate": {"detect_records_glob": "records/*.json", "allow_synthetic_minimal_ground_truth": False}},
        Path("run_root"),
    )

    assert calibration_result["reason"] == "synthetic_minimal_ground_truth_disabled"
    assert evaluate_result["reason"] == "synthetic_minimal_ground_truth_disabled"


def test_metrics_and_detect_stats_reject_legacy_event_attestation_alias() -> None:
    """
    功能：验证 metrics 与 detect stats 读取端都拒绝 legacy event_attestation_statistics_score。

    Validate that both metrics and detect-statistics readers reject the legacy
    event_attestation_statistics_score alias.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(ValueError, match="requires rerun"):
        eval_metrics._extract_score_value_for_metrics({}, "event_attestation_statistics_score")

    with pytest.raises(ValueError, match="requires rerun"):
        detect_orchestrator._extract_score_for_stats({}, "event_attestation_statistics_score")


def test_verify_attestation_no_longer_writes_legacy_alias_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：验证 verify_attestation 输出的 final_event_attested_decision 不再写 legacy alias。

    Validate that verify_attestation no longer writes legacy alias fields into
    final_event_attested_decision.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import main.watermarking.provenance.attestation_statement as attestation_statement
    import main.watermarking.provenance.key_derivation as key_derivation
    import main.watermarking.content_chain.low_freq_coder as low_freq_coder
    import main.watermarking.content_chain.high_freq_embedder as high_freq_embedder

    monkeypatch.setattr(attestation_statement, "verify_statement_fields", lambda _statement: True)
    monkeypatch.setattr(
        attestation_statement,
        "statement_from_dict",
        lambda _statement: SimpleNamespace(plan_digest="plan_digest"),
    )
    monkeypatch.setattr(attestation_statement, "compute_attestation_digest", lambda _statement: "a" * 64)
    monkeypatch.setattr(
        attestation_statement,
        "verify_signed_attestation_bundle",
        lambda _bundle, _k_master: {"status": "ok"},
    )
    monkeypatch.setattr(
        key_derivation,
        "derive_attestation_keys",
        lambda *_args, **_kwargs: SimpleNamespace(k_lf="lf", k_hf="hf", event_binding_digest="b" * 64),
    )
    monkeypatch.setattr(
        low_freq_coder,
        "compute_lf_attestation_score",
        lambda **_kwargs: {"status": "ok", "lf_attestation_score": 0.9},
    )
    monkeypatch.setattr(
        high_freq_embedder,
        "compute_hf_attestation_score",
        lambda **_kwargs: {
            "status": "ok",
            "hf_attestation_score": 0.8,
            "hf_attestation_decision_score": 0.8,
            "hf_attestation_trace": {},
        },
    )

    verification = detect_orchestrator.verify_attestation(
        k_master="c" * 64,
        candidate_statement={"plan_digest": "plan_digest"},
        cfg={"attestation": {"use_trajectory_mix": False}},
        attestation_bundle={"trace_commit": "trace_commit"},
        hf_values=[0.1, 0.2, 0.3],
        lf_latent_features=[0.1, 0.2, 0.3],
        geo_score=0.4,
        attestation_decision_mode="content_primary_geo_rescue",
    )

    final_decision = verification["final_event_attested_decision"]
    assert final_decision["event_attestation_score_name"] == "event_attestation_score"
    assert "event_attestation_statistics_score" not in final_decision
    assert "event_attestation_statistics_score_name" not in final_decision
    assert "event_attestation_statistics_score_semantics" not in final_decision