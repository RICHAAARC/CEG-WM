"""
文件目的：验证 stage 01 outputs audit 的冻结规则与 wrapper-level gate 判定。
Module type: General module
"""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path
from typing import Any, Dict

from scripts.notebook_runtime_common import finalize_stage_package, write_json_atomic


REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_SCRIPT_PATH = REPO_ROOT / "scripts" / "01_audit_outputs.py"
CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH = "artifacts/stage_01_canonical_source_pool/source_pool_manifest.json"
CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT = "artifacts/stage_01_canonical_source_pool/entries"


def _load_stage_01_audit_module() -> Any:
    """
    功能：按文件路径加载 stage 01 audit 脚本模块。 

    Load the stage-01 audit script module from its filesystem path.

    Args:
        None.

    Returns:
        Loaded module object.
    """
    spec = importlib.util.spec_from_file_location("stage_01_audit_outputs_test", AUDIT_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {AUDIT_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_yaml_text(path_obj: Path, content: str) -> None:
    """
    功能：写入最小 YAML 文本文件。 

    Write a minimal YAML text file.

    Args:
        path_obj: Destination path.
        content: YAML text content.

    Returns:
        None.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(content, encoding="utf-8")


def _write_json_to_roots(relative_path: str, payload: Dict[str, Any], *roots: Path) -> None:
    """
    功能：向多个根目录写入同一份 JSON 文件。 

    Write one JSON payload to multiple roots using the same relative path.

    Args:
        relative_path: Relative file path.
        payload: JSON payload.
        roots: Destination roots.

    Returns:
        None.
    """
    for root in roots:
        write_json_atomic(root / relative_path, payload)


def _touch_file(path_obj: Path, content: str = "placeholder\n") -> None:
    """
    功能：写入最小文本文件。 

    Write a minimal text file.

    Args:
        path_obj: Destination path.
        content: Text content.

    Returns:
        None.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(content, encoding="utf-8")


def _build_stage_01_audit_fixture(
    base_dir: Path,
    *,
    include_root_records: bool = True,
    include_stage_02_contract: bool = True,
    attestation_evidence_status: str = "ok",
    include_diagnostics: bool = False,
) -> Dict[str, Path]:
    """
    功能：构造 stage 01 audit 所需的最小 formal outputs。 

    Build the minimal stage-01 formal outputs required by the audit script.

    Args:
        base_dir: Temporary base directory.
        include_root_records: Whether representative root records are emitted.
        include_stage_02_contract: Whether the stage-02 source contract is emitted.
        attestation_evidence_status: Attestation-evidence status token.
        include_diagnostics: Whether diagnostics artifacts are present.

    Returns:
        Mapping of key output paths.
    """
    run_root = base_dir / "run_root"
    package_root = base_dir / "package_root"
    export_root = base_dir / "exports"
    diagnostics_root = base_dir / "diagnostics"
    stage_run_id = "stage01_audit"

    representative_root_records = {
        "view_role": "representative_summary_view",
        "contract_mode": "strong_compatibility",
        "source_truth": "canonical_source_pool",
        "root_records_required": True,
        "root_embed_record_package_relative_path": "records/embed_record.json",
        "root_detect_record_package_relative_path": "records/detect_record.json",
        "source_prompt_index": 0,
        "source_prompt_sha256": "prompt_sha256_00",
        "source_entry_package_relative_path": f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/000_source_entry.json",
        "source_embed_record_package_relative_path": "artifacts/stage_01_source_pool_embed_records/000_embed_record.json",
        "source_detect_record_package_relative_path": "artifacts/stage_01_source_pool_detect_records/000_detect_record.json",
    }
    attestation_resolution = {
        "overall_status": attestation_evidence_status,
        "required_entry_count": 1,
        "checked_entry_count": 1 if attestation_evidence_status == "ok" else 0,
        "missing_evidence_count": 0 if attestation_evidence_status == "ok" else 1,
        "failing_prompt_indices": [] if attestation_evidence_status == "ok" else [0],
        "failing_source_entry_paths": [] if attestation_evidence_status == "ok" else [
            f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/000_source_entry.json"
        ],
        "summary_reason": "ok" if attestation_evidence_status == "ok" else "attestation_evidence_failed",
        "failure_reason": None if attestation_evidence_status == "ok" else "attestation_evidence_failed",
        "representative_root_summary": {
            **representative_root_records,
            "resolution_role": "representative_summary_view_only",
        },
    }

    diagnostics_summary_path = diagnostics_root / "failure_diagnostics_summary.json"
    diagnostics_manifest_path = diagnostics_root / "failure_diagnostics_manifest.json"
    diagnostics_package_path = diagnostics_root / "failure_diagnostics.zip"
    diagnostics_status = "generated" if include_diagnostics else "not_generated"
    if include_diagnostics:
        write_json_atomic(diagnostics_summary_path, {"stage_name": "01_Paper_Full_Cuda", "status": "failed"})
        write_json_atomic(diagnostics_manifest_path, {"artifact_type": "failure_diagnostics_manifest"})
        diagnostics_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(diagnostics_package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("failure_diagnostics_summary.json", json.dumps({"status": "failed"}, ensure_ascii=False))

    stage_manifest = {
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": stage_run_id,
        "stage_status": "ok",
        "workflow_summary_status": "ok",
        "workflow_summary_path": (run_root / "artifacts" / "workflow_summary.json").as_posix(),
        "formal_package_status": "generated",
        "formal_package_role": "formal_stage_package",
        "formal_package_discovery_scope": "discoverable_formal_only",
        "runtime_config_snapshot_path": (run_root / "runtime_metadata" / "runtime_config_snapshot.yaml").as_posix(),
        "prompt_snapshot_path": (run_root / "runtime_metadata" / "prompt_snapshot" / "prompt.txt").as_posix(),
        "thresholds_path": (run_root / "artifacts" / "thresholds" / "thresholds_artifact.json").as_posix(),
        "threshold_metadata_artifact_path": (run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json").as_posix(),
        "evaluation_report_path": (run_root / "artifacts" / "evaluation_report.json").as_posix(),
        "run_closure_path": (run_root / "artifacts" / "run_closure.json").as_posix(),
        "stage_01_canonical_source_pool_manifest_path": (run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH).as_posix(),
        "stage_01_canonical_source_pool_manifest_package_relative_path": CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH,
        "stage_01_canonical_source_pool_entry_count": 1,
        "stage_01_root_contract_mode": "strong_compatibility",
        "stage_01_root_records_required": True,
        "stage_01_source_truth": "canonical_source_pool",
        "stage_01_representative_root_role": "representative_summary_view",
        "stage_01_representative_root_records": representative_root_records,
        "attestation_evidence_status": attestation_evidence_status,
        "attestation_evidence_summary_reason": attestation_resolution["summary_reason"],
        "attestation_evidence_failure_reason": attestation_resolution["failure_reason"],
        "parallel_attestation_statistics_input_contract_package_relative_path": "artifacts/parallel_attestation_statistics_input_contract.json",
        "diagnostics_status": diagnostics_status,
        "diagnostics_summary_path": diagnostics_summary_path.as_posix() if include_diagnostics else "<absent>",
        "diagnostics_manifest_path": diagnostics_manifest_path.as_posix() if include_diagnostics else "<absent>",
        "diagnostics_package_path": diagnostics_package_path.as_posix() if include_diagnostics else "<absent>",
    }
    workflow_summary = {
        "stage_name": "01_Paper_Full_Cuda_mainline",
        "stage_run_id": stage_run_id,
        "status": "ok",
        "summary_reason": "ok" if attestation_evidence_status == "ok" else "failed_attestation_evidence",
        "failure_reason": None if attestation_evidence_status == "ok" else "failed_attestation_evidence",
        "canonical_source_pool_entry_count": 1,
        "representative_root_records": representative_root_records,
        "attestation_evidence_resolution": attestation_resolution,
        "required_artifacts_ok": include_root_records and include_stage_02_contract,
    }
    canonical_source_entry = {
        "artifact_type": "stage_01_canonical_source_entry",
        "entry_role": "canonical_source_entry",
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": stage_run_id,
        "path": (run_root / CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT / "000_source_entry.json").as_posix(),
        "source_entry_package_relative_path": f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/000_source_entry.json",
        "prompt_index": 0,
        "prompt_text": "prompt 0",
        "prompt_sha256": "prompt_sha256_00",
        "prompt_file": "prompts/paper_small.txt",
        "record_usage": "stage_01_direct_source_pool",
        "label": True,
        "content_chain_score_available": True,
        "event_attestation_score_available": True,
        "prompt_run_root": (run_root / "source_pool" / "000_prompt").as_posix(),
        "runtime_config_path": (run_root / "runtime_metadata" / "runtime_config_snapshot.yaml").as_posix(),
        "runtime_config_package_relative_path": "runtime_metadata/runtime_config_snapshot.yaml",
        "embed_record_path": (run_root / "artifacts" / "stage_01_source_pool_embed_records" / "000_embed_record.json").as_posix(),
        "embed_record_package_relative_path": "artifacts/stage_01_source_pool_embed_records/000_embed_record.json",
        "detect_record_path": (run_root / "artifacts" / "stage_01_source_pool_detect_records" / "000_detect_record.json").as_posix(),
        "detect_record_package_relative_path": "artifacts/stage_01_source_pool_detect_records/000_detect_record.json",
        "attestation_statement": {
            "exists": True,
            "path": (run_root / "artifacts" / "stage_01_canonical_source_pool" / "attestation" / "000_statement.json").as_posix(),
            "package_relative_path": "artifacts/stage_01_canonical_source_pool/attestation/000_statement.json",
        },
        "attestation_bundle": {
            "exists": True,
            "path": (run_root / "artifacts" / "stage_01_canonical_source_pool" / "attestation" / "000_bundle.json").as_posix(),
            "package_relative_path": "artifacts/stage_01_canonical_source_pool/attestation/000_bundle.json",
        },
        "attestation_result": {
            "exists": True,
            "path": (run_root / "artifacts" / "stage_01_canonical_source_pool" / "attestation" / "000_result.json").as_posix(),
            "package_relative_path": "artifacts/stage_01_canonical_source_pool/attestation/000_result.json",
        },
        "source_image": {
            "exists": True,
            "path": (run_root / "artifacts" / "stage_01_canonical_source_pool" / "source_images" / "000.png").as_posix(),
            "package_relative_path": "artifacts/stage_01_canonical_source_pool/source_images/000.png",
        },
    }
    canonical_source_pool_manifest = {
        "artifact_type": "stage_01_canonical_source_pool",
        "artifact_role": "canonical_source_pool_root",
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": stage_run_id,
        "source_truth": "canonical_source_pool",
        "root_contract_mode": "strong_compatibility",
        "root_records_required": True,
        "representative_root_role": "representative_summary_view",
        "prompt_file": "prompts/paper_small.txt",
        "canonical_source_pool_root_path": (run_root / "artifacts" / "stage_01_canonical_source_pool").as_posix(),
        "canonical_source_pool_root_package_relative_path": "artifacts/stage_01_canonical_source_pool",
        "manifest_path": (run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH).as_posix(),
        "manifest_package_relative_path": CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH,
        "entries_root_path": (run_root / CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT).as_posix(),
        "entries_package_relative_root": CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT,
        "prompt_count": 1,
        "entry_count": 1,
        "entries": [
            {
                "prompt_index": 0,
                "prompt_text": "prompt 0",
                "prompt_sha256": "prompt_sha256_00",
                "source_entry_package_relative_path": f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/000_source_entry.json",
                "detect_record_path": canonical_source_entry["detect_record_path"],
                "detect_record_package_relative_path": canonical_source_entry["detect_record_package_relative_path"],
                "embed_record_path": canonical_source_entry["embed_record_path"],
                "embed_record_package_relative_path": canonical_source_entry["embed_record_package_relative_path"],
                "runtime_config_path": canonical_source_entry["runtime_config_path"],
                "runtime_config_package_relative_path": canonical_source_entry["runtime_config_package_relative_path"],
                "attestation_statement": canonical_source_entry["attestation_statement"],
                "attestation_bundle": canonical_source_entry["attestation_bundle"],
                "attestation_result": canonical_source_entry["attestation_result"],
                "source_image": canonical_source_entry["source_image"],
                "representative_root_records_alias": True,
            }
        ],
        "representative_root_records": representative_root_records,
    }
    source_contract = {
        "artifact_type": "parallel_attestation_statistics_input_contract",
        "contract_role": "source_contract",
        "contract_version": "v1",
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": stage_run_id,
        "status": "ok",
        "reason": "stage_01_direct_source_pool_ready",
        "source_authority": "canonical_source_pool",
        "contract_view_role": "stage_02_compatibility_view",
        "canonical_source_pool_manifest_package_relative_path": CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH,
        "canonical_source_pool_entries_package_relative_root": CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT,
        "representative_root_records": representative_root_records,
        "score_name": "event_attestation_score",
        "threshold_score_name": "content_chain_score",
        "source_records_available": True,
        "record_count": 1,
        "label_summary": {"positive": 1, "negative": 0, "unknown": 0, "label_balanced": False},
        "score_availability": {
            "content_chain_score": {"available_record_count": 1, "missing_record_count": 0},
            "event_attestation_score": {"available_record_count": 1, "missing_record_count": 0},
        },
        "direct_stats_ready": False,
        "direct_stats_reason": "parallel_attestation_statistics_requires_label_balanced_detect_records",
        "records": [
            {
                "record_role": "direct_source_record",
                "usage": "stage_01_direct_source_pool",
                "package_relative_path": "artifacts/stage_01_source_pool_detect_records/000_detect_record.json",
                "path": canonical_source_entry["detect_record_path"],
                "sha256": "detect_sha256_00",
                "label": True,
                "prompt_index": 0,
                "prompt_text": "prompt 0",
                "prompt_sha256": "prompt_sha256_00",
                "prompt_file": "prompts/paper_small.txt",
                "canonical_source_entry_package_relative_path": f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/000_source_entry.json",
                "embed_record_package_relative_path": "artifacts/stage_01_source_pool_embed_records/000_embed_record.json",
                "runtime_config_package_relative_path": "runtime_metadata/runtime_config_snapshot.yaml",
                "score_name": "event_attestation_score",
                "score_available": True,
                "threshold_score_name": "content_chain_score",
                "threshold_score_available": True,
            }
        ],
    }

    json_payloads = {
        "artifacts/stage_manifest.json": stage_manifest,
        "artifacts/workflow_summary.json": workflow_summary,
        CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH: canonical_source_pool_manifest,
        f"{CANONICAL_SOURCE_POOL_ENTRIES_RELATIVE_ROOT}/000_source_entry.json": canonical_source_entry,
        "artifacts/stage_01_source_pool_embed_records/000_embed_record.json": {"status": "ok"},
        "artifacts/stage_01_source_pool_detect_records/000_detect_record.json": {"status": "ok", "label": True},
        "artifacts/stage_01_canonical_source_pool/attestation/000_statement.json": {"status": "ok"},
        "artifacts/stage_01_canonical_source_pool/attestation/000_bundle.json": {"status": "ok"},
        "artifacts/stage_01_canonical_source_pool/attestation/000_result.json": {"status": "ok"},
        "artifacts/thresholds/thresholds_artifact.json": {"threshold_value": 0.5, "thresholds_digest": "thr01"},
        "artifacts/thresholds/threshold_metadata_artifact.json": {"threshold_metadata_digest": "meta01"},
        "artifacts/evaluation_report.json": {"status": "ok", "policy_path": "content_np_geo_rescue"},
        "artifacts/run_closure.json": {"status": "ok"},
    }
    if include_root_records:
        json_payloads["records/embed_record.json"] = {"record_type": "embed"}
        json_payloads["records/detect_record.json"] = {"record_type": "detect"}
    json_payloads["records/calibration_record.json"] = {"record_type": "calibration"}
    json_payloads["records/evaluate_record.json"] = {"record_type": "evaluate"}
    if include_stage_02_contract:
        json_payloads["artifacts/parallel_attestation_statistics_input_contract.json"] = source_contract

    for relative_path, payload in json_payloads.items():
        _write_json_to_roots(relative_path, payload, run_root, package_root)

    _write_yaml_text(run_root / "runtime_metadata" / "runtime_config_snapshot.yaml", "policy_path: content_np_geo_rescue\n")
    _write_yaml_text(package_root / "runtime_metadata" / "runtime_config_snapshot.yaml", "policy_path: content_np_geo_rescue\n")
    _touch_file(run_root / "runtime_metadata" / "prompt_snapshot" / "prompt.txt", "prompt 0\n")
    _touch_file(package_root / "runtime_metadata" / "prompt_snapshot" / "prompt.txt", "prompt 0\n")
    _touch_file(run_root / "artifacts" / "stage_01_canonical_source_pool" / "source_images" / "000.png", "png\n")
    _touch_file(package_root / "artifacts" / "stage_01_canonical_source_pool" / "source_images" / "000.png", "png\n")

    package_manifest_path = run_root / "artifacts" / "package_manifest.json"
    package_manifest = finalize_stage_package(
        stage_name="01_Paper_Full_Cuda",
        stage_run_id=stage_run_id,
        package_root=package_root,
        export_root=export_root,
        source_stage_run_id=None,
        source_stage_package_path=None,
        package_manifest_path=package_manifest_path,
    )
    return {
        "run_root": run_root,
        "stage_manifest_path": run_root / "artifacts" / "stage_manifest.json",
        "workflow_summary_path": run_root / "artifacts" / "workflow_summary.json",
        "package_manifest_path": package_manifest_path,
        "package_path": Path(str(package_manifest["package_path"])),
        "audit_output_path": run_root / "artifacts" / "stage_01_audit_summary.json",
    }


def test_strong_compatibility_contract_passes(tmp_path: Path) -> None:
    """
    功能：验证 strong_compatibility contract 正常满足时 audit 通过。 

    Verify that the audit passes when the strong-compatibility contract and
    downstream readiness checks all succeed.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_01_audit_module()
    fixture = _build_stage_01_audit_fixture(tmp_path / "case_pass")

    summary = module.run_stage_01_output_audit(
        run_root=fixture["run_root"],
        stage_manifest_path=fixture["stage_manifest_path"],
        workflow_summary_path=fixture["workflow_summary_path"],
        package_manifest_path=fixture["package_manifest_path"],
        output_path=fixture["audit_output_path"],
    )

    assert summary["overall_status"] == "passed"
    assert summary["definition_status"] == "passed"
    assert summary["strong_compatibility_status"] == "passed"
    assert summary["stage_02_ready"] is True
    assert summary["stage_03_ready"] is True
    assert summary["stage_04_ready"] is True
    assert summary["formal_package_policy_status"] == "passed"
    assert summary["canonical_source_pool_status"] == "passed"
    assert summary["representative_root_status"] == "passed"
    assert summary["blocking_reasons"] == []


def test_root_records_missing_hard_fail_even_when_canonical_source_pool_exists(tmp_path: Path) -> None:
    """
    功能：验证 root records 缺失时必须 hard fail。 

    Verify that missing root records cause a hard failure even when the
    canonical source pool remains complete.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_01_audit_module()
    fixture = _build_stage_01_audit_fixture(tmp_path / "case_root_missing", include_root_records=False)

    summary = module.run_stage_01_output_audit(
        run_root=fixture["run_root"],
        stage_manifest_path=fixture["stage_manifest_path"],
        workflow_summary_path=fixture["workflow_summary_path"],
        package_manifest_path=fixture["package_manifest_path"],
        output_path=fixture["audit_output_path"],
    )

    assert summary["overall_status"] == "blocked"
    assert summary["strong_compatibility_status"] == "blocked"
    assert any(reason.startswith("strong_compatibility.") for reason in summary["blocking_reasons"])


def test_stage_02_contract_missing_blocks_readiness(tmp_path: Path) -> None:
    """
    功能：验证 stage 02 input contract 缺失时 stage_02_ready 为 false。 

    Verify that a missing stage-02 source contract blocks overall readiness.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_01_audit_module()
    fixture = _build_stage_01_audit_fixture(tmp_path / "case_stage02_missing", include_stage_02_contract=False)

    summary = module.run_stage_01_output_audit(
        run_root=fixture["run_root"],
        stage_manifest_path=fixture["stage_manifest_path"],
        workflow_summary_path=fixture["workflow_summary_path"],
        package_manifest_path=fixture["package_manifest_path"],
        output_path=fixture["audit_output_path"],
    )

    assert summary["stage_02_ready"] is False
    assert summary["overall_status"] == "blocked"
    assert any(reason.startswith("stage_02.") for reason in summary["blocking_reasons"])


def test_attestation_failure_blocks_stage_04_readiness(tmp_path: Path) -> None:
    """
    功能：验证 attestation_evidence_status 非成功态时 stage_04_ready 必须失败。 

    Verify that a failed attestation-evidence status blocks stage-04 readiness.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_01_audit_module()
    fixture = _build_stage_01_audit_fixture(
        tmp_path / "case_attestation_failed",
        attestation_evidence_status="failed",
    )

    summary = module.run_stage_01_output_audit(
        run_root=fixture["run_root"],
        stage_manifest_path=fixture["stage_manifest_path"],
        workflow_summary_path=fixture["workflow_summary_path"],
        package_manifest_path=fixture["package_manifest_path"],
        output_path=fixture["audit_output_path"],
    )

    assert summary["attestation_evidence_status"] == "failed"
    assert summary["stage_04_ready"] is False
    assert summary["overall_status"] == "blocked"
    assert "stage_04.attestation_evidence_status_not_success" in summary["blocking_reasons"]


def test_diagnostics_presence_does_not_break_formal_package_policy(tmp_path: Path) -> None:
    """
    功能：验证 diagnostics 存在不会破坏 formal package readiness。 

    Verify that diagnostics presence does not invalidate the formal package
    policy or overall readiness when the formal package itself is valid.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_01_audit_module()
    fixture = _build_stage_01_audit_fixture(
        tmp_path / "case_diagnostics_present",
        include_diagnostics=True,
    )

    summary = module.run_stage_01_output_audit(
        run_root=fixture["run_root"],
        stage_manifest_path=fixture["stage_manifest_path"],
        workflow_summary_path=fixture["workflow_summary_path"],
        package_manifest_path=fixture["package_manifest_path"],
        output_path=fixture["audit_output_path"],
    )

    assert summary["formal_package_policy_status"] == "passed"
    assert summary["overall_status"] == "passed"
    assert "diagnostics_package_present_but_not_formal_input" in summary["warnings"]