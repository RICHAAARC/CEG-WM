"""
文件目的：验证 00_main notebook 的最小运行合同与 stage package 解析。
Module type: General module

职责边界：
1. 覆盖 01 notebook 的 validation contract，以及 03 / 04 notebook 的 package 解析与执行绑定。
2. 通过直接解析 notebook JSON 与执行关键 code cell，验证 notebook 层语义，不修改 scripts 层 formal contract。
3. 不重跑主链，不触发真实 Colab 或 Google Drive 依赖。
"""

from __future__ import annotations

import json
import os
import subprocess
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from scripts.notebook_runtime_common import (
    build_repo_import_subprocess_env,
    build_failure_diagnostics_filename,
    compute_file_sha256,
    finalize_stage_package,
    resolve_export_package_manifest_path,
    resolve_stage_package_input_or_discover,
    write_json_atomic,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_01_PATH = REPO_ROOT / "notebook" / "00_main" / "01_Paper_Full_Cuda.ipynb"
NOTEBOOK_02_PATH = REPO_ROOT / "notebook" / "00_main" / "02_Parallel_Attestation_Statistics.ipynb"
NOTEBOOK_03_PATH = REPO_ROOT / "notebook" / "00_main" / "03_Experiment_Matrix_Full.ipynb"
NOTEBOOK_04_PATH = REPO_ROOT / "notebook" / "00_main" / "04_Release_And_Signoff.ipynb"


def _load_notebook(notebook_path: Path) -> Dict[str, Any]:
    """
    功能：读取 notebook JSON 文档。 

    Load one notebook JSON document.

    Args:
        notebook_path: Notebook path.

    Returns:
        Parsed notebook mapping.
    """
    notebook_obj = json.loads(notebook_path.read_text(encoding="utf-8"))
    if not isinstance(notebook_obj, dict):
        raise AssertionError(f"notebook root must be dict: {notebook_path}")
    return notebook_obj


def _find_code_cell_source(notebook_path: Path, marker: str) -> str:
    """
    功能：按标记文本定位 notebook code cell。 

    Locate one code-cell source block by a stable marker string.

    Args:
        notebook_path: Notebook path.
        marker: Marker text expected inside one code cell.

    Returns:
        Joined code-cell source string.
    """
    if not isinstance(marker, str) or not marker:
        raise TypeError("marker must be non-empty str")

    notebook_obj = _load_notebook(notebook_path)
    cells = notebook_obj.get("cells")
    if not isinstance(cells, list):
        raise AssertionError(f"cells must be list: {notebook_path}")
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if not isinstance(source, list):
            continue
        source_text = "\n".join(str(line) for line in source)
        if marker in source_text:
            return source_text
    raise AssertionError(f"code cell marker not found: {marker}")


def _assert_execute_source_uses_repo_import_context(execute_source: str) -> None:
    """
    功能：断言 notebook execute cell 显式补齐 repo import context。

    Assert one notebook execute cell constructs a subprocess environment that
    preserves the current environment while prepending the repository root to
    PYTHONPATH.

    Args:
        execute_source: Joined notebook code-cell source.

    Returns:
        None.
    """
    if not isinstance(execute_source, str) or not execute_source:
        raise TypeError("execute_source must be non-empty str")

    assert "build_repo_import_subprocess_env" in execute_source
    assert "NOTEBOOK_SUBPROCESS_ENV = build_repo_import_subprocess_env(repo_root=REPO_ROOT)" in execute_source
    assert "env=NOTEBOOK_SUBPROCESS_ENV" in execute_source
    assert "env=os.environ.copy()" not in execute_source


def _create_formal_stage_package(
    base_dir: Path,
    *,
    stage_name: str,
    stage_run_id: str,
    extra_files: Dict[str, Any] | None = None,
) -> Path:
    """
    功能：构造最小可发现 formal stage package。 

    Create one minimal discoverable formal stage package for notebook tests.

    Args:
        base_dir: Base test directory.
        stage_name: Stage name.
        stage_run_id: Stage run identifier.
        extra_files: Optional extra packaged files keyed by relative path.

    Returns:
        Formal package ZIP path.
    """
    package_root = base_dir / "package_root" / stage_name / stage_run_id
    export_root = base_dir / "exports" / stage_name / stage_run_id
    run_root = base_dir / "run_root" / stage_name / stage_run_id
    package_root.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    write_json_atomic(
        package_root / "artifacts" / "stage_manifest.json",
        {
            "stage_name": stage_name,
            "stage_run_id": stage_run_id,
            "stage_status": "ok",
            "workflow_summary_path": f"/drive/runs/{stage_name}/{stage_run_id}/artifacts/workflow_summary.json",
        },
    )
    write_json_atomic(
        package_root / "artifacts" / "workflow_summary.json",
        {
            "stage_name": stage_name,
            "stage_run_id": stage_run_id,
            "status": "ok",
        },
    )
    for relative_path, payload in (extra_files or {}).items():
        target_path = package_root / relative_path
        if isinstance(payload, dict):
            write_json_atomic(target_path, payload)
        elif isinstance(payload, str):
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(payload, encoding="utf-8")
        else:
            raise TypeError("extra_files payload must be dict or str")

    package_manifest = finalize_stage_package(
        stage_name=stage_name,
        stage_run_id=stage_run_id,
        package_root=package_root,
        export_root=export_root,
        source_stage_run_id=None,
        source_stage_package_path=None,
        package_manifest_path=run_root / "artifacts" / "package_manifest.json",
    )
    return Path(str(package_manifest["package_path"]))


def _create_failure_diagnostics_package(base_dir: Path, *, stage_name: str, stage_run_id: str) -> Path:
    """
    功能：构造最小 diagnostics package。 

    Create one minimal diagnostics package that must stay excluded from formal
    discovery.

    Args:
        base_dir: Base test directory.
        stage_name: Stage name.
        stage_run_id: Stage run identifier.

    Returns:
        Diagnostics package ZIP path.
    """
    export_root = base_dir / "exports" / stage_name / stage_run_id
    export_root.mkdir(parents=True, exist_ok=True)
    package_path = export_root / build_failure_diagnostics_filename(stage_name, stage_run_id)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "failure_diagnostics_summary.json",
            json.dumps(
                {
                    "stage_name": stage_name,
                    "stage_run_id": stage_run_id,
                    "stage_status": "failed",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        archive.writestr(
            "artifacts/stage_manifest.json",
            json.dumps(
                {
                    "stage_name": stage_name,
                    "stage_run_id": stage_run_id,
                    "stage_status": "failed",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    write_json_atomic(
        resolve_export_package_manifest_path(export_root),
        {
            "stage_name": stage_name,
            "stage_run_id": stage_run_id,
            "package_filename": package_path.name,
            "package_path": package_path.as_posix(),
            "package_sha256": compute_file_sha256(package_path),
            "package_role": "failure_diagnostics_package",
            "package_discovery_scope": "excluded_from_formal_discovery",
            "diagnostics_package_path": package_path.as_posix(),
        },
    )
    return package_path


def _execute_stage_04_precheck_cell(
    tmp_path: Path,
    *,
    drive_project_root: Path,
    stage_01_package_path: str | None,
    stage_02_package_path: str | None,
    stage_03_package_path: str | None,
    require_stage_02: bool,
    require_stage_03: bool,
) -> Dict[str, Any]:
    """
    功能：在测试命名空间中执行 stage 04 的 precheck code cell。 

    Execute the stage-04 notebook precheck code cell inside a test namespace.

    Args:
        tmp_path: Temporary pytest directory.
        drive_project_root: Fake drive project root.
        stage_01_package_path: Optional manual stage-01 package path.
        stage_02_package_path: Optional manual stage-02 package path.
        stage_03_package_path: Optional manual stage-03 package path.
        require_stage_02: Whether stage 02 is required.
        require_stage_03: Whether stage 03 is required.

    Returns:
        Execution namespace populated by the cell.
    """
    config_path = tmp_path / "default.yaml"
    script_path = tmp_path / "04_Release_And_Signoff.py"
    config_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")
    script_path.write_text("pass\n", encoding="utf-8")

    precheck_source = _find_code_cell_source(NOTEBOOK_04_PATH, "PRECHECK_RESULTS: list[dict[str, Any]] = []")
    namespace: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "Any": Any,
        "Path": Path,
        "json": json,
        "CONFIG_PATH": config_path,
        "SCRIPT_PATH": script_path,
        "DRIVE_PROJECT_ROOT": drive_project_root,
        "STAGE_01_PACKAGE_PATH": stage_01_package_path,
        "STAGE_02_PACKAGE_PATH": stage_02_package_path,
        "STAGE_03_PACKAGE_PATH": stage_03_package_path,
        "REQUIRE_STAGE_02": require_stage_02,
        "REQUIRE_STAGE_03": require_stage_03,
        "print_json": lambda *_args, **_kwargs: None,
    }
    exec(precheck_source, namespace)
    return namespace


def test_build_repo_import_subprocess_env_preserves_existing_entries() -> None:
    """
    功能：验证 repo import helper 保留原环境并前置仓库根目录。

    Verify the shared notebook subprocess helper preserves existing environment
    entries while prepending the repository root to PYTHONPATH.

    Args:
        None.

    Returns:
        None.
    """
    existing_pythonpath = os.pathsep.join(["/tmp/site-packages", "/tmp/custom"]) 
    source_env = {
        "PATH": "test_path_value",
        "CUSTOM_FLAG": "enabled",
        "PYTHONPATH": existing_pythonpath,
    }

    env_mapping = build_repo_import_subprocess_env(base_env=source_env, repo_root=REPO_ROOT)
    pythonpath_entries = env_mapping["PYTHONPATH"].split(os.pathsep)

    assert env_mapping["PATH"] == "test_path_value"
    assert env_mapping["CUSTOM_FLAG"] == "enabled"
    assert pythonpath_entries[0] == str(REPO_ROOT.resolve())
    assert pythonpath_entries[1:] == ["/tmp/site-packages", "/tmp/custom"]
    assert source_env["PYTHONPATH"] == existing_pythonpath


def test_build_repo_import_subprocess_env_deduplicates_repo_root() -> None:
    """
    功能：验证 repo import helper 不会重复追加仓库根目录。

    Verify the shared notebook subprocess helper keeps exactly one repository
    root entry even when the incoming PYTHONPATH already contains duplicates.

    Args:
        None.

    Returns:
        None.
    """
    repo_root_text = str(REPO_ROOT.resolve())
    source_env = {
        "PYTHONPATH": os.pathsep.join([repo_root_text, ".", repo_root_text, "relative/module/path"]),
    }

    env_mapping = build_repo_import_subprocess_env(base_env=source_env, repo_root=REPO_ROOT)
    pythonpath_entries = env_mapping["PYTHONPATH"].split(os.pathsep)

    assert pythonpath_entries[0] == repo_root_text
    assert pythonpath_entries.count(repo_root_text) == 1
    assert pythonpath_entries[1:] == [".", "relative/module/path"]


def _execute_stage_02_precheck_cell(
    tmp_path: Path,
    *,
    drive_project_root: Path,
    source_package_path: str | None,
) -> Dict[str, Any]:
    """
    功能：在测试命名空间中执行 stage 02 的 precheck code cell。

    Execute the stage-02 notebook precheck code cell inside a test namespace.

    Args:
        tmp_path: Temporary pytest directory.
        drive_project_root: Fake drive project root.
        source_package_path: Optional manual stage-01 package path.

    Returns:
        Execution namespace populated by the cell.
    """
    config_path = tmp_path / "default.yaml"
    script_path = tmp_path / "02_Parallel_Attestation_Statistics.py"
    config_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")
    script_path.write_text("pass\n", encoding="utf-8")

    precheck_source = _find_code_cell_source(NOTEBOOK_02_PATH, "PRECHECK_RESULTS = []")
    namespace: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "Path": Path,
        "json": json,
        "shutil": shutil,
        "CONFIG_PATH": config_path,
        "SCRIPT_PATH": script_path,
        "DRIVE_PROJECT_ROOT": drive_project_root,
        "SOURCE_PACKAGE_PATH": source_package_path,
        "REPO_ROOT": tmp_path,
        "print_json": lambda *_args, **_kwargs: None,
    }
    exec(precheck_source, namespace)
    return namespace


def _execute_stage_03_precheck_cell(
    tmp_path: Path,
    *,
    drive_project_root: Path,
    source_package_path: str | None,
) -> Dict[str, Any]:
    """
    功能：在测试命名空间中执行 stage 03 的 precheck code cell。

    Execute the stage-03 notebook precheck code cell inside a test namespace.

    Args:
        tmp_path: Temporary pytest directory.
        drive_project_root: Fake drive project root.
        source_package_path: Optional manual stage-01 package path.

    Returns:
        Execution namespace populated by the cell.
    """
    config_path = tmp_path / "default.yaml"
    script_path = tmp_path / "03_Experiment_Matrix_Full.py"
    config_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")
    script_path.write_text("pass\n", encoding="utf-8")

    fake_model_info = type("FakeModelInfo", (), {"id": "test-model"})
    fake_hf_api = type("FakeHfApi", (), {"model_info": lambda self, _repo_id: fake_model_info()})
    precheck_source = _find_code_cell_source(NOTEBOOK_03_PATH, "PRECHECK_RESULTS = []")
    fake_nvidia_smi = tmp_path / "nvidia-smi"
    fake_nvidia_smi.write_text("", encoding="utf-8")
    import scripts.workflow_acceptance_common as workflow_acceptance_common

    original_which = workflow_acceptance_common.shutil.which
    workflow_acceptance_common.shutil.which = lambda _command: str(fake_nvidia_smi)
    namespace: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "Path": Path,
        "json": json,
        "shutil": shutil,
        "CONFIG_PATH": config_path,
        "SCRIPT_PATH": script_path,
        "DRIVE_PROJECT_ROOT": drive_project_root,
        "SOURCE_PACKAGE_PATH": source_package_path,
        "REPO_ROOT": tmp_path,
        "HfApi": fake_hf_api,
        "MODEL_IDENTITY": {"model_id": "test-model"},
        "WEIGHT_PATH": tmp_path / "weights" / "ckpt_base.pth",
        "PERSISTENT_WEIGHT_PATH": tmp_path / "cache" / "ckpt_base.pth",
        "is_valid_weight_file": lambda _path_obj: True,
        "print_json": lambda *_args, **_kwargs: None,
    }
    try:
        exec(precheck_source, namespace)
    finally:
        workflow_acceptance_common.shutil.which = original_which
    return namespace


def _write_stage_01_precheck_config(config_path: Path) -> Path:
    """
    功能：写出 stage 01 precheck 所需的最小配置。 

    Write the minimal config consumed by the stage-01 notebook precheck tests.

    Args:
        config_path: Destination config path.

    Returns:
        Written config path.
    """
    cfg_obj = {
        "policy_path": "content_np_geo_rescue",
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "model_source": "hf",
        "hf_revision": "main",
        "inference_prompt_file": "prompts/paper_small.txt",
        "attestation": {
            "enabled": True,
            "k_master_env_var": "CEG_WM_K_MASTER",
            "k_prompt_env_var": "CEG_WM_K_PROMPT",
            "k_seed_env_var": "CEG_WM_K_SEED",
        },
        "stage_01_source_pool": {
            "enabled": True,
            "use_inference_prompt_file": True,
        },
        "stage_01_pooled_threshold_build": {
            "enabled": True,
            "target_pair_count": 16,
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg_obj, sort_keys=False), encoding="utf-8")
    return config_path


def _execute_stage_01_precheck_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_snapshot_path: Path,
    allow_failure: bool = False,
) -> Dict[str, Any]:
    """
    功能：在测试命名空间中执行 stage 01 的 precheck code cell。 

    Execute the stage-01 notebook precheck code cell inside a test namespace.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.
        model_snapshot_path: Notebook-visible model snapshot path.
        allow_failure: Whether RuntimeError raised by notebook hard-fail should
            be captured instead of re-raised.

    Returns:
        Execution namespace populated by the cell.
    """
    config_path = _write_stage_01_precheck_config(tmp_path / "default.yaml")
    script_path = tmp_path / "01_Paper_Full_Cuda.py"
    script_path.write_text("pass\n", encoding="utf-8")
    prompt_source_path = tmp_path / "prompts" / "paper_small.txt"
    prompt_source_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_source_path.write_text("prompt 0\n", encoding="utf-8")
    drive_project_root = tmp_path / "drive_project_root"
    drive_project_root.mkdir(parents=True, exist_ok=True)

    precheck_source = _find_code_cell_source(NOTEBOOK_01_PATH, "PRECHECK_RESULTS = []")
    fake_model_info = type("FakeModelInfo", (), {"id": "stabilityai/stable-diffusion-3.5-medium"})
    fake_hf_api = type("FakeHfApi", (), {"model_info": lambda self, _repo_id: fake_model_info()})
    fake_nvidia_smi = tmp_path / "nvidia-smi"
    fake_nvidia_smi.write_text("", encoding="utf-8")
    import scripts.workflow_acceptance_common as workflow_acceptance_common

    monkeypatch.setattr(workflow_acceptance_common.shutil, "which", lambda _command: str(fake_nvidia_smi))
    monkeypatch.setenv("CEG_WM_K_MASTER", "a" * 64)
    monkeypatch.setenv("CEG_WM_K_PROMPT", "b" * 32)
    monkeypatch.setenv("CEG_WM_K_SEED", "c" * 32)

    captured_json: Dict[str, Any] = {}
    namespace: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "json": json,
        "os": os,
        "shutil": shutil,
        "CONFIG_PATH": config_path,
        "SCRIPT_PATH": script_path,
        "PROMPT_SOURCE_PATH": prompt_source_path,
        "DRIVE_PROJECT_ROOT": drive_project_root,
        "NOTEBOOK_NAME": "01_Paper_Full_Cuda",
        "REPO_ROOT": tmp_path,
        "MODEL_SNAPSHOT_PATH": model_snapshot_path.as_posix(),
        "HfApi": fake_hf_api,
        "MODEL_IDENTITY": {"model_id": "stabilityai/stable-diffusion-3.5-medium"},
        "WEIGHT_PATH": tmp_path / "weights" / "ckpt_base.pth",
        "PERSISTENT_WEIGHT_PATH": tmp_path / "cache" / "ckpt_base.pth",
        "is_valid_weight_file": lambda _path_obj: True,
        "print_json": lambda name, payload: captured_json.__setitem__(str(name), payload),
    }
    try:
        exec(precheck_source, namespace)
    except RuntimeError as exc:
        namespace["execution_error"] = exc
        if not allow_failure:
            raise
    namespace["captured_json"] = captured_json
    return namespace


def _execute_stage_01_validation_cell(tmp_path: Path, *, include_root_records: bool) -> Dict[str, Any]:
    """
    功能：在测试命名空间中执行 stage 01 的 validation code cell。

    Execute the stage-01 notebook validation code cell inside a minimal test
    namespace.

    Args:
        tmp_path: Temporary pytest directory.
        include_root_records: Whether representative root records exist.

    Returns:
        Execution namespace populated by the cell.
    """
    validation_source = _find_code_cell_source(NOTEBOOK_01_PATH, 'required_formal_files_check')
    run_root = tmp_path / "run_root"
    export_root = tmp_path / "exports"
    package_path = export_root / "01_Paper_Full_Cuda__stage01_notebook_validation.zip"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "artifacts/stage_manifest.json",
            json.dumps({"stage_name": "01_Paper_Full_Cuda"}, ensure_ascii=False, indent=2),
        )

    stage_manifest_path = run_root / "artifacts" / "stage_manifest.json"
    package_manifest_path = export_root / "package_manifest.json"
    audit_summary_path = run_root / "artifacts" / "stage_01_audit_summary.json"
    calibration_record_path = run_root / "records" / "calibration_record.json"
    evaluate_record_path = run_root / "records" / "evaluate_record.json"
    embed_record_path = run_root / "records" / "embed_record.json"
    detect_record_path = run_root / "records" / "detect_record.json"
    thresholds_artifact_path = run_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
    threshold_metadata_artifact_path = run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json"
    canonical_source_pool_manifest_path = (
        run_root / "artifacts" / "stage_01_canonical_source_pool" / "source_pool_manifest.json"
    )
    source_contract_path = run_root / "artifacts" / "parallel_attestation_statistics_input_contract.json"
    pooled_threshold_build_contract_path = run_root / "artifacts" / "stage_01_pooled_threshold_build_contract.json"
    evaluation_report_path = run_root / "artifacts" / "evaluation_report.json"
    run_closure_path = run_root / "artifacts" / "run_closure.json"
    workflow_summary_path = run_root / "artifacts" / "workflow_summary.json"
    runtime_config_snapshot_path = run_root / "runtime_metadata" / "runtime_config_snapshot.yaml"
    prompt_snapshot_path = run_root / "runtime_metadata" / "prompt_snapshot" / "prompt.txt"

    for output_path, payload in [
        (audit_summary_path, {"overall_status": "passed"}),
        (calibration_record_path, {"record_type": "calibration"}),
        (evaluate_record_path, {"record_type": "evaluate"}),
        (thresholds_artifact_path, {"threshold": 0.5}),
        (threshold_metadata_artifact_path, {"threshold_metadata": True}),
        (
            canonical_source_pool_manifest_path,
            {
                "artifact_role": "canonical_source_pool_root",
                "source_truth": "canonical_source_pool",
                "root_contract_mode": "compatibility_view",
                "root_records_required": False,
            },
        ),
        (source_contract_path, {"contract_role": "source_contract", "source_authority": "canonical_source_pool"}),
        (pooled_threshold_build_contract_path, {"contract_role": "pooled_threshold_build_contract"}),
        (evaluation_report_path, {"status": "ok"}),
        (run_closure_path, {"status": {"ok": True, "reason": "ok"}}),
        (
            workflow_summary_path,
            {
                "stage_name": "01_Paper_Full_Cuda",
                "stage_run_id": "stage01_notebook_validation",
                "status": "ok",
            },
        ),
    ]:
        write_json_atomic(output_path, payload)

    runtime_config_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_config_snapshot_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")
    prompt_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_snapshot_path.write_text("prompt 0\n", encoding="utf-8")
    if include_root_records:
        write_json_atomic(embed_record_path, {"record_type": "embed"})
        write_json_atomic(detect_record_path, {"record_type": "detect"})

    stage_manifest = {
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": "stage01_notebook_validation",
        "config_source_path": "/content/ceg_wm_workspace/configs/default.yaml",
        "thresholds_path": thresholds_artifact_path.as_posix(),
        "threshold_metadata_artifact_path": threshold_metadata_artifact_path.as_posix(),
        "evaluation_report_path": evaluation_report_path.as_posix(),
        "run_closure_path": run_closure_path.as_posix(),
        "workflow_summary_path": workflow_summary_path.as_posix(),
        "runtime_config_snapshot_path": runtime_config_snapshot_path.as_posix(),
        "prompt_snapshot_path": prompt_snapshot_path.as_posix(),
        "stage_01_canonical_source_pool_manifest_path": canonical_source_pool_manifest_path.as_posix(),
        "parallel_attestation_statistics_input_contract_path": source_contract_path.as_posix(),
        "stage_01_pooled_threshold_build_contract_path": pooled_threshold_build_contract_path.as_posix(),
        "stage_01_source_truth": "canonical_source_pool",
        "records": {
            "embed_record": {"path": embed_record_path.as_posix()},
            "detect_record": {"path": detect_record_path.as_posix()},
            "calibration_record": {"path": calibration_record_path.as_posix()},
            "evaluate_record": {"path": evaluate_record_path.as_posix()},
        },
    }
    write_json_atomic(stage_manifest_path, stage_manifest)

    package_manifest = {
        "stage_name": "01_Paper_Full_Cuda",
        "package_sha256": compute_file_sha256(package_path),
    }
    write_json_atomic(package_manifest_path, package_manifest)

    stage_summary = {
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": "stage01_notebook_validation",
        "package_path": package_path.as_posix(),
        "stage_manifest_path": stage_manifest_path.as_posix(),
        "package_manifest_path": package_manifest_path.as_posix(),
        "audit_status": "passed",
        "run_root": run_root.as_posix(),
        "log_root": (tmp_path / "logs").as_posix(),
        "runtime_state_root": (tmp_path / "runtime_state").as_posix(),
        "export_root": export_root.as_posix(),
    }
    audit_summary = {
        "overall_status": "passed",
        "definition_status": "passed",
        "strong_compatibility_status": "passed",
        "stage_02_ready": True,
        "stage_03_ready": True,
        "stage_04_ready": True,
    }
    namespace: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "Path": Path,
        "STAGE_SUMMARY": stage_summary,
        "STAGE_MANIFEST": stage_manifest,
        "PACKAGE_MANIFEST": package_manifest,
        "AUDIT_SUMMARY": audit_summary,
        "AUDIT_SUMMARY_PATH": audit_summary_path,
        "NOTEBOOK_NAME": "01_Paper_Full_Cuda",
        "STAGE_RUN_ID": "stage01_notebook_validation",
        "print_json": lambda *_args, **_kwargs: None,
    }
    exec(validation_source, namespace)
    return namespace


def _execute_stage_01_diagnostics_cell(tmp_path: Path, *, include_root_records: bool) -> Dict[str, Any]:
    """
    功能：在测试命名空间中执行 stage 01 的 diagnostics code cell。

    Execute the stage-01 notebook diagnostics code cell after the validation
    cell has populated the expected namespace.

    Args:
        tmp_path: Temporary pytest directory.
        include_root_records: Whether representative root records exist.

    Returns:
        Execution namespace populated by the diagnostics cell.
    """
    namespace = _execute_stage_01_validation_cell(tmp_path, include_root_records=include_root_records)
    log_root = Path(str(namespace["STAGE_SUMMARY"]["log_root"]))
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / "stage_01.log").write_text("stage 01 ok\n", encoding="utf-8")
    diagnostics_source = _find_code_cell_source(NOTEBOOK_01_PATH, 'DIAGNOSTIC_RESULT = {')
    exec(diagnostics_source, namespace)
    return namespace


def test_stage_01_validation_moves_root_records_to_optional_compatibility_views(tmp_path: Path) -> None:
    """
    功能：验证 stage 01 notebook 将 root records 视为 optional compatibility views。

    Verify the stage-01 notebook validation cell treats representative root
    records as optional compatibility views instead of required formal files.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    validation_source = _find_code_cell_source(NOTEBOOK_01_PATH, 'required_formal_files_check')
    namespace = _execute_stage_01_validation_cell(tmp_path, include_root_records=False)

    assert 'REQUIRED_FORMAL_FILES = {' in validation_source
    assert 'OPTIONAL_COMPATIBILITY_FILES = {' in validation_source
    assert 'required_formal_files_check' in validation_source
    assert 'optional_compatibility_views_check' in validation_source
    assert "embed_record" not in namespace["REQUIRED_FORMAL_FILES"]
    assert "detect_record" not in namespace["REQUIRED_FORMAL_FILES"]
    assert set(namespace["OPTIONAL_COMPATIBILITY_FILES"]) == {"embed_record", "detect_record"}
    assert namespace["MISSING_REQUIRED_FORMAL_FILES"] == []
    assert set(namespace["MISSING_OPTIONAL_COMPATIBILITY_FILES"]) == {"embed_record", "detect_record"}
    assert namespace["VALIDATION_RESULT"]["status"] == "ok"
    assert namespace["VALIDATION_RESULT"]["source_truth"] == "canonical_source_pool"
    assert set(namespace["VALIDATION_RESULT"]["missing_optional_compatibility_files"]) == {
        "embed_record",
        "detect_record",
    }


def test_stage_01_precheck_uses_bound_config_view_and_persists_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 stage 01 precheck 先固化 bound config 再执行 formal preflight。 

    Verify the stage-01 notebook precheck persists a bound config artifact and
    runs detect_stage_01_preflight against that bound view instead of the raw
    config path.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    precheck_source = _find_code_cell_source(NOTEBOOK_01_PATH, "PRECHECK_RESULTS = []")
    snapshot_dir = tmp_path / "model_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    namespace = _execute_stage_01_precheck_cell(tmp_path, monkeypatch, model_snapshot_path=snapshot_dir)

    bound_config_path = Path(str(namespace["PRECHECK_BOUND_CONFIG_PATH"]))
    bound_cfg = yaml.safe_load(bound_config_path.read_text(encoding="utf-8"))
    environment_precheck = namespace["captured_json"]["environment_precheck"]

    assert "detect_stage_01_preflight(CONFIG_PATH)" not in precheck_source
    assert "apply_notebook_model_snapshot_binding" in precheck_source
    assert "write_yaml_mapping(PRECHECK_BOUND_CONFIG_PATH, PRECHECK_BOUND_CFG)" in precheck_source
    assert "STAGE_01_PREFLIGHT = detect_stage_01_preflight(PRECHECK_BOUND_CONFIG_PATH)" in precheck_source
    assert namespace["STAGE_01_PREFLIGHT"]["cfg_path"] == bound_config_path.resolve().as_posix()
    assert bound_cfg["model_snapshot_path"] == snapshot_dir.resolve().as_posix()
    assert bound_cfg["model_source_binding"]["binding_status"] == "bound"
    assert environment_precheck["config_path"] == str(namespace["CONFIG_PATH"])
    assert environment_precheck["precheck_bound_config_path"] == str(bound_config_path)
    assert environment_precheck["model_snapshot_path"] == snapshot_dir.as_posix()
    assert environment_precheck["notebook_model_snapshot_binding_applied"] is True
    assert environment_precheck["precheck_bound_config"]["model_snapshot_path"] == snapshot_dir.resolve().as_posix()
    assert environment_precheck["precheck_bound_config"]["model_source_binding"]["binding_status"] == "bound"
    assert namespace["hard_fail"] == []


def test_stage_01_precheck_binding_semantics_match_execute_stage_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 stage 01 precheck 与 execute 共享同一模型绑定语义。 

    Verify the stage-01 notebook precheck uses the same model snapshot binding
    contract as the execute cell instead of checking an unbound raw config.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    precheck_source = _find_code_cell_source(NOTEBOOK_01_PATH, "PRECHECK_RESULTS = []")
    execute_source = _find_code_cell_source(NOTEBOOK_01_PATH, "STAGE_RUN_ID = make_stage_run_id(NOTEBOOK_NAME)")
    snapshot_dir = tmp_path / "model_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    namespace = _execute_stage_01_precheck_cell(tmp_path, monkeypatch, model_snapshot_path=snapshot_dir)
    bound_cfg = yaml.safe_load(Path(str(namespace["PRECHECK_BOUND_CONFIG_PATH"])).read_text(encoding="utf-8"))

    assert 'PRECHECK_BINDING_ENV["CEG_WM_MODEL_SNAPSHOT_PATH"] = str(MODEL_SNAPSHOT_PATH)' in precheck_source
    assert 'NOTEBOOK_SUBPROCESS_ENV["CEG_WM_MODEL_SNAPSHOT_PATH"] = str(MODEL_SNAPSHOT_PATH)' in execute_source
    assert bound_cfg["model_source_binding"]["binding_env_var"] == "CEG_WM_MODEL_SNAPSHOT_PATH"
    assert bound_cfg["model_source_binding"]["binding_source"] == "notebook_snapshot_download"
    assert namespace["STAGE_01_PREFLIGHT"]["model_source_binding_present"] is True
    assert namespace["STAGE_01_PREFLIGHT"]["model_source_binding_status"] == "bound"


def test_stage_01_precheck_hard_fails_when_model_snapshot_path_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 stage 01 precheck 在 notebook snapshot 缺失时仍然正式失败。 

    Verify the stage-01 notebook precheck still hard-fails when
    MODEL_SNAPSHOT_PATH points to a missing directory.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    missing_snapshot_path = tmp_path / "missing_model_snapshot"
    namespace = _execute_stage_01_precheck_cell(
        tmp_path,
        monkeypatch,
        model_snapshot_path=missing_snapshot_path,
        allow_failure=True,
    )
    environment_precheck = namespace["captured_json"]["environment_precheck"]

    assert isinstance(namespace.get("execution_error"), RuntimeError)
    assert namespace["STAGE_01_PREFLIGHT"]["ok"] is False
    assert namespace["STAGE_01_PREFLIGHT"]["model_source_binding_present"] is True
    assert namespace["STAGE_01_PREFLIGHT"]["model_source_binding_status"] == "invalid"
    assert "stage_01_model_source_binding_not_bound" in namespace["STAGE_01_PREFLIGHT"]["failed_checks"]
    assert "stage_01_model_snapshot_path_missing_or_not_directory" in namespace["STAGE_01_PREFLIGHT"]["failed_checks"]
    assert environment_precheck["notebook_model_snapshot_binding_applied"] is True
    assert namespace["hard_fail"] == [
        {
            "name": "stage 01 preflight",
            "detail": json.dumps(namespace["STAGE_01_PREFLIGHT"], ensure_ascii=False, sort_keys=True),
        }
    ]


def test_stage_01_execute_cell_uses_repo_import_subprocess_env() -> None:
    """
    功能：验证 stage 01 execute cell 显式补齐 repo import context。

    Verify the stage-01 notebook execute cell passes an explicit subprocess
    environment that includes the repository import context.

    Args:
        None.

    Returns:
        None.
    """
    execute_source = _find_code_cell_source(NOTEBOOK_01_PATH, "STAGE_RUN_ID = make_stage_run_id(NOTEBOOK_NAME)")

    _assert_execute_source_uses_repo_import_context(execute_source)
    assert 'NOTEBOOK_SUBPROCESS_ENV["CEG_WM_MODEL_SNAPSHOT_PATH"] = str(MODEL_SNAPSHOT_PATH)' in execute_source
    assert "command_result = subprocess.run(" in execute_source


def test_stage_01_validation_reports_optional_root_records_when_present(tmp_path: Path) -> None:
    """
    功能：验证 stage 01 notebook 会报告已导出的 optional compatibility views。

    Verify the stage-01 notebook validation cell reports representative root
    records as present when the optional compatibility views are exported.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    namespace = _execute_stage_01_validation_cell(tmp_path, include_root_records=True)

    assert namespace["MISSING_REQUIRED_FORMAL_FILES"] == []
    assert namespace["MISSING_OPTIONAL_COMPATIBILITY_FILES"] == []
    assert namespace["OPTIONAL_COMPATIBILITY_STATUS"]["embed_record"]["exists"] is True
    assert namespace["OPTIONAL_COMPATIBILITY_STATUS"]["detect_record"]["exists"] is True
    assert namespace["VALIDATION_RESULT"]["optional_compatibility_views"]["embed_record"]["exists"] is True
    assert namespace["VALIDATION_RESULT"]["optional_compatibility_views"]["detect_record"]["exists"] is True


def test_stage_01_diagnostics_uses_current_validation_fields(tmp_path: Path) -> None:
    """
    功能：验证 stage 01 diagnostics cell 使用当前 validation 字段。

    Verify the stage-01 diagnostics cell consumes the current validation
    fields and no longer references the removed missing_files key.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    diagnostics_source = _find_code_cell_source(NOTEBOOK_01_PATH, 'DIAGNOSTIC_RESULT = {')
    namespace = _execute_stage_01_diagnostics_cell(tmp_path, include_root_records=False)

    assert 'VALIDATION_RESULT["missing_files"]' not in diagnostics_source
    assert 'VALIDATION_RESULT.get("missing_required_formal_files", [])' in diagnostics_source
    assert 'VALIDATION_RESULT.get("missing_optional_compatibility_files", [])' in diagnostics_source
    assert namespace["DIAGNOSTIC_RESULT"]["missing_required_formal_files"] == []
    assert set(namespace["DIAGNOSTIC_RESULT"]["missing_optional_compatibility_files"]) == {
        "embed_record",
        "detect_record",
    }
    assert namespace["DIAGNOSTIC_RESULT"]["status"] == "optional_compatibility_missing"


def test_stage_02_precheck_persists_resolved_source_package(tmp_path: Path) -> None:
    """
    功能：验证 stage 02 precheck 会固化解析后的 source package。

    Verify the stage-02 notebook precheck persists the resolved source package
    path and resolution metadata for later execution.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_package_path = _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_for_stage02_auto",
        extra_files={
            "artifacts/parallel_attestation_statistics_input_contract.json": {
                "contract_role": "source_contract",
                "source_authority": "canonical_source_pool",
            }
        },
    )

    namespace = _execute_stage_02_precheck_cell(
        tmp_path,
        drive_project_root=drive_project_root,
        source_package_path=None,
    )

    assert namespace["RESOLVED_SOURCE_PACKAGE_PATH"] == stage_01_package_path
    assert namespace["RESOLVED_SOURCE_PACKAGE_SOURCE"] == "auto_discovered"
    assert namespace["PRECHECK_SOURCE_PACKAGE_SUMMARY"]["requested_package_input"] == "<absent>"
    assert Path(str(namespace["PRECHECK_SOURCE_PACKAGE_SUMMARY"]["resolved_package_path"])) == stage_01_package_path
    assert namespace["PRECHECK_SOURCE_PACKAGE_SUMMARY"]["resolution_source"] == "auto_discovered"


def test_stage_02_precheck_prefers_manual_source_package(tmp_path: Path) -> None:
    """
    功能：验证 stage 02 precheck 保持手工输入优先。

    Verify the stage-02 notebook precheck keeps manual source package input as
    the winning resolution when both manual and auto-discovered packages exist.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    manual_package_path = _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_for_stage02_manual",
        extra_files={
            "artifacts/parallel_attestation_statistics_input_contract.json": {
                "contract_role": "source_contract",
                "source_authority": "canonical_source_pool",
            }
        },
    )
    time.sleep(0.02)
    _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_for_stage02_latest_auto",
        extra_files={
            "artifacts/parallel_attestation_statistics_input_contract.json": {
                "contract_role": "source_contract",
                "source_authority": "canonical_source_pool",
            }
        },
    )

    namespace = _execute_stage_02_precheck_cell(
        tmp_path,
        drive_project_root=drive_project_root,
        source_package_path=str(manual_package_path),
    )

    assert namespace["RESOLVED_SOURCE_PACKAGE_PATH"] == manual_package_path
    assert namespace["RESOLVED_SOURCE_PACKAGE_SOURCE"] == "manual"
    assert Path(str(namespace["PRECHECK_SOURCE_PACKAGE_SUMMARY"]["requested_package_input"])) == manual_package_path
    assert Path(str(namespace["PRECHECK_SOURCE_PACKAGE_SUMMARY"]["resolved_package_path"])) == manual_package_path
    assert namespace["PRECHECK_SOURCE_PACKAGE_SUMMARY"]["resolution_source"] == "manual"


def test_stage_02_execute_cell_uses_resolved_source_package_path() -> None:
    """
    功能：验证 stage 02 execute cell 复用 precheck 已解析路径。

    Verify the stage-02 notebook execute cell uses the resolved source package
    variables and does not rediscover the package.

    Args:
        None.

    Returns:
        None.
    """
    execute_source = _find_code_cell_source(NOTEBOOK_02_PATH, "STAGE_RUN_ID = make_stage_run_id(NOTEBOOK_NAME)")

    assert "RESOLVED_SOURCE_PACKAGE_PATH" in execute_source
    assert "RESOLVED_SOURCE_PACKAGE_SOURCE" in execute_source
    assert "PRECHECK_SOURCE_PACKAGE_RESOLUTION" in execute_source
    assert "discover_stage_packages" not in execute_source
    assert "select_latest_stage_package" not in execute_source
    assert "resolve_stage_package_input_or_discover" not in execute_source
    _assert_execute_source_uses_repo_import_context(execute_source)


def test_stage_03_precheck_persists_resolved_source_package(tmp_path: Path) -> None:
    """
    功能：验证 stage 03 precheck 会固化解析后的 source package。

    Verify the stage-03 notebook precheck persists the resolved source package
    path and resolution metadata for later execution.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_package_path = _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_for_stage03_auto",
        extra_files={
            "artifacts/thresholds/thresholds_artifact.json": {"threshold": 0.5}
        },
    )

    namespace = _execute_stage_03_precheck_cell(
        tmp_path,
        drive_project_root=drive_project_root,
        source_package_path=None,
    )

    assert namespace["RESOLVED_SOURCE_PACKAGE_PATH"] == stage_01_package_path
    assert namespace["RESOLVED_SOURCE_PACKAGE_SOURCE"] == "auto_discovered"
    assert namespace["PRECHECK_SOURCE_PACKAGE_SUMMARY"]["requested_package_input"] == "<absent>"
    assert Path(str(namespace["PRECHECK_SOURCE_PACKAGE_SUMMARY"]["resolved_package_path"])) == stage_01_package_path
    assert namespace["PRECHECK_SOURCE_PACKAGE_SUMMARY"]["resolution_source"] == "auto_discovered"


def test_stage_03_precheck_prefers_manual_source_package(tmp_path: Path) -> None:
    """
    功能：验证 stage 03 precheck 保持手工输入优先。

    Verify the stage-03 notebook precheck keeps manual source package input as
    the winning resolution when both manual and auto-discovered packages exist.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    manual_package_path = _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_for_stage03_manual",
        extra_files={
            "artifacts/thresholds/thresholds_artifact.json": {"threshold": 0.5}
        },
    )
    time.sleep(0.02)
    _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_for_stage03_latest_auto",
        extra_files={
            "artifacts/thresholds/thresholds_artifact.json": {"threshold": 0.7}
        },
    )

    namespace = _execute_stage_03_precheck_cell(
        tmp_path,
        drive_project_root=drive_project_root,
        source_package_path=str(manual_package_path),
    )

    assert namespace["RESOLVED_SOURCE_PACKAGE_PATH"] == manual_package_path
    assert namespace["RESOLVED_SOURCE_PACKAGE_SOURCE"] == "manual"
    assert Path(str(namespace["PRECHECK_SOURCE_PACKAGE_SUMMARY"]["requested_package_input"])) == manual_package_path
    assert Path(str(namespace["PRECHECK_SOURCE_PACKAGE_SUMMARY"]["resolved_package_path"])) == manual_package_path
    assert namespace["PRECHECK_SOURCE_PACKAGE_SUMMARY"]["resolution_source"] == "manual"


def test_stage_03_precheck_rejects_diagnostics_package_as_formal_input(tmp_path: Path) -> None:
    """
    功能：验证 stage 03 precheck 不会把 diagnostics package 当成 formal 输入。

    Verify the stage-03 notebook precheck does not resolve a diagnostics
    package as a valid formal source package.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    _create_failure_diagnostics_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_diagnostics_only_for_stage03",
    )

    with pytest.raises(RuntimeError):
        _execute_stage_03_precheck_cell(
            tmp_path,
            drive_project_root=drive_project_root,
            source_package_path=None,
        )


def test_stage_03_execute_cell_uses_resolved_source_package_path() -> None:
    """
    功能：验证 stage 03 execute cell 复用 precheck 已解析路径。

    Verify the stage-03 notebook execute cell uses the resolved source package
    variables and does not rediscover the package.

    Args:
        None.

    Returns:
        None.
    """
    execute_source = _find_code_cell_source(NOTEBOOK_03_PATH, "STAGE_RUN_ID = make_stage_run_id(NOTEBOOK_NAME)")

    assert "RESOLVED_SOURCE_PACKAGE_PATH" in execute_source
    assert "RESOLVED_SOURCE_PACKAGE_SOURCE" in execute_source
    assert "PRECHECK_SOURCE_PACKAGE_RESOLUTION" in execute_source
    assert "resolve_stage_package_input_or_discover" not in execute_source
    assert 'NOTEBOOK_SUBPROCESS_ENV["CEG_WM_MODEL_SNAPSHOT_PATH"] = str(MODEL_SNAPSHOT_PATH)' in execute_source
    _assert_execute_source_uses_repo_import_context(execute_source)


def test_stage_03_notebook_defines_source_package_path_as_none() -> None:
    """
    功能：验证 stage 03 notebook 显式初始化 SOURCE_PACKAGE_PATH。 

    Verify the stage-03 notebook explicitly initializes SOURCE_PACKAGE_PATH to
    None so later cells do not raise NameError in the default flow.

    Args:
        None.

    Returns:
        None.
    """
    config_source = _find_code_cell_source(NOTEBOOK_03_PATH, 'NOTEBOOK_NAME = "03_Experiment_Matrix_Full"')
    namespace: Dict[str, Any] = {"__builtins__": __builtins__}
    exec(config_source, namespace)

    assert "SOURCE_PACKAGE_PATH" in namespace
    assert namespace["SOURCE_PACKAGE_PATH"] is None


def test_stage_04_precheck_auto_discovers_stage_01_package_when_manual_input_absent(tmp_path: Path) -> None:
    """
    功能：验证 stage 04 precheck 能自动发现 stage 01 formal package。 

    Verify the stage-04 notebook precheck discovers the latest valid stage-01
    package when no manual input is provided.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_package_path = _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_auto",
    )

    namespace = _execute_stage_04_precheck_cell(
        tmp_path,
        drive_project_root=drive_project_root,
        stage_01_package_path=None,
        stage_02_package_path=None,
        stage_03_package_path=None,
        require_stage_02=False,
        require_stage_03=False,
    )

    stage_01_summary = namespace["PRECHECK_STAGE_PACKAGE_SUMMARY"]["stage_01"]
    assert namespace["hard_fail"] == []
    assert stage_01_summary["manual_input_used"] is False
    assert stage_01_summary["required"] is True
    assert stage_01_summary["formal_package_valid"] is True
    assert stage_01_summary["selected_package_path"] == stage_01_package_path.as_posix()
    assert stage_01_summary["auto_discovered_package_path"] == stage_01_package_path.as_posix()
    assert namespace["RESOLVED_STAGE_01_PACKAGE_PATH"] == stage_01_package_path


def test_stage_04_precheck_allows_missing_optional_stage_02_package(tmp_path: Path) -> None:
    """
    功能：验证 stage 02 非必需时，缺失不应触发 hard fail。 

    Verify the stage-04 notebook precheck does not hard fail when stage 02 is
    optional and absent while the required stages resolve successfully.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_optional_stage02",
    )
    stage_03_package_path = _create_formal_stage_package(
        drive_project_root,
        stage_name="03_Experiment_Matrix_Full",
        stage_run_id="stage03_required",
    )

    namespace = _execute_stage_04_precheck_cell(
        tmp_path,
        drive_project_root=drive_project_root,
        stage_01_package_path=None,
        stage_02_package_path=None,
        stage_03_package_path=None,
        require_stage_02=False,
        require_stage_03=True,
    )

    stage_02_summary = namespace["PRECHECK_STAGE_PACKAGE_SUMMARY"]["stage_02"]
    stage_03_summary = namespace["PRECHECK_STAGE_PACKAGE_SUMMARY"]["stage_03"]
    assert namespace["hard_fail"] == []
    assert stage_02_summary["required"] is False
    assert stage_02_summary["formal_package_valid"] is False
    assert namespace["RESOLVED_STAGE_02_PACKAGE_PATH"] is None
    assert stage_03_summary["formal_package_valid"] is True
    assert namespace["RESOLVED_STAGE_03_PACKAGE_PATH"] == stage_03_package_path


def test_stage_04_precheck_allows_missing_optional_stage_03_package(tmp_path: Path) -> None:
    """
    功能：验证 stage 03 非必需时，缺失不应触发 hard fail。 

    Verify the stage-04 notebook precheck does not hard fail when stage 03 is
    optional and absent while the required stages resolve successfully.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_optional_stage03",
    )
    stage_02_package_path = _create_formal_stage_package(
        drive_project_root,
        stage_name="02_Parallel_Attestation_Statistics",
        stage_run_id="stage02_required",
    )

    namespace = _execute_stage_04_precheck_cell(
        tmp_path,
        drive_project_root=drive_project_root,
        stage_01_package_path=None,
        stage_02_package_path=None,
        stage_03_package_path=None,
        require_stage_02=True,
        require_stage_03=False,
    )

    stage_02_summary = namespace["PRECHECK_STAGE_PACKAGE_SUMMARY"]["stage_02"]
    stage_03_summary = namespace["PRECHECK_STAGE_PACKAGE_SUMMARY"]["stage_03"]
    assert namespace["hard_fail"] == []
    assert stage_02_summary["formal_package_valid"] is True
    assert namespace["RESOLVED_STAGE_02_PACKAGE_PATH"] == stage_02_package_path
    assert stage_03_summary["required"] is False
    assert stage_03_summary["formal_package_valid"] is False
    assert namespace["RESOLVED_STAGE_03_PACKAGE_PATH"] is None


def test_stage_04_helper_does_not_discover_diagnostics_package(tmp_path: Path) -> None:
    """
    功能：验证 diagnostics package 不会被误识别为 formal package。 

    Verify diagnostics packages remain excluded from notebook auto discovery
    when no formal package exists.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    _create_failure_diagnostics_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_diagnostics_only",
    )

    resolution = resolve_stage_package_input_or_discover(
        None,
        drive_project_root / "exports" / "01_Paper_Full_Cuda",
        expected_stage_name="01_Paper_Full_Cuda",
    )

    assert resolution["selected_package_valid"] is False
    assert resolution["selected_package_path"] == "<absent>"
    assert resolution["candidates"] == []


def test_stage_04_helper_prefers_manual_package_input_over_auto_discovery(tmp_path: Path) -> None:
    """
    功能：验证手工指定 package 时优先于自动发现结果。 

    Verify manual package input wins over automatic discovery when both are
    available.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_project_root"
    manual_package_path = _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_manual",
    )
    time.sleep(0.02)
    _create_formal_stage_package(
        drive_project_root,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_latest_auto",
    )

    resolution = resolve_stage_package_input_or_discover(
        str(manual_package_path),
        drive_project_root / "exports" / "01_Paper_Full_Cuda",
        expected_stage_name="01_Paper_Full_Cuda",
    )

    assert resolution["manual_input_used"] is True
    assert resolution["selected_package_valid"] is True
    assert resolution["selected_package_path"] == manual_package_path.as_posix()
    assert resolution["auto_discovered_package_path"] == "<skipped_manual_override>"
    assert resolution["selection_reason"] == "manual package path override"


def test_stage_04_execute_cell_uses_resolved_package_paths() -> None:
    """
    功能：验证 stage 04 执行 cell 复用 precheck 已解析的 package 路径。 

    Verify the stage-04 execute cell uses the pre-resolved package variables
    instead of re-reading the raw manual-input variables.

    Args:
        None.

    Returns:
        None.
    """
    execute_source = _find_code_cell_source(NOTEBOOK_04_PATH, "STAGE_RUN_ID = make_stage_run_id(NOTEBOOK_NAME)")

    assert "RESOLVED_STAGE_01_PACKAGE_PATH" in execute_source
    assert "str(RESOLVED_STAGE_01_PACKAGE_PATH)" in execute_source
    assert "str(Path(STAGE_01_PACKAGE_PATH))" not in execute_source
    assert "if RESOLVED_STAGE_02_PACKAGE_PATH is not None:" in execute_source
    assert "if RESOLVED_STAGE_03_PACKAGE_PATH is not None:" in execute_source
    _assert_execute_source_uses_repo_import_context(execute_source)


@pytest.mark.parametrize(
    "script_name",
    [
        "01_Paper_Full_Cuda.py",
        "02_Parallel_Attestation_Statistics.py",
        "03_Experiment_Matrix_Full.py",
        "04_Release_And_Signoff.py",
    ],
)
def test_stage_scripts_help_accept_repo_pythonpath(script_name: str) -> None:
    """
    功能：验证 stage 脚本在补齐 repo PYTHONPATH 后可完成入口解析。

    Verify each stage script can complete top-level imports and argparse help
    rendering when the repository root is explicitly injected into PYTHONPATH.

    Args:
        script_name: Stage script filename.

    Returns:
        None.
    """
    script_path = REPO_ROOT / "scripts" / script_name
    env_mapping = build_repo_import_subprocess_env(repo_root=REPO_ROOT)
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=str(REPO_ROOT),
        env=env_mapping,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined_output = f"{result.stdout}\n{result.stderr}"

    assert result.returncode == 0
    assert "usage:" in combined_output.lower()
    assert "ModuleNotFoundError: No module named 'scripts'" not in combined_output
