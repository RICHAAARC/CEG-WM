"""
File purpose: Contract tests for the PW04 notebook entrypoint and parameter wiring.
Module type: General module
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, cast

import pytest

from scripts.notebook_runtime_common import REPO_ROOT


NOTEBOOK_PW04_PREPARE_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW04_Attack_Merge_And_Metrics - 1.Prepare.ipynb"
NOTEBOOK_PW04_QUALITY_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW04_Attack_Merge_And_Metrics - 2.Quality.ipynb"
NOTEBOOK_PW04_FINALIZE_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW04_Attack_Merge_And_Metrics - 3.Finalize.ipynb"
NOTEBOOK_PW04_PATHS = [
    (NOTEBOOK_PW04_PREPARE_PATH, "prepare"),
    (NOTEBOOK_PW04_QUALITY_PATH, "quality_shard"),
    (NOTEBOOK_PW04_FINALIZE_PATH, "finalize"),
]


def _load_notebook(notebook_path: Path) -> Dict[str, Any]:
    """
    Load one notebook JSON object.

    Args:
        notebook_path: Notebook path.

    Returns:
        Parsed notebook payload.
    """
    notebook_payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    if not isinstance(notebook_payload, dict):
        raise AssertionError(f"notebook root must be object: {notebook_path}")
    return cast(Dict[str, Any], notebook_payload)


def _find_cell_sources(notebook_path: Path, marker: str, cell_type: str) -> List[str]:
    """
    Find notebook cell sources by marker text and cell type.

    Args:
        notebook_path: Notebook path.
        marker: Marker text.
        cell_type: Notebook cell type.

    Returns:
        Matching cell source texts.
    """
    notebook_payload = _load_notebook(notebook_path)
    cells = notebook_payload.get("cells")
    if not isinstance(cells, list):
        raise AssertionError(f"cells must be list: {notebook_path}")
    matches: List[str] = []
    for cell_node in cast(List[object], cells):
        if not isinstance(cell_node, dict):
            continue
        if cell_node.get("cell_type") != cell_type:
            continue
        source_node = cell_node.get("source")
        if not isinstance(source_node, list):
            continue
        source_text = "\n".join(str(line) for line in cast(List[object], source_node))
        while "\n\n" in source_text:
            source_text = source_text.replace("\n\n", "\n")
        if marker in source_text:
            matches.append(source_text)
    return matches


def _find_code_cell_source(notebook_path: Path, marker: str) -> str:
    """
    Find one code cell by marker text.

    Args:
        notebook_path: Notebook path.
        marker: Marker text.

    Returns:
        Joined code source text.
    """
    matches = _find_cell_sources(notebook_path, marker, "code")
    if not matches:
        raise AssertionError(f"code cell marker not found: {marker}")
    return matches[0]


@pytest.mark.parametrize(("notebook_path", "expected_mode"), NOTEBOOK_PW04_PATHS)
def test_pw04_notebook_binds_expected_script_and_parameters(
    notebook_path: Path,
    expected_mode: str,
) -> None:
    """
    Verify the PW04 notebook binds the real script and required parameters.

    Args:
        None.

    Returns:
        None.
    """
    constants_source = _find_code_cell_source(notebook_path, "SCRIPT_PATH = REPO_ROOT")
    bootstrap_source = _find_code_cell_source(notebook_path, "prepare_local_runtime_for_stage(")
    precheck_source = _find_code_cell_source(notebook_path, "PRECHECK_RESULTS = []")
    execute_source = _find_code_cell_source(notebook_path, "build_pw04_command(")
    result_source = _find_code_cell_source(notebook_path, "read_pw04_result_summary(")
    quality_runtime_matches = _find_cell_sources(
        notebook_path,
        "resolve_pw04_quality_runtime_summary(",
        "code",
    )

    assert '"PW04_Attack_Merge_And_Metrics.py"' in constants_source
    assert 'PERSISTENT_DRIVE_PROJECT_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "CEG_WM_PaperWorkflow"' in constants_source
    assert 'DRIVE_PROJECT_ROOT = PERSISTENT_DRIVE_PROJECT_ROOT' in constants_source
    assert 'LOCAL_HF_HOME = REPO_ROOT / "huggingface_cache"' in constants_source
    assert 'LOCAL_HF_HUB_CACHE = LOCAL_HF_HOME / "hub"' in constants_source
    assert 'LOCAL_TRANSFORMERS_CACHE = LOCAL_HF_HOME / "transformers"' in constants_source
    assert 'FAMILY_ID = "paper_eval_family_pilot_v1"' in constants_source
    assert 'PW04_MODE =' not in constants_source
    assert 'FORCE_RERUN = False' in constants_source

    if expected_mode == "prepare":
        assert 'QUALITY_SHARD_COUNT = None' in constants_source
        assert 'ENABLE_TAIL_ESTIMATION = False' in constants_source
        assert 'QUALITY_SHARD_INDEX' not in constants_source
        assert 'QUALITY_DEVICE_OVERRIDE' not in constants_source
        assert 'QUALITY_LPIPS_BATCH_SIZE' not in constants_source
        assert 'QUALITY_CLIP_BATCH_SIZE' not in constants_source
        assert 'QUALITY_PSNR_SSIM_BATCH_SIZE' not in constants_source
        assert 'QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET' not in constants_source
        assert 'PW04_MODE = "prepare"' in precheck_source
        assert 'QUALITY_SHARD_INDEX = 0' in precheck_source
        assert quality_runtime_matches == []
        assert 'quality_runtime_summary=QUALITY_RUNTIME_SUMMARY' not in execute_source
        assert 'pw04_mode=PW04_MODE' in execute_source
        assert 'PW04_COMMAND_KWARGS["quality_shard_count"] = QUALITY_SHARD_COUNT' in execute_source
        assert 'pw04_prepare_runtime_diagnostics.json' in execute_source
    elif expected_mode == "quality_shard":
        assert 'QUALITY_SHARD_INDEX = 0' in constants_source
        assert 'QUALITY_DEVICE_OVERRIDE = "auto"' in constants_source
        assert 'QUALITY_LPIPS_BATCH_SIZE = 256' in constants_source
        assert 'QUALITY_CLIP_BATCH_SIZE = 400' in constants_source
        assert 'QUALITY_PSNR_SSIM_DEVICE = None' in constants_source
        assert 'QUALITY_PSNR_SSIM_BATCH_SIZE = None' in constants_source
        assert 'QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET = None' in constants_source
        assert 'QUALITY_SHARD_COUNT' not in constants_source
        assert 'ENABLE_TAIL_ESTIMATION' not in constants_source
        assert 'PW04_MODE = "quality_shard"' in precheck_source
        assert 'ENABLE_TAIL_ESTIMATION = False' in precheck_source
        assert len(quality_runtime_matches) == 1
        quality_runtime_source = quality_runtime_matches[0]
        assert 'quality_device_override=QUALITY_DEVICE_OVERRIDE' in quality_runtime_source
        assert 'quality_lpips_batch_size_override=QUALITY_LPIPS_BATCH_SIZE' in quality_runtime_source
        assert 'quality_clip_batch_size_override=QUALITY_CLIP_BATCH_SIZE' in quality_runtime_source
        assert 'quality_psnr_ssim_device_override=QUALITY_PSNR_SSIM_DEVICE' in quality_runtime_source
        assert 'quality_psnr_ssim_batch_size_override=QUALITY_PSNR_SSIM_BATCH_SIZE' in quality_runtime_source
        assert 'quality_psnr_ssim_batch_element_budget_override=QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET' in quality_runtime_source
        assert 'base_env=os.environ' in quality_runtime_source
        assert 'quality_runtime_summary=QUALITY_RUNTIME_SUMMARY' in execute_source
        assert 'pw04_mode=PW04_MODE' in execute_source
        assert 'lpips_batch_size_source' in execute_source
        assert 'clip_batch_size_source' in execute_source
        assert 'batch_default_reason' in execute_source
        assert 'pw04_quality_shard_' in execute_source
        assert 'unit_label="quality_pairs"' in execute_source
    else:
        assert 'QUALITY_SHARD_INDEX' not in constants_source
        assert 'QUALITY_SHARD_COUNT' not in constants_source
        assert 'ENABLE_TAIL_ESTIMATION' not in constants_source
        assert 'QUALITY_DEVICE_OVERRIDE' not in constants_source
        assert 'QUALITY_LPIPS_BATCH_SIZE' not in constants_source
        assert 'QUALITY_CLIP_BATCH_SIZE' not in constants_source
        assert 'QUALITY_PSNR_SSIM_BATCH_SIZE' not in constants_source
        assert 'QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET' not in constants_source
        assert 'PW04_MODE = "finalize"' in precheck_source
        assert 'QUALITY_SHARD_INDEX = 0' in precheck_source
        assert 'ENABLE_TAIL_ESTIMATION = False' in precheck_source
        assert quality_runtime_matches == []
        assert 'quality_runtime_summary=QUALITY_RUNTIME_SUMMARY' not in execute_source
        assert 'pw04_mode=PW04_MODE' in execute_source
        assert 'pw04_finalize_runtime_diagnostics.json' in execute_source
        assert 'unit_label="quality_shards"' in execute_source

    assert 'prepare_local_runtime_for_stage(' in bootstrap_source
    assert 'resolve_notebook_model_cache_layout(DRIVE_MOUNT_ROOT, REPO_ROOT, create_directories=True)' in bootstrap_source
    assert 'run_checked([sys.executable, "-m", "pip", "install", "lpips", "open_clip_torch"], cwd=REPO_ROOT)' in bootstrap_source
    assert 'os.environ["HUGGINGFACE_HUB_CACHE"] = str(LOCAL_HF_HUB_CACHE)' in bootstrap_source
    assert 'snapshot_download(' not in bootstrap_source
    assert 'bootstrap_notebook_model_cache(' not in bootstrap_source
    assert 'quality_dependency_check' not in bootstrap_source

    assert 'build_pw04_command(' in execute_source
    assert 'PW04_COMMAND_KWARGS = {' in execute_source
    assert 'build_pw04_subprocess_env(' in execute_source
    assert 'resolve_pw04_expected_output(' in execute_source
    assert 'load_gpu_peak_summary(' in execute_source
    assert 'build_gpu_peak_notebook_summary(' in execute_source
    assert 'build_stage_runtime_diagnostics_payload(' in execute_source
    assert 'build_stage_runtime_workload_summary(' in execute_source
    assert 'write_stage_runtime_diagnostics(' in execute_source
    assert 'PW04_RUNTIME_DIAGNOSTICS_PATH =' in execute_source
    assert 'PW04_ARCHIVE_STAGE_NAME = f"PW04_Attack_Merge_And_Metrics_{PW04_MODE}"' in execute_source
    assert 'PW04_COUNT_SUMMARY = {' in execute_source
    assert '"quality_shard_index": QUALITY_SHARD_INDEX' in execute_source
    assert '"force_rerun": FORCE_RERUN' in execute_source
    assert '"enable_tail_estimation": ENABLE_TAIL_ESTIMATION' in execute_source
    assert 'def load_gpu_peak_summary(' not in execute_source
    assert 'def build_gpu_peak_notebook_summary(' not in execute_source

    assert 'read_pw04_result_summary(' in result_source
    assert 'prepare_manifest_path=PREPARE_MANIFEST_PATH' in result_source
    assert 'selected_quality_shard_path=SELECTED_QUALITY_SHARD_PATH' in result_source
    assert 'pw04_summary_path=PW04_SUMMARY_PATH' in result_source
    assert 'gpu_peak_notebook_summary=GPU_PEAK_NOTEBOOK_SUMMARY' in result_source


def test_pw04_notebook_markdown_matches_stage_specific_runtime_scope() -> None:
    """
    Verify PW04 notebook markdown exposes only the parameters and runtime guidance relevant to each stage.

    Args:
        None.

    Returns:
        None.
    """
    prepare_params = _find_cell_sources(
        NOTEBOOK_PW04_PREPARE_PATH,
        "## 运行参数与基础路径说明",
        "markdown",
    )[0]
    quality_params = _find_cell_sources(
        NOTEBOOK_PW04_QUALITY_PATH,
        "## 运行参数与基础路径说明",
        "markdown",
    )[0]
    quality_runtime_markdown = _find_cell_sources(
        NOTEBOOK_PW04_QUALITY_PATH,
        "## Quality Runtime 配置说明",
        "markdown",
    )[0]
    finalize_params = _find_cell_sources(
        NOTEBOOK_PW04_FINALIZE_PATH,
        "## 运行参数与基础路径说明",
        "markdown",
    )[0]

    assert 'QUALITY_SHARD_INDEX' not in prepare_params
    assert 'QUALITY_DEVICE_OVERRIDE' not in prepare_params
    assert 'QUALITY_LPIPS_BATCH_SIZE' not in prepare_params
    assert 'QUALITY_CLIP_BATCH_SIZE' not in prepare_params
    assert 'QUALITY_PSNR_SSIM_BATCH_SIZE' not in prepare_params
    assert 'QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET' not in prepare_params

    assert 'QUALITY_SHARD_INDEX' in quality_params
    assert 'QUALITY_DEVICE_OVERRIDE' in quality_params
    assert 'QUALITY_LPIPS_BATCH_SIZE' in quality_params
    assert 'QUALITY_CLIP_BATCH_SIZE' in quality_params
    assert 'QUALITY_PSNR_SSIM_DEVICE' in quality_params
    assert 'QUALITY_PSNR_SSIM_BATCH_SIZE' in quality_params
    assert 'QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET' in quality_params
    assert 'QUALITY_SHARD_COUNT' not in quality_params
    assert 'ENABLE_TAIL_ESTIMATION' not in quality_params
    assert 'None 表示不覆盖' in quality_runtime_markdown
    assert 'notebook override 优先于环境变量' in quality_runtime_markdown
    assert 'LPIPS=128' in quality_runtime_markdown
    assert 'CLIP=256' in quality_runtime_markdown
    assert 'PW_QUALITY_PSNR_SSIM_DEVICE' in quality_runtime_markdown
    assert 'PW_QUALITY_PSNR_SSIM_BATCH_SIZE' in quality_runtime_markdown
    assert 'PW_QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET' in quality_runtime_markdown
    assert '低显存' not in quality_runtime_markdown

    assert 'QUALITY_SHARD_INDEX' not in finalize_params
    assert 'QUALITY_SHARD_COUNT' not in finalize_params
    assert 'ENABLE_TAIL_ESTIMATION' not in finalize_params
    assert 'QUALITY_DEVICE_OVERRIDE' not in finalize_params
    assert 'QUALITY_LPIPS_BATCH_SIZE' not in finalize_params
    assert 'QUALITY_CLIP_BATCH_SIZE' not in finalize_params
    assert 'QUALITY_PSNR_SSIM_BATCH_SIZE' not in finalize_params
    assert 'QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET' not in finalize_params


def test_pw04_quality_notebook_wires_psnr_ssim_batch_overrides() -> None:
    """
    Verify the PW04 quality notebook exposes and forwards PSNR/SSIM batch override parameters.

    Args:
        None.

    Returns:
        None.
    """
    constants_source = _find_code_cell_source(NOTEBOOK_PW04_QUALITY_PATH, "SCRIPT_PATH = REPO_ROOT")
    quality_runtime_source = _find_code_cell_source(
        NOTEBOOK_PW04_QUALITY_PATH,
        "resolve_pw04_quality_runtime_summary(",
    )
    quality_runtime_markdown = _find_cell_sources(
        NOTEBOOK_PW04_QUALITY_PATH,
        "## Quality Runtime 配置说明",
        "markdown",
    )[0]

    assert 'QUALITY_PSNR_SSIM_BATCH_SIZE = None' in constants_source
    assert 'QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET = None' in constants_source
    assert 'QUALITY_PSNR_SSIM_DEVICE = None' in constants_source
    assert 'quality_psnr_ssim_device_override=QUALITY_PSNR_SSIM_DEVICE' in quality_runtime_source
    assert 'quality_psnr_ssim_batch_size_override=QUALITY_PSNR_SSIM_BATCH_SIZE' in quality_runtime_source
    assert 'quality_psnr_ssim_batch_element_budget_override=QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET' in quality_runtime_source
    assert 'PW_QUALITY_PSNR_SSIM_DEVICE' in quality_runtime_markdown
    assert 'PW_QUALITY_PSNR_SSIM_BATCH_SIZE' in quality_runtime_markdown
    assert 'PW_QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET' in quality_runtime_markdown


@pytest.mark.parametrize("notebook_path", [path for path, _ in NOTEBOOK_PW04_PATHS])
def test_pw04_notebook_reads_pw02_inputs_and_pw04_outputs(notebook_path: Path) -> None:
    """
    Verify the PW04 notebook precheck and summary cells read the expected artifacts.

    Args:
        None.

    Returns:
        None.
    """
    precheck_source = _find_code_cell_source(notebook_path, "PRECHECK_RESULTS = []")
    summary_source = _find_code_cell_source(notebook_path, "read_pw04_result_summary(")

    assert 'PW02_SUMMARY_PATH = FAMILY_ROOT / "runtime_state" / "pw02_summary.json"' in precheck_source
    assert 'FINALIZE_MANIFEST_PATH = FAMILY_ROOT / "exports" / "pw02" / "paper_source_finalize_manifest.json"' in precheck_source
    assert 'ATTACK_SHARD_PLAN_PATH = FAMILY_ROOT / "manifests" / "attack_shard_plan.json"' in precheck_source
    assert 'CONTENT_THRESHOLD_EXPORT_PATH = FAMILY_ROOT / "exports" / "pw02" / "thresholds" / "content" / "thresholds.json"' in precheck_source
    assert 'ATTESTATION_THRESHOLD_EXPORT_PATH = FAMILY_ROOT / "exports" / "pw02" / "thresholds" / "attestation" / "thresholds.json"' in precheck_source
    assert 'PROJECT_ROOT_PRECHECK_LABEL = "项目运行根目录存在" if LOCAL_RUNTIME_ENABLED else "Drive 项目根目录存在"' in precheck_source
    assert 'record_precheck(PROJECT_ROOT_PRECHECK_LABEL, DRIVE_PROJECT_ROOT.exists(), str(DRIVE_PROJECT_ROOT))' in precheck_source
    assert 'PREPARE_MANIFEST_PATH = FAMILY_ROOT / "exports" / "pw04" / "manifests" / "pw04_prepare_manifest.json"' in precheck_source
    assert 'QUALITY_PAIR_PLAN_PATH = QUALITY_ROOT / "quality_pair_plan.json"' in precheck_source
    assert 'SELECTED_QUALITY_SHARD_PATH = QUALITY_ROOT / "shards" / f"quality_shard_{QUALITY_SHARD_INDEX:04d}.json"' in precheck_source
    assert '所有计划内 PW03 shard manifest 存在且 completed' in precheck_source
    assert 'expected_attack_event_count == discovered_attack_event_count' in precheck_source
    assert 'PW04_MODE == "prepare"' in precheck_source
    assert 'PW04_MODE == "quality_shard"' in precheck_source
    assert '全部计划内 quality shard 已完成' in precheck_source

    assert 'PW04_SUMMARY_PATH = FAMILY_ROOT / "runtime_state" / "pw04_summary.json"' in precheck_source
    assert 'prepare_manifest_path=PREPARE_MANIFEST_PATH' in summary_source
    assert 'selected_quality_shard_path=SELECTED_QUALITY_SHARD_PATH' in summary_source
    assert 'pw04_summary_path=PW04_SUMMARY_PATH' in summary_source
    assert 'gpu_peak_notebook_summary=GPU_PEAK_NOTEBOOK_SUMMARY' in summary_source
    assert 'if PW04_MODE == "prepare":' not in summary_source
    assert 'FORMAL_ATTACK_FINAL_DECISION_METRICS_PATH' not in summary_source
    assert 'PAPER_SCOPE_REGISTRY_PATH' not in summary_source
    assert 'TAIL_ESTIMATION_PATHS' not in summary_source
    assert 'matplotlib' not in summary_source
    assert 'np.random' not in summary_source


def test_pw04_notebook_runtime_diagnostics_names_are_mode_specific() -> None:
    """
    Verify PW04 notebooks bind distinct runtime diagnostics file names by mode.

    Args:
        None.

    Returns:
        None.
    """
    prepare_execute = _find_code_cell_source(NOTEBOOK_PW04_PREPARE_PATH, "build_pw04_command(")
    quality_execute = _find_code_cell_source(NOTEBOOK_PW04_QUALITY_PATH, "build_pw04_command(")
    finalize_execute = _find_code_cell_source(NOTEBOOK_PW04_FINALIZE_PATH, "build_pw04_command(")

    assert 'pw04_prepare_runtime_diagnostics.json' in prepare_execute
    assert 'pw04_quality_shard_' in quality_execute
    assert 'pw04_finalize_runtime_diagnostics.json' in finalize_execute


def test_pw04_prepare_notebook_has_single_main_execution_cell() -> None:
    """
    Verify the PW04 prepare notebook contains only one main execution cell.

    Args:
        None.

    Returns:
        None.
    """
    execute_cells = _find_cell_sources(
        NOTEBOOK_PW04_PREPARE_PATH,
        "build_pw04_command(",
        "code",
    )

    assert len(execute_cells) == 1
    assert 'quality_runtime_summary=QUALITY_RUNTIME_SUMMARY' not in execute_cells[0]


def test_pw04_wrapper_delegates_to_run_function(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Verify the PW04 wrapper only parses args and delegates to the run function.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    wrapper_module = importlib.import_module("paper_workflow.scripts.PW04_Attack_Merge_And_Metrics")
    captured: Dict[str, Any] = {}

    def fake_run_pw04_merge_attack_event_shards(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(wrapper_module, "run_pw04_merge_attack_event_shards", fake_run_pw04_merge_attack_event_shards)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "PW04_Attack_Merge_And_Metrics.py",
            "--drive-project-root",
            str(tmp_path / "drive_root"),
            "--family-id",
            "family_pw04_demo",
            "--pw04-mode",
            "quality_shard",
            "--quality-shard-index",
            "3",
            "--force-rerun",
            "--enable-tail-estimation",
        ],
    )

    assert wrapper_module.main() == 0
    assert captured["drive_project_root"] == tmp_path / "drive_root"
    assert captured["family_id"] == "family_pw04_demo"
    assert captured["pw04_mode"] == "quality_shard"
    assert captured["quality_shard_index"] == 3
    assert captured["quality_shard_count"] is None
    assert captured["force_rerun"] is True
    assert captured["enable_tail_estimation"] is True


def test_pw04_wrapper_prepare_mode_forwards_quality_shard_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Verify the PW04 wrapper forwards quality_shard_count only for prepare mode.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    wrapper_module = importlib.import_module("paper_workflow.scripts.PW04_Attack_Merge_And_Metrics")
    captured: Dict[str, Any] = {}

    def fake_run_pw04_merge_attack_event_shards(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(wrapper_module, "run_pw04_merge_attack_event_shards", fake_run_pw04_merge_attack_event_shards)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "PW04_Attack_Merge_And_Metrics.py",
            "--drive-project-root",
            str(tmp_path / "drive_root"),
            "--family-id",
            "family_pw04_demo",
            "--pw04-mode",
            "prepare",
            "--quality-shard-count",
            "1",
        ],
    )

    assert wrapper_module.main() == 0
    assert captured["drive_project_root"] == tmp_path / "drive_root"
    assert captured["family_id"] == "family_pw04_demo"
    assert captured["pw04_mode"] == "prepare"
    assert captured["quality_shard_index"] is None
    assert captured["quality_shard_count"] == 1
