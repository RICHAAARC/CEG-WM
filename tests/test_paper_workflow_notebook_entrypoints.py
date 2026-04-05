"""
File purpose: Contract tests for paper_workflow notebook entrypoints and script guards.
Module type: General module
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, cast

import pytest

from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest
from paper_workflow.scripts.pw01_run_source_event_shard import run_pw01_source_event_shard
from scripts.notebook_runtime_common import REPO_ROOT, build_repo_import_subprocess_env


NOTEBOOK_PW00_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW00_Paper_Eval_Family_Manifest.ipynb"
NOTEBOOK_PW01_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW01_Source_Event_Shards.ipynb"
NOTEBOOK_PW02_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW02_Source_Merge_And_Global_Thresholds.ipynb"


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
    if cell_type not in {"code", "markdown"}:
        raise ValueError(f"unsupported cell_type: {cell_type}")

    notebook_payload = _load_notebook(notebook_path)
    cells = notebook_payload.get("cells")
    if not isinstance(cells, list):
        raise AssertionError(f"cells must be list: {notebook_path}")

    matches: List[str] = []
    for cell_node in cast(List[object], cells):
        cell = cast(Dict[str, Any], cell_node) if isinstance(cell_node, dict) else None
        if cell is None:
            continue
        if cell.get("cell_type") != cell_type:
            continue
        source_node = cell.get("source")
        if not isinstance(source_node, list):
            continue
        source_text = "\n".join(str(line) for line in cast(List[object], source_node))
        if marker in source_text:
            matches.append(source_text)
    return matches


def _find_code_cell_sources(notebook_path: Path, marker: str) -> List[str]:
    """
    Find all matching code cell source texts.

    Args:
        notebook_path: Notebook path.
        marker: Marker text.

    Returns:
        Matching code cell source texts.
    """
    return _find_cell_sources(notebook_path, marker, "code")


def _find_code_cell_source(notebook_path: Path, marker: str) -> str:
    """
    Find one code cell by marker text.

    Args:
        notebook_path: Notebook path.
        marker: Marker text.

    Returns:
        Joined code source text.
    """
    for source_text in _find_code_cell_sources(notebook_path, marker):
        return source_text
    raise AssertionError(f"code cell marker not found: {marker}")


def _find_code_cell_index(notebook_path: Path, marker: str) -> int:
    """
    Find the index of one matching code cell.

    Args:
        notebook_path: Notebook path.
        marker: Marker text.

    Returns:
        Zero-based notebook cell index.
    """
    notebook_payload = _load_notebook(notebook_path)
    cells = notebook_payload.get("cells")
    if not isinstance(cells, list):
        raise AssertionError(f"cells must be list: {notebook_path}")

    for index, cell_node in enumerate(cast(List[object], cells)):
        cell = cast(Dict[str, Any], cell_node) if isinstance(cell_node, dict) else None
        if cell is None:
            continue
        if cell.get("cell_type") != "code":
            continue
        source_node = cell.get("source")
        if not isinstance(source_node, list):
            continue
        source_text = "\n".join(str(line) for line in cast(List[object], source_node))
        if marker in source_text:
            return index
    raise AssertionError(f"code cell marker not found: {marker}")


def _find_markdown_cell_source(notebook_path: Path, marker: str) -> str:
    """
    Find one markdown cell by marker text.

    Args:
        notebook_path: Notebook path.
        marker: Marker text.

    Returns:
        Joined markdown source text.
    """
    for source_text in _find_cell_sources(notebook_path, marker, "markdown"):
        return source_text
    raise AssertionError(f"markdown cell marker not found: {marker}")


def test_paper_workflow_notebook_entrypoints_bind_expected_scripts() -> None:
    """
    Verify notebook code cells bind expected script paths and args.

    Args:
        None.

    Returns:
        None.
    """
    pw00_constants = _find_code_cell_source(NOTEBOOK_PW00_PATH, "SCRIPT_PATH = REPO_ROOT")
    pw00_bootstrap = _find_code_cell_source(
        NOTEBOOK_PW00_PATH,
        'from scripts.notebook_runtime_common import resolve_notebook_model_cache_layout',
    )
    pw00_execute = _find_code_cell_source(NOTEBOOK_PW00_PATH, "COMMAND = [")
    pw01_constants = _find_code_cell_source(NOTEBOOK_PW01_PATH, "SCRIPT_PATH = REPO_ROOT")
    pw01_repo_bootstrap = _find_code_cell_source(
        NOTEBOOK_PW01_PATH,
        'from scripts.notebook_runtime_common import resolve_notebook_model_cache_layout',
    )
    pw01_execute = _find_code_cell_source(NOTEBOOK_PW01_PATH, "COMMAND = [")
    pw02_constants = _find_code_cell_source(NOTEBOOK_PW02_PATH, "SCRIPT_PATH = REPO_ROOT")
    pw02_bootstrap = _find_code_cell_source(
        NOTEBOOK_PW02_PATH,
        'from scripts.notebook_runtime_common import resolve_notebook_model_cache_layout',
    )
    pw02_execute = _find_code_cell_source(NOTEBOOK_PW02_PATH, "COMMAND = [")

    assert '"PW00_Paper_Eval_Family_Manifest.py"' in pw00_constants
    assert 'DRIVE_MODELS_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "Models"' in pw00_constants
    assert 'PERSISTENT_HF_ROOT = DRIVE_MODELS_ROOT / "Huggingface"' in pw00_constants
    assert 'LOCAL_HF_HOME = REPO_ROOT / "huggingface_cache"' in pw00_constants
    assert 'snapshot_download(' not in pw00_bootstrap
    assert '"--drive-project-root"' in pw00_execute
    assert '"--family-id"' in pw00_execute
    assert '"--prompt-file"' in pw00_execute
    assert '"--seed-list"' in pw00_execute
    assert '"--source-shard-count"' in pw00_execute

    assert '"PW01_Source_Event_Shards.py"' in pw01_constants
    assert 'DRIVE_MODELS_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "Models"' in pw01_constants
    assert 'PERSISTENT_HF_ROOT = DRIVE_MODELS_ROOT / "Huggingface"' in pw01_constants
    assert 'LOCAL_HF_HOME = REPO_ROOT / "huggingface_cache"' in pw01_constants
    assert 'SAMPLE_ROLE = "positive_source"' in pw01_constants
    assert '"planner_conditioned_control_negative": "control_negative"' in pw01_constants
    assert 'STAGE_01_WORKER_COUNT = 2' in pw01_constants
    assert 'MODEL_CACHE_LAYOUT = resolve_notebook_model_cache_layout(DRIVE_MOUNT_ROOT, REPO_ROOT, create_directories=True)' in pw01_repo_bootstrap
    assert '"model_cache_mode": "local_session_primary"' in pw01_repo_bootstrap
    assert '"persistent_hf_root_role": "compatibility_only"' in pw01_repo_bootstrap
    assert '"--drive-project-root"' in pw01_execute
    assert '"--family-id"' in pw01_execute
    assert '"--sample-role"' in pw01_execute
    assert 'SAMPLE_ROLE' in pw01_execute
    assert '"--shard-index"' in pw01_execute
    assert '"--shard-count"' in pw01_execute
    assert '"--stage-01-worker-count"' in pw01_execute
    assert 'str(STAGE_01_WORKER_COUNT)' in pw01_execute
    assert '"--bound-config-path"' in pw01_execute
    assert 'str(PW01_BOUND_CONFIG_PATH)' in pw01_execute
    assert '"--force-rerun"' in pw01_execute

    assert '"PW02_Source_Merge_And_Global_Thresholds.py"' in pw02_constants
    assert 'DRIVE_MODELS_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "Models"' in pw02_constants
    assert 'PERSISTENT_HF_ROOT = DRIVE_MODELS_ROOT / "Huggingface"' in pw02_constants
    assert 'LOCAL_HF_HOME = REPO_ROOT / "huggingface_cache"' in pw02_constants
    assert 'os.environ["HUGGINGFACE_HUB_CACHE"] = str(LOCAL_HF_HUB_CACHE)' in pw02_bootstrap
    assert 'snapshot_download(' not in pw02_bootstrap
    assert '"--drive-project-root"' in pw02_execute
    assert '"--family-id"' in pw02_execute


def test_pw01_notebook_passes_precheck_bound_config_to_execute_and_parallel_plan() -> None:
    """
    Verify PW01 notebook promotes the precheck-bound config into the execute
    command and the parallel extension example.

    Args:
        None.

    Returns:
        None.
    """
    pw01_precheck = _find_code_cell_source(NOTEBOOK_PW01_PATH, "PRECHECK_RESULTS = []")
    pw01_execute = _find_code_cell_source(NOTEBOOK_PW01_PATH, "COMMAND = [")
    pw01_parallel_plan = _find_code_cell_source(NOTEBOOK_PW01_PATH, "parallel_plan = []")

    assert "PW01_BOUND_CONFIG_PATH = PRECHECK_BOUND_CONFIG_PATH" in pw01_precheck
    assert "write_yaml_mapping(PW01_BOUND_CONFIG_PATH, PRECHECK_BOUND_CFG)" in pw01_precheck
    assert "STAGE_01_PREFLIGHT = detect_stage_01_preflight(PW01_BOUND_CONFIG_PATH)" in pw01_precheck
    assert '"sample_role 合法"' in pw01_precheck
    assert 'SAMPLE_ROLE in manifest_sample_roles' in pw01_precheck
    assert '"--bound-config-path"' in pw01_execute
    assert 'str(PW01_BOUND_CONFIG_PATH)' in pw01_execute
    assert '"--bound-config-path"' in pw01_parallel_plan
    assert 'str(PW01_BOUND_CONFIG_PATH)' in pw01_parallel_plan
    assert '"--sample-role"' in pw01_parallel_plan
    assert 'SAMPLE_ROLE' in pw01_parallel_plan


def test_pw01_notebook_restores_bootstrap_before_single_formal_precheck() -> None:
    """
    Verify PW01 notebook restores the bootstrap cell before one formal precheck.

    Args:
        None.

    Returns:
        None.
    """
    pw01_intro = _find_markdown_cell_source(NOTEBOOK_PW01_PATH, "用途：")
    pw01_parallel_markdown = _find_markdown_cell_source(NOTEBOOK_PW01_PATH, "扩展规则：")
    pw01_bootstrap = _find_code_cell_source(
        NOTEBOOK_PW01_PATH,
        'MODEL_CACHE_BOOTSTRAP = bootstrap_notebook_model_cache(',
    )
    pw01_precheck = _find_code_cell_source(NOTEBOOK_PW01_PATH, "PRECHECK_RESULTS = []")

    assert "formal 主流程角色为 positive_source 与 clean_negative" in pw01_intro
    assert "planner_conditioned_control_negative 仅作为 optional diagnostic cohort 按需执行" in pw01_intro
    assert "source_shards/positive/shard_xxxx、source_shards/negative/shard_xxxx 或 source_shards/control_negative/shard_xxxx" in pw01_parallel_markdown
    assert "planner_conditioned_control_negative 仅在需要 diagnostic pool 时按需补跑" in pw01_parallel_markdown

    assert 'from huggingface_hub import HfApi' in pw01_bootstrap
    assert 'bootstrap_notebook_model_cache' in pw01_bootstrap
    assert 'os.environ["HUGGINGFACE_HUB_CACHE"] = str(LOCAL_HF_HUB_CACHE)' in pw01_bootstrap
    assert 'MODEL_CACHE_BOOTSTRAP = bootstrap_notebook_model_cache(' in pw01_bootstrap
    assert 'MODEL_SNAPSHOT_PATH = str(MODEL_CACHE_BOOTSTRAP["local_snapshot_path"])' in pw01_bootstrap
    assert 'MODEL_DOWNLOAD_SUMMARY = dict(MODEL_CACHE_BOOTSTRAP["model_audit_summary"])' in pw01_bootstrap
    assert 'WEIGHT_DOWNLOAD_SUMMARY = collect_weight_summary(REPO_ROOT, CFG_OBJ)' in pw01_bootstrap
    assert '"snapshot_source": MODEL_CACHE_BOOTSTRAP["snapshot_source"]' in pw01_bootstrap
    assert '"model_source_binding": MODEL_CACHE_BOOTSTRAP["model_source_binding"]' in pw01_bootstrap
    assert 'build_directory_digest_summary(Path(MODEL_SNAPSHOT_PATH))' not in pw01_bootstrap
    assert 'print_json("model_cache_bootstrap", MODEL_CACHE_BOOTSTRAP)' in pw01_bootstrap
    assert 'ATTESTATION_BOOTSTRAP = ensure_attestation_env_bootstrap(' in pw01_bootstrap
    assert 'print_json("attestation_env_bootstrap", ATTESTATION_BOOTSTRAP)' in pw01_bootstrap
    assert 'run_checked(["nvidia-smi"])' in pw01_bootstrap

    assert len(_find_code_cell_sources(NOTEBOOK_PW01_PATH, "PRECHECK_RESULTS = []")) == 1
    assert _find_code_cell_index(
        NOTEBOOK_PW01_PATH,
        'MODEL_CACHE_BOOTSTRAP = bootstrap_notebook_model_cache(',
    ) < _find_code_cell_index(
        NOTEBOOK_PW01_PATH,
        "PRECHECK_RESULTS = []",
    )
    assert 'SOURCE_SPLIT_PLAN_PATH = FAMILY_ROOT / "manifests" / "source_split_plan.json"' in pw01_precheck
    assert 'SAMPLE_ROLE in manifest_sample_roles' in pw01_precheck
    assert 'persistent Huggingface 路径仅兼容保留' in pw01_precheck
    assert '模型 snapshot 来源为本地会话缓存' in pw01_precheck
    assert 'Path(str(MODEL_SNAPSHOT_PATH)).exists() and Path(str(MODEL_SNAPSHOT_PATH)).is_dir()' in pw01_precheck
    assert 'str(Path(str(MODEL_SNAPSHOT_PATH)).resolve()).startswith(str(LOCAL_HF_HOME.resolve()))' in pw01_precheck
    assert 'MODEL_DOWNLOAD_SUMMARY["binding_status"] = PRECHECK_MODEL_SOURCE_BINDING.get("binding_status", "<absent>")' in pw01_precheck
    assert 'Path(str(ATTESTATION_BOOTSTRAP.get("attestation_env_path", ""))).exists()' in pw01_precheck
    assert 'Path(str(ATTESTATION_BOOTSTRAP.get("attestation_env_info_path", ""))).exists()' in pw01_precheck
    assert 'nvidia_smi_result = subprocess.run(' in pw01_precheck


def test_pw01_notebook_wraps_command_with_gpu_peak_monitor_and_reads_shard_manifest_contract() -> None:
    """
    Verify that the PW01 notebook routes the shard command through the GPU
    peak wrapper and reads the formal shard manifest artifact after success.

    Args:
        None.

    Returns:
        None.
    """
    pw01_execute = _find_code_cell_source(NOTEBOOK_PW01_PATH, "COMMAND = [")

    assert 'GPU_PEAK_SCRIPT_PATH = REPO_ROOT / "scripts" / "gpu_session_peak.py"' in pw01_execute
    assert 'GPU_PEAK_SUMMARY_PATH = SHARD_ROOT / "artifacts" / "gpu_session_peak.json"' in pw01_execute
    assert 'MONITORED_COMMAND = [' in pw01_execute
    assert 'PW01_RESULT = subprocess.run(' in pw01_execute
    assert 'MONITORED_COMMAND,' in pw01_execute
    assert 'SHARD_ROOT = FAMILY_ROOT / "source_shards" / SAMPLE_ROLE_DIRNAME' in pw01_execute
    assert 'f"shard_{SHARD_INDEX:04d}"' in pw01_execute
    assert 'PW01_SHARD_MANIFEST_PATH = SHARD_ROOT / "shard_manifest.json"' in pw01_execute
    assert 'if not PW01_SHARD_MANIFEST_PATH.exists():' in pw01_execute
    assert 'PW01_SUMMARY = json.loads(PW01_SHARD_MANIFEST_PATH.read_text(encoding="utf-8"))' in pw01_execute
    assert 'shard_manifest_path={PW01_SHARD_MANIFEST_PATH} stdout={PW01_RESULT.stdout} stderr={PW01_RESULT.stderr}' in pw01_execute
    assert 'pw01_stdout_text = PW01_RESULT.stdout.strip()' not in pw01_execute
    assert 'PW01_SUMMARY = json.loads(pw01_stdout_text)' not in pw01_execute
    assert 'GPU_PEAK_SUMMARY = json.loads(GPU_PEAK_SUMMARY_PATH.read_text(encoding="utf-8"))' in pw01_execute
    assert 'print_json("gpu_session_peak_summary", GPU_PEAK_NOTEBOOK_SUMMARY)' in pw01_execute


def test_pw02_notebook_reads_summary_from_runtime_state() -> None:
    """
    Verify PW02 notebook binds the expected script and reads the runtime summary.

    Args:
        None.

    Returns:
        None.
    """
    pw02_execute = _find_code_cell_source(NOTEBOOK_PW02_PATH, "COMMAND = [")

    assert 'PW02_SUMMARY_PATH = FAMILY_ROOT / "runtime_state" / "pw02_summary.json"' in _find_code_cell_source(
        NOTEBOOK_PW02_PATH,
        "PW02_SUMMARY_PATH = FAMILY_ROOT",
    )
    assert '"--drive-project-root"' in pw02_execute
    assert '"--family-id"' in pw02_execute
    assert 'PW02_RESULT = subprocess.run(' in pw02_execute
    assert 'if not PW02_SUMMARY_PATH.exists():' in pw02_execute
    assert 'PW02_SUMMARY = json.loads(PW02_SUMMARY_PATH.read_text(encoding="utf-8"))' in pw02_execute
    assert 'summary 会显式记录 planner_conditioned_control_negative 的 cohort_status' in _find_markdown_cell_source(
        NOTEBOOK_PW02_PATH,
        'summary 会显式记录 planner_conditioned_control_negative 的 cohort_status',
    )


def test_pw00_and_pw02_notebooks_explain_formal_vs_optional_control_boundary() -> None:
    """
    Verify PW00 and PW02 notebook markdown explains the formal/optional cohort boundary.

    Args:
        None.

    Returns:
        None.
    """
    pw00_intro = _find_markdown_cell_source(NOTEBOOK_PW00_PATH, "用途：")
    pw02_intro = _find_markdown_cell_source(NOTEBOOK_PW02_PATH, "用途：")
    pw02_precheck_markdown = _find_markdown_cell_source(NOTEBOOK_PW02_PATH, "本单元只把")

    assert "formal 主流程所需的双 role source event grid" in pw00_intro
    assert "optional diagnostic cohort planner_conditioned_control_negative" in pw00_intro

    assert "formal 双 role shard（positive_source 与 clean_negative）" in pw02_intro
    assert "完整 planner_conditioned_control_negative cohort 时额外导出 optional diagnostic pool" in pw02_intro

    assert "positive_source 与 clean_negative 视为 PW02 formal 主流程硬依赖" in pw02_precheck_markdown
    assert "planner_conditioned_control_negative 缺失可接受" in pw02_precheck_markdown
    assert "若只完成部分 shard，PW02 CLI 会快速失败" in pw02_precheck_markdown


@pytest.mark.parametrize(
    ("script_relative_path", "expected_help_text"),
    [
        (
            "paper_workflow/scripts/PW00_Paper_Eval_Family_Manifest.py",
            "optional planner_conditioned_control_negative diagnostic cohort",
        ),
        (
            "paper_workflow/scripts/PW01_Source_Event_Shards.py",
            "Optional advanced diagnostic value: planner_conditioned_control_negative.",
        ),
        (
            "paper_workflow/scripts/PW02_Source_Merge_And_Global_Thresholds.py",
            "formal mainline requires positive_source and clean_negative only",
        ),
    ],
)
def test_paper_workflow_script_help_entrypoints(
    script_relative_path: str,
    expected_help_text: str,
) -> None:
    """
    Verify paper_workflow scripts expose help entrypoints.

    Args:
        script_relative_path: Script path relative to repository root.

    Returns:
        None.
    """
    script_path = REPO_ROOT / script_relative_path
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=REPO_ROOT,
        env=build_repo_import_subprocess_env(repo_root=REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined_output = f"{result.stdout}\n{result.stderr}"
    normalized_output = " ".join(combined_output.split())
    assert result.returncode == 0
    assert "usage:" in combined_output.lower()
    assert expected_help_text in normalized_output


def test_pw01_errors_when_family_manifest_missing(tmp_path: Path) -> None:
    """
    Verify PW01 fails clearly when family manifest is missing.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    with pytest.raises(FileNotFoundError, match="paper eval family manifest"):
        run_pw01_source_event_shard(
            drive_project_root=tmp_path / "drive",
            family_id="missing_family",
            shard_index=0,
            shard_count=1,
        )


def test_pw00_errors_when_prompt_file_missing(tmp_path: Path) -> None:
    """
    Verify PW00 fails clearly when prompt file is missing.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    with pytest.raises(FileNotFoundError, match="prompt file not found"):
        run_pw00_build_family_manifest(
            drive_project_root=tmp_path / "drive",
            family_id="missing_prompt_family",
            prompt_file=str(tmp_path / "missing_prompts.txt"),
            seed_list=[0],
            source_shard_count=1,
        )
