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
NOTEBOOK_PW03_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW03_Attack_Event_Shards.ipynb"
NOTEBOOK_PW04_PREPARE_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW04_Attack_Merge_And_Metrics - 1.Prepare.ipynb"
NOTEBOOK_PW04_QUALITY_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW04_Attack_Merge_And_Metrics - 2.Quality.ipynb"
NOTEBOOK_PW04_FINALIZE_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW04_Attack_Merge_And_Metrics - 3.Finalize.ipynb"
NOTEBOOK_PW05_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW05_Release_And_Signoff.ipynb"
README_PATH = REPO_ROOT / "paper_workflow" / "README.md"
PAPER_PILOT_PROMPT_PATH = REPO_ROOT / "prompts" / "paper_pilot_10.txt"


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


def test_notebook_runtime_common_exports_read_optional_json() -> None:
    """
    Verify notebook runtime common exports the optional JSON helper used by PW00.

    Args:
        None.

    Returns:
        None.
    """
    from scripts.notebook_runtime_common import read_optional_json

    assert callable(read_optional_json)


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
        'from scripts.notebook_runtime_common import ensure_attestation_env_bootstrap, load_yaml_mapping',
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
    pw03_constants = _find_code_cell_source(NOTEBOOK_PW03_PATH, "SCRIPT_PATH = REPO_ROOT")
    pw05_constants = _find_code_cell_source(NOTEBOOK_PW05_PATH, "SCRIPT_PATH = REPO_ROOT")

    assert '"PW00_Paper_Eval_Family_Manifest.py"' in pw00_constants
    assert 'HF_HOME = REPO_ROOT / "huggingface_cache"' in pw00_constants
    assert 'HF_HUB_CACHE = HF_HOME / "hub"' in pw00_constants
    assert 'TRANSFORMERS_CACHE = HF_HOME / "transformers"' in pw00_constants
    assert 'FAMILY_ID = "paper_eval_family_pilot_v1"' in pw00_constants
    assert 'PROMPT_FILE = "prompts/paper_pilot_10.txt"' in pw00_constants
    assert 'PW_BASE_CONFIG_PATH = "paper_workflow/configs/pw_base_pilot.yaml"' in pw00_constants
    assert 'SEED_LIST = [100, 101, 102, 103, 104, 105, 106, 107]' in pw00_constants
    assert 'SOURCE_SHARD_COUNT = 4' in pw00_constants
    assert 'ATTACK_SHARD_COUNT = 16' in pw00_constants
    assert 'run_checked([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], cwd=REPO_ROOT)' in pw00_bootstrap
    assert 'ensure_attestation_env_bootstrap(' in pw00_bootstrap
    assert 'snapshot_download(' not in pw00_bootstrap
    assert '"--drive-project-root"' in pw00_execute
    assert '"--family-id"' in pw00_execute
    assert '"--pw-base-config-path"' in pw00_execute
    assert 'PW_BASE_CONFIG_PATH' in pw00_execute
    assert '"--prompt-file"' in pw00_execute
    assert '"--seed-list"' in pw00_execute
    assert '"--source-shard-count"' in pw00_execute
    assert '"--attack-shard-count"' in pw00_execute
    assert '"--attack-shard-count"' in pw00_execute

    assert '"PW01_Source_Event_Shards.py"' in pw01_constants
    assert 'DRIVE_MODELS_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "Models"' in pw01_constants
    assert 'PERSISTENT_HF_ROOT = DRIVE_MODELS_ROOT / "Huggingface"' in pw01_constants
    assert 'LOCAL_HF_HOME = REPO_ROOT / "huggingface_cache"' in pw01_constants
    assert 'FAMILY_ID = "paper_eval_family_pilot_v1"' in pw01_constants
    assert 'SAMPLE_ROLE = "positive_source"' in pw01_constants
    assert '"planner_conditioned_control_negative": "control_negative"' in pw01_constants
    assert 'SHARD_COUNT = 4' in pw01_constants
    assert 'PW01_WORKER_COUNT = 2' in pw01_constants
    assert 'MODEL_CACHE_LAYOUT = resolve_notebook_model_cache_layout(DRIVE_MOUNT_ROOT, REPO_ROOT, create_directories=True)' in pw01_repo_bootstrap
    assert '"model_cache_mode": "local_session_primary"' in pw01_repo_bootstrap
    assert '"persistent_hf_root_role": "compatibility_only"' in pw01_repo_bootstrap
    assert '"--drive-project-root"' in pw01_execute
    assert '"--family-id"' in pw01_execute
    assert '"--sample-role"' in pw01_execute
    assert 'SAMPLE_ROLE' in pw01_execute
    assert '"--shard-index"' in pw01_execute
    assert '"--shard-count"' in pw01_execute
    assert '"--pw01-worker-count"' in pw01_execute
    assert 'str(PW01_WORKER_COUNT)' in pw01_execute
    assert '"--bound-config-path"' in pw01_execute
    assert 'str(PW01_BOUND_CONFIG_PATH)' in pw01_execute
    assert '"--force-rerun"' in pw01_execute

    assert '"PW02_Source_Merge_And_Global_Thresholds.py"' in pw02_constants
    assert 'DRIVE_MODELS_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "Models"' in pw02_constants
    assert 'PERSISTENT_HF_ROOT = DRIVE_MODELS_ROOT / "Huggingface"' in pw02_constants
    assert 'LOCAL_HF_HOME = REPO_ROOT / "huggingface_cache"' in pw02_constants
    assert 'FAMILY_ID = "paper_eval_family_pilot_v1"' in pw02_constants
    assert 'os.environ["HUGGINGFACE_HUB_CACHE"] = str(LOCAL_HF_HUB_CACHE)' in pw02_bootstrap
    assert 'snapshot_download(' not in pw02_bootstrap
    assert '"--drive-project-root"' in pw02_execute
    assert '"--family-id"' in pw02_execute

    assert '"PW03_Attack_Event_Shards.py"' in pw03_constants
    assert 'FAMILY_ID = "paper_eval_family_pilot_v1"' in pw03_constants
    assert 'ATTACK_SHARD_COUNT = 16' in pw03_constants
    assert 'ATTACK_LOCAL_WORKER_COUNT = 4' in pw03_constants
    assert 'ATTACK_FAMILY_ALLOWLIST = None' in pw03_constants

    assert '"PW05_Release_And_Signoff.py"' in pw05_constants
    assert 'FAMILY_ID = "paper_eval_family_pilot_v1"' in pw05_constants
    assert 'FORCE_RERUN = False' in pw05_constants


@pytest.mark.parametrize(
    (
        "notebook_path",
        "prepare_stage_name",
        "archive_stage_name",
        "prepare_tokens",
        "archive_tokens",
    ),
    [
        (
            NOTEBOOK_PW00_PATH,
            "PW00_Paper_Eval_Family_Manifest",
            "PW00_Paper_Eval_Family_Manifest",
            [],
            [],
        ),
        (
            NOTEBOOK_PW01_PATH,
            "PW01_Source_Event_Shards",
            "PW01_Source_Event_Shards",
            ["sample_role=SAMPLE_ROLE", "shard_index=SHARD_INDEX", "shard_count=SHARD_COUNT"],
            ["sample_role=SAMPLE_ROLE", "shard_index=SHARD_INDEX", "shard_count=SHARD_COUNT"],
        ),
        (
            NOTEBOOK_PW02_PATH,
            "PW02_Source_Merge_And_Global_Thresholds",
            "PW02_Source_Merge_And_Global_Thresholds",
            [],
            [],
        ),
        (
            NOTEBOOK_PW03_PATH,
            "PW03_Attack_Event_Shards",
            "PW03_Attack_Event_Shards",
            ["shard_index=ATTACK_SHARD_INDEX", "shard_count=ATTACK_SHARD_COUNT"],
            ["shard_index=ATTACK_SHARD_INDEX", "shard_count=ATTACK_SHARD_COUNT"],
        ),
        (
            NOTEBOOK_PW04_PREPARE_PATH,
            "PW04_Attack_Merge_And_Metrics_prepare",
            "PW04_Attack_Merge_And_Metrics_prepare",
            [],
            [],
        ),
        (
            NOTEBOOK_PW04_QUALITY_PATH,
            "PW04_Attack_Merge_And_Metrics_quality_shard",
            "PW04_Attack_Merge_And_Metrics_quality_shard",
            ["shard_index=QUALITY_SHARD_INDEX"],
            ["shard_index=QUALITY_SHARD_INDEX"],
        ),
        (
            NOTEBOOK_PW04_FINALIZE_PATH,
            "PW04_Attack_Merge_And_Metrics_finalize",
            "PW04_Attack_Merge_And_Metrics_finalize",
            [],
            [],
        ),
        (
            NOTEBOOK_PW05_PATH,
            "PW05_Release_And_Signoff",
            "PW05_Release_And_Signoff",
            [],
            [],
        ),
    ],
)
def test_paper_workflow_notebooks_enable_local_runtime_bundle_mode(
    notebook_path: Path,
    prepare_stage_name: str,
    archive_stage_name: str,
    prepare_tokens: List[str],
    archive_tokens: List[str],
) -> None:
    """
    Verify notebooks expose local runtime variables and bundle prepare/archive hooks.

    Args:
        notebook_path: Notebook path.
        prepare_stage_name: Expected prepare stage name.
        archive_stage_name: Expected archive stage name.
        prepare_tokens: Additional prepare-call tokens.
        archive_tokens: Additional archive-call tokens.

    Returns:
        None.
    """
    constants_source = _find_code_cell_source(notebook_path, "LOCAL_RUNTIME_ENABLED = True")
    prepare_source = _find_code_cell_source(notebook_path, "prepare_local_runtime_for_stage(")
    archive_source = _find_code_cell_source(notebook_path, "archive_local_runtime_for_stage(")

    assert 'LOCAL_PROJECT_ROOT = Path("/content/CEG_WM_PaperWorkflow")' in constants_source
    assert 'PERSISTENT_DRIVE_PROJECT_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "CEG_WM_PaperWorkflow"' in constants_source
    assert 'DRIVE_BUNDLE_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "CEG_WM_PaperWorkflow_Bundles"' in constants_source
    assert 'if LOCAL_RUNTIME_ENABLED:' in constants_source
    assert 'DRIVE_PROJECT_ROOT = LOCAL_PROJECT_ROOT' in constants_source
    assert 'DRIVE_PROJECT_ROOT = PERSISTENT_DRIVE_PROJECT_ROOT' in constants_source

    assert 'from paper_workflow.scripts.pw_local_runtime import prepare_local_runtime_for_stage' in prepare_source
    assert f'stage_name="{prepare_stage_name}"' in prepare_source
    assert 'local_project_root=LOCAL_PROJECT_ROOT' in prepare_source
    assert 'drive_bundle_root=DRIVE_BUNDLE_ROOT' in prepare_source
    assert 'clean_before_run=True' in prepare_source
    assert 'if LOCAL_RUNTIME_ENABLED:' in prepare_source
    for token in prepare_tokens:
        assert token in prepare_source

    assert 'from paper_workflow.scripts.pw_local_runtime import archive_local_runtime_for_stage' in archive_source
    if notebook_path in {NOTEBOOK_PW04_PREPARE_PATH, NOTEBOOK_PW04_QUALITY_PATH, NOTEBOOK_PW04_FINALIZE_PATH}:
        assert 'stage_name=PW04_ARCHIVE_STAGE_NAME' in archive_source
    else:
        assert f'stage_name="{archive_stage_name}"' in archive_source
    assert 'local_project_root=LOCAL_PROJECT_ROOT' in archive_source
    assert 'drive_bundle_root=DRIVE_BUNDLE_ROOT' in archive_source
    assert 'clean_after_archive=False' in archive_source
    assert 'if LOCAL_RUNTIME_ENABLED:' in archive_source
    for token in archive_tokens:
        assert token in archive_source


def test_pw00_notebook_has_no_legacy_attestation_bootstrap_call() -> None:
    """
    Verify the PW00 notebook no longer contains the legacy attestation bootstrap call.

    Args:
        None.

    Returns:
        None.
    """
    notebook_text = NOTEBOOK_PW00_PATH.read_text(encoding="utf-8")

    assert 'notebook_name=NOTEBOOK_NAME' not in notebook_text
    assert 'ensure_attestation_env_bootstrap(CONFIG_PATH, DRIVE_PROJECT_ROOT' not in notebook_text


def test_pw00_notebook_uses_current_attestation_bootstrap_fields() -> None:
    """
    Verify the PW00 notebook uses the current attestation bootstrap field names.

    Args:
        None.

    Returns:
        None.
    """
    constants_source = _find_code_cell_source(NOTEBOOK_PW00_PATH, "LOCAL_RUNTIME_ENABLED = True")
    bootstrap_source = _find_code_cell_source(
        NOTEBOOK_PW00_PATH,
        "ATTESTATION_BOOTSTRAP = ensure_attestation_env_bootstrap(",
    )
    precheck_source = _find_code_cell_source(NOTEBOOK_PW00_PATH, "precheck_results = []")

    assert 'ATTESTATION_BOOTSTRAP["attestation_env_root"]' not in precheck_source
    assert 'ATTESTATION_BOOTSTRAP["attestation_env_snapshot"]' not in precheck_source
    assert 'record_precheck("attestation_env_root 存在"' not in precheck_source
    assert 'record_precheck("attestation_env_snapshot 存在"' not in precheck_source

    assert 'ATTESTATION_BOOTSTRAP.get("attestation_env_path")' in precheck_source
    assert 'ATTESTATION_BOOTSTRAP.get("attestation_env_info_path")' in precheck_source
    assert 'attestation_env 文件存在' in precheck_source
    assert 'attestation_env_info 文件存在' in precheck_source

    assert 'PERSISTENT_DRIVE_PROJECT_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "CEG_WM_PaperWorkflow"' in constants_source
    assert 'ATTESTATION_PROJECT_ROOT = PERSISTENT_DRIVE_PROJECT_ROOT' in constants_source
    assert 'ensure_attestation_env_bootstrap(' in bootstrap_source
    assert '    ATTESTATION_PROJECT_ROOT,' in bootstrap_source


def test_pw00_notebook_passes_seed_list_as_single_argument() -> None:
    """
    Verify the PW00 notebook passes seed_list as one CLI argument string.

    Args:
        None.

    Returns:
        None.
    """
    constants_source = _find_code_cell_source(NOTEBOOK_PW00_PATH, 'PROMPT_FILE = "prompts/paper_pilot_10.txt"')
    execute_source = _find_code_cell_source(NOTEBOOK_PW00_PATH, "COMMAND = [")

    assert 'SEED_LIST = [100, 101, 102, 103, 104, 105, 106, 107]' in constants_source
    assert '"--seed-list"' in execute_source
    assert 'json.dumps(SEED_LIST, ensure_ascii=False)' in execute_source
    assert '*(str(seed) for seed in SEED_LIST)' not in execute_source
    assert '"--seed-list",\n    *(str(seed) for seed in SEED_LIST)' not in execute_source


def test_pw00_notebook_passes_repo_import_env_to_subprocess() -> None:
    """
    Verify the PW00 notebook passes the repo import environment to its subprocess.

    Args:
        None.

    Returns:
        None.
    """
    source = _find_code_cell_source(NOTEBOOK_PW00_PATH, "COMMAND = [")

    assert "build_repo_import_subprocess_env" in source
    assert "PW00_SUBPROCESS_ENV = build_repo_import_subprocess_env(repo_root=REPO_ROOT)" in source
    assert "env=PW00_SUBPROCESS_ENV" in source


def test_pw00_pw01_pw02_pw03_pw04_pw05_notebooks_write_runtime_diagnostics_before_archive() -> None:
    """
    Verify all PW notebooks write runtime diagnostics before bundle archive.

    Args:
        None.

    Returns:
        None.
    """
    pw00_execute = _find_code_cell_source(NOTEBOOK_PW00_PATH, "COMMAND = [")
    pw01_execute = _find_code_cell_source(NOTEBOOK_PW01_PATH, "COMMAND = [")
    pw02_execute = _find_code_cell_source(NOTEBOOK_PW02_PATH, "COMMAND = [")
    pw03_execute = _find_code_cell_source(NOTEBOOK_PW03_PATH, "COMMAND = [")
    pw04_prepare_execute = _find_code_cell_source(NOTEBOOK_PW04_PREPARE_PATH, "build_pw04_command(")
    pw04_quality_execute = _find_code_cell_source(NOTEBOOK_PW04_QUALITY_PATH, "build_pw04_command(")
    pw04_finalize_execute = _find_code_cell_source(NOTEBOOK_PW04_FINALIZE_PATH, "build_pw04_command(")
    pw05_execute = _find_code_cell_source(NOTEBOOK_PW05_PATH, "COMMAND = [")

    for source_text, diagnostics_name in [
        (pw00_execute, "pw00_runtime_diagnostics.json"),
        (pw01_execute, "pw01_"),
        (pw02_execute, "pw02_runtime_diagnostics.json"),
        (pw03_execute, "pw03_attack_shard_"),
        (pw04_prepare_execute, "pw04_prepare_runtime_diagnostics.json"),
        (pw04_quality_execute, "pw04_quality_shard_"),
        (pw04_finalize_execute, "pw04_finalize_runtime_diagnostics.json"),
        (pw05_execute, "pw05_runtime_diagnostics.json"),
    ]:
        assert 'from datetime import datetime, timezone' in source_text
        assert 'import time' in source_text
        assert 'RUN_STARTED_AT_UTC = datetime.now(timezone.utc).isoformat()' in source_text
        assert 'RUN_FINISHED_AT_UTC = datetime.now(timezone.utc).isoformat()' in source_text
        assert 'RUN_ELAPSED_SECONDS = time.perf_counter() - RUN_STARTED_AT_MONOTONIC' in source_text
        assert 'build_stage_runtime_diagnostics_payload' in source_text
        assert 'build_stage_runtime_workload_summary' in source_text
        assert 'write_stage_runtime_diagnostics' in source_text
        assert 'count_summary=' in source_text
        assert 'workload_summary=' in source_text
        assert diagnostics_name in source_text
        assert source_text.index('write_stage_runtime_diagnostics(') < source_text.index('archive_local_runtime_for_stage(')


@pytest.mark.parametrize(
    "notebook_path",
    [NOTEBOOK_PW00_PATH, NOTEBOOK_PW01_PATH, NOTEBOOK_PW03_PATH],
)
def test_notebooks_use_persistent_attestation_project_root(notebook_path: Path) -> None:
    """
    Verify attestation bootstrap notebooks bind a persistent Drive root that is
    distinct from the local runtime root.

    Args:
        notebook_path: Notebook path.

    Returns:
        None.
    """
    constants_source = _find_code_cell_source(notebook_path, "LOCAL_RUNTIME_ENABLED = True")
    bootstrap_source = _find_code_cell_source(
        notebook_path,
        "ATTESTATION_BOOTSTRAP = ensure_attestation_env_bootstrap(",
    )

    assert 'PERSISTENT_DRIVE_PROJECT_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "CEG_WM_PaperWorkflow"' in constants_source
    assert 'ATTESTATION_PROJECT_ROOT = PERSISTENT_DRIVE_PROJECT_ROOT' in constants_source
    assert 'DRIVE_PROJECT_ROOT = LOCAL_PROJECT_ROOT' in constants_source
    assert 'DRIVE_PROJECT_ROOT = PERSISTENT_DRIVE_PROJECT_ROOT' in constants_source
    assert '    ATTESTATION_PROJECT_ROOT,' in bootstrap_source
    assert '    DRIVE_PROJECT_ROOT,' not in bootstrap_source


@pytest.mark.parametrize(
    "notebook_path",
    [NOTEBOOK_PW00_PATH, NOTEBOOK_PW01_PATH, NOTEBOOK_PW03_PATH],
)
def test_attestation_project_root_is_not_local_runtime_root(notebook_path: Path) -> None:
    """
    Verify the attestation root remains bound to persistent Drive storage rather
    than the local runtime root.

    Args:
        notebook_path: Notebook path.

    Returns:
        None.
    """
    constants_source = _find_code_cell_source(notebook_path, "LOCAL_RUNTIME_ENABLED = True")

    assert 'LOCAL_PROJECT_ROOT = Path("/content/CEG_WM_PaperWorkflow")' in constants_source
    assert 'PERSISTENT_DRIVE_PROJECT_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "CEG_WM_PaperWorkflow"' in constants_source
    assert 'ATTESTATION_PROJECT_ROOT = PERSISTENT_DRIVE_PROJECT_ROOT' in constants_source
    assert 'ATTESTATION_PROJECT_ROOT = LOCAL_PROJECT_ROOT' not in constants_source


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
    pw01_parallel_plan = _find_code_cell_source(NOTEBOOK_PW01_PATH, 'print("[PW01 后续并行运行说明]")')

    assert "PW01_BOUND_CONFIG_PATH = PRECHECK_BOUND_CONFIG_PATH" in pw01_precheck
    assert "write_yaml_mapping(PW01_BOUND_CONFIG_PATH, PRECHECK_BOUND_CFG)" in pw01_precheck
    assert "PW01_PREFLIGHT = detect_pw01_preflight(PW01_BOUND_CONFIG_PATH)" in pw01_precheck
    assert '"PW01 preflight"' in pw01_precheck
    assert '"pw01_preflight": PW01_PREFLIGHT' in pw01_precheck
    assert "detect_stage_01_preflight" not in pw01_precheck
    assert '"sample_role 合法"' in pw01_precheck
    assert 'SAMPLE_ROLE in manifest_sample_roles' in pw01_precheck
    assert '"--bound-config-path"' in pw01_execute
    assert 'str(PW01_BOUND_CONFIG_PATH)' in pw01_execute
    assert '当前 shard 设置：SHARD_COUNT = {SHARD_COUNT}' in pw01_parallel_plan
    assert '当前 notebook 只运行：{SAMPLE_ROLE} shard {SHARD_INDEX}' in pw01_parallel_plan
    assert 'PW02 运行条件：positive_source 与 clean_negative 的全部 shard 均完成后，运行 PW02 一次。' in pw01_parallel_plan


def test_pw00_notebook_exposes_independent_attack_shard_count_controls() -> None:
    """
    Verify the PW00 notebook exposes independent attack shard controls.

    Args:
        None.

    Returns:
        None.
    """
    pw00_constants = _find_code_cell_source(NOTEBOOK_PW00_PATH, 'PROMPT_FILE = "prompts/paper_pilot_10.txt"')
    pw00_precheck = _find_code_cell_source(NOTEBOOK_PW00_PATH, "precheck_results = []")
    pw00_execute = _find_code_cell_source(NOTEBOOK_PW00_PATH, "COMMAND = [")
    pw00_output_check = _find_code_cell_source(NOTEBOOK_PW00_PATH, "output_check = {")
    pw00_parallel_markdown = _find_markdown_cell_source(NOTEBOOK_PW00_PATH, "扩展规则：")
    pw00_parallel_guide = _find_code_cell_source(NOTEBOOK_PW00_PATH, 'print("[后续并行运行说明]")')

    assert 'FAMILY_ID = "paper_eval_family_pilot_v1"' in pw00_constants
    assert 'PROMPT_FILE = "prompts/paper_pilot_10.txt"' in pw00_constants
    assert 'PW_BASE_CONFIG_PATH = "paper_workflow/configs/pw_base_pilot.yaml"' in pw00_constants
    assert 'SEED_LIST = [100, 101, 102, 103, 104, 105, 106, 107]' in pw00_constants
    assert 'SOURCE_SHARD_COUNT = 4' in pw00_constants
    assert 'ATTACK_SHARD_COUNT = 16' in pw00_constants
    assert 'pilot 默认冻结为 16' in pw00_constants
    assert 'RESOLVED_ATTACK_SHARD_COUNT = SOURCE_SHARD_COUNT if ATTACK_SHARD_COUNT is None else ATTACK_SHARD_COUNT' in pw00_constants
    assert 'PROJECT_ROOT_PRECHECK_LABEL = "项目运行根目录存在" if LOCAL_RUNTIME_ENABLED else "Drive 项目根目录存在"' in pw00_precheck
    assert 'record_precheck("prompt 文件存在"' in pw00_precheck
    assert 'record_precheck("attestation 持久根目录存在"' in pw00_precheck
    assert 'ATTESTATION_BOOTSTRAP.get("attestation_env_path")' in pw00_precheck
    assert 'ATTESTATION_BOOTSTRAP.get("attestation_env_info_path")' in pw00_precheck
    assert 'record_precheck("attestation_env 文件存在"' in pw00_precheck
    assert 'record_precheck("attestation_env_info 文件存在"' in pw00_precheck
    assert 'record_precheck("attestation_env_root 存在"' not in pw00_precheck
    assert 'record_precheck("attestation_env_snapshot 存在"' not in pw00_precheck
    assert 'print_json("pw00_precheck", precheck_results)' in pw00_precheck
    assert 'failed_prechecks = [item for item in precheck_results if not item["passed"]]' in pw00_precheck
    assert '"--pw-base-config-path"' in pw00_execute
    assert 'PW_BASE_CONFIG_PATH' in pw00_execute
    assert '"--seed-list"' in pw00_execute
    assert 'json.dumps(SEED_LIST, ensure_ascii=False)' in pw00_execute
    assert '*(str(seed) for seed in SEED_LIST)' not in pw00_execute
    assert '"--attack-shard-count"' in pw00_execute
    assert 'str(RESOLVED_ATTACK_SHARD_COUNT)' in pw00_execute
    assert 'ATTACK_SHARD_PLAN_PATH = Path(PW00_SUMMARY["attack_shard_plan_path"])' in pw00_output_check
    assert '"attack_shard_plan": {"path": str(ATTACK_SHARD_PLAN_PATH), "exists": ATTACK_SHARD_PLAN_PATH.exists()}' in pw00_output_check
    assert 'SOURCE_SHARD_COUNT 控制 source shard' in pw00_parallel_markdown
    assert 'ATTACK_SHARD_COUNT 控制 attack shard' in pw00_parallel_markdown
    assert '若未显式提供 ATTACK_SHARD_COUNT，则默认回退到 SOURCE_SHARD_COUNT' in pw00_parallel_markdown
    assert 'PW01 需要设置：SHARD_COUNT = {SOURCE_SHARD_COUNT}' in pw00_parallel_guide
    assert 'PW03 需要设置：ATTACK_SHARD_COUNT = {RESOLVED_ATTACK_SHARD_COUNT}' in pw00_parallel_guide
    assert 'PW04 运行顺序：prepare 一次 → quality_shard 全部完成 → finalize 一次。' in pw00_parallel_guide
    assert 'PW05 运行条件：PW04 finalize 完成后，运行 PW05 一次。' in pw00_parallel_guide


def test_paper_pilot_prompt_file_has_ten_non_empty_prompts_in_five_categories() -> None:
    """
    Verify the pilot prompt file exists and freezes ten prompts across five categories.

    Args:
        None.

    Returns:
        None.
    """
    prompt_lines = [
        line.strip()
        for line in PAPER_PILOT_PROMPT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    category_counts: Dict[str, int] = {}
    expected_category_counts = {
        "still_life_low_texture": 2,
        "high_texture_object": 2,
        "indoor_scene": 2,
        "outdoor_nature": 2,
        "structured_geometry": 2,
    }

    assert PAPER_PILOT_PROMPT_PATH.exists()
    assert len(prompt_lines) == 10

    for prompt_line in prompt_lines:
        category, separator, prompt_text = prompt_line.partition(":")
        assert separator == ":"
        assert prompt_text.strip()
        category_counts[category.strip()] = category_counts.get(category.strip(), 0) + 1

    assert category_counts == expected_category_counts


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
    assert '    ATTESTATION_PROJECT_ROOT,' in pw01_bootstrap
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
    assert 'PROJECT_ROOT_PRECHECK_LABEL = "项目运行根目录存在" if LOCAL_RUNTIME_ENABLED else "Drive 项目根目录存在"' in pw01_precheck
    assert 'record_precheck(PROJECT_ROOT_PRECHECK_LABEL, DRIVE_PROJECT_ROOT.exists(), str(DRIVE_PROJECT_ROOT))' in pw01_precheck
    assert 'record_precheck("attestation 持久根目录存在", ATTESTATION_PROJECT_ROOT.exists(), str(ATTESTATION_PROJECT_ROOT))' in pw01_precheck
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


def test_paper_workflow_all_notebook_script_paths_use_existing_wrapper_entrypoints() -> None:
    """
    Verify all PW00-PW05 notebooks bind existing uppercase wrapper entrypoints.

    Args:
        None.

    Returns:
        None.
    """
    notebook_expectations = [
        (NOTEBOOK_PW00_PATH, "PW00_Paper_Eval_Family_Manifest.py"),
        (NOTEBOOK_PW01_PATH, "PW01_Source_Event_Shards.py"),
        (NOTEBOOK_PW02_PATH, "PW02_Source_Merge_And_Global_Thresholds.py"),
        (NOTEBOOK_PW03_PATH, "PW03_Attack_Event_Shards.py"),
        (NOTEBOOK_PW04_PREPARE_PATH, "PW04_Attack_Merge_And_Metrics.py"),
        (NOTEBOOK_PW04_QUALITY_PATH, "PW04_Attack_Merge_And_Metrics.py"),
        (NOTEBOOK_PW04_FINALIZE_PATH, "PW04_Attack_Merge_And_Metrics.py"),
        (NOTEBOOK_PW05_PATH, "PW05_Release_And_Signoff.py"),
    ]

    for notebook_path, script_name in notebook_expectations:
        constants_source = _find_code_cell_source(notebook_path, "SCRIPT_PATH = REPO_ROOT")
        assert f'"{script_name}"' in constants_source
        assert (REPO_ROOT / "paper_workflow" / "scripts" / script_name).exists()


def test_paper_workflow_wrapper_files_exist() -> None:
    """
    Verify all paper_workflow wrapper files exist on disk.

    Args:
        None.

    Returns:
        None.
    """
    wrapper_paths = [
        REPO_ROOT / "paper_workflow" / "scripts" / "PW00_Paper_Eval_Family_Manifest.py",
        REPO_ROOT / "paper_workflow" / "scripts" / "PW01_Source_Event_Shards.py",
        REPO_ROOT / "paper_workflow" / "scripts" / "PW02_Source_Merge_And_Global_Thresholds.py",
        REPO_ROOT / "paper_workflow" / "scripts" / "PW03_Attack_Event_Shards.py",
        REPO_ROOT / "paper_workflow" / "scripts" / "PW04_Attack_Merge_And_Metrics.py",
        REPO_ROOT / "paper_workflow" / "scripts" / "PW05_Release_And_Signoff.py",
    ]

    for wrapper_path in wrapper_paths:
        assert wrapper_path.exists(), wrapper_path


def test_paper_workflow_readme_and_pw00_command_template_use_wrapper_entrypoints() -> None:
    """
    Verify README and the PW00 command template use wrapper entrypoints only.

    Args:
        None.

    Returns:
        None.
    """
    readme_text = README_PATH.read_text(encoding="utf-8")
    for old_name in [
        "pw00_paper_eval_family_manifest.py",
        "pw01_source_event_shards.py",
        "pw02_source_merge_and_global_thresholds.py",
    ]:
        assert old_name not in readme_text

    for wrapper_relative_path in [
        "paper_workflow/scripts/PW00_Paper_Eval_Family_Manifest.py",
        "paper_workflow/scripts/PW01_Source_Event_Shards.py",
        "paper_workflow/scripts/PW02_Source_Merge_And_Global_Thresholds.py",
        "paper_workflow/scripts/PW03_Attack_Event_Shards.py",
        "paper_workflow/scripts/PW04_Attack_Merge_And_Metrics.py",
        "paper_workflow/scripts/PW05_Release_And_Signoff.py",
    ]:
        assert wrapper_relative_path in readme_text

    pw00_constants = _find_code_cell_source(NOTEBOOK_PW00_PATH, "SCRIPT_PATH = REPO_ROOT")
    pw01_constants = _find_code_cell_source(NOTEBOOK_PW01_PATH, "SCRIPT_PATH = REPO_ROOT")
    pw02_constants = _find_code_cell_source(NOTEBOOK_PW02_PATH, "SCRIPT_PATH = REPO_ROOT")
    assert '"PW00_Paper_Eval_Family_Manifest.py"' in pw00_constants
    assert '"PW01_Source_Event_Shards.py"' in pw01_constants
    assert '"PW02_Source_Merge_And_Global_Thresholds.py"' in pw02_constants


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
        (
            "paper_workflow/scripts/PW03_Attack_Event_Shards.py",
            "finalized positive source pool",
        ),
        (
            "paper_workflow/scripts/PW04_Attack_Merge_And_Metrics.py",
            "optional tail estimation exports",
        ),
        (
            "paper_workflow/scripts/PW05_Release_And_Signoff.py",
            "release package and signoff outputs",
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
