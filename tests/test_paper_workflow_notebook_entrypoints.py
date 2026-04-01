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


def _find_code_cell_source(notebook_path: Path, marker: str) -> str:
    """
    Find one code cell by marker text.

    Args:
        notebook_path: Notebook path.
        marker: Marker text.

    Returns:
        Joined code source text.
    """
    notebook_payload = _load_notebook(notebook_path)
    cells = notebook_payload.get("cells")
    if not isinstance(cells, list):
        raise AssertionError(f"cells must be list: {notebook_path}")
    for cell_node in cast(List[object], cells):
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
            return source_text
    raise AssertionError(f"code cell marker not found: {marker}")


def test_paper_workflow_notebook_entrypoints_bind_expected_scripts() -> None:
    """
    Verify notebook code cells bind expected script paths and args.

    Args:
        None.

    Returns:
        None.
    """
    pw00_constants = _find_code_cell_source(NOTEBOOK_PW00_PATH, "SCRIPT_PATH = REPO_ROOT")
    pw00_execute = _find_code_cell_source(NOTEBOOK_PW00_PATH, "COMMAND = [")
    pw01_constants = _find_code_cell_source(NOTEBOOK_PW01_PATH, "SCRIPT_PATH = REPO_ROOT")
    pw01_execute = _find_code_cell_source(NOTEBOOK_PW01_PATH, "COMMAND = [")

    assert '"PW00_Paper_Eval_Family_Manifest.py"' in pw00_constants
    assert '"--drive-project-root"' in pw00_execute
    assert '"--family-id"' in pw00_execute
    assert '"--prompt-file"' in pw00_execute
    assert '"--seed-list"' in pw00_execute
    assert '"--source-shard-count"' in pw00_execute

    assert '"PW01_Source_Event_Shards.py"' in pw01_constants
    assert '"--drive-project-root"' in pw01_execute
    assert '"--family-id"' in pw01_execute
    assert '"--shard-index"' in pw01_execute
    assert '"--shard-count"' in pw01_execute
    assert '"--force-rerun"' in pw01_execute


@pytest.mark.parametrize(
    "script_relative_path",
    [
        "paper_workflow/scripts/PW00_Paper_Eval_Family_Manifest.py",
        "paper_workflow/scripts/PW01_Source_Event_Shards.py",
    ],
)
def test_paper_workflow_script_help_entrypoints(script_relative_path: str) -> None:
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
    assert result.returncode == 0
    assert "usage:" in combined_output.lower()


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
