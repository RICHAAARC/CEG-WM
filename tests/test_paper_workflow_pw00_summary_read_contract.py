"""
File purpose: Contract test for PW00 notebook summary read behavior.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, cast

from scripts.notebook_runtime_common import REPO_ROOT


NOTEBOOK_PW00_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW00_Paper_Eval_Family_Manifest.ipynb"


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


def test_pw00_notebook_reads_formal_summary_file_instead_of_stdout_tail() -> None:
    """
    Verify PW00 notebook reads the formal summary artifact.

    Args:
        None.

    Returns:
        None.
    """
    pw00_execute = _find_code_cell_source(NOTEBOOK_PW00_PATH, "PW00_RESULT = subprocess.run(")

    assert 'FAMILY_ROOT = DRIVE_PROJECT_ROOT / "paper_workflow" / "families" / FAMILY_ID' in pw00_execute
    assert 'PW00_SUMMARY_PATH = FAMILY_ROOT / "runtime_state" / "pw00_summary.json"' in pw00_execute
    assert 'if not PW00_SUMMARY_PATH.exists():' in pw00_execute
    assert 'PW00_SUMMARY = json.loads(PW00_SUMMARY_PATH.read_text(encoding="utf-8"))' in pw00_execute
    assert 'summary_path={PW00_SUMMARY_PATH} stdout={PW00_RESULT.stdout} stderr={PW00_RESULT.stderr}' in pw00_execute
    assert 'pw00_stdout_lines = [line.strip() for line in PW00_RESULT.stdout.splitlines() if line.strip()]' not in pw00_execute
    assert 'json.loads(pw00_stdout_lines[-1])' not in pw00_execute