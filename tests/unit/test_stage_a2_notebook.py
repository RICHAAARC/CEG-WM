from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

_NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "stage_a2_hf_colab.ipynb"


@pytest.mark.unit
def test_stage_a2_notebook_has_one_thin_runnable_terminal_path() -> None:
    document = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
    sources = ["".join(cell["source"]) for cell in code_cells]

    assert len(code_cells) == 4
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
    trees = [ast.parse(source) for source in sources]

    runner_index = next(index for index, source in enumerate(sources) if "subprocess.Popen" in source)
    terminal_index = next(index for index, source in enumerate(sources) if "zipfile.ZipFile" in source)
    assert runner_index < terminal_index == len(code_cells) - 1

    summary_assignments = [
        node
        for node in ast.walk(trees[terminal_index])
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "summary" for target in node.targets)
    ]
    assert len(summary_assignments) == 1
    summary_value = summary_assignments[0].value
    assert isinstance(summary_value, ast.Dict)
    assert 1 <= len(summary_value.keys) <= 8
    assert all(isinstance(key, ast.Constant) and isinstance(key.value, str) for key in summary_value.keys)

    terminal = sources[terminal_index]
    nonzero_boundary = terminal.index("if runner_rc != 0:")
    assert terminal.index("summary =") < terminal.index("print(summary)") < nonzero_boundary
    assert "raise RuntimeError" in terminal[nonzero_boundary:]
