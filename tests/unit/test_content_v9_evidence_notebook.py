import ast
import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOK = _ROOT / "notebooks" / "content_v9_formal_initial_colab.ipynb"
_RUNNER = _ROOT / "experiments/run_content_v9_stability.py"


def test_content_v9_evidence_notebook_is_self_contained() -> None:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    source = "\n".join(code_cells)

    ast.parse(source)
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert 'BRANCH = "Content-V9-Evidence"' in source
    assert 'RUNNER_MODULE = "experiments.run_content_v9_stability"' in source
    assert source.count("RUNNER_RC = subprocess.run(") == 1
    assert "RUNNER_RC = subprocess.run(" not in code_cells[-1]
    assert _RUNNER.is_file()
