import ast
import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOK = _ROOT / "notebooks" / "content_v6_formal_initial_colab.ipynb"
_RUNNER = _ROOT / "experiments/run_content_v6_formal_initial.py"


def test_content_v6_evidence_notebook_is_self_contained() -> None:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    source = "\n".join(code_cells)

    ast.parse(source)
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert 'BRANCH = "Content-V6-Evidence"' in source
    assert 'RUNNER_MODULE = "experiments.run_content_v6_formal_initial"' in source
    assert source.count("subprocess.Popen(") == 1
    assert "subprocess.Popen(" not in code_cells[-1]
    assert _RUNNER.is_file()
