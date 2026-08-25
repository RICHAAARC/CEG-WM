from __future__ import annotations

import json
from pathlib import Path


def test_geometry_notebook_is_thin_and_exact_bound() -> None:
    path = Path("notebooks/geometry_v1_qk_operational_preflight_colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(cell.get("execution_count") is None and cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "beb5be85e53dda539e055ede98cda9da0ffd00c3" in source
    assert "BRANCH = 'Geometry-V1'" in source
    assert "PREPARED_NOT_EXECUTABLE_FROM_COLAB" in source
    assert "drive.mount" not in source
    assert "CEG_WM_ROOT_KEY" in source and "HF_TOKEN" in source
    assert "subprocess.run" in source and "stderr=subprocess.DEVNULL" in source
