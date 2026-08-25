from __future__ import annotations

import json
from pathlib import Path


def test_geometry_notebook_is_thin_and_exact_bound() -> None:
    path = Path("notebooks/geometry_v1_qk_operational_preflight_colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(cell.get("execution_count") is None and cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "30c04f98e7a6b30e58f3d105412ef534e6742deb" in source
    assert "BRANCH='Geometry-V1'" in source
    assert "PREPARED_NOT_EXECUTED" in source
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "drive.mount('/content/drive')" in "".join(code_cells[0]["source"])
    assert "Content" not in source and "content_v" not in source
    assert "CEG_WM_ROOT_KEY" in source and "HF_TOKEN" in source
    assert "subprocess.run" in source and "stderr=subprocess.DEVNULL" in source
    assert "pip','install'" in source and "verify_checkout()" in source
    assert "--expected-exact" in source and "geometry-v1-b2b-30c04f98e7a6-operational-01" in source
    assert "receipt.json" in source and "manifest.json" in source and "SHA256SUMS" in source
