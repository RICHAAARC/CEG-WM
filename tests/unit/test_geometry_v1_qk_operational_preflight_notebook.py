from __future__ import annotations

import json
from pathlib import Path


def test_geometry_notebook_is_prepared_and_create_only() -> None:
    notebook = json.loads(Path("notebooks/geometry_v1_qk_operational_preflight_colab.ipynb").read_text())
    codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert notebook["nbformat"] == 4
    assert all(cell.get("execution_count") is None and not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "drive.mount('/content/drive', force_remount=False)" in codes[0]
    assert "4e39ec28fc2c1f8cc2848c62360f3ca096184658" in source
    assert "PREPARED_NOT_EXECUTED" in source and "content_v" not in source and "Content" not in source
    assert "open('xb')" in source and "copy2" not in source and "write_bytes" not in source
    assert "receipt.json" in source and "manifest.json" in source and "SHA256SUMS" in source
