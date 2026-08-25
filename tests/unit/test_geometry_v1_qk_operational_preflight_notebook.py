from __future__ import annotations

import json
from pathlib import Path


def test_notebook_is_complete_prepared_create_only_handoff() -> None:
    value = json.loads(Path("notebooks/geometry_v1_qk_operational_preflight_colab.ipynb").read_text())
    codes = ["".join(cell["source"]) for cell in value["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in value["cells"])
    assert value["nbformat"] == 4
    assert all(cell.get("execution_count") is None and not cell.get("outputs") for cell in value["cells"] if cell["cell_type"] == "code")
    assert 'drive.mount("/content/drive", force_remount=False)' in codes[0]
    assert "f1b89c00f19fa561235170aaeb342671a69c5906" in source
    assert 'PROPOSED_PENDING_FINAL_USER_CONFIRMATION="/content/drive/MyDrive/CEG-WM/Geometry-V1/Batch2B"' in source
    assert "PREPARED_NOT_EXECUTED" in source and "try: pass" not in source
    assert "git','checkout','--detach',EXECUTION_EXACT" in source and "pip','install'" in source
    assert source.index("try:") < source.index("files.upload()") < source.index("userdata.get('HF_TOKEN')")
    assert "CEGWM_GEOMETRY_V1_OPERATIONAL_PREFLIGHT " in source and "CEGWM_GEOMETRY_V1_OPERATIONAL_FAILURE " in source
    assert "receipt/return-code mismatch" in source and "status=='failure'" in source
    assert "receipt.json" in source and "manifest.json" in source and "SHA256SUMS" in source
    assert "allowed_filenames" in source and "source.open('rb')" in source and "target.open('xb')" in source
    assert "copy2" not in source and "write_bytes" not in source and "content_v" not in source and "Content" not in source
    assert "process.kill" in source and "runner_env.pop('HF_TOKEN'" in source and "input_dir.rmdir" in source
