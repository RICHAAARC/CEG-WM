from __future__ import annotations

import json
from pathlib import Path
import re

import pytest


NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "geometry_v2_neural_corner_sync_n0_colab.ipynb"


@pytest.mark.unit
def test_n0_notebook_is_exact_bound_single_runner_create_only_handoff() -> None:
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = document["cells"]
    assert cells[0]["cell_type"] == "code"
    assert cells[0]["source"] == ["from google.colab import drive\n", "drive.mount('/content/drive')\n"]
    source = "".join("".join(cell.get("source", [])) for cell in cells)
    match = re.search(r"N0_RUNNER_EXACT = '([^']+)'", source)
    assert match and match.group(1) == "054953fe20d717faf740ba4e51a71d9b21e8f44a"
    assert "git', 'checkout', '--detach', N0_RUNNER_EXACT" in source
    assert "execution_commit != N0_RUNNER_EXACT" in source and "git_output(repo, 'status', '--porcelain')" in source
    assert source.count("subprocess.Popen(") == 1 and "RUNNER_ATTEMPTED" in source
    assert "'--expected-exact', execution_commit" in source and "'--control-fd', str(control_write)" in source
    assert "MAX_CONTROL_BYTES = 1024" in source and "timeout=7200" in source
    assert "stdout=subprocess.DEVNULL" in source and "stderr=subprocess.DEVNULL" in source
    assert "/content/drive/MyDrive/CEG-WM/Geometry-V2/N0" in source and "run_dir.exists()" in source
    assert "userdata.get('CEGWM_GEOMETRY_KEY')" in source and "secrets.token_bytes(32)" in source
    assert "CEGWM_GEOMETRY_KEY_HEX" in source and "geometry_secret = None; derived_key_hex = ''" in source
    assert "force_remount" not in source and "retry" not in source.lower() and "fallback" not in source.lower()
    assert "science_denominator': 0" in source and "CEGWM_GEOMETRY_V2_N0_TERMINAL" in source


@pytest.mark.unit
def test_n0_notebook_does_not_persist_or_display_sensitive_runtime_material() -> None:
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert "torch.save" not in source and "checkpoint" not in source.lower()
    assert "print(geometry_secret" not in source and "print(derived_key_hex" not in source
    assert "raw_qk" not in source.lower() and "image_bytes" not in source.lower()
