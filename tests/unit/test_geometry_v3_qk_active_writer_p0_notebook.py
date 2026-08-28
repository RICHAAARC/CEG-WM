from __future__ import annotations

import json
from pathlib import Path

import pytest


NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "geometry_v3_qk_active_writer_p0_colab.ipynb"
RUNNER = Path(__file__).resolve().parents[2] / "experiments" / "run_geometry_v3_qk_active_writer_p0.py"


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.mark.unit
def test_first_executable_cell_is_exact_independent_drive_mount() -> None:
    cells = _code_cells()
    assert cells[0] == "from google.colab import drive\ndrive.mount('/content/drive')"
    assert "force_remount" not in cells[0]


@pytest.mark.unit
def test_notebook_binds_checkout_runner_tokens_control_and_drive_contract() -> None:
    code = "\n".join(_code_cells())
    assert RUNNER.is_file()
    assert "P0_RUNNER_EXACT = '9b5085c805b6e3580fadc153598aac93fcc41eab'" in code
    assert "git', 'checkout', '--detach', P0_RUNNER_EXACT" in code
    assert "resolved != P0_RUNNER_EXACT or dirty" in code
    assert "runner_path = checkout / RUNNER_RELATIVE_PATH" in code
    assert "userdata.get('HF_TOKEN')" in code
    assert "userdata.get('CEGWM_GEOMETRY_KEY')" in code
    assert "'/content/drive/MyDrive/CEG-WM/Geometry-V3/P0'" in code
    assert code.count("subprocess.Popen(") == 1
    assert "'--control-fd', str(write_fd)" in code
    assert "pass_fds=(write_fd,)" in code
    assert "os.read(read_fd, MAX_CONTROL_BYTES + 1)" in code
    assert "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL" in code
    assert "child.wait(timeout=7200)" in code
    assert "'handoff_status': control.get('status')" in code
    assert "'status': control.get('p0_status')" in code
    assert "retry" not in code.lower()
    assert "fallback" not in code.lower()


@pytest.mark.unit
def test_notebook_does_not_persist_or_print_credentials_or_method_inputs() -> None:
    code = "\n".join(_code_cells())
    plan_block = code[code.index("plan = {") : code.index("plan_file =")]
    terminal_block = code[code.index("terminal = {") : code.index("print(")]
    for secret_name in ("HF_TOKEN", "CEGWM_GEOMETRY_KEY"):
        assert secret_name not in plan_block
        assert secret_name not in terminal_block
    assert "P0_PROMPT_TEXT" not in code
    assert "raw_qk" not in code.lower()
    assert "images" not in plan_block.lower()
