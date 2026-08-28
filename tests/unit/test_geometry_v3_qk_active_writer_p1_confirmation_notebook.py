from __future__ import annotations

import json
from pathlib import Path

import pytest


NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "geometry_v3_qk_active_writer_p1_confirmation_colab.ipynb"
RUNNER = Path(__file__).resolve().parents[2] / "experiments" / "run_geometry_v3_qk_active_writer_p1_confirmation.py"
RUNNER_EXACT = "f4a78c6f3e820ea3ef3de25e741dfbbaddf0dfbd"


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.mark.unit
def test_first_executable_cell_is_exact_independent_drive_mount() -> None:
    cells = _code_cells()
    assert cells[0] == "from google.colab import drive\ndrive.mount('/content/drive')"
    assert "force_remount" not in cells[0]


@pytest.mark.unit
def test_notebook_binds_runner_source_output_and_bounded_control() -> None:
    code = "\n".join(_code_cells())
    assert RUNNER.is_file()
    assert f"P1_RUNNER_EXACT = '{RUNNER_EXACT}'" in code
    assert "git', 'checkout', '--detach', P1_RUNNER_EXACT" in code
    assert "resolved != P1_RUNNER_EXACT or dirty" in code
    assert "runner_path = checkout / RUNNER_RELATIVE_PATH" in code
    assert "'expected_exact': P1_RUNNER_EXACT" in code
    assert "'execution_exact': P1_RUNNER_EXACT" in code
    assert "Geometry-V3/P0/Geometry-V3-P0-9b5085c805b6-20260828T122005Z" in code
    assert "DRIVE_ROOT = Path('/content/drive/MyDrive/CEG-WM/Geometry-V3/P1')" in code
    assert "'source_directory': str(SOURCE_ROOT)" in code
    assert "'output_directory': str(drive_directory)" in code
    assert code.count("subprocess.Popen(") == 1
    assert "'--control-fd', str(write_fd)" in code
    assert "pass_fds=(write_fd,)" in code
    assert "os.read(read_fd, MAX_CONTROL_BYTES + 1)" in code
    assert "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL" in code
    assert "child.wait(timeout=7200)" in code
    assert "'status': control.get('p1_status')" in code
    assert "'failure_point': control.get('failure_point')" in code
    assert "'error_class': control.get('error_class')" in code
    assert "retry" not in code.lower()
    assert "fallback" not in code.lower()


@pytest.mark.unit
def test_notebook_keeps_credentials_out_of_plan_and_terminal() -> None:
    code = "\n".join(_code_cells())
    assert "userdata.get('HF_TOKEN')" in code
    assert "userdata.get('CEGWM_GEOMETRY_KEY')" in code
    plan_block = code[code.index("plan = {") : code.index("plan_file =")]
    terminal_block = code[code.index("terminal = {") : code.index("print(")]
    for secret_name in ("HF_TOKEN", "CEGWM_GEOMETRY_KEY"):
        assert secret_name not in plan_block
        assert secret_name not in terminal_block
    assert "P1_PROMPT_TEXT" not in code
    assert "raw_qk" not in code.lower()
    assert "images" not in plan_block.lower()


@pytest.mark.unit
def test_notebook_uses_one_frozen_candidate_without_switching() -> None:
    code = "\n".join(_code_cells())
    assert "block4-qk" not in code and "block20-qk" not in code
    assert "relative_rms_budget" not in code
    assert "selected_config" not in code
    assert "fixed_config_id" in code
