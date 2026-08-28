from __future__ import annotations

import json
from pathlib import Path

import pytest


NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "geometry_v3_qk_active_writer_p0d_colab.ipynb"
)
RUNNER = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_geometry_v3_qk_active_writer_p0d.py"
)
RUNNER_EXACT = "20e664f291bfab8c8fb571c767a3a58263542c5f"


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


@pytest.mark.unit
def test_first_executable_cell_is_exact_independent_drive_mount() -> None:
    cells = _code_cells()
    assert cells[0] == "from google.colab import drive\ndrive.mount('/content/drive')"
    assert "force_remount" not in cells[0]


@pytest.mark.unit
def test_notebook_binds_exact_single_runner_control_and_drive_contract() -> None:
    code = "\n".join(_code_cells())
    assert RUNNER.is_file()
    assert f"P0D_RUNNER_EXACT = '{RUNNER_EXACT}'" in code
    assert "git', 'checkout', '--detach', P0D_RUNNER_EXACT" in code
    assert "resolved != P0D_RUNNER_EXACT or dirty" in code
    assert "runner_path = checkout / RUNNER_RELATIVE_PATH" in code
    assert "'/content/drive/MyDrive/CEG-WM/Geometry-V3/P0D'" in code
    assert code.count("subprocess.Popen(") == 1
    assert "'--control-fd', str(write_fd)" in code
    assert "pass_fds=(write_fd,)" in code
    assert "os.read(read_fd, MAX_CONTROL_BYTES + 1)" in code
    assert "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL" in code
    assert "child.wait(timeout=7200)" in code
    assert "'status': control.get('p0d_status')" in code
    assert "'failure_point': control.get('failure_point')" in code
    assert "'error_class': control.get('error_class')" in code
    assert "'counters': control.get('counters')" in code
    assert "retry" not in code.lower()
    assert "fallback" not in code.lower()


@pytest.mark.unit
def test_notebook_keeps_credentials_out_of_plan_terminal_and_output() -> None:
    code = "\n".join(_code_cells())
    assert "userdata.get('HF_TOKEN')" in code
    assert "userdata.get('CEGWM_GEOMETRY_KEY')" in code
    plan_block = code[code.index("plan = {") : code.index("plan_file =")]
    terminal_block = code[code.index("terminal = {") : code.index("print(")]
    for secret_name in ("HF_TOKEN", "CEGWM_GEOMETRY_KEY"):
        assert secret_name not in plan_block
        assert secret_name not in terminal_block
    assert "P0_PROMPT_TEXT" not in code
    assert "raw_qk" not in code.lower()
    assert "anchor" not in terminal_block.lower()
    assert "latent" not in terminal_block.lower()
    assert "weights" not in terminal_block.lower()


@pytest.mark.unit
def test_notebook_plan_is_single_fixed_execution_without_method_choices() -> None:
    code = "\n".join(_code_cells())
    plan_block = code[code.index("plan = {") : code.index("plan_file =")]
    assert set(
        line.strip().split(":", 1)[0].strip("'")
        for line in plan_block.splitlines()[1:]
        if ":" in line
    ) == {"expected_exact", "execution_exact", "output_directory"}
    for forbidden in (
        "placement",
        "budget",
        "candidate",
        "baseline",
        "attack",
        "selection",
    ):
        assert forbidden not in plan_block.lower()

