from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "geometry_v3_qk_active_writer_p1m0_mechanism_audit_colab.ipynb"
RUNNER = ROOT / "experiments" / "run_geometry_v3_qk_active_writer_p1m0_mechanism_audit.py"
RUNNER_EXACT = "0ae56ab89204d63b4f3f0ee8544d82a19faa8d8f"


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.mark.unit
def test_first_executable_cell_is_exact_two_line_drive_mount() -> None:
    cells = _code_cells()
    assert cells[0] == "from google.colab import drive\ndrive.mount('/content/drive')"
    assert "force_remount" not in cells[0]


@pytest.mark.unit
def test_notebook_binds_runner_both_sources_and_create_only_output() -> None:
    code = "\n".join(_code_cells())
    assert RUNNER.is_file()
    assert f"P1M0_RUNNER_EXACT = '{RUNNER_EXACT}'" in code
    assert "git', 'checkout', '--detach', P1M0_RUNNER_EXACT" in code
    assert "resolved != P1M0_RUNNER_EXACT or dirty" in code
    assert "runner_path = checkout / RUNNER_RELATIVE_PATH" in code
    assert "'expected_exact': P1M0_RUNNER_EXACT" in code
    assert "'execution_exact': P1M0_RUNNER_EXACT" in code
    assert "Geometry-V3/P0/Geometry-V3-P0-9b5085c805b6-20260828T122005Z" in code
    assert "Geometry-V3/P1/Geometry-V3-P1-517ba73993f1-20260828T131759Z" in code
    assert "DRIVE_ROOT = Path('/content/drive/MyDrive/CEG-WM/Geometry-V3/P1M0')" in code
    assert "if drive_directory.exists()" in code
    assert "'p0_source_directory': str(P0_SOURCE_ROOT)" in code
    assert "'p1_source_directory': str(P1_SOURCE_ROOT)" in code
    assert "'output_directory': str(drive_directory)" in code


@pytest.mark.unit
def test_notebook_has_one_bounded_child_without_retry_or_fallback() -> None:
    code = "\n".join(_code_cells())
    assert code.count("subprocess.Popen(") == 1
    assert "'--control-fd', str(write_fd)" in code
    assert "pass_fds=(write_fd,)" in code
    assert "os.read(read_fd, MAX_CONTROL_BYTES + 1)" in code
    assert "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL" in code
    assert "child.wait(timeout=7200)" in code
    assert "'status': control.get('p1m0_status')" in code
    assert "'failure_point': control.get('failure_point')" in code
    assert "'error_class': control.get('error_class')" in code
    assert "retry" not in code.lower()
    assert "fallback" not in code.lower()


@pytest.mark.unit
def test_notebook_keeps_secrets_and_private_material_out_of_public_plan() -> None:
    code = "\n".join(_code_cells())
    assert "userdata.get('HF_TOKEN')" in code
    assert "userdata.get('CEGWM_GEOMETRY_KEY')" in code
    plan = code[code.index("plan = {") : code.index("plan_file =")]
    terminal = code[code.index("terminal = {") : code.index("print(")]
    for secret_name in ("HF_TOKEN", "CEGWM_GEOMETRY_KEY"):
        assert secret_name not in plan
        assert secret_name not in terminal
    for forbidden in ("raw_qk", "prompt_text", "latent", "anchor", "pattern", "model_weights"):
        assert forbidden not in plan.lower()
    assert "block4-qk" not in code and "block20-qk" not in code
    assert "relative_rms_budget" not in code
