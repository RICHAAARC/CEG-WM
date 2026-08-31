from __future__ import annotations

import ast
import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOK = _ROOT / "notebooks/geometry_v5_m0_diagnostic_colab.ipynb"
_EXACT = "97c4ca7744784969e51ef57601990dde2a4a4a17"


def _notebook() -> dict[str, object]:
    return json.loads(_NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict[str, object]) -> list[dict[str, object]]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]  # type: ignore[index]


def test_diagnostic_notebook_has_a_clean_colab_mount_cell_and_unexecuted_nbformat() -> None:
    notebook = _notebook()
    cells = _code_cells(notebook)
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert len(cells) == 3
    assert cells[0]["source"] == "from google.colab import drive\ndrive.mount('/content/drive')"
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in cells)
    for cell in cells:
        ast.parse(cell["source"])  # type: ignore[arg-type]


def test_diagnostic_notebook_pins_detached_clean_source_once_and_calls_fixed_runner_from_repo() -> None:
    source = "\n".join(cell["source"] for cell in _code_cells(_notebook()))  # type: ignore[arg-type]
    assert source.count(_EXACT) == 1
    assert "https://github.com/RICHAAARC/CEG-WM.git" in source
    assert "'checkout', '--detach', SOURCE_EXACT" in source
    assert "'branch', '--show-current'" in source and "'status', '--porcelain'" in source
    assert source.count("'-m', 'pip', 'install', '.'") == 1
    assert source.count("experiments.geometry_v5_m0_diagnostic") == 1
    assert "cwd=REPO" in source and "stderr=subprocess.DEVNULL" in source
    assert source.index("torch.cuda.is_available") < source.index("experiments.geometry_v5_m0_diagnostic")


def test_diagnostic_notebook_uses_single_create_only_drive_json_and_has_no_execution_governance_expansion() -> None:
    source = "\n".join(cell["source"] for cell in _code_cells(_notebook()))  # type: ignore[arg-type]
    assert "/content/drive/MyDrive/CEG-WM/Geometry-V5/M0-Diagnostic/runs" in source
    assert "OUTPUT.exists()" in source and "FileExistsError" in source
    assert "SOURCE_EXACT}_seed7501_" in source
    assert "lines[-1]" in source and "len(summary_line.encode('utf-8')) > 4096" in source
    for forbidden in ("force_remount", "sys.path", "retry", "fallback", "resume", "files.download", "artifact", "aggregate", "44"):
        assert forbidden not in source
    prose = "\n".join(cell["source"] for cell in _notebook()["cells"])  # type: ignore[arg-type,index]
    assert "RELIABLE" in prose and "science denominator is 0" in prose and "content score" not in prose


def test_diagnostic_notebook_reports_only_per_case_json_fields_after_runner_failure() -> None:
    source = "\n".join(cell["source"] for cell in _code_cells(_notebook()))  # type: ignore[arg-type]
    failure = source.index("if completed.returncode != 0:")
    success_summary = source.index("summary_line = lines[-1]")
    assert failure < success_summary
    assert "OUTPUT.read_text(encoding='utf-8')" in source
    assert "for case in cases:" in source
    assert "('attack_id', 'failure_stage', 'error_class')" in source
    assert "report['raw.status'] = raw['status']" in source
    assert "report['diagnostics'] = raw['diagnostics']" in source
    assert "diagnostic incomplete; per-case status printed" in source
    for forbidden in ("completed.stderr", "traceback", "original_prompt", "initial_z_t", "final_rgb"):
        assert forbidden not in source
