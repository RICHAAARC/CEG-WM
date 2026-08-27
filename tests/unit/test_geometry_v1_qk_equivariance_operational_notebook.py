"""Static contract checks for the thin Geometry-V1 QK-E0 Colab handoff."""
from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = Path("notebooks/geometry_v1_qk_equivariance_operational_colab.ipynb")


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook: dict) -> str:
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_notebook_is_unexecuted_nbformat_with_exact_drive_first_code_cell() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4 and notebook["cells"]
    first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert first_code["source"] == ["from google.colab import drive\n", "drive.mount('/content/drive')\n"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []


def test_notebook_builds_the_fixed_eight_pair_plan_and_calls_existing_runner_once() -> None:
    code = _code(_notebook())
    assert "ATTENTION_LAYER_PATHS = ['transformer_blocks.0.attn', 'transformer_blocks.23.attn']" in code
    assert "TRANSFORMS = ('identity', 'd4', 'similarity', 'crop_rescale')" in code
    assert "len(plan['pairs']) != 8" in code
    assert code.count("subprocess.Popen(") == 1
    assert "RUNNER_PATH = 'experiments/run_geometry_v1_qk_equivariance_operational.py'" in code
    assert "Known H is evaluation truth only" in code


def test_fixed_layer_pair_has_one_assignment_and_flows_directly_to_the_plan() -> None:
    tree = ast.parse(_code(_notebook()))
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ATTENTION_LAYER_PATHS" for target in node.targets)
    ]
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.List) and [element.value for element in value.elts if isinstance(element, ast.Constant)] == ["transformer_blocks.0.attn", "transformer_blocks.23.attn"]
    assert not any(isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "ATTENTION_LAYER_PATHS" for node in ast.walk(tree))
    plan_values = [
        value for node in ast.walk(tree) if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "attention_layer_paths"
    ]
    assert len(plan_values) == 1 and isinstance(plan_values[0], ast.Name) and plan_values[0].id == "ATTENTION_LAYER_PATHS"


def test_runner_command_and_compact_control_are_bound_to_the_actual_checkout() -> None:
    code = _code(_notebook())
    assert "command = [sys.executable, RUNNER_PATH, '--plan', str(plan_path), '--repo-root', str(repo)," in code
    assert "'--expected-exact', execution_commit, '--output-root', str(run_dir)," in code
    assert "'--control-fd', str(control_write)]" in code
    assert code.count("subprocess.Popen(") == 1
    assert "line = os.read(control_read, MAX_CONTROL_BYTES + 1)" in code
    assert "len(line) > MAX_CONTROL_BYTES or not line.endswith" in code
    assert "receipt.get('run_id') != run_id" in code


def test_notebook_is_thin_fail_closed_and_keeps_the_evidence_ceiling() -> None:
    notebook = _notebook()
    code = _code(notebook)
    markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    for forbidden in ("observe_sd35", "evaluate_unit", "zipfile", "read_bytes", "hashlib", "revision", "to_q", "to_k"):
        assert forbidden not in code
    assert "userdata.get('HF_TOKEN')" in code
    assert "print(hf_token" not in code and "runner_env.pop('HF_TOKEN', None)" in code
    assert "if RUNNER_ATTEMPTED:" in code and "receipt_status != 'success'" in code
    for forbidden in ("retry", "fallback", "tuning"):
        assert forbidden not in code.lower()
    assert "science_denominator=0" in markdown
    assert "does not establish method, detector, or scientific success" in markdown
    assert "does not inspect ZIP contents" in markdown
