from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

_NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "stage_a2_hf_colab.ipynb"


@pytest.mark.unit
def test_stage_a_frequency_response_notebook_has_one_thin_handoff_path() -> None:
    document = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
    sources = ["".join(cell["source"]) for cell in code_cells]

    assert len(code_cells) == 4
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
    trees = [ast.parse(source) for source in sources]
    combined = "\n".join(sources)
    assert "refs/heads/stage-a-standalone-lf-hf-frequency-response-resumable-v2" in combined
    assert "/content/drive/MyDrive/CEG-WM/stage_a_standalone_lf_hf_frequency_response" in combined
    assert "slhfr-[0-9a-f]{24}" in combined
    assert "experiments.stage_a_frequency_response.run_colab" in combined
    for flag in ("--repo-root", "--expected-exact", "--local-work-root", "--artifact-sink"):
        assert flag in combined

    runner_index = next(index for index, source in enumerate(sources) if "subprocess.Popen" in source)
    terminal_index = len(code_cells) - 1
    assert runner_index < terminal_index == len(code_cells) - 1
    popen_calls = [
        node
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "Popen"
    ]
    assert len(popen_calls) == 1
    assert len(popen_calls[0].args) >= 1
    assert isinstance(popen_calls[0].args[0], ast.Name)
    assert popen_calls[0].args[0].id == "command"
    cwd_keywords = [keyword.value for keyword in popen_calls[0].keywords if keyword.arg == "cwd"]
    assert len(cwd_keywords) == 1
    cwd_value = cwd_keywords[0]
    assert isinstance(cwd_value, ast.Call)
    assert isinstance(cwd_value.func, ast.Name) and cwd_value.func.id == "str"
    assert len(cwd_value.args) == 1 and isinstance(cwd_value.args[0], ast.Name)
    assert cwd_value.args[0].id == "repo" and not cwd_value.keywords

    command_assignments = [
        node
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "command"
    ]
    assert len(command_assignments) == 1
    command_value = command_assignments[0].value
    assert isinstance(command_value, ast.List)
    command_elements = command_value.elts
    assert len(command_elements) == 11
    assert isinstance(command_elements[0], ast.Attribute)
    assert isinstance(command_elements[0].value, ast.Name)
    assert command_elements[0].value.id == "sys" and command_elements[0].attr == "executable"
    assert [element.value for element in command_elements[1:3] if isinstance(element, ast.Constant)] == ["-m", "experiments.stage_a_frequency_response.run_colab"]
    assert [element.value for element in command_elements[3::2] if isinstance(element, ast.Constant)] == ["--repo-root", "--expected-exact", "--local-work-root", "--artifact-sink"]
    assert all(isinstance(element, ast.Call) for element in (command_elements[4], command_elements[8], command_elements[10]))
    assert all(isinstance(element.func, ast.Name) and element.func.id == "str" and len(element.args) == 1 and isinstance(element.args[0], ast.Name) and not element.keywords for element in (command_elements[4], command_elements[8], command_elements[10]))
    assert command_elements[4].args[0].id == "repo"
    assert isinstance(command_elements[6], ast.Name) and command_elements[6].id == "resolved_exact"
    assert command_elements[8].args[0].id == "local_work_root"
    assert command_elements[10].args[0].id == "run_store_root"

    runner = sources[runner_index]
    assert runner.index("CEGWM_PROGRESS ") < runner.index("CEGWM_SUMMARY ")
    assert "if terminal_event is not None:" in runner
    assert "terminal_event['rc'] != runner_rc" in runner

    summary_assignments = [
        node
        for node in ast.walk(trees[terminal_index])
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "summary" for target in node.targets)
    ]
    assert len(summary_assignments) == 1
    summary_value = summary_assignments[0].value
    assert isinstance(summary_value, ast.Dict)
    assert 1 <= len(summary_value.keys) <= 8
    assert all(isinstance(key, ast.Constant) and isinstance(key.value, str) for key in summary_value.keys)

    terminal = sources[terminal_index]
    terminal_names = {
        node.id for node in ast.walk(trees[terminal_index]) if isinstance(node, ast.Name)
    }
    terminal_attributes = {
        node.attr for node in ast.walk(trees[terminal_index]) if isinstance(node, ast.Attribute)
    }
    assert not ({"hashlib", "zipfile", "receipt", "result"} & terminal_names)
    assert not ({"read_bytes", "read_text"} & terminal_attributes)
    nonzero_boundary = terminal.index("if runner_rc != 0:")
    assert terminal.index("summary =") < terminal.index("print(summary)") < nonzero_boundary
    assert "raise RuntimeError" in terminal[nonzero_boundary:]
