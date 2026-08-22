from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

_NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "stage_a2_hf_colab.ipynb"
_BRANCH = "stage-a-content-adaptive-dual-branch-v2"
_RUNNER = "experiments.run_content_adaptive_dual_branch_v2_clean"
_LOCAL_ROOT = "/content/cegwm-stage-a-content-adaptive-dual-branch-v2-local"
_ARTIFACT_SINK = "/content/drive/MyDrive/CEG-WM/stage_a_content_adaptive_dual_branch_v2_clean"
_SCOPE_PARAGRAPH = "Content V2 clean mechanism only, with a fixed 8 units/16 records: no attacks, complementarity, superiority, geometry, fixed-FPR calibration, Stage-A/content-chain completion, scientific self-promotion, or paper promotion; local/Colab results are engineering evidence only, and any returned ZIP+SHA requires external supervisor validation."


def _popen_calls(trees: list[ast.AST]) -> list[ast.Call]:
    return [
        node
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "Popen"
    ]


@pytest.mark.unit
def test_stage_a_content_v2_notebook_has_one_thin_handoff_path() -> None:
    document = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
    sources = ["".join(cell["source"]) for cell in code_cells]
    trees = [ast.parse(source) for source in sources]
    combined = "\n".join(sources)

    assert len(code_cells) == 4
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
    assert document["cells"][0]["source"][-1] == _SCOPE_PARAGRAPH
    assert f"refs/heads/{_BRANCH}" in combined
    assert _RUNNER in combined
    assert _LOCAL_ROOT in combined
    assert _ARTIFACT_SINK in combined
    assert combined.count(_RUNNER) == 1
    checkout = next(source for source in sources if "refs/heads/" in source)
    assert checkout.count(f"refs/heads/{_BRANCH}") == 1
    assert checkout.count("'rev-parse'") == 1
    assert "'checkout', '--detach', 'FETCH_HEAD'" in checkout
    assert "'status', '--porcelain'" in checkout

    runner_index = next(index for index, source in enumerate(sources) if "subprocess.Popen" in source)
    terminal_index = len(code_cells) - 1
    assert runner_index < terminal_index
    assert "CEGWM_PROGRESS " in sources[runner_index]
    assert "CEGWM_SUMMARY " in sources[runner_index]

    popen_calls = _popen_calls(trees)
    assert len(popen_calls) == 1
    cwd_keywords = [keyword.value for keyword in popen_calls[0].keywords if keyword.arg == "cwd"]
    assert len(cwd_keywords) == 1
    cwd_value = cwd_keywords[0]
    assert isinstance(cwd_value, ast.Call)
    assert isinstance(cwd_value.func, ast.Name) and cwd_value.func.id == "str"
    assert len(cwd_value.args) == 1 and isinstance(cwd_value.args[0], ast.Name)
    assert cwd_value.args[0].id == "repo" and not cwd_value.keywords

    runner = sources[runner_index]
    for argument in ("--repo-root", "--expected-exact", "--local-work-root", "--artifact-sink"):
        assert runner.count(repr(argument)) == 1
    command_source = next(line for line in runner.splitlines() if line.startswith("command ="))
    assert "'CEG_WM_ROOT_KEY'" not in command_source
    assert "'HF_TOKEN'" not in command_source
    assert "runner_env['CEG_WM_ROOT_KEY'] = root_key" in runner
    assert "runner_env['HF_TOKEN'] = hf_token" in runner
    assert "runner_env.pop('CEG_WM_ROOT_KEY', None)" in runner
    assert "runner_env.pop('HF_TOKEN', None)" in runner
    assert "root_key = hf_token = ''" in runner
    assert "set(progress) != {'run_id', 'committed', 'fixed_total', 'phase'}" in runner
    assert "set(runner_summary) != {'run_id', 'committed', 'fixed_total', 'rc', 'phase'}" in runner
    assert runner.count("content-adaptive-v2-[0-9a-f]{12}-[0-9a-f]{12}") == 2

    terminal = sources[terminal_index]
    terminal_tree = trees[terminal_index]
    summary_assignments = [
        node
        for node in ast.walk(terminal_tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "summary" for target in node.targets)
    ]
    assert len(summary_assignments) == 1
    summary_value = summary_assignments[0].value
    assert isinstance(summary_value, ast.Dict)
    assert [key.value for key in summary_value.keys if isinstance(key, ast.Constant)] == [
        "run_id", "resolved_exact", "runner_rc", "zip_path", "checksum_path", "pair_present",
    ]
    assert not ({"hashlib", "zipfile", "receipt", "result"} & {
        node.id for node in ast.walk(terminal_tree) if isinstance(node, ast.Name)
    })
    assert not ({"read_bytes", "read_text", "open"} & {
        node.attr for node in ast.walk(terminal_tree) if isinstance(node, ast.Attribute)
    })
    assert terminal.index("summary =") < terminal.index("print(summary)")
    assert terminal.index("print(summary)") < terminal.index("raise RuntimeError")
