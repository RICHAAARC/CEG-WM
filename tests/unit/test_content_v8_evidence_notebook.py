import ast
import json
import re
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOK = _ROOT / "notebooks" / "content_v8_formal_initial_colab.ipynb"
_RUNNER = _ROOT / "experiments/run_content_v8_formal_initial.py"


def _notebook_code() -> tuple[dict, list[str], str]:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    return notebook, code_cells, "\n".join(code_cells)


def _summary_parser(source: str):
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_runner_summary"
    )
    namespace = {
        "json": json,
        "re": re,
        "RUNNER_SUMMARY_PREFIX": "CEGWM_CONTENT_V8_FORMAL_SUMMARY ",
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook>", "exec"), namespace)
    return namespace["parse_runner_summary"]


def test_content_v8_evidence_notebook_is_self_contained() -> None:
    notebook, code_cells, source = _notebook_code()

    ast.parse(source)
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert 'BRANCH = "Content-V8-Evidence"' in source
    assert 'RUNNER_MODULE = "experiments.run_content_v8_formal_initial"' in source
    assert 'RUN_ID = None' in source
    assert 'RUNNER_TERMINAL_SHA256 = None' in source
    assert 'RUN_ID = "content-v8-' not in source
    assert source.index("parse_runner_summary(captured, execution_short)") < source.index("captured.clear()")
    assert 'pairs[0]["sha256"] != RUNNER_TERMINAL_SHA256' in source
    assert source.count("subprocess.Popen(") == 1
    assert "subprocess.Popen(" not in code_cells[-1]
    assert _RUNNER.is_file()


def test_content_v8_evidence_notebook_uses_runner_dynamic_identity() -> None:
    _, _, source = _notebook_code()
    parse_runner_summary = _summary_parser(source)
    summary = {
        "evaluation_rc_in_order": [0, 0],
        "rc": 0,
        "run_id": "content-v8-0123456789ab-fedcba987654",
        "terminal_sha256": "a" * 64,
    }
    captured = (
        b'CEGWM_CONTENT_V8_RUNTIME_ASSET {"fit_sample_count":32}\n'
        + b"CEGWM_CONTENT_V8_FORMAL_SUMMARY "
        + json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("ascii")
        + b"\n"
    )

    assert parse_runner_summary(captured, "0123456789ab") == (
        summary["run_id"], summary["terminal_sha256"]
    )

    try:
        parse_runner_summary(captured, "ba9876543210")
    except RuntimeError:
        pass
    else:
        raise AssertionError("mismatched execution identity must fail closed")

    summary["unexpected"] = True
    invalid_schema = (
        b"CEGWM_CONTENT_V8_FORMAL_SUMMARY "
        + json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("ascii")
        + b"\n"
    )
    try:
        parse_runner_summary(invalid_schema, "0123456789ab")
    except RuntimeError:
        pass
    else:
        raise AssertionError("unknown runner summary fields must fail closed")
