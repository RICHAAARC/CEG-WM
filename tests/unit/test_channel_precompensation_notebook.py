"""Packaging and notebook control-flow checks: no network/model execution."""

import base64
import io
import json
from pathlib import Path
import types
import zipfile

import pytest

from scripts.build_channel_precompensation_notebook import (
    ROOT, OUTPUT, SOURCE_FILES, build_notebook, source_snapshot,
)

pytestmark = pytest.mark.unit


def namespace():
    notebook = build_notebook()
    scope = {}
    source = next(cell for cell in notebook["cells"] if cell["id"] == "helpers")["source"]
    exec(compile("".join(source), "notebook:helpers", "exec"), scope)
    return scope


def config():
    return json.loads((ROOT / "configs/channel_precompensation_v1_colab.json").read_text())


def test_delivered_notebook_contains_current_sources_and_no_executed_cells():
    expected = build_notebook()
    actual = json.loads(OUTPUT.read_text())
    assert actual == expected, "Rebuild notebook after changing its bundled source files"
    code = [cell for cell in actual["cells"] if cell["cell_type"] == "code"]
    assert "".join(code[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in code:
        assert cell["execution_count"] is None and cell["outputs"] == []
        compile("".join(cell["source"]), cell["id"], "exec")
    scope = {}
    exec("".join(next(cell for cell in code if cell["id"] == "embedded_source")["source"]), scope)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(scope["SOURCE_BUNDLE_B64"]))) as archive:
        assert set(archive.namelist()) == set(SOURCE_FILES)
        for name in SOURCE_FILES:
            assert archive.read(name) == (ROOT / name).read_bytes()


def test_unpack_does_not_escape_destination(tmp_path):
    scope = namespace()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="unsafe"):
        scope["unpack_source"](buffer.getvalue(), tmp_path / "unpack")
    assert not (tmp_path / "outside.txt").exists()


def test_embedded_snapshot_has_importable_dependencies(tmp_path):
    scope = namespace()
    path = scope["unpack_source"](source_snapshot(), tmp_path / "source")
    assert (path / "src/cegwm/shared/numerics.py").is_file()
    assert scope["read_json"](path / scope["CONFIG_RELATIVE"])["quality"]["minimum_psnr_db"] is None


@pytest.mark.parametrize("terminal", [True, False])
def test_existing_results_are_read_only_even_when_incomplete(tmp_path, terminal):
    scope, settings = namespace(), config()
    root = tmp_path / "run"
    (root / "batch").mkdir(parents=True)
    summary = {"stages": {"prepare": "complete" if terminal else "started", "cleanup": "complete"},
               "actual_calls": {"final_generation_attempted": 4}, "sources": []}
    scope["write_json"](root / "config.json", settings)
    scope["write_json"](root / "batch/summary.json", summary)
    def forbidden(*args):
        raise AssertionError("Existing results must not install dependencies or run models")
    scope["ensure_runtime"] = forbidden
    assert scope["run_or_show"](settings, tmp_path, root, b"unused") == summary
    aggregate, archive = scope["package_results"](settings, root)
    assert aggregate["counts_terminal"] is terminal
    assert (aggregate["actual_calls"] is not None) is terminal
    assert len(aggregate["sources"]) == 4
    assert archive.is_file()


def test_launch_uses_fixed_config_and_separate_new_output(tmp_path):
    scope, settings = namespace(), config()
    checkout = scope["unpack_source"](source_snapshot(), tmp_path / "source")
    root = tmp_path / "run"
    calls = []
    scope["ensure_runtime"] = lambda *args: {"HF_TOKEN": "fake-only"}
    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("--output") + 1])
        assert not output.exists()
        output.mkdir()
        scope["write_json"](output / "summary.json", {"stages": {"cleanup": "complete"}, "sources": []})
        return types.SimpleNamespace(returncode=1)
    scope["subprocess"] = types.SimpleNamespace(run=run)
    scope["run_or_show"](settings, checkout, root, source_snapshot())
    assert len(calls) == 1
    assert calls[0][0][1:3] == ["-m", "experiments.run_channel_precompensation_v1"]
    assert calls[0][1]["check"] is False
    assert (root / "source_snapshot.zip").is_file()


def test_bundle_can_execute_plan_in_isolated_python_path(tmp_path):
    # A subprocess imports only the bundled runtime's CPU planning path. It
    # never selects --execute and must not construct model pipelines.
    import os
    import subprocess
    import sys
    scope = namespace()
    checkout = scope["unpack_source"](source_snapshot(), tmp_path / "source")
    environment = dict(os.environ, PYTHONPATH=str(checkout / "src") + os.pathsep + str(checkout))
    completed = subprocess.run([sys.executable, "-m", "experiments.run_channel_precompensation_v1",
                                "--config", str(checkout / scope["CONFIG_RELATIVE"])],
                               cwd=checkout, env=environment, text=True, capture_output=True, check=True)
    plan = json.loads(completed.stdout)
    assert plan["planned_budget"]["final_generation_calls"] == 20
    assert plan["quality"]["minimum_psnr_db"] is None
