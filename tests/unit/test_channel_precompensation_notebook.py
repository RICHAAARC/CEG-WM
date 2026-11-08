"""Packaging and notebook control-flow checks: no network/model execution."""

import base64
import io
import json
from pathlib import Path
import types
import zipfile

import pytest

from scripts.build_channel_precompensation_notebook import (
    ROOT, OUTPUT, CONTINUATION_OUTPUT, SOURCE_FILES, build_notebook,
    build_continuation_notebook, source_snapshot,
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


def continuation_namespace():
    notebook = build_continuation_notebook()
    scope = {}
    source = next(cell for cell in notebook["cells"] if cell["id"] == "helpers")["source"]
    exec(compile("".join(source), "continuation:helpers", "exec"), scope)
    return scope


class FakeRunnerProcess:
    def __init__(self, returncode, output=""):
        self.returncode = returncode
        self.stdout = io.StringIO(output)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stdout.close()

    def wait(self):
        return self.returncode


def test_continuation_notebook_is_unexecuted_and_contains_current_source():
    expected = build_continuation_notebook()
    assert json.loads(CONTINUATION_OUTPUT.read_text()) == expected
    code = [cell for cell in expected["cells"] if cell["cell_type"] == "code"]
    assert "".join(code[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in code:
        assert cell["execution_count"] is None and cell["outputs"] == []
        compile("".join(cell["source"]), cell["id"], "exec")


def test_continuation_launch_reuses_saved_rgb_checkpoint_without_overwriting_metadata(tmp_path):
    scope = continuation_namespace()
    checkout = scope["unpack_source"](source_snapshot(), tmp_path / "source")
    root = tmp_path / "run"
    original = root / scope["RESUME_NAME"]
    original.mkdir(parents=True)
    scope["write_json"](original / "config.json", config())
    scope["write_json"](original / "summary.json", {"stages": {"cleanup": "complete"},
                                                  "actual_calls": {"final_generation_attempted": 4}})
    (root / "environment.json").write_text("original environment")
    (root / "source_snapshot.zip").write_bytes(b"original code")
    earlier_metadata = root / (scope["RESUME_NAME"] + "_metadata") / "launch_0001"
    earlier_metadata.mkdir(parents=True)
    (earlier_metadata / "runner.log").write_text("saved source suffix failure")
    metadata_root = root / (scope["CONTINUATION_NAME"] + "_metadata")
    metadata_root.mkdir()
    (metadata_root / "source_snapshot.zip").write_bytes(b"earlier continuation source")
    (metadata_root / "environment.json").write_text("earlier continuation environment")
    (metadata_root / "notebook_error.json").write_text("earlier continuation error")
    previous_launch = metadata_root / "launch_0001"
    previous_launch.mkdir()
    (previous_launch / "runner.log").write_text("retained early launch error")
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    calls = []
    def ensure(checkout, metadata_dir):
        assert metadata_dir == metadata_root / "launch_0002"
        scope["write_json"](metadata_dir / "environment.json", {"kind": "continuation"})
        return {"HF_TOKEN": "fake-only"}
    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert command[command.index("--resume-from") + 1] == str(original)
        assert "--config" not in command
        output = Path(command[command.index("--output") + 1])
        assert output == root / "continuation_suffix_device_v1" and not output.exists()
        output.mkdir()
        scope["write_json"](output / "summary.json", {"stages": {"cleanup": "complete"}, "sources": []})
        return FakeRunnerProcess(1, "retained numerical outcome\n")
    scope["ensure_runtime"] = ensure
    scope["subprocess"] = types.SimpleNamespace(Popen=run, PIPE=-1, STDOUT=-2)
    scope["run_or_show_continuation"](checkout, root, source_snapshot())
    assert len(calls) == 1
    metadata = metadata_root / "launch_0002"
    assert scope["read_json"](metadata / "runner_process.json")["returncode"] == 1
    assert (metadata / "runner.log").read_text() == "retained numerical outcome\n"
    for relative, content in before.items():
        assert (root / relative).read_bytes() == content


def test_continuation_early_exit_preserves_both_streams_and_returncode(tmp_path, capsys):
    import subprocess
    import sys

    scope = continuation_namespace()
    root = tmp_path / "run"
    original = root / scope["RESUME_NAME"]
    original.mkdir(parents=True)
    scope["write_json"](original / "config.json", config())
    scope["write_json"](original / "summary.json", {"actual_calls": {"final_generation_attempted": 4}})
    scope["ensure_runtime"] = lambda *args: {}
    calls = []

    def launch(command, **kwargs):
        calls.append(command)
        assert command[1:4] == ["-u", "-m", "experiments.run_channel_precompensation_v1"]
        # A tiny subprocess exercises stdout/stderr capture; no project runner,
        # model import, GPU work or network call is executed.
        return subprocess.Popen(
            [sys.executable, "-u", "-c",
             "import sys; print('checkpoint loading'); print('saved failure detail', file=sys.stderr); sys.exit(7)"],
            **kwargs,
        )

    scope["subprocess"] = types.SimpleNamespace(Popen=launch, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
    metadata = root / (scope["CONTINUATION_NAME"] + "_metadata") / "launch_0001"
    with pytest.raises(RuntimeError, match="runner return code 7") as caught:
        scope["run_or_show_continuation"](tmp_path, root, b"retained source snapshot")
    assert len(calls) == 1
    assert str(metadata / "runner.log") in str(caught.value)
    log = (metadata / "runner.log").read_text()
    assert "checkpoint loading" in log and "saved failure detail" in log
    visible = capsys.readouterr().out
    assert "checkpoint loading" in visible and "saved failure detail" in visible
    assert scope["read_json"](metadata / "runner_process.json") == {
        "returncode": 7, "log_path": str(metadata / "runner.log")}
    assert not (root / scope["CONTINUATION_NAME"]).exists()
    report, archive = scope["package_continuation"](config(), root)
    assert report["counts_terminal"] is False and report["actual_calls"] is None
    with zipfile.ZipFile(archive) as saved:
        assert saved.read("run/" + str(metadata.relative_to(root)) + "/runner.log").decode() == log


def test_existing_interrupted_continuation_is_not_rerun_and_keeps_prior_counts(tmp_path):
    scope = continuation_namespace()
    root = tmp_path / "run"
    (root / "batch").mkdir(parents=True)
    scope["write_json"](root / "batch/summary.json", {
        "actual_calls": {"final_generation_attempted": 4, "reconstruction_attempted": 12}})
    (root / "batch/original_failure.pt").write_bytes(b"retained")
    checkpoint = root / scope["RESUME_NAME"]
    checkpoint.mkdir()
    scope["write_json"](checkpoint / "config.json", config())
    prior = {"actual_calls": {"final_generation_attempted": 0, "reconstruction_attempted": 4,
                              "rgb_vjps_attempted": 8},
             "cumulative_actual_calls": {"final_generation_attempted": 4, "reconstruction_attempted": 16,
                                         "rgb_vjps_attempted": 15}}
    scope["write_json"](checkpoint / "summary.json", prior)
    (checkpoint / "suffix_failure.pt").write_bytes(b"retained suffix failure")
    output = root / scope["CONTINUATION_NAME"]
    output.mkdir()
    pending = {"stages": {"prepare": "started", "cleanup": "not_started"},
               "actual_calls": {"rgb_vjps_attempted": 2}, "sources": []}
    scope["write_json"](output / "summary.json", pending)
    def forbidden(*args):
        raise AssertionError("Interrupted continuation must not be retried")
    scope["ensure_runtime"] = forbidden
    assert scope["run_or_show_continuation"](tmp_path, root, b"unused") == pending
    report, archive = scope["package_continuation"](config(), root)
    assert report["actual_calls"] is None and report["cumulative_actual_calls"] is None
    assert report["prior_actual_calls"] == prior["cumulative_actual_calls"]
    assert report["last_recorded_calls"] == pending["actual_calls"]
    assert report["resume_from"] == str(checkpoint)
    assert report["scope"] == "suffix_device_fix_continuation_of_saved_rgb_gradients"
    assert archive.name == "run-continuation-suffix-device-v1-results.zip"
    with zipfile.ZipFile(archive) as saved:
        assert "run/batch/original_failure.pt" in saved.namelist()
        assert "run/continuation_layout_v2/suffix_failure.pt" in saved.namelist()
        assert "run/continuation_suffix_device_v1_summary.json" in saved.namelist()
    assert not (root / "measurement_summary.json").exists()


def test_continuation_requires_saved_batch_before_environment_setup(tmp_path):
    scope = continuation_namespace()
    def forbidden(*args):
        raise AssertionError("Missing original must not launch setup or generation")
    scope["ensure_runtime"] = forbidden
    with pytest.raises(RuntimeError, match="saved continuation_layout_v2 checkpoint"):
        scope["run_or_show_continuation"](tmp_path, tmp_path / "missing", b"unused")


def test_suffix_continuation_uses_current_checkpoint_and_documents_cached_work():
    notebook = build_continuation_notebook()
    scope = continuation_namespace()
    assert scope["RESUME_NAME"] == "continuation_layout_v2"
    assert scope["RESUME_BATCH"] == scope["RUN_ROOT"] / "continuation_layout_v2"
    assert scope["CONTINUATION_NAME"] == "continuation_suffix_device_v1"
    intro = "".join(next(cell for cell in notebook["cells"] if cell["id"] == "intro")["source"])
    assert "8 条 RGB 梯度" in intro and "4 次源末段前向及 8 次潜变量 VJP" in intro
    assert "16 次正式生成" in intro and "32 次正式重建、48 次公开读出" in intro
    prepare = "".join(next(cell for cell in notebook["cells"] if cell["id"] == "prepare_source")["source"])
    assert '"expected_reused_rgb_gradients": 8' in prepare
    assert '"expected_new_rgb_vjps": 0' in prepare
    assert '"expected_new_proxy_reconstructions": 0' in prepare
