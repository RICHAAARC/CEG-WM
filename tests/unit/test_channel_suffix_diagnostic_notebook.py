"""Independent diagnostic notebook packaging/control flow; no models or GPU."""

import base64
import io
import json
from pathlib import Path
import types
import zipfile

import pytest

from scripts.build_channel_suffix_diagnostic_notebook import (
    ROOT, OUTPUT, SOURCE_FILES, HELPERS, build_notebook, source_snapshot,
)


pytestmark = pytest.mark.unit


def namespace():
    scope = {}
    exec(compile(HELPERS, "suffix_diagnostic:helpers", "exec"), scope)
    return scope


def prior(scope, root, *, counts=None):
    directory = root / scope["RESUME_NAME"]
    directory.mkdir(parents=True)
    scope["write_json"](directory / "config.json", {"source": "retained fixture"})
    scope["write_json"](directory / "summary.json", {
        "cumulative_actual_calls": counts or {"final_generation_attempted": 4, "rgb_vjps_attempted": 8},
        "stages": {"cleanup": "complete"}})
    (directory / "original_trace.pt").write_bytes(b"unchanged evidence")
    return directory


class FakeProcess:
    def __init__(self, returncode, text):
        self.returncode, self.stdout = returncode, io.StringIO(text)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stdout.close()

    def wait(self):
        return self.returncode


def test_delivered_notebook_is_current_unexecuted_and_drive_mount_is_first_code():
    expected = build_notebook()
    assert json.loads(OUTPUT.read_text()) == expected, "Rebuild the diagnostic notebook after source edits"
    code = [entry for entry in expected["cells"] if entry["cell_type"] == "code"]
    assert "".join(code[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    for entry in code:
        assert entry["execution_count"] is None and entry["outputs"] == []
        compile("".join(entry["source"]), entry["id"], "exec")
    all_code = "\n".join("".join(entry["source"]) for entry in code)
    assert "files.upload(" not in all_code and "CODE_REF =" not in all_code
    assert "experiments.run_channel_precompensation_v1" not in HELPERS
    embedded = next(entry for entry in code if entry["id"] == "embedded_source")
    scope = {}
    exec("".join(embedded["source"]), scope)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(scope["SOURCE_BUNDLE_B64"]))) as archive:
        assert set(archive.namelist()) == set(SOURCE_FILES)
        for name in SOURCE_FILES:
            assert archive.read(name) == (ROOT / name).read_bytes()


def test_notebook_fixed_scope_and_helpers_do_not_start_models(monkeypatch):
    # Executing only the helper definitions must not start subprocesses or load
    # checkpoints. No torch/diffusers import is needed by these definitions.
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("unexpected subprocess"))
    scope = namespace()
    assert scope["RESUME_NAME"] == "continuation_suffix_device_v1"
    assert scope["DIAGNOSTIC_NAME"] == "suffix_diagnostic_v1"
    assert scope["SOURCE_ID"] == "source05_smooth_stilllife"
    assert scope["ARM_ORDER"] == ("NG1", "NG2", "G")
    intro = "".join(build_notebook(payload=b"unused")["cells"][0]["source"])
    assert "3 次 suffix" in intro and "3 个末步" in intro and "3 次 VAE decode" in intro
    assert "完整生成、SDXL、VJP、公开读出均为 0" in intro and "没有反向" in intro


@pytest.mark.parametrize("name", ["../outside.txt", "/outside.txt", "unsafe\\file.py"])
def test_embedded_extraction_rejects_escaping_paths(tmp_path, name):
    scope, buffer = namespace(), io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, "bad")
    with pytest.raises(ValueError, match="unsafe"):
        scope["unpack_source"](buffer.getvalue(), tmp_path / "source")
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.parametrize("summary", [None, {"stages": {"traces": "started", "cleanup": "not_started"}}])
def test_existing_output_is_read_only_even_without_summary(tmp_path, summary):
    scope, root = namespace(), tmp_path / "run"
    output = root / scope["DIAGNOSTIC_NAME"]
    output.mkdir(parents=True)
    (output / "trace.pt").write_bytes(b"retained")
    if summary is not None:
        scope["write_json"](output / "summary.json", summary)
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    scope["ensure_runtime"] = lambda *args: pytest.fail("existing output must not start environment setup")
    result = scope["run_or_show_diagnostic"](tmp_path, root, b"unused")
    assert result == (summary or {})
    assert not (root / (scope["DIAGNOSTIC_NAME"] + "_metadata")).exists()
    assert before == {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}


def test_launch_has_fixed_paths_and_appends_metadata_without_replacing_prior_evidence(tmp_path):
    scope, root = namespace(), tmp_path / "run"
    original = prior(scope, root)
    old_metadata = root / (scope["DIAGNOSTIC_NAME"] + "_metadata") / "launch_0001"
    old_metadata.mkdir(parents=True)
    (old_metadata / "notebook_error.json").write_text("saved environment setup failure")
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    calls = []
    scope["ensure_runtime"] = lambda *args: {"HF_TOKEN": "fake-only"}

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        assert command[1:4] == ["-u", "-m", "experiments.run_channel_suffix_diagnostic_v1"]
        assert command[command.index("--resume-from") + 1] == str(original)
        assert "--config" not in command and "--execute" in command
        output = Path(command[command.index("--output") + 1])
        assert output == root / "suffix_diagnostic_v1" and not output.exists()
        output.mkdir()
        scope["write_json"](output / "summary.json", {"stages": {"cleanup": "complete"}, "actual_calls": {}})
        return FakeProcess(0, "NG1\nNG2\nG forward only\n")

    scope["subprocess"] = types.SimpleNamespace(Popen=launch, PIPE=-1, STDOUT=-2)
    scope["run_or_show_diagnostic"](tmp_path, root, b"source snapshot")
    assert len(calls) == 1
    metadata = old_metadata.parent / "launch_0002"
    assert (metadata / "source_snapshot.zip").read_bytes() == b"source snapshot"
    assert scope["read_json"](metadata / "runner_process.json")["returncode"] == 0
    assert (metadata / "runner.log").read_text() == "NG1\nNG2\nG forward only\n"
    assert all(path.read_bytes() == value for path, value in before.items())


def test_early_failure_keeps_stdout_stderr_returncode_and_marks_counts_unknown(tmp_path, capsys):
    import subprocess
    import sys

    scope, root = namespace(), tmp_path / "run"
    prior(scope, root)
    scope["ensure_runtime"] = lambda *args: {}

    def launch(command, **kwargs):
        # A tiny process exercises the logging transport only; never invokes a
        # project execute path, imports a model, or accesses GPU/network.
        return subprocess.Popen([sys.executable, "-u", "-c",
            "import sys; print('loading saved capture'); print('decoder error detail', file=sys.stderr); sys.exit(7)"],
            **kwargs)

    scope["subprocess"] = types.SimpleNamespace(Popen=launch, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
    with pytest.raises(RuntimeError, match="runner return code 7; counts unknown"):
        scope["run_or_show_diagnostic"](tmp_path, root, b"source snapshot")
    metadata = root / "suffix_diagnostic_v1_metadata/launch_0001"
    log = (metadata / "runner.log").read_text()
    assert "loading saved capture" in log and "decoder error detail" in log
    visible = capsys.readouterr().out
    assert "loading saved capture" in visible and "decoder error detail" in visible
    assert scope["read_json"](metadata / "runner_process.json")["returncode"] == 7
    report, archive = scope["package_diagnostic"](root)
    assert report["actual_calls"] is None and report["cumulative_actual_calls"] is None
    assert report["new_counts_status"] == "unknown" and report["counts_terminal"] is False
    assert report["prior_actual_calls"]["final_generation_attempted"] == 4
    with zipfile.ZipFile(archive) as saved:
        assert saved.read("run/suffix_diagnostic_v1_metadata/launch_0001/runner.log").decode() == log


def test_package_is_small_self_contained_diagnostic_and_preserves_prior_counts(tmp_path):
    scope, root = namespace(), tmp_path / "run"
    original = prior(scope, root, counts={"final_generation_attempted": 4, "sd35_suffix_forward_calls": 8})
    output = root / scope["DIAGNOSTIC_NAME"]
    (output / "NG1").mkdir(parents=True)
    (output / "NG1/trace.pt").write_bytes(b"diagnostic trace")
    (output / "baseline.pt").write_bytes(b"copied reference capture")
    (output / "reference_zero.png").write_bytes(b"copied reference RGB")
    (output / "external-results.zip").write_bytes(b"must not nest an external result ZIP")
    (root / "previous-results.zip").write_bytes(b"outside diagnostic")
    summary = {"stages": {"load": "complete", "traces": "complete", "cleanup": "complete"},
               "actual_calls": {"sd35_suffix_forward_calls": 3, "final_generation_attempted": 0},
               "cumulative_actual_calls": {"sd35_suffix_forward_calls": 11, "final_generation_attempted": 4},
               "classification": {"kind": "diagnostic_only"}}
    scope["write_json"](output / "summary.json", summary)
    report, destination = scope["package_diagnostic"](root)
    assert report["counts_terminal"] is True
    assert report["actual_calls"] == summary["actual_calls"]
    assert report["cumulative_actual_calls"] == summary["cumulative_actual_calls"]
    assert report["prior_actual_calls"]["sd35_suffix_forward_calls"] == 8
    assert report["resume_from"] == str(original)
    with zipfile.ZipFile(destination) as saved:
        names = saved.namelist()
        assert "run/suffix_diagnostic_v1/NG1/trace.pt" in names
        assert "run/suffix_diagnostic_v1/baseline.pt" in names
        assert "run/suffix_diagnostic_v1/reference_zero.png" in names
        assert all("continuation_suffix_device_v1/" not in name for name in names)
        assert all(not name.endswith(".zip") for name in names)


def test_missing_input_cannot_install_dependencies_or_launch_models(tmp_path):
    scope = namespace()
    scope["ensure_runtime"] = lambda *args: pytest.fail("missing input must not start setup")
    with pytest.raises(RuntimeError, match="saved continuation_suffix_device_v1 checkpoint"):
        scope["run_or_show_diagnostic"](tmp_path, tmp_path / "missing", b"unused")


def test_embedded_bundle_plan_is_readable_in_isolated_cpu_process(tmp_path):
    import os
    import subprocess
    import sys
    from experiments import run_channel_precompensation_v1 as collector
    from test_channel_precompensation_runner import PreWriteProvider, config

    settings = config()
    settings["generation"].update(num_inference_steps=20, injection_step=18)
    settings["sources"][0]["source_id"] = "source05_smooth_stilllife"
    original = tmp_path / "original"
    collector.execute_measurement(settings, original, provider=PreWriteProvider(settings))
    scope = namespace()
    checkout = scope["unpack_source"](source_snapshot(), tmp_path / "source")
    environment = dict(os.environ, PYTHONPATH=str(checkout / "src") + os.pathsep + str(checkout),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run([sys.executable, "-m", "experiments.run_channel_suffix_diagnostic_v1",
                                "--resume-from", str(original)],
                               cwd=checkout, env=environment, text=True, capture_output=True, check=True)
    plan = json.loads(completed.stdout)
    assert plan["mode"] == "plan_only" and plan["source_id"] == "source05_smooth_stilllife"
    assert [row["arm"] for row in plan["arms"]] == ["NG1", "NG2", "G"]
    assert plan["planned_budget"]["suffix_forwards"] == 3
    assert all(plan["planned_budget"][key] == 0 for key in ("full_generation", "reconstruction", "public_reads", "vjps"))
