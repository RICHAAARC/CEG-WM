"""Single-source notebook packaging and fake launch checks; no GPU/models."""

import base64
import io
import json
from pathlib import Path
import types
import zipfile

import pytest

from scripts.build_channel_precompensation_single_source_notebook import (
    ROOT, OUTPUT, SOURCE_FILES, HELPERS, build_notebook, source_snapshot,
)


pytestmark = pytest.mark.unit


def namespace():
    scope = {}
    exec(compile(HELPERS, "single_source:helpers", "exec"), scope)
    return scope


def prior(scope, root):
    directory = root / scope["RESUME_NAME"]
    directory.mkdir(parents=True)
    scope["write_json"](directory / "config.json", {"source": "retained four-source fixture"})
    scope["write_json"](directory / "summary.json", {
        "cumulative_actual_calls": {"final_generation_attempted": 4, "rgb_vjps_attempted": 8},
        "stages": {"cleanup": "complete"}})
    (directory / "original_evidence.pt").write_bytes(b"unchanged")
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
    assert json.loads(OUTPUT.read_text()) == expected, "Rebuild the single-source notebook after bundled source edits"
    code = [entry for entry in expected["cells"] if entry["cell_type"] == "code"]
    assert "".join(code[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    for entry in code:
        assert entry["execution_count"] is None and entry["outputs"] == []
        compile("".join(entry["source"]), entry["id"], "exec")
    all_code = "\n".join("".join(entry["source"]) for entry in code)
    assert "files.upload(" not in all_code and "CODE_REF =" not in all_code
    assert "experiments.run_channel_precompensation_v1" not in HELPERS
    assert "experiments.run_channel_suffix_diagnostic_v1" not in HELPERS
    embedded = next(entry for entry in code if entry["id"] == "embedded_source")
    scope = {}
    exec("".join(embedded["source"]), scope)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(scope["SOURCE_BUNDLE_B64"]))) as archive:
        assert set(archive.namelist()) == set(SOURCE_FILES)
        assert "src/cegwm/runtime/channel_precompensation_single_source.py" in archive.namelist()
        for name in SOURCE_FILES:
            assert archive.read(name) == (ROOT / name).read_bytes()


def test_notebook_scope_is_single_source_current_zero_and_no_old_gradient_reuse(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("unexpected subprocess"))
    scope = namespace()
    assert scope["RESUME_NAME"] == "continuation_suffix_device_v1"
    assert scope["METHOD_NAME"] == "single_source_method_v1"
    assert scope["SOURCE_ID"] == "source05_smooth_stilllife"
    assert scope["METHODS"] == ("joint", "gradient") and scope["BITS"] == (0, 1)
    notebook = build_notebook(payload=b"unused")
    for entry in notebook["cells"]:
        if entry["cell_type"] == "code":
            compile("".join(entry["source"]), entry["id"], "exec")
    intro = "".join(notebook["cells"][0]["source"])
    assert "6 次 suffix" in intro and "2 次潜变量 VJP、2 次 RGB VJP" in intro
    assert "11 次重建、15 次公开读出" in intro and "完整生成次数为 0" in intro
    assert "旧梯度保留，不跨基点使用" in intro and "另外三个来源本轮不测" in intro
    prepare = "".join(next(entry for entry in notebook["cells"] if entry["id"] == "prepare_source")["source"])
    assert '"old_rgb_gradients_reused": False' in prepare
    assert '"new_zero_reference": "current_suffix"' in prepare


@pytest.mark.parametrize("name", ["../outside.txt", "/outside.txt", "unsafe\\file.py"])
def test_embedded_extraction_rejects_escaping_paths(tmp_path, name):
    scope, buffer = namespace(), io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, "bad")
    with pytest.raises(ValueError, match="unsafe"):
        scope["unpack_source"](buffer.getvalue(), tmp_path / "source")
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.parametrize("summary", [None, {"stages": {"prepare": "started", "cleanup": "not_started"}}])
def test_existing_output_is_read_only_even_without_summary(tmp_path, summary):
    scope, root = namespace(), tmp_path / "run"
    output = root / scope["METHOD_NAME"]
    output.mkdir(parents=True)
    (output / "partial_write.pt").write_bytes(b"retained")
    if summary is not None:
        scope["write_json"](output / "summary.json", summary)
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    scope["ensure_runtime"] = lambda *args: pytest.fail("existing output must not start setup")
    assert scope["run_or_show_single_source"](tmp_path, root, b"unused") == (summary or {})
    assert not (root / (scope["METHOD_NAME"] + "_metadata")).exists()
    assert before == {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}


def test_launch_uses_single_source_module_and_appends_logs_without_overwriting_original(tmp_path):
    scope, root = namespace(), tmp_path / "run"
    original = prior(scope, root)
    old_metadata = root / (scope["METHOD_NAME"] + "_metadata") / "launch_0001"
    old_metadata.mkdir(parents=True)
    (old_metadata / "notebook_error.json").write_text("previous environment setup failure")
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    calls = []
    scope["ensure_runtime"] = lambda *args: {"HF_TOKEN": "fake-only"}

    def launch(command, **kwargs):
        calls.append(command)
        assert command[1:4] == ["-u", "-m", "experiments.run_channel_precompensation_single_source_v1"]
        assert command[command.index("--resume-from") + 1] == str(original)
        assert "--config" not in command and "--execute" in command
        output = Path(command[command.index("--output") + 1])
        assert output == root / "single_source_method_v1" and not output.exists()
        output.mkdir()
        scope["write_json"](output / "summary.json", {"stages": {"cleanup": "complete"}, "actual_calls": {}})
        return FakeProcess(1, "source05 new zero\nfixed four write arms\n")

    scope["subprocess"] = types.SimpleNamespace(Popen=launch, PIPE=-1, STDOUT=-2)
    scope["run_or_show_single_source"](tmp_path, root, b"source snapshot")
    assert len(calls) == 1
    metadata = old_metadata.parent / "launch_0002"
    assert (metadata / "source_snapshot.zip").read_bytes() == b"source snapshot"
    assert scope["read_json"](metadata / "runner_process.json")["returncode"] == 1
    assert (metadata / "runner.log").read_text() == "source05 new zero\nfixed four write arms\n"
    assert all(path.read_bytes() == value for path, value in before.items())


def test_early_failure_preserves_both_streams_returncode_and_unknown_counts(tmp_path, capsys):
    import subprocess
    import sys

    scope, root = namespace(), tmp_path / "run"
    prior(scope, root)
    scope["ensure_runtime"] = lambda *args: {}

    def launch(command, **kwargs):
        return subprocess.Popen([sys.executable, "-u", "-c",
            "import sys; print('reading source05'); print('saved startup detail', file=sys.stderr); sys.exit(7)"], **kwargs)

    scope["subprocess"] = types.SimpleNamespace(Popen=launch, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
    with pytest.raises(RuntimeError, match="runner return code 7; counts unknown"):
        scope["run_or_show_single_source"](tmp_path, root, b"source snapshot")
    metadata = root / "single_source_method_v1_metadata/launch_0001"
    log = (metadata / "runner.log").read_text()
    assert "reading source05" in log and "saved startup detail" in log
    assert "saved startup detail" in capsys.readouterr().out
    assert scope["read_json"](metadata / "runner_process.json")["returncode"] == 7
    report, archive = scope["package_single_source"](root)
    assert report["actual_calls"] is None and report["cumulative_actual_calls"] is None
    assert report["new_counts_status"] == "unknown" and report["counts_terminal"] is False
    assert report["prior_actual_calls"]["final_generation_attempted"] == 4
    with zipfile.ZipFile(archive) as saved:
        assert saved.read("run/single_source_method_v1_metadata/launch_0001/runner.log").decode() == log


def test_package_keeps_new_evidence_and_distinguishes_four_source_prior_from_one_source_calls(tmp_path):
    scope, root = namespace(), tmp_path / "run"
    original = prior(scope, root)
    output = root / scope["METHOD_NAME"]
    output.mkdir()
    (output / "write.pt").write_bytes(b"actual write")
    (output / "external-results.zip").write_bytes(b"not nested")
    (root / "previous-results.zip").write_bytes(b"not included")
    summary = {"stages": {"prepare": "complete", "cleanup": "complete"},
               "actual_calls": {"sd35_suffix_forward_calls": 6, "final_generation_attempted": 0},
               "cumulative_actual_calls": {"sd35_suffix_forward_calls": 14, "final_generation_attempted": 4},
               "method_summaries": {"joint": {"expected_bit_trials": 2}}}
    scope["write_json"](output / "summary.json", summary)
    report, destination = scope["package_single_source"](root)
    assert report["counts_terminal"] is True and report["actual_calls"] == summary["actual_calls"]
    assert report["cumulative_actual_calls"] == summary["cumulative_actual_calls"]
    assert report["prior_actual_calls"]["final_generation_attempted"] == 4
    assert "four-source" in report["prior_calls_scope"] and report["new_calls_scope"] == "source05 only"
    assert report["resume_from"] == str(original)
    with zipfile.ZipFile(destination) as saved:
        assert "run/single_source_method_v1/write.pt" in saved.namelist()
        assert all("continuation_suffix_device_v1/" not in name for name in saved.namelist())
        assert all(not name.endswith(".zip") for name in saved.namelist())


def test_missing_input_cannot_install_dependencies_or_start_models(tmp_path):
    scope = namespace()
    scope["ensure_runtime"] = lambda *args: pytest.fail("missing input must not start setup")
    with pytest.raises(RuntimeError, match="saved continuation_suffix_device_v1 checkpoint"):
        scope["run_or_show_single_source"](tmp_path, tmp_path / "missing", b"unused")


def test_bundle_imports_provider_and_plans_without_model_or_checkpoint_execution(tmp_path):
    import os
    import subprocess
    import sys

    scope = namespace()
    checkout = scope["unpack_source"](source_snapshot(), tmp_path / "source")
    original = tmp_path / "saved"
    original.mkdir()
    # Planning consumes the old four-source config only. Missing captured-state
    # files ensure this path cannot accidentally execute checkpoint/model work.
    settings = json.loads((ROOT / "configs/channel_precompensation_v1_colab.json").read_text())
    scope["write_json"](original / "config.json", settings)
    environment = dict(os.environ, PYTHONPATH=str(checkout / "src") + os.pathsep + str(checkout),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run([sys.executable, "-c",
        "import sys; import cegwm.runtime.channel_precompensation_single_source; "
        "from experiments import run_channel_precompensation_single_source_v1 as runner; "
        "raise SystemExit(runner.main(['--resume-from', sys.argv[1]]))", str(original)],
        cwd=checkout, env=environment, text=True, capture_output=True, check=True)
    plan = json.loads(completed.stdout)
    assert plan["mode"] == "plan_only" and plan["source_id"] == "source05_smooth_stilllife"
    assert len(plan["full_roster"]) == 4
    assert sum(row["scope_status"] == "selected_by_fixed_order" for row in plan["full_roster"]) == 1
    assert plan["methods"] == ["joint", "gradient"] and plan["bits"] == [0, 1]
    budget = plan["planned_budget"]
    assert budget["source_suffix_forward_calls"] == 6 and budget["final_generation_calls"] == 0
    assert budget["final_reconstruction_calls"] == 10 and budget["total_reconstruction_forward_calls"] == 11
    assert budget["public_feature_reads"] == 15 and budget["latent_vjps"] == budget["rgb_vjps"] == 2
