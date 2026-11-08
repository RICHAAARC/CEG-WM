"""Static and simulated Colab checks; no network, model, CUDA, or subprocess runs."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/reconstruction_channel_v1_colab.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(cell_id):
    return "".join(next(cell for cell in _notebook()["cells"] if cell["id"] == cell_id)["source"])


def _namespace():
    namespace = {"__name__": "simulated_colab_notebook"}
    for cell_id in ("source_helpers", "batch_helpers", "summary_helpers"):
        exec(compile(_code(cell_id), f"notebook:{cell_id}", "exec"), namespace)
    return namespace


def _config():
    return json.loads((ROOT / "configs/reconstruction_channel_v1_colab.json").read_text(encoding="utf-8"))


def _colab(monkeypatch, *, files=None, userdata=None, drive=None):
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.files, colab.userdata, colab.drive = files, userdata, drive
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def _fake_git(namespace, *, resolved_commit="a" * 40, fail_operation=None):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert command[0] == "git"
        assert kwargs == {"check": True, "text": True, "capture_output": True}
        if fail_operation and fail_operation in command:
            raise subprocess.CalledProcessError(128, command, stderr="fatal: requested remote reference is unavailable")
        if command[1] == "clone":
            checkout = Path(command[-1])
            (checkout / ".git").mkdir(parents=True)
            (checkout / "configs").mkdir()
            (checkout / "configs/reconstruction_channel_v1_colab.json").write_text(json.dumps(_config()), encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout=resolved_commit + "\n" if "rev-parse" in command else "", stderr="")
    namespace["subprocess"] = types.SimpleNamespace(run=run, CalledProcessError=subprocess.CalledProcessError)
    return calls


def _write_batch(output, config, *, complete=True, diagnostic_invalid=False, response=0.0, extra_calls=0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config), encoding="utf-8")
    n, seed_count = len(config["sources"]), len(config["reconstruction"]["seeds"])
    counts = {
        "generation_attempted": 3 * n + extra_calls, "generation_returned": 3 * n,
        "reconstruction_attempted": 3 * n * seed_count, "reconstruction_returned": 3 * n * seed_count,
        "feature_reads_attempted": 3 * n * (1 + seed_count), "feature_reads_saved": 3 * n * (1 + seed_count),
    }
    rows = []
    for source in config["sources"]:
        directory = output / source["source_id"]
        directory.mkdir()
        row = {
            "source_id": source["source_id"], "matched_state": complete, "collection_complete": complete,
            "stage": "complete",
            "status": "failed" if not complete else ("incomplete" if diagnostic_invalid else "measured"),
            "measurement": {"status": "INCOMPLETE" if diagnostic_invalid else "MEASURED",
                            "clean": {"response_secant": [response], "response_secant_l2": abs(response)}},
        }
        (directory / "measurement.json").write_text(json.dumps(row), encoding="utf-8")
        for channel in ("clean", *(f"reconstruction_{seed}" for seed in config["reconstruction"]["seeds"])):
            (directory / channel).mkdir()
            for arm in ("zero", "plus", "minus"):
                for extension in (".png", ".pt"):
                    (directory / channel / (arm + extension)).write_bytes(b"simulated evidence")
        rows.append(row)
    summary = {"mode": "measurement", "actual_calls": counts, "sources": rows,
               "stages": {"generation_setup": "complete", "generation": "complete",
                          "generation_release": "complete", "reconstruction_setup": "complete",
                          "reconstruction": "complete" if complete else "completed_with_errors", "cleanup": "complete"}}
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def _fake_execution(namespace, *, complete=True, diagnostic_invalid=False, response=0.0, extra_calls=0):
    calls = []
    namespace["ensure_runtime"] = lambda checkout, root: {"PYTHONPATH": str(checkout / "src"), "HF_TOKEN": "test-only"}
    def run(command, **kwargs):
        assert "--execute" in command
        assert kwargs["check"] is False
        assert kwargs["env"]["HF_TOKEN"] == "test-only"
        config = json.loads(Path(command[command.index("--config") + 1]).read_text())
        output = Path(command[command.index("--output") + 1])
        calls.append((output.name, [source["source_id"] for source in config["sources"]]))
        _write_batch(output, config, complete=complete, diagnostic_invalid=diagnostic_invalid,
                     response=response, extra_calls=extra_calls)
        return types.SimpleNamespace(returncode=1 if diagnostic_invalid or not complete else 0)
    namespace["subprocess"] = types.SimpleNamespace(run=run)
    return calls


def test_notebook_is_unexecuted_valid_and_every_code_cell_compiles():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "".join(code_cells[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"notebook:{cell['id']}", "exec")
    try:
        import nbformat
    except ImportError:
        pass
    else:
        nbformat.validate(nbformat.from_dict(notebook))
    combined = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "files.upload" not in combined
    assert "BUNDLE_PATH" not in combined and "BUNDLE_NAME" not in combined
    assert 'DRIVE_PARENT / "inputs"' not in combined
    assert '"https://github.com/RICHAAARC/CEG-WM.git"' in combined
    assert '"/content/cegwm-reconstruction-channel-v1-published"' in combined
    assert "CEG_WM_ROOT_KEY" not in combined
    assert "torch==" not in combined
    ids = [cell["id"] for cell in notebook["cells"]]
    assert ids.index("summary_helpers") < ids.index("execute_batches")


def test_mount_cell_calls_only_the_required_colab_mount(monkeypatch):
    calls = []
    _colab(monkeypatch, drive=types.SimpleNamespace(mount=calls.append))
    exec(_code("drive_mount"), {})
    assert calls == ["/content/drive"]


def test_source_bootstrap_uses_public_git_and_keeps_fixed_config_read_only(tmp_path):
    namespace = _namespace()
    namespace.update(CHECKOUT=tmp_path / "published", RUN_ROOT=tmp_path / "results")
    calls = _fake_git(namespace)
    exec(_code("source_setup"), namespace)
    checkout = str(namespace["CHECKOUT"])
    expected = [
        ["git", "clone", "--no-checkout", namespace["SOURCE_REPO"], checkout],
        ["git", "-C", checkout, "fetch", "origin", namespace["CODE_REF"]],
        ["git", "-C", checkout, "checkout", "--detach", "FETCH_HEAD"],
        ["git", "-C", checkout, "rev-parse", "HEAD"],
    ]
    assert [command for command, _ in calls] == expected
    assert namespace["CONFIG"] == _config()
    config_path = namespace["CHECKOUT"] / namespace["CONFIG_RELATIVE"]
    before = config_path.read_bytes()
    exec(_code("source_setup"), namespace)
    assert [command for command, _ in calls[4:]] == expected[1:]
    assert config_path.read_bytes() == before
    metadata = json.loads((namespace["RUN_ROOT"] / "code_version.json").read_text())
    assert metadata["resolved_commit"] == "a" * 40
    assert metadata["requested_ref"] == namespace["CODE_REF"]


@pytest.mark.parametrize("operation", ["clone", "fetch"])
def test_unavailable_remote_has_clear_failure_before_any_measurement(tmp_path, operation):
    namespace = _namespace()
    calls = _fake_git(namespace, fail_operation=operation)
    checkout, run_root = tmp_path / "published", tmp_path / "results"
    with pytest.raises(RuntimeError, match="Could not fetch the published source") as caught:
        namespace["fetch_source"](namespace["SOURCE_REPO"], namespace["CODE_REF"], checkout, run_root)
    assert namespace["SOURCE_REPO"] in str(caught.value)
    assert namespace["CODE_REF"] in str(caught.value)
    assert not run_root.exists()
    assert not any("--execute" in command for command, _ in calls)


def test_bootstrap_preserves_existing_non_repository_data(tmp_path):
    namespace = _namespace()
    checkout = tmp_path / "published"
    checkout.mkdir()
    user_file = checkout / "user_data.txt"
    user_file.write_text("keep me")
    calls = _fake_git(namespace)
    with pytest.raises(RuntimeError, match="non-repository data"):
        namespace["fetch_source"](namespace["SOURCE_REPO"], namespace["CODE_REF"], checkout, tmp_path / "results")
    assert user_file.read_text() == "keep me"
    assert calls == []


def test_first_code_version_record_is_preserved_without_hash_admission_gate(tmp_path):
    namespace = _namespace()
    checkout, run_root = tmp_path / "published", tmp_path / "results"
    (run_root / "first_source").mkdir(parents=True)
    _fake_git(namespace, resolved_commit="a" * 40)
    namespace["fetch_source"](namespace["SOURCE_REPO"], "first-ref", checkout, run_root)
    first_record = (run_root / "code_version.json").read_bytes()
    _fake_git(namespace, resolved_commit="b" * 40)
    current = namespace["fetch_source"](namespace["SOURCE_REPO"], "second-ref", checkout, run_root)
    assert current["resolved_commit"] == "b" * 40
    assert (run_root / "code_version.json").read_bytes() == first_record
    metadata = json.loads(first_record)
    assert metadata["preexisting_batch_directories"] == ["first_source"]
    assert "not proof of the producer" in metadata["scope"]


@pytest.mark.parametrize("response,invalid", [(0.0, False), (-3.0, False), (0.0, True)])
def test_zero_negative_and_quantization_invalid_diagnostics_do_not_filter_sources(tmp_path, response, invalid):
    namespace, config = _namespace(), _config()
    calls = _fake_execution(namespace, response=response, diagnostic_invalid=invalid)
    first, rest = namespace["run_fixed_roster"](config, tmp_path / "checkout", tmp_path / "results")
    identifiers = [source["source_id"] for source in config["sources"]]
    assert calls == [("first_source", identifiers[:1]), ("remaining_sources", identifiers[1:])]
    assert len(first["sources"]) == 1 and len(rest["sources"]) == 3
    assert config == _config()


def test_complete_existing_batches_are_read_only_without_setup_or_subprocess(tmp_path):
    namespace, config = _namespace(), _config()
    first, rest = copy.deepcopy(config), copy.deepcopy(config)
    first["sources"], rest["sources"] = config["sources"][:1], config["sources"][1:]
    run_root = tmp_path / "results"
    _write_batch(run_root / "first_source", first, diagnostic_invalid=True)
    _write_batch(run_root / "remaining_sources", rest)
    (run_root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    before = {str(path): path.read_bytes() for path in run_root.rglob("*") if path.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError("existing completed data must not trigger setup or a subprocess")
    namespace["ensure_runtime"] = forbidden
    namespace["subprocess"] = types.SimpleNamespace(run=forbidden)
    namespace["run_fixed_roster"](config, tmp_path / "checkout", run_root)
    after = {str(path): path.read_bytes() for path in run_root.rglob("*") if path.is_file()}
    assert before == after


@pytest.mark.parametrize("existing_kind", ["failed", "started", "changed_config", "missing_feature"])
def test_existing_incomplete_or_changed_batch_never_retries(tmp_path, existing_kind):
    namespace, config = _namespace(), _config()
    first = copy.deepcopy(config)
    first["sources"] = config["sources"][:1]
    output = tmp_path / "results/first_source"
    if existing_kind == "started":
        output.mkdir(parents=True)
    else:
        _write_batch(output, first, complete=existing_kind != "failed")
        if existing_kind == "changed_config":
            stored = copy.deepcopy(first)
            stored["sources"][0]["seed"] += 1
            (output / "config.json").write_text(json.dumps(stored))
        if existing_kind == "missing_feature":
            (output / first["sources"][0]["source_id"] / "clean/plus.pt").unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError("existing incomplete batch must not run")
    namespace["ensure_runtime"] = forbidden
    namespace["subprocess"] = types.SimpleNamespace(run=forbidden)
    with pytest.raises(RuntimeError, match="retained without retry"):
        namespace["run_fixed_roster"](config, tmp_path / "checkout", tmp_path / "results")
    assert not (tmp_path / "results/remaining_sources").exists()


def test_new_collection_failure_stops_rest_and_still_packages_fixed_roster(tmp_path):
    namespace, config = _namespace(), _config()
    calls = _fake_execution(namespace, complete=False)
    run_root = tmp_path / "measurement-20261108"
    with pytest.raises(RuntimeError, match="remaining sources will not start"):
        namespace["run_fixed_roster"](config, tmp_path / "checkout", run_root)
    assert len(calls) == 1
    aggregate, archive_path = namespace["aggregate_and_package"](config, run_root)
    assert aggregate["planned_calls"] == {"generation_calls": 12, "reconstruction_calls": 12, "public_feature_calls": 24}
    assert [row["status"] for row in aggregate["sources"]] == ["failed", "missing", "missing", "missing"]
    assert len(aggregate["sources"]) == 4
    assert aggregate["actual_calls"]["generation_attempted"] == 3
    assert archive_path.parent == run_root.parent
    with zipfile.ZipFile(archive_path) as archive:
        assert any(name.endswith("measurement_summary.json") for name in archive.namelist())
        assert not any(name.endswith("-results.zip") for name in archive.namelist())


def test_extra_calls_are_not_accepted_as_complete_collection(tmp_path):
    namespace, config = _namespace(), _config()
    calls = _fake_execution(namespace, extra_calls=1)
    with pytest.raises(RuntimeError, match="not exactly the planned calls"):
        namespace["run_fixed_roster"](config, tmp_path / "checkout", tmp_path / "results")
    assert len(calls) == 1


def test_failure_before_runner_output_is_recorded_and_not_retried(tmp_path):
    namespace, config, calls = _namespace(), _config(), []
    namespace["ensure_runtime"] = lambda *args: {}
    def no_output(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(returncode=2)
    namespace["subprocess"] = types.SimpleNamespace(run=no_output)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            namespace["run_fixed_roster"](config, tmp_path / "checkout", tmp_path / "results")
    assert len(calls) == 1
    saved = json.loads((tmp_path / "results/first_source/summary.json").read_text())
    assert saved["sources"][0]["status"] == "launch_failed"
    aggregate, _ = namespace["aggregate_and_package"](config, tmp_path / "results")
    assert aggregate["actual_calls"]["generation_attempted"] is None


def test_dependency_setup_is_simulated_pinned_and_never_persists_token(tmp_path, monkeypatch):
    namespace, commands, secret_requests = _namespace(), [], []
    fake_torch = types.SimpleNamespace(__version__="2.6.0+cu124", cuda=types.SimpleNamespace(is_available=lambda: True))
    namespace["importlib"] = types.SimpleNamespace(import_module=lambda name: fake_torch if name == "torch" else None)
    namespace["version"] = lambda name: "simulated-" + name
    namespace["subprocess"] = types.SimpleNamespace(run=lambda argv, **kwargs: commands.append((argv, kwargs)))
    def secret(name):
        secret_requests.append(name)
        return "never-persist-this-token"
    _colab(monkeypatch, userdata=types.SimpleNamespace(get=secret))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    environment = namespace["ensure_runtime"](tmp_path / "checkout", tmp_path / "results")
    namespace["ensure_runtime"](tmp_path / "checkout", tmp_path / "results")
    assert len(commands) == 1
    argv, kwargs = commands[0]
    assert kwargs == {"check": True}
    assert argv[1:4] == ["-m", "pip", "install"]
    assert argv[4:] == ["-r", str(tmp_path / "checkout/requirements-reconstruction-channel-colab.txt")]
    requirements = (ROOT / "requirements-reconstruction-channel-colab.txt").read_text()
    assert "diffusers==0.39.0" in requirements and "transformers==4.57.6" in requirements and "accelerate==1.1.1" in requirements
    assert not any(argument == "torch" or argument.startswith("torch=") for argument in argv)
    assert secret_requests == ["HF_TOKEN"]
    assert environment["HF_TOKEN"] == "never-persist-this-token"
    assert str(tmp_path / "checkout/src") in environment["PYTHONPATH"]
    environment_json = (tmp_path / "results/environment.json").read_text()
    assert "never-persist-this-token" not in environment_json and "HF_TOKEN" not in environment_json


@pytest.mark.parametrize("torch_version,cuda", [("2.5.0", True), ("2.6.0", False)])
def test_incompatible_colab_runtime_stops_before_install_without_replacing_torch(tmp_path, torch_version, cuda):
    namespace = _namespace()
    fake_torch = types.SimpleNamespace(__version__=torch_version, cuda=types.SimpleNamespace(is_available=lambda: cuda))
    namespace["importlib"] = types.SimpleNamespace(import_module=lambda _: fake_torch)
    def forbidden(*args, **kwargs):
        raise AssertionError("incompatible runtime must not install or execute")
    namespace["subprocess"] = types.SimpleNamespace(run=forbidden)
    with pytest.raises(RuntimeError, match="standard Colab GPU"):
        namespace["ensure_runtime"](tmp_path / "checkout", tmp_path / "results")
    assert (tmp_path / "results/environment.json").is_file()


def test_aggregate_does_not_relabel_a_different_batch_config_as_current_roster(tmp_path):
    namespace, config = _namespace(), _config()
    first = copy.deepcopy(config)
    first["sources"] = first["sources"][:1]
    first["sources"][0]["seed"] += 1
    run_root = tmp_path / "results"
    original = _write_batch(run_root / "first_source", first)
    aggregate, _ = namespace["aggregate_and_package"](config, run_root)
    assert aggregate["sources"][0]["status"] == "config_mismatch"
    assert aggregate["sources"][0]["collection_complete"] is False
    assert aggregate["batches"]["first_source"]["summary"] == original
    assert aggregate["batches"]["first_source"]["saved_config"] == first
    assert aggregate["actual_calls"]["generation_attempted"] is None
    assert (run_root / "config.json").read_text() != (run_root / "first_source/config.json").read_text()


@pytest.mark.parametrize("stage_marker", ["global_started", "row_started", "cleanup_not_started"])
def test_interrupted_saved_counts_remain_unknown_and_cannot_release_rest(tmp_path, stage_marker):
    namespace, config = _namespace(), _config()
    first = copy.deepcopy(config)
    first["sources"] = first["sources"][:1]
    run_root = tmp_path / "results"
    summary = _write_batch(run_root / "first_source", first)
    if stage_marker == "global_started":
        summary["stages"]["reconstruction"] = "started"
    elif stage_marker == "row_started":
        summary["sources"][0]["stage"] = "reconstruction"
    else:
        summary["stages"]["cleanup"] = "not_started"
    (run_root / "first_source/summary.json").write_text(json.dumps(summary), encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("interrupted batch must not release the remaining sources")
    namespace["ensure_runtime"] = forbidden
    with pytest.raises(RuntimeError, match="terminal"):
        namespace["run_fixed_roster"](config, tmp_path / "checkout", run_root)
    aggregate, _ = namespace["aggregate_and_package"](config, run_root)
    assert aggregate["actual_calls"]["generation_attempted"] is None
    assert aggregate["unknown_actual_call_batches"] == ["first_source"]
    assert aggregate["last_recorded_counts"]["first_source"] == summary["actual_calls"]


def test_full_run_config_is_checked_before_any_new_batch(tmp_path):
    namespace, config = _namespace(), _config()
    run_root = tmp_path / "results"
    run_root.mkdir()
    other = copy.deepcopy(config)
    other["sources"][3]["seed"] += 1
    (run_root / "config.json").write_text(json.dumps(other), encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("different full config must stop before setup")
    namespace["ensure_runtime"] = forbidden
    with pytest.raises(RuntimeError, match="Run-root config differs"):
        namespace["run_fixed_roster"](config, tmp_path / "checkout", run_root)
    assert not (run_root / "first_source").exists()
