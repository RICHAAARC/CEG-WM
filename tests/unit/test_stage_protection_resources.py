import json
from pathlib import Path
import sys

import pytest

from cegwm.runtime.stage_protection_colab import run_logged
from cegwm.runtime import stage_protection_resources as resources

pytestmark = pytest.mark.unit


def files(directory, oom=0, killed=0):
    directory.mkdir(exist_ok=True)
    (directory / "memory.events").write_text(f"low 0\nhigh 0\noom {oom}\noom_kill {killed}\n")
    (directory / "memory.current").write_text("100")
    (directory / "memory.peak").write_text("200")
    (directory / "memory.max").write_text("max")


def test_cgroup_resolution_and_event_delta_never_imply_process_attribution(tmp_path):
    proc, cg = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    files(cg, 2, 1)
    (proc / "self/cgroup").write_text("0::/container\n")
    (proc / "self/mountinfo").write_text(f"1 0 0:1 /container {cg} rw - cgroup2 cgroup rw\n")
    before = resources.cgroup_memory_snapshot(proc_root=proc)
    files(cg, 3, 2)
    after = resources.cgroup_memory_snapshot(directory=before["cgroup_path"])
    assert before["memory_limit_bytes"] == "max" and after["memory_peak_bytes"] == 200
    assert resources.event_delta(before, after)["oom_kill"] == 1
    files(cg, 0, 0)
    assert resources.event_delta(after, resources.cgroup_memory_snapshot(directory=cg))["oom_kill"] is None


def test_missing_and_denied_events_remain_unknown(tmp_path, monkeypatch):
    missing = resources.cgroup_memory_snapshot(directory=tmp_path)
    assert missing["memory_events"] is missing["memory_limit_bytes"] is None
    assert resources.event_delta(missing, missing) is None
    original = Path.read_text
    def denied(path, *args, **kwargs):
        if path.name == "memory.events":
            raise PermissionError("synthetic denied")
        return original(path, *args, **kwargs)
    files(tmp_path)
    monkeypatch.setattr(Path, "read_text", denied)
    state = resources.cgroup_memory_snapshot(directory=tmp_path)
    assert state["memory_events"] is None and "PermissionError" in state["errors"]["memory.events"]


def test_parent_keeps_resource_record_when_child_is_sigkilled(tmp_path, monkeypatch):
    cg = tmp_path / "cg"
    files(cg)
    snapshot = resources.cgroup_memory_snapshot
    monkeypatch.setattr(resources, "cgroup_memory_snapshot", lambda **kwargs: snapshot(directory=cg))
    status = run_logged([sys.executable, "-u", "-c",
                         "import os,signal; print('before kill', flush=True); os.kill(os.getpid(), signal.SIGKILL)"],
                        cwd=tmp_path, output=tmp_path)
    assert status == -9
    records = [json.loads(line) for line in (tmp_path / "runner_resources.jsonl").read_text().splitlines()]
    assert records[0]["event"] == "before_launch" and records[-1]["event"] == "child_exit"
    assert records[-1]["returncode"] == -9 and records[-1]["child_pid"] > 0
    assert records[-1]["memory_events_delta"]["oom_kill"] == 0  # Read twice, not inferred from -9.
    assert "not process-specific OOM attribution" in records[-1]["scope"]
    assert "before kill" in (tmp_path / "runner.log").read_text()


def test_source_rss_failure_is_unknown_and_does_not_change_execution(monkeypatch, capsys):
    def denied(*args, **kwargs):
        raise PermissionError("unavailable")
    monkeypatch.setattr(Path, "read_text", denied)
    resources.source_memory_event("before_step", step=18, saved_tensors={"activation_cpu_copy_bytes": 12})
    record = json.loads(capsys.readouterr().out.removeprefix("SOURCE_MEMORY "))
    assert record["rss_bytes"] is None and record["step"] == 18
    assert record["errors"]["rss"].startswith("PermissionError")


def test_unwritable_telemetry_neither_blocks_child_nor_masks_exit(tmp_path, monkeypatch):
    original = Path.open
    def deny_resource_log(path, *args, **kwargs):
        if path.name == "runner_resources.jsonl":
            raise PermissionError("synthetic telemetry log denied")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", deny_resource_log)
    status = run_logged([sys.executable, "-u", "-c",
                         "print('child did run', flush=True); raise RuntimeError('original child error')"],
                        cwd=tmp_path, output=tmp_path)
    assert status == 1
    log = (tmp_path / "runner.log").read_text()
    assert "child did run" in log and "original child error" in log
    assert "RESOURCE_RECORD_UNAVAILABLE" in log and "synthetic telemetry log denied" in log
def test_cgroup_parent_escape_stays_unavailable(tmp_path):
    from cegwm.runtime.stage_protection_resources import resolve_cgroup
    proc = tmp_path / "self"
    proc.mkdir()
    (proc / "cgroup").write_text("0::/../../jupyter-children\n")
    (proc / "mountinfo").write_text("1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n")
    directory, error = resolve_cgroup(proc_root=tmp_path)
    assert directory is None and "escapes visible namespace" in error
