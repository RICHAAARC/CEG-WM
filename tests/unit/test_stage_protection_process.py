import json
import os
from pathlib import Path
import signal

import pytest
import torch

from cegwm.runtime.stage_protection_process import ProcessProvider, cleanup_owned_scratch
from cegwm.runtime.stage_protection_store import OperationStore, Unavailable, atomic_json

pytestmark = pytest.mark.unit


class TinyWorkerProvider:
    """Importable by real subprocesses; no model or network calls."""
    def __init__(self, config, *, counters, disk_storage, resource_observer):
        self.config, self.counters, self.disk = config, counters, disk_storage

    def read(self, image):
        self.counters["fake_read_attempted"] = self.counters.get("fake_read_attempted", 0) + 1
        packed = self.disk.pack(torch.ones(1024))
        if self.config.get("kill"):
            os.kill(os.getpid(), signal.SIGKILL)
        self.counters["fake_read_returned"] = self.counters.get("fake_read_returned", 0) + 1
        return dict(feature=self.disk.unpack(packed), score=1., worker_pid=os.getpid())


@pytest.fixture
def process_setup(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(root / "src"), str(root), str(Path(__file__).parent)]))
    def create(config=None):
        config = config or {}
        store = OperationStore(tmp_path / "output", config)
        provider = ProcessProvider(config, counters=store.counters, directory=store.directory,
            scratch_root=tmp_path / "scratch", factory="test_stage_protection_process:TinyWorkerProvider")
        return store, provider
    return create


def test_models_only_in_separate_reaped_workers_and_cached_operation_no_compute(process_setup, tmp_path):
    store, provider = process_setup()
    first = store.call("source/clean", lambda: provider.read(None))
    again = store.call("source/clean", lambda: pytest.fail("cache must not start worker"))
    second = store.call("source2/clean", lambda: provider.read(None))
    assert first["worker_pid"] == again["worker_pid"] != second["worker_pid"] != os.getpid()
    for result in (first, second):
        with pytest.raises(ProcessLookupError):
            os.kill(result["worker_pid"], 0)
    receipts = list((store.directory / "workers").glob("*/parent_status.json"))
    assert len(receipts) == 2 and store.counters["fake_read_attempted"] == 2
    for path in receipts:
        record = json.loads(path.read_text())
        assert record["status"] == "RETURNED" and record["child_reaped"] and record["scratch_removed"]
        assert record["disk_saved_tensors"]["saved_files"] == 1
    assert not list((tmp_path / "scratch").glob("job-*"))
    assert not hasattr(provider, "_source") and not hasattr(provider, "_vae")


def test_kill_retains_exit_partial_cost_and_cleans_scratch(process_setup, tmp_path):
    store, provider = process_setup({"kill": True})
    with pytest.raises(Unavailable, match="exit -9"):
        store.call("source/clean", lambda: provider.read(None))
    assert store.counters["fake_read_attempted"] == 1
    assert store.counters.get("fake_read_returned", 0) == 0
    record = json.loads(next((store.directory / "workers").glob("*/parent_status.json")).read_text())
    assert record["returncode"] == -9 and record["child_reaped"] and record["scratch_removed"]
    assert record["disk_saved_tensors"]["scratch_peak_bytes"] > 0
    assert not list((tmp_path / "scratch").glob("job-*"))
    with pytest.raises(Unavailable, match="no automatic retry"):
        store.call("source/clean", lambda: pytest.fail("no repeat after kill"))


def test_cleanup_only_own_dead_scratch(tmp_path):
    scratch = tmp_path / "scratch"
    output = tmp_path / "output"
    for name, owner in (("job-own", "cegwm-isolated-v1"), ("job-other", "other")):
        folder = scratch / name
        folder.mkdir(parents=True)
        atomic_json(folder / "owner.json", dict(owner=owner, receipt=str(output / "workers" / name), coordinator_pid=None))
    removed = cleanup_owned_scratch(scratch, output)
    assert removed == [str(scratch / "job-own")]
    assert (scratch / "job-other").exists()
