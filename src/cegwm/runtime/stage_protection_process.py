"""One model operation per subprocess, with durable receipts and local scratch."""
from __future__ import annotations

import ctypes
import importlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import uuid

import torch

from .stage_protection_disk import DiskSavedTensors, local_scratch
from .stage_protection_store import DurableCounters, atomic_json, atomic_torch

METHODS = ("generate", "render", "read", "reconstruct", "rgb_gradient", "latent_gradients")


def cleanup_owned_scratch(scratch, directory):
    """Remove only dead jobs whose ownership marker names this output's workers."""
    removed = []
    expected = (Path(directory) / "workers").resolve()
    for job in Path(scratch).glob("job-*"):
        if not job.is_dir() or job.is_symlink() or not (job / "owner.json").exists():
            continue
        owner = json.loads((job / "owner.json").read_text())
        if owner.get("owner") != "cegwm-isolated-v1" or Path(owner["receipt"]).resolve().parent != expected:
            continue
        for pid in (owner.get("child_pid"), owner.get("coordinator_pid")):
            if pid and pid != os.getpid():
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f"previous owned scratch still has live process {pid}; no overlapping worker")
        shutil.rmtree(job)
        removed.append(str(job))
    return removed


def memory_snapshot():
    try:
        row = next(x for x in Path("/proc/self/status").read_text().splitlines() if x.startswith("VmRSS:"))
        rss = int(row.split()[1]) * 1024
    except (OSError, StopIteration, ValueError):
        rss = None
    gpu = {}
    if torch.cuda.is_initialized():
        gpu = dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return dict(pid=os.getpid(), rss_bytes=rss, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, gpu=gpu)


def worker_memory_event(event, *, step=None, saved_tensors=None):
    try:
        print("WORKER_MEMORY " + json.dumps(dict(event=event, step=step, memory=memory_snapshot(),
            saved_tensor_allocations=saved_tensors), allow_nan=False), flush=True)
    except OSError:
        pass  # Observation transport is not a scientific admission condition.


class WorkerFailed(RuntimeError):
    def __init__(self, message, partial=None):
        super().__init__(message)
        self.partial = partial or {}


def _parent_death(parent_pid):
    if sys.platform != "linux":
        raise RuntimeError("isolated model worker requires Linux parent-death handling")
    # Kill orphan workers if the coordinating process is killed; no silent
    # background model process may survive and overlap the next invocation.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    if os.getppid() != parent_pid:
        raise RuntimeError("coordinator exited before worker startup")


def run_worker(request_path):
    request = torch.load(request_path, map_location="cpu", weights_only=False)
    _parent_death(request["parent_pid"])
    method = request["method"]
    if method not in METHODS:
        raise ValueError("unsupported isolated operation")
    receipt = Path(request["receipt"])
    scratch = Path(request_path).parent
    counters = DurableCounters(request["counters_path"])
    provider = None
    backend = DiskSavedTensors(scratch, metrics_path=scratch / "disk_metrics.json")
    try:
        module, name = request["factory"].split(":")
        factory = getattr(importlib.import_module(module), name)
        provider = factory(request["config"], counters=counters, disk_storage=backend,
                           resource_observer=worker_memory_event)
        worker_memory_event("operation_start")
        result = getattr(provider, method)(*request["args"], **request["kwargs"])
        worker_memory_event("operation_returned")
        atomic_torch(receipt / "result.pt", result)
        del result
        atomic_json(receipt / "worker_status.json", dict(status="RETURNED", method=method,
                    counts_after=dict(counters), memory=memory_snapshot()))
        return 0
    except Exception as error:
        partial = getattr(error, "partial", {})
        if partial:
            atomic_torch(receipt / "partial.pt", partial)
        atomic_json(receipt / "worker_status.json", dict(status="FAILED", method=method,
            error=f"{type(error).__name__}: {error}", counts_after=dict(counters), memory=memory_snapshot()))
        raise
    finally:
        # Process exit releases model allocations. Do not call the legacy close
        # path here: it moves the retained reader VAE to CPU unnecessarily.
        provider = None
        backend.close()
        worker_memory_event("worker_cleanup_returned")


class ProcessProvider:
    """Coordinator owns only CPU arguments/results; it never constructs a model."""
    def __init__(self, config, *, counters, directory, scratch_root="/content/ceg_stage_protection_scratch",
                 factory="cegwm.runtime.stage_protection_v2:StageProtectionProvider"):
        self.config, self.counters = config, counters
        self.directory = Path(directory)
        self.scratch = local_scratch(scratch_root)
        self.removed_stale_scratch = cleanup_owned_scratch(self.scratch, self.directory)
        self.factory = factory
        self.operation_key = None
        self.active = None

    def _call(self, method, *args, **kwargs):
        if self.active is not None:
            raise RuntimeError("one model worker at a time")
        identifier = uuid.uuid4().hex
        receipt = self.directory / "workers" / identifier
        receipt.mkdir(parents=True)
        scratch = self.scratch / ("job-" + identifier)
        scratch.mkdir()
        owner = dict(owner="cegwm-isolated-v1", receipt=str(receipt.resolve()), coordinator_pid=os.getpid())
        request = dict(config=self.config, method=method, args=args, kwargs=kwargs,
            counters_path=str(self.counters.path.resolve()), receipt=str(receipt.resolve()),
            factory=self.factory, parent_pid=os.getpid())
        try:
            atomic_json(scratch / "owner.json", owner)
            atomic_torch(scratch / "request.pt", request)
        except BaseException:
            shutil.rmtree(scratch)
            raise
        record = dict(status="STARTED", method=method, operation_key=self.operation_key,
            counts_before=dict(self.counters), scratch=str(scratch),
            scratch_free_before=shutil.disk_usage(scratch).free)
        atomic_json(receipt / "parent_status.json", record)
        started = time.monotonic()
        try:
            command = [sys.executable, "-u", "-m", "cegwm.runtime.stage_protection_process", str(scratch / "request.pt")]
            self.active = subprocess.Popen(command, start_new_session=True)
            atomic_json(scratch / "owner.json", dict(owner, child_pid=self.active.pid))
            record["child_pid"] = self.active.pid
            atomic_json(receipt / "parent_status.json", record)
            # stdout/stderr inherit the existing append-only runner stream.
            returncode = self.active.wait()
            record.update(returncode=returncode, child_reaped=True, elapsed_seconds=time.monotonic() - started)
            if returncode != 0:
                detail = json.loads((receipt / "worker_status.json").read_text()) if (receipt / "worker_status.json").exists() else {}
                partial = torch.load(receipt / "partial.pt", map_location="cpu", weights_only=False) if (receipt / "partial.pt").exists() else {}
                raise WorkerFailed(f"isolated {method} exit {returncode}: {detail.get('error', 'no returned worker status')}", partial)
            status = json.loads((receipt / "worker_status.json").read_text())
            if status["status"] != "RETURNED":
                raise WorkerFailed("worker exited without RETURNED result")
            record["status"] = "RETURNED"
            # Reading a result is strictly after wait/reap, not just file arrival.
            return torch.load(receipt / "result.pt", map_location="cpu", weights_only=False)
        except BaseException as error:
            record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if self.active is not None and self.active.poll() is None:
                os.killpg(self.active.pid, signal.SIGKILL)
                self.active.wait()
            if self.active is not None:
                record.update(returncode=self.active.returncode, child_reaped=True)
            self.active = None
            # Child increments are durable. Refresh the parent dictionary without
            # rewriting the authoritative counter file or losing partial cost.
            self.counters.clear()
            self.counters.update(json.loads(self.counters.path.read_text()))
            record["counts_after"] = dict(self.counters)
            if (scratch / "disk_metrics.json").exists():
                record["disk_saved_tensors"] = json.loads((scratch / "disk_metrics.json").read_text())
            record["scratch_bytes_before_cleanup"] = sum(p.stat().st_size for p in scratch.rglob("*") if p.is_file())
            shutil.rmtree(scratch)  # Only the directory allocated above, after reap.
            record["scratch_removed"] = True
            atomic_json(receipt / "parent_status.json", record)

    def close(self):
        if self.active is not None:
            raise RuntimeError("cannot finish while a model worker is active")


def _method(name):
    def invoke(self, *args, **kwargs):
        return self._call(name, *args, **kwargs)
    return invoke


for _name in METHODS:
    setattr(ProcessProvider, _name, _method(_name))


if __name__ == "__main__":
    raise SystemExit(run_worker(sys.argv[1]))
