"""Small resumable operation store: STARTED work is never silently retried.

Own-run torch artifacts contain tensors/PIL/captures and are trusted local
checkpoints, not an interchange format for untrusted pickle uploads.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch(path, value):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class DurableCounters(dict):
    def __init__(self, path):
        self.path = Path(path)
        super().__init__(json.loads(self.path.read_text()) if self.path.exists() else {})

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        atomic_json(self.path, self)


class Unavailable(RuntimeError):
    pass


class OperationStore:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.operations = self.directory / "operations"
        self.operations.mkdir(exist_ok=True)
        path = self.directory / "config.json"
        if path.exists():
            if json.loads(path.read_text()) != config:
                raise ValueError("resume configuration differs; no new parameters in an existing run")
        else:
            atomic_json(path, config)
        self.counters = DurableCounters(self.directory / "physical_counts.json")
        self.cache_reuses = []

    def call(self, key, function):
        folder = self.operations / key
        folder.mkdir(parents=True, exist_ok=True)
        status_file, payload = folder / "status.json", folder / "result.pt"
        if status_file.exists():
            status = json.loads(status_file.read_text())
            if status["status"] == "RETURNED":
                self.cache_reuses.append(key)
                return torch.load(payload, map_location="cpu", weights_only=False)
            raise Unavailable(f"{key}: {status['status']} (no automatic retry)")
        atomic_json(status_file, {"status": "STARTED", "key": key,
                                  "counts_before": dict(self.counters)})
        print(f"STARTED {key}", flush=True)
        started = time.monotonic()
        try:
            result = function()
            atomic_torch(payload, result)
            atomic_json(status_file, {"status": "RETURNED", "key": key,
                                     "counts_after": dict(self.counters),
                                     "elapsed_seconds": time.monotonic() - started})
            return result
        except Exception as error:
            partial = getattr(error, "partial", {})
            if partial:
                atomic_torch(folder / "partial.pt", partial)
            atomic_json(status_file, {"status": "FAILED", "key": key,
                "error": f"{type(error).__name__}: {error}", "counts_after": dict(self.counters)})
            raise Unavailable(f"{key}: {type(error).__name__}: {error}") from error

    def inventory(self):
        return [json.loads(path.read_text()) for path in sorted(self.operations.rglob("status.json"))]
