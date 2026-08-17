"""CPU checks for neutral experiment-execution support helpers."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiment_execution.delivery_support import (
    DeliverySupportError,
    _canonical_bytes,
    _session_runtime_identity,
    _sha256_file,
)
from scripts.experiment_execution.server_support import (
    ExperimentExecutionServerError,
    _absolute_directory,
    _paths_overlap,
    _probe_resources,
    _write_json_create_only,
)


pytestmark = pytest.mark.unit


def test_delivery_support_canonical_bytes_are_stable() -> None:
    assert _canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'


def test_delivery_support_hashes_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")
    assert _sha256_file(path) == sha256(path.read_bytes()).hexdigest()


def test_delivery_support_normalizes_runtime_identity() -> None:
    assert _session_runtime_identity(role="gpu", display_value="NVIDIA A100") == "gpu_nvidia_a100"
    with pytest.raises(DeliverySupportError):
        _session_runtime_identity(role="model", display_value="A100")


def test_server_support_resource_preflight_uses_absolute_roots_and_one_exact_gpu_query(
    monkeypatch,
    tmp_path: Path,
) -> None:
    persistent_root = tmp_path / "persistent"
    cache_root = tmp_path / "cache"
    assert _absolute_directory(persistent_root, "persistent_root") == persistent_root.resolve()
    assert _absolute_directory(cache_root, "cache_root") == cache_root.resolve()
    with pytest.raises(ExperimentExecutionServerError):
        _absolute_directory("relative", "persistent_root")
    disk_roots: list[Path] = []
    disk_results = iter((SimpleNamespace(free=101), SimpleNamespace(free=202)))
    commands: list[tuple[str, ...]] = []
    command_kwargs: list[dict[str, object]] = []

    def disk_usage(root: Path) -> SimpleNamespace:
        disk_roots.append(root)
        return next(disk_results)

    def probe(command, **kwargs):
        commands.append(tuple(command))
        command_kwargs.append(kwargs)
        return SimpleNamespace(stdout="NVIDIA L4, 23034\n")

    monkeypatch.setattr("scripts.experiment_execution.server_support.shutil.disk_usage", disk_usage)
    monkeypatch.setattr("scripts.experiment_execution.server_support.subprocess.run", probe)
    observed = _probe_resources(
        persistent_root=persistent_root,
        cache_root=cache_root,
    )

    assert commands == [
        (
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        )
    ]
    assert command_kwargs == [{"check": True, "capture_output": True, "text": True}]
    assert disk_roots == [persistent_root, cache_root]
    assert observed["cuda_device_name"] == "NVIDIA L4"
    assert observed["cuda_total_memory_bytes"] == 23034 * 1024 * 1024
    assert observed["free_disk_bytes"] == {
        str(persistent_root): 101,
        str(cache_root): 202,
    }


def test_server_support_writes_create_only_receipt(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    _write_json_create_only(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text("utf-8")) == {"a": 1, "b": 2}
    assert not _paths_overlap(tmp_path / "left", tmp_path / "right")
