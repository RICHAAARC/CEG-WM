"""CPU checks for neutral experiment-execution support helpers."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

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


def test_server_support_requires_absolute_directory(tmp_path: Path) -> None:
    assert _absolute_directory(tmp_path, "persistent_root") == tmp_path.resolve()
    with pytest.raises(ExperimentExecutionServerError):
        _absolute_directory("relative", "persistent_root")


def test_server_support_writes_create_only_receipt(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    _write_json_create_only(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text("utf-8")) == {"a": 1, "b": 2}
    assert not _paths_overlap(tmp_path / "left", tmp_path / "right")
