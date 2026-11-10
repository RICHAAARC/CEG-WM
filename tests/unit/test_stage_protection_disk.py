import gc
import json
import weakref

import pytest
import torch

from cegwm.runtime.stage_protection_disk import DiskSavedTensors, local_scratch
from cegwm.runtime.channel_precompensation_sd35 import _cpu_saved_tensors

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "holes", "offset", "overlap", "singleton", "empty", "scalar"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_exact_copy_layout_multiple_unpacks_and_no_cpu_cache(tmp_path, monkeypatch, layout, dtype):
    base = torch.arange(24, dtype=dtype).reshape(4, 6)
    value = dict(contiguous=base, transpose=base.T, holes=base[:, ::2], offset=base[1:],
                 overlap=base.as_strided((4, 6), (0, 1)), singleton=base.as_strided((1, 6), (99, 1)),
                 empty=base[:0], scalar=base.reshape(-1)[0])[layout]
    saved_refs = []
    actual_save = torch.save
    def observe_save(tensor, stream):
        saved_refs.append(weakref.ref(tensor))
        return actual_save(tensor, stream)
    monkeypatch.setattr(torch, "save", observe_save)
    with DiskSavedTensors(tmp_path) as disk:
        packed = disk.pack(value)
        gc.collect()
        assert saved_refs[0]() is None
        expected = value.detach().to("cpu", copy=True)
        first, second = disk.unpack(packed), disk.unpack(packed)
        assert torch.equal(first, expected) and torch.equal(second, expected)
        assert first.stride() == second.stride() == expected.stride()
        assert first.dtype == second.dtype == dtype
        assert packed[0].exists()
        ref = weakref.ref(first)
        del first
        gc.collect()
        assert ref() is None  # No loaded CPU tensor held by the backend.
        assert disk.stats["saved_tensor_bytes"] == value.numel() * value.element_size()
        assert disk.stats["largest_cpu_copy_bytes"] <= value.numel() * value.element_size()
    assert not packed[0].exists()
    assert disk.stats["cleaned"] and disk.stats["scratch_current_bytes"] == 0


def test_two_row_vjp_retain_graph_exact_and_stable_weight_reference(tmp_path):
    module = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64).eval().requires_grad_(False)
    module.weight.copy_(torch.arange(16).reshape(4, 4) / 20)
    results = []
    for mode in ("cpu", "disk"):
        disk = DiskSavedTensors(tmp_path) if mode == "disk" else None
        try:
            x = torch.arange(8, dtype=torch.float64).reshape(2, 4).requires_grad_(True)
            with _cpu_saved_tensors(stable_modules=(module,), retain_stable_storage=True, disk_storage=disk) as stats:
                y = torch.sin(x @ module.weight.T) + (x @ module.weight).square()
                a, = torch.autograd.grad(y.sum(), x, retain_graph=True)
                b, = torch.autograd.grad((y * 2).sum(), x)
            results.append((y.detach(), a, b, stats))
            if disk:
                assert disk.stats["unpack_calls"] > disk.stats["saved_files"]
                assert disk.stats["saved_files"] > 0
                assert stats["stable_parameter_reference_saves"] == 2
        finally:
            if disk:
                disk.close()
    for first, second in zip(results[0][:3], results[1][:3]):
        assert torch.equal(first, second)


def test_space_failure_and_exception_cleanup(tmp_path, monkeypatch):
    import cegwm.runtime.stage_protection_disk as backend
    with pytest.raises(OSError, match="insufficient"):
        with DiskSavedTensors(tmp_path) as disk:
            monkeypatch.setattr(backend.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 0})())
            disk.pack(torch.ones(16))
    assert disk.stats["cleaned"]
    assert not list(tmp_path.glob("saved-*"))


def test_cache_advice_failure_is_diagnostic(tmp_path, monkeypatch):
    import cegwm.runtime.stage_protection_disk as backend
    monkeypatch.setattr(backend.os, "posix_fadvise", lambda *a: (_ for _ in ()).throw(OSError("unsupported")))
    with DiskSavedTensors(tmp_path) as disk:
        packed = disk.pack(torch.ones(4))
        assert torch.equal(disk.unpack(packed), torch.ones(4))
        assert disk.stats["cache_advice_failures"] == 2


def test_drive_and_memory_filesystems_rejected(tmp_path, monkeypatch):
    from pathlib import Path
    with pytest.raises(ValueError, match="never Google Drive"):
        local_scratch("/content/drive/MyDrive/scratch")
    real = Path.read_text
    for filesystem in ("tmpfs", "ramfs", "fuse.drivefs"):
        monkeypatch.setattr(Path, "read_text", lambda path, *a, **k: f"none / {filesystem} rw 0 0\n" if str(path) == "/proc/mounts" else real(path, *a, **k))
        with pytest.raises(ValueError, match="must be local"):
            local_scratch(tmp_path)
