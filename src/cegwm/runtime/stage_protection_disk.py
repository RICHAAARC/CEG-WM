"""Exact saved-tensor files on local scratch; no graph-wide CPU tensor cache."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import time

import torch

from .stage_protection_store import atomic_json


def local_scratch(path):
    path = Path(path).resolve()
    if path == Path("/content/drive") or Path("/content/drive") in path.parents:
        raise ValueError("saved tensors must use local scratch, never Google Drive")
    # Reject remote/FUSE mounts as well as the ordinary Colab Drive path.
    if Path("/proc/mounts").exists():
        mounts = []
        for line in Path("/proc/mounts").read_text().splitlines():
            parts = line.split()
            mount = Path(parts[1].replace(r"\040", " "))
            if path == mount or mount in path.parents:
                mounts.append((len(str(mount)), parts[2]))
        if mounts and max(mounts)[1].startswith(("fuse", "nfs", "cifs", "9p", "tmpfs", "ramfs")):
            raise ValueError("saved tensor scratch filesystem must be local")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _drop_file_cache(path):
    # An advisory kernel-cache hint, not a guarantee about RSS/cgroup memory.
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            with Path(path).open("rb") as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError as error:
            return f"{type(error).__name__}: {error}"
    return None


class DiskSavedTensors:
    """pack retains paths only; files survive all unpacks until context exit.

    One old-semantics CPU copy is serialized at a time. Unpack allocations are
    held only by the active backward consumer, never by this backend. This
    bounds retained saved CPU copies, not model/workspace/allocator/page-cache
    memory or the simultaneous operands of a backward kernel.
    """
    def __init__(self, scratch, *, metrics_path=None, reserve_bytes=64 * 1024**2):
        self.parent = local_scratch(scratch)
        free = shutil.disk_usage(self.parent).free
        if free < reserve_bytes:
            raise OSError(f"local scratch free bytes {free} below startup reserve {reserve_bytes}")
        self.directory = Path(tempfile.mkdtemp(prefix="saved-", dir=self.parent))
        self.metrics_path = Path(metrics_path) if metrics_path else self.parent / "saved_tensor_metrics.json"
        self.reserve_bytes = reserve_bytes
        self.stats = dict(saved_files=0, unpack_calls=0, saved_tensor_bytes=0,
            written_file_bytes=0, read_file_bytes=0, scratch_peak_bytes=0,
            scratch_current_bytes=0, largest_cpu_copy_bytes=0, write_seconds=0., read_seconds=0.,
            initial_free_bytes=free, minimum_observed_free_bytes=free,
            retained_cpu_tensor_bytes=0, cleaned=False,
            cache_advice_failures=0, last_cache_advice_error=None,
            memory_scope="one serial pack copy; unpack tensors belong to active consumers; excludes models/workspaces/allocator/kernel page cache")
        self.active = True
        self.publish()

    def publish(self):
        atomic_json(self.metrics_path, self.stats)

    def cache_hint(self, path):
        error = _drop_file_cache(path)
        if error:
            self.stats["cache_advice_failures"] += 1
            self.stats["last_cache_advice_error"] = error

    def pack(self, value):
        if not self.active:
            raise RuntimeError("disk saved-tensor context is closed")
        if value.layout != torch.strided:
            raise ValueError("disk saved tensors support dense strided tensors only")
        needed = value.numel() * value.element_size()
        free = shutil.disk_usage(self.directory).free
        self.stats["minimum_observed_free_bytes"] = min(free, self.stats["minimum_observed_free_bytes"])
        if free < needed + self.reserve_bytes:
            self.publish()
            raise OSError(f"local saved-tensor scratch insufficient: free={free}, tensor={needed}, reserve={self.reserve_bytes}")
        started = time.monotonic()
        # Exactly the previous CPU copy operation. Views do not serialize their
        # entire original storage; preserve_format retains dense strides, and
        # compacts holey/overlapping/offset layouts exactly as before.
        copied = value.detach().to("cpu", copy=True)
        path = self.directory / f"{self.stats['saved_files']:08d}.pt"
        layout = (tuple(copied.shape), tuple(copied.stride()), copied.dtype)
        size = copied.untyped_storage().nbytes()
        self.stats["largest_cpu_copy_bytes"] = max(size, self.stats["largest_cpu_copy_bytes"])
        try:
            with path.open("wb") as stream:
                torch.save(copied, stream)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            del copied
        file_bytes = path.stat().st_size
        self.stats["saved_files"] += 1
        self.stats["saved_tensor_bytes"] += needed
        self.stats["written_file_bytes"] += file_bytes
        self.stats["scratch_current_bytes"] += file_bytes
        self.stats["scratch_peak_bytes"] = max(self.stats["scratch_peak_bytes"], self.stats["scratch_current_bytes"])
        self.stats["write_seconds"] += time.monotonic() - started
        self.cache_hint(path)
        self.publish()
        return (path, value.device, layout)

    def unpack(self, packed):
        if not self.active:
            raise RuntimeError("saved tensor used after disk context exit")
        path, device, layout = packed
        started = time.monotonic()
        copied = torch.load(path, map_location="cpu", weights_only=True)
        if (tuple(copied.shape), tuple(copied.stride()), copied.dtype) != layout:
            raise ValueError("saved tensor metadata changed")
        result = copied.to(device)
        del copied
        self.stats["unpack_calls"] += 1
        self.stats["read_file_bytes"] += path.stat().st_size
        self.stats["read_seconds"] += time.monotonic() - started
        self.cache_hint(path)
        self.publish()
        return result

    def close(self):
        if self.active:
            self.active = False
            shutil.rmtree(self.directory)
            self.stats.update(scratch_current_bytes=0, cleaned=True)
            self.publish()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
