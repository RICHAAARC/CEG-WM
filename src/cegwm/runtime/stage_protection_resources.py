"""Read-only Linux memory evidence. Missing observations are null, never zero.

Cgroup counters cover a group of processes and do not attribute an OOM to this
runner. These records do not control writes, acceptance, budgets or retries.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path


def _read(path, parser, errors, name):
    try:
        return parser(Path(path).read_text().strip())
    except (OSError, ValueError) as error:
        errors[name] = f"{type(error).__name__}: {error}"
        return None


def resolve_cgroup(*, proc_root=Path("/proc")):
    """Resolve the current process's unified hierarchy in its mount namespace."""
    try:
        entry = next(line.split(":", 2)[2] for line in (proc_root / "self/cgroup").read_text().splitlines()
                     if line.startswith("0::"))
        for line in (proc_root / "self/mountinfo").read_text().splitlines():
            left, separator, right = line.partition(" - ")
            if separator and right.split()[0] == "cgroup2":
                fields = left.split()
                root, mount = fields[3], fields[4]
                # mountinfo uses octal escapes for whitespace/backslashes.
                for encoded, literal in ((r"\040", " "), (r"\011", "\t"), (r"\134", "\\")):
                    root, mount = root.replace(encoded, literal), mount.replace(encoded, literal)
                relative = Path(entry).relative_to(root)
                return str(Path(mount) / relative), None
        return None, "unified cgroup2 mount unavailable"
    except (OSError, ValueError, StopIteration, IndexError) as error:
        return None, f"{type(error).__name__}: {error}"


def cgroup_memory_snapshot(*, directory=None, proc_root=Path("/proc")):
    path, resolution_error = (str(directory), None) if directory is not None else resolve_cgroup(proc_root=proc_root)
    errors = {}
    result = dict(cgroup_path=path, memory_events=None, memory_current_bytes=None,
                  memory_peak_bytes=None, memory_limit_bytes=None, errors=errors)
    if path is None:
        errors["cgroup"] = resolution_error
        return result
    folder = Path(path)
    def events(text):
        return {key: int(value) for key, value in (line.split() for line in text.splitlines())}
    result["memory_events"] = _read(folder / "memory.events", events, errors, "memory.events")
    for field, filename in (("memory_current_bytes", "memory.current"),
                            ("memory_peak_bytes", "memory.peak"), ("memory_limit_bytes", "memory.max")):
        result[field] = _read(folder / filename, lambda text: "max" if text == "max" else int(text), errors, filename)
    return result


def event_delta(before, after):
    first, last = before.get("memory_events"), after.get("memory_events")
    if first is None or last is None or before["cgroup_path"] != after["cgroup_path"]:
        return None
    # Missing event fields or a reset remain unknown, not an invented zero.
    return {key: last[key] - first[key] if key in first and key in last and last[key] >= first[key] else None
            for key in sorted(first.keys() | last.keys())}


def append_parent_record(path, event, *, before=None, pid=None, returncode=None, fallback_log=None):
    """Best-effort telemetry; its IO must not gate a run or mask an exit code."""
    try:
        snapshot = cgroup_memory_snapshot(directory=before["cgroup_path"] if before else None)
    except Exception as error:
        snapshot = dict(cgroup_path=None, memory_events=None, memory_current_bytes=None,
                        memory_peak_bytes=None, memory_limit_bytes=None,
                        errors={"observation": f"{type(error).__name__}: {error}"})
    record = dict(event=event, utc=datetime.now(timezone.utc).isoformat(), child_pid=pid,
                  returncode=returncode, memory=snapshot,
                  memory_events_delta=event_delta(before, snapshot) if before else None,
                  scope="parent cgroup, normally inherited by child; shared counters, not process-specific OOM attribution")
    try:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception as error:
        snapshot["errors"]["resource_record_write"] = f"{type(error).__name__}: {error}"
        if fallback_log is not None:
            try:
                fallback_log.write("RESOURCE_RECORD_UNAVAILABLE " + json.dumps(record, allow_nan=False) + "\n")
                fallback_log.flush()
            except Exception:
                pass
    return snapshot


def source_memory_event(event, *, step=None, saved_tensors=None):
    errors = {}
    def rss(text):
        row = next(line for line in text.splitlines() if line.startswith("VmRSS:"))
        return int(row.split()[1]) * 1024
    try:
        value = _read("/proc/self/status", rss, errors, "rss")
    except StopIteration:
        value = None
        errors["rss"] = "VmRSS unavailable"
    record = dict(event=event, step=step, pid=os.getpid(), rss_bytes=value,
                  saved_tensors=saved_tensors, errors=errors,
                  byte_semantics="CPU copy bytes are cumulative allocations in this VJP, not live RSS")
    try:
        print("SOURCE_MEMORY " + json.dumps(record, allow_nan=False), flush=True)
    except OSError:
        pass  # Observation transport must not become a candidate decision.
