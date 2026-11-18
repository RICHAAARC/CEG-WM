"""Atomic completion markers and identity checks; safe for interrupted Drive writes."""
import json
import os
from pathlib import Path
import re
import shutil
import time

from .colab_config import file_sha


def redact(text):
    secret = os.environ.get("HF_TOKEN")
    if secret:
        text = str(text).replace(secret, "[REDACTED]")
    return re.sub(r"hf_[A-Za-z0-9]{8,}", "[REDACTED]", str(text))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with open(temporary, "w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def quarantine(paths, root, reason):
    root = Path(root).resolve()
    target = root/"quarantine"/str(time.time_ns())
    moved = []
    for path in paths:
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("quarantine target escapes run directory")
        if path.exists():
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target/path.name))
            moved.append(path.name)
    if moved:
        atomic_json(target/"reason.json", {"reason": reason, "files": moved})
    return moved


def completed_asset(root, name, identity):
    root = Path(root)
    asset, marker = root/name, root/(name+".json")
    if marker.exists():
        metadata = read_json(marker)
        if metadata.get("identity") != identity:
            raise ValueError("asset identity conflict: " + name)
        if not asset.is_file() or file_sha(asset) != metadata["sha256"]:
            raise ValueError("completed asset hash mismatch: " + name)
        return metadata
    if asset.exists():
        quarantine([asset], root, "orphan asset without completion marker; regenerate")
    return None


def write_asset(root, name, identity, write):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root/name
    temporary = root/(name+".tmp")
    write(temporary)
    with open(temporary, "rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    metadata = {"identity": identity, "sha256": file_sha(path)}
    atomic_json(root/(name+".json"), metadata)
    return metadata
