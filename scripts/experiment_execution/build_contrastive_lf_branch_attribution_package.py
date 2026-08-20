"""Build the deterministic exact-revision Stage-A execution package."""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Sequence
from zipfile import ZIP_STORED, ZipFile, ZipInfo


PROFILE = "contrastive_lf_branch_attribution_stage_a"
EMBEDDED_MANIFEST = "contrastive_lf_branch_attribution_package_manifest.json"
REVISION = re.compile(r"^[0-9a-f]{40}$")
LOCAL_PATH = re.compile(rb"(?<![A-Za-z0-9_])(?:/(?:home|Users|mnt|tmp|var|opt|root)/|[A-Za-z]:[\\/])")
SEED_FILES = frozenset(
    {
        "scripts/experiment_execution/contrastive_lf_branch_attribution_bootstrap.py",
        "scripts/experiment_execution/contrastive_lf_branch_attribution_entrypoint.py",
        "scripts/experiment_execution/contrastive_lf_branch_attribution_server.py",
        "experiments/runners/contrastive_lf_branch_attribution.py",
        "experiments/attacks/contrastive_lf_branch_attribution.py",
        "experiments/metrics/contrastive_lf_branch_attribution.py",
        "experiments/methods/ceg_wm.py",
    }
)
DATA_FILES = frozenset(
    {
        "configs/experiments/contrastive_lf_branch_attribution.json",
        "configs/experiments/contrastive_lf_branch_attribution_prompt_roster.json",
        "configs/experiments/contrastive_lf_null_fit_manifest.json",
        "configs/experiments/contrastive_lf_candidate_selection_manifest.json",
        "configs/experiments/contrastive_lf_branch_attribution_execution.json",
        "configs/experiments/contrastive_lf_branch_attribution_execution_components.json",
        "configs/experiments/internal_execution_components.json",
        "configs/runtime/runtime_sd35_flowmatch.json",
        "main/shared/normal_quantile_table20_float32_be.txt",
        "requirements_semantic_texture_operational_preflight.txt",
        "requirements_semantic_texture_operational_preflight_overlay.txt",
    }
)
ALLOWED_ROOTS = frozenset({"configs", "experiments", "main", "runtime", "scripts"})
FORBIDDEN_PARTS = frozenset({".agents", ".codex", ".git", "governance", "notebooks", "outputs", "tests", "paper_artifacts", "__pycache__"})


class ContrastiveLfPackageError(RuntimeError):
    pass


def _git(root: Path, *arguments: str, text: bool = True) -> str | bytes:
    try:
        result = subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True, text=text)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContrastiveLfPackageError("source repository identity is unavailable") from exc
    return result.stdout.strip() if text else result.stdout


def _blob(root: Path, revision: str, path: str) -> bytes:
    value = _git(root, "show", f"{revision}:{path}", text=False)
    if not isinstance(value, bytes):
        raise ContrastiveLfPackageError("tracked source blob is unavailable")
    return value


def _module_path(module: str) -> str | None:
    path = module.replace(".", "/")
    return path + ".py" if path else None


def _available_paths(root: Path, revision: str) -> set[str]:
    raw = _git(root, "ls-tree", "-r", "--name-only", revision)
    assert isinstance(raw, str)
    return set(raw.splitlines())


def _resolve_imports(path: str, blob: bytes, available: set[str]) -> set[str]:
    try:
        tree = ast.parse(blob, filename=path)
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise ContrastiveLfPackageError("package Python source is not parseable") from exc
    module = path[:-3].replace("/", ".")
    is_package = path.endswith("/__init__.py")
    package_parts = module.split(".") if is_package else module.split(".")[:-1]
    discovered: set[str] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - node.level + 1
                base_parts = package_parts[:keep]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                candidates.append(base)
                candidates.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
        for candidate in candidates:
            module_file = _module_path(candidate)
            package_file = candidate.replace(".", "/") + "/__init__.py"
            if module_file in available:
                discovered.add(module_file)
            if package_file in available:
                discovered.add(package_file)
            parts = candidate.split(".")
            for count in range(1, len(parts)):
                parent = "/".join(parts[:count]) + "/__init__.py"
                if parent in available:
                    discovered.add(parent)
    return discovered


def _source_closure(root: Path, revision: str) -> tuple[str, ...]:
    available = _available_paths(root, revision)
    pending = list(SEED_FILES)
    closure: set[str] = set()
    while pending:
        path = pending.pop()
        if path in closure:
            continue
        if path not in available:
            raise ContrastiveLfPackageError(f"required package source is missing: {path}")
        closure.add(path)
        if path.endswith(".py"):
            for dependency in _resolve_imports(path, _blob(root, revision, path), available):
                if dependency not in closure:
                    pending.append(dependency)
    closure.update(DATA_FILES)
    if not closure <= available:
        raise ContrastiveLfPackageError("required package data is missing")
    return tuple(sorted(closure))


def _safe(path_text: str, blob: bytes) -> None:
    path = PurePosixPath(path_text)
    declared_root_data_file = len(path.parts) == 1 and path_text in DATA_FILES
    if (
        path.is_absolute()
        or not path.parts
        or (path.parts[0] not in ALLOWED_ROOTS and not declared_root_data_file)
        or any(part in FORBIDDEN_PARTS for part in path.parts)
        or ".." in path.parts
        or path.suffix in {".pyc", ".pyo"}
        or LOCAL_PATH.search(blob)
    ):
        raise ContrastiveLfPackageError("package member is unsafe")


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_contrastive_lf_branch_attribution_package(
    *, repository_root: str | Path, source_revision: str, output: str | Path
) -> dict[str, object]:
    root, target = Path(repository_root).resolve(), Path(output).resolve()
    delivery = target.with_suffix(target.suffix + ".manifest.json")
    if REVISION.fullmatch(source_revision) is None or _git(root, "rev-parse", "HEAD") != source_revision or _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ContrastiveLfPackageError("source checkout must be exact and clean")
    if target == root or root in target.parents or target.exists() or delivery.exists():
        raise ContrastiveLfPackageError("package output target is unsafe or occupied")
    paths = _source_closure(root, source_revision)
    entries = []
    blobs: dict[str, bytes] = {}
    for path in paths:
        blob = _blob(root, source_revision, path)
        _safe(path, blob)
        blobs[path] = blob
        entries.append({"path": path, "sha256": sha256(blob).hexdigest(), "size_bytes": len(blob)})
    identity = sha256(_canonical({"copied_files": entries, "profile_name": PROFILE, "source_revision": source_revision})).hexdigest()
    manifest = {
        "copied_files": entries,
        "entrypoint_path": "scripts/experiment_execution/contrastive_lf_branch_attribution_entrypoint.py",
        "package_identity": identity,
        "package_ready": True,
        "package_schema_version": 1,
        "profile_name": PROFILE,
        "source_revision": source_revision,
        "stage_a_actions": ["null_fit", "candidate_selection"],
    }
    manifest_blob = _canonical(manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_STORED) as archive:
            for path in paths:
                archive.writestr(_zip_info(path), blobs[path])
            archive.writestr(_zip_info(EMBEDDED_MANIFEST), manifest_blob)
        os.link(temporary, target)
    except FileExistsError as exc:
        raise ContrastiveLfPackageError("package target must remain absent") from exc
    finally:
        temporary.unlink(missing_ok=True)
    result = {
        "archive_filename": target.name,
        "archive_sha256": sha256(target.read_bytes()).hexdigest(),
        "archive_size_bytes": target.stat().st_size,
        "copied_file_count": len(entries),
        "embedded_manifest_sha256": sha256(manifest_blob).hexdigest(),
        "package_identity": identity,
        "package_ready": True,
        "profile_name": PROFILE,
        "source_revision": source_revision,
    }
    with delivery.open("xb") as handle:
        handle.write(_canonical(result))
    return {**result, "delivery_manifest_path": str(delivery)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    print(json.dumps(build_contrastive_lf_branch_attribution_package(repository_root=arguments.repository_root, source_revision=arguments.source_revision, output=arguments.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
