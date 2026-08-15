"""Build one deterministic exact-revision salient-local-LF execution package."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Sequence
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


PACKAGE_SCHEMA_VERSION = 1
PACKAGE_PROFILE = "salient_local_lf_mask_write_validation_execution_package"
REVISION = re.compile(r"^[0-9a-f]{40}$")
_EXACT_ROOT_FILES = {
    "pyproject.toml",
    "requirements_inspyrenet_salient_local_lf_gpu_execution.txt",
    "requirements_sd35_gpu_execution.txt",
}
_PREFIXES = (
    "main/", "runtime/", "experiments/", "configs/runtime/",
    "configs/experiments/internal_execution_components.json",
    "configs/experiments/salient_local_lf_mask_write_validation",
    "scripts/experiment_execution/__init__.py",
    "scripts/experiment_execution/development_exploration_entrypoint.py",
    "scripts/experiment_execution/salient_local_lf_mask_write_validation",
)
_README_SOURCE = "templates/release_readmes/salient_local_lf_mask_write_validation_package.md"


class SalientLocalLfPackageBuildError(RuntimeError):
    """The exact source tree cannot produce the frozen package."""


def _git(root: Path, *args: str, text: bool = True) -> str | bytes:
    try:
        result = subprocess.run(("git", *args), cwd=root, check=True, capture_output=True, text=text)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SalientLocalLfPackageBuildError("exact Git source is unavailable") from exc
    return result.stdout.strip() if text else result.stdout


def _included(path: str) -> bool:
    return path in _EXACT_ROOT_FILES or path == _README_SOURCE or any(path.startswith(prefix) for prefix in _PREFIXES)


def _tree(root: Path, revision: str) -> tuple[tuple[str, str, bytes], ...]:
    raw = _git(root, "ls-tree", "-r", "-z", revision, text=False)
    if type(raw) is not bytes:
        raise SalientLocalLfPackageBuildError("Git tree output is invalid")
    entries = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        metadata, encoded_path = item.split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split()
        path = encoded_path.decode("utf-8")
        if not _included(path):
            continue
        pure = PurePosixPath(path)
        if mode != "100644" or kind != "blob" or pure.is_absolute() or ".." in pure.parts:
            raise SalientLocalLfPackageBuildError("package source path is unsafe")
        payload = _git(root, "cat-file", "blob", blob, text=False)
        if type(payload) is not bytes:
            raise SalientLocalLfPackageBuildError("package blob bytes are invalid")
        archive_path = "README.md" if path == _README_SOURCE else path
        entries.append((archive_path, blob, payload))
    config_payload = next(
        (payload for path, _blob, payload in entries
         if path == "configs/experiments/salient_local_lf_mask_write_validation.json"),
        None,
    )
    if config_payload is None:
        raise SalientLocalLfPackageBuildError("package protocol config is unavailable")
    try:
        config = json.loads(config_payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SalientLocalLfPackageBuildError("package protocol config is invalid") from exc
    for authority in config.get("historical_prior_authorities", []):
        if type(authority) is not dict or type(authority.get("paths")) is not list:
            raise SalientLocalLfPackageBuildError("historical package authority is invalid")
        revision_identity = authority.get("producer_revision")
        if type(revision_identity) is not str or REVISION.fullmatch(revision_identity) is None:
            raise SalientLocalLfPackageBuildError("historical producer revision is invalid")
        for binding in authority["paths"]:
            if type(binding) is not dict:
                raise SalientLocalLfPackageBuildError("historical package path binding is invalid")
            source_path = binding.get("path")
            archive_path = binding.get("package_member_path")
            if type(source_path) is not str or type(archive_path) is not str:
                raise SalientLocalLfPackageBuildError("historical package path binding is invalid")
            tree_line = str(_git(root, "ls-tree", revision_identity, "--", source_path)).split()
            if (len(tree_line) != 4 or tree_line[:2] != ["100644", "blob"]
                    or tree_line[2] != binding.get("git_blob_sha")):
                raise SalientLocalLfPackageBuildError("historical producer Git blob drifted")
            historical_payload = _git(root, "cat-file", "blob", tree_line[2], text=False)
            if type(historical_payload) is not bytes or sha256(historical_payload).hexdigest() != binding.get("raw_sha256"):
                raise SalientLocalLfPackageBuildError("historical producer bytes drifted")
            pure = PurePosixPath(archive_path)
            if pure.is_absolute() or ".." in pure.parts or not archive_path.startswith("historical_authorities/"):
                raise SalientLocalLfPackageBuildError("historical package member path is unsafe")
            entries.append((archive_path, tree_line[2], historical_payload))
    entries.sort(key=lambda item: item[0])
    names = tuple(item[0] for item in entries)
    required = {
        "README.md",
        "configs/experiments/salient_local_lf_mask_write_validation.json",
        "configs/experiments/salient_local_lf_mask_write_validation_manifest.json",
        "experiments/protocol/salient_local_lf_mask_write_validation.py",
        "experiments/metrics/salient_local_lf_mask_write_validation.py",
        "experiments/runners/salient_local_lf_mask_write_validation.py",
        "scripts/experiment_execution/salient_local_lf_mask_write_validation_entrypoint.py",
        "runtime/inspyrenet_saliency.py",
        "runtime/_vendor/transparent_background/SOURCE.json",
        "historical_authorities/925c2cbc727e3b18e91c0b3981eeed1b470a955a/configs/experiments/content_routing_directional_diagnosis.json",
        "historical_authorities/925c2cbc727e3b18e91c0b3981eeed1b470a955a/configs/experiments/content_routing_directional_diagnosis_manifest.json",
        "historical_authorities/7c0d86d6eac5ffcfc4a30f2f5fb22884aaa848da/configs/experiments/content_uniform_combination_directional_diagnosis.json",
        "historical_authorities/7c0d86d6eac5ffcfc4a30f2f5fb22884aaa848da/configs/experiments/content_uniform_combination_directional_diagnosis_manifest.json",
    }
    if not required.issubset(names) or len(names) != len(set(names)):
        raise SalientLocalLfPackageBuildError("package source closure is incomplete")
    return tuple(entries)


def build_salient_local_lf_mask_write_validation_package(
    repository_root: str | Path, output_path: str | Path, revision: str,
) -> dict[str, object]:
    root = Path(repository_root)
    destination = Path(output_path)
    if REVISION.fullmatch(revision) is None or _git(root, "rev-parse", f"{revision}^{{commit}}") != revision:
        raise SalientLocalLfPackageBuildError("revision is not an exact commit")
    entries = _tree(root, revision)
    manifest_entries = [
        {"path": path, "git_blob_sha": blob, "sha256": sha256(payload).hexdigest(), "size": len(payload)}
        for path, blob, payload in entries
    ]
    manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION, "package_profile": PACKAGE_PROFILE,
        "committed_revision": revision, "entries": manifest_entries,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SalientLocalLfPackageBuildError("package destination already exists")
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path, _blob, payload in (*entries, ("PACKAGE_MANIFEST.json", "", manifest_bytes)):
            info = ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    package_sha = sha256(destination.read_bytes()).hexdigest()
    return {"package_path": str(destination), "package_sha256": package_sha,
            "package_size_bytes": destination.stat().st_size,
            "package_manifest_sha256": sha256(manifest_bytes).hexdigest(),
            "committed_revision": revision, "package_profile": PACKAGE_PROFILE}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_salient_local_lf_mask_write_validation_package(
        args.repository_root, args.output_path, args.revision), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
