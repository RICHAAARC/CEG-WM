"""Build one deterministic exact-revision salient-local-LF execution package."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
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
_SERVER_STARTUP_FILES = {
    "scripts/experiment_execution/__init__.py",
    "scripts/experiment_execution/build_salient_local_lf_mask_write_validation_package.py",
    "scripts/experiment_execution/development_exploration_entrypoint.py",
    "scripts/experiment_execution/development_exploration_server.py",
    "scripts/experiment_execution/salient_local_lf_mask_write_validation_entrypoint.py",
    "scripts/experiment_execution/salient_local_lf_mask_write_validation_server.py",
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
_CURRENT_AUTHORITY_IDENTITY = (
    "current_experiment_inputs_at_salient_local_lf_authorization_base"
)
_HISTORICAL_AUTHORITY_IDENTITIES = (
    "historical_content_routing_directional_negative",
    "historical_content_uniform_combination_negative",
)


class SalientLocalLfPackageBuildError(RuntimeError):
    """The exact source tree cannot produce the frozen package."""


def resolve_required_git_authority_revisions(
    *, execution_revision: str, config_payload: bytes | str,
) -> tuple[str, ...]:
    """Resolve the exact execution/current/historical commits from frozen config."""

    if type(execution_revision) is not str or REVISION.fullmatch(execution_revision) is None:
        raise SalientLocalLfPackageBuildError("execution revision is invalid")
    try:
        config = json.loads(
            config_payload.decode("utf-8")
            if type(config_payload) is bytes
            else config_payload
        )
    except (AttributeError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise SalientLocalLfPackageBuildError("package protocol config is invalid") from exc
    if type(config) is not dict:
        raise SalientLocalLfPackageBuildError("package protocol config is invalid")

    current = config.get("current_experiment_authority")
    historical = config.get("historical_prior_authorities")
    if (
        type(current) is not dict
        or current.get("authority_identity") != _CURRENT_AUTHORITY_IDENTITY
        or type(historical) is not list
        or len(historical) != len(_HISTORICAL_AUTHORITY_IDENTITIES)
    ):
        raise SalientLocalLfPackageBuildError("required Git authority identity drifted")
    observed_historical_identities = tuple(
        item.get("authority_identity") if type(item) is dict else None
        for item in historical
    )
    if observed_historical_identities != _HISTORICAL_AUTHORITY_IDENTITIES:
        raise SalientLocalLfPackageBuildError("required Git authority identity drifted")

    revisions = (
        execution_revision,
        current.get("producer_revision"),
        *(item.get("producer_revision") for item in historical),
    )
    if any(type(value) is not str or REVISION.fullmatch(value) is None for value in revisions):
        raise SalientLocalLfPackageBuildError("required Git authority revision is invalid")
    if len(revisions) != len(set(revisions)):
        raise SalientLocalLfPackageBuildError("required Git authority revision is duplicated")
    return revisions


def _git(root: Path, *args: str, text: bool = True) -> str | bytes:
    try:
        result = subprocess.run(("git", *args), cwd=root, check=True, capture_output=True, text=text)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SalientLocalLfPackageBuildError("exact Git source is unavailable") from exc
    return result.stdout.strip() if text else result.stdout


def _included(path: str) -> bool:
    return (path in _EXACT_ROOT_FILES or path in _SERVER_STARTUP_FILES
            or path == _README_SOURCE or any(path.startswith(prefix) for prefix in _PREFIXES))


def _tree(root: Path, revision: str) -> tuple[tuple[tuple[str, str, bytes], ...], dict[str, str]]:
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
    resolve_required_git_authority_revisions(
        execution_revision=revision,
        config_payload=config_payload,
    )
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
    current = config.get("current_experiment_authority")
    if type(current) is not dict or type(current.get("paths")) is not list:
        raise SalientLocalLfPackageBuildError("current experiment package authority is invalid")
    current_revision = current.get("producer_revision")
    expected_tree_oid = current.get("configs_experiments_tree_oid")
    if (type(current_revision) is not str or REVISION.fullmatch(current_revision) is None
            or type(expected_tree_oid) is not str or REVISION.fullmatch(expected_tree_oid) is None):
        raise SalientLocalLfPackageBuildError("current experiment package authority is invalid")
    root_tree = str(_git(root, "ls-tree", current_revision, "--", "configs/experiments")).split()
    if len(root_tree) != 4 or root_tree[:2] != ["040000", "tree"] or root_tree[2] != expected_tree_oid:
        raise SalientLocalLfPackageBuildError("current experiment root tree drifted")
    raw_current_tree = _git(root, "ls-tree", "-r", "-z", current_revision, "--", "configs/experiments", text=False)
    if type(raw_current_tree) is not bytes:
        raise SalientLocalLfPackageBuildError("current experiment tree output is invalid")
    observed_current = []
    for item in raw_current_tree.split(b"\0"):
        if not item:
            continue
        metadata, encoded_path = item.split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split()
        if kind != "blob":
            raise SalientLocalLfPackageBuildError("current experiment tree contains a non-blob")
        observed_current.append((encoded_path.decode("utf-8"), mode, blob))
    expected_current = [
        (item.get("path"), item.get("mode"), item.get("git_blob_sha"))
        for item in current["paths"] if type(item) is dict
    ]
    if (len(expected_current) != current.get("tracked_path_count")
            or observed_current != expected_current):
        raise SalientLocalLfPackageBuildError("current experiment Git inventory drifted")
    for binding in current["paths"]:
        source_path = binding.get("path")
        archive_path = binding.get("package_member_path")
        blob = binding.get("git_blob_sha")
        if (type(source_path) is not str or type(archive_path) is not str or type(blob) is not str
                or binding.get("mode") != "100644"):
            raise SalientLocalLfPackageBuildError("current experiment package binding is invalid")
        payload = _git(root, "cat-file", "blob", blob, text=False)
        if type(payload) is not bytes or sha256(payload).hexdigest() != binding.get("raw_sha256"):
            raise SalientLocalLfPackageBuildError("current experiment bytes drifted")
        expected_prefix = f"authority_inputs/current_{current_revision}/"
        pure = PurePosixPath(archive_path)
        if (pure.is_absolute() or ".." in pure.parts
                or archive_path != expected_prefix + source_path):
            raise SalientLocalLfPackageBuildError("current experiment package member path is unsafe")
        entries.append((archive_path, blob, payload))
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
        "historical_authorities/925c2cbc727e3b18e91c0b3981eeed1b470a955a/configs/experiments/content_routing_reference_fit_manifest.json",
        "historical_authorities/7c0d86d6eac5ffcfc4a30f2f5fb22884aaa848da/configs/experiments/content_uniform_combination_directional_diagnosis.json",
        "historical_authorities/7c0d86d6eac5ffcfc4a30f2f5fb22884aaa848da/configs/experiments/content_uniform_combination_directional_diagnosis_manifest.json",
        "historical_authorities/7c0d86d6eac5ffcfc4a30f2f5fb22884aaa848da/configs/experiments/content_uniform_combination_reference_fit_manifest.json",
    } | _SERVER_STARTUP_FILES
    current_members = {item.get("package_member_path") for item in current["paths"]}
    if (not required.issubset(names) or not current_members.issubset(names)
            or len(names) != len(set(names))):
        raise SalientLocalLfPackageBuildError("package source closure is incomplete")
    return tuple(entries), {current_revision: expected_tree_oid}


def _package_bytes(root: Path, revision: str) -> tuple[bytes, bytes]:
    entries, authority_roots = _tree(root, revision)
    manifest_entries = [
        {"path": path, "mode": "100644", "git_blob_sha": blob,
         "raw_sha256": sha256(payload).hexdigest(), "sha256": sha256(payload).hexdigest(),
         "size": len(payload)}
        for path, blob, payload in entries
    ]
    manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION, "package_profile": PACKAGE_PROFILE,
        "committed_revision": revision, "authority_root_tree_oids": authority_roots,
        "entries": manifest_entries,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode("utf-8")
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path, _blob, payload in (*entries, ("PACKAGE_MANIFEST.json", "", manifest_bytes)):
            info = ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return stream.getvalue(), manifest_bytes


def build_salient_local_lf_mask_write_validation_package(
    repository_root: str | Path, output_path: str | Path, revision: str,
) -> dict[str, object]:
    root = Path(repository_root)
    destination = Path(output_path)
    if REVISION.fullmatch(revision) is None or _git(root, "rev-parse", f"{revision}^{{commit}}") != revision:
        raise SalientLocalLfPackageBuildError("revision is not an exact commit")
    package_bytes, manifest_bytes = _package_bytes(root, revision)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SalientLocalLfPackageBuildError("package destination already exists")
    with destination.open("xb") as stream:
        stream.write(package_bytes)
    package_sha = sha256(destination.read_bytes()).hexdigest()
    return {"package_path": str(destination), "package_sha256": package_sha,
            "package_size_bytes": destination.stat().st_size,
            "package_manifest_sha256": sha256(manifest_bytes).hexdigest(),
            "committed_revision": revision, "package_profile": PACKAGE_PROFILE}


def verify_salient_local_lf_mask_write_validation_package(
    repository_root: str | Path, package_path: str | Path, revision: str,
) -> dict[str, object]:
    root = Path(repository_root)
    package = Path(package_path)
    if not package.is_file():
        raise SalientLocalLfPackageBuildError("execution package is unavailable")
    expected, manifest_bytes = _package_bytes(root, revision)
    actual = package.read_bytes()
    if actual != expected:
        raise SalientLocalLfPackageBuildError("existing execution package bytes drifted")
    return {
        "package_path": str(package), "package_sha256": sha256(actual).hexdigest(),
        "package_size_bytes": len(actual),
        "package_manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "committed_revision": revision, "package_profile": PACKAGE_PROFILE,
    }


def verify_extracted_salient_local_lf_mask_write_validation_package(
    repository_root: str | Path, revision: str,
) -> dict[str, object]:
    """Verify the extracted package from its frozen internal byte manifest."""

    root = Path(repository_root)
    manifest_path = root / "PACKAGE_MANIFEST.json"
    if not root.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise SalientLocalLfPackageBuildError("extracted package manifest is unavailable")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SalientLocalLfPackageBuildError("extracted package manifest is invalid") from exc
    if (type(manifest) is not dict
            or manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION
            or manifest.get("package_profile") != PACKAGE_PROFILE
            or manifest.get("committed_revision") != revision
            or json.dumps(
                manifest, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8") != manifest_bytes):
        raise SalientLocalLfPackageBuildError("extracted package identity drifted")
    entries = manifest.get("entries")
    if type(entries) is not list:
        raise SalientLocalLfPackageBuildError("extracted package inventory is invalid")
    expected_paths = []
    for entry in entries:
        if type(entry) is not dict:
            raise SalientLocalLfPackageBuildError("extracted package inventory is invalid")
        path = entry.get("path")
        pure = PurePosixPath(path) if type(path) is str else None
        if (pure is None or pure.is_absolute() or not pure.parts or ".." in pure.parts
                or "." in pure.parts or entry.get("mode") != "100644"):
            raise SalientLocalLfPackageBuildError("extracted package member identity is unsafe")
        member = root.joinpath(*pure.parts)
        if member.is_symlink() or not member.is_file():
            raise SalientLocalLfPackageBuildError("extracted package member is unavailable")
        payload = member.read_bytes()
        digest = sha256(payload).hexdigest()
        if (entry.get("size") != len(payload) or entry.get("raw_sha256") != digest
                or entry.get("sha256") != digest):
            raise SalientLocalLfPackageBuildError("extracted package member bytes drifted")
        expected_paths.append(path)
    if expected_paths != sorted(expected_paths) or len(expected_paths) != len(set(expected_paths)):
        raise SalientLocalLfPackageBuildError("extracted package inventory order drifted")
    observed_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )
    if observed_paths != sorted((*expected_paths, "PACKAGE_MANIFEST.json")):
        raise SalientLocalLfPackageBuildError("extracted package file set drifted")
    return manifest


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
