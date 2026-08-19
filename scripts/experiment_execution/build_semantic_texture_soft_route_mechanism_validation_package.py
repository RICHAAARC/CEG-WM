"""Build the exact-revision, Git-less soft-route mechanism package."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Sequence
import zipfile


PACKAGE_PROFILE = "semantic_texture_soft_route_mechanism_validation"
EMBEDDED_MANIFEST_PATH = "semantic_texture_soft_route_mechanism_validation_manifest.json"
REVISION = re.compile(r"^[0-9a-f]{40}$")
LOCAL_ABSOLUTE_PATH = re.compile(
    rb"(?<![A-Za-z0-9_])(?:/(?:home|Users|mnt|tmp|var|opt|root)/|[A-Za-z]:[\\/])"
)
EXCLUDED_PARTS = frozenset(
    {".agents", ".codex", ".git", "governance", "notebooks", "outputs", "paper_artifacts", "tests"}
)
ENTRYPOINTS = {
    "candidate_selection": "scripts/experiment_execution/semantic_texture_soft_route_candidate_selection_entrypoint.py",
    "untouched_confirmation": "scripts/experiment_execution/semantic_texture_soft_route_untouched_confirmation_entrypoint.py",
}
SOFT_ROUTE_MECHANISM_EXACT_SOURCE_FILES = frozenset(
    {
        "configs/experiments/semantic_texture_soft_route_mechanism_validation.json",
        "configs/experiments/semantic_texture_soft_route_candidate_selection_manifest.json",
        "configs/experiments/semantic_texture_soft_route_untouched_confirmation_manifest.json",
        "configs/experiments/hf_only_reference_prompt_roster.json",
        "configs/experiments/internal_execution_components.json",
        "configs/runtime/runtime_sd35_flowmatch.json",
        "experiments/__init__.py",
        "experiments/attacks/__init__.py",
        "experiments/attacks/geometric.py",
        "experiments/methods/__init__.py",
        "experiments/methods/ceg_wm.py",
        "experiments/metrics/__init__.py",
        "experiments/metrics/lf_whitened_score_screening.py",
        "experiments/protocol/internal_splits.py",
        "experiments/protocol/semantic_texture_soft_detector_assets.py",
        "experiments/protocol/semantic_texture_soft_route_mechanism_validation.py",
        "experiments/runners/semantic_texture_soft_route_mechanism_validation.py",
        "main/__init__.py",
        "main/content_chain/__init__.py",
        "main/content_chain/detector.py",
        "main/content_chain/embedder.py",
        "main/content_chain/hf_carrier.py",
        "main/content_chain/hf_detector.py",
        "main/content_chain/lf_carrier.py",
        "main/content_chain/lf_detector.py",
        "main/content_chain/lf_whitening.py",
        "main/content_chain/routing.py",
        "main/geometry_chain/__init__.py",
        "main/geometry_chain/qk_sync.py",
        "main/geometry_chain/rectifier.py",
        "main/geometry_chain/reliability.py",
        "main/geometry_chain/transform_estimator.py",
        "main/joint_decision/__init__.py",
        "main/joint_decision/detector.py",
        "main/shared/__init__.py",
        "main/shared/key_schedule.py",
        "main/shared/normal_quantile_table20_float32_be.txt",
        "main/shared/rgb8.py",
        "requirements_semantic_texture_operational_preflight.txt",
        "requirements_semantic_texture_operational_preflight_overlay.txt",
        "runtime/__init__.py",
        "runtime/adapter.py",
        "runtime/backend.py",
        "runtime/configuration.py",
        "runtime/content_write.py",
        "runtime/geometry_synchronization.py",
        "runtime/qk_observation.py",
        "runtime/routing_observation.py",
        "runtime/sd35_backend.py",
        "runtime/_vendor/transparent_background/InSPyReNet.py",
        "runtime/_vendor/transparent_background/LICENSE",
        "runtime/_vendor/transparent_background/SOURCE.json",
        "runtime/_vendor/transparent_background/__init__.py",
        "runtime/_vendor/transparent_background/backbones/SwinTransformer.py",
        "runtime/_vendor/transparent_background/backbones/__init__.py",
        "runtime/_vendor/transparent_background/modules/__init__.py",
        "runtime/_vendor/transparent_background/modules/attention_module.py",
        "runtime/_vendor/transparent_background/modules/context_module.py",
        "runtime/_vendor/transparent_background/modules/decoder_module.py",
        "runtime/_vendor/transparent_background/modules/layers.py",
        "scripts/experiment_execution/__init__.py",
        "scripts/experiment_execution/README.md",
        "scripts/experiment_execution/semantic_texture_operational_preflight_bootstrap.py",
        "scripts/experiment_execution/semantic_texture_soft_route_candidate_selection_bootstrap.py",
        "scripts/experiment_execution/semantic_texture_soft_route_candidate_selection_entrypoint.py",
        "scripts/experiment_execution/semantic_texture_soft_route_candidate_selection_server.py",
        "scripts/experiment_execution/semantic_texture_soft_route_untouched_confirmation_bootstrap.py",
        "scripts/experiment_execution/semantic_texture_soft_route_untouched_confirmation_entrypoint.py",
        "scripts/experiment_execution/semantic_texture_soft_route_untouched_confirmation_server.py",
    }
)


class SemanticTextureSoftRouteSoftRouteMechanismPackageError(RuntimeError):
    """The soft-route mechanism validation package is not an exact safe tracked-blob release."""


def _git(root: Path, *arguments: str, text: bool = True) -> str | bytes:
    try:
        result = subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True, text=text)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SemanticTextureSoftRouteSoftRouteMechanismPackageError("source repository identity is unavailable") from exc
    return result.stdout.strip() if text else result.stdout


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _safe(path_text: str) -> None:
    path = PurePosixPath(path_text)
    if (
        not path_text
        or path.is_absolute()
        or ".." in path.parts
        or "__pycache__" in path.parts
        or path.suffix in {".pyc", ".pyo"}
        or any(part in EXCLUDED_PARTS for part in path.parts)
    ):
        raise SemanticTextureSoftRouteSoftRouteMechanismPackageError("unsafe package path")


def _tracked_blobs(root: Path, revision: str) -> tuple[tuple[str, bytes], ...]:
    result: list[tuple[str, bytes]] = []
    for path_text in sorted(SOFT_ROUTE_MECHANISM_EXACT_SOURCE_FILES):
        _safe(path_text)
        metadata = _git(root, "ls-tree", revision, path_text, text=False)
        if not isinstance(metadata, bytes) or not metadata.strip():
            raise SemanticTextureSoftRouteSoftRouteMechanismPackageError(f"required package source is missing: {path_text}")
        blob = _git(root, "show", f"{revision}:{path_text}", text=False)
        if not isinstance(blob, bytes) or LOCAL_ABSOLUTE_PATH.search(blob):
            raise SemanticTextureSoftRouteSoftRouteMechanismPackageError("package source content is unsafe")
        result.append((path_text, blob))
    return tuple(result)


def _zip_info(path_text: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path_text, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_semantic_texture_soft_route_mechanism_validation_package(*, repository_root: str | Path, source_revision: str, output: str | Path, split: str) -> dict[str, object]:
    """Write one create-only archive from exact tracked blobs."""
    root, output_path = Path(repository_root).resolve(), Path(output).resolve()
    delivery_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if split not in ENTRYPOINTS or REVISION.fullmatch(source_revision) is None:
        raise SemanticTextureSoftRouteSoftRouteMechanismPackageError("soft-route mechanism validation package identity is invalid")
    if _git(root, "rev-parse", "HEAD") != source_revision or _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise SemanticTextureSoftRouteSoftRouteMechanismPackageError("source checkout must be exact and clean")
    if output_path == root or root in output_path.parents or output_path.exists() or delivery_path.exists():
        raise SemanticTextureSoftRouteSoftRouteMechanismPackageError("package output target is unsafe or occupied")
    entries = _tracked_blobs(root, source_revision)
    copied = [{"path": path, "sha256": sha256(blob).hexdigest(), "size_bytes": len(blob)} for path, blob in entries]
    identity = sha256(_canonical({"copied_files": copied, "profile_name": PACKAGE_PROFILE, "source_revision": source_revision, "split": split})).hexdigest()
    manifest = {
        "copied_files": copied,
        "entrypoint_path": ENTRYPOINTS[split],
        "package_identity": identity,
        "package_ready": True,
        "package_schema_version": 1,
        "profile_name": PACKAGE_PROFILE,
        "source_revision": source_revision,
        "split": split,
    }
    manifest_blob = _canonical(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for path_text, blob in entries:
                archive.writestr(_zip_info(path_text), blob)
            archive.writestr(_zip_info(EMBEDDED_MANIFEST_PATH), manifest_blob)
        os.link(temporary, output_path)
    except FileExistsError as exc:
        raise SemanticTextureSoftRouteSoftRouteMechanismPackageError("package output target must remain absent") from exc
    finally:
        temporary.unlink(missing_ok=True)
    delivery = {
        "archive_filename": output_path.name,
        "archive_sha256": sha256(output_path.read_bytes()).hexdigest(),
        "archive_size_bytes": output_path.stat().st_size,
        "embedded_manifest_sha256": sha256(manifest_blob).hexdigest(),
        "package_identity": identity,
        "profile_name": PACKAGE_PROFILE,
        "source_revision": source_revision,
        "split": split,
    }
    with delivery_path.open("xb") as handle:
        handle.write(_canonical(delivery))
    return {**delivery, "copied_file_count": len(copied), "delivery_manifest_path": str(delivery_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=tuple(ENTRYPOINTS), required=True)
    arguments = parser.parse_args(argv)
    print(json.dumps(build_semantic_texture_soft_route_mechanism_validation_package(repository_root=arguments.repository_root, source_revision=arguments.source_revision, output=arguments.output, split=arguments.split), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
