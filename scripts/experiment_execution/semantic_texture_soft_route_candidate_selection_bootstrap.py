"""Thin fresh-checkout bootstrap for soft-route candidate selection."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from scripts.experiment_execution.semantic_texture_operational_preflight_bootstrap import (
    SemanticTextureOperationalBootstrapError,
    _execution_environment,
    _regular_checkpoint,
    _repository_revision,
    _require_overlay_imports,
    _run_checked,
    _PYPI_INDEX_URL,
    _PYTORCH_INDEX_URL,
    _NVIDIA_INDEX_URL,
)


ENTRYPOINT_PATH = "scripts/experiment_execution/semantic_texture_soft_route_candidate_selection_entrypoint.py"
PACKAGE_MANIFEST = "semantic_texture_soft_route_mechanism_validation_manifest.json"


def _single_argument(arguments: Sequence[str], name: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise SemanticTextureOperationalBootstrapError("integrity_blocked")
    return arguments[positions[0] + 1]


def _package_revision(repository: Path) -> str:
    """Authenticate either an exact checkout or a Git-less extracted package."""

    if (repository / ".git").is_dir():
        return _repository_revision(repository)
    try:
        manifest = json.loads((repository / PACKAGE_MANIFEST).read_text(encoding="utf-8"))
        copied = manifest["copied_files"]
        revision = manifest["source_revision"]
        if (
            manifest["profile_name"] != "semantic_texture_soft_route_mechanism_validation"
            or type(copied) is not list
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise ValueError
        expected = {entry["path"] for entry in copied} | {PACKAGE_MANIFEST}
        observed = {
            path.relative_to(repository).as_posix()
            for path in repository.rglob("*")
            if path.is_file()
        }
        if observed != expected:
            raise ValueError
        for entry in copied:
            path = repository / entry["path"]
            blob = path.read_bytes()
            if len(blob) != entry["size_bytes"] or sha256(blob).hexdigest() != entry["sha256"]:
                raise ValueError
        return revision
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise SemanticTextureOperationalBootstrapError("integrity_blocked") from None


def bootstrap_soft_route_mechanism_candidate_selection(*, repository_root: str | Path, checkpoint: str | Path, execution_root: str | Path, entrypoint_args: Sequence[str], entrypoint_path: str = ENTRYPOINT_PATH) -> tuple[int, dict[str, object]]:
    """Install only the registered overlay, then execute the exact checked-out entrypoint."""

    repository, root = Path(repository_root).resolve(), Path(execution_root).resolve()
    revision = "unavailable"
    try:
        if root.exists() or not entrypoint_args or len(entrypoint_args) > 20:
            raise SemanticTextureOperationalBootstrapError("integrity_blocked")
        root.mkdir(parents=True)
        for name in ("cache", "dependencies", "persistent", "tmp"):
            (root / name).mkdir()
        revision = _package_revision(repository)
        environment = _execution_environment(repository, root, _regular_checkpoint(Path(checkpoint)))
        _run_checked([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--index-url", _PYPI_INDEX_URL, "--extra-index-url", _PYTORCH_INDEX_URL, "--extra-index-url", _NVIDIA_INDEX_URL, "--requirement", str(repository / "requirements_semantic_texture_operational_preflight_overlay.txt"), "--no-deps", "--target", str(root / "dependencies")], cwd=repository, environment=environment)
        _require_overlay_imports(repository, root / "dependencies")
        if entrypoint_path not in {ENTRYPOINT_PATH, "scripts/experiment_execution/semantic_texture_soft_route_untouched_confirmation_entrypoint.py"}:
            raise SemanticTextureOperationalBootstrapError("integrity_blocked")
        if "--observed-repository-revision" in entrypoint_args:
            raise SemanticTextureOperationalBootstrapError("identity_blocked")
        completed = subprocess.run(
            [
                sys.executable,
                str(repository / entrypoint_path),
                "--observed-repository-revision",
                revision,
                *entrypoint_args,
            ],
            cwd=repository,
            env=environment,
            check=False,
        )
        return completed.returncode, {"observed_repository_revision": revision, "status": "passed" if completed.returncode == 0 else "blocked", "stage": "entrypoint"}
    except Exception as error:
        blocked = error.blocked_class if isinstance(error, SemanticTextureOperationalBootstrapError) else "implementation_blocked"
        receipt: dict[str, object] = {"blocked_class": blocked, "status": "blocked", "stage": "bootstrap"}
        try:
            output_root = _single_argument(entrypoint_args, "--output-root")
            run_id = _single_argument(entrypoint_args, "--run-id")
            if entrypoint_path == ENTRYPOINT_PATH:
                from scripts.experiment_execution.semantic_texture_soft_route_candidate_selection_server import finalize_soft_route_mechanism_failure_delivery
                _code, persisted = finalize_soft_route_mechanism_failure_delivery(
                    observed_repository_revision=revision,
                    run_id=run_id,
                    output_root=output_root,
                    stage="bootstrap",
                    failure_reason=blocked,
                )
            else:
                from scripts.experiment_execution.semantic_texture_soft_route_untouched_confirmation_server import finalize_soft_route_mechanism_untouched_confirmation_failure_delivery
                _code, persisted = finalize_soft_route_mechanism_untouched_confirmation_failure_delivery(
                    observed_repository_revision=revision,
                    run_id=run_id,
                    output_root=output_root,
                    stage="bootstrap",
                    failure_reason=blocked,
                )
            receipt.update(persisted)
        except Exception:
            receipt["failure_delivery_status"] = "not_created"
        return 2, receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--execution-root", required=True)
    parser.add_argument("--entrypoint-args", nargs=argparse.REMAINDER, required=True)
    arguments = parser.parse_args(argv)
    code, receipt = bootstrap_soft_route_mechanism_candidate_selection(repository_root=arguments.repository_root, checkpoint=arguments.checkpoint, execution_root=arguments.execution_root, entrypoint_args=arguments.entrypoint_args)
    print(json.dumps(receipt, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
