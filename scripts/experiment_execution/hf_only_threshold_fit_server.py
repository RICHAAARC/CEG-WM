"""Server and Colab-neutral entrypoint for one HF-only threshold-fit shard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.experiment_execution.build_experiment_execution_package import (
    build_experiment_execution_package,
)
from scripts.experiment_execution import experiment_execution_bootstrap as bootstrap


REVISION = re.compile(r"^[0-9a-f]{40}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
RUNTIME_CONFIG_PATH = Path("configs/runtime/runtime_sd35_flowmatch.json")
EXECUTION_CONFIG_PATH = Path(
    "configs/experiments/hf_only_threshold_fit_gpu_execution.json"
)


class HfOnlyThresholdFitServerError(RuntimeError):
    """The direct execution preflight or receipt boundary failed closed."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        failure_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.safe_message = message
        self.failure_type = failure_type or type(self).__name__


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HfOnlyThresholdFitServerError(
            "repository",
            "repository identity is unavailable",
            failure_type=type(exc).__name__,
        ) from exc
    return completed.stdout.strip()


def _absolute_root(value: str | Path, role: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise HfOnlyThresholdFitServerError(
            "arguments",
            f"{role} must be absolute",
        )
    return path.resolve()


def _overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _verify_repository(root: Path, expected_revision: str) -> None:
    if not root.is_dir() or REVISION.fullmatch(expected_revision) is None:
        raise HfOnlyThresholdFitServerError(
            "repository",
            "repository root or exact revision is invalid",
        )
    if _git(root, "rev-parse", "HEAD") != expected_revision:
        raise HfOnlyThresholdFitServerError(
            "repository",
            "repository HEAD differs from expected_revision",
        )
    if _git(root, "status", "--porcelain"):
        raise HfOnlyThresholdFitServerError(
            "repository",
            "repository worktree must be clean",
        )


def _load_execution_bindings(root: Path) -> tuple[str, str, int]:
    try:
        runtime = json.loads((root / RUNTIME_CONFIG_PATH).read_text("utf-8"))
        execution = json.loads((root / EXECUTION_CONFIG_PATH).read_text("utf-8"))
        model_id = runtime["model_id"]
        model_revision = runtime["model_revision"]
        minimum_vram_bytes = execution["resource_plan"]["minimum_vram_bytes"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise HfOnlyThresholdFitServerError(
            "configuration",
            "frozen runtime or execution configuration is unreadable",
            failure_type=type(exc).__name__,
        ) from exc
    if (
        type(model_id) is not str
        or not model_id
        or type(model_revision) is not str
        or REVISION.fullmatch(model_revision) is None
        or type(minimum_vram_bytes) is not int
        or isinstance(minimum_vram_bytes, bool)
        or minimum_vram_bytes <= 0
    ):
        raise HfOnlyThresholdFitServerError(
            "configuration",
            "frozen runtime or execution configuration is invalid",
        )
    return model_id, model_revision, minimum_vram_bytes


def _probe_resources(
    *,
    roots: tuple[Path, ...],
    minimum_vram_bytes: int,
) -> dict[str, object]:
    free_bytes: dict[str, int] = {}
    try:
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)
            observed = int(shutil.disk_usage(root).free)
            if observed <= 0:
                raise HfOnlyThresholdFitServerError(
                    "resource_preflight",
                    "execution storage has no available space",
                )
            free_bytes[str(root)] = observed
    except HfOnlyThresholdFitServerError:
        raise
    except OSError as exc:
        raise HfOnlyThresholdFitServerError(
            "resource_preflight",
            "execution storage identity is unavailable",
            failure_type=type(exc).__name__,
        ) from exc
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
        name, memory_mib = (part.strip() for part in rows[0].rsplit(",", 1))
        total_memory_bytes = int(memory_mib) * 1024 * 1024
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError) as exc:
        raise HfOnlyThresholdFitServerError(
            "resource_preflight",
            "cuda device identity is unavailable",
            failure_type=type(exc).__name__,
        ) from exc
    if total_memory_bytes < minimum_vram_bytes:
        raise HfOnlyThresholdFitServerError(
            "resource_preflight",
            "cuda:0 is below the frozen model-agnostic VRAM floor",
        )
    return {
        "cuda_device_name": name,
        "cuda_total_memory_bytes": total_memory_bytes,
        "free_disk_bytes": free_bytes,
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_path_identity(value: str, role: str) -> str:
    if SAFE_ID.fullmatch(value):
        return value
    return f"invalid-{role}-{sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _preflight_failure_receipt(
    *,
    output_root: str | Path,
    expected_revision: str,
    run_id: str,
    shard_index: int,
    error: HfOnlyThresholdFitServerError,
) -> dict[str, object]:
    output = Path(output_root).resolve()
    safe_revision = (
        expected_revision
        if REVISION.fullmatch(expected_revision)
        else _safe_path_identity(expected_revision, "revision")
    )
    safe_run_id = _safe_path_identity(run_id, "run")
    safe_shard = shard_index if isinstance(shard_index, int) else -1
    finished = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    diagnostic_path = (
        output
        / "server_preflight_failures"
        / safe_revision
        / safe_run_id
        / f"shard_{safe_shard:02d}"
        / f"ceg_wm_server_preflight_failure_{safe_run_id}_{finished}.zip"
    )
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic = {
        "artifact_kind": "server_preflight_failure",
        "committed_revision": expected_revision,
        "failure_message": error.safe_message,
        "failure_stage": error.stage,
        "failure_type": error.failure_type,
        "run_id": run_id,
        "shard_index": shard_index,
        "scientific_claims_supported": False,
        "tau_approval": False,
        "confirmation_unlock": False,
    }
    with zipfile.ZipFile(
        diagnostic_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            "server_preflight_diagnostic.json",
            json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
        )
    receipt_path = (
        output
        / "execution_receipts"
        / safe_revision
        / safe_run_id
        / f"shard_{safe_shard:02d}"
        / "execution_receipt.json"
    )
    receipt = {
        **diagnostic,
        "artifact_path": str(diagnostic_path),
        "artifact_sha256": _sha256_file(diagnostic_path),
        "bootstrap_exit_code": 3,
        "receipt_path": str(receipt_path),
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def execute_server_threshold_fit_shard(
    *,
    repository_root: str | Path,
    expected_revision: str,
    scratch_root: str | Path,
    cache_root: str | Path,
    output_root: str | Path,
    run_id: str,
    shard_index: int,
    environment: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    """Build, verify, and execute one formal shard through the existing bootstrap."""

    repository = _absolute_root(repository_root, "repository_root")
    scratch = _absolute_root(scratch_root, "scratch_root")
    cache = _absolute_root(cache_root, "cache_root")
    output = _absolute_root(output_root, "output_root")
    roots = (scratch, cache, output)
    if any(
        _overlap(first, second)
        for index, first in enumerate(roots)
        for second in roots[index + 1 :]
    ) or any(_overlap(repository, root) for root in roots):
        raise HfOnlyThresholdFitServerError(
            "arguments",
            "repository, scratch, cache, and output roots must be disjoint",
        )
    if SAFE_ID.fullmatch(run_id) is None or not 0 <= shard_index < 16:
        raise HfOnlyThresholdFitServerError(
            "arguments",
            "run_id or shard_index is invalid",
        )
    _verify_repository(repository, expected_revision)
    model_id, model_revision, minimum_vram_bytes = _load_execution_bindings(
        repository
    )
    runtime_environment = dict(os.environ if environment is None else environment)
    if not runtime_environment.get("HF_TOKEN") or not runtime_environment.get(
        "CEG_WM_ROOT_KEY"
    ):
        raise HfOnlyThresholdFitServerError(
            "secrets",
            "HF_TOKEN and CEG_WM_ROOT_KEY are required",
        )
    resource_facts = _probe_resources(
        roots=roots,
        minimum_vram_bytes=minimum_vram_bytes,
    )
    build_root = Path(
        tempfile.mkdtemp(dir=scratch, prefix="hf_only_threshold_fit_build_")
    )
    package_zip = build_root / "ceg_wm_hf_only_threshold_fit.zip"
    try:
        build = build_experiment_execution_package(
            root=repository,
            output_zip=package_zip,
            committed_revision=expected_revision,
        )
    except Exception as exc:
        raise HfOnlyThresholdFitServerError(
            "package_build",
            "dedicated execution package build failed",
            failure_type=type(exc).__name__,
        ) from exc
    sidecar = Path(str(build["delivery_manifest_path"]))
    bootstrap_path = (
        repository
        / "scripts/experiment_execution/experiment_execution_bootstrap.py"
    )
    bootstrap_sha256 = _sha256_file(bootstrap_path)
    bootstrap_exit_code, bootstrap_result = bootstrap.run_bootstrap(
        package_zip=package_zip,
        delivery_manifest_path=sidecar,
        expected_archive_sha256=str(build["archive_sha256"]),
        expected_delivery_manifest_sha256=_sha256_file(sidecar),
        expected_embedded_manifest_sha256=str(build["embedded_manifest_sha256"]),
        expected_bootstrap_identity=bootstrap.BOOTSTRAP_IDENTITY,
        expected_bootstrap_schema_version=bootstrap.BOOTSTRAP_SCHEMA_VERSION,
        expected_bootstrap_sha256=bootstrap_sha256,
        expected_revision=expected_revision,
        ephemeral_root=build_root / "bootstrap_scratch",
        persistent_root=output,
        model_cache_root=cache,
        prepare_frozen_model=True,
        run_id=run_id,
        shard_index=shard_index,
        environment=runtime_environment,
    )
    artifact_value = bootstrap_result.get("result_zip") or bootstrap_result.get(
        "diagnostic_zip"
    )
    if type(artifact_value) is not str:
        raise HfOnlyThresholdFitServerError(
            "bootstrap_result",
            "bootstrap did not return a result or diagnostic ZIP",
        )
    artifact_path = Path(artifact_value).resolve()
    if not artifact_path.is_file() or output not in artifact_path.parents:
        raise HfOnlyThresholdFitServerError(
            "bootstrap_result",
            "bootstrap artifact is unavailable or outside output_root",
        )
    receipt_path = (
        output
        / "execution_receipts"
        / expected_revision
        / run_id
        / f"shard_{shard_index:02d}"
        / "execution_receipt.json"
    )
    receipt = {
        "artifact_kind": bootstrap_result["artifact_kind"],
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256_file(artifact_path),
        "bootstrap_exit_code": bootstrap_exit_code,
        "bootstrap_sha256": bootstrap_sha256,
        "archive_sha256": build["archive_sha256"],
        "delivery_manifest_sha256": _sha256_file(sidecar),
        "embedded_manifest_sha256": build["embedded_manifest_sha256"],
        "candidate_config_digest": build["candidate_config_digest"],
        "execution_config_digest": build["execution_config_digest"],
        "input_manifest_digest": build["input_manifest_digest"],
        "committed_revision": expected_revision,
        "model_id": model_id,
        "model_revision": model_revision,
        "run_id": run_id,
        "shard_index": shard_index,
        "resource_facts": resource_facts,
        "scientific_claims_supported": False,
        "tau_approval": False,
        "confirmation_unlock": False,
    }
    _atomic_json(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    return bootstrap_exit_code, receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shard-index", required=True, type=int)
    arguments = parser.parse_args(argv)
    try:
        exit_code, receipt = execute_server_threshold_fit_shard(
            repository_root=arguments.repository_root,
            expected_revision=arguments.expected_revision,
            scratch_root=arguments.scratch_root,
            cache_root=arguments.cache_root,
            output_root=arguments.output_root,
            run_id=arguments.run_id,
            shard_index=arguments.shard_index,
        )
    except HfOnlyThresholdFitServerError as error:
        receipt = _preflight_failure_receipt(
            output_root=arguments.output_root,
            expected_revision=arguments.expected_revision,
            run_id=arguments.run_id,
            shard_index=arguments.shard_index,
            error=error,
        )
        exit_code = 3
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
