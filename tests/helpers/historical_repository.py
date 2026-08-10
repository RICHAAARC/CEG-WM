"""Local-Git-only materialization for producer-bound research tests."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Iterable


REVISION = re.compile(r"[0-9a-f]{40}")
HF_REFERENCE_PRODUCER_REVISION = "cc9af5df0d9a63d349402d56ddd6bb81d117d1e8"
HF_REFERENCE_PRODUCER_PATHS = (
    "configs/experiments/hf_only_reference_validation.json",
    "configs/experiments/hf_only_reference_metrics.json",
    "configs/experiments/hf_only_threshold_fit_gpu_execution.json",
    "configs/experiments/hf_only_reference_prompt_roster.json",
    "configs/experiments/hf_only_content_threshold_fit_manifest.json",
    "configs/experiments/hf_only_untouched_confirmation_manifest.json",
    "configs/experiments/assets/parti_prompts_dataset_snapshot.txt",
    "configs/experiments/internal_execution_components.json",
    "configs/runtime/runtime_sd35_flowmatch.json",
    "docs/design/candidate_specifications.md",
    "main/shared/key_schedule.py",
    "main/content_chain/hf_carrier.py",
    "main/content_chain/embedder.py",
    "main/content_chain/hf_detector.py",
    "main/content_chain/detector.py",
    "experiments/protocol/hf_only_reference_protocol.py",
    "experiments/metrics/hf_only_reference_metrics.py",
    "experiments/runners/hf_only_threshold_fit_gpu_execution.py",
)
HF_REFERENCE_SOURCE_EQUIVALENCE_PATHS = (
    "experiments/protocol/hf_only_reference_protocol.py",
    "experiments/metrics/hf_only_reference_metrics.py",
    "experiments/runners/hf_only_threshold_fit_gpu_execution.py",
)


class HistoricalRepositoryError(RuntimeError):
    """A requested producer tree cannot be materialized from local Git."""


def _git(root: Path, *arguments: str, text: bool = False) -> bytes | str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HistoricalRepositoryError(
            "historical producer is unavailable in the local Git object database"
        ) from exc
    return completed.stdout


def materialize_historical_repository(
    *,
    source_root: str | Path,
    revision: str,
    destination: str | Path,
    paths: Iterable[str],
    source_equivalence_paths: Iterable[str] = (),
) -> Path:
    """Write exact committed blobs to ``destination`` without fetch or fallback."""

    root = Path(source_root).resolve()
    target = Path(destination).resolve()
    if REVISION.fullmatch(revision) is None:
        raise HistoricalRepositoryError("historical producer revision is invalid")
    if target.exists():
        raise HistoricalRepositoryError("historical materialization target already exists")
    requested = tuple(paths)
    if not requested or len(requested) != len(set(requested)):
        raise HistoricalRepositoryError("historical path allowlist is empty or duplicated")
    normalized: list[str] = []
    for path_text in requested:
        path = PurePosixPath(path_text)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != path_text:
            raise HistoricalRepositoryError("historical path is unsafe")
        normalized.append(path_text)

    listed = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        revision,
        "--",
        *normalized,
        text=True,
    )
    if not isinstance(listed, str) or set(listed.splitlines()) != set(normalized):
        raise HistoricalRepositoryError("historical producer path set is incomplete")

    blobs: dict[str, bytes] = {}
    for path_text in normalized:
        blob = _git(root, "show", f"{revision}:{path_text}")
        if not isinstance(blob, bytes):
            raise HistoricalRepositoryError("historical producer blob is invalid")
        blobs[path_text] = blob
    for path_text in source_equivalence_paths:
        if path_text not in blobs:
            raise HistoricalRepositoryError(
                "source-equivalence path is outside the historical allowlist"
            )
        current = root / path_text
        if not current.is_file() or current.read_bytes() != blobs[path_text]:
            raise HistoricalRepositoryError(
                f"current source differs from historical producer: {path_text}"
            )

    target.mkdir(parents=True)
    for path_text, blob in blobs.items():
        output = target / path_text
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(blob)
    return target


def materialize_historical_tree(
    *,
    source_root: str | Path,
    revision: str,
    destination: str | Path,
    source_equivalence_paths: Iterable[str] = (),
) -> Path:
    """Materialize every tracked file from one locally available producer."""

    root = Path(source_root).resolve()
    listed = _git(root, "ls-tree", "-r", "--name-only", revision, text=True)
    if not isinstance(listed, str):
        raise HistoricalRepositoryError("historical producer tree is invalid")
    paths = tuple(path for path in listed.splitlines() if path)
    return materialize_historical_repository(
        source_root=root,
        revision=revision,
        destination=destination,
        paths=paths,
        source_equivalence_paths=source_equivalence_paths,
    )


__all__ = [
    "HF_REFERENCE_PRODUCER_PATHS",
    "HF_REFERENCE_PRODUCER_REVISION",
    "HF_REFERENCE_SOURCE_EQUIVALENCE_PATHS",
    "HistoricalRepositoryError",
    "materialize_historical_repository",
    "materialize_historical_tree",
]
