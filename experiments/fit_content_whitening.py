"""User-run producer for the content-whitening clean-null W asset."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import torch

from cegwm.method.content_whitening import (
    ASSET_ROLE_ID,
    FIT_MANIFEST_REPO_PATH,
    FIT_UNIT_COUNT,
    MODEL_ID,
    build_whitening_asset,
    fit_whitening_operator,
    load_fit_manifest,
    stable_json_bytes,
)
from cegwm.runtime.content_whitening_sd35 import run_clean_fit_observation
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline

TOKEN_ENV = "HF_TOKEN"
ASSET_FILENAME = f"{ASSET_ROLE_ID}.json"


def _git_exact(repo_root: Path, expected_exact: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected_exact) is None:
        raise ValueError("expected exact must be a lowercase 40-character revision")
    exact = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if exact != expected_exact:
        raise RuntimeError("resolved revision differs from approved producer exact")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("producer checkout must be clean")
    return exact


def _load_pipeline(token: str) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_real_content_whitening_fit")
    pipeline = load_sd35_pipeline(MODEL_ID, torch_dtype=torch.float16, token=token)
    pipeline.to("cuda")
    return pipeline


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _destinations(artifact_sink: Path, producer_exact: str) -> tuple[Path, Path]:
    execution_root = artifact_sink / producer_exact
    asset = execution_root / ASSET_FILENAME
    return asset, execution_root / f"{ASSET_FILENAME}.sha256"


def _require_create_only(asset_path: Path, checksum_path: Path) -> None:
    if asset_path.exists() or checksum_path.exists():
        raise FileExistsError("create-only whitening asset destination already exists")


def _publish_create_only(
    asset_path: Path,
    checksum_path: Path,
    payload: bytes,
) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    _require_create_only(asset_path, checksum_path)
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    created_asset = False
    try:
        with asset_path.open("xb") as handle:
            handle.write(payload)
        created_asset = True
        with checksum_path.open("xb") as handle:
            handle.write(f"{digest}  {ASSET_FILENAME}\n".encode("ascii"))
    except BaseException:
        if created_asset:
            asset_path.unlink(missing_ok=True)
        raise
    return digest


def _receipt(*, producer_exact: str, asset_sha256: str) -> None:
    payload = {
        "asset_sha256": asset_sha256,
        "producer_exact": producer_exact,
        "unit_count": FIT_UNIT_COUNT,
    }
    print(
        "CEGWM_CONTENT_WHITENING_RECEIPT "
        + stable_json_bytes(payload).decode("utf-8"),
        flush=True,
    )


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    artifact_sink = Path(args.artifact_sink).resolve()
    exact = _git_exact(repo_root, args.expected_exact)
    manifest = load_fit_manifest(repo_root / FIT_MANIFEST_REPO_PATH)
    asset_path, checksum_path = _destinations(artifact_sink, exact)
    _require_create_only(asset_path, checksum_path)

    token = os.environ.pop(TOKEN_ENV, "")
    if not token.strip():
        token = ""
        raise RuntimeError("HF_TOKEN_is_required_for_content_whitening_fit")
    try:
        pipeline = _load_pipeline(token)
    finally:
        token = ""

    observations = []
    for entry in manifest.entries:
        observations.append(
            run_clean_fit_observation(
                pipeline,
                entry,
                generator=_generator(entry.generation_seed),
            )
        )
    if len(observations) != FIT_UNIT_COUNT:
        raise RuntimeError("content-whitening whitening runner did not produce exactly 32 observations")
    fit = fit_whitening_operator(observations)
    observations.clear()
    asset = build_whitening_asset(exact, fit.words_be_hex)
    asset_sha256 = _publish_create_only(asset_path, checksum_path, asset.json_bytes)
    _receipt(producer_exact=exact, asset_sha256=asset_sha256)
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--artifact-sink", required=True)
    parser.add_argument("--expected-exact", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
