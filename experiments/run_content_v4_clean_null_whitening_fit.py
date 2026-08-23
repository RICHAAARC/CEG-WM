"""User-run current-stack producer for the Content V4 clean-null W asset."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import torch

from cegwm.method.content_whitening_v4 import (
    ASSET_ROLE_ID,
    FIT_MANIFEST_REPO_PATH,
    FIT_UNIT_COUNT,
    MODEL_ID,
    bind_fit_protocol,
    build_whitening_asset,
    fit_whitening_operator,
    load_fit_manifest,
    stable_json_bytes,
)
from cegwm.runtime.content_whitening_sd35_v4 import run_clean_fit_observation
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
        raise RuntimeError("cuda_required_for_real_Content_V4_whitening_fit")
    pipeline = load_sd35_pipeline(MODEL_ID, torch_dtype=torch.float16, token=token)
    pipeline.to("cuda")
    return pipeline


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _destinations(
    artifact_sink: Path,
    run_id: str,
    producer_exact: str,
) -> tuple[Path, Path]:
    execution_root = artifact_sink / run_id / producer_exact
    asset = execution_root / ASSET_FILENAME
    return asset, execution_root / f"{ASSET_FILENAME}.sha256"


def _require_create_only(asset_path: Path, checksum_path: Path) -> None:
    if asset_path.exists() or checksum_path.exists():
        raise FileExistsError("create-only whitening asset destination already exists")


def _publish_create_only(
    asset_path: Path,
    checksum_path: Path,
    payload: bytes,
    digest: str,
) -> None:
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("whitening asset payload digest differs before publication")
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


def _receipt(
    *,
    run_id: str,
    producer_exact: str,
    fit_protocol_digest: str,
    asset_digest: str,
) -> None:
    payload = {
        "asset_digest": asset_digest,
        "asset_role_id": ASSET_ROLE_ID,
        "fit_protocol_digest": fit_protocol_digest,
        "producer_exact": producer_exact,
        "run_id": run_id,
        "unit_count": FIT_UNIT_COUNT,
    }
    print(
        "CEGWM_CONTENT_V4_WHITENING_RECEIPT "
        + stable_json_bytes(payload).decode("utf-8"),
        flush=True,
    )


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    artifact_sink = Path(args.artifact_sink).resolve()
    exact = _git_exact(repo_root, args.expected_exact)
    manifest = load_fit_manifest(repo_root / FIT_MANIFEST_REPO_PATH)
    binding = bind_fit_protocol(manifest, exact)
    asset_path, checksum_path = _destinations(
        artifact_sink, binding.run_id, exact
    )
    _require_create_only(asset_path, checksum_path)

    token = os.environ.pop(TOKEN_ENV, "")
    if not token.strip():
        token = ""
        raise RuntimeError("HF_TOKEN_is_required_for_Content_V4_whitening_fit")
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
        raise RuntimeError("Content V4 whitening runner did not produce exactly 32 observations")
    fit = fit_whitening_operator(observations)
    observations.clear()
    asset = build_whitening_asset(binding, fit.words_be_hex)
    _publish_create_only(asset_path, checksum_path, asset.json_bytes, asset.digest)
    _receipt(
        run_id=binding.run_id,
        producer_exact=exact,
        fit_protocol_digest=binding.digest,
        asset_digest=asset.digest,
    )
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--artifact-sink", required=True)
    parser.add_argument("--expected-exact", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
