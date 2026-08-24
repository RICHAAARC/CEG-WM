"""User-run producer for the Content V6 detector-domain ISS g/m asset."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import torch

from experiments import run_content_v4_clean as v4_runner
from cegwm.method.content_iss_v6 import (
    ISS_ASSET_ROLE_ID,
    build_iss_asset,
    derive_development_key,
    fit_iss_gain_target,
    stable_json_bytes,
)
from cegwm.protocol.content_chain_v6 import (
    V6_PERSONAL_SPEC_SHA256,
    load_content_v6_data_contract,
)
from cegwm.runtime.content_iss_sd35_v6 import (
    ContentV6DevelopmentAssets,
    run_content_v6_development_pair,
)
from cegwm.shared.keys import public_key_digest

KEY_ENV = "CEG_WM_ROOT_KEY"
TOKEN_ENV = "HF_TOKEN"
ASSET_FILENAME = f"{ISS_ASSET_ROLE_ID}.json"
FIT_RECEIPT_PREFIX = "CEGWM_CONTENT_V6_ISS_FIT_RECEIPT"


def _git_exact(repo_root: Path, expected_exact: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected_exact) is None:
        raise ValueError("expected exact must be lowercase 40-hex")
    exact = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if exact != expected_exact or status:
        raise RuntimeError("Content V6 ISS producer checkout identity differs")
    return exact


def _destinations(artifact_sink: Path, producer_exact: str) -> tuple[Path, Path]:
    root = artifact_sink / producer_exact
    asset = root / ASSET_FILENAME
    return asset, root / f"{ASSET_FILENAME}.sha256"


def _require_create_only(asset_path: Path, sidecar_path: Path) -> None:
    if asset_path.exists() or sidecar_path.exists():
        raise FileExistsError("create-only Content V6 ISS asset destination exists")


def _publish_create_only(asset_path: Path, sidecar_path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    _require_create_only(asset_path, sidecar_path)
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    created_asset = False
    try:
        with asset_path.open("xb") as handle:
            handle.write(payload)
        created_asset = True
        with sidecar_path.open("xb") as handle:
            handle.write(f"{digest}  {ASSET_FILENAME}\n".encode("ascii"))
    except BaseException:
        if created_asset:
            asset_path.unlink(missing_ok=True)
        raise
    return digest


def _load_pipeline_and_assets(token: str) -> tuple[Any, ContentV6DevelopmentAssets]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_real_Content_V6_ISS_fit")
    pipeline, v4_assets = v4_runner._load_pipeline_and_assets(
        "stabilityai/stable-diffusion-3.5-medium", token
    )
    return pipeline, ContentV6DevelopmentAssets(
        v4_assets.embed_assets,
        v4_assets.lf_public_assets,
    )


def _receipt(*, producer_exact: str, asset_sha256: str, key_digest: str) -> None:
    payload = {
        "asset_sha256": asset_sha256,
        "development_public_key_digest": key_digest,
        "fit_sample_count": 32,
        "personal_spec_sha256": V6_PERSONAL_SPEC_SHA256,
        "producer_exact": producer_exact,
    }
    print(f"{FIT_RECEIPT_PREFIX} {stable_json_bytes(payload).decode('ascii')}", flush=True)


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    artifact_sink = Path(args.artifact_sink).resolve()
    exact = _git_exact(repo_root, args.expected_exact)
    contract = load_content_v6_data_contract(repo_root)
    asset_path, sidecar_path = _destinations(artifact_sink, exact)
    _require_create_only(asset_path, sidecar_path)

    root_key_text = os.environ.pop(KEY_ENV, "")
    token = os.environ.pop(TOKEN_ENV, "")
    if not root_key_text:
        token = ""
        raise RuntimeError("CEG_WM_ROOT_KEY_is_required_for_Content_V6_ISS_fit")
    development_key = derive_development_key(root_key_text)
    root_key_text = ""
    if not token.strip():
        development_key = b""
        raise RuntimeError("HF_TOKEN_is_required_for_Content_V6_ISS_fit")
    try:
        pipeline, assets = _load_pipeline_and_assets(token)
    finally:
        token = ""

    measurements = []
    for unit in contract.development:
        measurements.append(
            run_content_v6_development_pair(
                pipeline,
                unit,
                development_key,
                assets,
            )
        )
    if len(measurements) != 32:
        raise RuntimeError("Content V6 ISS fit did not complete exactly 32 units")
    fit = fit_iss_gain_target(measurements)
    measurements.clear()
    asset = build_iss_asset(exact, development_key, fit)
    key_digest = public_key_digest(development_key)
    development_key = b""
    asset_sha256 = _publish_create_only(asset_path, sidecar_path, asset.json_bytes)
    _receipt(producer_exact=exact, asset_sha256=asset_sha256, key_digest=key_digest)
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--artifact-sink", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
