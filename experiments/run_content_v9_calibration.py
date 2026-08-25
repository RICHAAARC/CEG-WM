"""User-run producer for the Content V9 paired-null calibration asset."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import torch

from experiments import run_content_v6_clean as v6_runner
from cegwm.method.content_weighted_joint_v9 import (
    build_calibration_asset,
    fit_weighted_joint_calibration,
    stable_json_bytes,
)
from cegwm.protocol.content_chain_v9 import (
    CONTENT_V9_CALIBRATION_ASSET_ROLE_ID,
    CONTENT_V9_CALIBRATION_KEY_DOMAIN,
    CONTENT_V9_CALIBRATION_RECEIPT_ID,
    CONTENT_V9_PAIRED_NULL_SCORE_COUNT,
    deterministic_calibration_run_id,
    load_content_v9_phase1_contract,
)
from cegwm.runtime.content_weighted_joint_sd35_v9 import (
    ContentV9CalibrationAssets,
    run_content_v9_calibration_unit,
)
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from cegwm.shared.prg import prg_bytes

KEY_ENV = "CEG_WM_ROOT_KEY"
TOKEN_ENV = "HF_TOKEN"
ASSET_FILENAME = f"{CONTENT_V9_CALIBRATION_ASSET_ROLE_ID}.json"
RECEIPT_PREFIX = "CEGWM_CONTENT_V9_CALIBRATION_RECEIPT"


def _git_exact(repo_root: Path, expected_exact: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected_exact) is None:
        raise ValueError("expected exact must be lowercase 40-hex")
    exact = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout
    if exact != expected_exact or status:
        raise RuntimeError("Content V9 calibration checkout identity differs")
    return exact


def _destinations(artifact_sink: Path, run_id: str) -> tuple[Path, Path]:
    run_root = artifact_sink / run_id
    asset_path = run_root / ASSET_FILENAME
    return asset_path, run_root / f"{ASSET_FILENAME}.sha256"


def _require_create_only(asset_path: Path, sidecar_path: Path) -> None:
    if asset_path.exists() or sidecar_path.exists():
        raise FileExistsError("create-only Content V9 calibration destination exists")


def _publish_create_only(asset_path: Path, sidecar_path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    _require_create_only(asset_path, sidecar_path)
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    created_asset = False
    created_sidecar = False
    try:
        with asset_path.open("xb") as handle:
            handle.write(payload)
        created_asset = True
        with sidecar_path.open("xb") as handle:
            created_sidecar = True
            handle.write(f"{digest}  {ASSET_FILENAME}\n".encode("ascii"))
    except BaseException:
        if created_sidecar:
            sidecar_path.unlink(missing_ok=True)
        if created_asset:
            asset_path.unlink(missing_ok=True)
        raise
    return digest


def derive_calibration_key(root_key: str | bytes | bytearray | memoryview) -> bytes:
    return prg_bytes(normalize_detection_key(root_key), CONTENT_V9_CALIBRATION_KEY_DOMAIN, 32)


def _load_pipeline_and_assets(token: str) -> tuple[Any, ContentV9CalibrationAssets]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_real_Content_V9_calibration")
    pipeline, assets = v6_runner._load_pipeline_and_assets(
        "stabilityai/stable-diffusion-3.5-medium", token
    )
    return pipeline, ContentV9CalibrationAssets(assets.evaluation_assets)


def _receipt(
    *, asset_sha256: str, exact: str, protocol_digest: str,
    public_digest: str, run_id: str,
) -> None:
    payload = {
        "asset_sha256": asset_sha256,
        "calibration_pair_count": CONTENT_V9_PAIRED_NULL_SCORE_COUNT,
        "calibration_public_key_digest": public_digest,
        "producer_exact": exact,
        "protocol_digest": protocol_digest,
        "receipt_contract_id": CONTENT_V9_CALIBRATION_RECEIPT_ID,
        "run_id": run_id,
        "status": "calibration_asset_created",
    }
    print(f"{RECEIPT_PREFIX} {stable_json_bytes(payload).decode('ascii')}", flush=True)


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    sink = Path(args.artifact_sink).resolve()
    exact = _git_exact(repo_root, args.expected_exact)
    contract = load_content_v9_phase1_contract(repo_root)

    root_key_text = os.environ.pop(KEY_ENV, "")
    token = os.environ.pop(TOKEN_ENV, "")
    if not root_key_text:
        token = ""
        raise RuntimeError("CEG_WM_ROOT_KEY_is_required_for_Content_V9_calibration")
    calibration_key = derive_calibration_key(root_key_text)
    root_key_text = ""
    public_digest = public_key_digest(calibration_key)
    run_id = deterministic_calibration_run_id(contract.protocol_digest, public_digest)
    asset_path, sidecar_path = _destinations(sink, run_id)
    _require_create_only(asset_path, sidecar_path)
    if not token.strip():
        calibration_key = b""
        raise RuntimeError("HF_TOKEN_is_required_for_Content_V9_calibration")
    try:
        pipeline, assets = _load_pipeline_and_assets(token)
    finally:
        token = ""

    pairs = []
    for unit in contract.calibration:
        pairs.extend(run_content_v9_calibration_unit(pipeline, unit, calibration_key, assets))
    if len(pairs) != CONTENT_V9_PAIRED_NULL_SCORE_COUNT:
        raise RuntimeError("Content V9 calibration did not produce exactly 1056 pairs")
    fit = fit_weighted_joint_calibration(pairs)
    pairs.clear()
    asset = build_calibration_asset(
        producer_exact=exact,
        protocol_digest=contract.protocol_digest,
        public_key_digest=public_digest,
        fit=fit,
    )
    calibration_key = b""
    asset_sha256 = _publish_create_only(asset_path, sidecar_path, asset.json_bytes)
    _receipt(
        asset_sha256=asset_sha256,
        exact=exact,
        protocol_digest=contract.protocol_digest,
        public_digest=public_digest,
        run_id=run_id,
    )
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--artifact-sink", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
