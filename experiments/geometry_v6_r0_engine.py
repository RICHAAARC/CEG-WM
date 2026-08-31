"""Geometry-V6 R0 clean-identity diagnostic; never a science adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from cegwm.method.content_adaptive import score_content_image
from cegwm.method.content_weighted_joint import load_calibration_asset, weighted_joint_score
from cegwm.method.geometry_v6_roundtrip import (
    GEOMETRY_V6_METHOD_ID,
    R0_AMPLITUDE_CANDIDATES,
    blind_geometry_observation,
    derive_geometry_keys,
)
from cegwm.runtime.content_adaptive_sd35 import ContentEmbedAssets, load_dino_content_assets
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline
from cegwm.runtime.geometry_v6_sd35 import run_sd35_geometry_v6_r0_arm
from cegwm.shared.keys import public_key_digest

CONTENT_KEY_ENV = "CEG_WM_ROOT_KEY"
GEOMETRY_KEY_ENV = "CEG_WM_GEOMETRY_KEY"
WRONG_GEOMETRY_KEY_ENV = "CEG_WM_GEOMETRY_WRONG_KEY"
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
ARMS = ("content_only", "content_geometry", "geometry_only", "unwatermarked")


def _exact(repo_root: Path, expected: str) -> str:
    if len(expected) != 40 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected exact must be a lowercase 40-character commit")
    current = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    clean = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, check=True, capture_output=True, text=True).stdout
    if current != expected or clean:
        raise RuntimeError("R0 diagnostic requires its approved detached, clean exact")
    return current


def _load_assets(token: str) -> tuple[Any, ContentEmbedAssets]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_user_run_nonformal_r0_diagnostic")
    # Reuse the frozen content implementation's real asset construction, without
    # changing its writer/detector files.
    from cegwm.method.hf import FrozenHFPublicAssets
    from cegwm.method.lf import (
        LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        FrozenLFPublicAssets,
    )

    pipeline = load_sd35_pipeline(MODEL_ID, torch_dtype=torch.float16, token=token)
    pipeline.to("cuda")
    vae, processor = pipeline.vae, pipeline.image_processor
    hf = FrozenHFPublicAssets(vae=vae, image_processor=processor, image_processor_id=f"{MODEL_ID}:image_processor")
    lf = FrozenLFPublicAssets(vae=vae, image_processor=processor, image_processor_id=f"{MODEL_ID}:image_processor", candidate_id=LF_BALANCED_BLOCKS_CARRIER_METHOD_ID, detector_statistic_id=LF_BLOCKNORM_DETECTOR_STATISTIC_ID, evaluated_candidate_id=LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID)
    dino_model, dino_processor = load_dino_content_assets(token=token)
    dino_model.to("cuda").eval()
    return pipeline, ContentEmbedAssets(dino_model, dino_processor, hf, lf)


def _record_content(image: Any, content_key: str, assets: ContentEmbedAssets, calibration: Any) -> dict[str, Any]:
    score = score_content_image(image, content_key, assets.hf_public_assets, assets.lf_public_assets)
    return {
        "lf": score.lf,
        "hf": score.hf,
        "weighted_joint": weighted_joint_score(score.lf, score.hf, calibration),
        # A single clean R0 unit has neither the frozen formal roster nor the
        # ordered wrong/null evidence needed for the existing content judgement.
        "original_decision": "NOT_ADJUDICATED",
    }


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("R0 output is create-only and already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    path.write_bytes(encoded + b"\n")
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(encoded + b"\n").hexdigest(), "method_id": GEOMETRY_V6_METHOD_ID}, sort_keys=True))


def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    exact = _exact(repo_root, args.expected_exact)
    content_key = os.environ.get(CONTENT_KEY_ENV)
    geometry_key = os.environ.get(GEOMETRY_KEY_ENV)
    wrong_geometry_key = os.environ.get(WRONG_GEOMETRY_KEY_ENV)
    token = os.environ.get("HF_TOKEN")
    if not all(isinstance(value, str) and value.strip() for value in (content_key, geometry_key, wrong_geometry_key, token)):
        raise RuntimeError("R0 diagnostic requires separate content, geometry, wrong-geometry, and HF credentials")
    if geometry_key == content_key or wrong_geometry_key in {geometry_key, content_key}:
        raise ValueError("Geometry-V6 requires distinct content, geometry, and wrong-geometry key roots")
    if args.amplitude not in R0_AMPLITUDE_CANDIDATES:
        raise ValueError("amplitude must be one predeclared global R0 candidate")
    pipeline, assets = _load_assets(token)
    calibration = load_calibration_asset(
        repo_root / "configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json",
        repo_root / "configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json.sha256",
    )
    images: dict[str, Any] = {}
    for arm in ARMS:
        # Resetting the generator gives every physical arm the same frozen seed.
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        output = run_sd35_geometry_v6_r0_arm(
            pipeline, args.prompt, arm, content_key=content_key if "content" in arm else None,
            geometry_key=geometry_key if "geometry" in arm else None,
            amplitude=args.amplitude if "geometry" in arm else None,
            content_assets=assets if "content" in arm else None,
            height=args.height, width=args.width, generator=generator,
        )
        images[arm] = output.image
    records: dict[str, Any] = {}
    for arm, image in images.items():
        # Each blind invocation is deliberately separate; truth/arm comparison
        # happens only after these frozen observations have been collected.
        records[arm] = {
            "content": _record_content(image, content_key, assets, calibration),
            "geometry_correct": blind_geometry_observation(image, geometry_key, assets.hf_public_assets.image_processor, assets.hf_public_assets.vae).score,
            "geometry_wrong": blind_geometry_observation(image, wrong_geometry_key, assets.hf_public_assets.image_processor, assets.hf_public_assets.vae).score,
            "geometry_no_key": blind_geometry_observation(image, None, assets.hf_public_assets.image_processor, assets.hf_public_assets.vae).score,
        }
    keys = derive_geometry_keys(geometry_key)
    return {
        "method_id": GEOMETRY_V6_METHOD_ID,
        "stage": "R0_clean_identity_only",
        "evidence_ceiling": "user_run_nonformal_colab_diagnostic; science_denominator=0",
        "exact": exact,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "seed": args.seed,
        "amplitude": args.amplitude,
        "geometry_key_public_digest": public_key_digest(geometry_key),
        "geometry_subkey_digests": {
            "k_search": hashlib.sha256(keys.search).hexdigest(),
            "k_fit": hashlib.sha256(keys.fit).hexdigest(),
            "k_validate": hashlib.sha256(keys.validate).hexdigest(),
        },
        "arms": records,
        "content_geometry_compatibility": "NOT_ADJUDICATED_NO_FROZEN_CONTENT_ROSTER",
        "carrier_window": "NOT_ADJUDICATED_NO_AUTHORIZED_GEOMETRY_OR_QUALITY_THRESHOLD",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validate-static", action="store_true")
    parser.add_argument("--run-diagnostic", action="store_true")
    parser.add_argument("--expected-exact")
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--amplitude", type=float)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    if args.validate_static:
        if args.run_diagnostic:
            raise ValueError("static validation cannot run a diagnostic")
        print(json.dumps({"method_id": GEOMETRY_V6_METHOD_ID, "amplitudes": R0_AMPLITUDE_CANDIDATES, "status": "STATIC_VALIDATED", "science_denominator": 0}, sort_keys=True))
        return 0
    if not args.run_diagnostic or not all((args.expected_exact, args.prompt, args.seed is not None, args.amplitude is not None, args.output_json)):
        raise ValueError("R0 diagnostic requires exact, prompt, seed, amplitude, and create-only output path")
    _write_create_only(args.output_json, _run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
