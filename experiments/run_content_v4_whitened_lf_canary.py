"""One-unit, non-roster operational canary for the real Content V4 runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.run_content_v4_clean import (
    _load_pipeline_and_assets,
    _run_joint,
)
from cegwm.method.content_adaptive_v3 import (
    BRANCH_SHARE_SUM_ABSOLUTE_TOLERANCE,
    DINO_ASSET_ID,
)
from cegwm.method.content_whitening_v4 import (
    ASSET_ROLE_ID,
    ASSET_SCHEMA_ID,
    ASSET_SHA256,
    ASSET_SIDECAR_SHA256,
    CONTENT_V4_EVALUATED_CANDIDATE_ID,
    CONTENT_V4_LF_SCORER_ID,
    CONTENT_V4_METHOD_ID,
    FIT_MANIFEST_REPO_PATH,
    OBSERVATION_CONTRACT_ID,
    FrozenContentV4LFPublicAssets,
    load_fit_manifest,
    score_content_v4_lf_image,
)
from cegwm.method.hf import FrozenHFPublicAssets, score_hf_image
from cegwm.protocol.content_chain_v4 import (
    CONTENT_V4_PROTOCOL_DIGEST,
    CONTENT_V4_PROTOCOL_ID,
    CONTENT_V4_RUN_PREFIX,
    ContentChainProtocol,
    load_content_v4_clean_protocol,
)
from cegwm.runtime.diffusers_sd35 import run_sd35_plain
from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key, public_key_digest

CANARY_ID = "content-v4-whitened-lf-full-runtime-non-roster-canary-v1"
UNIT_ID = "content-v4-whitened-lf-canary-0001"
SOURCE_ID = "content-v4-whitened-lf-canary-prompt-9001"
PROMPT = "A book conservator examining an illuminated manuscript under neutral studio light"
SEED = 1415149
HEIGHT = 512
WIDTH = 512
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
DINO_ID = DINO_ASSET_ID
BRANCH = "stage-a-content-adaptive-dual-branch-v4-whitened-lf-canary"
KEY_ENV = "CEG_WM_ROOT_KEY"
TOKEN_ENV = "HF_TOKEN"
PREFIX = "CEGWM_CONTENT_V4_CANARY_RESULT"
CLAIM_CEILING = "full_non_roster_runtime_canary_only"
_FAILURE_STAGES = {
    "identity_validation",
    "runtime_asset_load",
    "joint_generation",
    "primary_null_generation",
    "blind_scoring",
    "budget_quality_validation",
}
_ERROR_CLASSES = {
    "FileNotFoundError",
    "ImportError",
    "MemoryError",
    "ModuleNotFoundError",
    "OSError",
    "OutOfMemoryError",
    "RuntimeError",
    "TimeoutError",
    "TypeError",
    "ValueError",
}
_SUCCESS_FIELDS = (
    "status",
    "claim_ceiling",
    "canary_id",
    "unit_id",
    "source_id",
    "exact",
    "public_key_digest",
    "run_id",
    "protocol_id",
    "protocol_digest",
    "content_method_id",
    "evaluated_candidate_id",
    "model_id",
    "dino_asset_id",
    "lf_scorer_id",
    "observation_contract_id",
    "whitening_asset_role_id",
    "whitening_asset_schema_id",
    "whitening_asset_sha256",
    "whitening_asset_sidecar_sha256",
    "seed",
    "height",
    "width",
    "combined_actual_dtype_relative_l2",
    "lf_effective_relative_l2",
    "hf_effective_relative_l2",
    "lf_branch_share",
    "hf_branch_share",
    "minimum_counterfactual_effect",
    "probe_evaluation_count",
    "paired_rgb_psnr_db",
    "joint_registered_lf_score",
    "joint_registered_hf_score",
    "joint_registered_joint_score",
    "primary_null_registered_lf_score",
    "primary_null_registered_hf_score",
    "primary_null_registered_joint_score",
    "formal_roster_member",
    "scientific_denominator_units",
)


def _finite_scalar(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _run_git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _resolve_exact(repo_root: Path) -> str:
    if not repo_root.is_dir():
        raise ValueError("repo root must be an existing directory")
    exact = _run_git(repo_root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", exact) is None:
        raise RuntimeError("resolved revision is not an exact commit")
    return exact


def _validate_execution_identity(repo_root: Path, expected_exact: str, exact: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected_exact) is None:
        raise ValueError("expected exact must be a lowercase 40-character revision")
    if _run_git(repo_root, "rev-parse", "--show-toplevel") != str(repo_root):
        raise RuntimeError("repo root must be the checkout root")
    if exact != expected_exact:
        raise RuntimeError("resolved revision differs from expected exact")
    if _run_git(repo_root, "branch", "--show-current") != BRANCH:
        raise RuntimeError("execution branch differs from the canary branch")
    if _run_git(repo_root, "status", "--porcelain"):
        raise RuntimeError("execution checkout must be clean")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda is required for the operational canary")


def _assert_non_roster_identity(repo_root: Path) -> ContentChainProtocol:
    config_root = repo_root / "configs" / "content_chain"
    protocol = load_content_v4_clean_protocol(
        config_root / "content_v4_clean_v1.json",
        config_root / "content_adaptive_dual_branch_v2_clean.jsonl",
    )
    if (
        protocol.protocol_id != CONTENT_V4_PROTOCOL_ID
        or protocol.protocol_digest != CONTENT_V4_PROTOCOL_DIGEST
    ):
        raise RuntimeError("canary protocol identity differs from Content V4")
    for unit in protocol.roster:
        if (
            unit.unit_id == UNIT_ID
            or unit.source_id == SOURCE_ID
            or unit.prompt == PROMPT
            or unit.seed == SEED
        ):
            raise RuntimeError("canary identity overlaps the frozen scientific roster")
    fit_manifest = load_fit_manifest(repo_root / FIT_MANIFEST_REPO_PATH)
    for entry in fit_manifest.entries:
        if (
            entry.unit_id in {UNIT_ID, SOURCE_ID}
            or entry.prompt == PROMPT
            or entry.generation_seed == SEED
        ):
            raise RuntimeError("canary identity overlaps the frozen whitening fit manifest")
    return protocol


def _validate_runtime_assets(assets: Any) -> None:
    hf_assets = getattr(assets, "hf_public_assets", None)
    lf_assets = getattr(assets, "lf_public_assets", None)
    if not isinstance(hf_assets, FrozenHFPublicAssets):
        raise TypeError("canary requires frozen HF public assets")
    if not isinstance(lf_assets, FrozenContentV4LFPublicAssets):
        raise TypeError("canary requires frozen Content V4 LF public assets")
    carrier = lf_assets.carrier_assets
    if hf_assets.vae is not carrier.vae or hf_assets.image_processor is not carrier.image_processor:
        raise ValueError("canary HF and LF scores must share public observation assets")
    if (
        lf_assets.whitening_asset.payload["schema_version"] != ASSET_SCHEMA_ID
        or lf_assets.whitening_asset.payload["observation_contract_id"]
        != OBSERVATION_CONTRACT_ID
    ):
        raise ValueError("canary whitening asset identity differs")


def _registered_scores(
    image: Any,
    detection_key: bytes,
    hf_assets: FrozenHFPublicAssets,
    lf_assets: FrozenContentV4LFPublicAssets,
) -> dict[str, float]:
    ordinary = require_ordinary_rgb_image(image)
    if ordinary.size != (WIDTH, HEIGHT):
        raise ValueError("blind score image dimensions differ from the canary")
    if not isinstance(hf_assets, FrozenHFPublicAssets):
        raise TypeError("blind HF scoring requires frozen public assets")
    if not isinstance(lf_assets, FrozenContentV4LFPublicAssets):
        raise TypeError("blind LF scoring requires frozen Content V4 public assets")
    lf_score = _finite_scalar(
        score_content_v4_lf_image(ordinary, detection_key, lf_assets), "LF score"
    )
    hf_score = _finite_scalar(score_hf_image(ordinary, detection_key, hf_assets), "HF score")
    if not -1.0 <= lf_score <= 1.0 or not -1.0 <= hf_score <= 1.0:
        raise ValueError("blind scores must lie in the normalized interval")
    return {"lf": lf_score, "hf": hf_score, "joint": min(lf_score, hf_score)}


def _paired_psnr(first: Any, second: Any) -> float:
    first_rgb = require_ordinary_rgb_image(first)
    second_rgb = require_ordinary_rgb_image(second)
    if first_rgb.size != (WIDTH, HEIGHT) or second_rgb.size != (WIDTH, HEIGHT):
        raise ValueError("paired image dimensions differ from the canary")
    first_pixels = np.asarray(first_rgb, dtype=np.float64) / 255.0
    second_pixels = np.asarray(second_rgb, dtype=np.float64) / 255.0
    mse = float(np.mean(np.square(first_pixels - second_pixels)))
    if not math.isfinite(mse) or not 0.0 < mse <= 1.0:
        raise ValueError("paired images require a finite nonzero RGB error")
    psnr = -10.0 * math.log10(mse)
    if not math.isfinite(psnr) or psnr < 0.0:
        raise ValueError("paired RGB PSNR is invalid")
    return psnr


def _validated_metrics(measurement: Any, psnr: float) -> dict[str, float | int]:
    combined = _finite_scalar(measurement.combined_budget.relative_l2, "combined budget")
    lf_effective = _finite_scalar(measurement.lf_effective_relative_l2, "LF effective budget")
    hf_effective = _finite_scalar(measurement.hf_effective_relative_l2, "HF effective budget")
    lf_share = _finite_scalar(measurement.lf_branch_share, "LF branch share")
    hf_share = _finite_scalar(measurement.hf_branch_share, "HF branch share")
    minimum_effect = _finite_scalar(
        measurement.minimum_counterfactual_effect, "minimum counterfactual effect"
    )
    if not 0.0 < combined <= 0.012:
        raise ValueError("combined actual-dtype budget is outside the canary bound")
    if lf_effective <= 0.0 or hf_effective <= 0.0:
        raise ValueError("both effective branch budgets must be positive")
    if not 0.0 < lf_share < 1.0 or not 0.0 < hf_share < 1.0:
        raise ValueError("both branch shares must be strictly internal")
    if not math.isclose(
        lf_share + hf_share,
        1.0,
        rel_tol=0.0,
        abs_tol=BRANCH_SHARE_SUM_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError("branch shares do not sum to one within the frozen tolerance")
    if minimum_effect < 0.0:
        raise ValueError("minimum counterfactual effect must be nonnegative")
    probe_count = measurement.probe_evaluation_count
    if not isinstance(probe_count, int) or isinstance(probe_count, bool) or probe_count != 64:
        raise ValueError("probe evaluation count must be exactly 64")
    psnr = _finite_scalar(psnr, "paired RGB PSNR")
    if psnr < 0.0:
        raise ValueError("paired RGB PSNR must be nonnegative")
    return {
        "combined_actual_dtype_relative_l2": combined,
        "lf_effective_relative_l2": lf_effective,
        "hf_effective_relative_l2": hf_effective,
        "lf_branch_share": lf_share,
        "hf_branch_share": hf_share,
        "minimum_counterfactual_effect": minimum_effect,
        "probe_evaluation_count": probe_count,
        "paired_rgb_psnr_db": psnr,
    }


def _public_error_class(error: Exception) -> str:
    name = type(error).__name__
    return name if name in _ERROR_CLASSES else "OtherOperationalError"


def _emit(payload: Mapping[str, Any]) -> None:
    line = f"{PREFIX} {json.dumps(dict(payload), sort_keys=True, separators=(',', ':'))}"
    if "\n" in line or "\r" in line or len(line) > 4096:
        raise RuntimeError("canary output escaped the bounded line contract")
    print(line, flush=True)


def execute(args: argparse.Namespace) -> int:
    stage = "identity_validation"
    exact: str | None = None
    raw_key = os.environ.pop(KEY_ENV, "")
    token = os.environ.pop(TOKEN_ENV, "")
    detection_key = b""
    try:
        if not raw_key or not token:
            raise RuntimeError("required canary environment is absent")
        detection_key = normalize_detection_key(raw_key)
        raw_key = ""
        repo_root = Path(args.repo_root).resolve()
        exact = _resolve_exact(repo_root)
        _validate_execution_identity(repo_root, args.expected_exact, exact)
        protocol = _assert_non_roster_identity(repo_root)
        key_digest = public_key_digest(detection_key)
        run_id = f"{CONTENT_V4_RUN_PREFIX}-{protocol.protocol_digest[:12]}-{key_digest[:12]}"

        stage = "runtime_asset_load"
        pipeline, assets = _load_pipeline_and_assets(MODEL_ID, token)
        _validate_runtime_assets(assets)
        token = ""

        stage = "joint_generation"
        joint = _run_joint(
            pipeline,
            PROMPT,
            detection_key,
            assets,
            height=HEIGHT,
            width=WIDTH,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
        )

        stage = "primary_null_generation"
        primary_null = run_sd35_plain(
            pipeline,
            PROMPT,
            height=HEIGHT,
            width=WIDTH,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
        )

        stage = "blind_scoring"
        joint_scores = _registered_scores(
            joint.image, detection_key, assets.hf_public_assets, assets.lf_public_assets
        )
        null_scores = _registered_scores(
            primary_null, detection_key, assets.hf_public_assets, assets.lf_public_assets
        )

        stage = "budget_quality_validation"
        metrics = _validated_metrics(joint.measurement, _paired_psnr(joint.image, primary_null))
        result = {
            "status": "operational_canary_pass",
            "claim_ceiling": CLAIM_CEILING,
            "canary_id": CANARY_ID,
            "unit_id": UNIT_ID,
            "source_id": SOURCE_ID,
            "exact": exact,
            "public_key_digest": key_digest,
            "run_id": run_id,
            "protocol_id": protocol.protocol_id,
            "protocol_digest": protocol.protocol_digest,
            "content_method_id": CONTENT_V4_METHOD_ID,
            "evaluated_candidate_id": CONTENT_V4_EVALUATED_CANDIDATE_ID,
            "model_id": MODEL_ID,
            "dino_asset_id": DINO_ID,
            "lf_scorer_id": CONTENT_V4_LF_SCORER_ID,
            "observation_contract_id": OBSERVATION_CONTRACT_ID,
            "whitening_asset_role_id": ASSET_ROLE_ID,
            "whitening_asset_schema_id": ASSET_SCHEMA_ID,
            "whitening_asset_sha256": ASSET_SHA256,
            "whitening_asset_sidecar_sha256": ASSET_SIDECAR_SHA256,
            "seed": SEED,
            "height": HEIGHT,
            "width": WIDTH,
            **metrics,
            "joint_registered_lf_score": joint_scores["lf"],
            "joint_registered_hf_score": joint_scores["hf"],
            "joint_registered_joint_score": joint_scores["joint"],
            "primary_null_registered_lf_score": null_scores["lf"],
            "primary_null_registered_hf_score": null_scores["hf"],
            "primary_null_registered_joint_score": null_scores["joint"],
            "formal_roster_member": False,
            "scientific_denominator_units": 0,
        }
        if tuple(result) != _SUCCESS_FIELDS:
            raise RuntimeError("success result fields differ from the canary contract")
        detection_key = b""
        _emit(result)
        return 0
    except Exception as error:  # noqa: BLE001 - public failure is deliberately sanitized
        failure: dict[str, Any] = {"status": "operational_failure", "canary_id": CANARY_ID}
        if exact is not None:
            failure["exact"] = exact
        failure["stage"] = stage if stage in _FAILURE_STAGES else "identity_validation"
        failure["error_class"] = _public_error_class(error)
        _emit(failure)
        return 1
    finally:
        raw_key = ""
        token = ""
        detection_key = b""


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return execute(_arguments(argv))


if __name__ == "__main__":
    raise SystemExit(main())
