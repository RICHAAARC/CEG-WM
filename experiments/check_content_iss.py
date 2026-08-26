"""One-unit, non-roster operational canary for the real content ISS runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import torch

from experiments import content_adaptive_engine as engine
from experiments.content_iss_engine import (
    _load_pipeline_and_assets,
    _run_pair,
)
from cegwm.method.content_unweighted import BRANCH_SHARE_SUM_ABSOLUTE_TOLERANCE
from cegwm.method.content_iss import (
    CONTENT_ISS_EVALUATED_CANDIDATE_ID,
    CONTENT_ISS_METHOD_ID,
    ISS_ASSET_SHA256,
    ISS_ASSET_SIDECAR_SHA256,
)
from cegwm.method.content_whitening import (
    CONTENT_WHITENING_LF_SCORER_ID,
    score_content_whitened_lf_image,
)
from cegwm.method.hf import score_hf_image
from cegwm.protocol.content_iss import (
    CONTENT_ISS_PROTOCOL_DIGEST,
    CONTENT_ISS_PROTOCOL_ID,
    CONTENT_ISS_RUN_PREFIX,
    _historical_units,
    load_content_iss_protocol,
    load_content_iss_data_contract,
)
from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key, public_key_digest

CANARY_ID = "content-v6-iss-detector-domain-operational-canary-v1"
UNIT_ID = "content-v6-iss-canary-0001"
SOURCE_ID = "content-v6-iss-canary-source-0001"
PROMPT = "A simple white ceramic cup on a wooden table in soft daylight"
SEED = 2026082600
HEIGHT = 512
WIDTH = 512
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
KEY_ENV = "CEG_WM_ROOT_KEY"
TOKEN_ENV = "HF_TOKEN"
PREFIX = "CEGWM_CONTENT_ISS_CHECK_RESULT"
CLAIM_CEILING = "full_non_roster_runtime_canary_only"
_FAILURE_STAGES = {
    "identity_validation",
    "runtime_asset_load",
    "paired_generation",
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
    if _run_git(repo_root, "status", "--porcelain"):
        raise RuntimeError("execution checkout must be clean")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda is required for the operational canary")


def _assert_non_roster_identity(repo_root: Path) -> None:
    protocol = load_content_iss_protocol(repo_root)
    if (
        protocol.protocol_id != CONTENT_ISS_PROTOCOL_ID
        or protocol.protocol_digest != CONTENT_ISS_PROTOCOL_DIGEST
    ):
        raise RuntimeError("canary protocol identity differs from content ISS")
    contract = load_content_iss_data_contract(repo_root)
    root = repo_root / "configs" / "content_chain"
    all_units = (*contract.development, *contract.evaluation, *_historical_units(root))
    for unit in all_units:
        if (
            unit.unit_id in {CANARY_ID, UNIT_ID, SOURCE_ID}
            or unit.source_id in {CANARY_ID, UNIT_ID, SOURCE_ID}
            or unit.prompt == PROMPT
            or unit.seed == SEED
        ):
            raise RuntimeError("canary identity overlaps frozen research data")


def _registered_scores(image: Any, detection_key: bytes, assets: Any) -> dict[str, float]:
    ordinary = require_ordinary_rgb_image(image)
    if ordinary.size != (WIDTH, HEIGHT):
        raise ValueError("blind score image dimensions differ from the canary")
    lf_score = _finite_scalar(
        score_content_whitened_lf_image(ordinary, detection_key, assets.lf_public_assets),
        "LF score",
    )
    hf_score = _finite_scalar(
        score_hf_image(ordinary, detection_key, assets.hf_public_assets),
        "HF score",
    )
    if not -1.0 <= lf_score <= 1.0 or not -1.0 <= hf_score <= 1.0:
        raise ValueError("blind scores must lie in the normalized interval")
    return {"lf": lf_score, "hf": hf_score, "joint": min(lf_score, hf_score)}


def _validated_metrics(output: Any) -> dict[str, float | int]:
    measurement = output.measurement
    metrics = engine._candidate_aggregate_metrics(
        UNIT_ID,
        measurement,
        engine._psnr(output.image, output.primary_null),
        share_sum_absolute_tolerance=BRANCH_SHARE_SUM_ABSOLUTE_TOLERANCE,
    )
    if not 0.0 < float(metrics["combined_relative_l2"]) <= 0.012:
        raise ValueError("combined actual-dtype budget is outside the canary bound")
    if float(metrics["lf_effective_relative_l2"]) <= 0.0:
        raise ValueError("LF effective budget must be positive")
    if float(metrics["hf_effective_relative_l2"]) <= 0.0:
        raise ValueError("HF effective budget must be positive")
    return {
        "combined_actual_dtype_relative_l2": float(metrics["combined_relative_l2"]),
        "lf_effective_relative_l2": float(metrics["lf_effective_relative_l2"]),
        "hf_effective_relative_l2": float(metrics["hf_effective_relative_l2"]),
        "lf_branch_share": float(metrics["lf_branch_share"]),
        "hf_branch_share": float(metrics["hf_branch_share"]),
        "minimum_counterfactual_effect": float(metrics["minimum_counterfactual_effect"]),
        "probe_evaluation_count": int(metrics["probe_evaluation_count"]),
        "paired_rgb_psnr_db": float(metrics["paired_rgb_psnr_db"]),
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
        _assert_non_roster_identity(repo_root)
        key_digest = public_key_digest(detection_key)
        run_id = f"{CONTENT_ISS_RUN_PREFIX}-{CONTENT_ISS_PROTOCOL_DIGEST[:12]}-{key_digest[:12]}"

        stage = "runtime_asset_load"
        pipeline, assets = _load_pipeline_and_assets(MODEL_ID, token)
        token = ""

        stage = "paired_generation"
        output = _run_pair(
            pipeline,
            PROMPT,
            detection_key,
            assets,
            height=HEIGHT,
            width=WIDTH,
            seed=SEED,
        )

        stage = "blind_scoring"
        joint_scores = _registered_scores(output.image, detection_key, assets)
        null_scores = _registered_scores(output.primary_null, detection_key, assets)

        stage = "budget_quality_validation"
        result = {
            "status": "operational_canary_pass",
            "claim_ceiling": CLAIM_CEILING,
            "canary_id": CANARY_ID,
            "unit_id": UNIT_ID,
            "source_id": SOURCE_ID,
            "exact": exact,
            "public_key_digest": key_digest,
            "run_id": run_id,
            "protocol_id": CONTENT_ISS_PROTOCOL_ID,
            "protocol_digest": CONTENT_ISS_PROTOCOL_DIGEST,
            "content_method_id": CONTENT_ISS_METHOD_ID,
            "evaluated_candidate_id": CONTENT_ISS_EVALUATED_CANDIDATE_ID,
            "model_id": MODEL_ID,
            "lf_scorer_id": CONTENT_WHITENING_LF_SCORER_ID,
            "iss_asset_sha256": ISS_ASSET_SHA256,
            "iss_asset_sidecar_sha256": ISS_ASSET_SIDECAR_SHA256,
            "seed": SEED,
            "height": HEIGHT,
            "width": WIDTH,
            **_validated_metrics(output),
            "joint_registered_lf_score": joint_scores["lf"],
            "joint_registered_hf_score": joint_scores["hf"],
            "joint_registered_joint_score": joint_scores["joint"],
            "primary_null_registered_lf_score": null_scores["lf"],
            "primary_null_registered_hf_score": null_scores["hf"],
            "primary_null_registered_joint_score": null_scores["joint"],
            "formal_roster_member": False,
            "scientific_denominator_units": 0,
        }
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
