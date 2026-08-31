"""Geometry-V6 R0 clean-identity diagnostic; never a science adjudication."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import traceback
from pathlib import Path
from typing import Any

import torch

from cegwm.method.content_weighted_joint import (
    LFHFScorePair,
    load_calibration_asset,
    weighted_gate_evidence,
    weighted_joint_score,
)
from cegwm.method.content_whitening import (
    FrozenContentWhiteningLFPublicAssets,
    load_frozen_content_whitening_asset,
    score_content_whitened_lf_image,
)
from cegwm.method.hf import score_hf_image
from cegwm.method.geometry_v6_roundtrip import (
    GEOMETRY_V6_METHOD_ID,
    PUBLIC_PILOT_SPEC_ID,
    R0_AMPLITUDE_CANDIDATES,
    blind_geometry_observation,
)
from cegwm.runtime.content_adaptive_sd35 import ContentEmbedAssets, load_dino_content_assets
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline
from cegwm.runtime.geometry_v6_sd35 import run_sd35_geometry_v6_r0_arm
from cegwm.runtime.content_weighted_joint_sd35 import derive_stability_wrong_keys

CONTENT_KEY_ENV = "CEG_WM_ROOT_KEY"
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
ARMS = ("content_only", "content_geometry", "geometry_only", "unwatermarked")
FAILURE_STAGE = "run_sd35_geometry_v6_r0_arm"
FAILURE_MESSAGE_LIMIT = 512
FAILURE_TRACEBACK_TAIL_LIMIT = 4096
UNAVAILABLE = "UNAVAILABLE"
_RUNTIME_ENVIRONMENT_FIELDS = frozenset({
    "torch_version", "torch_cuda_runtime_version", "cuda_device_name",
    "diffusers_version", "transformers_version", "model_id", "pipeline_class",
    "vae_class", "vae_parameter_dtype", "vae_scaling_factor", "vae_shift_factor",
})


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


def _sanitize_diagnostic(value: str, secrets: tuple[str, ...]) -> str:
    """Remove the exact sensitive values used by this R0 diagnostic."""

    sanitized = value
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(secret, "[REDACTED]")
    return sanitized


def _failure_diagnostic(error: Exception, content_key: str, token: str, prompt: str) -> dict[str, str]:
    """Return a finite, JSON-safe description of the exception from this arm."""

    secrets = (content_key, token, prompt)
    message = _sanitize_diagnostic(str(error), secrets)[:FAILURE_MESSAGE_LIMIT]
    rendered_traceback = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    traceback_tail = _sanitize_diagnostic(rendered_traceback, secrets)[-FAILURE_TRACEBACK_TAIL_LIMIT:]
    return {
        "failure_class": type(error).__name__,
        "failure_stage": FAILURE_STAGE,
        "sanitized_message": message,
        "sanitized_traceback_tail": traceback_tail,
    }


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return UNAVAILABLE


def _vae_parameter_dtype(vae: Any) -> str:
    try:
        return str(next(vae.parameters()).dtype)
    except (AttributeError, StopIteration, TypeError):
        return UNAVAILABLE


def _vae_config_value(vae: Any, name: str) -> float | str:
    value = getattr(getattr(vae, "config", None), name, None)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else UNAVAILABLE


def _cuda_device_name() -> str:
    try:
        return str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else UNAVAILABLE
    except (AssertionError, RuntimeError):
        return UNAVAILABLE


def _runtime_environment(pipeline: Any) -> dict[str, Any]:
    """Expose only the public runtime facts needed to diagnose an R0 failure."""

    vae = getattr(pipeline, "vae", None)
    environment = {
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime_version": str(torch.version.cuda) if torch.version.cuda else UNAVAILABLE,
        "cuda_device_name": _cuda_device_name(),
        "diffusers_version": _package_version("diffusers"),
        "transformers_version": _package_version("transformers"),
        "model_id": MODEL_ID,
        "pipeline_class": type(pipeline).__name__,
        "vae_class": type(vae).__name__ if vae is not None else UNAVAILABLE,
        "vae_parameter_dtype": _vae_parameter_dtype(vae),
        "vae_scaling_factor": _vae_config_value(vae, "scaling_factor"),
        "vae_shift_factor": _vae_config_value(vae, "shift_factor"),
    }
    if set(environment) != _RUNTIME_ENVIRONMENT_FIELDS:
        raise AssertionError("runtime environment fields changed without authorization")
    return environment


def _content_raw(
    image: Any,
    content_key: str,
    assets: ContentEmbedAssets,
    whitening_lf_assets: FrozenContentWhiteningLFPublicAssets,
    calibration: Any,
) -> dict[str, dict[str, float]]:
    """Use the unchanged blind content scorer for registered plus 16 frozen wrong keys."""

    keys = (content_key, *derive_stability_wrong_keys(content_key))
    labels = ("registered", *(f"wrong_{index:02d}" for index in range(16)))
    records: dict[str, dict[str, float]] = {}
    for label, key in zip(labels, keys, strict=True):
        lf = float(score_content_whitened_lf_image(image, key, whitening_lf_assets))
        hf = float(score_hf_image(image, key, assets.hf_public_assets))
        records[label] = {
            "lf": lf,
            "hf": hf,
            "weighted_joint": weighted_joint_score(lf, hf, calibration),
        }
    return records


def _per_unit_content_evidence(candidate: dict[str, dict[str, float]], primary_null: dict[str, dict[str, float]], calibration: Any) -> dict[str, Any]:
    """Apply the existing 16-wrong-key weighted-gate primitive without new tau."""

    registered = candidate["registered"]
    wrong = tuple(candidate[f"wrong_{index:02d}"] for index in range(16))
    evidence = weighted_gate_evidence(
        LFHFScorePair(registered["lf"], registered["hf"]),
        tuple(LFHFScorePair(item["lf"], item["hf"]) for item in wrong),
        LFHFScorePair(primary_null["registered"]["lf"], primary_null["registered"]["hf"]),
        calibration,
    )
    return {
        "per_unit_frozen_content_evidence": {
            "weighted_gate_a": evidence.weighted_gate_a,
            "weighted_gate_b": evidence.weighted_gate_b,
            "lf_gate_a_diagnostic": evidence.lf_gate_a_diagnostic,
            "lf_gate_b_diagnostic": evidence.lf_gate_b_diagnostic,
            "hf_gate_a_diagnostic": evidence.hf_gate_a_diagnostic,
            "hf_gate_b_diagnostic": evidence.hf_gate_b_diagnostic,
        },
        "per_unit_frozen_content_positive": evidence.weighted_gate_a and evidence.weighted_gate_b,
        "science_adjudication": "NOT_ADJUDICATED",
    }


def _geometry_raw(image: Any, assets: ContentEmbedAssets) -> dict[str, float | int]:
    processor = assets.hf_public_assets.image_processor
    vae = assets.hf_public_assets.vae
    observation = blind_geometry_observation(image, processor, vae)
    return {name: getattr(observation, name) for name in observation.__dataclass_fields__}


def _pilot_present_vs_absent(present: dict[str, Any], absent: dict[str, Any]) -> dict[str, Any]:
    if present.get("status") != "success" or absent.get("status") != "success":
        return {"status": "NOT_EVALUABLE_OPERATIONAL_FAILURE"}
    left, right = present["public_pilot_observation"], absent["public_pilot_observation"]
    return {
        "status": "RAW_OBSERVABILITY_ONLY",
        "aggregate_delta": left["aggregate_score"] - right["aggregate_score"],
        "search_delta": left["search_score"] - right["search_score"],
        "fit_delta": left["fit_score"] - right["fit_score"],
        "validate_delta": left["validate_score"] - right["validate_score"],
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
    token = os.environ.get("HF_TOKEN")
    if not all(isinstance(value, str) and value.strip() for value in (content_key, token)):
        raise RuntimeError("R0 diagnostic requires content and HF credentials")
    pipeline, assets = _load_assets(token)
    runtime_environment = _runtime_environment(pipeline)
    calibration = load_calibration_asset(
        repo_root / "configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json",
        repo_root / "configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json.sha256",
    )
    whitening_lf_assets = FrozenContentWhiteningLFPublicAssets(
        assets.lf_public_assets, load_frozen_content_whitening_asset(repo_root)
    )
    def generate(arm: str, amplitude: float | None) -> tuple[Any | None, dict[str, str] | None]:
        # Every physical arm gets an independent generator reset to the same seed.
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        try:
            output = run_sd35_geometry_v6_r0_arm(
                pipeline, args.prompt, arm, content_key=content_key if "content" in arm else None,
                amplitude=amplitude if "geometry" in arm else None,
                content_assets=assets if "content" in arm else None,
                height=args.height, width=args.width, generator=generator,
            )
            return output.image, None
        except Exception as error:  # Record, do not retry or silently replace a failed physical arm.
            return None, _failure_diagnostic(error, content_key, token, args.prompt)

    def record_arm(image: Any | None, failure: dict[str, str] | None) -> dict[str, Any]:
        if failure is not None:
            return {"status": "operational_failure", "failure_reason": failure}
        if image is None:
            raise RuntimeError("successful R0 arm omitted its image")
        return {
            "status": "success",
            "content_raw": _content_raw(image, content_key, assets, whitening_lf_assets, calibration),
            "public_pilot_observation": _geometry_raw(image, assets),
        }

    content_only_image, content_only_failure = generate("content_only", None)
    unwatermarked_image, unwatermarked_failure = generate("unwatermarked", None)
    content_only_record = record_arm(content_only_image, content_only_failure)
    unwatermarked_record = record_arm(unwatermarked_image, unwatermarked_failure)
    if content_only_record["status"] == unwatermarked_record["status"] == "success":
        content_only_record["content_evidence"] = _per_unit_content_evidence(
            content_only_record["content_raw"], unwatermarked_record["content_raw"], calibration
        )
    else:
        content_only_record["content_evidence"] = {"science_adjudication": "NOT_ADJUDICATED", "per_unit_frozen_content_evidence": "NOT_EVALUABLE_OPERATIONAL_FAILURE"}
    amplitudes: list[dict[str, Any]] = []
    for amplitude in R0_AMPLITUDE_CANDIDATES:
        combined_image, combined_failure = generate("content_geometry", amplitude)
        geometry_only_image, geometry_only_failure = generate("geometry_only", amplitude)
        combined_record = record_arm(combined_image, combined_failure)
        geometry_only_record = record_arm(geometry_only_image, geometry_only_failure)
        if combined_record["status"] == geometry_only_record["status"] == "success":
            combined_record["content_evidence"] = _per_unit_content_evidence(
                combined_record["content_raw"], geometry_only_record["content_raw"], calibration
            )
        else:
            combined_record["content_evidence"] = {"science_adjudication": "NOT_ADJUDICATED", "per_unit_frozen_content_evidence": "NOT_EVALUABLE_OPERATIONAL_FAILURE"}
        combined_positive = combined_record["content_evidence"].get("per_unit_frozen_content_positive")
        baseline_positive = content_only_record["content_evidence"].get("per_unit_frozen_content_positive")
        compatibility = "NOT_EVALUABLE_OPERATIONAL_FAILURE"
        if isinstance(baseline_positive, bool) and isinstance(combined_positive, bool):
            compatibility = "FAIL_CLOSED_CONTENT_POSITIVE_TO_NEGATIVE" if baseline_positive and not combined_positive else "NO_POSITIVE_TO_NEGATIVE_FLIP"
        amplitudes.append({
            "amplitude": amplitude,
            "content_geometry": combined_record,
            "geometry_only": geometry_only_record,
            "pilot_present_vs_absent": {
                "content_geometry_vs_content_only": _pilot_present_vs_absent(combined_record, content_only_record),
                "geometry_only_vs_unwatermarked": _pilot_present_vs_absent(geometry_only_record, unwatermarked_record),
            },
            "content_compatibility": compatibility,
        })
    return {
        "method_id": GEOMETRY_V6_METHOD_ID,
        "stage": "R0_clean_identity_only",
        "evidence_ceiling": "user_run_nonformal_colab_diagnostic; science_denominator=0",
        "exact": exact,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "seed": args.seed,
        "amplitude_sequence": R0_AMPLITUDE_CANDIDATES,
        "public_pilot_spec_id": PUBLIC_PILOT_SPEC_ID,
        "runtime_environment": runtime_environment,
        "baselines": {
            "content_only": content_only_record,
            "unwatermarked": unwatermarked_record,
        },
        "amplitudes": amplitudes,
        "carrier_window": "NOT_ADJUDICATED_NO_AUTHORIZED_GEOMETRY_OR_QUALITY_THRESHOLD",
        "conditional_flow_fpr": "NOT_ADJUDICATED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validate-static", action="store_true")
    parser.add_argument("--run-diagnostic", action="store_true")
    parser.add_argument("--expected-exact")
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    if args.validate_static:
        if args.run_diagnostic:
            raise ValueError("static validation cannot run a diagnostic")
        print(json.dumps({"method_id": GEOMETRY_V6_METHOD_ID, "amplitudes": R0_AMPLITUDE_CANDIDATES, "status": "STATIC_VALIDATED", "science_denominator": 0}, sort_keys=True))
        return 0
    if not args.run_diagnostic or not all((args.expected_exact, args.prompt, args.seed is not None, args.output_json)):
        raise ValueError("R0 diagnostic requires exact, prompt, seed, and create-only output path")
    _write_create_only(args.output_json, _run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
