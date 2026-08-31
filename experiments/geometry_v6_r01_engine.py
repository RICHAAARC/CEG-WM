"""Geometry-V6 R0.1 fixed four-unit carrier-window candidate diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from experiments import geometry_v6_r0_engine as r0
from cegwm.method.geometry_v6_roundtrip import R0_AMPLITUDE_CANDIDATES
from cegwm.runtime.geometry_v6_sd35 import run_sd35_geometry_v6_r0_arm

R01_METHOD_ID = "geometry_v6_public_fixed_unkeyed_roundtrip_r01_carrier_candidate_v1"
ROSTER_MANIFEST = "configs/content_chain/content_chain_novel_seed_stability.jsonl"
ROSTER_MANIFEST_SHA256 = "33613cb24de87c86a573ac0dda80523912e001c922494051f5d89a9e2851831b"
ROSTER_MODEL_CONFIG = "configs/content_chain/content_v6_iss_clean_v1.json"
ROSTER_SIZE = 4
QUALITY_PSNR_MIN = 40.0
QUALITY_SSIM_MIN = 0.98
_DELTA_FIELDS = ("search_score", "fit_score", "validate_score", "aggregate_score")
_EXPECTED_ROSTER = (
    ("content-chain-seed-01-0001", "A cartographer inking river contours beside a brass compass", 2026101000),
    ("content-chain-seed-01-0002", "A falconer repairing a leather glove in a stone courtyard", 2026101001),
    ("content-chain-seed-01-0003", "A mycologist arranging forest specimens on a slate table", 2026101002),
    ("content-chain-seed-01-0004", "A piano technician voicing felt hammers under a workshop lamp", 2026101003),
)


def _load_roster(repo_root: Path) -> tuple[dict[str, Any], ...]:
    manifest = repo_root / ROSTER_MANIFEST
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != ROSTER_MANIFEST_SHA256:
        raise RuntimeError("R0.1 roster manifest identity differs")
    model = json.loads((repo_root / ROSTER_MODEL_CONFIG).read_text(encoding="utf-8"))
    if model.get("generation_runtime", {}).get("model_id") != r0.MODEL_ID:
        raise RuntimeError("R0.1 roster is not bound to the R0 generation model")
    rows = tuple(json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines())[:ROSTER_SIZE]
    received = tuple((row.get("unit_id"), row.get("prompt"), row.get("seed")) for row in rows)
    if len(rows) != ROSTER_SIZE or received != _EXPECTED_ROSTER:
        raise RuntimeError("R0.1 ordered holdout roster differs")
    if any(row.get("height") != 512 or row.get("width") != 512 for row in rows):
        raise RuntimeError("R0.1 roster dimensions differ")
    return rows


def _rgb_array(image: Any) -> np.ndarray:
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise TypeError("R0.1 quality requires ordinary RGB images")
    value = np.asarray(image, dtype=np.float64) / 255.0
    if value.ndim != 3 or value.shape[2] != 3 or not np.isfinite(value).all():
        raise ValueError("R0.1 quality requires finite RGB arrays")
    return value


def _v4_rgb_quality(first: Any, second: Any) -> dict[str, float]:
    """Reuse Geometry-V4's fixed RGB PSNR/SSIM scalar semantics exactly."""

    clean, marked = _rgb_array(first), _rgb_array(second)
    if clean.shape != marked.shape:
        raise ValueError("R0.1 matched RGB shape differs")
    mse = float(np.mean((marked - clean) ** 2))
    psnr = 300.0 if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    mx, my = float(clean.mean()), float(marked.mean())
    vx, vy = float(clean.var()), float(marked.var())
    covariance = float(((clean - mx) * (marked - my)).mean())
    ssim = ((2 * mx * my + 1e-4) * (2 * covariance + 9e-4)) / ((mx * mx + my * my + 1e-4) * (vx + vy + 9e-4))
    if not math.isfinite(psnr) or not math.isfinite(ssim):
        raise ValueError("R0.1 RGB quality is non-finite")
    return {"psnr_db": psnr, "ssim": ssim}


def _matched_carrier(present: dict[str, Any], absent: dict[str, Any]) -> dict[str, Any]:
    if present.get("status") != "success" or absent.get("status") != "success":
        return {"status": "FAIL_CLOSED_OPERATIONAL_FAILURE", "raw_deltas": {}}
    left, right = present.get("public_pilot_observation"), absent.get("public_pilot_observation")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return {"status": "FAIL_CLOSED_MISSING_OBSERVATION", "raw_deltas": {}}
    try:
        deltas = {field: float(left[field]) - float(right[field]) for field in _DELTA_FIELDS}
    except (KeyError, TypeError, ValueError):
        return {"status": "FAIL_CLOSED_MISSING_OBSERVATION", "raw_deltas": {}}
    passed = all(math.isfinite(value) and value > 0.0 for value in deltas.values())
    return {"status": "PASS" if passed else "FAIL_CLOSED_NONPOSITIVE_OR_NONFINITE_DELTA", "raw_deltas": deltas}


def _matched_quality(present: Any | None, absent: Any | None) -> dict[str, Any]:
    try:
        metrics = _v4_rgb_quality(absent, present)
    except (TypeError, ValueError):
        return {"status": "FAIL_CLOSED_QUALITY_UNAVAILABLE", "metrics": {}}
    passed = metrics["psnr_db"] > QUALITY_PSNR_MIN and metrics["ssim"] > QUALITY_SSIM_MIN
    return {"status": "PASS" if passed else "FAIL_CLOSED_QUALITY_THRESHOLD", "metrics": metrics}


def _unit_amplitude(
    content_only: dict[str, Any], unwatermarked: dict[str, Any], combined: dict[str, Any], geometry_only: dict[str, Any],
    content_only_image: Any | None, unwatermarked_image: Any | None, combined_image: Any | None, geometry_only_image: Any | None,
) -> dict[str, Any]:
    content = combined.get("content_evidence", {})
    content_pass = content_only.get("content_evidence", {}).get("per_unit_frozen_content_positive") is True and content.get("per_unit_frozen_content_positive") is True
    comparisons = {
        "content_geometry_vs_content_only": {"carrier": _matched_carrier(combined, content_only), "quality": _matched_quality(combined_image, content_only_image)},
        "geometry_only_vs_unwatermarked": {"carrier": _matched_carrier(geometry_only, unwatermarked), "quality": _matched_quality(geometry_only_image, unwatermarked_image)},
    }
    pairs_pass = all(item["carrier"]["status"] == "PASS" and item["quality"]["status"] == "PASS" for item in comparisons.values())
    return {"status": "PASS" if content_pass and pairs_pass else "FAIL_CLOSED", "content_compatibility_pass": content_pass, "matched_pairs": comparisons}


def _generate(pipeline: Any, assets: Any, content_key: str, prompt: str, seed: int, arm: str, amplitude: float | None) -> tuple[Any | None, dict[str, str] | None]:
    try:
        r0._reset_pipeline_offload_state(pipeline)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = run_sd35_geometry_v6_r0_arm(pipeline, prompt, arm, content_key=content_key if "content" in arm else None, amplitude=amplitude if "geometry" in arm else None, content_assets=assets if "content" in arm else None, height=512, width=512, generator=generator)
        return result.image, None
    except Exception as error:
        return None, r0._failure_diagnostic(error, content_key, "", prompt)


def _record(image: Any | None, failure: dict[str, str] | None, content_key: str, assets: Any, whitening: Any, calibration: Any) -> dict[str, Any]:
    if failure is not None:
        return {"status": "operational_failure", "failure_reason": failure}
    if image is None:
        raise RuntimeError("R0.1 successful arm omitted its image")
    return {"status": "success", "content_raw": r0._content_raw(image, content_key, assets, whitening, calibration), "public_pilot_observation": r0._geometry_raw(image, assets)}


def _run_unit(pipeline: Any, assets: Any, whitening: Any, calibration: Any, content_key: str, unit: dict[str, Any]) -> dict[str, Any]:
    prompt, seed = unit["prompt"], unit["seed"]
    content_image, content_failure = _generate(pipeline, assets, content_key, prompt, seed, "content_only", None)
    null_image, null_failure = _generate(pipeline, assets, content_key, prompt, seed, "unwatermarked", None)
    content = _record(content_image, content_failure, content_key, assets, whitening, calibration)
    null = _record(null_image, null_failure, content_key, assets, whitening, calibration)
    if content["status"] == null["status"] == "success":
        content["content_evidence"] = r0._per_unit_content_evidence(content["content_raw"], null["content_raw"], calibration)
    else:
        content["content_evidence"] = {"science_adjudication": "NOT_ADJUDICATED", "per_unit_frozen_content_evidence": "NOT_EVALUABLE_OPERATIONAL_FAILURE"}
    amplitudes = []
    for amplitude in R0_AMPLITUDE_CANDIDATES:
        combined_image, combined_failure = _generate(pipeline, assets, content_key, prompt, seed, "content_geometry", amplitude)
        geometry_image, geometry_failure = _generate(pipeline, assets, content_key, prompt, seed, "geometry_only", amplitude)
        combined = _record(combined_image, combined_failure, content_key, assets, whitening, calibration)
        geometry = _record(geometry_image, geometry_failure, content_key, assets, whitening, calibration)
        if combined["status"] == geometry["status"] == "success":
            combined["content_evidence"] = r0._per_unit_content_evidence(combined["content_raw"], geometry["content_raw"], calibration)
        else:
            combined["content_evidence"] = {"science_adjudication": "NOT_ADJUDICATED", "per_unit_frozen_content_evidence": "NOT_EVALUABLE_OPERATIONAL_FAILURE"}
        amplitudes.append({"amplitude": amplitude, "content_geometry": combined, "geometry_only": geometry, "r01_gate": _unit_amplitude(content, null, combined, geometry, content_image, null_image, combined_image, geometry_image)})
    return {"unit_id": unit["unit_id"], "prompt": prompt, "seed": seed, "baselines": {"content_only": content, "unwatermarked": null}, "amplitudes": amplitudes}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    exact = r0._exact(repo, args.expected_exact)
    content_key, token = r0.os.environ.get(r0.CONTENT_KEY_ENV), r0.os.environ.get("HF_TOKEN")
    if not all(isinstance(value, str) and value.strip() for value in (content_key, token)):
        raise RuntimeError("R0.1 diagnostic requires content and HF credentials")
    roster = _load_roster(repo)
    pipeline, assets = r0._load_assets(token)
    calibration = r0.load_calibration_asset(repo / "configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json", repo / "configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json.sha256")
    whitening = r0.FrozenContentWhiteningLFPublicAssets(assets.lf_public_assets, r0.load_frozen_content_whitening_asset(repo))
    units = [_run_unit(pipeline, assets, whitening, calibration, content_key, unit) for unit in roster]
    amplitude_summaries = []
    for index, amplitude in enumerate(R0_AMPLITUDE_CANDIDATES):
        passed = sum(unit["amplitudes"][index]["r01_gate"]["status"] == "PASS" for unit in units)
        amplitude_summaries.append({"amplitude": amplitude, "passed_units": passed, "required_units": ROSTER_SIZE, "status": "PASS" if passed == ROSTER_SIZE else "FAIL_CLOSED"})
    candidate = all(item["status"] == "PASS" for item in amplitude_summaries)
    return {"method_id": R01_METHOD_ID, "stage": "R0.1_fixed_carrier_candidate_only", "evidence_ceiling": "fixed_four_unit_carrier_window_candidate_only; science_denominator=0; no_formal_fpr_or_robustness_claim", "exact": exact, "runtime_environment": r0._runtime_environment(pipeline), "roster_manifest": ROSTER_MANIFEST, "roster_manifest_sha256": ROSTER_MANIFEST_SHA256, "ordered_roster": [{key: unit[key] for key in ("unit_id", "prompt", "seed", "height", "width")} for unit in roster], "amplitude_sequence": R0_AMPLITUDE_CANDIDATES, "units": units, "amplitude_summaries": amplitude_summaries, "carrier_window": "CARRIER_WINDOW_CANDIDATE" if candidate else "FAIL_CLOSED_NO_CARRIER_WINDOW_CANDIDATE", "conditional_flow_fpr": "NOT_ADJUDICATED", "science_denominator": 0}


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists(): raise FileExistsError("R0.1 output is create-only and already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    path.write_bytes(encoded + b"\n")
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(encoded + b"\n").hexdigest(), "method_id": R01_METHOD_ID}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validate-static", action="store_true")
    parser.add_argument("--run-diagnostic", action="store_true")
    parser.add_argument("--expected-exact")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    if args.validate_static:
        if args.run_diagnostic: raise ValueError("static validation cannot run a diagnostic")
        print(json.dumps({"method_id": R01_METHOD_ID, "roster_size": ROSTER_SIZE, "amplitudes": R0_AMPLITUDE_CANDIDATES, "science_denominator": 0, "status": "STATIC_VALIDATED"}, sort_keys=True)); return 0
    if not args.run_diagnostic or not args.expected_exact or args.output_json is None: raise ValueError("R0.1 diagnostic requires exact and create-only output")
    _write_create_only(args.output_json, _run(args)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
