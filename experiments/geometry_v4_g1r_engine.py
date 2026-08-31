"""V4-G1R CPU canary and real fixed-development runner; no fallback."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from cegwm.method.geometry_v4_g1r import _detect_g1r_engineering, _probe_g1r_at_truth, measure_g1r_final_rgb, write_g1r_rgb
from cegwm.method.geometry_v4_generative import build_reused_weighted_joint_content_adapter
from cegwm.method.geometry_v4_proxy import _sample_h, _similarity_h
from cegwm.protocol.geometry_v4_g1r import (
    ATTACKS,
    DEVELOPMENT_ARTIFACT_FILES,
    DEVELOPMENT_CORRECT_SAFE_REQUIRED,
    DEVELOPMENT_NOTEBOOK_ID,
    DEVELOPMENT_SOURCE_REQUIRED,
    MODEL_ID,
    PLACEMENT,
    WRITER_ID,
    contract_sha256,
    load_contract,
    require_split,
    unsafe_geometry,
)
from cegwm.runtime.geometry_v4_g1r_sd35 import run_g1r_sd35_pair
from cegwm.shared.keys import normalize_detection_key

CPU_KEY = b"geometry-v4-g1r-cpu-key-v1"
CPU_WRONG_KEY = b"geometry-v4-g1r-cpu-wrong-key-v1"
CPU_CARRIER_IDS = ("gradient_shapes", "crosshatch", "radial_objects", "colored_texture")


@dataclass(frozen=True, slots=True)
class BlindArms:
    correct: Mapping[str, object]
    wrong: Mapping[str, object]
    negative: Mapping[str, object]


def build_real_roster(repo_root: str | Path, split: str) -> tuple[tuple[int, str, str], ...]:
    """Return exactly one complete frozen split; subset inputs do not exist."""
    return require_split(load_contract(repo_root), split)


def _carrier(carrier_id: str, size: int = 128) -> np.ndarray:
    if carrier_id not in CPU_CARRIER_IDS:
        raise ValueError("unfrozen V4-G1R CPU carrier")
    yy, xx = np.mgrid[:size, :size]
    x, y = (xx + .5) / size, (yy + .5) / size
    if carrier_id == "gradient_shapes":
        base = .34 + .18 * x + .12 * y + .07 * np.sin(2 * math.pi * (2 * x + y))
        object_mask = ((x - .30) ** 2 + (y - .62) ** 2 < .10**2) | ((abs(x - .72) < .11) & (abs(y - .35) < .08))
        channels = (base + .12 * object_mask, base - .04 * object_mask, base + .02)
    elif carrier_id == "crosshatch":
        hatch = .07 * np.sin(2 * math.pi * 3 * x) * np.cos(2 * math.pi * 5 * y)
        diagonal = .05 * np.sin(2 * math.pi * (4 * x + 3 * y))
        channels = (.48 + hatch + diagonal, .42 - hatch + .5 * diagonal, .38 + .4 * hatch - diagonal)
    elif carrier_id == "radial_objects":
        radius = np.sqrt((x - .52) ** 2 + (y - .48) ** 2)
        rings = .06 * np.cos(2 * math.pi * 5 * radius)
        bars = .08 * ((abs(x - .25) < .035) | (abs(y - .76) < .04))
        channels = (.36 + rings + bars, .50 - .6 * rings, .44 + .5 * rings - .4 * bars)
    else:
        texture = .05 * np.sin(2 * math.pi * (7 * x + 5 * y)) + .04 * np.cos(2 * math.pi * (5 * x - 6 * y))
        blobs = .09 * np.exp(-((x - .28) ** 2 + (y - .30) ** 2) / .012) - .07 * np.exp(-((x - .72) ** 2 + (y - .67) ** 2) / .02)
        channels = (.44 + texture + blobs, .38 - .5 * texture + .6 * blobs, .52 + .3 * texture - .7 * blobs)
    return np.clip(np.stack(channels, axis=-1), 0.08, 0.92).astype(np.float64)


def _apply_attack(rgb: np.ndarray, attack: str) -> np.ndarray:
    """Apply one frozen image operation and return only its RGB result."""
    if attack == "identity":
        output_to_input = np.eye(3, dtype=np.float64)
    elif attack == "rotation_5":
        output_to_input = np.linalg.inv(_similarity_h(5.0, 1.0))
    elif attack == "scale_0.9":
        output_to_input = np.linalg.inv(_similarity_h(0.0, .9))
    elif attack == "translation_0.08_0":
        output_to_input = np.linalg.inv(_similarity_h(0.0, 1.0, .08, 0.0))
    elif attack == "crop_0.9":
        output_to_input = np.linalg.inv(_similarity_h(0.0, 1.0 / .9))
    else:
        raise ValueError("unfrozen V4-G1R attack")
    return np.clip(_sample_h(rgb, output_to_input, 0.0), 0.0, 1.0)


def _truth_for_attack(attack: str) -> np.ndarray:
    """Truth-only evaluator input; call strictly after blind arms are frozen."""
    if attack == "identity":
        canonical_to_attacked = np.eye(3, dtype=np.float64)
    elif attack == "rotation_5":
        canonical_to_attacked = _similarity_h(5.0, 1.0)
    elif attack == "scale_0.9":
        canonical_to_attacked = _similarity_h(0.0, .9)
    elif attack == "translation_0.08_0":
        canonical_to_attacked = _similarity_h(0.0, 1.0, .08, 0.0)
    elif attack == "crop_0.9":
        canonical_to_attacked = _similarity_h(0.0, 1.0 / .9)
    else:
        raise ValueError("unfrozen V4-G1R truth attack")
    return np.linalg.inv(canonical_to_attacked)


def _blind_arms_for_keys(attacked_marked: np.ndarray, attacked_negative: np.ndarray, correct_key: object, wrong_key: object) -> BlindArms:
    """Freeze three independent attacked-RGB/key-only arms before truth exists."""
    correct, correct_diagnostics = _detect_g1r_engineering(attacked_marked, correct_key)
    wrong, wrong_diagnostics = _detect_g1r_engineering(attacked_marked, wrong_key)
    negative, negative_diagnostics = _detect_g1r_engineering(attacked_negative, correct_key)
    return BlindArms(
        correct={**correct, "engineering_diagnostics": correct_diagnostics},
        wrong={**wrong, "engineering_diagnostics": wrong_diagnostics},
        negative={**negative, "engineering_diagnostics": negative_diagnostics},
    )


def _blind_arms(attacked_marked: np.ndarray, attacked_negative: np.ndarray) -> BlindArms:
    """CPU-canary binding of the same complete three-arm blind path."""
    return _blind_arms_for_keys(attacked_marked, attacked_negative, CPU_KEY, CPU_WRONG_KEY)


def _points(homography: np.ndarray) -> np.ndarray:
    points = []
    for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (.5, .5)):
        point = homography @ np.asarray((x, y, 1.0), dtype=np.float64)
        points.append(point[:2] / point[2])
    return np.asarray(points)


def _angle_scale(homography: np.ndarray) -> tuple[float, float]:
    return math.degrees(math.atan2(float(homography[1, 0]), float(homography[0, 0]))), math.hypot(float(homography[0, 0]), float(homography[1, 0]))


def _truth_metrics(arm: Mapping[str, object], truth_attacked_to_canonical: np.ndarray) -> Mapping[str, float]:
    if arm.get("status") != "RELIABLE" or arm.get("H_hat") is None:
        return {"mapped_corner_error": 0.0, "center_reprojection_error": 0.0, "rotation_abs_error_degrees": 0.0, "log_scale_abs_error": 0.0}
    estimate = np.asarray(arm["H_hat"], dtype=np.float64).reshape(3, 3)
    predicted, truth = _points(estimate), _points(truth_attacked_to_canonical)
    distances = np.linalg.norm(predicted - truth, axis=1) / math.sqrt(2.0)
    predicted_angle, predicted_scale = _angle_scale(estimate)
    truth_angle, truth_scale = _angle_scale(truth_attacked_to_canonical)
    rotation_error = abs((predicted_angle - truth_angle + 90.0) % 180.0 - 90.0)
    return {"mapped_corner_error": float(np.max(distances[:4])), "center_reprojection_error": float(distances[4]), "rotation_abs_error_degrees": rotation_error, "log_scale_abs_error": abs(math.log(predicted_scale) - math.log(truth_scale))}


def _evaluate_frozen_arms(arms: BlindArms, truth_attacked_to_canonical: np.ndarray) -> Mapping[str, object]:
    evaluated = {}
    for name in ("correct", "wrong", "negative"):
        arm = getattr(arms, name)
        metrics = _truth_metrics(arm, truth_attacked_to_canonical)
        evaluated[name] = {**arm, "truth_metrics": metrics, "unsafe": unsafe_geometry(str(arm["status"]), metrics)}
    return evaluated


def _truth_probe(attacked_marked: np.ndarray, correct_key: object, truth_attacked_to_canonical: np.ndarray) -> Mapping[str, object]:
    """Post-freeze runner diagnostic; never feeds an arm, rank, or summary."""
    return _probe_g1r_at_truth(attacked_marked, correct_key, truth_attacked_to_canonical)


def run_cpu_canary() -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for carrier_id in CPU_CARRIER_IDS:
        ordinary = _carrier(carrier_id)
        marked, budget = write_g1r_rgb(ordinary, CPU_KEY)
        for attack in ATTACKS:
            base = {"carrier": carrier_id, "attack": attack, "failure": None, "budget": budget}
            try:
                attacked_marked = _apply_attack(marked, attack)
                attacked_negative = _apply_attack(ordinary, attack)
                arms = _blind_arms(attacked_marked, attacked_negative)
                evaluation = _evaluate_frozen_arms(arms, _truth_for_attack(attack))
                records.append({**base, "arms": evaluation})
            except Exception as error:
                records.append({**base, "failure": f"{type(error).__name__}: {error}", "arms": None})
    if len(records) != 20:
        raise RuntimeError("V4-G1R CPU denominator differs")
    return tuple(records)


def summarize_cpu_canary(records: tuple[Mapping[str, object], ...]) -> Mapping[str, object]:
    if len(records) != 20 or tuple(record["attack"] for record in records) != ATTACKS * 4:
        raise ValueError("V4-G1R CPU records differ from fixed 4x5 roster")
    failures = sum(record["failure"] is not None for record in records)
    correct_safe = sum(record["arms"] is not None and record["arms"]["correct"]["status"] == "RELIABLE" and not record["arms"]["correct"]["unsafe"] for record in records)
    correct_unsafe = sum(record["arms"] is not None and record["arms"]["correct"]["unsafe"] for record in records)
    wrong_unsafe = sum(record["arms"] is not None and record["arms"]["wrong"]["unsafe"] for record in records)
    negative_unsafe = sum(record["arms"] is not None and record["arms"]["negative"]["unsafe"] for record in records)
    by_attack = {attack: sum(record["attack"] == attack and record["arms"] is not None and record["arms"]["correct"]["status"] == "RELIABLE" and not record["arms"]["correct"]["unsafe"] for record in records) for attack in ATTACKS}
    top5 = {attack: sum(record["attack"] == attack and record["arms"] is not None and bool(record["arms"]["correct"].get("engineering_diagnostics", {}).get("top5_hit", False)) for record in records) for attack in ATTACKS}
    identity_psr = sum(record["attack"] == "identity" and record["arms"] is not None and float(record["arms"]["correct"].get("engineering_diagnostics", {}).get("translation_psr", 0.0)) >= 8.0 for record in records)
    passed = failures == 0 and correct_safe >= 18 and correct_unsafe == wrong_unsafe == negative_unsafe == 0 and sum(top5.values()) >= 18 and all(value >= 3 for value in by_attack.values()) and all(value >= 3 for value in top5.values()) and identity_psr >= 3
    return {"stage": "V4-G1R", "evidence_label": "CPU_SYNTHETIC_ENGINEERING_EXIT", "evidence": "synthetic_only", "formal_denominator": 0, "cpu_carrier_ids": CPU_CARRIER_IDS, "units": 20, "failures": failures, "stops": failures, "correct_safe_reliable": correct_safe, "correct_unsafe": correct_unsafe, "wrong_unsafe": wrong_unsafe, "negative_unsafe": negative_unsafe, "correct_safe_by_attack": by_attack, "correct_rs_top5_by_attack": top5, "correct_rs_top5": sum(top5.values()), "identity_translation_psr_ge_8": identity_psr, "exit": passed, "status": "CPU_ENGINEERING_EXIT" if passed else "CPU_METHOD_PARTIAL"}


def _rgb(image: Image.Image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("V4-G1R real runner requires PIL RGB outputs")
    return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def _failure(error: BaseException, scope: str) -> Mapping[str, str]:
    return {"scope": scope, "type": type(error).__name__}


def _load_real_pipeline_and_assets(hf_token: str) -> tuple[Any, Any]:
    from experiments.content_iss_engine import _load_pipeline_and_assets

    pipeline, assets = _load_pipeline_and_assets(MODEL_ID, hf_token)
    return pipeline.to("cuda"), assets


def _cuda_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _failed_development(roster: tuple[tuple[int, str, str], ...], error: BaseException, scope: str) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    failure = _failure(error, scope)
    sources = tuple({"seed": seed, "prompt": prompt, "failure": failure, "final_rgb": None} for seed, prompt, _ in roster[:: len(ATTACKS)])
    records = tuple({"seed": seed, "prompt": prompt, "attack": attack, "failure": failure, "arms": None} for seed, prompt, attack in roster)
    return sources, records


def run_real_development(
    detection_key: object,
    wrong_key: object,
    *,
    repo_root: str | Path,
    hf_token: str,
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...], Mapping[str, str]]:
    """Run exactly four real pairs and the contract-internal 20 development units."""
    contract = load_contract(repo_root)
    roster = require_split(contract, "development")
    if len(roster) != 20 or tuple(item[2] for item in roster) != ATTACKS * 4:
        raise RuntimeError("V4-G1R development roster differs")
    normalized = normalize_detection_key(detection_key)
    normalized_wrong = normalize_detection_key(wrong_key)
    if normalized == normalized_wrong:
        raise ValueError("V4-G1R real wrong key collides")
    if not torch.cuda.is_available():
        raise RuntimeError("V4-G1R development requires CUDA")
    try:
        pipeline, assets = _load_real_pipeline_and_assets(hf_token)
        content_detector = build_reused_weighted_joint_content_adapter(assets, repo_root)
    except Exception as error:
        sources, records = _failed_development(roster, error, "model_or_asset_load")
        return sources, records, {}

    sources: list[Mapping[str, object]] = []
    records: list[Mapping[str, object]] = []
    for seed, prompt, first_attack in roster[:: len(ATTACKS)]:
        if first_attack != ATTACKS[0]:
            raise RuntimeError("V4-G1R source boundary differs")
        source_base = {"seed": seed, "prompt": prompt, "failure": None}
        try:
            pair = run_g1r_sd35_pair(pipeline, prompt, normalized, height=512, width=512, generator=_cuda_generator(seed))
            clean_rgb, marked_rgb = _rgb(pair.clean), _rgb(pair.marked)
        except Exception as error:
            failure = _failure(error, "source_generation")
            sources.append({**source_base, "failure": failure, "final_rgb": None})
            records.extend({"seed": seed, "prompt": prompt, "attack": attack, "failure": failure, "arms": None} for attack in ATTACKS)
            continue
        try:
            observation = measure_g1r_final_rgb(clean_rgb, marked_rgb, normalized, normalized_wrong, content_detector)
            final_rgb: Mapping[str, object] = {"passed": observation.passed, **asdict(observation)}
            source_failure = None
        except Exception as error:
            final_rgb = {"passed": False, "failure": _failure(error, "final_rgb_observability")}
            source_failure = final_rgb["failure"]
        sources.append({**source_base, "failure": source_failure, "final_rgb": final_rgb})
        for attack in ATTACKS:
            unit_base = {"seed": seed, "prompt": prompt, "attack": attack, "failure": None}
            try:
                attacked_marked = _apply_attack(marked_rgb, attack)
                attacked_negative = _apply_attack(clean_rgb, attack)
                arms = _blind_arms_for_keys(attacked_marked, attacked_negative, normalized, normalized_wrong)
                truth = _truth_for_attack(attack)
                evaluation = _evaluate_frozen_arms(arms, truth)
                try:
                    truth_probe: Mapping[str, object] = _truth_probe(attacked_marked, normalized, truth)
                except Exception as error:
                    truth_probe = {"failure": _failure(error, "truth_probe")}
                records.append({**unit_base, "arms": evaluation, "truth_probe": truth_probe})
            except Exception as error:
                records.append({**unit_base, "failure": _failure(error, "attacked_unit"), "arms": None})
    if len(sources) != 4 or len(records) != 20:
        raise RuntimeError("V4-G1R real denominator differs")
    return tuple(sources), tuple(records), content_detector.identities()


def summarize_real_development(sources: tuple[Mapping[str, object], ...], records: tuple[Mapping[str, object], ...]) -> Mapping[str, object]:
    if len(sources) != 4 or len(records) != 20 or tuple(record["attack"] for record in records) != ATTACKS * 4:
        raise ValueError("V4-G1R real development records differ")
    source_passed = sum(bool(source.get("final_rgb", {}).get("passed")) for source in sources if isinstance(source.get("final_rgb"), Mapping))
    source_failures = sum(source.get("failure") is not None for source in sources)
    unit_failures = sum(record.get("failure") is not None for record in records)
    correct_safe = sum(record.get("arms") is not None and record["arms"]["correct"]["status"] == "RELIABLE" and not record["arms"]["correct"]["unsafe"] for record in records)
    correct_unsafe = sum(record.get("arms") is not None and record["arms"]["correct"]["unsafe"] for record in records)
    wrong_unsafe = sum(record.get("arms") is not None and record["arms"]["wrong"]["unsafe"] for record in records)
    negative_unsafe = sum(record.get("arms") is not None and record["arms"]["negative"]["unsafe"] for record in records)
    failures = source_failures + unit_failures
    passed = source_passed == DEVELOPMENT_SOURCE_REQUIRED and correct_safe == DEVELOPMENT_CORRECT_SAFE_REQUIRED and correct_unsafe == wrong_unsafe == negative_unsafe == failures == 0
    return {
        "stage": "development",
        "status": "PASS" if passed else "GATE_FAILED",
        "sources": 4,
        "source_observability_passed": source_passed,
        "units": 20,
        "correct_safe_reliable": correct_safe,
        "correct_unsafe": correct_unsafe,
        "wrong_unsafe": wrong_unsafe,
        "negative_unsafe": negative_unsafe,
        "failures": failures,
        "source_failures": source_failures,
        "unit_failures": unit_failures,
    }


def _checkout_state(repo_root: str | Path) -> tuple[str, str, bool]:
    root = Path(repo_root)

    def git(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True, text=True).stdout.strip()

    return git("rev-parse", "HEAD"), git("branch", "--show-current"), git("status", "--porcelain") == ""


def _secret_environment(environ: dict[str, str]) -> tuple[bytes, bytes, str]:
    root_key = environ.pop("CEG_WM_ROOT_KEY", "")
    hf_token = environ.pop("HF_TOKEN", "")
    try:
        if not isinstance(root_key, str) or not root_key or not isinstance(hf_token, str) or not hf_token:
            raise RuntimeError("V4-G1R required secrets are unavailable")
        normalized = normalize_detection_key(root_key)
        from cegwm.runtime.content_weighted_joint_sd35 import derive_stability_wrong_keys

        wrong_key = derive_stability_wrong_keys(normalized)[0]
        return normalized, wrong_key, hf_token
    finally:
        root_key = ""
        environ.pop("CEG_WM_ROOT_KEY", None)
        environ.pop("HF_TOKEN", None)


def _environment_metadata() -> Mapping[str, object]:
    metadata: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        metadata["gpu"] = {"name": properties.name, "vram_bytes": int(properties.total_memory)}
    return metadata


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")


def _write_create_only(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)


def write_development_artifacts(
    artifact_root: str | Path,
    *,
    source_exact: str,
    repo_root: str | Path,
    sources: tuple[Mapping[str, object], ...],
    records: tuple[Mapping[str, object], ...],
    summary: Mapping[str, object],
    content_detector: Mapping[str, str],
) -> Mapping[str, object]:
    root = Path(artifact_root)
    records_name, summary_name, manifest_name = DEVELOPMENT_ARTIFACT_FILES
    digest = contract_sha256(repo_root)
    records_payload = _json_bytes({"stage": "development", "source_exact": source_exact, "config_sha256": digest, "content_detector": content_detector, "sources": sources, "records": records})
    summary_payload = _json_bytes({**summary, "source_exact": source_exact, "config_sha256": digest})
    records_sha256 = hashlib.sha256(records_payload).hexdigest()
    summary_sha256 = hashlib.sha256(summary_payload).hexdigest()
    manifest = {
        "schema": "cegwm_geometry_v4_g1r_development_artifact_v1",
        "stage": "development",
        "source_exact": source_exact,
        "config_sha256": digest,
        "model_id": MODEL_ID,
        "placement": PLACEMENT,
        "writer_id": WRITER_ID,
        "notebook_identity": DEVELOPMENT_NOTEBOOK_ID,
        "seeds": [seed for seed, _, _ in require_split(load_contract(repo_root), "development")[:: len(ATTACKS)]],
        "prompts": [prompt for _, prompt, _ in require_split(load_contract(repo_root), "development")[:: len(ATTACKS)]],
        "attacks": list(ATTACKS),
        "units": 20,
        "environment": _environment_metadata(),
        "files": {records_name: records_sha256, summary_name: summary_sha256},
    }
    manifest_payload = _json_bytes(manifest)
    payloads = {records_name: records_payload, summary_name: summary_payload, manifest_name: manifest_payload}
    root.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        _write_create_only(root / name, payload)
        sidecar = f"{hashlib.sha256(payload).hexdigest()}  {name}\n".encode("ascii")
        _write_create_only(root / f"{name}.sha256", sidecar)
    return manifest


def main(argv: list[str] | None = None, *, environ: dict[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Geometry-V4 G1R real development runner")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    args = parser.parse_args(argv)
    stage = str(args.stage)
    try:
        if stage != "development":
            raise RuntimeError("V4-G1R CLI permits development only")
        if Path(args.artifact_root).exists():
            raise FileExistsError("V4-G1R artifact root already exists")
        exact, branch, clean = _checkout_state(args.repo_root)
        if exact != args.expected_exact or branch != "" or not clean:
            raise RuntimeError("V4-G1R detached checkout exact or clean state differs")
        env = os.environ if environ is None else environ
        detection_key, wrong_key, hf_token = _secret_environment(env)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                sources, records, detector_identity = run_real_development(detection_key, wrong_key, repo_root=args.repo_root, hf_token=hf_token)
        finally:
            hf_token = ""
            env.pop("HF_TOKEN", None)
        summary = summarize_real_development(sources, records)
        write_development_artifacts(args.artifact_root, source_exact=exact, repo_root=args.repo_root, sources=sources, records=records, summary=summary, content_detector=detector_identity)
        compact = {**summary, "source_exact": exact}
        print(json.dumps(compact, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
        return 0 if summary["status"] == "PASS" else 2
    except Exception as error:
        print(json.dumps({"stage": stage, "status": "STOPPED", "error": type(error).__name__}, sort_keys=True, separators=(",", ":")), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
