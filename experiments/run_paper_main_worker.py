"""Formal main-method worker with frozen paper threshold and minimal ablations."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from cegwm.formal_ablation import EMBEDDING_ABLATIONS, run_formal_ablation_pair
from cegwm.formal_experiment import (
    CLEAN_TEST_NEGATIVES,
    EVALUATION_PAIRS,
    FORMAL_CONDITIONS,
    FormalRunStore,
    OperationalUnitError,
    PreflightFailed,
    apply_attack,
    empty_binary_summary,
    execute_job_preflight,
    expand_rosters,
    freeze_threshold,
    load_or_recover_pair,
    load_formal_config,
    publish_job_state,
    raise_classified_operational,
    summarize_binary,
    summarize_quality,
)
from cegwm.geometry_v7.r1b import rectify_attacked_rgb
from cegwm.runtime.blind_detection import (
    _detect_core,
    _geometry_disposition,
    _raw_h,
    _score_current_rgb,
)
from cegwm.runtime.content_iss_sd35 import run_content_iss_evaluation_pair
from cegwm.runtime.diffusers_sd35 import run_sd35_plain
from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from experiments.run_blind_detection_v1 import (
    CONTENT_CHAIN_PUBLIC_KEY_DIGEST,
    build_production_runtime,
    load_runtime_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/paper_experiment/formal_experiment_v1.json"
METHOD_ID = "cegwm_blind_detection_v1_paper"
SYNCSEAL_RESIDUAL_MULTIPLIER = 0.75
PREFLIGHT_PROMPT = "A neutral geometric still life for an engineering preflight."
PREFLIGHT_SEED = 2027000000


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def verify_expected_exact(expected_exact: str) -> None:
    if _git_head() != expected_exact:
        raise RuntimeError("checked-out PaperFPR producer exact differs from notebook expectation")
    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("formal main worker requires a clean detached checkout")


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_png_create_only(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        require_ordinary_rgb_image(image).save(stream, format="PNG")
        stream.flush()
        os.fsync(stream.fileno())


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        result = image.convert("RGB").copy()
    return require_ordinary_rgb_image(result)


def _identity(job_id: str, stage: str, exact: str) -> dict[str, str]:
    return {
        "schema_version": "cegwm_formal_job_v1",
        "job_id": job_id,
        "run_id": f"{job_id}:{METHOD_ID}:{stage}",
        "method_id": METHOD_ID,
        "stage": stage,
        "expected_exact": exact,
    }


def _generator(seed: int, device: str) -> Any:
    import torch
    return torch.Generator(device=device).manual_seed(seed)


def _plain(runtime: Mapping[str, Any], prompt: str, seed: int) -> Image.Image:
    return require_ordinary_rgb_image(run_sd35_plain(
        runtime["pipeline"], prompt, height=512, width=512,
        generator=_generator(seed, runtime["device"]),
    ))


def _main_pair(runtime: Mapping[str, Any], prompt: str, seed: int) -> tuple[Image.Image, Image.Image]:
    pair = run_content_iss_evaluation_pair(
        runtime["pipeline"], prompt, runtime["key"], runtime["assets"].content_assets.iss_assets,
        height=512, width=512, seed=seed,
    )
    clean = require_ordinary_rgb_image(pair.primary_null).copy()
    content = require_ordinary_rgb_image(pair.image)
    watermarked = runtime["assets"].geometry_backend.embed_final_rgb(
        content, SYNCSEAL_RESIDUAL_MULTIPLIER
    )
    return clean, require_ordinary_rgb_image(watermarked).copy()


def _ablation_pair(runtime: Mapping[str, Any], prompt: str, seed: int, variant: str) -> tuple[Image.Image, Image.Image]:
    pair = run_formal_ablation_pair(
        runtime["pipeline"], prompt, runtime["key"],
        runtime["assets"].content_assets.iss_assets,
        variant, seed=seed,
    )
    watermarked = runtime["assets"].geometry_backend.embed_final_rgb(
        pair.image, SYNCSEAL_RESIDUAL_MULTIPLIER
    )
    return pair.primary_null.copy(), require_ordinary_rgb_image(watermarked).copy()


def _calibration_score(runtime: Mapping[str, Any], image: Image.Image) -> tuple[float, str]:
    try:
        pre = _score_current_rgb(image, runtime["key"], runtime["assets"])
    except Exception as error:
        raise_classified_operational(error, "content_pre")
    try:
        geometry = runtime["assets"].geometry_backend.detect_geometry(image)
    except Exception as error:
        raise_classified_operational(error, "geometry")
    disposition, detail = _geometry_disposition(geometry)
    if disposition == "OPERATIONAL":
        raise OperationalUnitError("MODEL_RUNTIME_TRANSIENT", "geometry", detail or "geometry operational failure")
    if disposition == "INVALID_H":
        return pre.value, "GEOMETRY_FAIL_CLOSED"
    try:
        matrix = _raw_h(geometry)
    except LookupError:
        return pre.value, "GEOMETRY_NO_H"
    except (TypeError, ValueError):
        return pre.value, "GEOMETRY_FAIL_CLOSED"
    try:
        recovered = rectify_attacked_rgb(image, matrix)
    except Exception:
        return pre.value, "RECTIFICATION_FAIL_CLOSED"
    try:
        post = _score_current_rgb(recovered, runtime["key"], runtime["assets"])
    except Exception as error:
        raise_classified_operational(error, "content_post")
    return max(pre.value, post.value), "GEOMETRY_RECOVERED"


def _detect_payload(runtime: Mapping[str, Any], image: Image.Image, tau: float, *, no_geometry: bool = False) -> dict[str, Any]:
    if no_geometry:
        try:
            pre = _score_current_rgb(image, runtime["key"], runtime["assets"])
        except Exception as error:
            raise_classified_operational(error, "content_pre")
        return {
            "normalized_score": pre.value,
            "decision": pre.value > tau,
            "route": "CONTROLLED_NO_GEOMETRY",
            "threshold": tau,
            "method_complete": True,
        }
    record = _detect_core(image, runtime["key"], runtime["assets"], tau)
    if not record.method_complete:
        raise OperationalUnitError(
            "MODEL_RUNTIME_TRANSIENT", "detection", record.operational_error or "unknown detection interruption"
        )
    score = record.pre.value if record.pre is not None else None
    if record.post is not None:
        score = max(float(score), record.post.value)
    if score is None or not math.isfinite(float(score)):
        raise OperationalUnitError("MODEL_RUNTIME_TRANSIENT", "detection", "complete detection lacks finite score")
    return {
        "normalized_score": float(score),
        "decision": record.positive,
        "route": record.route,
        "threshold": tau,
        "method_complete": True,
    }


_LPIPS_MODEL: Any | None = None


def _quality(clean: Image.Image, watermarked: Image.Image) -> dict[str, float]:
    import torch
    import lpips
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )
    global _LPIPS_MODEL
    device = torch.device("cuda")
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(device).eval()
    first = torch.from_numpy(np.asarray(clean, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    second = torch.from_numpy(np.asarray(watermarked, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        return {
            "psnr": float(peak_signal_noise_ratio(second, first, data_range=1.0).item()),
            "ssim": float(structural_similarity_index_measure(second, first, data_range=1.0).item()),
            "lpips": float(_LPIPS_MODEL(second * 2.0 - 1.0, first * 2.0 - 1.0).reshape(-1)[0].item()),
        }


def _threshold_from_stage(store: FormalRunStore, output: Path, exact: str) -> dict[str, Any] | None:
    if output.exists():
        value = json.loads(output.read_text(encoding="utf-8"))
        if value.get("method_id") != METHOD_ID or value.get("producer_exact") != exact:
            raise RuntimeError("paper main threshold identity differs")
        return value
    rows = store.rows()
    if len(rows) != 2000 or any(row["terminal_status"] != "SCORED" for row in rows):
        return None
    threshold = {
        "schema_version": "cegwm_paper_threshold_v1",
        "method_id": METHOD_ID,
        "producer_exact": exact,
        "score_id": "max_pre_post_registered_weighted_joint_minus_exact16_wrong_max",
        **freeze_threshold([row["normalized_score"] for row in rows]),
    }
    _write_json_create_only(output, threshold)
    return threshold


def _empty_evaluation() -> dict[str, Any]:
    return {
        f"{condition}:{role}": empty_binary_summary(
            truth_positive=role == "positive", planned=EVALUATION_PAIRS
        )
        for condition in FORMAL_CONDITIONS
        for role in ("negative", "positive")
    }


def _empty_ablations(config: Mapping[str, Any]) -> dict[str, Any]:
    subset_size = config["ablations"]["subset_size"]
    return {
        f"{variant}:{condition}:{role}": empty_binary_summary(
            truth_positive=role == "positive", planned=subset_size
        )
        for variant, definition in config["ablations"]["variants"].items()
        for condition in definition["conditions"]
        for role in ("negative", "positive")
    }


def _build_runtime(runtime_root: Path) -> dict[str, Any]:
    root_key = os.environ.get("CEG_WM_ROOT_KEY", "")
    token = os.environ.get("HF_TOKEN", "")
    if not root_key or not token:
        raise RuntimeError("CEG_WM_ROOT_KEY and HF_TOKEN are required")
    key = normalize_detection_key(root_key)
    if public_key_digest(key) != CONTENT_CHAIN_PUBLIC_KEY_DIGEST:
        raise RuntimeError("content-chain detection key identity differs")
    config = load_runtime_config(REPO_ROOT)
    pipeline, assets = build_production_runtime(
        REPO_ROOT, config, hf_token=token, runtime_root=runtime_root
    )
    return {"pipeline": pipeline, "assets": assets, "key": key, "device": config["device"]}


def _prepare_runtime(runtime_root: Path) -> dict[str, Any]:
    runtime = _build_runtime(runtime_root)
    clean, marked = _main_pair(runtime, PREFLIGHT_PROMPT, PREFLIGHT_SEED)
    _calibration_score(runtime, clean)
    _detect_payload(runtime, marked, 0.0)
    _quality(clean, marked)
    return runtime


def run_worker(*, job_id: str, expected_exact: str, drive_root: Path, runtime_root: Path) -> int:
    verify_expected_exact(expected_exact)
    config = load_formal_config(CONFIG_PATH)
    rosters = expand_rosters(REPO_ROOT, config)
    job_root = drive_root / job_id
    final_path = job_root / "method_final.json"
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if final.get("producer_exact") != expected_exact:
            raise RuntimeError("existing main final exact differs")
        print(json.dumps({"method_id": METHOD_ID, "status": final["status"], "terminal": True}, sort_keys=True))
        return 0

    try:
        runtime = execute_job_preflight(
            job_root,
            _identity(job_id, "preflight", expected_exact),
            lambda: _prepare_runtime(runtime_root),
        )
    except PreflightFailed as error:
        print(json.dumps({
            "method_id": METHOD_ID,
            "status": error.state["status"],
            "science_denominator": 0,
        }, sort_keys=True))
        return 3

    def get_runtime() -> dict[str, Any]:
        return runtime

    calibration_units = rosters["threshold_calibration"]
    calibration = FormalRunStore(
        job_root / "threshold_calibration", _identity(job_id, "threshold_calibration", expected_exact),
        [unit.unit_id for unit in calibration_units],
    )
    threshold_path = job_root / "threshold.json"
    threshold = _threshold_from_stage(calibration, threshold_path, expected_exact)
    if threshold is None and not calibration.completed_result():
        by_id = {unit.unit_id: unit for unit in calibration_units}

        def calibrate(unit_id: str, attempt: int) -> dict[str, Any]:
            del attempt
            unit = by_id[unit_id]
            score, route = _calibration_score(get_runtime(), _plain(get_runtime(), unit.prompt, unit.seed))
            return {"normalized_score": score, "calibration_route": route}

        calibration.run(calibrate)
        status = "COMPLETE" if all(row["terminal_status"] == "SCORED" for row in calibration.rows()) else "INCOMPLETE_OPERATIONAL"
        calibration.finalize({"status": status, "threshold_created": False})
        threshold = _threshold_from_stage(calibration, threshold_path, expected_exact)
    if threshold is None:
        _write_json_create_only(final_path, {
            "schema_version": "cegwm_formal_method_result_v1",
            "method_id": METHOD_ID,
            "producer_exact": expected_exact,
            "threshold": None,
            "threshold_status": "INCOMPLETE_THRESHOLD",
            "clean_negative_test": empty_binary_summary(
                truth_positive=False, planned=CLEAN_TEST_NEGATIVES
            ),
            "evaluation": _empty_evaluation(),
            "ablations": _empty_ablations(config),
            "quality": summarize_quality((), planned=EVALUATION_PAIRS),
            "quality_source": "not_available_threshold_stage_incomplete",
            "status": "INCOMPLETE_OPERATIONAL",
            "reason": "paper threshold unavailable after terminal calibration",
            "fpr_policy": "report_only_nonblocking",
            "result_package_produced": True,
        })
        print(json.dumps({
            "method_id": METHOD_ID,
            "status": "INCOMPLETE_OPERATIONAL",
            "terminal": True,
        }, sort_keys=True))
        publish_job_state(
            job_root, _identity(job_id, "method_terminal", expected_exact),
            "TERMINAL_INCOMPLETE", reason="threshold unavailable",
        )
        return 0
    tau = float(threshold["tau"])

    clean_units = rosters["clean_negative_test"]
    clean_store = FormalRunStore(
        job_root / "clean_negative_test", _identity(job_id, "clean_negative_test", expected_exact),
        [unit.unit_id for unit in clean_units],
    )
    if not clean_store.completed_result():
        by_id = {unit.unit_id: unit for unit in clean_units}

        def clean_detect(unit_id: str, attempt: int) -> dict[str, Any]:
            del attempt
            unit = by_id[unit_id]
            return _detect_payload(get_runtime(), _plain(get_runtime(), unit.prompt, unit.seed), tau)

        clean_store.run(clean_detect)
        summary = summarize_binary(clean_store.rows(), truth_positive=False, planned=CLEAN_TEST_NEGATIVES)
        clean_store.finalize({"summary": summary, "status": summary["status"]})

    eval_units = rosters["formal_evaluation_pairs"]
    image_root = job_root / "evaluation_images"
    generation = FormalRunStore(
        job_root / "evaluation_generation", _identity(job_id, "evaluation_generation", expected_exact),
        [unit.unit_id for unit in eval_units],
    )
    if not generation.completed_result():
        by_id = {unit.unit_id: unit for unit in eval_units}

        def generate(unit_id: str, attempt: int) -> dict[str, Any]:
            del attempt
            unit = by_id[unit_id]
            clean_path = image_root / f"{unit.roster_index:04d}" / "clean.png"
            marked_path = image_root / f"{unit.roster_index:04d}" / "watermarked.png"
            clean, marked, pair_state = load_or_recover_pair(
                clean_path,
                marked_path,
                lambda: _main_pair(get_runtime(), unit.prompt, unit.seed),
                _load_rgb,
                _atomic_png_create_only,
            )
            return {
                "artifact_status": "GENERATED",
                "clean_image": str(clean_path.relative_to(job_root)),
                "watermarked_image": str(marked_path.relative_to(job_root)),
                "pair_state": pair_state,
                "quality": _quality(clean, marked),
            }

        generation.run(generate)
        generation.finalize({
            "status": "COMPLETE" if all(row["terminal_status"] == "SCORED" for row in generation.rows()) else "INCOMPLETE_OPERATIONAL"
        })

    eval_items = [
        (unit, condition, role)
        for unit in eval_units for condition in FORMAL_CONDITIONS for role in ("negative", "positive")
    ]
    eval_ids = [f"{unit.unit_id}__{condition}__{role}" for unit, condition, role in eval_items]
    detection = FormalRunStore(
        job_root / "evaluation_detection", _identity(job_id, "evaluation_detection", expected_exact), eval_ids,
    )
    if not detection.completed_result():
        item_by_id = dict(zip(eval_ids, eval_items, strict=True))

        def detect(item_id: str, attempt: int) -> dict[str, Any]:
            del attempt
            unit, condition, role = item_by_id[item_id]
            source = image_root / f"{unit.roster_index:04d}" / ("watermarked.png" if role == "positive" else "clean.png")
            return {
                **_detect_payload(get_runtime(), apply_attack(_load_rgb(source), condition), tau),
                "physical_unit_id": unit.unit_id,
                "condition": condition,
                "truth_role": role,
            }

        detection.run(detect)
        summaries: dict[str, Any] = {}
        for condition in FORMAL_CONDITIONS:
            for role in ("negative", "positive"):
                subset = [row for row in detection.rows() if row.get("condition") == condition and row.get("truth_role") == role]
                summaries[f"{condition}:{role}"] = summarize_binary(
                    subset, truth_positive=role == "positive", planned=EVALUATION_PAIRS
                )
        detection.finalize({
            "summaries": summaries,
            "status": "COMPLETE" if all(value["status"] == "COMPLETE" for value in summaries.values()) else "INCOMPLETE_OPERATIONAL",
        })

    ablation_config = config["ablations"]
    ablation_units = eval_units[:ablation_config["subset_size"]]
    ablation_image_root = job_root / "ablation_images"
    generation_items = [(variant, unit) for variant in EMBEDDING_ABLATIONS for unit in ablation_units]
    generation_ids = [f"{variant}__{unit.unit_id}" for variant, unit in generation_items]
    ablation_generation = FormalRunStore(
        job_root / "ablation_generation", _identity(job_id, "ablation_generation", expected_exact), generation_ids,
    )
    if not ablation_generation.completed_result():
        item_by_id = dict(zip(generation_ids, generation_items, strict=True))

        def generate_ablation(item_id: str, attempt: int) -> dict[str, Any]:
            del attempt
            variant, unit = item_by_id[item_id]
            path = ablation_image_root / variant / f"{unit.roster_index:04d}.png"
            if path.exists():
                marked = _load_rgb(path)
            else:
                _, marked = _ablation_pair(get_runtime(), unit.prompt, unit.seed, variant)
                _atomic_png_create_only(path, marked)
            return {"artifact_status": "GENERATED", "variant": variant, "watermarked_image": str(path.relative_to(job_root))}

        ablation_generation.run(generate_ablation)
        ablation_generation.finalize({
            "status": "COMPLETE" if all(row["terminal_status"] == "SCORED" for row in ablation_generation.rows()) else "INCOMPLETE_OPERATIONAL"
        })

    variants = tuple(ablation_config["variants"])
    ablation_items = [
        (variant, unit, condition, role)
        for variant in variants
        for unit in ablation_units
        for condition in ablation_config["variants"][variant]["conditions"]
        for role in ("negative", "positive")
    ]
    ablation_ids = [f"{variant}__{unit.unit_id}__{condition}__{role}" for variant, unit, condition, role in ablation_items]
    ablation_detection = FormalRunStore(
        job_root / "ablation_detection", _identity(job_id, "ablation_detection", expected_exact), ablation_ids,
    )
    if not ablation_detection.completed_result():
        item_by_id = dict(zip(ablation_ids, ablation_items, strict=True))

        def detect_ablation(item_id: str, attempt: int) -> dict[str, Any]:
            del attempt
            variant, unit, condition, role = item_by_id[item_id]
            if role == "negative" or variant == "no_geometry":
                source = image_root / f"{unit.roster_index:04d}" / ("watermarked.png" if role == "positive" else "clean.png")
            else:
                source = ablation_image_root / variant / f"{unit.roster_index:04d}.png"
            return {
                **_detect_payload(
                    get_runtime(), apply_attack(_load_rgb(source), condition), tau,
                    no_geometry=variant == "no_geometry",
                ),
                "physical_unit_id": unit.unit_id,
                "condition": condition,
                "truth_role": role,
                "variant": variant,
                "threshold_role": ablation_config["variants"][variant]["threshold_role"],
            }

        ablation_detection.run(detect_ablation)
        summaries: dict[str, Any] = {}
        for variant in variants:
            for condition in ablation_config["variants"][variant]["conditions"]:
                for role in ("negative", "positive"):
                    subset = [row for row in ablation_detection.rows() if row.get("variant") == variant and row.get("condition") == condition and row.get("truth_role") == role]
                    summaries[f"{variant}:{condition}:{role}"] = summarize_binary(
                        subset, truth_positive=role == "positive", planned=len(ablation_units)
                    )
        ablation_detection.finalize({
            "summaries": summaries,
            "status": "COMPLETE" if all(value["status"] == "COMPLETE" for value in summaries.values()) else "INCOMPLETE_OPERATIONAL",
        })

    clean_result = json.loads(clean_store.final_path.read_text(encoding="utf-8"))
    eval_result = json.loads(detection.final_path.read_text(encoding="utf-8"))
    ablation_result = json.loads(ablation_detection.final_path.read_text(encoding="utf-8"))
    quality = summarize_quality(generation.rows(), planned=EVALUATION_PAIRS)
    statuses = (
        clean_result["status"], eval_result["status"], ablation_result["status"],
        quality["status"],
    )
    _write_json_create_only(final_path, {
        "schema_version": "cegwm_formal_method_result_v1",
        "method_id": METHOD_ID,
        "producer_exact": expected_exact,
        "threshold": threshold,
        "clean_negative_test": clean_result["summary"],
        "evaluation": eval_result["summaries"],
        "ablations": ablation_result["summaries"],
        "quality": quality,
        "quality_source": "evaluation_generation_clean_pairs",
        "status": "COMPLETE" if all(value == "COMPLETE" for value in statuses) else "INCOMPLETE_OPERATIONAL",
        "fpr_policy": "report_only_nonblocking",
        "result_package_produced": True,
    })
    publish_job_state(
        job_root, _identity(job_id, "method_terminal", expected_exact),
        "TERMINAL_COMPLETE" if all(value == "COMPLETE" for value in statuses) else "TERMINAL_INCOMPLETE",
    )
    print(json.dumps({"method_id": METHOD_ID, "status": statuses, "terminal": True}, sort_keys=True))
    return 0


def run_engineering_canary(
    *, job_id: str, expected_exact: str, drive_root: Path, runtime_root: Path,
) -> int:
    """Exercise real generation, detection, checkpoint, and resume outside paper denominators."""

    verify_expected_exact(expected_exact)
    load_formal_config(CONFIG_PATH)
    root = drive_root / job_id
    final_path = root / "canary_final.json"
    if final_path.exists():
        value = json.loads(final_path.read_text(encoding="utf-8"))
        if value.get("producer_exact") != expected_exact or value.get("science_denominator") != 0:
            raise RuntimeError("main canary final identity differs")
        print(json.dumps({"status": value["status"], "terminal": True}, sort_keys=True))
        return 0 if value["status"] == "ENGINEERING_CANARY_COMPLETE" else 4
    try:
        runtime = execute_job_preflight(
            root,
            _identity(job_id, "engineering_canary_preflight", expected_exact),
            lambda: _prepare_runtime(runtime_root),
        )
    except PreflightFailed as error:
        print(json.dumps({"status": error.state["status"], "science_denominator": 0}, sort_keys=True))
        return 3

    clean_path = root / "images" / "clean.png"
    marked_path = root / "images" / "watermarked.png"
    generation_identity = _identity(job_id, "engineering_canary_generation", expected_exact)
    generation = FormalRunStore(root / "generation", generation_identity, ("engineering-canary-pair",))

    def generate(unit_id: str, attempt: int) -> dict[str, Any]:
        del unit_id, attempt
        clean, marked, pair_state = load_or_recover_pair(
            clean_path, marked_path,
            lambda: _main_pair(runtime, PREFLIGHT_PROMPT, PREFLIGHT_SEED),
            _load_rgb, _atomic_png_create_only,
        )
        return {
            "artifact_status": "GENERATED", "pair_state": pair_state,
            "quality": _quality(clean, marked),
        }

    generation.run(generate)
    generation_status = (
        "COMPLETE"
        if all(row["terminal_status"] == "SCORED" for row in generation.rows())
        else "INCOMPLETE_OPERATIONAL"
    )
    generation.finalize({"status": generation_status, "science_denominator": 0})
    resumed_generation = FormalRunStore(
        root / "generation", generation_identity, ("engineering-canary-pair",)
    )
    resumed_generation.run(lambda unit_id, attempt: (_ for _ in ()).throw(
        AssertionError("scored canary generation unit reran")
    ))

    detection_identity = _identity(job_id, "engineering_canary_detection", expected_exact)
    detection = FormalRunStore(
        root / "detection", detection_identity,
        ("engineering-canary-negative", "engineering-canary-positive"),
    )

    def detect(unit_id: str, attempt: int) -> dict[str, Any]:
        del attempt
        role = "negative" if unit_id.endswith("negative") else "positive"
        image = clean_path if role == "negative" else marked_path
        return {**_detect_payload(runtime, _load_rgb(image), 0.0), "truth_role": role}

    detection.run(detect)
    detection_status = (
        "COMPLETE"
        if all(row["terminal_status"] == "SCORED" for row in detection.rows())
        else "INCOMPLETE_OPERATIONAL"
    )
    detection.finalize({"status": detection_status, "science_denominator": 0})
    resumed_detection = FormalRunStore(
        root / "detection", detection_identity,
        ("engineering-canary-negative", "engineering-canary-positive"),
    )
    resumed_detection.run(lambda unit_id, attempt: (_ for _ in ()).throw(
        AssertionError("scored canary detection unit reran")
    ))
    checkpoint_count = sum(
        len(tuple((root / stage / "checkpoints").glob("checkpoint-*.json")))
        for stage in ("generation", "detection")
    )
    if checkpoint_count < 2:
        raise RuntimeError("main canary did not publish both stage checkpoints")
    canary_status = (
        "ENGINEERING_CANARY_COMPLETE"
        if generation_status == detection_status == "COMPLETE"
        else "ENGINEERING_CANARY_INCOMPLETE"
    )
    _write_json_create_only(final_path, {
        "schema_version": "cegwm_engineering_canary_result_v1",
        "method_id": METHOD_ID,
        "producer_exact": expected_exact,
        "status": canary_status,
        "science_denominator": 0,
        "generation_verified": generation_status == "COMPLETE",
        "detection_verified": detection_status == "COMPLETE",
        "checkpoint_count": checkpoint_count,
        "resume_verified": True,
    })
    publish_job_state(
        root, _identity(job_id, "engineering_canary_terminal", expected_exact),
        canary_status,
    )
    print(json.dumps({"status": canary_status, "terminal": True}, sort_keys=True))
    return 0 if canary_status == "ENGINEERING_CANARY_COMPLETE" else 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--engineering-canary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_formal_config(CONFIG_PATH)
    rosters = expand_rosters(REPO_ROOT, config)
    if args.validate_only:
        print(json.dumps({
            "status": "VALID",
            "counts": {name: len(units) for name, units in rosters.items()},
            "conditions": list(FORMAL_CONDITIONS),
            "ablation_subset": config["ablations"]["subset_size"],
            "ablation_variants": list(config["ablations"]["variants"]),
            "model_execution": False,
        }, sort_keys=True))
        return 0
    if args.engineering_canary:
        return run_engineering_canary(
            job_id=args.job_id, expected_exact=args.expected_exact,
            drive_root=Path(args.drive_root), runtime_root=Path(args.runtime_root),
        )
    return run_worker(
        job_id=args.job_id,
        expected_exact=args.expected_exact,
        drive_root=Path(args.drive_root),
        runtime_root=Path(args.runtime_root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
