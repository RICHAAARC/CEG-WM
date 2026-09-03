"""Formal Baseline-V1 worker for one fixed method and stable Drive JOB_ID.

This is the only model-backed baseline entry used by the paper notebooks.  It
does not share thresholds, inspect outcomes to choose retries, or implement an
FPR admission gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from cegwm.formal_experiment import (
    CLEAN_TEST_NEGATIVES,
    EVALUATION_PAIRS,
    FORMAL_CONDITIONS,
    FormalRunStore,
    OperationalUnitError,
    PreflightFailed,
    apply_attack,
    decide,
    empty_binary_summary,
    execute_job_preflight,
    expand_rosters,
    freeze_threshold,
    load_or_recover_pair,
    load_formal_config,
    publish_job_state,
    summarize_binary,
    summarize_quality,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/paper_experiment/formal_experiment_v1.json"
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
MODEL_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
PREFLIGHT_PROMPT = "A neutral geometric still life for an engineering preflight."
PREFLIGHT_SEED = 2027000000

METHOD_SPECS: dict[str, dict[str, Any]] = {
    "tree_ring": {
        "official_url": "https://github.com/YuxinWenRick/tree-ring-watermark.git",
        "official_exact": "3015283d9cf82e90b628f02ad2121bd37408ca9a",
        "score_id": "negative_fourier_key_l1_distance",
        "watermark_seed": 999999,
    },
    "gaussian_shading": {
        "official_url": "https://github.com/bsmhmmlf/Gaussian-Shading.git",
        "official_exact": "09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
        "score_id": "watermark_bit_accuracy",
        "watermark_seed": 20260622,
    },
    "shallow_diffuse": {
        "official_url": "https://github.com/liwd190019/Shallow-Diffuse.git",
        "official_exact": "c80c553fdf66fda8db735d77a9d56538b7a0ade8",
        "score_id": "negative_mask_l1diff_mean",
        "watermark_seed": 42,
    },
    "t2smark": {
        "official_url": "https://github.com/0xD009/T2SMark.git",
        "official_exact": "0c1fbfd50fcd1fba135477a2c016e284d5d7914d",
        "score_id": "norm1_w_master_key",
        "watermark_seed": 9173,
    },
}


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def verify_expected_exact(expected_exact: str) -> None:
    if _git_head() != expected_exact:
        raise RuntimeError("checked-out Baseline-V1 exact differs from notebook expectation")
    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("formal baseline worker requires a clean detached checkout")


def _atomic_png_create_only(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        image.convert("RGB").save(stream, format="PNG")
        stream.flush()
        os.fsync(stream.fileno())


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        result = image.convert("RGB").copy()
    if result.size != (512, 512):
        raise ValueError("formal baseline image dimensions differ")
    return result


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _ensure_official_source(runtime_root: Path, method: str) -> Path:
    spec = METHOD_SPECS[method]
    source = runtime_root / "official_source"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", spec["official_url"], str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", spec["official_exact"]], check=True)
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != spec["official_exact"] or dirty:
        raise RuntimeError("official baseline source exact or clean state differs")
    return source


class _ExternalRuntime:
    def __init__(self, method: str, hf_token: str) -> None:
        import torch
        from cegwm.baselines.external_sd35 import (
            GaussianShadingCarrier,
            ShallowDiffuseCarrier,
            TreeRingCarrier,
        )
        from cegwm.baselines.sd35_runtime import load_sd3_pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for formal baseline execution")
        self.torch = torch
        self.method = method
        self.pipeline = load_sd3_pipeline(MODEL_ID, MODEL_REVISION, token=hf_token)
        carrier_type = {
            "tree_ring": TreeRingCarrier,
            "gaussian_shading": GaussianShadingCarrier,
            "shallow_diffuse": ShallowDiffuseCarrier,
        }[method]
        self.carrier = carrier_type.fixed(
            seed=METHOD_SPECS[method]["watermark_seed"], device="cuda"
        )

    def _base(self, seed: int) -> Any:
        from cegwm.baselines.external_sd35 import SD35_SHAPE
        generator = self.torch.Generator("cuda").manual_seed(seed)
        return self.torch.randn(
            SD35_SHAPE, generator=generator, device="cuda", dtype=self.torch.float16
        )

    def plain(self, prompt: str, seed: int) -> Image.Image:
        base = self._base(seed)
        common = {
            "prompt": prompt, "height": 512, "width": 512,
            "guidance_scale": 4.5, "num_inference_steps": 20,
        }
        return self.pipeline(latents=base, **common).images[0].convert("RGB")

    def pair(self, prompt: str, seed: int) -> tuple[Image.Image, Image.Image]:
        base = self._base(seed)
        if self.method == "shallow_diffuse":
            edit_index = 16
            pre = self.pipeline.denoise_segment(
                base, prompt=prompt, guidance=4.5, steps=20, start=0, end=edit_index
            )
            marked_edit = self.carrier.inject(pre.clone())
            clean_latent = self.pipeline.denoise_segment(
                pre, prompt=prompt, guidance=1.0, steps=20, start=edit_index, end=20
            )
            marked_branch = self.pipeline.denoise_segment(
                marked_edit, prompt=prompt, guidance=1.0, steps=20, start=edit_index, end=20
            )
            marked_latent = clean_latent.clone()
            marked_latent[:, 0] = marked_branch[:, 0]
            return (
                self.pipeline.decode_latents(clean_latent).convert("RGB"),
                self.pipeline.decode_latents(marked_latent).convert("RGB"),
            )
        marked = (
            self.carrier.inject(base)
            if self.method == "tree_ring"
            else self.carrier.create_strict_paired_latents(base)
        )
        common = {
            "prompt": prompt, "height": 512, "width": 512,
            "guidance_scale": 4.5, "num_inference_steps": 20,
        }
        clean = self.pipeline(latents=base, **common).images[0].convert("RGB")
        watermarked = self.pipeline(latents=marked, **common).images[0].convert("RGB")
        return clean, watermarked

    def score(self, image: Image.Image) -> float:
        from cegwm.baselines.external_sd35 import score_rgb
        return float(score_rgb(
            np.asarray(image.convert("RGB"), dtype=np.uint8),
            self.pipeline, self.carrier, inversion_steps=20,
        ))


class _T2SRuntime:
    def __init__(self, source: Path, hf_token: str) -> None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for formal T2SMark execution")
        sys.path.insert(0, str(source))
        from src.inversion.inverse_diffusion3 import InversionDiffusion3Pipeline
        self.torch = torch
        self.pipeline = InversionDiffusion3Pipeline.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16,
            token=hf_token,
        ).to("cuda")
        self.pipeline.set_progress_bar_config(disable=True)
        keygen = torch.Generator("cuda").manual_seed(
            METHOD_SPECS["t2smark"]["watermark_seed"]
        )
        self.master_key = torch.randint(0, 2, (16,), generator=keygen, device="cuda")
        self.session_key = torch.randint(0, 2, (16,), generator=keygen, device="cuda")
        self.message = torch.randint(0, 2, (256,), generator=keygen, device="cuda")

    def _base(self, seed: int) -> Any:
        return self.torch.randn(
            (1, 16, 64, 64), generator=self.torch.Generator("cuda").manual_seed(seed),
            device="cuda", dtype=self.torch.float16,
        )

    def plain(self, prompt: str, seed: int) -> Image.Image:
        base = self._base(seed)
        return self.pipeline(
            prompt=prompt, latents=base, height=512, width=512,
            guidance_scale=4.0, num_inference_steps=40,
        ).images[0].convert("RGB")

    def pair(self, prompt: str, seed: int) -> tuple[Image.Image, Image.Image]:
        from cegwm.baselines.t2smark import embed_t2smark_sd35
        base = self._base(seed)
        marked = embed_t2smark_sd35(
            base, self.master_key, self.session_key, self.message
        )
        common = {
            "prompt": prompt, "height": 512, "width": 512,
            "guidance_scale": 4.0, "num_inference_steps": 40,
        }
        return (
            self.pipeline(latents=base, **common).images[0].convert("RGB"),
            self.pipeline(latents=marked, **common).images[0].convert("RGB"),
        )

    def score(self, image: Image.Image) -> float:
        from cegwm.baselines.t2smark import score_t2smark_rgb
        return float(score_t2smark_rgb(
            np.asarray(image.convert("RGB"), dtype=np.uint8),
            self.pipeline, self.master_key, 10,
        ))


def _build_runtime(method: str, runtime_root: Path, token: str) -> Any:
    try:
        if method == "t2smark":
            return _T2SRuntime(_ensure_official_source(runtime_root, method), token)
        _ensure_official_source(runtime_root, method)
        return _ExternalRuntime(method, token)
    except Exception as error:
        raise RuntimeError(f"baseline runtime construction failed: {type(error).__name__}: {error}") from error


def _identity(job_id: str, method: str, stage: str, exact: str) -> dict[str, str]:
    return {
        "schema_version": "cegwm_formal_job_v1",
        "job_id": job_id,
        "run_id": f"{job_id}:{method}:{stage}",
        "method_id": method,
        "stage": stage,
        "expected_exact": exact,
    }


def _score_payload(runtime: Any, image: Image.Image, tau: float | None = None) -> dict[str, Any]:
    score = runtime.score(image)
    if not math.isfinite(score):
        raise OperationalUnitError("MODEL_RUNTIME_TRANSIENT", "score", "nonfinite native score")
    payload: dict[str, Any] = {
        "native_score": score,
        "normalized_score": score,
        "score_id": METHOD_SPECS[runtime.method]["score_id"] if hasattr(runtime, "method") else METHOD_SPECS["t2smark"]["score_id"],
    }
    if tau is not None:
        payload["decision"] = decide(score, tau)
        payload["threshold"] = tau
    return payload


def _threshold_from_stage(stage: FormalRunStore, output: Path, method: str, exact: str) -> dict[str, Any] | None:
    if output.exists():
        value = json.loads(output.read_text(encoding="utf-8"))
        if value.get("method_id") != method or value.get("producer_exact") != exact:
            raise RuntimeError("formal threshold identity differs")
        return value
    rows = stage.rows()
    if len(rows) != 2000:
        return None
    if any(row["terminal_status"] != "SCORED" for row in rows):
        return None
    threshold = {
        "schema_version": "cegwm_paper_threshold_v1",
        "method_id": method,
        "producer_exact": exact,
        "score_id": METHOD_SPECS[method]["score_id"],
        **freeze_threshold([row["normalized_score"] for row in rows]),
    }
    _write_json_create_only(output, threshold)
    return threshold


_LPIPS_MODEL: Any | None = None


def _quality(clean: Image.Image, watermarked: Image.Image) -> dict[str, float]:
    import torch
    import lpips
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )
    global _LPIPS_MODEL
    first = torch.from_numpy(np.asarray(clean, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    second = torch.from_numpy(np.asarray(watermarked, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    device = torch.device("cuda")
    first, second = first.to(device), second.to(device)
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(device).eval()
    with torch.no_grad():
        return {
            "psnr": float(peak_signal_noise_ratio(second, first, data_range=1.0).item()),
            "ssim": float(structural_similarity_index_measure(second, first, data_range=1.0).item()),
            "lpips": float(_LPIPS_MODEL(second * 2.0 - 1.0, first * 2.0 - 1.0).reshape(-1)[0].item()),
        }


def _empty_evaluation() -> dict[str, Any]:
    return {
        f"{condition}:{role}": empty_binary_summary(
            truth_positive=role == "positive", planned=EVALUATION_PAIRS
        )
        for condition in FORMAL_CONDITIONS
        for role in ("negative", "positive")
    }


def _prepare_runtime(method: str, runtime_root: Path, token: str) -> Any:
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    runtime = _build_runtime(method, runtime_root, token)
    if not hasattr(runtime, "method"):
        runtime.method = "t2smark"
    clean, watermarked = runtime.pair(PREFLIGHT_PROMPT, PREFLIGHT_SEED)
    _score_payload(runtime, clean)
    _score_payload(runtime, watermarked, 0.0)
    _quality(clean, watermarked)
    return runtime


def run_worker(*, method: str, job_id: str, expected_exact: str, drive_root: Path, runtime_root: Path) -> int:
    verify_expected_exact(expected_exact)
    config = load_formal_config(CONFIG_PATH)
    rosters = expand_rosters(REPO_ROOT, config)
    job_root = drive_root / job_id
    final_path = job_root / "method_final.json"
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if final.get("method_id") != method or final.get("producer_exact") != expected_exact:
            raise RuntimeError("existing formal method final identity differs")
        print(json.dumps({"status": final["status"], "method_id": method, "terminal": True}, sort_keys=True))
        return 0

    token = os.environ.get("HF_TOKEN", "")
    try:
        runtime = execute_job_preflight(
            job_root,
            _identity(job_id, method, "preflight", expected_exact),
            lambda: _prepare_runtime(method, runtime_root, token),
        )
    except PreflightFailed as error:
        print(json.dumps({
            "method_id": method,
            "status": error.state["status"],
            "science_denominator": 0,
        }, sort_keys=True))
        return 3

    calibration_units = rosters["threshold_calibration"]
    calibration = FormalRunStore(
        job_root / "threshold_calibration",
        _identity(job_id, method, "threshold_calibration", expected_exact),
        [unit.unit_id for unit in calibration_units],
    )
    threshold_path = job_root / "threshold.json"
    threshold = _threshold_from_stage(calibration, threshold_path, method, expected_exact)
    def get_runtime() -> Any:
        return runtime

    if threshold is None and not calibration.completed_result():
        by_id = {unit.unit_id: unit for unit in calibration_units}
        calibration.run(lambda unit_id, attempt: _score_payload(
            get_runtime(), get_runtime().plain(by_id[unit_id].prompt, by_id[unit_id].seed)
        ))
        calibration.finalize({
            "status": "COMPLETE" if all(row["terminal_status"] == "SCORED" for row in calibration.rows()) else "INCOMPLETE_OPERATIONAL",
            "threshold_created": False,
        })
        threshold = _threshold_from_stage(calibration, threshold_path, method, expected_exact)

    if threshold is None:
        _write_json_create_only(final_path, {
            "schema_version": "cegwm_formal_method_result_v1",
            "method_id": method,
            "producer_exact": expected_exact,
            "threshold": None,
            "threshold_status": "INCOMPLETE_THRESHOLD",
            "clean_negative_test": empty_binary_summary(
                truth_positive=False, planned=CLEAN_TEST_NEGATIVES
            ),
            "evaluation": _empty_evaluation(),
            "quality": summarize_quality((), planned=EVALUATION_PAIRS),
            "status": "INCOMPLETE_OPERATIONAL",
            "reason": "paper threshold unavailable after terminal calibration",
            "fpr_policy": "report_only_nonblocking",
            "result_package_produced": True,
        })
        print(json.dumps({
            "method_id": method,
            "status": "INCOMPLETE_OPERATIONAL",
            "terminal": True,
        }, sort_keys=True))
        publish_job_state(
            job_root, _identity(job_id, method, "method_terminal", expected_exact),
            "TERMINAL_INCOMPLETE", reason="threshold unavailable",
        )
        return 0

    tau = float(threshold["tau"])
    clean_units = rosters["clean_negative_test"]
    clean_stage = FormalRunStore(
        job_root / "clean_negative_test",
        _identity(job_id, method, "clean_negative_test", expected_exact),
        [unit.unit_id for unit in clean_units],
    )
    if not clean_stage.completed_result():
        by_id = {unit.unit_id: unit for unit in clean_units}
        clean_stage.run(lambda unit_id, attempt: _score_payload(
            get_runtime(), get_runtime().plain(by_id[unit_id].prompt, by_id[unit_id].seed), tau
        ))
        clean_summary = summarize_binary(clean_stage.rows(), truth_positive=False, planned=CLEAN_TEST_NEGATIVES)
        clean_stage.finalize({"summary": clean_summary, "status": clean_summary["status"]})

    eval_units = rosters["formal_evaluation_pairs"]
    generation_stage = FormalRunStore(
        job_root / "evaluation_generation",
        _identity(job_id, method, "evaluation_generation", expected_exact),
        [unit.unit_id for unit in eval_units],
    )
    image_root = job_root / "evaluation_images"
    if not generation_stage.completed_result():
        by_id = {unit.unit_id: unit for unit in eval_units}

        def generate_pair(unit_id: str, attempt: int) -> dict[str, Any]:
            del attempt
            unit = by_id[unit_id]
            clean_path = image_root / f"{unit.roster_index:04d}" / "clean.png"
            watermarked_path = image_root / f"{unit.roster_index:04d}" / "watermarked.png"
            clean, watermarked, pair_state = load_or_recover_pair(
                clean_path,
                watermarked_path,
                lambda: get_runtime().pair(unit.prompt, unit.seed),
                _load_rgb,
                _atomic_png_create_only,
            )
            return {
                "artifact_status": "GENERATED",
                "clean_image": str(clean_path.relative_to(job_root)),
                "watermarked_image": str(watermarked_path.relative_to(job_root)),
                "pair_state": pair_state,
                "quality": _quality(clean, watermarked),
            }

        generation_stage.run(generate_pair)
        generation_stage.finalize({
            "status": "COMPLETE" if all(row["terminal_status"] == "SCORED" for row in generation_stage.rows()) else "INCOMPLETE_OPERATIONAL"
        })

    detection_items = [
        (unit, condition, role)
        for unit in eval_units
        for condition in FORMAL_CONDITIONS
        for role in ("negative", "positive")
    ]
    detection_ids = [f"{unit.unit_id}__{condition}__{role}" for unit, condition, role in detection_items]
    detection_stage = FormalRunStore(
        job_root / "evaluation_detection",
        _identity(job_id, method, "evaluation_detection", expected_exact),
        detection_ids,
    )
    if not detection_stage.completed_result():
        item_by_id = dict(zip(detection_ids, detection_items, strict=True))

        def detect_item(item_id: str, attempt: int) -> dict[str, Any]:
            del attempt
            unit, condition, role = item_by_id[item_id]
            source = image_root / f"{unit.roster_index:04d}" / ("watermarked.png" if role == "positive" else "clean.png")
            image = apply_attack(_load_rgb(source), condition)
            return {
                **_score_payload(get_runtime(), image, tau),
                "physical_unit_id": unit.unit_id,
                "condition": condition,
                "truth_role": role,
            }

        detection_stage.run(detect_item)
        summaries: dict[str, Any] = {}
        rows = detection_stage.rows()
        for condition in FORMAL_CONDITIONS:
            for role in ("negative", "positive"):
                subset = [row for row in rows if row.get("condition") == condition and row.get("truth_role") == role]
                summaries[f"{condition}:{role}"] = summarize_binary(
                    subset, truth_positive=role == "positive", planned=EVALUATION_PAIRS
                )
        detection_stage.finalize({
            "summaries": summaries,
            "status": "COMPLETE" if all(value["status"] == "COMPLETE" for value in summaries.values()) else "INCOMPLETE_OPERATIONAL",
        })

    clean_result = json.loads(clean_stage.final_path.read_text(encoding="utf-8"))
    eval_result = json.loads(detection_stage.final_path.read_text(encoding="utf-8"))
    quality = summarize_quality(generation_stage.rows(), planned=EVALUATION_PAIRS)
    statuses = (clean_result["status"], eval_result["status"], quality["status"])
    _write_json_create_only(final_path, {
        "schema_version": "cegwm_formal_method_result_v1",
        "method_id": method,
        "producer_exact": expected_exact,
        "threshold": threshold,
        "clean_negative_test": clean_result["summary"],
        "evaluation": eval_result["summaries"],
        "quality": quality,
        "quality_source": "evaluation_generation_clean_pairs",
        "status": "COMPLETE" if all(value == "COMPLETE" for value in statuses) else "INCOMPLETE_OPERATIONAL",
        "fpr_policy": "report_only_nonblocking",
        "result_package_produced": True,
    })
    publish_job_state(
        job_root, _identity(job_id, method, "method_terminal", expected_exact),
        "TERMINAL_COMPLETE" if all(value == "COMPLETE" for value in statuses) else "TERMINAL_INCOMPLETE",
    )
    print(json.dumps({"status": statuses, "method_id": method, "terminal": True}, sort_keys=True))
    return 0


def run_engineering_canary(
    *, method: str, job_id: str, expected_exact: str, drive_root: Path,
    runtime_root: Path,
) -> int:
    """Exercise one real baseline pair, detection, checkpoint, and resume at N=0 science."""

    verify_expected_exact(expected_exact)
    load_formal_config(CONFIG_PATH)
    root = drive_root / job_id
    final_path = root / "canary_final.json"
    if final_path.exists():
        value = json.loads(final_path.read_text(encoding="utf-8"))
        if (
            value.get("method_id") != method
            or value.get("producer_exact") != expected_exact
            or value.get("science_denominator") != 0
        ):
            raise RuntimeError("baseline canary final identity differs")
        print(json.dumps({"status": value["status"], "terminal": True}, sort_keys=True))
        return 0 if value["status"] == "ENGINEERING_CANARY_COMPLETE" else 4
    token = os.environ.get("HF_TOKEN", "")
    try:
        runtime = execute_job_preflight(
            root,
            _identity(job_id, method, "engineering_canary_preflight", expected_exact),
            lambda: _prepare_runtime(method, runtime_root, token),
        )
    except PreflightFailed as error:
        print(json.dumps({"status": error.state["status"], "science_denominator": 0}, sort_keys=True))
        return 3

    clean_path = root / "images" / "clean.png"
    marked_path = root / "images" / "watermarked.png"
    generation_identity = _identity(job_id, method, "engineering_canary_generation", expected_exact)
    generation = FormalRunStore(root / "generation", generation_identity, ("engineering-canary-pair",))

    def generate(unit_id: str, attempt: int) -> dict[str, Any]:
        del unit_id, attempt
        clean, marked, pair_state = load_or_recover_pair(
            clean_path, marked_path,
            lambda: runtime.pair(PREFLIGHT_PROMPT, PREFLIGHT_SEED),
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
    FormalRunStore(
        root / "generation", generation_identity, ("engineering-canary-pair",)
    ).run(lambda unit_id, attempt: (_ for _ in ()).throw(
        AssertionError("scored canary generation unit reran")
    ))

    detection_identity = _identity(job_id, method, "engineering_canary_detection", expected_exact)
    detection = FormalRunStore(
        root / "detection", detection_identity,
        ("engineering-canary-negative", "engineering-canary-positive"),
    )

    def detect(unit_id: str, attempt: int) -> dict[str, Any]:
        del attempt
        role = "negative" if unit_id.endswith("negative") else "positive"
        image = clean_path if role == "negative" else marked_path
        return {**_score_payload(runtime, _load_rgb(image), 0.0), "truth_role": role}

    detection.run(detect)
    detection_status = (
        "COMPLETE"
        if all(row["terminal_status"] == "SCORED" for row in detection.rows())
        else "INCOMPLETE_OPERATIONAL"
    )
    detection.finalize({"status": detection_status, "science_denominator": 0})
    FormalRunStore(
        root / "detection", detection_identity,
        ("engineering-canary-negative", "engineering-canary-positive"),
    ).run(lambda unit_id, attempt: (_ for _ in ()).throw(
        AssertionError("scored canary detection unit reran")
    ))
    checkpoint_count = sum(
        len(tuple((root / stage / "checkpoints").glob("checkpoint-*.json")))
        for stage in ("generation", "detection")
    )
    if checkpoint_count < 2:
        raise RuntimeError("baseline canary did not publish both stage checkpoints")
    canary_status = (
        "ENGINEERING_CANARY_COMPLETE"
        if generation_status == detection_status == "COMPLETE"
        else "ENGINEERING_CANARY_INCOMPLETE"
    )
    _write_json_create_only(final_path, {
        "schema_version": "cegwm_engineering_canary_result_v1",
        "method_id": method,
        "producer_exact": expected_exact,
        "status": canary_status,
        "science_denominator": 0,
        "generation_verified": generation_status == "COMPLETE",
        "detection_verified": detection_status == "COMPLETE",
        "checkpoint_count": checkpoint_count,
        "resume_verified": True,
    })
    publish_job_state(
        root, _identity(job_id, method, "engineering_canary_terminal", expected_exact),
        canary_status,
    )
    print(json.dumps({"status": canary_status, "terminal": True}, sort_keys=True))
    return 0 if canary_status == "ENGINEERING_CANARY_COMPLETE" else 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=tuple(METHOD_SPECS))
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
            "method": args.method,
            "counts": {name: len(units) for name, units in rosters.items()},
            "conditions": list(FORMAL_CONDITIONS),
            "model_execution": False,
        }, sort_keys=True))
        return 0
    if args.engineering_canary:
        return run_engineering_canary(
            method=args.method, job_id=args.job_id, expected_exact=args.expected_exact,
            drive_root=Path(args.drive_root), runtime_root=Path(args.runtime_root),
        )
    return run_worker(
        method=args.method,
        job_id=args.job_id,
        expected_exact=args.expected_exact,
        drive_root=Path(args.drive_root),
        runtime_root=Path(args.runtime_root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
