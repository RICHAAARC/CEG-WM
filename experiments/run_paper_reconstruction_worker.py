"""Supplementary SDXL image-to-image reconstruction attack for the main method."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from cegwm.formal_experiment import (
    FormalRunStore,
    OperationalUnitError,
    empty_binary_summary,
    expand_rosters,
    load_formal_config,
    summarize_binary,
)
from cegwm.runtime.observation import require_ordinary_rgb_image
from experiments.run_paper_main_worker import (
    CONFIG_PATH,
    METHOD_ID,
    REPO_ROOT,
    _build_runtime,
    _detect_payload,
    _load_rgb,
)


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _verify_exact(expected: str) -> None:
    if _git_head() != expected:
        raise RuntimeError("reconstruction worker exact differs")
    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("reconstruction worker requires clean detached checkout")


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_png_create_only(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        require_ordinary_rgb_image(image).save(stream, format="PNG")
        stream.flush()
        os.fsync(stream.fileno())


def _identity(job_id: str, stage: str, exact: str) -> dict[str, str]:
    return {
        "schema_version": "cegwm_formal_job_v1",
        "job_id": job_id,
        "run_id": f"{job_id}:main_reconstruction:{stage}",
        "method_id": METHOD_ID,
        "stage": stage,
        "expected_exact": exact,
    }


def _load_reconstruction_pipeline(config: Mapping[str, Any]) -> Any:
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for reconstruction attack")
    attack = config["reconstruction"]
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        attack["model_id"], revision=attack["model_revision"],
        torch_dtype=torch.float16, token=os.environ.get("HF_TOKEN", ""),
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def run_worker(*, job_id: str, main_job_id: str, expected_exact: str, drive_root: Path, runtime_root: Path) -> int:
    _verify_exact(expected_exact)
    config = load_formal_config(CONFIG_PATH)
    rosters = expand_rosters(REPO_ROOT, config)
    attack = config["reconstruction"]
    if attack["resolved_pair_count"] != 100 or attack["fpr_resolution"] != 0.01:
        raise RuntimeError("reconstruction denominator or resolution differs")
    root = drive_root / "reconstruction" / job_id
    final_path = root / "reconstruction_final.json"
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        print(json.dumps({"status": final["status"], "terminal": True}, sort_keys=True))
        return 0

    main_root = drive_root / "main" / main_job_id
    threshold_path = main_root / "threshold.json"
    generation_final = main_root / "evaluation_generation" / "final_result.json"
    threshold: dict[str, Any] | None = None
    prerequisite_reason: str | None = None
    if not threshold_path.exists():
        prerequisite_reason = "main paper threshold is unavailable; reconstruction model was not loaded"
    else:
        threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
        if threshold.get("method_id") != METHOD_ID or threshold.get("producer_exact") != expected_exact:
            raise RuntimeError("reconstruction threshold identity differs")
    if prerequisite_reason is None and not generation_final.exists():
        prerequisite_reason = "formal evaluation generation result is unavailable; reconstruction model was not loaded"
    elif prerequisite_reason is None:
        generation_status = json.loads(generation_final.read_text(encoding="utf-8"))
        if generation_status.get("status") != "COMPLETE":
            prerequisite_reason = "formal evaluation image generation is incomplete; reconstruction model was not loaded"
    if prerequisite_reason is not None:
        _write_json_create_only(final_path, {
            "schema_version": "cegwm_reconstruction_supplement_v1",
            "method_id": METHOD_ID,
            "producer_exact": expected_exact,
            "threshold": threshold,
            "attack": attack,
            "summaries": {
                role: empty_binary_summary(truth_positive=role == "positive", planned=100)
                for role in ("negative", "positive")
            },
            "status": "INCOMPLETE_OPERATIONAL",
            "reason": prerequisite_reason,
            "model_loaded": False,
            "fpr_resolution": 0.01,
            "claim_ceiling": "supplementary_reconstruction_only_not_0.1_percent_attacked_fpr_validation",
            "result_package_produced": True,
        })
        print(json.dumps({"status": "INCOMPLETE_OPERATIONAL", "terminal": True}, sort_keys=True))
        return 0

    units = rosters["formal_evaluation_pairs"][:100]
    items = [(unit, role) for unit in units for role in ("negative", "positive")]
    ids = [f"{unit.unit_id}__{role}" for unit, role in items]
    generated_root = root / "images"
    generation = FormalRunStore(
        root / "generation", _identity(job_id, "generation", expected_exact), ids
    )
    if not generation.completed_result():
        item_by_id = dict(zip(ids, items, strict=True))
        pipeline: Any | None = None

        def reconstruct(item_id: str, attempt: int) -> dict[str, Any]:
            nonlocal pipeline
            unit, role = item_by_id[item_id]
            source = main_root / "evaluation_images" / f"{unit.roster_index:04d}" / ("watermarked.png" if role == "positive" else "clean.png")
            output = generated_root / f"{unit.roster_index:04d}" / f"{role}.png"
            if output.exists():
                _load_rgb(output)
            else:
                if pipeline is None:
                    pipeline = _load_reconstruction_pipeline(config)
                try:
                    import torch
                    result = pipeline(
                        prompt=attack["prompt"], image=_load_rgb(source),
                        strength=attack["strength"], guidance_scale=attack["guidance_scale"],
                        num_inference_steps=attack["num_inference_steps"],
                        generator=torch.Generator("cuda").manual_seed(unit.seed + (1 if role == "positive" else 0)),
                    ).images[0]
                except torch.cuda.OutOfMemoryError as error:
                    raise OperationalUnitError("CUDA_OOM_TRANSIENT", "reconstruction", str(error)) from error
                except RuntimeError as error:
                    raise OperationalUnitError("MODEL_RUNTIME_TRANSIENT", "reconstruction", str(error)) from error
                _write_png_create_only(output, require_ordinary_rgb_image(result))
            return {
                "artifact_status": "GENERATED", "physical_unit_id": unit.unit_id,
                "truth_role": role, "image": str(output.relative_to(root)),
            }

        generation.run(reconstruct)
        generation.finalize({
            "status": "COMPLETE" if all(row["terminal_status"] == "SCORED" for row in generation.rows()) else "INCOMPLETE_OPERATIONAL"
        })
        del pipeline
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    detection = FormalRunStore(
        root / "detection", _identity(job_id, "detection", expected_exact), ids
    )
    if not detection.completed_result():
        item_by_id = dict(zip(ids, items, strict=True))
        runtime: dict[str, Any] | None = None

        def detect(item_id: str, attempt: int) -> dict[str, Any]:
            nonlocal runtime
            del attempt
            unit, role = item_by_id[item_id]
            if runtime is None:
                runtime = _build_runtime(runtime_root)
            image = generated_root / f"{unit.roster_index:04d}" / f"{role}.png"
            return {
                **_detect_payload(runtime, _load_rgb(image), float(threshold["tau"])),
                "physical_unit_id": unit.unit_id,
                "truth_role": role,
            }

        detection.run(detect)
        summaries = {
            role: summarize_binary(
                [row for row in detection.rows() if row.get("truth_role") == role],
                truth_positive=role == "positive", planned=100,
            )
            for role in ("negative", "positive")
        }
        detection.finalize({
            "summaries": summaries,
            "status": "COMPLETE" if all(value["status"] == "COMPLETE" for value in summaries.values()) else "INCOMPLETE_OPERATIONAL",
        })

    detection_result = json.loads(detection.final_path.read_text(encoding="utf-8"))
    _write_json_create_only(final_path, {
        "schema_version": "cegwm_reconstruction_supplement_v1",
        "method_id": METHOD_ID,
        "producer_exact": expected_exact,
        "threshold": threshold,
        "attack": attack,
        "summaries": detection_result["summaries"],
        "status": detection_result["status"],
        "fpr_resolution": 0.01,
        "claim_ceiling": "supplementary_reconstruction_only_not_0.1_percent_attacked_fpr_validation",
        "result_package_produced": True,
    })
    print(json.dumps({"status": detection_result["status"], "terminal": True}, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--main-job-id", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_formal_config(CONFIG_PATH)
    if args.validate_only:
        attack = config["reconstruction"]
        print(json.dumps({
            "status": "VALID", "model_execution": False,
            "pair_count": attack["resolved_pair_count"],
            "fpr_resolution": attack["fpr_resolution"],
            "model_id": attack["model_id"], "model_revision": attack["model_revision"],
        }, sort_keys=True))
        return 0
    return run_worker(
        job_id=args.job_id, main_job_id=args.main_job_id,
        expected_exact=args.expected_exact, drive_root=Path(args.drive_root),
        runtime_root=Path(args.runtime_root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
