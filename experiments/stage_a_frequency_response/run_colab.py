"""Explicit GPU runner for finite descriptive LF/HF frequency-response evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
import torch

from cegwm.method.hf import FrozenHFPublicAssets, HF_CANDIDATE_ID, score_hf_image
from cegwm.method.lf import (
    FrozenLFPublicAssets,
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    score_lf_image,
)
from cegwm.protocol.records import StageARecord
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline, run_sd35_hf, run_sd35_lf, run_sd35_plain
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from cegwm.shared.prg import prg_bytes

from experiments.stage_a_frequency_response.attack_transforms import apply_condition, public_noise_domain
from experiments.stage_a_frequency_response.protocol import (
    CONDITIONS,
    EVIDENCE_CONTRACT,
    HF_ARM,
    LF_ARM,
    RECORD_ARMS,
    FrequencyResponsePlan,
    FrequencyResponseUnit,
    expected_pairs,
    load_plan,
)

KEY_ENV = "CEG_WM_ROOT_KEY"
TOKEN_ENV = "HF_TOKEN"
_BUDGET_MAX = 0.012


def _git_exact(repo_root: Path, expected_exact: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected_exact) is None:
        raise ValueError("expected exact must be a lowercase 40-character revision")
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    if actual != expected_exact:
        raise RuntimeError("resolved revision differs from expected execution exact")
    if subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, check=True, capture_output=True, text=True).stdout:
        raise RuntimeError("execution checkout must be clean")
    return actual


def _load_pipeline_and_assets(model_id: str, hf_token: str) -> tuple[Any, FrozenHFPublicAssets, FrozenLFPublicAssets]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_frequency_response_execution")
    pipeline = load_sd35_pipeline(model_id, torch_dtype=torch.float16, token=hf_token)
    vae, image_processor = getattr(pipeline, "vae", None), getattr(pipeline, "image_processor", None)
    hf_assets = FrozenHFPublicAssets(vae=vae, image_processor=image_processor, image_processor_id=f"{model_id}:image_processor")
    lf_assets = FrozenLFPublicAssets(
        vae=vae, image_processor=image_processor, image_processor_id=f"{model_id}:image_processor",
        candidate_id=LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        detector_statistic_id=LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        evaluated_candidate_id=LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    pipeline.to("cuda")
    return pipeline, hf_assets, lf_assets


def _wrong_keys(detection_key: bytes) -> tuple[bytes, ...]:
    return tuple(prg_bytes(detection_key, f"stage-a/frequency-response/wrong-key/v1/index={index}", 32) for index in range(16))


def _scores(image: Any, detection_key: bytes, wrong_keys: tuple[bytes, ...], assets: FrozenHFPublicAssets | FrozenLFPublicAssets) -> dict[str, float]:
    scorer = score_hf_image if isinstance(assets, FrozenHFPublicAssets) else score_lf_image
    values = {"registered": float(scorer(image, detection_key, assets))}
    values.update({f"wrong_{index:02d}": float(scorer(image, key, assets)) for index, key in enumerate(wrong_keys)})
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("blind scores must be finite")
    return values


def _psnr(first: Any, second: Any) -> float | None:
    left, right = np.asarray(first, dtype=np.float64) / 255.0, np.asarray(second, dtype=np.float64) / 255.0
    if left.shape != right.shape:
        raise ValueError("ordinary RGB image shapes differ")
    mse = float(np.mean(np.square(left - right)))
    if not math.isfinite(mse):
        raise ValueError("ordinary RGB PSNR is nonfinite")
    return None if mse == 0.0 else -10.0 * math.log10(mse)


def _failure_transaction(unit: FrequencyResponseUnit, *, run_id: str, revision: str, plan: FrequencyResponsePlan, key_digest: str, reason: str) -> list[StageARecord]:
    return [StageARecord(run_id=run_id, unit_id=unit.unit_id, source_cluster_id=unit.source_id, arm=arm, condition=condition, code_revision=revision, config_digest=plan.config_digest, key_public_digest=key_digest, status="operational_failure", failure_reason=reason) for condition, arm in expected_pairs()]


def _unit_transaction(unit: FrequencyResponseUnit, *, pipeline: Any, detection_key: bytes, wrong_keys: tuple[bytes, ...], hf_assets: FrozenHFPublicAssets, lf_assets: FrozenLFPublicAssets, run_id: str, revision: str, plan: FrequencyResponsePlan, key_digest: str) -> list[StageARecord]:
    """Generate independently, then score only attacked ordinary RGB images."""

    hf = run_sd35_hf(pipeline, unit.prompt, detection_key, hf_assets, height=unit.height, width=unit.width, generator=torch.Generator(device="cuda").manual_seed(unit.seed))
    lf = run_sd35_lf(pipeline, unit.prompt, detection_key, lf_assets, height=unit.height, width=unit.width, generator=torch.Generator(device="cuda").manual_seed(unit.seed))
    plain = run_sd35_plain(pipeline, unit.prompt, height=unit.height, width=unit.width, generator=torch.Generator(device="cuda").manual_seed(unit.seed))
    hf_budget, lf_budget = float(hf.injection_budget.relative_l2), float(lf.injection_budget.relative_l2)
    if not all(math.isfinite(value) and 0.0 < value <= _BUDGET_MAX for value in (hf_budget, lf_budget)):
        raise ValueError("independent actual-callback-dtype relative L2 budget invalid")
    records: list[StageARecord] = []
    for condition in CONDITIONS:
        domain = public_noise_domain(protocol_id=plan.protocol_id, condition=condition, unit_id=unit.unit_id, source_id=unit.source_id, generation_seed=unit.seed, height=unit.height, width=unit.width) if condition.startswith("gaussian_noise_") else None
        hf_image, lf_image, plain_image = (apply_condition(image, condition, noise_domain=domain) for image in (hf.image, lf.image, plain))
        common = dict(run_id=run_id, unit_id=unit.unit_id, source_cluster_id=unit.source_id, condition=condition, code_revision=revision, config_digest=plan.config_digest, key_public_digest=key_digest, status="success")
        hf_metrics = {"actual_callback_dtype_relative_l2": hf_budget}
        lf_metrics = {"actual_callback_dtype_relative_l2": lf_budget}
        for metrics, method_image in ((hf_metrics, hf_image), (lf_metrics, lf_image)):
            effect = _psnr(method_image, plain_image)
            if effect is not None:
                metrics["candidate_vs_plain_psnr"] = effect
        records.extend((
            StageARecord(arm=HF_ARM, scores=_scores(hf_image, detection_key, wrong_keys, hf_assets), metrics=hf_metrics, **common),
            StageARecord(arm=f"primary_null__{HF_ARM}", scores=_scores(plain_image, detection_key, wrong_keys, hf_assets), **common),
            StageARecord(arm=LF_ARM, scores=_scores(lf_image, detection_key, wrong_keys, lf_assets), metrics=lf_metrics, **common),
            StageARecord(arm=f"primary_null__{LF_ARM}", scores=_scores(plain_image, detection_key, wrong_keys, lf_assets), **common),
        ))
    if [(record.condition, record.arm) for record in records] != list(expected_pairs()):
        raise RuntimeError("40-record atomic unit order differs")
    return records


def _median(values: list[float]) -> float | None:
    return None if not values else float(np.median(np.asarray(values, dtype=np.float64)))


def _descriptive_response(records: list[StageARecord]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Per-detector response/effect facts only; no cross-method conclusion is computed."""

    output: dict[str, dict[str, dict[str, float | int | None]]] = {"hf": {}, "lf": {}}
    for method, candidate_arm, null_arm in (("hf", HF_ARM, f"primary_null__{HF_ARM}"), ("lf", LF_ARM, f"primary_null__{LF_ARM}")):
        for condition in CONDITIONS:
            candidates = [record for record in records if record.condition == condition and record.arm == candidate_arm and record.status == "success"]
            nulls = [record for record in records if record.condition == condition and record.arm == null_arm and record.status == "success"]
            margins = [float(record.scores["registered"] - max(value for name, value in record.scores.items() if name.startswith("wrong_"))) for record in candidates]
            lifts = [float(candidate.scores["registered"] - null.scores["registered"]) for candidate, null in zip(candidates, nulls, strict=True)]
            output[method][condition] = {"successful_candidate_records": len(candidates), "median_registered_score": _median([float(record.scores["registered"]) for record in candidates]), "median_registered_minus_wrong_key_max": _median(margins), "median_candidate_minus_primary_null_registered": _median(lifts)}
    return output


def _run_id(revision: str, plan: FrequencyResponsePlan, key_digest: str) -> str:
    framed = json.dumps({"revision": revision, "config_digest": plan.config_digest, "key_public_digest": key_digest}, sort_keys=True, separators=(",", ":"))
    return "frequency-response-" + hashlib.sha256(framed.encode("utf-8")).hexdigest()[:20]


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    revision = _git_exact(repo_root, args.expected_exact)
    plan = load_plan(repo_root / "configs/stage_a_frequency_response/standalone_lf_hf_frequency_response_v1.json", repo_root / "configs/stage_a_frequency_response/standalone_lf_hf_frequency_response_v1.jsonl")
    raw_key, hf_token = os.environ.pop(KEY_ENV, None), os.environ.pop(TOKEN_ENV, None)
    if not isinstance(raw_key, str) or not raw_key.strip() or not isinstance(hf_token, str) or not hf_token.strip():
        raise RuntimeError("CEG_WM_ROOT_KEY and HF_TOKEN environment inputs are required")
    detection_key = normalize_detection_key(raw_key)
    del raw_key
    key_digest = public_key_digest(detection_key)
    run_id = _run_id(revision, plan, key_digest)
    try:
        pipeline, hf_assets, lf_assets = _load_pipeline_and_assets(plan.model_id, hf_token)
    except Exception:
        pipeline = hf_assets = lf_assets = None
        failed = True
    else:
        failed = False
    del hf_token
    wrong_keys = _wrong_keys(detection_key)
    records: list[StageARecord] = []
    for unit in plan.units:
        if pipeline is None:
            transaction = _failure_transaction(unit, run_id=run_id, revision=revision, plan=plan, key_digest=key_digest, reason="runtime_initialization_failure")
        else:
            try:
                transaction = _unit_transaction(unit, pipeline=pipeline, detection_key=detection_key, wrong_keys=wrong_keys, hf_assets=hf_assets, lf_assets=lf_assets, run_id=run_id, revision=revision, plan=plan, key_digest=key_digest)
            except Exception:
                failed = True
                transaction = _failure_transaction(unit, run_id=run_id, revision=revision, plan=plan, key_digest=key_digest, reason="unit_execution_failure")
        records.extend(transaction)
    del detection_key, wrong_keys
    if len(records) != 320 or [(record.condition, record.arm) for record in records[:40]] != list(expected_pairs()):
        raise RuntimeError("fixed 320-record export cannot be formed")
    rc = 2 if failed else 0
    result = {
        "evidence_contract": EVIDENCE_CONTRACT, "run_id": run_id, "resolved_exact": revision,
        "rc": rc, "complete": rc == 0, "protocol_id": plan.protocol_id, "protocol_digest": plan.config_digest,
        "key_public_digest": key_digest, "fixed_unit_count": 8, "fixed_condition_count": 10,
        "fixed_record_count": 320, "condition_order": list(CONDITIONS), "record_arms_in_exact_condition_order": list(RECORD_ARMS),
        "records": [record.to_dict() for record in records], "descriptive_per_method_response": _descriptive_response(records),
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "frequency_response_evidence.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    raise SystemExit(execute(_parser().parse_args()))


if __name__ == "__main__":
    main()
