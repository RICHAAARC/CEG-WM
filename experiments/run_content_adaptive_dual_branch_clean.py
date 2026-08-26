"""Real clean Stage-A runner for content-adaptive dual-branch evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
import torch

from cegwm.method.content_adaptive import JOINT_EVALUATED_CANDIDATE_ID
from cegwm.method.hf import FrozenHFPublicAssets, score_hf_image
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
    score_lf_image,
)
from cegwm.protocol.content_chain import ContentChainProtocol, load_content_adaptive_dual_branch_clean_protocol
from cegwm.protocol.records import StageARecord
from cegwm.runtime.content_adaptive_sd35 import (
    ContentEmbedAssets,
    load_dino_content_assets,
    run_sd35_content_adaptive,
)
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline, run_sd35_plain
from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from cegwm.shared.prg import prg_bytes

KEY_ENV = "CEG_WM_ROOT_KEY"
TOKEN_ENV = "HF_TOKEN"
EXECUTION_SCOPE_ID = "content_adaptive_dual_branch_clean_engineering_and_stage_a_evaluation_v1"
COMPLETE_EXECUTION = "complete_for_content_adaptive_dual_branch_clean_evaluation"
INCOMPLETE_EXECUTION = "incomplete_operational_execution"
ARMS = (JOINT_EVALUATED_CANDIDATE_ID, f"primary_null__{JOINT_EVALUATED_CANDIDATE_ID}")
BRANCHES = ("lf", "hf", "joint")


def _git_exact(repo_root: Path, expected_exact: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected_exact) is None:
        raise ValueError("expected exact must be a lowercase 40-character revision")
    exact = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if exact != expected_exact:
        raise RuntimeError("resolved revision differs from approved execution exact")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout
    if status:
        raise RuntimeError("execution checkout must be clean")
    return exact


def _load_protocol(repo_root: Path) -> ContentChainProtocol:
    root = repo_root / "configs" / "content_chain"
    return load_content_adaptive_dual_branch_clean_protocol(
        root / "content_adaptive_dual_branch_clean_v1.json",
        root / "content_adaptive_dual_branch_clean.jsonl",
    )


def _load_pipeline_and_assets(model_id: str, token: str) -> tuple[Any, ContentEmbedAssets]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_real_content_adaptive_execution")
    pipeline = load_sd35_pipeline(model_id, torch_dtype=torch.float16, token=token)
    pipeline.to("cuda")
    vae = getattr(pipeline, "vae", None)
    processor = getattr(pipeline, "image_processor", None)
    hf = FrozenHFPublicAssets(
        vae=vae,
        image_processor=processor,
        image_processor_id=f"{model_id}:image_processor",
    )
    lf = FrozenLFPublicAssets(
        vae=vae,
        image_processor=processor,
        image_processor_id=f"{model_id}:image_processor",
        candidate_id=LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        detector_statistic_id=LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        evaluated_candidate_id=LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    dino_model, dino_processor = load_dino_content_assets(token=token)
    dino_model.to("cuda")
    dino_model.eval()
    return pipeline, ContentEmbedAssets(dino_model, dino_processor, hf, lf)


def _wrong_keys(key: bytes, protocol: ContentChainProtocol) -> tuple[bytes, ...]:
    keying = protocol.config["keying"]
    return tuple(
        prg_bytes(key, f"{keying['wrong_key_derivation_domain']}/index={index}", 32)
        for index in range(keying["wrong_key_count"])
    )


def _blind_scores(
    image: Any,
    key: bytes,
    wrong_keys: tuple[bytes, ...],
    hf_public_assets: FrozenHFPublicAssets,
    lf_public_assets: FrozenLFPublicAssets,
) -> dict[str, dict[str, float]]:
    ordinary_image = require_ordinary_rgb_image(image)
    if not isinstance(hf_public_assets, FrozenHFPublicAssets):
        raise TypeError("blind HF score requires FrozenHFPublicAssets")
    if not isinstance(lf_public_assets, FrozenLFPublicAssets):
        raise TypeError("blind LF score requires FrozenLFPublicAssets")
    if len(wrong_keys) != 16 or any(not isinstance(item, bytes) for item in wrong_keys):
        raise ValueError("blind score requires exactly 16 normalized external wrong keys")
    lf = {"registered": float(score_lf_image(ordinary_image, key, lf_public_assets))}
    hf = {"registered": float(score_hf_image(ordinary_image, key, hf_public_assets))}
    for index, wrong_key in enumerate(wrong_keys):
        label = f"wrong_{index:02d}"
        lf[label] = float(score_lf_image(ordinary_image, wrong_key, lf_public_assets))
        hf[label] = float(score_hf_image(ordinary_image, wrong_key, hf_public_assets))
    joint = {label: min(lf[label], hf[label]) for label in lf}
    values = {"lf": lf, "hf": hf, "joint": joint}
    if not all(math.isfinite(value) for branch in values.values() for value in branch.values()):
        raise ValueError("nonfinite_blind_score")
    return values


def _flat_scores(values: dict[str, dict[str, float]]) -> dict[str, float]:
    expected_labels = {"registered", *(f"wrong_{index:02d}" for index in range(16))}
    if set(values) != set(BRANCHES) or any(set(values[branch]) != expected_labels for branch in BRANCHES):
        raise ValueError("blind score fields differ from the fixed 3-by-17 roster")
    return {
        f"{branch}__{label}": float(values[branch][label])
        for branch in BRANCHES
        for label in ("registered", *(f"wrong_{index:02d}" for index in range(16)))
    }


def _psnr(first: Any, second: Any) -> float:
    first_pixels = np.asarray(first, dtype=np.float64) / 255.0
    second_pixels = np.asarray(second, dtype=np.float64) / 255.0
    if first_pixels.shape != second_pixels.shape:
        raise ValueError("paired_image_shape_mismatch")
    mse = float(np.mean(np.square(first_pixels - second_pixels)))
    if not math.isfinite(mse) or mse <= 0.0:
        raise ValueError("paired_psnr_requires_finite_nonidentical_images")
    value = -10.0 * math.log10(mse)
    if not math.isfinite(value):
        raise ValueError("paired_psnr_not_finite")
    return value


def _candidate_aggregate_metrics(
    unit_id: str,
    measurement: Any,
    paired_rgb_psnr_db: float,
    *,
    share_sum_absolute_tolerance: float,
) -> dict[str, Any]:
    """Pass through the fixed public aggregates and reject invalid identities."""

    lf_share = measurement.lf_branch_share
    hf_share = measurement.hf_branch_share
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in (lf_share, hf_share)
    ):
        raise ValueError("public branch shares must be finite real scalars")
    if not 0.0 < float(lf_share) < 1.0 or not 0.0 < float(hf_share) < 1.0:
        raise ValueError("public branch shares must be strictly between zero and one")
    if not math.isclose(
        float(lf_share) + float(hf_share),
        1.0,
        rel_tol=0.0,
        abs_tol=share_sum_absolute_tolerance,
    ):
        raise ValueError("public branch shares do not sum to one within the frozen tolerance")
    for name, value in (
        (
            "semantic_attention_counterfactual_effect",
            measurement.semantic_attention_counterfactual_effect,
        ),
        (
            "texture_energy_counterfactual_effect",
            measurement.texture_energy_counterfactual_effect,
        ),
        (
            "lf_probe_response_counterfactual_effect",
            measurement.lf_probe_response_counterfactual_effect,
        ),
        (
            "hf_probe_response_counterfactual_effect",
            measurement.hf_probe_response_counterfactual_effect,
        ),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be finite and strictly positive")
    return {
        "unit_id": unit_id,
        "combined_relative_l2": measurement.combined_budget.relative_l2,
        "lf_effective_relative_l2": measurement.lf_effective_relative_l2,
        "hf_effective_relative_l2": measurement.hf_effective_relative_l2,
        "lf_branch_share": lf_share,
        "hf_branch_share": hf_share,
        "semantic_attention_counterfactual_effect": (
            measurement.semantic_attention_counterfactual_effect
        ),
        "texture_energy_counterfactual_effect": measurement.texture_energy_counterfactual_effect,
        "lf_probe_response_counterfactual_effect": (
            measurement.lf_probe_response_counterfactual_effect
        ),
        "hf_probe_response_counterfactual_effect": (
            measurement.hf_probe_response_counterfactual_effect
        ),
        "minimum_counterfactual_effect": measurement.minimum_counterfactual_effect,
        "probe_evaluation_count": measurement.probe_evaluation_count,
        "paired_rgb_psnr_db": paired_rgb_psnr_db,
    }


def _branch_share_population_summary(
    unit_metrics: list[dict[str, Any]],
    expected_unit_ids: tuple[str, ...],
    *,
    rc: int,
    share_sum_absolute_tolerance: float,
    population_std_absolute_tolerance: float,
) -> tuple[float | None, float | None, bool, bool]:
    """Compute two independent fixed-roster ddof=0 summaries only for valid RC0."""

    unavailable = (None, None, False, False)
    if rc != 0 or len(unit_metrics) != 8 or len(expected_unit_ids) != 8:
        return unavailable
    try:
        received_unit_ids = tuple(metric["unit_id"] for metric in unit_metrics)
    except KeyError:
        return unavailable
    if received_unit_ids != expected_unit_ids:
        return unavailable
    try:
        lf_values = np.asarray([metric["lf_branch_share"] for metric in unit_metrics], dtype=np.float64)
        hf_values = np.asarray([metric["hf_branch_share"] for metric in unit_metrics], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return unavailable
    if (
        lf_values.shape != (8,)
        or hf_values.shape != (8,)
        or not np.all(np.isfinite(lf_values))
        or not np.all(np.isfinite(hf_values))
    ):
        return unavailable
    if (
        not np.all((0.0 < lf_values) & (lf_values < 1.0))
        or not np.all((0.0 < hf_values) & (hf_values < 1.0))
        or not np.allclose(
            lf_values + hf_values,
            np.ones(8, dtype=np.float64),
            rtol=0.0,
            atol=share_sum_absolute_tolerance,
        )
    ):
        return unavailable
    lf_mean = float(np.sum(lf_values) / 8.0)
    lf_population_std = math.sqrt(float(np.sum(np.square(lf_values - lf_mean)) / 8.0))
    hf_mean = float(np.sum(hf_values) / 8.0)
    hf_population_std = math.sqrt(float(np.sum(np.square(hf_values - hf_mean)) / 8.0))
    lf_reference = float(np.std(lf_values, ddof=0))
    hf_reference = float(np.std(hf_values, ddof=0))
    if not all(math.isfinite(value) for value in (lf_population_std, hf_population_std)):
        return unavailable
    if not (
        math.isclose(
            lf_population_std,
            lf_reference,
            rel_tol=0.0,
            abs_tol=population_std_absolute_tolerance,
        )
        and math.isclose(
            hf_population_std,
            hf_reference,
            rel_tol=0.0,
            abs_tol=population_std_absolute_tolerance,
        )
        and math.isclose(
            lf_population_std,
            hf_population_std,
            rel_tol=0.0,
            abs_tol=population_std_absolute_tolerance,
        )
    ):
        return unavailable
    supports_nonidentical = lf_population_std > 0.0 and hf_population_std > 0.0
    return lf_population_std, hf_population_std, supports_nonidentical, True


def _gate_evidence(records: list[dict[str, Any]], unit_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    by_unit: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_unit.setdefault(record["unit_id"], {})[record["arm"]] = record
    gates: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        gate_a = 0
        gate_b = 0
        for transaction in by_unit.values():
            joint = transaction[ARMS[0]]["scores"]
            primary_null = transaction[ARMS[1]]["scores"]
            registered = float(joint[f"{branch}__registered"])
            wrong = [float(joint[f"{branch}__wrong_{index:02d}"]) for index in range(16)]
            gate_a += int(registered > max(wrong))
            gate_b += int(registered > float(primary_null[f"{branch}__registered"]))
        gates[branch] = {
            "gate_a_pass_units": gate_a,
            "gate_b_pass_units": gate_b,
            "gate_a_pass": gate_a >= 7,
            "gate_b_pass": gate_b >= 7,
            "strict_ties_fail": True,
        }
    budget_units = sum(metric["combined_relative_l2"] <= 0.012 for metric in unit_metrics)
    nonzero_units = sum(
        metric["lf_effective_relative_l2"] > 0.0 and metric["hf_effective_relative_l2"] > 0.0
        for metric in unit_metrics
    )
    psnr_units = sum(metric["paired_rgb_psnr_db"] >= 30.0 for metric in unit_metrics)
    pass_all = (
        len(records) == 16
        and len(unit_metrics) == 8
        and all(item[gate] for item in gates.values() for gate in ("gate_a_pass", "gate_b_pass"))
        and budget_units == nonzero_units == psnr_units == 8
    )
    return {
        "branches": gates,
        "combined_budget_pass_units": budget_units,
        "both_nonzero_branches_pass_units": nonzero_units,
        "paired_rgb_psnr_pass_units": psnr_units,
        "all_predeclared_gates_pass": pass_all,
        "formal_fpr_claim": False,
    }


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()
    key_text = os.environ.get(KEY_ENV, "")
    token = os.environ.get(TOKEN_ENV, "")
    if not key_text.strip() or not token.strip():
        raise RuntimeError("CEG_WM_ROOT_KEY_and_HF_TOKEN_are_required")
    exact = _git_exact(repo_root, args.expected_exact)
    protocol = _load_protocol(repo_root)
    aggregate_contract = protocol.config["aggregate_measurement"]
    key = normalize_detection_key(key_text)
    key_digest = public_key_digest(key)
    run_id = f"content-adaptive-{protocol.protocol_digest[:12]}-{key_digest[:12]}"
    wrong_keys = _wrong_keys(key, protocol)
    pipeline, assets = _load_pipeline_and_assets(
        protocol.config["generation_runtime"]["model_id"], token
    )
    records: list[dict[str, Any]] = []
    unit_metrics: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for unit in protocol.roster:
        try:
            joint_generator = torch.Generator(device="cuda").manual_seed(unit.seed)
            null_generator = torch.Generator(device="cuda").manual_seed(unit.seed)
            output = run_sd35_content_adaptive(
                pipeline, unit.prompt, key, assets,
                height=unit.height, width=unit.width, generator=joint_generator,
            )
            primary_null = run_sd35_plain(
                pipeline, unit.prompt,
                height=unit.height, width=unit.width, generator=null_generator,
            )
            joint_scores = _blind_scores(
                output.image, key, wrong_keys,
                assets.hf_public_assets, assets.lf_public_assets,
            )
            null_scores = _blind_scores(
                primary_null, key, wrong_keys,
                assets.hf_public_assets, assets.lf_public_assets,
            )
            measurement = output.measurement
            metrics = _candidate_aggregate_metrics(
                unit.unit_id,
                measurement,
                _psnr(output.image, primary_null),
                share_sum_absolute_tolerance=aggregate_contract[
                    "branch_share_sum_absolute_tolerance"
                ],
            )
            transaction = [
                StageARecord(
                    run_id=run_id, unit_id=unit.unit_id, source_cluster_id=unit.source_id,
                    arm=ARMS[0], condition="clean", code_revision=exact,
                    config_digest=protocol.protocol_digest, key_public_digest=key_digest,
                    status="success", scores=_flat_scores(joint_scores),
                    metrics={key: float(value) for key, value in metrics.items() if key != "unit_id"},
                ).to_dict(),
                StageARecord(
                    run_id=run_id, unit_id=unit.unit_id, source_cluster_id=unit.source_id,
                    arm=ARMS[1], condition="clean", code_revision=exact,
                    config_digest=protocol.protocol_digest, key_public_digest=key_digest,
                    status="success", scores=_flat_scores(null_scores),
                    metrics={"paired_rgb_psnr_db": metrics["paired_rgb_psnr_db"]},
                ).to_dict(),
            ]
            unit_metrics.append(metrics)
            records.extend(transaction)
        except Exception as error:  # noqa: BLE001 - fixed-denominator unit failure is evidence
            error_type = type(error).__name__
            failures.append({"unit_id": unit.unit_id, "status": "failed", "error_type": error_type})
            for arm in ARMS:
                records.append(StageARecord(
                    run_id=run_id, unit_id=unit.unit_id, source_cluster_id=unit.source_id,
                    arm=arm, condition="clean", code_revision=exact,
                    config_digest=protocol.protocol_digest, key_public_digest=key_digest,
                    status="operational_failure", failure_reason=error_type,
                ).to_dict())
    rc = 0 if not failures and len(records) == 16 and len(unit_metrics) == 8 else 2
    lf_share_std, hf_share_std, supports_nonidentical, population_summary_valid = (
        _branch_share_population_summary(
            unit_metrics,
            tuple(unit.unit_id for unit in protocol.roster),
            rc=rc,
            share_sum_absolute_tolerance=aggregate_contract[
                "branch_share_sum_absolute_tolerance"
            ],
            population_std_absolute_tolerance=aggregate_contract[
                "population_std_absolute_tolerance"
            ],
        )
    )
    if rc == 0 and not population_summary_valid:
        rc = 1
        lf_share_std = hf_share_std = None
        supports_nonidentical = False
    gates = _gate_evidence(records, unit_metrics) if rc == 0 else None
    payload = {
        "rc": rc,
        "completeness": COMPLETE_EXECUTION if rc == 0 else INCOMPLETE_EXECUTION,
        "scientific_outcome_allowed": rc == 0,
        "scientific_status": "not_adjudicated" if rc == 0 else "not_evaluable",
        "execution_scope_id": EXECUTION_SCOPE_ID,
        "exact": exact,
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol.protocol_digest,
        "public_key_digest": key_digest,
        "fixed_denominator_units": 8,
        "fixed_records": 16,
        "lf_branch_share_population_std": lf_share_std,
        "hf_branch_share_population_std": hf_share_std,
        "fixed_roster_allocation_not_all_identical_supported": supports_nonidentical,
        "records": records,
        "unit_aggregate_metrics": unit_metrics,
        "failed_units": failures,
        "gate_evidence": gates,
        "limitations": list(protocol.config["limitations"]),
    }
    _write_result(output_path, payload)
    return rc


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
