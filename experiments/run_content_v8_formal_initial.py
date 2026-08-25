"""One-shot real Content V8 fit followed by two independent formal rosters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from experiments import run_content_adaptive_dual_branch_v2_clean as v2_runner
from cegwm.method.content_adaptive_v2 import COUNTERFACTUAL_EFFECT_FIELDS
from cegwm.method.content_iss_v8 import (
    CONTENT_V8_EVALUATED_CANDIDATE_ID,
    ISS_ASSET_ROLE_ID,
    ISSDevelopmentMeasurement,
    build_iss_asset,
    derive_development_key,
    derive_wrong_keys,
    fit_iss_gain_target,
    stable_json_bytes,
)
from cegwm.method.hf import score_hf_image
from cegwm.method.lf import score_lf_image
from cegwm.protocol.content_chain_v8 import (
    ContentV8Protocol,
    ContentV8Roster,
    ContentV8Unit,
    load_content_v8_protocol,
)
from cegwm.runtime.content_iss_sd35_v8 import (
    run_content_v8_development_pair,
    run_content_v8_evaluation_pair,
)
from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key, public_key_digest

KEY_ENV = "CEG_WM_ROOT_KEY"
TOKEN_ENV = "HF_TOKEN"
FIT_RECEIPT_PREFIX = "CEGWM_CONTENT_V8_RUNTIME_ASSET"
SUMMARY_PREFIX = "CEGWM_CONTENT_V8_FORMAL_SUMMARY"
ARMS = (
    CONTENT_V8_EVALUATED_CANDIDATE_ID,
    f"primary_null__{CONTENT_V8_EVALUATED_CANDIDATE_ID}",
)
BRANCHES = ("lf", "hf", "joint")
RECORD_CONTRACT_ID = "content_v8_v2_spatial_lf_iss_formal_record_v1"
ASSET_FILENAME = f"{ISS_ASSET_ROLE_ID}.json"
_PUBLIC_FAILURES = {
    "FileNotFoundError", "ImportError", "MemoryError", "OSError",
    "OutOfMemoryError", "RuntimeError", "TimeoutError", "TypeError", "ValueError",
}


def _git_exact(repo_root: Path, expected_exact: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected_exact) is None:
        raise ValueError("expected exact must be lowercase 40-hex")
    exact = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root,
        check=True, capture_output=True, text=True,
    ).stdout
    if exact != expected_exact or status:
        raise RuntimeError("Content V8 formal producer checkout identity differs")
    return exact


def _run_id(exact: str, protocol_digest: str) -> str:
    digest = hashlib.sha256(
        f"{exact}/{protocol_digest}/content-v8-formal-initial-v1".encode("ascii")
    ).hexdigest()[:12]
    return f"content-v8-{exact[:12]}-{digest}"


def _paths(artifact_sink: Path, run_id: str) -> tuple[Path, Path, Path, Path]:
    run_root = artifact_sink / run_id
    asset = run_root / "runtime_asset" / ASSET_FILENAME
    archive = run_root / "terminal" / f"{run_id}.zip"
    return run_root, asset, asset.with_name(f"{ASSET_FILENAME}.sha256"), archive


def _require_create_only(
    run_root: Path,
    asset_path: Path,
    sidecar_path: Path,
    archive_path: Path,
) -> None:
    if (
        run_root.exists()
        or asset_path.exists()
        or sidecar_path.exists()
        or archive_path.exists()
        or archive_path.with_name(f"{archive_path.name}.sha256").exists()
    ):
        raise FileExistsError("create-only Content V8 run-chain destination exists")


def _publish_runtime_asset(
    run_root: Path,
    asset_payload: bytes,
) -> tuple[Path, Path, str]:
    """Publish asset+sidecar together by one create-only directory rename."""

    digest = hashlib.sha256(asset_payload).hexdigest()
    run_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_root.name}.asset-", dir=run_root.parent))
    try:
        asset_dir = staging / "runtime_asset"
        asset_dir.mkdir()
        staged_asset = asset_dir / ASSET_FILENAME
        staged_sidecar = asset_dir / f"{ASSET_FILENAME}.sha256"
        with staged_asset.open("xb") as handle:
            handle.write(asset_payload)
        with staged_sidecar.open("xb") as handle:
            handle.write(f"{digest}  {ASSET_FILENAME}\n".encode("ascii"))
        if run_root.exists():
            raise FileExistsError("create-only Content V8 run chain appeared")
        staging.rename(run_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    asset_path = run_root / "runtime_asset" / ASSET_FILENAME
    return asset_path, asset_path.with_name(f"{ASSET_FILENAME}.sha256"), digest


def _load_pipeline_and_assets(token: str) -> tuple[Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_real_Content_V8_formal_initial")
    return v2_runner._load_pipeline_and_assets(
        "stabilityai/stable-diffusion-3.5-medium", token
    )


def _blind_scores(
    image: Any,
    key: bytes,
    wrong_keys: tuple[bytes, ...],
    assets: Any,
) -> dict[str, float]:
    ordinary = require_ordinary_rgb_image(image)
    labels = (("registered", key), *(
        (f"wrong_{index:02d}", wrong) for index, wrong in enumerate(wrong_keys)
    ))
    lf = {
        label: float(score_lf_image(ordinary, score_key, assets.lf_public_assets))
        for label, score_key in labels
    }
    hf = {
        label: float(score_hf_image(ordinary, score_key, assets.hf_public_assets))
        for label, score_key in labels
    }
    if len(lf) != 17 or len(hf) != 17:
        raise RuntimeError("Content V8 blind score denominator differs")
    scores = {
        f"{branch}__{label}": value
        for branch, values in (
            ("lf", lf),
            ("hf", hf),
            ("joint", {label: min(lf[label], hf[label]) for label in lf}),
        )
        for label, value in values.items()
    }
    if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in scores.values()):
        raise ValueError("Content V8 blind scores must be finite in [-1, 1]")
    return scores


def _psnr(first: Any, second: Any) -> float:
    left = np.asarray(first, dtype=np.float64) / 255.0
    right = np.asarray(second, dtype=np.float64) / 255.0
    if left.shape != right.shape:
        raise ValueError("Content V8 paired image shape differs")
    mse = float(np.mean(np.square(left - right)))
    if not math.isfinite(mse) or mse <= 0.0:
        raise ValueError("Content V8 PSNR requires finite nonidentical images")
    value = -10.0 * math.log10(mse)
    if not math.isfinite(value):
        raise ValueError("Content V8 PSNR is nonfinite")
    return value


def _metrics(measurement: Any, psnr_db: float) -> dict[str, float]:
    values = {
        "combined_relative_l2": float(measurement.combined_budget.relative_l2),
        "lf_effective_relative_l2": float(measurement.lf_effective_relative_l2),
        "hf_effective_relative_l2": float(measurement.hf_effective_relative_l2),
        "probe_evaluation_count": float(measurement.probe_evaluation_count),
        "paired_rgb_psnr_db": float(psnr_db),
    }
    if (
        not all(math.isfinite(value) for value in values.values())
        or not 0.0 < values["combined_relative_l2"] <= 0.012
        or not 0.0 < values["lf_effective_relative_l2"] <= 0.012
        or not 0.0 < values["hf_effective_relative_l2"] <= 0.012
        or values["probe_evaluation_count"] != 64.0
        or values["paired_rgb_psnr_db"] < 0.0
    ):
        raise ValueError("Content V8 public measurements differ")
    # The six V2 effects are validated but deliberately not exported.
    for name in COUNTERFACTUAL_EFFECT_FIELDS:
        effect = getattr(measurement, name)
        if (
            not isinstance(effect, (int, float))
            or isinstance(effect, bool)
            or not math.isfinite(float(effect))
            or float(effect) < 0.0
        ):
            raise ValueError("Content V8 V2 counterfactual measurement differs")
    return values


def _record(
    *,
    run_id: str,
    roster_role: str,
    unit: ContentV8Unit,
    arm: str,
    exact: str,
    protocol_digest: str,
    key_digest: str,
    status: str,
    failure_reason: str | None = None,
    scores: Mapping[str, float] | None = None,
    metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    record = {
        "run_id": run_id,
        "roster_role": roster_role,
        "unit_id": unit.unit_id,
        "source_cluster_id": unit.source_id,
        "arm": arm,
        "condition": "clean",
        "code_revision": exact,
        "config_digest": protocol_digest,
        "key_public_digest": key_digest,
        "status": status,
        "failure_reason": failure_reason,
        "scores": dict(scores or {}),
        "metrics": dict(metrics or {}),
        "record_contract_id": RECORD_CONTRACT_ID,
    }
    if arm not in ARMS or status not in {"success", "operational_failure"}:
        raise ValueError("Content V8 record arm or status differs")
    if status == "operational_failure":
        if failure_reason is None or record["scores"] or record["metrics"]:
            raise ValueError("Content V8 failure record differs")
    else:
        if failure_reason is not None or len(record["scores"]) != 51:
            raise ValueError("Content V8 successful record differs")
    return record


def _failure_class(error: Exception) -> str:
    return type(error).__name__ if type(error).__name__ in _PUBLIC_FAILURES else "OtherOperationalError"


def _unit_transaction(
    *,
    pipeline: Any,
    assets: Any,
    iss_asset: Any,
    unit: ContentV8Unit,
    key: bytes,
    wrong_keys: tuple[bytes, ...],
    identity: Mapping[str, str],
    roster_role: str,
) -> list[dict[str, Any]]:
    output = run_content_v8_evaluation_pair(
        pipeline, unit, key, assets, iss_asset
    )
    psnr_db = _psnr(output.image, output.primary_null)
    candidate_metrics = _metrics(output.measurement, psnr_db)
    return [
        _record(
            run_id=identity["run_id"], roster_role=roster_role, unit=unit,
            arm=ARMS[0], exact=identity["exact"],
            protocol_digest=identity["protocol_digest"],
            key_digest=identity["key_digest"], status="success",
            scores=_blind_scores(output.image, key, wrong_keys, assets),
            metrics=candidate_metrics,
        ),
        _record(
            run_id=identity["run_id"], roster_role=roster_role, unit=unit,
            arm=ARMS[1], exact=identity["exact"],
            protocol_digest=identity["protocol_digest"],
            key_digest=identity["key_digest"], status="success",
            scores=_blind_scores(output.primary_null, key, wrong_keys, assets),
            metrics={"paired_rgb_psnr_db": psnr_db},
        ),
    ]


def _gate_evidence(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_unit: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_unit.setdefault(record["unit_id"], {})[record["arm"]] = record
    branches: dict[str, Any] = {}
    for branch in BRANCHES:
        rank = 0
        paired = 0
        for transaction in by_unit.values():
            candidate = transaction[ARMS[0]]["scores"]
            null = transaction[ARMS[1]]["scores"]
            registered = candidate[f"{branch}__registered"]
            wrong = [candidate[f"{branch}__wrong_{index:02d}"] for index in range(16)]
            rank += int(registered > max(wrong))
            paired += int(registered > null[f"{branch}__registered"])
        branches[branch] = {
            "registered_top_rank_pass_units": rank,
            "registered_write_gt_primary_null_pass_units": paired,
            "registered_top_rank_gate_pass": rank >= 7,
            "registered_write_gt_primary_null_gate_pass": paired >= 7,
            "strict_ties_fail": True,
        }
    candidate_metrics = [
        record["metrics"] for record in records if record["arm"] == ARMS[0]
    ]
    budget = sum(item["combined_relative_l2"] <= 0.012 for item in candidate_metrics)
    nonzero = sum(
        item["lf_effective_relative_l2"] > 0.0
        and item["hf_effective_relative_l2"] > 0.0
        for item in candidate_metrics
    )
    probes = sum(item["probe_evaluation_count"] == 64.0 for item in candidate_metrics)
    psnr = sum(item["paired_rgb_psnr_db"] >= 30.0 for item in candidate_metrics)
    return {
        "branches": branches,
        "combined_budget_pass_units": budget,
        "both_nonzero_branches_pass_units": nonzero,
        "probe_evaluation_count_64_pass_units": probes,
        "paired_rgb_psnr_pass_units": psnr,
        "all_predeclared_gates_pass": (
            len(records) == 16
            and len(candidate_metrics) == 8
            and all(
                value[name]
                for value in branches.values()
                for name in (
                    "registered_top_rank_gate_pass",
                    "registered_write_gt_primary_null_gate_pass",
                )
            )
            and budget == nonzero == probes == psnr == 8
        ),
        "formal_fpr_claim": False,
    }


def _evaluate_roster(
    *,
    roster: ContentV8Roster,
    pipeline: Any,
    assets: Any,
    iss_asset: Any,
    key: bytes,
    wrong_keys: tuple[bytes, ...],
    identity: Mapping[str, str],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for unit in roster.units:
        try:
            records.extend(_unit_transaction(
                pipeline=pipeline, assets=assets, iss_asset=iss_asset,
                unit=unit, key=key, wrong_keys=wrong_keys,
                identity=identity, roster_role=roster.role,
            ))
        except Exception as error:
            reason = _failure_class(error)
            failures.append({"unit_id": unit.unit_id, "error_type": reason})
            records.extend(
                _record(
                    run_id=identity["run_id"], roster_role=roster.role, unit=unit,
                    arm=arm, exact=identity["exact"],
                    protocol_digest=identity["protocol_digest"],
                    key_digest=identity["key_digest"],
                    status="operational_failure", failure_reason=reason,
                )
                for arm in ARMS
            )
    rc = 0 if not failures and len(records) == 16 else 2
    return {
        "roster_role": roster.role,
        "manifest": roster.manifest,
        "manifest_sha256": roster.manifest_sha256,
        "rc": rc,
        "fixed_denominator_units": 8,
        "fixed_record_count": 16,
        "records": records,
        "failed_units": failures,
        "gate_evidence": _gate_evidence(records) if rc == 0 else None,
        "scientific_status": "not_adjudicated" if rc == 0 else "not_evaluable",
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _publish_terminal(
    archive_path: Path,
    *,
    receipt: Mapping[str, Any],
    result: Mapping[str, Any],
    asset_payload: bytes,
    asset_sidecar: bytes,
) -> str:
    terminal_root = archive_path.parent
    if terminal_root.exists():
        raise FileExistsError("create-only Content V8 terminal destination exists")
    staging = Path(tempfile.mkdtemp(prefix=".terminal-", dir=terminal_root.parent))
    temporary = staging / archive_path.name
    temporary_sidecar = staging / f"{archive_path.name}.sha256"
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_STORED) as archive:
            for name, payload in (
                ("receipt.json", _json_bytes(receipt)),
                ("result.json", _json_bytes(result)),
                (f"runtime_asset/{ASSET_FILENAME}", asset_payload),
                (f"runtime_asset/{ASSET_FILENAME}.sha256", asset_sidecar),
            ):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                archive.writestr(info, payload)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        with temporary_sidecar.open("xb") as handle:
            handle.write(f"{digest}  {archive_path.name}\n".encode("ascii"))
        if terminal_root.exists():
            raise FileExistsError("create-only Content V8 terminal appeared")
        staging.rename(terminal_root)
        return digest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    artifact_sink = Path(args.artifact_sink).resolve()
    exact = _git_exact(repo_root, args.expected_exact)
    protocol: ContentV8Protocol = load_content_v8_protocol(repo_root)
    run_id = _run_id(exact, protocol.protocol_digest)
    run_root, asset_path, sidecar_path, archive_path = _paths(artifact_sink, run_id)
    _require_create_only(run_root, asset_path, sidecar_path, archive_path)

    root_key_text = os.environ.pop(KEY_ENV, "")
    token = os.environ.pop(TOKEN_ENV, "")
    if not root_key_text:
        token = ""
        raise RuntimeError("CEG_WM_ROOT_KEY_is_required_for_Content_V8")
    formal_key = normalize_detection_key(root_key_text)
    development_key = derive_development_key(root_key_text)
    root_key_text = ""
    key_digest = public_key_digest(formal_key)
    if not token.strip():
        formal_key = development_key = b""
        raise RuntimeError("HF_TOKEN_is_required_for_Content_V8")
    try:
        pipeline, assets = _load_pipeline_and_assets(token)
    finally:
        token = ""

    measurements: list[ISSDevelopmentMeasurement] = []
    for unit in protocol.development:
        measurements.append(
            run_content_v8_development_pair(
                pipeline, unit, development_key, assets
            )
        )
    if len(measurements) != 32:
        raise RuntimeError("Content V8 fit did not produce exactly 32 measurements")
    fit = fit_iss_gain_target(measurements)
    measurements.clear()
    iss_asset = build_iss_asset(
        exact, protocol.protocol_digest, development_key, fit
    )
    development_key = b""
    asset_path, sidecar_path, asset_sha256 = _publish_runtime_asset(
        run_root, iss_asset.json_bytes
    )
    print(
        f"{FIT_RECEIPT_PREFIX} "
        + stable_json_bytes({
            "asset_sha256": asset_sha256,
            "fit_sample_count": 32,
            "producer_exact": exact,
            "run_id": run_id,
        }).decode("ascii"),
        flush=True,
    )

    wrong_keys = derive_wrong_keys(formal_key)
    identity = {
        "run_id": run_id,
        "exact": exact,
        "protocol_digest": protocol.protocol_digest,
        "key_digest": key_digest,
    }
    results = tuple(
        _evaluate_roster(
            roster=roster, pipeline=pipeline, assets=assets,
            iss_asset=iss_asset, key=formal_key, wrong_keys=wrong_keys,
            identity=identity,
        )
        for roster in protocol.evaluation_rosters
    )
    formal_key = b""
    result = {
        "run_id": run_id,
        "exact": exact,
        "execution_scope_id": protocol.execution_scope_id,
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol.protocol_digest,
        "public_key_digest": key_digest,
        "runtime_asset_sha256": asset_sha256,
        "evaluation_results_in_order": list(results),
        "cross_roster_pooling": False,
        "cross_roster_outcome_control": False,
        "scientific_status": "not_adjudicated",
        "limitations": list(protocol.config["limitations"]),
    }
    receipt = {
        "artifact_kind": "terminal",
        "run_id": run_id,
        "exact": exact,
        "protocol_digest": protocol.protocol_digest,
        "runtime_asset_member": f"runtime_asset/{ASSET_FILENAME}",
        "runtime_asset_sha256": asset_sha256,
        "result_member": "result.json",
        "evaluation_roster_roles_in_order": [
            roster.role for roster in protocol.evaluation_rosters
        ],
        "external_validation_required": True,
    }
    terminal_sha256 = _publish_terminal(
        archive_path,
        receipt=receipt,
        result=result,
        asset_payload=asset_path.read_bytes(),
        asset_sidecar=sidecar_path.read_bytes(),
    )
    rc = 0 if all(item["rc"] == 0 for item in results) else 2
    print(
        f"{SUMMARY_PREFIX} "
        + stable_json_bytes({
            "run_id": run_id,
            "terminal_sha256": terminal_sha256,
            "evaluation_rc_in_order": [item["rc"] for item in results],
            "rc": rc,
        }).decode("ascii"),
        flush=True,
    )
    return rc


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--artifact-sink", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
