"""Resumable GPU runner for finite descriptive LF/HF frequency response."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any
import zipfile

import numpy as np
import torch

from cegwm.method.hf import FrozenHFPublicAssets, score_hf_image
from cegwm.method.lf import FrozenLFPublicAssets, score_lf_image
from cegwm.protocol.records import StageARecord
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline, run_sd35_hf, run_sd35_lf, run_sd35_plain
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from cegwm.shared.prg import prg_bytes
from experiments.stage_a_frequency_response.attack_transforms import apply_condition, public_noise_domain
from experiments.stage_a_frequency_response.protocol import (
    CONDITIONS, EVIDENCE_CONTRACT, HF_ARM, LF_ARM, RECORD_ARMS,
    FrequencyResponsePlan, FrequencyResponseUnit, expected_pairs, load_plan,
)

KEY_ENV = "CEG_WM_ROOT_KEY"
TOKEN_ENV = "HF_TOKEN"
CHECKPOINT_INTERVAL_HOURS = 2.0
CHECKPOINT_SCHEMA = "standalone-lf-hf-frequency-response-checkpoint-v1"
UNIT_TRANSACTION_RECORD_COUNT = 40
FIXED_RECORD_COUNT = 320
_BUDGET_MAX = 0.012
_IDENTITY_FIELDS = (
    "checkpoint_schema", "run_id", "resolved_exact", "protocol_digest",
    "roster_digest", "model_id", "method_identities", "ordered_unit_ids",
    "ordered_source_ids", "condition_order", "record_arm_order",
    "unit_transaction_record_count", "fixed_record_count", "fixed_unit_count",
    "fixed_condition_count", "key_public_digest", "checkpoint_interval_hours",
)
_STATE_KEYS = {
    *_IDENTITY_FIELDS, "checkpoint_sequence", "committed_unit_count",
    "committed_unit_ids", "records",
}
_LF_IDENTITY_KEYS = {
    "carrier_method_id", "detector_statistic_id", "evaluated_candidate_id",
}
_SCORE_KEYS = {"registered", *(f"wrong_{index:02d}" for index in range(16))}
_CANDIDATE_METRIC_KEYS = {
    "actual_callback_dtype_relative_l2", "candidate_vs_plain_psnr",
}
_UNIT_FAILURE_REASONS = {
    "runtime_initialization_failure", "unit_execution_failure",
}
_FAILURE_CLASSES = {"hugging_face_token_missing"}
_LIMITATIONS = [
    "descriptive_per_method_response_only",
    "no_calibrated_threshold_or_fixed_fpr_claim",
    "no_winner_complementarity_joint_content_gate_or_robustness_promotion",
    "ordinary_rgb_attacks_only",
]
_FINAL_RESULT_KEYS = _STATE_KEYS | {
    "evidence_contract", "result_kind", "rc", "complete", "status",
    "scientific_evaluation_allowed", "claim_ceiling", "limitations",
    "descriptive_per_method_response",
}
_FAILURE_RESULT_KEYS = _STATE_KEYS | {
    "evidence_contract", "result_kind", "error_class", "rc", "complete",
    "status", "scientific_evaluation_allowed", "claim_ceiling", "limitations",
}
_FINAL_RECEIPT_FIELDS = (
    "run_id", "resolved_exact", "protocol_digest", "roster_digest",
    "key_public_digest", "rc", "status", "result_kind",
    "committed_unit_count", "fixed_record_count", "claim_ceiling",
)
_FAILURE_RECEIPT_FIELDS = (
    "run_id", "resolved_exact", "protocol_digest", "roster_digest",
    "key_public_digest", "rc", "status", "result_kind", "error_class",
    "committed_unit_count", "fixed_record_count", "claim_ceiling",
)


def _exact_json(actual: Any, expected: Any) -> bool:
    return json.dumps(actual, sort_keys=True, separators=(",", ":")) == json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    )


def _git_exact(repo_root: Path, expected_exact: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected_exact) is None:
        raise ValueError("expected exact must be a lowercase 40-character revision")
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if actual != expected_exact:
        raise RuntimeError("resolved revision differs from expected execution exact")
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout:
        raise RuntimeError("execution checkout must be clean")
    return actual


def _load_pipeline_and_assets(
    model_id: str, hf_token: str, lf_method_identity: dict[str, str],
) -> tuple[Any, FrozenHFPublicAssets, FrozenLFPublicAssets]:
    if not isinstance(lf_method_identity, dict) or set(lf_method_identity) != _LF_IDENTITY_KEYS:
        raise ValueError("LF protocol-bound method identity fields differ")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_frequency_response_execution")
    pipeline = load_sd35_pipeline(model_id, torch_dtype=torch.float16, token=hf_token)
    vae = getattr(pipeline, "vae", None)
    processor = getattr(pipeline, "image_processor", None)
    hf_assets = FrozenHFPublicAssets(
        vae=vae, image_processor=processor,
        image_processor_id=f"{model_id}:image_processor",
    )
    lf_assets = FrozenLFPublicAssets(
        vae=vae, image_processor=processor,
        image_processor_id=f"{model_id}:image_processor",
        candidate_id=lf_method_identity["carrier_method_id"],
        detector_statistic_id=lf_method_identity["detector_statistic_id"],
        evaluated_candidate_id=lf_method_identity["evaluated_candidate_id"],
    )
    pipeline.to("cuda")
    return pipeline, hf_assets, lf_assets


def _wrong_keys(detection_key: bytes) -> tuple[bytes, ...]:
    return tuple(prg_bytes(
        detection_key, f"stage-a/frequency-response/wrong-key/v1/index={index}", 32,
    ) for index in range(16))


def _scores(image: Any, detection_key: bytes, wrong_keys: tuple[bytes, ...], assets: Any) -> dict[str, float]:
    scorer = score_hf_image if isinstance(assets, FrozenHFPublicAssets) else score_lf_image
    values = {"registered": float(scorer(image, detection_key, assets))}
    values.update({f"wrong_{index:02d}": float(scorer(image, key, assets)) for index, key in enumerate(wrong_keys)})
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("blind scores must be finite")
    return values


def _psnr(first: Any, second: Any) -> float | None:
    left = np.asarray(first, dtype=np.float64) / 255.0
    right = np.asarray(second, dtype=np.float64) / 255.0
    if left.shape != right.shape:
        raise ValueError("ordinary RGB image shapes differ")
    mse = float(np.mean(np.square(left - right)))
    if not math.isfinite(mse):
        raise ValueError("ordinary RGB PSNR is nonfinite")
    return None if mse == 0.0 else -10.0 * math.log10(mse)


def _failure_transaction(
    unit: FrequencyResponseUnit, *, run_id: str, revision: str,
    plan: FrequencyResponsePlan, key_digest: str, reason: str,
) -> list[StageARecord]:
    return [StageARecord(
        run_id=run_id, unit_id=unit.unit_id, source_cluster_id=unit.source_id,
        arm=arm, condition=condition, code_revision=revision,
        config_digest=plan.protocol_digest, key_public_digest=key_digest,
        status="operational_failure", failure_reason=reason,
    ) for condition, arm in expected_pairs()]


def _unit_transaction(
    unit: FrequencyResponseUnit, *, pipeline: Any, detection_key: bytes,
    wrong_keys: tuple[bytes, ...], hf_assets: FrozenHFPublicAssets,
    lf_assets: FrozenLFPublicAssets, run_id: str, revision: str,
    plan: FrequencyResponsePlan, key_digest: str,
) -> list[StageARecord]:
    """Generate independently, then score only attacked ordinary RGB images."""
    hf = run_sd35_hf(
        pipeline, unit.prompt, detection_key, hf_assets, height=unit.height,
        width=unit.width, generator=torch.Generator(device="cuda").manual_seed(unit.seed),
    )
    lf = run_sd35_lf(
        pipeline, unit.prompt, detection_key, lf_assets, height=unit.height,
        width=unit.width, generator=torch.Generator(device="cuda").manual_seed(unit.seed),
    )
    plain = run_sd35_plain(
        pipeline, unit.prompt, height=unit.height, width=unit.width,
        generator=torch.Generator(device="cuda").manual_seed(unit.seed),
    )
    hf_budget = float(hf.injection_budget.relative_l2)
    lf_budget = float(lf.injection_budget.relative_l2)
    if not all(math.isfinite(value) and 0.0 < value <= _BUDGET_MAX for value in (hf_budget, lf_budget)):
        raise ValueError("independent actual-callback-dtype relative L2 budget invalid")
    records: list[StageARecord] = []
    for condition in CONDITIONS:
        domain = public_noise_domain(
            protocol_id=plan.protocol_id, condition=condition, unit_id=unit.unit_id,
            source_id=unit.source_id, generation_seed=unit.seed,
            height=unit.height, width=unit.width,
        ) if condition.startswith("gaussian_noise_") else None
        hf_image, lf_image, plain_image = (
            apply_condition(image, condition, noise_domain=domain)
            for image in (hf.image, lf.image, plain)
        )
        common = dict(
            run_id=run_id, unit_id=unit.unit_id, source_cluster_id=unit.source_id,
            condition=condition, code_revision=revision,
            config_digest=plan.protocol_digest, key_public_digest=key_digest,
            status="success",
        )
        hf_metrics = {"actual_callback_dtype_relative_l2": hf_budget}
        lf_metrics = {"actual_callback_dtype_relative_l2": lf_budget}
        for metrics, image in ((hf_metrics, hf_image), (lf_metrics, lf_image)):
            effect = _psnr(image, plain_image)
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


def _descriptive_response(records: list[StageARecord]) -> dict[str, Any]:
    """Per-detector facts only; no winner, joint, or complementarity conclusion."""
    output: dict[str, Any] = {"hf": {}, "lf": {}}
    for method, candidate_arm, null_arm in (
        ("hf", HF_ARM, f"primary_null__{HF_ARM}"),
        ("lf", LF_ARM, f"primary_null__{LF_ARM}"),
    ):
        for condition in CONDITIONS:
            candidates = [r for r in records if r.condition == condition and r.arm == candidate_arm and r.status == "success"]
            nulls = [r for r in records if r.condition == condition and r.arm == null_arm and r.status == "success"]
            margins = [float(r.scores["registered"] - max(v for n, v in r.scores.items() if n.startswith("wrong_"))) for r in candidates]
            lifts = [float(a.scores["registered"] - b.scores["registered"]) for a, b in zip(candidates, nulls, strict=True)]
            output[method][condition] = {
                "successful_candidate_records": len(candidates),
                "median_registered_score": _median([float(r.scores["registered"]) for r in candidates]),
                "median_registered_minus_wrong_key_max": _median(margins),
                "median_candidate_minus_primary_null_registered": _median(lifts),
            }
    return output


def _run_identity(revision: str, plan: FrequencyResponsePlan, key_digest: str) -> dict[str, Any]:
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "resolved_exact": revision,
        "protocol_digest": plan.protocol_digest,
        "roster_digest": plan.roster_digest,
        "model_id": plan.model_id,
        "method_identities": plan.method_identities,
        "ordered_unit_ids": [unit.unit_id for unit in plan.units],
        "ordered_source_ids": [unit.source_id for unit in plan.units],
        "condition_order": list(CONDITIONS),
        "record_arm_order": list(RECORD_ARMS),
        "unit_transaction_record_count": UNIT_TRANSACTION_RECORD_COUNT,
        "fixed_record_count": FIXED_RECORD_COUNT,
        "fixed_unit_count": len(plan.units),
        "fixed_condition_count": len(CONDITIONS),
        "key_public_digest": key_digest,
        "checkpoint_interval_hours": CHECKPOINT_INTERVAL_HOURS,
    }


def _run_id(revision: str, plan: FrequencyResponsePlan, key_digest: str) -> str:
    canonical = json.dumps(_run_identity(revision, plan, key_digest), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(b"CEG-WM/frequency-response/resumable-run/v1\x00" + canonical.encode()).hexdigest()
    return "frequency-response-" + digest[:20]


def _new_state(revision: str, plan: FrequencyResponsePlan, key_digest: str) -> dict[str, Any]:
    identity = _run_identity(revision, plan, key_digest)
    return {
        **identity, "run_id": _run_id(revision, plan, key_digest),
        "checkpoint_sequence": 0, "committed_unit_count": 0,
        "committed_unit_ids": [], "records": [],
    }


def _validate_transaction(payloads: list[dict[str, Any]], expected: dict[str, Any], unit_index: int) -> None:
    if len(payloads) != UNIT_TRANSACTION_RECORD_COUNT:
        raise ValueError("committed unit must contain exactly 40 records")
    try:
        records = [StageARecord(**payload) for payload in payloads]
    except TypeError as error:
        raise ValueError("committed record schema differs") from error
    if [(r.condition, r.arm) for r in records] != list(expected_pairs()):
        raise ValueError("committed 40-record transaction order differs")
    if {r.status for r in records} not in ({"success"}, {"operational_failure"}):
        raise ValueError("committed transaction status is incomplete or mixed")
    for record in records:
        if (
            record.run_id != expected["run_id"]
            or record.unit_id != expected["ordered_unit_ids"][unit_index]
            or record.source_cluster_id != expected["ordered_source_ids"][unit_index]
            or record.code_revision != expected["resolved_exact"]
            or record.config_digest != expected["protocol_digest"]
            or record.key_public_digest != expected["key_public_digest"]
        ):
            raise ValueError("committed transaction identity differs")
    if records[0].status == "operational_failure":
        reasons = {record.failure_reason for record in records}
        if len(reasons) != 1 or not reasons.issubset(_UNIT_FAILURE_REASONS):
            raise ValueError("committed failure transaction reason differs")
        if any(record.scores or record.metrics for record in records):
            raise ValueError("committed failure transaction must not contain observations")
        return
    for record in records:
        if set(record.scores) != _SCORE_KEYS:
            raise ValueError("committed public score schema differs")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in record.scores.values()
        ):
            raise ValueError("committed public scores must be finite numbers")
        if record.arm in (HF_ARM, LF_ARM):
            metric_keys = set(record.metrics)
            if (
                not {"actual_callback_dtype_relative_l2"}.issubset(metric_keys)
                or not metric_keys.issubset(_CANDIDATE_METRIC_KEYS)
            ):
                raise ValueError("committed candidate metric schema differs")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in record.metrics.values()
            ):
                raise ValueError("committed candidate metrics must be finite numbers")
            budget = float(record.metrics["actual_callback_dtype_relative_l2"])
            if not 0.0 < budget <= _BUDGET_MAX:
                raise ValueError("committed candidate budget differs")
        elif record.metrics:
            raise ValueError("committed primary-null record must not contain metrics")


def _validate_state(state: Any, expected: dict[str, Any], *, checkpoint: bool = False) -> dict[str, Any]:
    if not isinstance(state, dict) or set(state) != _STATE_KEYS:
        raise ValueError("checkpoint state schema differs")
    if any(not _exact_json(state[field], expected[field]) for field in _IDENTITY_FIELDS):
        raise ValueError("resume identity differs")
    committed = state["committed_unit_ids"]
    records = state["records"]
    sequence = state["checkpoint_sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0 or (checkpoint and sequence < 1):
        raise ValueError("checkpoint sequence is invalid")
    if not isinstance(committed, list) or committed != expected["ordered_unit_ids"][:len(committed)]:
        raise ValueError("committed units must be an ordered roster prefix")
    committed_count = state["committed_unit_count"]
    if isinstance(committed_count, bool) or not isinstance(committed_count, int) or committed_count != len(committed):
        raise ValueError("committed unit count differs")
    if checkpoint and not committed:
        raise ValueError("complete checkpoint cannot be empty")
    if not isinstance(records, list) or len(records) != len(committed) * UNIT_TRANSACTION_RECORD_COUNT:
        raise ValueError("committed record count differs from 40-record transactions")
    for index in range(len(committed)):
        start = index * UNIT_TRANSACTION_RECORD_COUNT
        _validate_transaction(records[start:start + UNIT_TRANSACTION_RECORD_COUNT], expected, index)
    return state


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_checksum(zip_path: Path, checksum_path: Path) -> str:
    parts = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != zip_path.name:
        raise ValueError("artifact checksum file is malformed")
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if parts[0] != digest:
        raise ValueError("artifact checksum mismatch")
    return digest


def _zip_payload(zip_path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        if tuple(archive.namelist()) != names:
            raise ValueError("artifact ZIP members differ")
        return {name: json.loads(archive.read(name)) for name in names}


def _write_zip_pair(directory: Path, stem: str, payloads: dict[str, dict[str, Any]]) -> tuple[Path, Path]:
    zip_path = directory / f"{stem}.zip"
    checksum_path = directory / f"{stem}.zip.sha256"
    if zip_path.exists() or checksum_path.exists():
        if not (zip_path.is_file() and checksum_path.is_file()):
            raise ValueError("local artifact pair is incomplete")
        _verify_checksum(zip_path, checksum_path)
        if not _exact_json(_zip_payload(zip_path, tuple(payloads)), payloads):
            raise ValueError("local create-only artifact content differs")
        return zip_path, checksum_path
    with zip_path.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in payloads.items():
                member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = 0o600 << 16
                archive.writestr(member, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    with checksum_path.open("x", encoding="utf-8") as stream:
        stream.write(f"{digest}  {zip_path.name}\n")
    return zip_path, checksum_path


def _publish_pair_create_only(zip_path: Path, checksum_path: Path, sink: Path) -> None:
    _verify_checksum(zip_path, checksum_path)
    destinations = (sink / zip_path.name, sink / checksum_path.name)
    if any(path.exists() for path in destinations):
        raise RuntimeError("artifact sink refuses overwrite")
    created: list[Path] = []
    try:
        for source, destination in zip((zip_path, checksum_path), destinations, strict=True):
            with source.open("rb") as source_stream, destination.open("xb") as target:
                created.append(destination)
                shutil.copyfileobj(source_stream, target)
            if source.read_bytes() != destination.read_bytes():
                raise RuntimeError("artifact sink copy verification failed")
    except BaseException:
        for destination in reversed(created):
            destination.unlink(missing_ok=True)
        raise


def _resume_state(zip_path: Path, checksum_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    _verify_checksum(zip_path, checksum_path)
    return _validate_state(_zip_payload(zip_path, ("state.json",))["state.json"], expected, checkpoint=True)


def _state_extends(later: dict[str, Any], earlier: dict[str, Any]) -> bool:
    count = earlier["committed_unit_count"]
    record_count = count * UNIT_TRANSACTION_RECORD_COUNT
    return (
        later["committed_unit_count"] >= count
        and later["committed_unit_ids"][:count] == earlier["committed_unit_ids"]
        and later["records"][:record_count] == earlier["records"]
    )


def _discover_sink(run_store: Path, expected: dict[str, Any]) -> tuple[dict[str, Any] | None, tuple[str, Path, Path] | None]:
    checkpoint_pattern = re.compile(r"checkpoint-(\d{4})-units-(\d{4})\.zip")
    failure_pattern = re.compile(r"failure-[a-z][a-z0-9_]*\.zip")
    final_name = f"{expected['run_id']}.zip"
    zips: dict[str, Path] = {}
    checksums: dict[str, Path] = {}
    for path in run_store.iterdir():
        if not path.is_file():
            raise ValueError("artifact sink contains a non-file entry")
        if path.name.endswith(".zip.sha256"):
            checksums[path.name[:-7]] = path
        elif path.name.endswith(".zip"):
            zips[path.name] = path
        else:
            raise ValueError("artifact sink contains an unexpected artifact")
    if set(zips) != set(checksums):
        raise ValueError("artifact sink contains an orphan ZIP or checksum")
    checkpoints: list[tuple[int, int, dict[str, Any]]] = []
    terminals: list[tuple[str, Path, Path]] = []
    for name, zip_path in sorted(zips.items()):
        checksum_path = checksums[name]
        match = checkpoint_pattern.fullmatch(name)
        if match:
            state = _resume_state(zip_path, checksum_path, expected)
            sequence, count = int(match.group(1)), int(match.group(2))
            if state["checkpoint_sequence"] != sequence or state["committed_unit_count"] != count:
                raise ValueError("checkpoint filename and state differ")
            checkpoints.append((sequence, count, state))
        elif name == final_name:
            terminals.append(("final", zip_path, checksum_path))
        elif failure_pattern.fullmatch(name):
            terminals.append(("failure", zip_path, checksum_path))
        else:
            raise ValueError("artifact sink contains an unexpected artifact")
    ordered = sorted(checkpoints)
    if [seq for seq, _, _ in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("checkpoint sequence is nonmonotone")
    for (_, first_count, first), (_, second_count, second) in zip(ordered, ordered[1:]):
        if second_count <= first_count or not _state_extends(second, first):
            raise ValueError("checkpoint history diverges or is nonmonotone")
    if len(terminals) > 1:
        raise ValueError("multiple terminal artifact pairs exist")
    return (ordered[-1][2] if ordered else None), (terminals[0] if terminals else None)


def _validate_terminal_pair(terminal: tuple[str, Path, Path], expected: dict[str, Any], checkpoint: dict[str, Any] | None) -> int:
    kind, zip_path, checksum_path = terminal
    if kind not in {"final", "failure"}:
        raise ValueError("terminal artifact kind differs")
    _verify_checksum(zip_path, checksum_path)
    payloads = _zip_payload(zip_path, ("receipt.json", "result.json"))
    receipt, result = payloads["receipt.json"], payloads["result.json"]
    if kind == "final":
        if not isinstance(result, dict) or set(result) != _FINAL_RESULT_KEYS:
            raise ValueError("final terminal result schema differs")
        state = {key: result[key] for key in _STATE_KEYS}
        _validate_state(state, expected)
        if state["committed_unit_count"] != expected["fixed_unit_count"]:
            raise ValueError("final artifact is not a complete unit roster")
        records = [StageARecord(**payload) for payload in state["records"]]
        rc = 2 if any(record.status != "success" for record in records) else 0
        expected_result = _result_payload(state, records, rc)
        if not _exact_json(result, expected_result):
            raise ValueError("final terminal result differs from committed public records")
        if not _exact_json(receipt, _receipt_payload(expected_result, failure=False)):
            raise ValueError("final terminal receipt differs from result")
        if checkpoint and not _state_extends(state, checkpoint):
            raise ValueError("terminal artifact diverges from checkpoints")
    else:
        if not isinstance(result, dict) or set(result) != _FAILURE_RESULT_KEYS:
            raise ValueError("failure terminal result schema differs")
        error_class = result["error_class"]
        if error_class not in _FAILURE_CLASSES:
            raise ValueError("failure terminal class differs")
        if zip_path.name != f"failure-{error_class}.zip":
            raise ValueError("failure filename and error class differ")
        state = {key: result[key] for key in _STATE_KEYS}
        _validate_state(state, expected)
        expected_result = _failure_result_payload(state, error_class)
        if not _exact_json(result, expected_result):
            raise ValueError("failure terminal result differs from frozen contract")
        if not _exact_json(receipt, _receipt_payload(expected_result, failure=True)):
            raise ValueError("failure terminal receipt differs from result")
        if checkpoint and not _state_extends(state, checkpoint):
            raise ValueError("failure artifact diverges from checkpoints")
        rc = 2
    return rc


def _select_resume_state(local: dict[str, Any] | None, sink: dict[str, Any] | None) -> dict[str, Any] | None:
    if local is None:
        return sink
    if sink is None:
        if local["checkpoint_sequence"] != 0:
            raise ValueError("local checkpoint sequence has no sink history")
        return local
    if not (_state_extends(local, sink) or _state_extends(sink, local)):
        raise ValueError("local and sink committed histories diverge")
    if local["checkpoint_sequence"] > sink["checkpoint_sequence"]:
        raise ValueError("local checkpoint sequence leads sink history")
    if local["committed_unit_count"] != sink["committed_unit_count"]:
        return local if local["committed_unit_count"] > sink["committed_unit_count"] else sink
    if local["records"] != sink["records"]:
        raise ValueError("local and sink same-count histories diverge")
    return sink if sink["checkpoint_sequence"] > local["checkpoint_sequence"] else local


def _checkpoint(state: dict[str, Any], local_run: Path, run_store: Path) -> None:
    payload = dict(state)
    payload["checkpoint_sequence"] = int(state["checkpoint_sequence"]) + 1
    stem = f"checkpoint-{payload['checkpoint_sequence']:04d}-units-{payload['committed_unit_count']:04d}"
    zip_path, checksum_path = _write_zip_pair(local_run, stem, {"state.json": payload})
    _publish_pair_create_only(zip_path, checksum_path, run_store)
    _atomic_json_write(local_run / "state.json", payload)
    state.clear()
    state.update(payload)


def _result_payload(state: dict[str, Any], records: list[StageARecord], rc: int) -> dict[str, Any]:
    any_failure = any(record.status != "success" for record in records)
    return {
        **state,
        "evidence_contract": EVIDENCE_CONTRACT,
        "result_kind": "complete_fixed_denominator",
        "rc": rc,
        "complete": rc == 0,
        "status": "complete_with_operational_failures" if any_failure else "complete_for_descriptive_adjudication",
        "scientific_evaluation_allowed": rc == 0,
        "claim_ceiling": "descriptive_per_method_response_only",
        "limitations": list(_LIMITATIONS),
        "descriptive_per_method_response": _descriptive_response(records),
    }


def _failure_result_payload(state: dict[str, Any], error_class: str) -> dict[str, Any]:
    if error_class not in _FAILURE_CLASSES:
        raise ValueError("failure class is not frozen")
    return {
        **state,
        "evidence_contract": EVIDENCE_CONTRACT,
        "result_kind": "operational_failure_not_scientific",
        "error_class": error_class,
        "rc": 2,
        "complete": False,
        "status": "operational_failure",
        "scientific_evaluation_allowed": False,
        "claim_ceiling": "descriptive_per_method_response_only",
        "limitations": list(_LIMITATIONS),
    }


def _receipt_payload(result: dict[str, Any], *, failure: bool) -> dict[str, Any]:
    fields = _FAILURE_RECEIPT_FIELDS if failure else _FINAL_RECEIPT_FIELDS
    return {key: result[key] for key in fields}


def _publish_final(state: dict[str, Any], records: list[StageARecord], rc: int, local_run: Path, run_store: Path) -> tuple[Path, str]:
    result = _result_payload(state, records, rc)
    receipt = _receipt_payload(result, failure=False)
    zip_path, checksum_path = _write_zip_pair(
        local_run, state["run_id"], {"receipt.json": receipt, "result.json": result},
    )
    digest = _verify_checksum(zip_path, checksum_path)
    _publish_pair_create_only(zip_path, checksum_path, run_store)
    return run_store / zip_path.name, digest


def _publish_failure(
    state: dict[str, Any], error_class: str, local_run: Path, run_store: Path,
) -> tuple[Path, str]:
    result = _failure_result_payload(state, error_class)
    receipt = _receipt_payload(result, failure=True)
    zip_path, checksum_path = _write_zip_pair(
        local_run, f"failure-{error_class}",
        {"receipt.json": receipt, "result.json": result},
    )
    digest = _verify_checksum(zip_path, checksum_path)
    _publish_pair_create_only(zip_path, checksum_path, run_store)
    return run_store / zip_path.name, digest


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    revision = _git_exact(repo_root, args.expected_exact)
    plan = load_plan(
        repo_root / "configs/stage_a_frequency_response/standalone_lf_hf_frequency_response_v1.json",
        repo_root / "configs/stage_a_frequency_response/standalone_lf_hf_frequency_response_v1.jsonl",
    )
    raw_key = os.environ.pop(KEY_ENV, None)
    hf_token = os.environ.pop(TOKEN_ENV, None)
    if not isinstance(raw_key, str) or not raw_key.strip():
        raise RuntimeError("CEG_WM_ROOT_KEY environment input is required")
    detection_key = normalize_detection_key(raw_key)
    raw_key = ""
    del raw_key
    key_digest = public_key_digest(detection_key)
    expected = _new_state(revision, plan, key_digest)
    run_id = expected["run_id"]
    output_value = getattr(args, "output_root", None) or getattr(args, "output_dir", None)
    sink_value = getattr(args, "run_store_root", None)
    if output_value is None or sink_value is None:
        raise ValueError("output_root and run_store_root are required")
    output_root, sink_root = Path(output_value).resolve(), Path(sink_value).resolve()
    if output_root == sink_root:
        raise ValueError("local active-state root and artifact sink root must differ")
    output_root.mkdir(parents=True, exist_ok=True)
    sink_root.mkdir(parents=True, exist_ok=True)
    local_run, run_store = output_root / run_id, sink_root / run_id
    local_run.mkdir(exist_ok=True)
    run_store.mkdir(exist_ok=True)
    if not local_run.is_dir() or not run_store.is_dir():
        raise ValueError("local or sink run path is not a directory")

    sink_state, terminal = _discover_sink(run_store, expected)
    if terminal is not None:
        hf_token = ""
        del hf_token, detection_key
        rc = _validate_terminal_pair(terminal, expected, sink_state)
        print(json.dumps({"run_id": run_id, "status": "terminal_pair_present_for_external_validation", "rc": rc}), flush=True)
        return rc
    local_path = local_run / "state.json"
    local_state = None
    if local_path.exists():
        if not local_path.is_file():
            raise ValueError("local active state is not a file")
        local_state = _validate_state(json.loads(local_path.read_text(encoding="utf-8")), expected)
    state = _select_resume_state(local_state, sink_state) or expected
    _atomic_json_write(local_path, state)
    records = [StageARecord(**payload) for payload in state["records"]]
    committed_count = state["committed_unit_count"]
    pending = plan.units[committed_count:]
    if pending and (not isinstance(hf_token, str) or not hf_token.strip()):
        hf_token = ""
        del hf_token, detection_key
        failure_zip, digest = _publish_failure(
            state, "hugging_face_token_missing", local_run, run_store,
        )
        print(json.dumps({
            "run_id": run_id, "rc": 2, "status": "operational_failure",
            "error_class": "hugging_face_token_missing",
            "zip_path": str(failure_zip), "zip_sha256": digest,
        }), flush=True)
        return 2
    wrong_keys = _wrong_keys(detection_key) if pending else ()
    pipeline = hf_assets = lf_assets = None
    model_load_failed = False
    if pending:
        try:
            pipeline, hf_assets, lf_assets = _load_pipeline_and_assets(
                plan.model_id, hf_token, plan.method_identities["lf"],
            )
        except Exception:
            model_load_failed = True
    hf_token = ""
    del hf_token
    last_checkpoint_time = time.monotonic()
    new_units = 0
    for unit_index, unit in enumerate(pending, start=committed_count):
        if model_load_failed:
            transaction = _failure_transaction(
                unit, run_id=run_id, revision=revision, plan=plan,
                key_digest=key_digest, reason="runtime_initialization_failure",
            )
        else:
            try:
                transaction = _unit_transaction(
                    unit, pipeline=pipeline, detection_key=detection_key,
                    wrong_keys=wrong_keys, hf_assets=hf_assets, lf_assets=lf_assets,
                    run_id=run_id, revision=revision, plan=plan, key_digest=key_digest,
                )
            except Exception:
                transaction = _failure_transaction(
                    unit, run_id=run_id, revision=revision, plan=plan,
                    key_digest=key_digest, reason="unit_execution_failure",
                )
        payloads = [record.to_dict() for record in transaction]
        _validate_transaction(payloads, expected, unit_index)
        next_state = dict(state)
        next_state["committed_unit_ids"] = [*state["committed_unit_ids"], unit.unit_id]
        next_state["committed_unit_count"] = len(next_state["committed_unit_ids"])
        next_state["records"] = [*state["records"], *payloads]
        _validate_state(next_state, expected)
        _atomic_json_write(local_path, next_state)
        state.clear()
        state.update(next_state)
        records.extend(transaction)
        new_units += 1
        now = time.monotonic()
        if new_units > 0 and now - last_checkpoint_time >= CHECKPOINT_INTERVAL_HOURS * 3600.0:
            _checkpoint(state, local_run, run_store)
            last_checkpoint_time = now
            new_units = 0
        print(json.dumps({
            "run_id": run_id, "committed_unit_count": state["committed_unit_count"],
            "fixed_unit_count": state["fixed_unit_count"],
        }), flush=True)
    del detection_key, wrong_keys
    if len(records) != FIXED_RECORD_COUNT or state["committed_unit_count"] != len(plan.units):
        raise RuntimeError("fixed 320-record final cannot be formed")
    rc = 2 if any(record.status != "success" for record in records) else 0
    final_zip, digest = _publish_final(state, records, rc, local_run, run_store)
    print(json.dumps({
        "run_id": run_id, "resolved_exact": revision, "rc": rc,
        "status": "complete_fixed_denominator", "zip_path": str(final_zip),
        "zip_sha256": digest,
    }), flush=True)
    return rc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-store-root", required=True)
    return parser


def main() -> None:
    raise SystemExit(execute(_parser().parse_args()))


if __name__ == "__main__":
    main()
