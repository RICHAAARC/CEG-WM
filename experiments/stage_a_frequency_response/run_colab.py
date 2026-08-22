"""Explicit GPU runner for finite descriptive LF/HF frequency-response evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
STATE_SCHEMA_ID = "standalone_frequency_response_resumable_state_v2"
CHECKPOINT_INTERVAL_HOURS = 2.0
RECORDS_PER_UNIT = 40
FIXED_UNIT_COUNT = 8
FIXED_RECORD_COUNT = 320
_CHECKPOINT_INTERVAL_SECONDS = CHECKPOINT_INTERVAL_HOURS * 60.0 * 60.0
_PUBLIC_FAILURE_CLASSES = {"runtime_initialization_failure", "unit_execution_failure"}
_RECORD_FIELDS = {
    "run_id", "unit_id", "source_cluster_id", "arm", "condition", "code_revision",
    "config_digest", "key_public_digest", "status", "failure_reason", "scores", "metrics",
    "schema_version",
}
_SCORE_FIELDS = {"registered", *(f"wrong_{index:02d}" for index in range(16))}


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


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _roster_digest(plan: FrequencyResponsePlan) -> str:
    return hashlib.sha256(_canonical_json_bytes([asdict(unit) for unit in plan.units])).hexdigest()


def _run_identity(revision: str, plan: FrequencyResponsePlan, key_digest: str) -> dict[str, object]:
    identity = {
        "state_schema_id": STATE_SCHEMA_ID,
        "resolved_exact": revision,
        "protocol_id": plan.protocol_id,
        "protocol_digest": plan.config_digest,
        "roster_digest": _roster_digest(plan),
        "ordered_unit_ids": [unit.unit_id for unit in plan.units],
        "key_public_digest": key_digest,
        "condition_order": list(CONDITIONS),
        "record_arms_in_exact_condition_order": list(RECORD_ARMS),
        "fixed_unit_count": FIXED_UNIT_COUNT,
        "records_per_unit": RECORDS_PER_UNIT,
        "fixed_record_count": FIXED_RECORD_COUNT,
    }
    identity["run_id"] = "slhfr-" + hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()[:24]
    return identity


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _record_from_payload(payload: object) -> StageARecord:
    if not isinstance(payload, dict) or set(payload) != _RECORD_FIELDS:
        raise ValueError("committed record fields differ")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("committed record schema differs")
    scores, metrics = payload.get("scores"), payload.get("metrics")
    if not isinstance(scores, dict) or not isinstance(metrics, dict):
        raise ValueError("committed score or metric map differs")
    if any(not isinstance(name, str) or not _finite_number(value) for name, value in scores.items()):
        raise ValueError("committed scores must be finite public numbers")
    if any(not isinstance(name, str) or not _finite_number(value) for name, value in metrics.items()):
        raise ValueError("committed metrics must be finite public numbers")
    return StageARecord(**payload)


def _validate_transaction(
    payloads: list[object], *, unit: FrequencyResponseUnit, identity: dict[str, object], plan: FrequencyResponsePlan,
) -> list[StageARecord]:
    if len(payloads) != RECORDS_PER_UNIT:
        raise ValueError("committed unit must contain exactly 40 records")
    records = [_record_from_payload(payload) for payload in payloads]
    if [(record.condition, record.arm) for record in records] != list(expected_pairs()):
        raise ValueError("committed unit record order differs")
    expected_common = {
        "run_id": identity["run_id"], "unit_id": unit.unit_id, "source_cluster_id": unit.source_id,
        "code_revision": identity["resolved_exact"], "config_digest": plan.config_digest,
        "key_public_digest": identity["key_public_digest"],
    }
    statuses = {record.status for record in records}
    if statuses == {"success"}:
        for record in records:
            if record.failure_reason is not None or set(record.scores) != _SCORE_FIELDS:
                raise ValueError("successful record score or failure shape differs")
            if record.arm in (HF_ARM, LF_ARM):
                if set(record.metrics) not in ({"actual_callback_dtype_relative_l2"}, {"actual_callback_dtype_relative_l2", "candidate_vs_plain_psnr"}):
                    raise ValueError("candidate metric fields differ")
                budget = record.metrics["actual_callback_dtype_relative_l2"]
                if not (0.0 < float(budget) <= _BUDGET_MAX):
                    raise ValueError("candidate budget differs")
            elif record.metrics:
                raise ValueError("primary-null metrics must be empty")
    elif statuses == {"operational_failure"}:
        for record in records:
            if record.failure_reason not in _PUBLIC_FAILURE_CLASSES or record.scores or record.metrics:
                raise ValueError("operational-failure record shape differs")
    else:
        raise ValueError("atomic unit mixes statuses")
    for record in records:
        for field, expected in expected_common.items():
            if getattr(record, field) != expected:
                raise ValueError(f"committed record {field} differs")
    return records


def _state_payload(
    identity: dict[str, object], records: list[StageARecord], *, checkpoint_sequence: int,
    checkpoint_anchor_unix_seconds: float,
) -> dict[str, object]:
    committed = len(records) // RECORDS_PER_UNIT
    return {
        **identity,
        "committed_unit_count": committed,
        "ordered_committed_unit_ids": list(identity["ordered_unit_ids"][:committed]),
        "checkpoint_sequence": checkpoint_sequence,
        "checkpoint_anchor_unix_seconds": checkpoint_anchor_unix_seconds,
        "records": [record.to_dict() for record in records],
    }


def _validate_state(payload: object, *, identity: dict[str, object], plan: FrequencyResponsePlan) -> tuple[list[StageARecord], int, float]:
    identity_fields = set(identity)
    expected_fields = identity_fields | {
        "committed_unit_count", "ordered_committed_unit_ids", "checkpoint_sequence",
        "checkpoint_anchor_unix_seconds", "records",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("resumable state fields differ")
    for name, expected in identity.items():
        if payload.get(name) != expected or type(payload.get(name)) is not type(expected):
            raise ValueError(f"resumable identity {name} differs")
    count, sequence = payload.get("committed_unit_count"), payload.get("checkpoint_sequence")
    if type(count) is not int or not 0 <= count <= FIXED_UNIT_COUNT:
        raise ValueError("committed unit count differs")
    if type(sequence) is not int or sequence < 0:
        raise ValueError("checkpoint sequence differs")
    anchor = payload.get("checkpoint_anchor_unix_seconds")
    if not _finite_number(anchor) or float(anchor) < 0.0:
        raise ValueError("checkpoint clock anchor differs")
    if payload.get("ordered_committed_unit_ids") != list(identity["ordered_unit_ids"][:count]):
        raise ValueError("committed unit prefix differs")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != count * RECORDS_PER_UNIT:
        raise ValueError("committed record count differs")
    records: list[StageARecord] = []
    for index in range(count):
        start = index * RECORDS_PER_UNIT
        records.extend(_validate_transaction(raw_records[start:start + RECORDS_PER_UNIT], unit=plan.units[index], identity=identity, plan=plan))
    return records, sequence, float(anchor)


def _atomic_json_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_zip(zip_path: Path, members: dict[str, object]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, _canonical_json_bytes(payload) + b"\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksum(path: Path, zip_path: Path, *, published_name: str | None = None) -> str:
    digest = _sha256_file(zip_path)
    path.write_text(f"{digest}  {published_name or zip_path.name}\n", encoding="utf-8")
    return digest


def _read_pair(zip_path: Path, checksum_path: Path) -> dict[str, object]:
    if zip_path.exists() != checksum_path.exists():
        raise RuntimeError("artifact sink contains an orphan package pair")
    declared = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(declared) != 2 or declared[1] != zip_path.name or declared[0] != _sha256_file(zip_path):
        raise RuntimeError("artifact package checksum differs")
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("artifact package contains duplicate members")
        return {name: json.loads(archive.read(name)) for name in names}


def _publish_pair(local_zip: Path, local_checksum: Path, sink_zip: Path, sink_checksum: Path) -> None:
    sink_zip.parent.mkdir(parents=True, exist_ok=True)
    if sink_zip.exists() or sink_checksum.exists():
        raise FileExistsError("artifact destination already exists")
    created: list[Path] = []
    try:
        for source, destination in ((local_zip, sink_zip), (local_checksum, sink_checksum)):
            with source.open("rb") as reader, destination.open("xb") as writer:
                created.append(destination)
                shutil.copyfileobj(reader, writer)
                writer.flush()
                os.fsync(writer.fileno())
    except BaseException:
        for destination in reversed(created):
            destination.unlink(missing_ok=True)
        raise


def _checkpoint_paths(run_sink: Path, run_id: str, sequence: int) -> tuple[Path, Path]:
    stem = f"{run_id}-checkpoint-{sequence:04d}"
    return run_sink / f"{stem}.zip", run_sink / f"{stem}.zip.sha256"


def _final_paths(run_sink: Path, run_id: str) -> tuple[Path, Path]:
    return run_sink / f"{run_id}.zip", run_sink / f"{run_id}.zip.sha256"


def _sink_checkpoints(run_sink: Path, run_id: str, *, identity: dict[str, object], plan: FrequencyResponsePlan) -> list[tuple[dict[str, object], list[StageARecord], int, float]]:
    if not run_sink.exists():
        return []
    zip_pattern = re.compile(rf"{re.escape(run_id)}-checkpoint-(\d{{4}})\.zip")
    sha_pattern = re.compile(rf"{re.escape(run_id)}-checkpoint-(\d{{4}})\.zip\.sha256")
    final_names = {f"{run_id}.zip", f"{run_id}.zip.sha256"}
    entries = list(run_sink.iterdir())
    unexpected = [
        path.name for path in entries
        if path.name not in final_names
        and zip_pattern.fullmatch(path.name) is None
        and sha_pattern.fullmatch(path.name) is None
    ]
    if unexpected:
        raise RuntimeError("artifact sink contains loose or unexpected state")
    zip_sequences = {int(match.group(1)) for path in entries if (match := zip_pattern.fullmatch(path.name))}
    sha_sequences = {int(match.group(1)) for path in entries if (match := sha_pattern.fullmatch(path.name))}
    if zip_sequences != sha_sequences:
        raise RuntimeError("artifact sink checkpoint history contains an orphan")
    if zip_sequences and sorted(zip_sequences) != list(range(1, max(zip_sequences) + 1)):
        raise RuntimeError("artifact sink checkpoint sequence has a gap")
    checkpoints: list[tuple[dict[str, object], list[StageARecord], int, float]] = []
    previous_records: list[dict[str, object]] = []
    for sequence in sorted(zip_sequences):
        zip_path, checksum_path = _checkpoint_paths(run_sink, run_id, sequence)
        members = _read_pair(zip_path, checksum_path)
        if set(members) != {"state.json"}:
            raise RuntimeError("checkpoint package members differ")
        state = members["state.json"]
        records, stored_sequence, anchor = _validate_state(state, identity=identity, plan=plan)
        if stored_sequence != sequence or sequence <= 0:
            raise RuntimeError("checkpoint sequence differs from package name")
        serialized = [record.to_dict() for record in records]
        if len(serialized) <= len(previous_records) or serialized[:len(previous_records)] != previous_records:
            raise RuntimeError("artifact sink checkpoint history diverges")
        previous_records = serialized
        checkpoints.append((state, records, sequence, anchor))
    return checkpoints


def _minimal_valid_final(run_sink: Path, *, identity: dict[str, object]) -> int | None:
    zip_path, checksum_path = _final_paths(run_sink, str(identity["run_id"]))
    if not zip_path.exists() and not checksum_path.exists():
        return None
    members = _read_pair(zip_path, checksum_path)
    if set(members) != {"receipt.json", "result.json"}:
        raise RuntimeError("final package members differ")
    receipt, result = members["receipt.json"], members["result.json"]
    if not isinstance(receipt, dict) or not isinstance(result, dict):
        raise RuntimeError("final package JSON shape differs")
    for payload in (receipt, result):
        if payload.get("run_id") != identity["run_id"] or payload.get("resolved_exact") != identity["resolved_exact"]:
            raise RuntimeError("final package run identity differs")
        if payload.get("fixed_unit_count") != FIXED_UNIT_COUNT or payload.get("fixed_record_count") != FIXED_RECORD_COUNT:
            raise RuntimeError("final package fixed denominator differs")
    records = result.get("records")
    if not isinstance(records, list) or len(records) != FIXED_RECORD_COUNT:
        raise RuntimeError("final package record structure differs")
    rc = result.get("rc")
    if type(rc) is not int or rc not in (0, 2) or receipt.get("rc") != rc:
        raise RuntimeError("final package return code differs")
    return rc


def _final_payload(identity: dict[str, object], records: list[StageARecord]) -> tuple[dict[str, object], dict[str, object]]:
    rc = 2 if any(record.status == "operational_failure" for record in records) else 0
    common = {
        "evidence_contract": EVIDENCE_CONTRACT,
        **identity,
        "rc": rc,
        "complete": rc == 0,
        "fixed_condition_count": len(CONDITIONS),
    }
    receipt = {**common, "committed_unit_count": FIXED_UNIT_COUNT, "record_count": FIXED_RECORD_COUNT}
    result = {
        **common,
        "records": [record.to_dict() for record in records],
        "descriptive_per_method_response": _descriptive_response(records),
    }
    return receipt, result


def _publish_checkpoint(
    *, run_dir: Path, run_sink: Path, identity: dict[str, object], records: list[StageARecord],
    sequence: int, anchor: float,
) -> None:
    state = _state_payload(identity, records, checkpoint_sequence=sequence, checkpoint_anchor_unix_seconds=anchor)
    package_dir = run_dir / "packages"
    local_zip = package_dir / f"checkpoint-{sequence:04d}.zip"
    local_checksum = package_dir / f"checkpoint-{sequence:04d}.zip.sha256"
    sink_zip, sink_checksum = _checkpoint_paths(run_sink, str(identity["run_id"]), sequence)
    _write_zip(local_zip, {"state.json": state})
    _write_checksum(local_checksum, local_zip, published_name=sink_zip.name)
    _publish_pair(local_zip, local_checksum, sink_zip, sink_checksum)


def _publish_final(*, run_dir: Path, run_sink: Path, identity: dict[str, object], records: list[StageARecord]) -> int:
    receipt, result = _final_payload(identity, records)
    local_zip = run_dir / "packages" / "final.zip"
    local_checksum = run_dir / "packages" / "final.zip.sha256"
    sink_zip, sink_checksum = _final_paths(run_sink, str(identity["run_id"]))
    _write_zip(local_zip, {"receipt.json": receipt, "result.json": result})
    _write_checksum(local_checksum, local_zip, published_name=sink_zip.name)
    _publish_pair(local_zip, local_checksum, sink_zip, sink_checksum)
    return int(result["rc"])


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
    identity = _run_identity(revision, plan, key_digest)
    run_id = str(identity["run_id"])
    local_work_root = Path(args.local_work_root).resolve()
    artifact_sink = Path(args.artifact_sink).resolve()
    run_dir = local_work_root / run_id
    run_sink = artifact_sink / run_id
    if run_dir == run_sink or run_dir in run_sink.parents or run_sink in run_dir.parents:
        raise ValueError("local active state and artifact sink must be separate")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_sink.mkdir(parents=True, exist_ok=True)
    checkpoints = _sink_checkpoints(run_sink, run_id, identity=identity, plan=plan)
    terminal_rc = _minimal_valid_final(run_sink, identity=identity)
    if terminal_rc is not None:
        del detection_key, hf_token
        return terminal_rc
    sink_records: list[StageARecord] = []
    sink_sequence = 0
    sink_anchor = float(time.time())
    if checkpoints:
        _, sink_records, sink_sequence, sink_anchor = checkpoints[-1]
    state_path = run_dir / "state.json"
    if state_path.exists():
        local_records, local_sequence, local_anchor = _validate_state(_read_json(state_path), identity=identity, plan=plan)
        serialized_local = [record.to_dict() for record in local_records]
        serialized_sink = [record.to_dict() for record in sink_records]
        if local_sequence != sink_sequence or len(local_records) < len(sink_records) or serialized_local[:len(sink_records)] != serialized_sink:
            raise RuntimeError("local and artifact-sink histories diverge or roll back")
        records, checkpoint_sequence, checkpoint_anchor = local_records, local_sequence, local_anchor
    else:
        records, checkpoint_sequence, checkpoint_anchor = sink_records, sink_sequence, sink_anchor
        _atomic_json_write(
            state_path,
            _state_payload(identity, records, checkpoint_sequence=checkpoint_sequence, checkpoint_anchor_unix_seconds=checkpoint_anchor),
        )

    if len(records) < FIXED_RECORD_COUNT:
        try:
            pipeline, hf_assets, lf_assets = _load_pipeline_and_assets(plan.model_id, hf_token)
        except Exception:
            pipeline = hf_assets = lf_assets = None
        del hf_token
    else:
        pipeline = hf_assets = lf_assets = None
        del hf_token
    wrong_keys = _wrong_keys(detection_key)
    for unit_index in range(len(records) // RECORDS_PER_UNIT, FIXED_UNIT_COUNT):
        unit = plan.units[unit_index]
        if pipeline is None:
            transaction = _failure_transaction(unit, run_id=run_id, revision=revision, plan=plan, key_digest=key_digest, reason="runtime_initialization_failure")
        else:
            try:
                transaction = _unit_transaction(unit, pipeline=pipeline, detection_key=detection_key, wrong_keys=wrong_keys, hf_assets=hf_assets, lf_assets=lf_assets, run_id=run_id, revision=revision, plan=plan, key_digest=key_digest)
            except Exception:
                transaction = _failure_transaction(unit, run_id=run_id, revision=revision, plan=plan, key_digest=key_digest, reason="unit_execution_failure")
        validated = _validate_transaction([record.to_dict() for record in transaction], unit=unit, identity=identity, plan=plan)
        records = [*records, *validated]
        now = float(time.time())
        _atomic_json_write(
            state_path,
            _state_payload(identity, records, checkpoint_sequence=checkpoint_sequence, checkpoint_anchor_unix_seconds=checkpoint_anchor),
        )
        if now - checkpoint_anchor >= _CHECKPOINT_INTERVAL_SECONDS and len(records) < FIXED_RECORD_COUNT:
            next_sequence = checkpoint_sequence + 1
            _publish_checkpoint(
                run_dir=run_dir, run_sink=run_sink, identity=identity, records=records,
                sequence=next_sequence, anchor=now,
            )
            checkpoint_sequence, checkpoint_anchor = next_sequence, now
            _atomic_json_write(
                state_path,
                _state_payload(identity, records, checkpoint_sequence=checkpoint_sequence, checkpoint_anchor_unix_seconds=checkpoint_anchor),
            )
    del detection_key, wrong_keys
    if len(records) != FIXED_RECORD_COUNT:
        raise RuntimeError("fixed 320-record export cannot be formed")
    return _publish_final(run_dir=run_dir, run_sink=run_sink, identity=identity, records=records)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--local-work-root", required=True)
    parser.add_argument("--artifact-sink", required=True)
    return parser


def main() -> None:
    raise SystemExit(execute(_parser().parse_args()))


if __name__ == "__main__":
    main()
