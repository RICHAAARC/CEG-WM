"""Thin Colab runner for incomplete Stage-A HF-anchor evidence collection."""

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
import time
from typing import Any
import zipfile

import numpy as np
import torch

from cegwm.method.hf import HF_CANDIDATE_ID, FrozenHFPublicAssets, score_hf_image
from cegwm.protocol.records import StageARecord
from cegwm.protocol.stage_a import StageAProtocol, load_stage_a_protocol
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline, run_sd35_hf, run_sd35_plain
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from cegwm.shared.prg import prg_bytes

KEY_ENV = "CEG_WM_ROOT_KEY"
TOKEN_ENV = "HF_TOKEN"
CHECKPOINT_INTERVAL_HOURS = 2.0
COMPLETENESS = "incomplete_for_hf_anchor"
SCIENTIFIC_STATUS = "not_evaluated"
LIMITATIONS = (
    "jpeg_q75_not_evaluated",
    "gaussian_blur_sigma_1_not_evaluated",
    "gaussian_noise_std_0_01_not_evaluated",
    "lpips_quality_gate_not_evaluated",
    "model_revision_and_weight_digest_not_recorded",
)
_FATAL_ERROR_BY_PHASE = {
    "initialization": "initialization_failure",
    "resume_validation": "resume_validation_failure",
    "runtime_execution": "runtime_execution_failure",
    "checkpoint": "checkpoint_failure",
    "final_export": "final_export_failure",
}


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _json_write(temporary, payload)
    os.replace(temporary, path)


def _git_exact(repo_root: Path, expected_exact: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected_exact) is None:
        raise ValueError("expected exact must be a lowercase 40-character revision")
    resolved = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != expected_exact:
        raise RuntimeError("resolved revision differs from approved execution exact")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("execution checkout must be clean")
    return resolved


def _load_protocol(repo_root: Path) -> StageAProtocol:
    config_root = repo_root / "configs" / "stage_a"
    return load_stage_a_protocol(
        config_root / "stage_a_v1.json",
        config_root / "candidate_selection.jsonl",
        config_root / "untouched_confirmation.jsonl",
    )


def _load_pipeline_and_assets(model_id: str, hf_token: str) -> tuple[Any, FrozenHFPublicAssets]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_colab_execution")
    pipeline = load_sd35_pipeline(model_id, torch_dtype=torch.float16, token=hf_token)
    vae = getattr(pipeline, "vae", None)
    image_processor = getattr(pipeline, "image_processor", None)
    assets = FrozenHFPublicAssets(
        vae=vae,
        image_processor=image_processor,
        image_processor_id=f"{model_id}:image_processor",
    )
    pipeline.to("cuda")
    return pipeline, assets


def _wrong_keys(detection_key: bytes) -> tuple[bytes, ...]:
    return tuple(
        prg_bytes(detection_key, f"stage-a/external-wrong-key/v1/index={index}", 32)
        for index in range(16)
    )


def _scores(image: Any, detection_key: bytes, wrong_keys: tuple[bytes, ...], assets: FrozenHFPublicAssets) -> dict[str, float]:
    values = {"registered": float(score_hf_image(image, detection_key, assets))}
    for index, wrong_key in enumerate(wrong_keys):
        values[f"wrong_{index:02d}"] = float(score_hf_image(image, wrong_key, assets))
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("nonfinite_blind_score")
    return values


def _psnr(first: Any, second: Any) -> float:
    first_pixels = np.asarray(first, dtype=np.float64) / 255.0
    second_pixels = np.asarray(second, dtype=np.float64) / 255.0
    if first_pixels.shape != second_pixels.shape:
        raise ValueError("paired_image_shape_mismatch")
    mse = float(np.mean(np.square(first_pixels - second_pixels)))
    if mse <= 0.0 or not math.isfinite(mse):
        raise ValueError("paired_psnr_not_finite")
    value = -10.0 * math.log10(mse)
    if not math.isfinite(value):
        raise ValueError("paired_psnr_not_finite")
    return value


def _failure_pair(
    unit: Any,
    protocol: StageAProtocol,
    run_id: str,
    revision: str,
    key_digest: str,
    reason: str,
) -> list[StageARecord]:
    return [
        StageARecord(
            run_id=run_id,
            unit_id=unit.unit_id,
            source_cluster_id=unit.source_id,
            arm=arm,
            condition="identity",
            code_revision=revision,
            config_digest=protocol.protocol_digest,
            key_public_digest=key_digest,
            status="operational_failure",
            failure_reason=reason,
        )
        for arm in ("hf_anchor", "primary_null")
    ]


def _new_state(
    *,
    run_id: str,
    resolved_exact: str,
    protocol: StageAProtocol,
    model_id: str,
    key_digest: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "resolved_exact": resolved_exact,
        "protocol_digest": protocol.protocol_digest,
        "hf_candidate_id": HF_CANDIDATE_ID,
        "ordered_roster_unit_ids": [unit.unit_id for unit in protocol.candidate_selection],
        "model_id": model_id,
        "key_public_digest": key_digest,
        "checkpoint_interval_hours": CHECKPOINT_INTERVAL_HOURS,
        "checkpoint_sequence": 0,
        "committed_unit_count": 0,
        "committed_unit_ids": [],
        "records": [],
    }


def _resume_state(
    resume_zip: Path,
    resume_checksum: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    checksum_parts = resume_checksum.read_text(encoding="utf-8").strip().split()
    if len(checksum_parts) != 2 or checksum_parts[1] != resume_zip.name:
        raise ValueError("resume checksum file is malformed")
    if hashlib.sha256(resume_zip.read_bytes()).hexdigest() != checksum_parts[0]:
        raise ValueError("resume checkpoint checksum mismatch")
    with zipfile.ZipFile(resume_zip) as archive:
        if archive.namelist() != ["state.json"]:
            raise ValueError("resume checkpoint must contain only state.json")
        state = json.loads(archive.read("state.json"))
    identity_fields = (
        "run_id",
        "resolved_exact",
        "protocol_digest",
        "hf_candidate_id",
        "ordered_roster_unit_ids",
        "model_id",
        "key_public_digest",
        "checkpoint_interval_hours",
    )
    if any(state.get(field) != expected.get(field) for field in identity_fields):
        raise ValueError("resume checkpoint identity mismatch")
    committed = state.get("committed_unit_ids")
    roster = expected["ordered_roster_unit_ids"]
    records = state.get("records")
    if not isinstance(committed, list) or committed != roster[: len(committed)]:
        raise ValueError("resume committed units must be an ordered roster prefix")
    if not committed:
        raise ValueError("resume checkpoint cannot be empty")
    if not isinstance(records, list) or len(records) != len(committed) * 2:
        raise ValueError("resume checkpoint record count mismatch")
    if state.get("committed_unit_count") != len(committed):
        raise ValueError("resume checkpoint committed count mismatch")
    for index, unit_id in enumerate(committed):
        pair = records[index * 2 : index * 2 + 2]
        if [record.get("unit_id") for record in pair] != [unit_id, unit_id]:
            raise ValueError("resume checkpoint record roster mismatch")
        validated_pair = [StageARecord(**record) for record in pair]
        if [record.arm for record in validated_pair] != ["hf_anchor", "primary_null"]:
            raise ValueError("resume checkpoint paired arms mismatch")
        for record in validated_pair:
            if (
                record.run_id != expected["run_id"]
                or record.code_revision != expected["resolved_exact"]
                or record.config_digest != expected["protocol_digest"]
                or record.key_public_digest != expected["key_public_digest"]
                or record.condition != "identity"
            ):
                raise ValueError("resume checkpoint record identity mismatch")
    sequence = state.get("checkpoint_sequence")
    if not isinstance(sequence, int) or sequence < 1:
        raise ValueError("resume checkpoint sequence is invalid")
    return state


def _deterministic_run_id(
    resolved_exact: str,
    protocol: StageAProtocol,
    model_id: str,
    key_digest: str,
) -> str:
    identity = {
        "resolved_exact": resolved_exact,
        "protocol_digest": protocol.protocol_digest,
        "hf_candidate_id": HF_CANDIDATE_ID,
        "model_id": model_id,
        "key_public_digest": key_digest,
        "checkpoint_interval_hours": CHECKPOINT_INTERVAL_HOURS,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(b"CEG-WM/stage-a2/run-id/v1\x00" + canonical.encode("utf-8"))
    return f"a2hf-{digest.hexdigest()[:24]}"


def _verify_checksum(zip_path: Path, checksum_path: Path) -> str:
    parts = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != zip_path.name:
        raise ValueError("artifact checksum file is malformed")
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if parts[0] != digest:
        raise ValueError("artifact checksum mismatch")
    return digest


def _validate_final(
    zip_path: Path,
    checksum_path: Path,
    expected: dict[str, Any],
) -> tuple[int, str]:
    _verify_checksum(zip_path, checksum_path)
    with zipfile.ZipFile(zip_path) as archive:
        if set(archive.namelist()) != {"receipt.json", "result.json"}:
            raise ValueError("final package members mismatch")
        receipt = json.loads(archive.read("receipt.json"))
        result = json.loads(archive.read("result.json"))
    identity_fields = (
        "run_id",
        "resolved_exact",
        "protocol_digest",
        "hf_candidate_id",
        "ordered_roster_unit_ids",
        "model_id",
        "key_public_digest",
        "checkpoint_interval_hours",
    )
    if any(
        receipt.get(field) != expected.get(field) or result.get(field) != expected.get(field)
        for field in identity_fields
    ):
        raise ValueError("final package identity mismatch")
    if receipt.get("rc") not in {0, 1} or result.get("rc") != receipt.get("rc"):
        raise ValueError("final package RC mismatch")
    if (
        receipt.get("completeness") != COMPLETENESS
        or result.get("completeness") != COMPLETENESS
        or receipt.get("scientific_status") != SCIENTIFIC_STATUS
        or result.get("scientific_status") != SCIENTIFIC_STATUS
        or receipt.get("limitations") != list(LIMITATIONS)
        or result.get("limitations") != list(LIMITATIONS)
    ):
        raise ValueError("final package scientific scope mismatch")
    if result.get("fixed_unit_count") != 8 or result.get("fixed_record_count") != 16:
        raise ValueError("final package fixed denominator mismatch")
    roster = expected["ordered_roster_unit_ids"]
    committed = result.get("committed_unit_ids")
    record_payloads = result.get("records")
    if committed != roster or result.get("committed_unit_count") != 8:
        raise ValueError("final package committed roster mismatch")
    if not isinstance(record_payloads, list) or len(record_payloads) != 16:
        raise ValueError("final package record count mismatch")
    records = [StageARecord(**record) for record in record_payloads]
    for index, unit_id in enumerate(roster):
        pair = records[index * 2 : index * 2 + 2]
        if [record.unit_id for record in pair] != [unit_id, unit_id]:
            raise ValueError("final package record roster mismatch")
        if [record.arm for record in pair] != ["hf_anchor", "primary_null"]:
            raise ValueError("final package paired arms mismatch")
        for record in pair:
            if (
                record.run_id != expected["run_id"]
                or record.code_revision != expected["resolved_exact"]
                or record.config_digest != expected["protocol_digest"]
                or record.key_public_digest != expected["key_public_digest"]
                or record.condition != "identity"
            ):
                raise ValueError("final package record identity mismatch")
    status = receipt.get("status")
    expected_status = (
        "complete_incomplete_scope" if receipt["rc"] == 0 else "complete_with_failures"
    )
    if status != expected_status or result.get("status") != expected_status:
        raise ValueError("final package status mismatch")
    has_failure = any(record.status != "success" for record in records)
    if has_failure != (receipt["rc"] == 1):
        raise ValueError("final package failure/RC mismatch")
    return int(receipt["rc"]), status


def _validate_fatal(
    zip_path: Path,
    checksum_path: Path,
    expected: dict[str, Any],
    error_class: str,
) -> str:
    if error_class not in _FATAL_ERROR_BY_PHASE.values():
        raise ValueError("fatal error class is not predeclared")
    digest = _verify_checksum(zip_path, checksum_path)
    with zipfile.ZipFile(zip_path) as archive:
        if set(archive.namelist()) != {"receipt.json", "result.json"}:
            raise ValueError("fatal package members mismatch")
        receipt = json.loads(archive.read("receipt.json"))
        result = json.loads(archive.read("result.json"))
    identity_fields = (
        "run_id",
        "resolved_exact",
        "protocol_digest",
        "hf_candidate_id",
        "ordered_roster_unit_ids",
        "model_id",
        "key_public_digest",
        "checkpoint_interval_hours",
    )
    if any(
        receipt.get(field) != expected.get(field) or result.get(field) != expected.get(field)
        for field in identity_fields
    ):
        raise ValueError("fatal package identity mismatch")
    if (
        receipt.get("approved_execution_exact") != expected["resolved_exact"]
        or result.get("approved_execution_exact") != expected["resolved_exact"]
        or receipt.get("rc") != 2
        or result.get("rc") != 2
        or receipt.get("status") != "operational_failure"
        or result.get("status") != "operational_failure"
        or receipt.get("result_kind") != "operational_failure_not_scientific"
        or result.get("result_kind") != "operational_failure_not_scientific"
        or receipt.get("error_class") != error_class
        or result.get("error_class") != error_class
    ):
        raise ValueError("fatal package operational status mismatch")
    if (
        receipt.get("completeness") != COMPLETENESS
        or result.get("completeness") != COMPLETENESS
        or receipt.get("scientific_status") != SCIENTIFIC_STATUS
        or result.get("scientific_status") != SCIENTIFIC_STATUS
        or receipt.get("limitations") != list(LIMITATIONS)
        or result.get("limitations") != list(LIMITATIONS)
    ):
        raise ValueError("fatal package scientific scope mismatch")
    roster = expected["ordered_roster_unit_ids"]
    committed = result.get("committed_unit_ids")
    record_payloads = result.get("records")
    if (
        not isinstance(committed, list)
        or committed != roster[: len(committed)]
        or receipt.get("committed_unit_ids") != committed
        or receipt.get("committed_unit_count") != len(committed)
        or result.get("committed_unit_count") != len(committed)
        or not isinstance(record_payloads, list)
        or len(record_payloads) != len(committed) * 2
        or result.get("fixed_unit_count") != 8
        or result.get("fixed_record_count") != 16
    ):
        raise ValueError("fatal package committed prefix mismatch")
    records = [StageARecord(**record) for record in record_payloads]
    for index, unit_id in enumerate(committed):
        pair = records[index * 2 : index * 2 + 2]
        if [record.unit_id for record in pair] != [unit_id, unit_id]:
            raise ValueError("fatal package record roster mismatch")
        if [record.arm for record in pair] != ["hf_anchor", "primary_null"]:
            raise ValueError("fatal package paired arms mismatch")
        for record in pair:
            if (
                record.run_id != expected["run_id"]
                or record.code_revision != expected["resolved_exact"]
                or record.config_digest != expected["protocol_digest"]
                or record.key_public_digest != expected["key_public_digest"]
                or record.condition != "identity"
            ):
                raise ValueError("fatal package record identity mismatch")
    return digest


def _discover_checkpoint(run_store: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    checkpoint_names: set[str] = set()
    pattern = re.compile(r"checkpoint-(\d{4})-units-(\d{4})\.zip")
    for zip_path in sorted(run_store.glob("checkpoint-*.zip")):
        match = pattern.fullmatch(zip_path.name)
        if match is None:
            raise ValueError("checkpoint filename is malformed")
        checksum_path = run_store / f"{zip_path.name}.sha256"
        if not checksum_path.is_file():
            raise ValueError("checkpoint checksum is missing")
        state = _resume_state(zip_path, checksum_path, expected)
        sequence = int(match.group(1))
        committed_count = int(match.group(2))
        if state["checkpoint_sequence"] != sequence:
            raise ValueError("checkpoint sequence filename mismatch")
        if state["committed_unit_count"] != committed_count:
            raise ValueError("checkpoint count filename mismatch")
        candidates.append((sequence, committed_count, state))
        checkpoint_names.update({zip_path.name, checksum_path.name})
    orphan_checksums = {
        path.name for path in run_store.glob("checkpoint-*.zip.sha256")
    } - checkpoint_names
    if orphan_checksums:
        raise ValueError("checkpoint checksum has no matching ZIP")
    ranks = [(sequence, count) for sequence, count, _ in candidates]
    if len({sequence for sequence, _ in ranks}) != len(ranks):
        raise ValueError("checkpoint sequence is ambiguous")
    ordered = sorted(ranks)
    if any(later[1] <= earlier[1] for earlier, later in zip(ordered, ordered[1:])):
        raise ValueError("checkpoint committed counts are ambiguous")
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _checkpoint(state: dict[str, Any], output_dir: Path, checkpoint_sink: Path) -> None:
    sequence = int(state["checkpoint_sequence"]) + 1
    committed_count = len(state["committed_unit_ids"])
    checkpoint_state = dict(state)
    checkpoint_state["checkpoint_sequence"] = sequence
    state_path = output_dir / "state.json"
    _atomic_json_write(state_path, checkpoint_state)
    stem = f"checkpoint-{sequence:04d}-units-{committed_count:04d}"
    zip_path = output_dir / f"{stem}.zip"
    checksum_path = output_dir / f"{stem}.zip.sha256"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(state_path, arcname="state.json")
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    checksum_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path) as archive:
        if json.loads(archive.read("state.json")) != checkpoint_state:
            raise RuntimeError("local checkpoint verification failed")
    for source in (zip_path, checksum_path):
        destination = checkpoint_sink / source.name
        if destination.exists():
            raise RuntimeError("checkpoint sink refuses overwrite")
        shutil.copy2(source, destination)
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError("checkpoint sink copy verification failed")
    state.clear()
    state.update(checkpoint_state)


def _export(output_dir: Path, receipt: dict[str, Any], records: list[StageARecord]) -> tuple[Path, str]:
    result = {
        "run_id": receipt["run_id"],
        "resolved_exact": receipt["resolved_exact"],
        "rc": receipt["rc"],
        "status": receipt["status"],
        "completeness": COMPLETENESS,
        "scientific_status": SCIENTIFIC_STATUS,
        "limitations": list(LIMITATIONS),
        "protocol_digest": receipt["protocol_digest"],
        "hf_candidate_id": receipt["hf_candidate_id"],
        "ordered_roster_unit_ids": receipt["ordered_roster_unit_ids"],
        "model_id": receipt["model_id"],
        "key_public_digest": receipt["key_public_digest"],
        "checkpoint_interval_hours": receipt["checkpoint_interval_hours"],
        "checkpoint_sequence": receipt["checkpoint_sequence"],
        "committed_unit_count": receipt["committed_unit_count"],
        "committed_unit_ids": receipt["committed_unit_ids"],
        "fixed_unit_count": 8,
        "fixed_record_count": 16,
        "records": [record.to_dict() for record in records],
    }
    if "error_class" in receipt:
        result.update({
            "result_kind": receipt["result_kind"],
            "error_class": receipt["error_class"],
            "approved_execution_exact": receipt["approved_execution_exact"],
            "resume_status": receipt["resume_status"],
        })
    _json_write(output_dir / "receipt.json", receipt)
    _json_write(output_dir / "result.json", result)
    zip_path = output_dir / f"{receipt['run_id']}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("receipt.json", "result.json"):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, (output_dir / name).read_bytes())
    zip_digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    return zip_path, zip_digest


def _publish_final(zip_path: Path, zip_digest: str, run_store: Path) -> None:
    checksum_path = zip_path.with_suffix(".zip.sha256")
    checksum_path.write_text(f"{zip_digest}  {zip_path.name}\n", encoding="utf-8")
    _verify_checksum(zip_path, checksum_path)
    for source in (zip_path, checksum_path):
        destination = run_store / source.name
        if destination.exists():
            raise RuntimeError("final run store refuses overwrite")
        shutil.copy2(source, destination)
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError("final run store copy verification failed")


def _publish_failure(
    zip_path: Path,
    checksum_path: Path,
    run_store: Path,
    expected: dict[str, Any],
    error_class: str,
) -> None:
    destination_zip = run_store / zip_path.name
    destination_checksum = run_store / checksum_path.name
    existing = (destination_zip.exists(), destination_checksum.exists())
    if any(existing):
        if not all(existing) or not destination_zip.is_file() or not destination_checksum.is_file():
            raise RuntimeError("failure run store pair is incomplete")
        _validate_fatal(destination_zip, destination_checksum, expected, error_class)
        if (
            zip_path.read_bytes() != destination_zip.read_bytes()
            or checksum_path.read_bytes() != destination_checksum.read_bytes()
        ):
            raise RuntimeError("failure run store refuses non-identical overwrite")
        return
    for source, destination in (
        (zip_path, destination_zip),
        (checksum_path, destination_checksum),
    ):
        with destination.open("xb") as target:
            target.write(source.read_bytes())
    _validate_fatal(destination_zip, destination_checksum, expected, error_class)


def _export_fatal(args: argparse.Namespace, context: dict[str, Any], error_class: str) -> tuple[Path, str]:
    if error_class not in _FATAL_ERROR_BY_PHASE.values():
        raise ValueError("fatal error class is not predeclared")
    run_id = context.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]{7,63}", run_id) is None:
        raise ValueError("unsafe run id cannot name a fatal package")
    output_dir = context.get("output_dir")
    if output_dir is None:
        output_dir = Path(args.output_root).resolve() / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        context["output_dir"] = output_dir
        context["output_dir_owned"] = True
    elif not context.get("output_dir_owned"):
        raise RuntimeError("fatal package refuses an unowned output directory")
    state = context.get("state") or {}
    record_payloads = state.get("records", [])
    records = [StageARecord(**record) for record in record_payloads]
    committed_ids = list(state.get("committed_unit_ids", []))
    if len(records) != len(committed_ids) * 2:
        raise ValueError("fatal package refuses inconsistent committed records")
    approved_exact = (
        args.expected_exact
        if re.fullmatch(r"[0-9a-f]{40}", args.expected_exact) is not None
        else None
    )
    interval = context.get("checkpoint_interval_hours")
    receipt: dict[str, Any] = {
        "run_id": run_id,
        "approved_execution_exact": approved_exact,
        "resolved_exact": context.get("resolved_exact"),
        "rc": 2,
        "status": "operational_failure",
        "result_kind": "operational_failure_not_scientific",
        "error_class": error_class,
        "completeness": COMPLETENESS,
        "scientific_status": SCIENTIFIC_STATUS,
        "protocol_digest": context.get("protocol_digest"),
        "hf_candidate_id": HF_CANDIDATE_ID,
        "ordered_roster_unit_ids": list(context.get("ordered_roster_unit_ids", [])),
        "model_id": context.get("model_id"),
        "key_public_digest": context.get("key_public_digest"),
        "checkpoint_interval_hours": interval,
        "checkpoint_sequence": int(state.get("checkpoint_sequence", 0)),
        "committed_unit_count": len(committed_ids),
        "committed_unit_ids": committed_ids,
        "resume_status": context.get("resume_status", "not_requested"),
        "limitations": list(LIMITATIONS),
    }
    expected = context.get("expected_state")
    run_store = context.get("run_store")
    if not isinstance(expected, dict) or not isinstance(run_store, Path):
        raise RuntimeError("fatal package requires resolved run identity and sink")
    base_zip, zip_digest = _export(output_dir, receipt, records)
    fatal_zip = output_dir / f"failure-{error_class}.zip"
    os.replace(base_zip, fatal_zip)
    fatal_checksum = output_dir / f"{fatal_zip.name}.sha256"
    fatal_checksum.write_text(f"{zip_digest}  {fatal_zip.name}\n", encoding="utf-8")
    _validate_fatal(fatal_zip, fatal_checksum, expected, error_class)
    _publish_failure(fatal_zip, fatal_checksum, run_store, expected, error_class)
    return fatal_zip, zip_digest


def execute(args: argparse.Namespace, *, fatal_context: dict[str, Any] | None = None) -> int:
    context = fatal_context if fatal_context is not None else {}
    context["phase"] = "initialization"
    repo_root = Path(args.repo_root).resolve()
    resolved_exact = _git_exact(repo_root, args.expected_exact)
    context["resolved_exact"] = resolved_exact
    protocol = _load_protocol(repo_root)
    context["protocol_digest"] = protocol.protocol_digest
    context["ordered_roster_unit_ids"] = [
        unit.unit_id for unit in protocol.candidate_selection
    ]
    runtime_config = protocol.config["generation_runtime"]
    budget_config = protocol.config["budget"]
    if runtime_config["model_id"] != "stabilityai/stable-diffusion-3.5-medium":
        raise RuntimeError("protocol_model_identity_mismatch")
    if runtime_config["public_asset_rule"] != (
        "protocol_model_id_default_hub_resolution_without_revision_or_weight_digest"
    ):
        raise RuntimeError("protocol_public_asset_rule_mismatch")
    if runtime_config["inference_steps"] != 20 or budget_config["total_relative_l2"] != 0.012:
        raise RuntimeError("protocol_runtime_identity_mismatch")
    if len(protocol.candidate_selection) != 8:
        raise RuntimeError("candidate_selection_roster_mismatch")
    model_id = runtime_config["model_id"]
    context["model_id"] = model_id
    context["checkpoint_interval_hours"] = CHECKPOINT_INTERVAL_HOURS
    run_store_root = Path(args.run_store_root).resolve()
    run_store_root.mkdir(parents=True, exist_ok=True)
    if not run_store_root.is_dir():
        raise ValueError("run store root must be a directory")

    raw_key = os.environ.pop(KEY_ENV, None)
    hf_token = os.environ.pop(TOKEN_ENV, None)
    if not isinstance(raw_key, str) or not raw_key.strip():
        hf_token = ""
        del hf_token
        raise RuntimeError("root_key_environment_input_required")
    detection_key = normalize_detection_key(raw_key)
    del raw_key
    key_digest = public_key_digest(detection_key)
    context["key_public_digest"] = key_digest
    run_id = _deterministic_run_id(resolved_exact, protocol, model_id, key_digest)
    context["run_id"] = run_id
    expected_state = _new_state(
        run_id=run_id,
        resolved_exact=resolved_exact,
        protocol=protocol,
        model_id=model_id,
        key_digest=key_digest,
    )
    context["expected_state"] = expected_state
    context["state"] = expected_state
    run_store = run_store_root / run_id
    if run_store.exists():
        if not run_store.is_dir():
            raise ValueError("deterministic run store path must be a directory")
    else:
        run_store.mkdir(parents=False, exist_ok=False)
    context["run_store"] = run_store
    if not isinstance(hf_token, str) or not hf_token.strip():
        hf_token = ""
        del hf_token, detection_key
        raise RuntimeError("hugging_face_token_environment_input_required")
    wrong_keys = _wrong_keys(detection_key)
    final_zip = run_store / f"{run_id}.zip"
    final_checksum = run_store / f"{run_id}.zip.sha256"
    final_names = {final_zip.name, final_checksum.name}
    checkpoint_name = re.compile(r"checkpoint-\d{4}-units-\d{4}\.zip(?:\.sha256)?")
    fatal_classes = "|".join(re.escape(value) for value in _FATAL_ERROR_BY_PHASE.values())
    failure_name = re.compile(rf"failure-(?:{fatal_classes})\.zip(?:\.sha256)?")
    context["phase"] = "resume_validation"
    for path in run_store.iterdir():
        if not path.is_file() or (
            path.name not in final_names
            and checkpoint_name.fullmatch(path.name) is None
            and failure_name.fullmatch(path.name) is None
        ):
            raise ValueError("run store contains an unexpected artifact")
    for error_class in _FATAL_ERROR_BY_PHASE.values():
        failure_zip = run_store / f"failure-{error_class}.zip"
        failure_checksum = run_store / f"{failure_zip.name}.sha256"
        if failure_zip.exists() or failure_checksum.exists():
            if not (failure_zip.is_file() and failure_checksum.is_file()):
                raise ValueError("failure package pair is incomplete")
            _validate_fatal(failure_zip, failure_checksum, expected_state, error_class)
    context["resume_status"] = "rejected"
    state = _discover_checkpoint(run_store, expected_state)
    context["state"] = state or expected_state
    if final_zip.exists() or final_checksum.exists():
        if not (final_zip.is_file() and final_checksum.is_file()):
            raise ValueError("final package pair is incomplete")
        final_rc, final_status = _validate_final(final_zip, final_checksum, expected_state)
        context["resume_status"] = "verified_final"
        hf_token = ""
        del detection_key, wrong_keys, hf_token
        print(
            "CEGWM_PROGRESS " + json.dumps({
                "run_id": run_id,
                "committed": 8,
                "fixed_total": 8,
                "phase": "verified_final",
            }),
            flush=True,
        )
        print(
            "CEGWM_SUMMARY " + json.dumps({
                "run_id": run_id,
                "resolved_exact": resolved_exact,
                "rc": final_rc,
                "status": final_status,
                "zip_path": str(final_zip),
                "zip_sha256": _verify_checksum(final_zip, final_checksum),
            }),
            flush=True,
        )
        return final_rc
    if state is None:
        state = expected_state
        context["resume_status"] = "fresh"
    else:
        context["resume_status"] = "accepted_checkpoint"
    context["state"] = state
    context["phase"] = "runtime_execution"
    output_dir = Path(args.output_root).resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    context["output_dir"] = output_dir
    context["output_dir_owned"] = True
    _atomic_json_write(output_dir / "state.json", state)
    receipt: dict[str, Any] = {
        "run_id": run_id,
        "resolved_exact": resolved_exact,
        "rc": None,
        "status": "running",
        "completeness": COMPLETENESS,
        "scientific_status": SCIENTIFIC_STATUS,
        "protocol_digest": protocol.protocol_digest,
        "hf_candidate_id": HF_CANDIDATE_ID,
        "ordered_roster_unit_ids": expected_state["ordered_roster_unit_ids"],
        "model_id": model_id,
        "key_public_digest": key_digest,
        "checkpoint_interval_hours": CHECKPOINT_INTERVAL_HOURS,
        "limitations": list(LIMITATIONS),
    }
    _json_write(output_dir / "receipt.json", receipt)
    records = [StageARecord(**record) for record in state["records"]]
    committed = set(state["committed_unit_ids"])
    pending_units = [unit for unit in protocol.candidate_selection if unit.unit_id not in committed]
    any_failure = any(record.status != "success" for record in records)
    pipeline = None
    assets = None
    model_load_failed = False
    last_checkpoint_time = time.monotonic()
    if pending_units:
        try:
            pipeline, assets = _load_pipeline_and_assets(model_id, hf_token)
        except Exception:
            model_load_failed = True
            any_failure = True
        finally:
            hf_token = ""
            del hf_token
    else:
        hf_token = ""
        del hf_token
    checkpoint_failure = False
    new_units_since_checkpoint = 0
    for unit in pending_units:
        if model_load_failed:
            pair = _failure_pair(
                unit,
                protocol,
                run_id,
                resolved_exact,
                key_digest,
                "model_load_failure",
            )
        else:
            try:
                hf_generator = torch.Generator(device="cuda").manual_seed(unit.seed)
                null_generator = torch.Generator(device="cuda").manual_seed(unit.seed)
                hf_output = run_sd35_hf(
                    pipeline,
                    unit.prompt,
                    detection_key,
                    assets,
                    height=unit.height,
                    width=unit.width,
                    generator=hf_generator,
                )
                null_image = run_sd35_plain(
                    pipeline,
                    unit.prompt,
                    height=unit.height,
                    width=unit.width,
                    generator=null_generator,
                )
                budget_value = float(hf_output.injection_budget.relative_l2)
                if not math.isfinite(budget_value) or not 0.0 < budget_value <= 0.012:
                    raise ValueError("actual_dtype_budget_invalid")
                psnr = _psnr(hf_output.image, null_image)
                hf_scores = _scores(hf_output.image, detection_key, wrong_keys, assets)
                null_scores = _scores(null_image, detection_key, wrong_keys, assets)
                common = dict(
                    run_id=run_id,
                    unit_id=unit.unit_id,
                    source_cluster_id=unit.source_id,
                    condition="identity",
                    code_revision=resolved_exact,
                    config_digest=protocol.protocol_digest,
                    key_public_digest=key_digest,
                    status="success",
                )
                pair = [
                    StageARecord(
                        arm="hf_anchor",
                        scores=hf_scores,
                        metrics={
                            "actual_dtype_relative_l2": budget_value,
                            "paired_rgb_psnr": psnr,
                        },
                        **common,
                    ),
                    StageARecord(
                        arm="primary_null",
                        scores=null_scores,
                        metrics={"paired_rgb_psnr": psnr},
                        **common,
                    ),
                ]
            except Exception:
                any_failure = True
                pair = _failure_pair(
                    unit,
                    protocol,
                    run_id,
                    resolved_exact,
                    key_digest,
                    "unit_execution_failure",
                )
        next_state = dict(state)
        next_state["committed_unit_ids"] = [*state["committed_unit_ids"], unit.unit_id]
        next_state["committed_unit_count"] = len(next_state["committed_unit_ids"])
        next_state["records"] = [
            *state["records"],
            *(record.to_dict() for record in pair),
        ]
        _atomic_json_write(output_dir / "state.json", next_state)
        state.clear()
        state.update(next_state)
        records.extend(pair)
        new_units_since_checkpoint += 1
        now = time.monotonic()
        if (
            new_units_since_checkpoint > 0
            and now - last_checkpoint_time >= CHECKPOINT_INTERVAL_HOURS * 3600.0
        ):
            try:
                context["phase"] = "checkpoint"
                _checkpoint(state, output_dir, run_store)
                last_checkpoint_time = now
                new_units_since_checkpoint = 0
            except Exception:
                checkpoint_failure = True
                any_failure = True
            finally:
                context["phase"] = "runtime_execution"
        print(
            "CEGWM_PROGRESS " + json.dumps({
                "run_id": run_id,
                "committed": len(state["committed_unit_ids"]),
                "fixed_total": 8,
            }),
            flush=True,
        )

    if len(records) != 16:
        raise RuntimeError("fixed_record_roster_not_preserved")
    receipt["rc"] = 1 if any_failure else 0
    receipt["checkpoint_sequence"] = state["checkpoint_sequence"]
    receipt["committed_unit_count"] = state["committed_unit_count"]
    receipt["committed_unit_ids"] = list(state["committed_unit_ids"])
    if checkpoint_failure:
        receipt["checkpoint_status"] = "failure"
    else:
        receipt["checkpoint_status"] = "complete"
    receipt["status"] = "complete_with_failures" if any_failure else "complete_incomplete_scope"
    context["phase"] = "final_export"
    zip_path, zip_digest = _export(output_dir, receipt, records)
    _discover_checkpoint(run_store, expected_state)
    _publish_final(zip_path, zip_digest, run_store)
    _validate_final(final_zip, final_checksum, expected_state)
    del detection_key, wrong_keys
    print(
        "CEGWM_SUMMARY " + json.dumps({
            "run_id": run_id,
            "resolved_exact": resolved_exact,
            "rc": receipt["rc"],
            "zip_path": str(final_zip),
            "zip_sha256": zip_digest,
        }),
        flush=True,
    )
    return int(receipt["rc"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--run-store-root", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    fatal_context: dict[str, Any] = {}
    try:
        return_code = execute(args, fatal_context=fatal_context)
    except (Exception, KeyboardInterrupt):
        phase = fatal_context.get("phase", "initialization")
        error_class = _FATAL_ERROR_BY_PHASE.get(phase, "initialization_failure")
        export_status = "unavailable"
        try:
            _export_fatal(args, fatal_context, error_class)
            export_status = "published"
        except Exception:
            pass
        print(
            "CEGWM_FATAL " + json.dumps({
                "run_id": fatal_context.get("run_id"),
                "error_class": error_class,
                "export_status": export_status,
            }),
            flush=True,
        )
        return_code = 2
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
