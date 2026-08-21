"""Thin Colab runner for incomplete Stage-A HF-anchor evidence collection."""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
from cegwm.runtime.diffusers_sd35 import run_sd35_hf, run_sd35_plain
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from cegwm.shared.prg import prg_bytes

KEY_ENV = "CEGWM_STAGE_A_DETECTION_KEY"
COMPLETENESS = "incomplete_for_hf_anchor"
SCIENTIFIC_STATUS = "not_evaluated"
LIMITATIONS = (
    "jpeg_q75_not_evaluated",
    "gaussian_blur_sigma_1_not_evaluated",
    "gaussian_noise_std_0_01_not_evaluated",
    "lpips_quality_gate_not_evaluated",
)


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


def _module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\x00")
        digest.update(str(value.dtype).encode("ascii") + b"\x00")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\x00")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_pipeline_and_assets(model_id: str, model_revision: str) -> tuple[Any, FrozenHFPublicAssets]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_colab_execution")
    try:
        diffusers = importlib.import_module("diffusers")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("diffusers_required_for_colab_execution") from error
    pipeline_class = getattr(diffusers, "StableDiffusion3Pipeline", None)
    if pipeline_class is None:
        raise RuntimeError("stable_diffusion_3_pipeline_unavailable")
    pipeline = pipeline_class.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch.float16,
    )
    vae = getattr(pipeline, "vae", None)
    image_processor = getattr(pipeline, "image_processor", None)
    assets = FrozenHFPublicAssets(
        vae=vae,
        image_processor=image_processor,
        model_revision=model_revision,
        vae_weight_digest=_module_digest(vae),
        image_processor_id=f"{model_id}@{model_revision}:image_processor",
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
    model_revision: str,
    key_digest: str,
    checkpoint_interval_hours: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "resolved_exact": resolved_exact,
        "protocol_digest": protocol.protocol_digest,
        "hf_candidate_id": HF_CANDIDATE_ID,
        "ordered_roster_unit_ids": [unit.unit_id for unit in protocol.candidate_selection],
        "model_revision": model_revision,
        "key_public_digest": key_digest,
        "checkpoint_interval_hours": checkpoint_interval_hours,
        "checkpoint_sequence": 0,
        "committed_unit_count": 0,
        "committed_unit_ids": [],
        "records": [],
        "vae_weight_digest": None,
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
        "model_revision",
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
        "completeness": COMPLETENESS,
        "scientific_status": SCIENTIFIC_STATUS,
        "limitations": list(LIMITATIONS),
        "checkpoint_interval_hours": receipt["checkpoint_interval_hours"],
        "checkpoint_sequence": receipt["checkpoint_sequence"],
        "committed_unit_count": receipt["committed_unit_count"],
        "fixed_unit_count": 8,
        "fixed_record_count": 16,
        "records": [record.to_dict() for record in records],
    }
    _json_write(output_dir / "receipt.json", receipt)
    _json_write(output_dir / "result.json", result)
    zip_path = output_dir / f"{receipt['run_id']}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output_dir / "receipt.json", arcname="receipt.json")
        archive.write(output_dir / "result.json", arcname="result.json")
    zip_digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    return zip_path, zip_digest


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    resolved_exact = _git_exact(repo_root, args.expected_exact)
    protocol = _load_protocol(repo_root)
    runtime_config = protocol.config["generation_runtime"]
    budget_config = protocol.config["budget"]
    if runtime_config["model_id"] != "stabilityai/stable-diffusion-3.5-medium":
        raise RuntimeError("protocol_model_identity_mismatch")
    if runtime_config["inference_steps"] != 20 or budget_config["total_relative_l2"] != 0.012:
        raise RuntimeError("protocol_runtime_identity_mismatch")
    if len(protocol.candidate_selection) != 8:
        raise RuntimeError("candidate_selection_roster_mismatch")
    if re.fullmatch(r"[0-9a-f]{40}", args.model_revision) is None:
        raise ValueError("model revision must be a lowercase 40-character revision")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{7,63}", args.run_id) is None:
        raise ValueError("run id must be 8-64 lowercase letters, digits, or hyphens")
    checkpoint_interval_hours = float(args.checkpoint_interval_hours)
    if not 1.0 <= checkpoint_interval_hours <= 2.0:
        raise ValueError("checkpoint interval hours must be in [1.0, 2.0]")
    checkpoint_sink = Path(args.checkpoint_sink).resolve()
    if not checkpoint_sink.is_dir():
        raise ValueError("checkpoint sink must be an existing directory")
    if any(path.suffix not in {".zip", ".sha256"} for path in checkpoint_sink.iterdir()):
        raise ValueError("checkpoint sink may contain only zip and sha256 files")
    if bool(args.resume_zip) != bool(args.resume_checksum):
        raise ValueError("resume requires both checkpoint zip and checksum")

    raw_key = os.environ.pop(KEY_ENV, None)
    if raw_key is None:
        raise RuntimeError("detection_key_environment_input_required")
    detection_key = normalize_detection_key(raw_key)
    del raw_key
    key_digest = public_key_digest(detection_key)
    wrong_keys = _wrong_keys(detection_key)
    output_dir = Path(args.output_root).resolve() / args.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    expected_state = _new_state(
        run_id=args.run_id,
        resolved_exact=resolved_exact,
        protocol=protocol,
        model_revision=args.model_revision,
        key_digest=key_digest,
        checkpoint_interval_hours=checkpoint_interval_hours,
    )
    if args.resume_zip:
        state = _resume_state(Path(args.resume_zip), Path(args.resume_checksum), expected_state)
    else:
        state = expected_state
    _atomic_json_write(output_dir / "state.json", state)
    receipt: dict[str, Any] = {
        "run_id": args.run_id,
        "resolved_exact": resolved_exact,
        "rc": None,
        "status": "running",
        "completeness": COMPLETENESS,
        "scientific_status": SCIENTIFIC_STATUS,
        "protocol_digest": protocol.protocol_digest,
        "model_id": runtime_config["model_id"],
        "model_revision": args.model_revision,
        "key_public_digest": key_digest,
        "checkpoint_interval_hours": checkpoint_interval_hours,
        "full_weight_digest": None,
        "full_weight_digest_status": "not_computed_nonblocking",
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
            pipeline, assets = _load_pipeline_and_assets(runtime_config["model_id"], args.model_revision)
            state["vae_weight_digest"] = assets.vae_weight_digest
        except Exception:
            model_load_failed = True
            any_failure = True
    checkpoint_failure = False
    new_units_since_checkpoint = 0
    for unit in pending_units:
        if model_load_failed:
            pair = _failure_pair(
                unit,
                protocol,
                args.run_id,
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
                    run_id=args.run_id,
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
                    args.run_id,
                    resolved_exact,
                    key_digest,
                    "unit_execution_failure",
                )
        records.extend(pair)
        state["committed_unit_ids"].append(unit.unit_id)
        state["committed_unit_count"] = len(state["committed_unit_ids"])
        state["records"].extend(record.to_dict() for record in pair)
        _atomic_json_write(output_dir / "state.json", state)
        new_units_since_checkpoint += 1
        now = time.monotonic()
        if (
            new_units_since_checkpoint > 0
            and now - last_checkpoint_time >= checkpoint_interval_hours * 3600.0
        ):
            try:
                _checkpoint(state, output_dir, checkpoint_sink)
                last_checkpoint_time = now
                new_units_since_checkpoint = 0
            except Exception:
                checkpoint_failure = True
                any_failure = True
        print(
            "CEGWM_PROGRESS " + json.dumps({
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
    if checkpoint_failure:
        receipt["checkpoint_status"] = "failure"
    else:
        receipt["checkpoint_status"] = "complete"
    receipt["status"] = "complete_with_failures" if any_failure else "complete_incomplete_scope"
    receipt["vae_weight_digest"] = (
        assets.vae_weight_digest if assets is not None else state.get("vae_weight_digest")
    )
    zip_path, zip_digest = _export(output_dir, receipt, records)
    del detection_key, wrong_keys
    print(
        "CEGWM_SUMMARY " + json.dumps({
            "run_id": args.run_id,
            "resolved_exact": resolved_exact,
            "rc": receipt["rc"],
            "zip_path": str(zip_path),
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
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-sink", required=True)
    parser.add_argument("--checkpoint-interval-hours", type=float, default=2.0)
    parser.add_argument("--resume-zip")
    parser.add_argument("--resume-checksum")
    return parser


def main() -> None:
    try:
        return_code = execute(_parser().parse_args())
    except Exception:
        print("CEGWM_FATAL " + json.dumps({"code": "initialization_or_export_failure"}), flush=True)
        return_code = 2
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
