"""Bounded runner for Geometry-V2 keyed neural corner synchronization N0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

import torch

from cegwm.geometry_v2.contracts import METHOD_IDENTITY, PROTOCOL_IDENTITY
from cegwm.geometry_v2.operational import (
    ATTACKS,
    BATCH_SIZE,
    CONFIRMATION_SEEDS,
    CONFIRMATION_UNIT_COUNT,
    EPOCHS,
    LEARNING_RATE,
    MINIMUM_SUPPORT,
    RELIABILITY_THRESHOLD,
    TRAINING_SEED,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
    package_n0,
    run_n0,
)


SUCCESS_PREFIX = "CEGWM_GEOMETRY_V2_N0 "
FAILURE_PREFIX = "CEGWM_GEOMETRY_V2_N0_FAILURE "
MAX_CONTROL_BYTES = 1024
MAX_PLAN_BYTES = 32768
KEY_ENVIRONMENT_VARIABLE = "CEGWM_GEOMETRY_KEY_HEX"


def _json(value: Any, maximum: int) -> bytes:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > maximum:
        raise ValueError("bounded_json_exceeded")
    return data


def build_plan() -> dict[str, Any]:
    return {
        "schema": "geometry-v2-neural-corner-sync-n0-plan-v1",
        "method": METHOD_IDENTITY,
        "protocol": PROTOCOL_IDENTITY,
        "image_size": 128,
        "train_seeds": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1], len(TRAIN_SEEDS)],
        "validation_seeds": [VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1], len(VALIDATION_SEEDS)],
        "confirmation_seeds": [CONFIRMATION_SEEDS[0], CONFIRMATION_SEEDS[-1], len(CONFIRMATION_SEEDS)],
        "attacks": list(ATTACKS),
        "training": {"seed": TRAINING_SEED, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "optimizer": "Adam", "lr": LEARNING_RATE},
        "loss": {"corner_smooth_l1": 1.0, "sync_reconstruction": 0.25, "residual_l2": 0.05},
        "residual_linf": 4.0 / 255.0,
        "reliability": {"formula": "clamp(1-mean_corner_error/0.25,0,1)", "threshold": RELIABILITY_THRESHOLD, "minimum_support": MINIMUM_SUPPORT},
        "candidate_gate": {"declared_unit_count": CONFIRMATION_UNIT_COUNT, "median_corner_error_lt": 0.05, "p95_corner_error_lt": 0.10, "reliable_fraction_gte": 0.75},
        "science_denominator": 0,
        "persist": ["bounded_public_metrics"],
        "forbidden_persist": ["raw_geometry_key", "images", "model_weights", "checkpoints"],
    }


def _execution_exact(expected: str, root: Path) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise ValueError("invalid_expected_exact")
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    if actual != expected or dirty:
        raise RuntimeError("execution_identity_mismatch")
    return actual


def _geometry_key() -> bytes:
    value = os.environ.get(KEY_ENVIRONMENT_VARIABLE, "")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("geometry_key_environment_invalid")
    return bytes.fromhex(value)


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device_must_be_auto_cpu_or_cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda_unavailable")
    return requested


def _public_error(error: BaseException) -> str:
    if isinstance(error, (TypeError, ValueError)):
        return "validation_error"
    if isinstance(error, (FileExistsError, FileNotFoundError, PermissionError, OSError)):
        return "filesystem_error"
    if isinstance(error, subprocess.SubprocessError):
        return "execution_identity_error"
    if isinstance(error, RuntimeError):
        return "runtime_error"
    return "unexpected_error"


def _emit(fd: int, prefix: str, value: dict[str, Any]) -> None:
    line = prefix.encode("ascii") + _json(value, MAX_CONTROL_BYTES - len(prefix) - 1) + b"\n"
    if len(line) > MAX_CONTROL_BYTES:
        raise ValueError("control_bound_exceeded")
    os.write(fd, line)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--control-fd", required=True, type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    stage = "plan"
    exact = args.expected_exact if re.fullmatch(r"[0-9a-f]{40}", args.expected_exact) else "invalid"
    run_id = f"geometry-v2-neural-corner-sync-n0-{exact[:12]}"
    try:
        plan_bytes = _json(build_plan(), MAX_PLAN_BYTES)
        plan_digest = hashlib.sha256(plan_bytes).hexdigest()
        stage = "execution_identity"
        exact = _execution_exact(args.expected_exact, Path(args.repo_root).resolve())
        run_id = f"geometry-v2-neural-corner-sync-n0-{exact[:12]}"
        stage = "geometry_key"
        key = _geometry_key()
        stage = "run_n0"
        result = run_n0(key, device_name=_device(args.device))
        key = b""
        stage = "artifact_packaging"
        package = package_n0(result, Path(args.output_root), execution_exact=exact)
        stage = "control_channel"
        _emit(args.control_fd, SUCCESS_PREFIX, {
            "status": "success", "run_id": run_id, "protocol": PROTOCOL_IDENTITY,
            "execution_exact": exact, "plan_digest": plan_digest,
            "n0_status": result.summary["n0_status"], "science_denominator": 0, **package,
        })
        return 0
    except BaseException as error:
        if stage == "control_channel":
            return 1
        try:
            _emit(args.control_fd, FAILURE_PREFIX, {
                "status": "failure", "run_id": run_id, "protocol": PROTOCOL_IDENTITY,
                "failure_point": stage, "error_class": _public_error(error), "science_denominator": 0,
            })
        except BaseException:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
