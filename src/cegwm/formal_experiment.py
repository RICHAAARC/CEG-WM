"""Pure formal-experiment protocol, statistics, roster, and recovery helpers.

The module is intentionally model-free.  GPU workers provide one-unit callbacks;
this layer freezes identities, retains every attempt, and publishes append-only
Drive state without checksum, receipt, lock, or concurrency machinery.
"""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter


ALPHA = 0.001
CONFIDENCE = 0.95
CALIBRATION_NEGATIVES = 2_000
CLEAN_TEST_NEGATIVES = 3_000
EVALUATION_PAIRS = 1_000
CHECKPOINT_INTERVAL_SECONDS = 2 * 60 * 60
CHECKPOINT_SHARD_SIZE = 25
MAX_UNIT_ATTEMPTS = 2

FORMAL_CONDITIONS = (
    "clean_no_attack",
    "jpeg_q50",
    "resize_50_bicubic_restore",
    "center_crop_80_restore",
    "gaussian_blur_sigma_1px",
    "rotation_10_bicubic_reflect_center_crop_v1",
)

RETRYABLE_OPERATIONAL_CODES = frozenset({
    "CUDA_OOM_TRANSIENT",
    "MODEL_RUNTIME_TRANSIENT",
})
QUALITY_METRICS = ("psnr", "ssim", "lpips")


@dataclass(frozen=True, slots=True)
class FormalUnit:
    partition: str
    roster_index: int
    unit_id: str
    prompt_id: str
    prompt: str
    seed: int
    height: int = 512
    width: int = 512


class OperationalUnitError(RuntimeError):
    """Typed unit failure; only frozen codes are eligible for one retry."""

    def __init__(self, code: str, stage: str, detail: str) -> None:
        if not all(isinstance(value, str) and value for value in (code, stage, detail)):
            raise TypeError("operational failure fields must be nonempty text")
        self.code = code
        self.stage = stage
        self.detail = detail
        super().__init__(f"{code}:{stage}:{detail}")


class PreflightFailed(RuntimeError):
    """Job-level recoverable preflight failure that consumes no formal unit."""

    def __init__(self, state: Mapping[str, Any]) -> None:
        self.state = dict(state)
        super().__init__(str(self.state.get("error", "formal worker preflight failed")))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_create_only(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_replaceable(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def classify_operational_exception(error: BaseException, stage: str) -> OperationalUnitError | None:
    """Map only recognizable transient runtime failures to frozen retry codes."""

    if isinstance(error, OperationalUnitError):
        return error
    detail = f"{type(error).__name__}: {error}"
    lowered = detail.lower()
    if type(error).__name__ == "OutOfMemoryError" or any(token in lowered for token in (
        "cuda out of memory", "cuda error: out of memory", "cublas_status_alloc_failed",
    )):
        return OperationalUnitError("CUDA_OOM_TRANSIENT", stage, detail)
    if isinstance(error, (TimeoutError, ConnectionError)) or any(token in lowered for token in (
        "timed out", "read timeout", "connect timeout", "temporary failure",
        "temporarily unavailable", "service unavailable", "connection error",
        "connection reset", "connection aborted",
        "cuda error", "cudnn_status", "cublas_status", "nccl error",
    )):
        return OperationalUnitError("MODEL_RUNTIME_TRANSIENT", stage, detail)
    return None


def raise_classified_operational(error: BaseException, stage: str) -> None:
    mapped = classify_operational_exception(error, stage)
    if mapped is not None:
        raise mapped from error
    raise error


def publish_job_state(
    root: str | Path,
    identity: Mapping[str, Any],
    status: str,
    **details: Any,
) -> dict[str, Any]:
    """Publish replaceable non-statistical job state outside formal unit records."""

    if not isinstance(status, str) or not status:
        raise ValueError("job state status must be nonempty text")
    payload = {
        "schema_version": "cegwm_formal_job_state_v1",
        "identity": dict(identity),
        "status": status,
        "science_denominator": 0,
        "updated_at_unix_seconds": time.time(),
        **details,
    }
    _write_json_replaceable(Path(root) / "job_state.json", payload)
    return payload


def execute_job_preflight(
    root: str | Path,
    identity: Mapping[str, Any],
    callback: Callable[[], Any],
) -> Any:
    """Build and probe a runtime before any formal stage is initialized."""

    publish_job_state(root, identity, "PREFLIGHT_RUNNING")
    try:
        result = callback()
    except Exception as error:
        mapped = classify_operational_exception(error, "preflight")
        state = publish_job_state(
            root,
            identity,
            "PREFLIGHT_FAILED_RECOVERABLE",
            error_code=mapped.code if mapped is not None else "PREFLIGHT_ERROR",
            error=f"{type(error).__name__}: {error}",
        )
        raise PreflightFailed(state) from error
    publish_job_state(root, identity, "PREFLIGHT_READY")
    return result


def load_formal_config(path: str | Path) -> dict[str, Any]:
    config = _read_json(Path(path))
    if not isinstance(config, dict) or config.get("schema_version") != "cegwm_formal_experiment_v1":
        raise ValueError("formal experiment config schema differs")
    expected = {
        "alpha": ALPHA,
        "calibration_negatives": CALIBRATION_NEGATIVES,
        "clean_test_negatives": CLEAN_TEST_NEGATIVES,
        "evaluation_pairs": EVALUATION_PAIRS,
        "checkpoint_interval_seconds": CHECKPOINT_INTERVAL_SECONDS,
        "checkpoint_shard_size": CHECKPOINT_SHARD_SIZE,
        "max_unit_attempts": MAX_UNIT_ATTEMPTS,
    }
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError("formal experiment scalar contract differs")
    if tuple(config.get("conditions", ())) != FORMAL_CONDITIONS:
        raise ValueError("formal attack matrix differs")
    if "rotation_scale" in " ".join(FORMAL_CONDITIONS):
        raise ValueError("rotation+scale must not enter the formal matrix")
    if frozenset(config.get("retryable_operational_codes", ())) != RETRYABLE_OPERATIONAL_CODES:
        raise ValueError("formal retry allowlist differs")
    if config.get("preflight") != {
        "before_formal_unit_state": True,
        "seed": 2027000000,
        "science_denominator": 0,
        "failure_status": "PREFLIGHT_FAILED_RECOVERABLE",
    }:
        raise ValueError("formal preflight contract differs")
    if config.get("engineering_canary") != {
        "science_denominator": 0,
        "drive_tree": "PaperFormal-V1-EngineeringCanary",
        "required_evidence": ["generation", "detection", "checkpoint", "resume"],
    }:
        raise ValueError("engineering canary contract differs")
    if config.get("finalization") != {
        "reconstruction_missing_prerequisite": "WAITING_FOR_PREREQUISITE",
        "missing_required_results": "WAITING_FOR_REQUIRED_RESULTS",
        "incomplete_close_requires_explicit_flag": True,
    }:
        raise ValueError("formal finalization contract differs")
    return config


def _load_prompts(repo_root: Path, paths: Sequence[str]) -> tuple[tuple[str, str], ...]:
    prompts: list[tuple[str, str]] = []
    for relative in paths:
        with (repo_root / relative).open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                prompt_id, prompt = row.get("unit_id"), row.get("prompt")
                if not isinstance(prompt_id, str) or not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("formal prompt source row is invalid")
                prompts.append((prompt_id, prompt))
    if len(prompts) != 64 or len({item[0] for item in prompts}) != 64:
        raise ValueError("formal prompt corpus must contain exactly 64 unique rows")
    return tuple(prompts)


def expand_rosters(repo_root: str | Path, config: Mapping[str, Any]) -> dict[str, tuple[FormalUnit, ...]]:
    root = Path(repo_root)
    prompt_paths = config.get("prompt_sources")
    if not isinstance(prompt_paths, list) or not all(isinstance(item, str) for item in prompt_paths):
        raise ValueError("formal prompt sources must be an ordered path list")
    prompts = _load_prompts(root, prompt_paths)
    definitions = config.get("partitions")
    if not isinstance(definitions, dict):
        raise ValueError("formal partition definitions are missing")
    expected_counts = {
        "threshold_calibration": CALIBRATION_NEGATIVES,
        "clean_negative_test": CLEAN_TEST_NEGATIVES,
        "formal_evaluation_pairs": EVALUATION_PAIRS,
    }
    result: dict[str, tuple[FormalUnit, ...]] = {}
    seen: set[tuple[str, int]] = set()
    for partition, count in expected_counts.items():
        definition = definitions.get(partition)
        if not isinstance(definition, dict) or definition.get("count") != count:
            raise ValueError(f"{partition} denominator differs")
        seed_start = definition.get("seed_start")
        if not isinstance(seed_start, int) or isinstance(seed_start, bool) or seed_start < 0:
            raise ValueError(f"{partition} seed start is invalid")
        units: list[FormalUnit] = []
        for index in range(count):
            prompt_id, prompt = prompts[index % len(prompts)]
            seed = seed_start + index
            key = (prompt_id, seed)
            if key in seen:
                raise ValueError("formal partitions overlap")
            seen.add(key)
            units.append(FormalUnit(
                partition=partition,
                roster_index=index,
                unit_id=f"{partition}-{index + 1:04d}",
                prompt_id=prompt_id,
                prompt=prompt,
                seed=seed,
            ))
        result[partition] = tuple(units)
    return result


def normalized_score(native_score: Any, direction: str) -> float:
    if isinstance(native_score, bool) or not isinstance(native_score, (int, float)):
        raise TypeError("native score must be a real scalar")
    value = float(native_score)
    if not math.isfinite(value):
        raise ValueError("native score must be finite")
    if direction == "higher_is_watermarked":
        return value
    if direction == "lower_is_watermarked":
        return -value
    raise ValueError("score direction is invalid")


def freeze_threshold(scores: Sequence[float]) -> dict[str, Any]:
    if len(scores) != CALIBRATION_NEGATIVES:
        raise ValueError("paper threshold requires exactly 2000 complete scores")
    ordered = sorted(normalized_score(value, "higher_is_watermarked") for value in scores)
    rank = math.ceil((1.0 - ALPHA) * len(ordered))
    tau = ordered[rank - 1]
    return {
        "alpha": ALPHA,
        "calibration_denominator": CALIBRATION_NEGATIVES,
        "estimator": "nearest_rank_empirical_quantile",
        "rank_one_based": rank,
        "tau": tau,
        "decision_rule": "positive_iff_normalized_score_strictly_greater_than_tau",
        "equality_decision": "negative",
    }


def decide(score: Any, tau: Any) -> bool:
    return normalized_score(score, "higher_is_watermarked") > normalized_score(tau, "higher_is_watermarked")


def _beta_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / max(d, tiny)
    h = d
    for step in range(1, 301):
        m, m2 = float(step), 2.0 * step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / max(1.0 + aa * d, tiny)
        c = max(1.0 + aa / c, tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / max(1.0 + aa * d, tiny)
        c = max(1.0 + aa / c, tiny)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            return h
    raise RuntimeError("incomplete beta fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_fraction(a, b, x) / a
    return 1.0 - front * _beta_fraction(b, a, 1.0 - x) / b


def _beta_inverse(probability: float, a: float, b: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(140):
        middle = (low + high) / 2.0
        if _regularized_beta(middle, a, b) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def clopper_pearson(successes: int, trials: int, confidence: float = CONFIDENCE) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("binomial count is invalid")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else _beta_inverse(alpha / 2.0, successes, trials - successes + 1.0)
    upper = 1.0 if successes == trials else _beta_inverse(1.0 - alpha / 2.0, successes + 1.0, trials - successes)
    return lower, upper


def summarize_binary(records: Iterable[Mapping[str, Any]], *, truth_positive: bool, planned: int) -> dict[str, Any]:
    rows = tuple(records)
    if planned <= 0 or len(rows) > planned:
        raise ValueError("planned denominator is invalid")
    scored_rows = tuple(row for row in rows if row.get("terminal_status") == "SCORED")
    failed = sum(row.get("terminal_status") == "OPERATIONAL_FAILURE" for row in rows)
    missing = planned - len(rows)
    positive = sum(row.get("decision") is True for row in scored_rows)
    negative = len(scored_rows) - positive
    successes = positive if truth_positive else positive
    rate = successes / len(scored_rows) if scored_rows else None
    interval = clopper_pearson(successes, len(scored_rows)) if scored_rows else (None, None)
    unresolved = failed + missing
    if truth_positive:
        lower, upper = positive / planned, (positive + unresolved) / planned
        names = {"tp": positive, "fn": negative, "scored_only_tpr": rate, "tpr_ci95": interval,
                 "planned_tpr_lower": lower, "planned_tpr_upper": upper}
    else:
        lower, upper = positive / planned, (positive + unresolved) / planned
        names = {"fp": positive, "tn": negative, "scored_only_fpr": rate, "fpr_ci95": interval,
                 "planned_fpr_lower": lower, "planned_fpr_upper": upper}
    return {
        "n_planned": planned,
        "n_scored": len(scored_rows),
        "n_failed": failed,
        "n_missing": missing,
        "coverage": len(scored_rows) / planned,
        "status": "COMPLETE" if unresolved == 0 else "INCOMPLETE_OPERATIONAL",
        **names,
    }


def empty_binary_summary(*, truth_positive: bool, planned: int) -> dict[str, Any]:
    """Return the frozen all-missing summary for a stage that could not start."""

    return summarize_binary((), truth_positive=truth_positive, planned=planned)


def summarize_quality(records: Iterable[Mapping[str, Any]], *, planned: int) -> dict[str, Any]:
    """Summarize the frozen PSNR/SSIM/LPIPS set without dropping failed pairs."""

    rows = tuple(records)
    if planned <= 0 or len(rows) > planned:
        raise ValueError("quality planned denominator is invalid")
    missing = planned - len(rows)
    metrics: dict[str, dict[str, Any]] = {}
    pair_valid = 0
    for row in rows:
        quality = row.get("quality")
        if row.get("terminal_status") == "SCORED" and isinstance(quality, Mapping) and all(
            isinstance(quality.get(metric), (int, float))
            and not isinstance(quality.get(metric), bool)
            and math.isfinite(float(quality[metric]))
            for metric in QUALITY_METRICS
        ):
            pair_valid += 1
    for metric in QUALITY_METRICS:
        values = [
            float(row["quality"][metric])
            for row in rows
            if row.get("terminal_status") == "SCORED"
            and isinstance(row.get("quality"), Mapping)
            and isinstance(row["quality"].get(metric), (int, float))
            and not isinstance(row["quality"].get(metric), bool)
            and math.isfinite(float(row["quality"][metric]))
        ]
        metrics[metric] = {
            "n_valid": len(values),
            "n_failed": len(rows) - len(values),
            "n_missing": missing,
            "mean": math.fsum(values) / len(values) if values else None,
        }
    return {
        "n_planned_pairs": planned,
        "n_valid_pairs": pair_valid,
        "n_failed_pairs": len(rows) - pair_valid,
        "n_missing_pairs": missing,
        "metrics": metrics,
        "status": "COMPLETE" if pair_valid == planned else "INCOMPLETE_OPERATIONAL",
    }


def load_or_recover_pair(
    clean_path: str | Path,
    watermarked_path: str | Path,
    generate: Callable[[], tuple[Any, Any]],
    load: Callable[[Path], Any],
    write: Callable[[Path, Any], None],
) -> tuple[Any, Any, str]:
    """Recover a deterministic one-file pair residue without replacing either arm."""

    clean_file, watermarked_file = Path(clean_path), Path(watermarked_path)
    clean_exists, watermarked_exists = clean_file.exists(), watermarked_file.exists()
    if clean_exists and watermarked_exists:
        return load(clean_file), load(watermarked_file), "PAIR_REUSED"
    generated_clean, generated_watermarked = generate()
    if clean_exists:
        clean = load(clean_file)
    else:
        write(clean_file, generated_clean)
        clean = load(clean_file)
    if watermarked_exists:
        watermarked = load(watermarked_file)
    else:
        write(watermarked_file, generated_watermarked)
        watermarked = load(watermarked_file)
    mode = "PAIR_PARTIAL_RECOVERED" if clean_exists or watermarked_exists else "PAIR_CREATED"
    return clean, watermarked, mode


def apply_attack(image: Image.Image, condition: str) -> Image.Image:
    rgb = image.convert("RGB")
    if condition == "clean_no_attack":
        return rgb.copy()
    if condition == "jpeg_q50":
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=50, subsampling=2, optimize=False, progressive=False)
        return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    width, height = rgb.size
    if condition == "resize_50_bicubic_restore":
        small = (max(1, round(width * 0.50)), max(1, round(height * 0.50)))
        return rgb.resize(small, Image.Resampling.BICUBIC).resize((width, height), Image.Resampling.BICUBIC)
    if condition == "center_crop_80_restore":
        scale = math.sqrt(0.80)
        crop_width = max(1, round(width * scale))
        crop_height = max(1, round(height * scale))
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        return rgb.crop((left, top, left + crop_width, top + crop_height)).resize(
            (width, height), Image.Resampling.BICUBIC
        )
    if condition == "gaussian_blur_sigma_1px":
        return rgb.filter(ImageFilter.GaussianBlur(radius=1.0))
    if condition == "rotation_10_bicubic_reflect_center_crop_v1":
        array = np.asarray(rgb, dtype=np.uint8)
        theta = math.radians(10.0)
        half_width, half_height = (width - 1) / 2.0, (height - 1) / 2.0
        pad_x = max(0, math.ceil(abs(math.cos(theta)) * half_width + abs(math.sin(theta)) * half_height + 2 - half_width))
        pad_y = max(0, math.ceil(abs(math.sin(theta)) * half_width + abs(math.cos(theta)) * half_height + 2 - half_height))
        if pad_x >= width or pad_y >= height:
            raise ValueError("rotation input is outside the reflect-padding domain")
        padded = np.pad(array, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="reflect")
        center = (pad_x + half_width, pad_y + half_height)
        rotated = Image.fromarray(padded).rotate(
            10.0, resample=Image.Resampling.BICUBIC, center=center, fillcolor=(0, 0, 0)
        )
        return rotated.crop((pad_x, pad_y, pad_x + width, pad_y + height))
    raise ValueError(f"unknown formal condition: {condition}")


def execute_with_frozen_retry(
    unit_id: str,
    callback: Callable[[int], Mapping[str, Any]],
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for attempt_number in range(1, MAX_UNIT_ATTEMPTS + 1):
        try:
            payload = dict(callback(attempt_number))
            if "normalized_score" in payload:
                normalized_score(payload["normalized_score"], "higher_is_watermarked")
            elif payload.get("artifact_status") != "GENERATED":
                raise ValueError("successful unit payload requires a score or generated artifact")
            attempts.append({"attempt": attempt_number, "status": "SCORED"})
            return {
                "unit_id": unit_id,
                "terminal_status": "SCORED",
                "attempts": attempts,
                **payload,
            }
        except Exception as raw_error:
            error = (
                raw_error
                if isinstance(raw_error, OperationalUnitError)
                else classify_operational_exception(raw_error, "unit_callback")
            )
            if error is None:
                attempts.append({
                    "attempt": attempt_number,
                    "status": "OPERATIONAL_FAILURE",
                    "failure_code": "UNCLASSIFIED_OPERATIONAL",
                    "failure_stage": "unit_callback",
                    "error": f"{type(raw_error).__name__}: {raw_error}",
                    "retryable_by_contract": False,
                })
                return {
                    "unit_id": unit_id,
                    "terminal_status": "OPERATIONAL_FAILURE",
                    "attempts": attempts,
                    "failure_code": "UNCLASSIFIED_OPERATIONAL",
                    "failure_stage": "unit_callback",
                    "error": f"{type(raw_error).__name__}: {raw_error}",
                }
            retryable = error.code in RETRYABLE_OPERATIONAL_CODES
            attempts.append({
                "attempt": attempt_number,
                "status": "OPERATIONAL_FAILURE",
                "failure_code": error.code,
                "failure_stage": error.stage,
                "error": error.detail,
                "retryable_by_contract": retryable,
            })
            if not retryable or attempt_number == MAX_UNIT_ATTEMPTS:
                return {
                    "unit_id": unit_id,
                    "terminal_status": "OPERATIONAL_FAILURE",
                    "attempts": attempts,
                    "failure_code": error.code,
                    "failure_stage": error.stage,
                    "error": error.detail,
                }
    raise AssertionError("frozen retry loop did not terminate")


class FormalRunStore:
    """Stable JOB_ID state with unit micro-commits and append-only checkpoints."""

    def __init__(self, root: str | Path, identity: Mapping[str, Any], unit_ids: Sequence[str]) -> None:
        self.root = Path(root)
        self.identity = dict(identity)
        self.unit_ids = tuple(unit_ids)
        if not self.unit_ids or len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError("formal unit ids must be nonempty and unique")
        required = ("schema_version", "job_id", "run_id", "method_id", "stage", "expected_exact")
        if any(not isinstance(self.identity.get(name), str) or not self.identity[name] for name in required):
            raise ValueError("formal run identity is incomplete")
        self._rows_cache: list[dict[str, Any]] | None = None

    @property
    def final_path(self) -> Path:
        return self.root / "final_result.json"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "run_config.json"
        contract = {"identity": self.identity, "unit_ids": list(self.unit_ids)}
        if path.exists():
            if _read_json(path) != contract:
                raise RuntimeError("stable JOB_ID contract drift")
        else:
            _write_json_create_only(path, contract)

    def completed_result(self) -> dict[str, Any] | None:
        if not self.final_path.exists():
            return None
        value = _read_json(self.final_path)
        if not isinstance(value, dict) or value.get("identity") != self.identity:
            raise RuntimeError("formal final result identity differs")
        return value

    def rows(self) -> tuple[dict[str, Any], ...]:
        if self._rows_cache is not None:
            return tuple(self._rows_cache)
        directory = self.root / "units"
        if not directory.exists():
            self._rows_cache = []
            return ()
        paths = tuple(sorted(directory.glob("*.json")))
        if len(paths) > len(self.unit_ids):
            raise RuntimeError("formal committed unit count exceeds the plan")
        rows: list[dict[str, Any]] = []
        for index, path in enumerate(paths):
            unit_id = self.unit_ids[index]
            if path.name != f"{index:06d}.json":
                raise RuntimeError("formal committed unit prefix has a gap")
            row = _read_json(path)
            if not isinstance(row, dict) or row.get("unit_id") != unit_id:
                raise RuntimeError("formal unit record identity differs")
            if row.get("terminal_status") not in {"SCORED", "OPERATIONAL_FAILURE"}:
                raise RuntimeError("formal unit record is not terminal")
            rows.append(row)
        self._rows_cache = rows
        return tuple(rows)

    def commit(self, row: Mapping[str, Any]) -> None:
        existing = self.rows()
        index = len(existing)
        if index >= len(self.unit_ids) or row.get("unit_id") != self.unit_ids[index]:
            raise RuntimeError("formal unit commit is out of order")
        committed = dict(row)
        _write_json_create_only(self.root / "units" / f"{index:06d}.json", committed)
        if self._rows_cache is None:
            raise AssertionError("formal row cache was not initialized")
        self._rows_cache.append(committed)
        self.write_progress("unit_committed")

    def _checkpoint_paths(self) -> tuple[Path, ...]:
        directory = self.root / "checkpoints"
        if not directory.exists():
            return ()
        paths = tuple(sorted(directory.glob("checkpoint-*.json")))
        for index, path in enumerate(paths):
            if path.name != f"checkpoint-{index:06d}.json":
                raise RuntimeError("formal checkpoint sequence is not contiguous")
        return paths

    def publish_checkpoint(self, reason: str) -> Path:
        rows = self.rows()
        paths = self._checkpoint_paths()
        path = self.root / "checkpoints" / f"checkpoint-{len(paths):06d}.json"
        payload = {
            "identity": self.identity,
            "checkpoint_sequence": len(paths),
            "committed_unit_count": len(rows),
            "reason": reason,
            "published_at_unix_seconds": time.time(),
            "terminal_status_counts": {
                "SCORED": sum(row["terminal_status"] == "SCORED" for row in rows),
                "OPERATIONAL_FAILURE": sum(row["terminal_status"] == "OPERATIONAL_FAILURE" for row in rows),
            },
        }
        _write_json_create_only(path, payload)
        self.write_progress("checkpoint_published")
        return path

    def write_progress(self, phase: str) -> None:
        _write_json_replaceable(self.root / "progress.json", {
            "monitoring_only": True,
            "identity": self.identity,
            "phase": phase,
            "committed_unit_count": len(self.rows()),
            "planned_unit_count": len(self.unit_ids),
            "updated_at_unix_seconds": time.time(),
        })

    def run(self, callback: Callable[[str, int], Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        self.initialize()
        if self.completed_result() is not None:
            return self.rows()
        rows = self.rows()
        checkpoint_anchor = time.monotonic()
        for index in range(len(rows), len(self.unit_ids)):
            unit_id = self.unit_ids[index]
            row = execute_with_frozen_retry(unit_id, lambda attempt: callback(unit_id, attempt))
            self.commit(row)
            committed = index + 1
            elapsed = time.monotonic() - checkpoint_anchor
            if committed % CHECKPOINT_SHARD_SIZE == 0 or elapsed >= CHECKPOINT_INTERVAL_SECONDS:
                self.publish_checkpoint("shard_end" if committed % CHECKPOINT_SHARD_SIZE == 0 else "two_hour_interval")
                checkpoint_anchor = time.monotonic()
        if len(self.rows()) % CHECKPOINT_SHARD_SIZE:
            self.publish_checkpoint("final_partial_shard")
        return self.rows()

    def finalize(self, payload: Mapping[str, Any]) -> Path:
        self.initialize()
        if self.final_path.exists():
            self.completed_result()
            return self.final_path
        rows = self.rows()
        if len(rows) != len(self.unit_ids):
            raise RuntimeError("formal final result requires every planned unit terminal")
        _write_json_create_only(self.final_path, {
            "identity": self.identity,
            "planned_unit_count": len(self.unit_ids),
            "records": list(rows),
            **dict(payload),
        })
        self.write_progress("terminal_result_published")
        return self.final_path


__all__ = [
    "ALPHA", "CALIBRATION_NEGATIVES", "CHECKPOINT_INTERVAL_SECONDS",
    "CHECKPOINT_SHARD_SIZE", "CLEAN_TEST_NEGATIVES", "EVALUATION_PAIRS",
    "FORMAL_CONDITIONS", "FormalRunStore", "FormalUnit", "MAX_UNIT_ATTEMPTS",
    "OperationalUnitError", "PreflightFailed", "QUALITY_METRICS",
    "RETRYABLE_OPERATIONAL_CODES", "apply_attack", "classify_operational_exception",
    "clopper_pearson", "decide", "empty_binary_summary", "execute_job_preflight",
    "execute_with_frozen_retry", "expand_rosters", "freeze_threshold",
    "load_formal_config", "load_or_recover_pair", "normalized_score",
    "publish_job_state", "raise_classified_operational", "summarize_binary",
    "summarize_quality",
]
