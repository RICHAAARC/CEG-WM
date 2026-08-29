"""Identity- and denominator-bound P1 proxy runner; no model or external I/O."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from cegwm.method.geometry_v4_proxy import apply_proxy_attack, detect_proxy, write_proxy
from cegwm.protocol.geometry_v4 import GEOMETRY_V4_METHOD_ID, GEOMETRY_V4_PROTOCOL_ID, GeometryV4Observation
from cegwm.protocol.geometry_v4_proxy import (
    P1_ATTACKS,
    P1_DIGEST,
    P1_RUNNER_ID,
    P1_SOURCE_ID,
    P1_SOURCE_SHAPE,
    P1_SPLITS,
    load_p1_proxy,
)
from cegwm.shared.keys import normalize_detection_key

_ROOT = Path(__file__).resolve().parents[1]
_ARM_NAMES = (
    "marked_correct_key",
    "attacked_unwatermarked_negative",
    "same_unit_wrong_key",
)


def load_runner_contract() -> Mapping[str, object]:
    return load_p1_proxy(_ROOT)


def _require_distinct_keys(detection_key: str | bytes, wrong_key: str | bytes) -> None:
    if normalize_detection_key(detection_key) == normalize_detection_key(wrong_key):
        raise ValueError("Geometry-V4 wrong-key control must differ after normalization")


def generate_procedural_source(seed: int) -> tuple[np.ndarray, dict[str, object]]:
    """Generate the sole deterministic, low-frequency P1 ordinary-RGB source."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Geometry-V4 source seed must be a non-negative integer")
    height, width, _ = P1_SOURCE_SHAPE
    generator = np.random.Generator(np.random.PCG64(seed))
    yy, xx = np.mgrid[:height, :width]
    x = xx / float(width - 1)
    y = yy / float(height - 1)
    rgb = np.empty(P1_SOURCE_SHAPE, dtype=np.float64)
    for channel in range(3):
        phase_x, phase_y = generator.uniform(0.0, 2.0 * math.pi, size=2)
        slope_x, slope_y = generator.uniform(-0.035, 0.035, size=2)
        low_frequency = 0.018 * np.sin(2.0 * math.pi * (channel + 1) * x + phase_x)
        low_frequency += 0.014 * np.cos(2.0 * math.pi * (channel + 1) * y + phase_y)
        rgb[..., channel] = 0.5 + slope_x * (x - 0.5) + slope_y * (y - 0.5) + low_frequency
    rgb = np.clip(rgb, 0.0, 1.0)
    framed = (
        f"{P1_SOURCE_ID}|{height}|{width}|{seed}|".encode("ascii")
        + np.asarray(rgb, dtype="<f4").tobytes(order="C")
    )
    source = {
        "source_id": P1_SOURCE_ID,
        "seed": seed,
        "shape": P1_SOURCE_SHAPE,
        "image_identity_sha256": hashlib.sha256(framed).hexdigest(),
    }
    return rgb, source


def _identity(mode: str, split: str, seed: int, attack: str, source: Mapping[str, object] | None) -> dict[str, object]:
    if split not in P1_SPLITS or isinstance(seed, bool) or seed not in P1_SPLITS[split]:
        raise ValueError("Geometry-V4 proxy seed is outside the selected split")
    if attack not in P1_ATTACKS:
        raise ValueError("Geometry-V4 proxy attack is outside the frozen roster")
    if mode == "engineering_canary":
        formal_member = False
        formal_split = None
        unit_id = f"engineering_canary:{seed}:{attack}"
    elif mode == f"{split}_full":
        formal_member = True
        formal_split = split
        unit_id = f"{split}_full:{seed}:{attack}"
    else:
        raise ValueError("Geometry-V4 proxy execution mode differs")
    return {
        "runner_id": P1_RUNNER_ID,
        "config_digest": P1_DIGEST,
        "execution_mode": mode,
        "formal_split": formal_split,
        "formal_denominator_member": formal_member,
        "seed": seed,
        "attack": attack,
        "unit_id": unit_id,
        "source": dict(source) if source is not None else None,
    }


def plan_full(split: str) -> tuple[dict[str, object], ...]:
    """Materialize the immutable 8x16 full identity plan without running outcomes."""

    load_runner_contract()
    if split not in P1_SPLITS:
        raise ValueError("Geometry-V4 proxy split must be P1D or P1C")
    sources = {seed: generate_procedural_source(seed)[1] for seed in P1_SPLITS[split]}
    plan = tuple(
        _identity(f"{split}_full", split, seed, attack, sources[seed])
        for seed in P1_SPLITS[split]
        for attack in P1_ATTACKS
    )
    if len(plan) != 128:
        raise RuntimeError("Geometry-V4 full plan denominator differs")
    return plan


def _stopped_detection(error: str) -> dict[str, object]:
    observation = GeometryV4Observation(None, (), 0, 0.0, "STOPPED")
    return {
        "method_id": GEOMETRY_V4_METHOD_ID,
        "protocol_id": GEOMETRY_V4_PROTOCOL_ID,
        "H_hat": observation.H_hat,
        "corners_hat": observation.corners_hat,
        "support": observation.support,
        "reliability": observation.reliability,
        "status": observation.status,
        "diagnostics": {"error": error},
    }


def _stopped_arm(error: str) -> dict[str, object]:
    return {"failure": error, "detection": _stopped_detection(error), "evaluation": None}


def _stopped_arms(error: str) -> dict[str, dict[str, object]]:
    return {name: _stopped_arm(error) for name in _ARM_NAMES}


def _evaluate_geometry(detection: Mapping[str, object], truth: tuple[float, ...]) -> dict[str, float] | None:
    """Compare attacked-to-canonical H on attacked unit-square corners."""

    estimated = detection.get("H_hat")
    if estimated is None:
        return None
    estimate_h = np.asarray(estimated, dtype=np.float64).reshape(3, 3)
    truth_h = np.asarray(truth, dtype=np.float64).reshape(3, 3)
    errors = []
    for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        point = np.asarray((x, y, 1.0), dtype=np.float64)
        estimate_point = estimate_h @ point
        truth_point = truth_h @ point
        estimate_point /= estimate_point[2]
        truth_point /= truth_point[2]
        errors.append(math.hypot(*(estimate_point[:2] - truth_point[:2])) / math.sqrt(2.0))
    return {"corner_error_mean_diagonal": float(np.mean(errors)), "corner_error_max_diagonal": float(np.max(errors))}


def _detect_arm(attacked: np.ndarray, key: str | bytes) -> dict[str, object]:
    try:
        return {"failure": None, "detection": detect_proxy(attacked, key), "evaluation": None}
    except Exception as error:  # every physical arm remains represented
        message = f"{type(error).__name__}: {error}"
        return _stopped_arm(message)


def _execute_generated_unit(
    detection_key: str | bytes,
    wrong_key: str | bytes,
    *,
    mode: str,
    split: str,
    seed: int,
    attack: str,
) -> dict[str, object]:
    identity = _identity(mode, split, seed, attack, None)
    record: dict[str, object] = {
        **identity,
        "failure": None,
        "budget": None,
        "arms": _stopped_arms("not_started"),
    }
    try:
        rgb, source = generate_procedural_source(seed)
        record["source"] = source
    except Exception as error:
        message = f"source:{type(error).__name__}: {error}"
        record["failure"] = message
        record["arms"] = _stopped_arms(message)
        return record
    try:
        marked, budget = write_proxy(rgb, detection_key)
        attacked_marked, truth = apply_proxy_attack(marked, attack)
        attacked_unwatermarked, negative_truth = apply_proxy_attack(rgb, attack)
        if truth != negative_truth:
            raise RuntimeError("matching attack truth differs between arms")
    except Exception as error:
        message = f"writer_or_attack:{type(error).__name__}: {error}"
        record["failure"] = message
        record["arms"] = _stopped_arms(message)
        return record
    correct = _detect_arm(attacked_marked, detection_key)
    negative = _detect_arm(attacked_unwatermarked, detection_key)
    wrong = _detect_arm(attacked_marked, wrong_key)
    # All blind detector calls complete before attacked-to-canonical truth is evaluated.
    if correct["failure"] is None:
        correct["evaluation"] = _evaluate_geometry(correct["detection"], truth)  # type: ignore[arg-type]
    record["budget"] = budget
    record["arms"] = {
        "marked_correct_key": correct,
        "attacked_unwatermarked_negative": negative,
        "same_unit_wrong_key": wrong,
    }
    if any(arm["failure"] is not None for arm in record["arms"].values()):  # type: ignore[union-attr]
        record["failure"] = "one_or_more_detector_arms_stopped"
    return record


def run_canary(
    detection_key: str | bytes,
    wrong_key: str | bytes,
    *,
    subset: Iterable[tuple[int, str]],
) -> tuple[dict[str, object], ...]:
    """Run an explicit non-formal subset that cannot impersonate a full split."""

    load_runner_contract()
    _require_distinct_keys(detection_key, wrong_key)
    selected = tuple(subset)
    if not selected or len(selected) >= 128 or len(set(selected)) != len(selected):
        raise ValueError("Geometry-V4 engineering canary requires a unique strict subset")
    resolved: list[tuple[str, int, str]] = []
    for seed, attack in selected:
        split = next((name for name, seeds in P1_SPLITS.items() if seed in seeds), "")
        if not split:
            raise ValueError("Geometry-V4 canary seed is outside every frozen seed roster")
        _identity("engineering_canary", split, seed, attack, None)
        resolved.append((split, seed, attack))
    return tuple(
        _execute_generated_unit(
            detection_key,
            wrong_key,
            mode="engineering_canary",
            split=split,
            seed=seed,
            attack=attack,
        )
        for split, seed, attack in resolved
    )


def run_full(
    detection_key: str | bytes, wrong_key: str | bytes, *, split: str
) -> tuple[dict[str, object], ...]:
    """Internally generate the exact 128-unit full split with no external subset."""

    load_runner_contract()
    _require_distinct_keys(detection_key, wrong_key)
    plan = plan_full(split)
    records = tuple(
        _execute_generated_unit(
            detection_key,
            wrong_key,
            mode=f"{split}_full",
            split=split,
            seed=int(item["seed"]),
            attack=str(item["attack"]),
        )
        for item in plan
    )
    if len(records) != 128:
        raise RuntimeError("Geometry-V4 full execution denominator differs")
    return records
