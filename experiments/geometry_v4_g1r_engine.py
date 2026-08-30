"""V4-G1R fixed CPU synthetic canary and roster guards; no model fallback."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from cegwm.method.geometry_v4_g1r import detect_g1r, write_g1r_rgb
from cegwm.method.geometry_v4_proxy import _sample_h, _similarity_h
from cegwm.protocol.geometry_v4_g1r import ATTACKS, load_contract, require_split, unsafe_geometry

CPU_KEY = b"geometry-v4-g1r-cpu-key-v1"
CPU_WRONG_KEY = b"geometry-v4-g1r-cpu-wrong-key-v1"
CPU_CARRIER_IDS = ("gradient_shapes", "crosshatch", "radial_objects", "colored_texture")


@dataclass(frozen=True, slots=True)
class BlindArms:
    correct: Mapping[str, object]
    wrong: Mapping[str, object]
    negative: Mapping[str, object]


def build_real_roster(repo_root: str | Path, split: str) -> tuple[tuple[int, str, str], ...]:
    """Return exactly one complete frozen split; subset inputs do not exist."""
    return require_split(load_contract(repo_root), split)


def _carrier(carrier_id: str, size: int = 128) -> np.ndarray:
    if carrier_id not in CPU_CARRIER_IDS:
        raise ValueError("unfrozen V4-G1R CPU carrier")
    yy, xx = np.mgrid[:size, :size]
    x, y = (xx + .5) / size, (yy + .5) / size
    if carrier_id == "gradient_shapes":
        base = .34 + .18 * x + .12 * y + .07 * np.sin(2 * math.pi * (2 * x + y))
        object_mask = ((x - .30) ** 2 + (y - .62) ** 2 < .10**2) | ((abs(x - .72) < .11) & (abs(y - .35) < .08))
        channels = (base + .12 * object_mask, base - .04 * object_mask, base + .02)
    elif carrier_id == "crosshatch":
        hatch = .07 * np.sin(2 * math.pi * 3 * x) * np.cos(2 * math.pi * 5 * y)
        diagonal = .05 * np.sin(2 * math.pi * (4 * x + 3 * y))
        channels = (.48 + hatch + diagonal, .42 - hatch + .5 * diagonal, .38 + .4 * hatch - diagonal)
    elif carrier_id == "radial_objects":
        radius = np.sqrt((x - .52) ** 2 + (y - .48) ** 2)
        rings = .06 * np.cos(2 * math.pi * 5 * radius)
        bars = .08 * ((abs(x - .25) < .035) | (abs(y - .76) < .04))
        channels = (.36 + rings + bars, .50 - .6 * rings, .44 + .5 * rings - .4 * bars)
    else:
        texture = .05 * np.sin(2 * math.pi * (7 * x + 5 * y)) + .04 * np.cos(2 * math.pi * (5 * x - 6 * y))
        blobs = .09 * np.exp(-((x - .28) ** 2 + (y - .30) ** 2) / .012) - .07 * np.exp(-((x - .72) ** 2 + (y - .67) ** 2) / .02)
        channels = (.44 + texture + blobs, .38 - .5 * texture + .6 * blobs, .52 + .3 * texture - .7 * blobs)
    return np.clip(np.stack(channels, axis=-1), 0.08, 0.92).astype(np.float64)


def _attack(rgb: np.ndarray, attack: str) -> tuple[np.ndarray, np.ndarray]:
    if attack == "identity":
        canonical_to_attacked = np.eye(3, dtype=np.float64)
    elif attack == "rotation_5":
        canonical_to_attacked = _similarity_h(5.0, 1.0)
    elif attack == "scale_0.9":
        canonical_to_attacked = _similarity_h(0.0, .9)
    elif attack == "translation_0.08_0":
        canonical_to_attacked = _similarity_h(0.0, 1.0, .08, 0.0)
    elif attack == "crop_0.9":
        canonical_to_attacked = _similarity_h(0.0, 1.0 / .9)
    else:
        raise ValueError("unfrozen V4-G1R attack")
    attacked_to_canonical = np.linalg.inv(canonical_to_attacked)
    attacked = _sample_h(rgb, attacked_to_canonical, 0.0)
    return np.clip(attacked, 0.0, 1.0), attacked_to_canonical


def _blind_arms(attacked_marked: np.ndarray, attacked_negative: np.ndarray) -> BlindArms:
    """Freeze all three attacked-RGB/key-only arms before truth exists downstream."""
    return BlindArms(
        correct=detect_g1r(attacked_marked, CPU_KEY),
        wrong=detect_g1r(attacked_marked, CPU_WRONG_KEY),
        negative=detect_g1r(attacked_negative, CPU_KEY),
    )


def _points(homography: np.ndarray) -> np.ndarray:
    points = []
    for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (.5, .5)):
        point = homography @ np.asarray((x, y, 1.0), dtype=np.float64)
        points.append(point[:2] / point[2])
    return np.asarray(points)


def _angle_scale(homography: np.ndarray) -> tuple[float, float]:
    return math.degrees(math.atan2(float(homography[1, 0]), float(homography[0, 0]))), math.hypot(float(homography[0, 0]), float(homography[1, 0]))


def _truth_metrics(arm: Mapping[str, object], truth_attacked_to_canonical: np.ndarray) -> Mapping[str, float]:
    if arm.get("status") != "RELIABLE" or arm.get("H_hat") is None:
        return {"mapped_corner_error": 0.0, "center_reprojection_error": 0.0, "rotation_abs_error_degrees": 0.0, "log_scale_abs_error": 0.0}
    estimate = np.asarray(arm["H_hat"], dtype=np.float64).reshape(3, 3)
    predicted, truth = _points(estimate), _points(truth_attacked_to_canonical)
    distances = np.linalg.norm(predicted - truth, axis=1) / math.sqrt(2.0)
    predicted_angle, predicted_scale = _angle_scale(estimate)
    truth_angle, truth_scale = _angle_scale(truth_attacked_to_canonical)
    rotation_error = abs((predicted_angle - truth_angle + 90.0) % 180.0 - 90.0)
    return {"mapped_corner_error": float(np.max(distances[:4])), "center_reprojection_error": float(distances[4]), "rotation_abs_error_degrees": rotation_error, "log_scale_abs_error": abs(math.log(predicted_scale) - math.log(truth_scale))}


def _evaluate_frozen_arms(arms: BlindArms, truth_attacked_to_canonical: np.ndarray) -> Mapping[str, object]:
    evaluated = {}
    for name in ("correct", "wrong", "negative"):
        arm = getattr(arms, name)
        metrics = _truth_metrics(arm, truth_attacked_to_canonical)
        evaluated[name] = {"status": arm["status"], "support": arm["support"], "truth_metrics": metrics, "unsafe": unsafe_geometry(str(arm["status"]), metrics)}
    return evaluated


def run_cpu_canary() -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for carrier_id in CPU_CARRIER_IDS:
        ordinary = _carrier(carrier_id)
        marked, budget = write_g1r_rgb(ordinary, CPU_KEY)
        for attack in ATTACKS:
            base = {"carrier": carrier_id, "attack": attack, "failure": None, "budget": budget}
            try:
                attacked_marked, truth = _attack(marked, attack)
                attacked_negative, negative_truth = _attack(ordinary, attack)
                if not np.allclose(truth, negative_truth):
                    raise RuntimeError("V4-G1R synthetic arm transforms differ")
                arms = _blind_arms(attacked_marked, attacked_negative)
                evaluation = _evaluate_frozen_arms(arms, truth)
                records.append({**base, "arms": evaluation})
            except Exception as error:
                records.append({**base, "failure": f"{type(error).__name__}: {error}", "arms": None})
    if len(records) != 20:
        raise RuntimeError("V4-G1R CPU denominator differs")
    return tuple(records)


def summarize_cpu_canary(records: tuple[Mapping[str, object], ...]) -> Mapping[str, object]:
    if len(records) != 20 or tuple(record["attack"] for record in records) != ATTACKS * 4:
        raise ValueError("V4-G1R CPU records differ from fixed 4x5 roster")
    failures = sum(record["failure"] is not None for record in records)
    correct_safe = sum(record["arms"] is not None and record["arms"]["correct"]["status"] == "RELIABLE" and not record["arms"]["correct"]["unsafe"] for record in records)
    correct_unsafe = sum(record["arms"] is not None and record["arms"]["correct"]["unsafe"] for record in records)
    wrong_unsafe = sum(record["arms"] is not None and record["arms"]["wrong"]["unsafe"] for record in records)
    negative_unsafe = sum(record["arms"] is not None and record["arms"]["negative"]["unsafe"] for record in records)
    by_attack = {attack: sum(record["attack"] == attack and record["arms"] is not None and record["arms"]["correct"]["status"] == "RELIABLE" and not record["arms"]["correct"]["unsafe"] for record in records) for attack in ATTACKS}
    passed = failures == 0 and correct_safe >= 18 and correct_unsafe == wrong_unsafe == negative_unsafe == 0 and all(value >= 3 for value in by_attack.values())
    return {"stage": "V4-G1R", "evidence": "synthetic_only", "formal_denominator": 0, "units": 20, "failures": failures, "correct_safe_reliable": correct_safe, "correct_unsafe": correct_unsafe, "wrong_unsafe": wrong_unsafe, "negative_unsafe": negative_unsafe, "correct_safe_by_attack": by_attack, "status": "CPU_ENGINEERING_EXIT" if passed else "CPU_METHOD_PARTIAL"}
