"""Denominator-preserving P1D/P1C proxy runner; no model or external I/O."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from cegwm.method.geometry_v4_proxy import apply_proxy_attack, detect_proxy, write_proxy
from cegwm.protocol.geometry_v4_proxy import P1_ATTACKS, P1_DIGEST, P1_RUNNER_ID, P1_SPLITS, load_p1_proxy

_ROOT = Path(__file__).resolve().parents[1]


def load_runner_contract() -> Mapping[str, object]:
    return load_p1_proxy(_ROOT)


def enumerate_unit_identities(split: str) -> tuple[dict[str, object], ...]:
    config = load_runner_contract()
    if split not in P1_SPLITS:
        raise ValueError("Geometry-V4 proxy split must be P1D or P1C")
    attacks = tuple(config["attacks"])  # type: ignore[index]
    return tuple(
        {
            "runner_id": P1_RUNNER_ID,
            "config_digest": P1_DIGEST,
            "split": split,
            "seed": seed,
            "attack": attack,
            "unit_id": f"{split}:{seed}:{attack}",
        }
        for seed in P1_SPLITS[split]
        for attack in attacks
    )


def _unit_identity(split: str, seed: int, attack: str) -> dict[str, object]:
    if split not in P1_SPLITS or isinstance(seed, bool) or seed not in P1_SPLITS[split]:
        raise ValueError("Geometry-V4 proxy seed is outside the selected split")
    if attack not in P1_ATTACKS:
        raise ValueError("Geometry-V4 proxy attack is outside the frozen roster")
    load_runner_contract()
    return {
        "runner_id": P1_RUNNER_ID,
        "config_digest": P1_DIGEST,
        "split": split,
        "seed": seed,
        "attack": attack,
        "unit_id": f"{split}:{seed}:{attack}",
    }


def _evaluate_geometry(detection: Mapping[str, object], truth: tuple[float, ...]) -> dict[str, float] | None:
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
        return {"failure": None, "detection": detect_proxy(attacked, key)}
    except Exception as error:  # every physical arm remains represented
        return {"failure": f"{type(error).__name__}: {error}", "detection": None}


def run_unit(
    rgb: np.ndarray,
    detection_key: str | bytes,
    wrong_key: str | bytes,
    *,
    split: str,
    seed: int,
    attack: str,
) -> dict[str, object]:
    """Run three blind arms; attack truth is read only after detection returns."""

    identity = _unit_identity(split, seed, attack)
    record: dict[str, object] = {**identity, "failure": None, "budget": None, "arms": {}}
    try:
        marked, budget = write_proxy(rgb, detection_key)
        attacked_marked, truth = apply_proxy_attack(marked, attack)
        attacked_unwatermarked, negative_truth = apply_proxy_attack(rgb, attack)
        if truth != negative_truth:
            raise RuntimeError("matching attack truth differs between arms")
    except Exception as error:
        record["failure"] = f"{type(error).__name__}: {error}"
        return record
    correct = _detect_arm(attacked_marked, detection_key)
    negative = _detect_arm(attacked_unwatermarked, detection_key)
    wrong = _detect_arm(attacked_marked, wrong_key)
    # The detector calls above are complete before truth is used for evaluation.
    if correct["detection"] is not None:
        correct["evaluation"] = _evaluate_geometry(correct["detection"], truth)  # type: ignore[arg-type]
    else:
        correct["evaluation"] = None
    record["budget"] = budget
    record["arms"] = {
        "marked_correct_key": correct,
        "attacked_unwatermarked_negative": negative,
        "same_unit_wrong_key": wrong,
    }
    return record


def run_split(
    images: Mapping[int, np.ndarray],
    detection_key: str | bytes,
    wrong_key: str | bytes,
    *,
    split: str,
    attacks: Iterable[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """Run a declared subset while retaining missing-image failures in its denominator."""

    config = load_runner_contract()
    if split not in P1_SPLITS:
        raise ValueError("Geometry-V4 proxy split must be P1D or P1C")
    selected = tuple(config["attacks"] if attacks is None else attacks)  # type: ignore[index]
    if len(set(selected)) != len(selected) or any(attack not in P1_ATTACKS for attack in selected):
        raise ValueError("Geometry-V4 proxy attack selection differs")
    records: list[dict[str, object]] = []
    for seed in P1_SPLITS[split]:
        for attack in selected:
            if seed not in images:
                identity = _unit_identity(split, seed, attack)
                records.append({**identity, "failure": "missing_image", "budget": None, "arms": {}})
                continue
            records.append(
                run_unit(images[seed], detection_key, wrong_key, split=split, seed=seed, attack=attack)
            )
    return tuple(records)
