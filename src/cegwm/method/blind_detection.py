"""BlindDetection-V1 statistic and fixed-N threshold contract.

This module is deliberately image-agnostic.  Runtime observation and geometry
live in :mod:`cegwm.runtime.blind_detection`; only content statistics can make
a positive decision here.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


BLIND_STATISTIC_ID = "registered_weighted_joint_minus_exact16_wrong_key_max_v1"
BLIND_THRESHOLD_SCHEMA_ID = "cegwm_blind_detection_v1_threshold_asset_v1"
BLIND_PRODUCTION_RUNTIME_ID = (
    "content_v9_weighted_joint_exact16__ordinary_rgb__syncseal_torchscript_direct_v1"
)
BLIND_DEV_DENOMINATOR = 256
BLIND_WRONG_KEY_COUNT = 16
BLIND_DEV_DISJOINT_FROM = (
    "geometry_v7_development",
    "future_paper_calibration",
    "future_paper_test",
)
_HEX16 = re.compile(r"[0-9a-f]{16}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def encode_binary64(value: Any, name: str = "value") -> str:
    return struct.pack(">d", _finite(value, name)).hex()


def decode_binary64(value: Any, name: str = "value") -> float:
    if not isinstance(value, str) or _HEX16.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase binary64 big-endian hex")
    scalar = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(scalar):
        raise ValueError(f"{name} binary64 value must be finite")
    return scalar


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class BlindStatistic:
    registered_weighted_joint: float
    wrong_key_weighted_joint: tuple[float, ...]
    wrong_key_max: float
    value: float


def registered_minus_wrong_key_max(
    registered_weighted_joint: Any,
    wrong_key_weighted_joint: Sequence[Any],
) -> BlindStatistic:
    """Return registered weighted-joint minus the exact-16 wrong-key maximum."""

    registered = _finite(registered_weighted_joint, "registered_weighted_joint")
    wrong = tuple(
        _finite(value, f"wrong_key_weighted_joint[{index}]")
        for index, value in enumerate(wrong_key_weighted_joint)
    )
    if len(wrong) != BLIND_WRONG_KEY_COUNT:
        raise ValueError("blind statistic requires exactly 16 wrong-key scores")
    wrong_max = max(wrong)
    statistic = registered - wrong_max
    if not math.isfinite(statistic):
        raise ValueError("blind statistic must be finite")
    return BlindStatistic(registered, wrong, wrong_max, statistic)


def statistic_from_weighted_scores(weighted_scores: Mapping[str, Any]) -> BlindStatistic:
    if not isinstance(weighted_scores, Mapping):
        raise TypeError("weighted scores must be a mapping")
    required = ("registered", *(f"wrong_{index:02d}" for index in range(16)))
    if tuple(weighted_scores) != required:
        raise ValueError("weighted scores must contain registered then exact wrong_00..wrong_15")
    return registered_minus_wrong_key_max(
        weighted_scores["registered"],
        tuple(weighted_scores[f"wrong_{index:02d}"] for index in range(16)),
    )


@dataclass(frozen=True, slots=True)
class BlindCalibrationUnit:
    unit_id: str
    source_stratum: str
    image_ref: str
    base_image_id: str

    def __post_init__(self) -> None:
        for name in ("unit_id", "source_stratum", "image_ref", "base_image_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"calibration {name} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class BlindCalibrationRoster:
    units: tuple[BlindCalibrationUnit, ...]
    disjoint_from: tuple[str, ...] = BLIND_DEV_DISJOINT_FROM

    def __post_init__(self) -> None:
        if len(self.units) != BLIND_DEV_DENOMINATOR:
            raise ValueError("blind development roster must contain exactly 256 units")
        if any(not isinstance(unit, BlindCalibrationUnit) for unit in self.units):
            raise TypeError("blind development roster contains an invalid unit")
        if tuple(unit.unit_id for unit in self.units) != tuple(
            dict.fromkeys(unit.unit_id for unit in self.units)
        ):
            raise ValueError("blind development unit ids must be unique and ordered")
        if len({unit.base_image_id for unit in self.units}) != BLIND_DEV_DENOMINATOR:
            raise ValueError("transforms of one base image cannot count as independent units")
        if self.disjoint_from != BLIND_DEV_DISJOINT_FROM:
            raise ValueError("blind development disjointness declaration differs")


@dataclass(frozen=True, slots=True)
class BlindCalibrationRow:
    roster_index: int
    unit_id: str
    source_stratum: str
    pre_score: float | None
    post_score: float | None
    geometry_outcome: str
    method_complete: bool
    operational_error: str | None = None

    @property
    def z(self) -> float:
        pre = _finite(self.pre_score, "pre_score")
        if self.post_score is None:
            return pre
        return max(pre, _finite(self.post_score, "post_score"))


@dataclass(frozen=True, slots=True)
class BlindReplayRow:
    roster_index: int
    unit_id: str
    source_stratum: str
    pre_score: float | None
    post_score: float | None
    route: str
    positive: bool
    recovered: bool
    method_complete: bool
    operational_error: str | None = None


@dataclass(frozen=True, slots=True)
class BlindThresholdAsset:
    payload: Mapping[str, Any]
    json_bytes: bytes
    test_only: bool = False

    def __post_init__(self) -> None:
        if self.test_only:
            expected = {
                "schema_version": "cegwm_blind_detection_v1_explicit_test_only_threshold_v1",
                "tau_blind_be_hex": self.payload.get("tau_blind_be_hex"),
            }
            if dict(self.payload) != expected or self.json_bytes != stable_json_bytes(expected):
                raise ValueError("explicit test-only threshold asset fields differ")
            decode_binary64(expected["tau_blind_be_hex"], "tau_blind")
            return
        _validate_threshold_payload(self.payload, self.json_bytes)

    @property
    def tau_blind(self) -> float:
        return decode_binary64(self.payload["tau_blind_be_hex"], "tau_blind")


def build_test_only_threshold_asset(tau_blind: Any) -> BlindThresholdAsset:
    """Create a conspicuously test-only threshold; never a production asset."""

    payload = {
        "schema_version": "cegwm_blind_detection_v1_explicit_test_only_threshold_v1",
        "tau_blind_be_hex": encode_binary64(tau_blind, "tau_blind"),
    }
    return BlindThresholdAsset(payload, stable_json_bytes(payload), test_only=True)


def validate_calibration_rows(
    rows: Sequence[BlindCalibrationRow], roster: BlindCalibrationRoster
) -> tuple[BlindCalibrationRow, ...]:
    received = tuple(rows)
    if len(received) != BLIND_DEV_DENOMINATOR:
        raise ValueError("threshold generation requires all fixed 256 rows")
    for index, (row, unit) in enumerate(zip(received, roster.units, strict=True)):
        if not isinstance(row, BlindCalibrationRow):
            raise TypeError(f"row[{index}] must be BlindCalibrationRow")
        if (row.roster_index, row.unit_id, row.source_stratum) != (
            index,
            unit.unit_id,
            unit.source_stratum,
        ):
            raise ValueError("calibration row identity/order differs from frozen roster")
        if not row.method_complete or row.operational_error is not None:
            raise ValueError("calibration rows contain an operational interruption")
        _finite(row.pre_score, f"row[{index}].pre_score")
        if row.geometry_outcome == "RECOVERED":
            _finite(row.post_score, f"row[{index}].post_score")
        elif row.post_score is not None:
            raise ValueError("fail-closed geometry row must not contain a post score")
        if row.geometry_outcome not in {
            "RECOVERED",
            "NO_H",
            "INVALID_H",
            "RECTIFICATION_ERROR",
        }:
            raise ValueError("calibration geometry outcome differs")
        _finite(row.z, f"row[{index}].z")
    return received


def candidate_tau_blind(
    rows: Sequence[BlindCalibrationRow], roster: BlindCalibrationRoster
) -> float:
    checked = validate_calibration_rows(rows, roster)
    return max(row.z for row in checked)


def validate_full_system_replay(
    rows: Sequence[BlindReplayRow],
    roster: BlindCalibrationRoster,
    calibration_rows: Sequence[BlindCalibrationRow],
    tau_blind: float,
) -> tuple[BlindReplayRow, ...]:
    received = tuple(rows)
    validate_calibration_rows(calibration_rows, roster)
    tau = _finite(tau_blind, "tau_blind")
    if len(received) != BLIND_DEV_DENOMINATOR:
        raise ValueError("full-system replay requires all fixed 256 rows")
    for index, (row, unit) in enumerate(zip(received, roster.units, strict=True)):
        if not isinstance(row, BlindReplayRow):
            raise TypeError(f"replay[{index}] must be BlindReplayRow")
        if (row.roster_index, row.unit_id, row.source_stratum) != (
            index,
            unit.unit_id,
            unit.source_stratum,
        ):
            raise ValueError("full-system replay identity/order differs from frozen roster")
        if not row.method_complete or row.operational_error is not None:
            raise ValueError("full-system replay contains an operational interruption")
        if not isinstance(row.route, str) or not row.route:
            raise ValueError("full-system replay route is absent")
        if not isinstance(row.positive, bool) or not isinstance(row.recovered, bool):
            raise TypeError("full-system replay decisions must be boolean")
        pre = _finite(row.pre_score, f"replay[{index}].pre_score")
        if row.route == "DIRECT_POSITIVE":
            if not pre > tau or not row.positive or row.recovered or row.post_score is not None:
                raise ValueError("full-system replay direct route semantics differ")
        elif row.route == "GEOMETRY_RECOVERED":
            post = _finite(row.post_score, f"replay[{index}].post_score")
            if pre > tau or not row.recovered or row.positive != (post > tau):
                raise ValueError("full-system replay recovered route semantics differ")
        elif row.route in {
            "GEOMETRY_NO_H", "GEOMETRY_FAIL_CLOSED", "RECTIFICATION_FAIL_CLOSED"
        }:
            if pre > tau or row.positive or row.recovered or row.post_score is not None:
                raise ValueError("full-system replay fail-closed route semantics differ")
        else:
            raise ValueError("full-system replay complete route differs")
    if sum(row.positive for row in received) != 0:
        raise ValueError("full-system replay must produce exactly 0/256 empirical false positives")
    return received


def _replay_rows_payload(rows: Sequence[BlindReplayRow]) -> list[dict[str, Any]]:
    return [
        {
            "method_complete": row.method_complete,
            "operational_error": row.operational_error,
            "post_score_be_hex": (
                None if row.post_score is None else encode_binary64(row.post_score, "post_score")
            ),
            "positive": row.positive,
            "pre_score_be_hex": encode_binary64(row.pre_score, "pre_score"),
            "recovered": row.recovered,
            "roster_index": row.roster_index,
            "route": row.route,
            "source_stratum": row.source_stratum,
            "unit_id": row.unit_id,
        }
        for row in rows
    ]


def build_threshold_asset(
    rows: Sequence[BlindCalibrationRow],
    roster: BlindCalibrationRoster,
    full_system_replay: Sequence[BlindReplayRow],
    *,
    producer_exact: str,
    calibration_key_digest: str,
) -> BlindThresholdAsset:
    """Freeze only after a distinct complete full-system 0/256 replay."""

    if not isinstance(producer_exact, str) or _HEX40.fullmatch(producer_exact) is None:
        raise ValueError("threshold producer exact must be a lowercase 40-hex commit")
    if not isinstance(calibration_key_digest, str) or _HEX64.fullmatch(calibration_key_digest) is None:
        raise ValueError("threshold calibration key digest must be lowercase 64-hex")
    checked = validate_calibration_rows(rows, roster)
    tau = candidate_tau_blind(checked, roster)
    replay = validate_full_system_replay(full_system_replay, roster, checked, tau)
    payload = dict(
        sorted(
            {
                "denominator": BLIND_DEV_DENOMINATOR,
                "calibration_key_digest": calibration_key_digest,
                "disjoint_from": list(BLIND_DEV_DISJOINT_FROM),
                "decision_rule": "positive_iff_m_strictly_greater_than_tau_blind",
                "fpr_experiment": "independent_unwatermarked_primary_null_only",
                "full_system_replay_rows": _replay_rows_payload(replay),
                "producer_exact": producer_exact,
                "production_runtime_id": BLIND_PRODUCTION_RUNTIME_ID,
                "replay_false_positives": 0,
                "replay_kind": "fresh_full_system_pre_geometry_post_same_roster_images_key_runtime",
                "schema_version": BLIND_THRESHOLD_SCHEMA_ID,
                "statistic_id": BLIND_STATISTIC_ID,
                "tau_blind_be_hex": encode_binary64(tau, "tau_blind"),
                "value_dtype": "IEEE-754_binary64_big_endian_hex",
                "wrong_key_attribution_experiment": "separate_fixed_denominator_experiment",
            }.items()
        )
    )
    raw = stable_json_bytes(payload)
    return BlindThresholdAsset(payload, raw)


def replay_empirical_false_positives(
    rows: Sequence[BlindCalibrationRow], asset: BlindThresholdAsset
) -> int:
    """Arithmetic cached-row cross-check; never the required full-system replay."""

    if not isinstance(asset, BlindThresholdAsset):
        raise TypeError("arithmetic cross-check requires a BlindThresholdAsset")
    if len(rows) != BLIND_DEV_DENOMINATOR:
        raise ValueError("threshold replay requires the complete 256-row denominator")
    tau = asset.tau_blind
    return sum(row.z > tau for row in rows)


def load_threshold_asset(path: str | Path) -> BlindThresholdAsset:
    asset_path = Path(path)
    if not asset_path.is_file():
        raise FileNotFoundError(
            "BlindDetection-V1 production threshold is absent; run the authorized N_dev=256 "
            "calibration and 0/256 replay before detection"
        )
    raw = asset_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("blind threshold asset must be UTF-8 JSON") from error
    return BlindThresholdAsset(payload, raw)


def _validate_threshold_payload(payload: Any, raw: Any) -> None:
    if not isinstance(payload, dict) or tuple(payload) != tuple(sorted(payload)):
        raise ValueError("blind threshold asset fields must use stable order")
    required = {
        "calibration_key_digest", "decision_rule", "denominator", "disjoint_from", "fpr_experiment",
        "full_system_replay_rows", "producer_exact", "replay_false_positives",
        "production_runtime_id", "replay_kind", "schema_version",
        "statistic_id", "tau_blind_be_hex", "value_dtype",
        "wrong_key_attribution_experiment",
    }
    if set(payload) != required:
        raise ValueError("blind threshold asset fields differ")
    if payload["schema_version"] != BLIND_THRESHOLD_SCHEMA_ID:
        raise ValueError("blind threshold schema differs")
    if payload["statistic_id"] != BLIND_STATISTIC_ID:
        raise ValueError("blind threshold statistic differs")
    if payload["denominator"] != BLIND_DEV_DENOMINATOR:
        raise ValueError("blind threshold denominator differs")
    if tuple(payload["disjoint_from"]) != BLIND_DEV_DISJOINT_FROM:
        raise ValueError("blind threshold disjointness differs")
    if payload["decision_rule"] != "positive_iff_m_strictly_greater_than_tau_blind":
        raise ValueError("blind threshold decision rule differs")
    if payload["replay_false_positives"] != 0:
        raise ValueError("blind threshold 0/256 replay gate differs")
    if payload["replay_kind"] != "fresh_full_system_pre_geometry_post_same_roster_images_key_runtime":
        raise ValueError("blind threshold replay kind differs")
    if payload["production_runtime_id"] != BLIND_PRODUCTION_RUNTIME_ID:
        raise ValueError("blind threshold production runtime differs")
    if payload["fpr_experiment"] != "independent_unwatermarked_primary_null_only":
        raise ValueError("blind threshold FPR experiment differs")
    if payload["wrong_key_attribution_experiment"] != "separate_fixed_denominator_experiment":
        raise ValueError("wrong-key attribution must remain separate from FPR")
    if not isinstance(payload["producer_exact"], str) or _HEX40.fullmatch(payload["producer_exact"]) is None:
        raise ValueError("blind threshold producer exact differs")
    if (
        not isinstance(payload["calibration_key_digest"], str)
        or _HEX64.fullmatch(payload["calibration_key_digest"]) is None
    ):
        raise ValueError("blind threshold calibration key digest differs")
    replay_rows = payload["full_system_replay_rows"]
    if not isinstance(replay_rows, list) or len(replay_rows) != BLIND_DEV_DENOMINATOR:
        raise ValueError("blind threshold full-system replay rows differ")
    decode_binary64(payload["tau_blind_be_hex"], "tau_blind")
    if payload["value_dtype"] != "IEEE-754_binary64_big_endian_hex":
        raise ValueError("blind threshold dtype differs")
    if stable_json_bytes(payload) != raw:
        raise ValueError("blind threshold asset must use stable JSON")


__all__ = [
    "BLIND_DEV_DENOMINATOR", "BLIND_DEV_DISJOINT_FROM", "BLIND_PRODUCTION_RUNTIME_ID",
    "BLIND_STATISTIC_ID",
    "BLIND_THRESHOLD_SCHEMA_ID", "BLIND_WRONG_KEY_COUNT", "BlindCalibrationRoster",
    "BlindCalibrationRow", "BlindCalibrationUnit", "BlindReplayRow", "BlindStatistic",
    "BlindThresholdAsset", "build_test_only_threshold_asset", "build_threshold_asset",
    "candidate_tau_blind", "decode_binary64",
    "encode_binary64", "load_threshold_asset",
    "registered_minus_wrong_key_max", "replay_empirical_false_positives",
    "stable_json_bytes", "statistic_from_weighted_scores", "validate_calibration_rows",
    "validate_full_system_replay",
]
