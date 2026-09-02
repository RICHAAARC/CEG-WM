from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from cegwm.method.blind_detection import (
    BLIND_DEV_DISJOINT_FROM,
    BlindCalibrationRoster,
    BlindCalibrationRow,
    BlindCalibrationUnit,
    build_test_only_threshold_asset,
    build_threshold_asset,
    decode_binary64,
    encode_binary64,
    load_threshold_asset,
    registered_minus_wrong_key_max,
    replay_empirical_false_positives,
    statistic_from_weighted_scores,
)
from cegwm.runtime import blind_detection as runtime

pytestmark = pytest.mark.unit


def _roster() -> BlindCalibrationRoster:
    return BlindCalibrationRoster(
        tuple(
            BlindCalibrationUnit(f"unit-{index:03d}", f"source-{index % 4}", f"img/{index}", f"base-{index}")
            for index in range(256)
        )
    )


def _rows() -> tuple[BlindCalibrationRow, ...]:
    return tuple(
        BlindCalibrationRow(
            index,
            f"unit-{index:03d}",
            f"source-{index % 4}",
            float(index) / 10.0,
            float(index) / 10.0 + 0.25 if index % 2 else None,
            "RECOVERED" if index % 2 else "NO_H",
            True,
        )
        for index in range(256)
    )


def test_exact_16_registered_minus_wrong_max_and_mapping_order() -> None:
    wrong = tuple(float(index) / 16.0 for index in range(16))
    score = registered_minus_wrong_key_max(2.0, wrong)
    assert score.wrong_key_max == 15.0 / 16.0
    assert score.value == 2.0 - 15.0 / 16.0
    mapping = {"registered": 2.0, **{f"wrong_{i:02d}": wrong[i] for i in range(16)}}
    assert statistic_from_weighted_scores(mapping) == score
    with pytest.raises(ValueError, match="exactly 16"):
        registered_minus_wrong_key_max(1.0, wrong[:-1])
    reordered = dict(reversed(tuple(mapping.items())))
    with pytest.raises(ValueError, match="registered then exact"):
        statistic_from_weighted_scores(reordered)


def test_roster_fixed_before_scoring_and_base_images_are_independent() -> None:
    roster = _roster()
    assert len(roster.units) == 256
    assert roster.disjoint_from == BLIND_DEV_DISJOINT_FROM
    duplicated = list(roster.units)
    duplicated[-1] = BlindCalibrationUnit("unit-255", "source-3", "different", "base-0")
    with pytest.raises(ValueError, match="cannot count as independent"):
        BlindCalibrationRoster(tuple(duplicated))


def test_threshold_is_exact_binary64_max_z_and_strict_replay_zero(tmp_path: Path) -> None:
    rows = _rows()
    asset = build_threshold_asset(rows, _roster(), producer_exact="a" * 40)
    assert asset.tau_blind == max(row.z for row in rows)
    assert decode_binary64(encode_binary64(asset.tau_blind)) == asset.tau_blind
    assert replay_empirical_false_positives(rows, asset) == 0
    assert asset.payload["wrong_key_attribution_experiment"] == "separate_fixed_denominator_experiment"
    path = tmp_path / "threshold.json"
    path.write_bytes(asset.json_bytes)
    assert load_threshold_asset(path).payload == asset.payload
    assert json.loads(path.read_text(encoding="ascii"))["replay_false_positives"] == 0


def test_threshold_generation_retains_no_h_but_blocks_missing_or_operational_rows() -> None:
    roster = _roster()
    rows = list(_rows())
    assert rows[0].geometry_outcome == "NO_H" and rows[0].z == rows[0].pre_score
    rows[1] = BlindCalibrationRow(1, "unit-001", "source-1", 0.1, None, "RECOVERED", False, "content_post:failed")
    with pytest.raises(ValueError, match="operational interruption"):
        build_threshold_asset(rows, roster, producer_exact="b" * 40)
    with pytest.raises(ValueError, match="all fixed 256"):
        build_threshold_asset(_rows()[:-1], roster, producer_exact="b" * 40)


def test_production_threshold_absence_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="N_dev=256"):
        load_threshold_asset(tmp_path / "missing.json")


def test_repository_has_no_placeholder_production_threshold() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/blind_detection/blind_detection_v1.json").read_text(encoding="utf-8")
    )
    assert config["threshold_asset_state"].startswith("ABSENT_UNTIL_AUTHORIZED")
    assert "tau_blind" not in config
    assert not (root / config["threshold_asset"]).exists()


def test_detection_signature_and_source_have_no_forbidden_runtime_inputs() -> None:
    assert tuple(inspect.signature(runtime.detect_watermark).parameters) == ("image", "key", "assets")
    source = inspect.getsource(runtime.detect_watermark)
    for forbidden in (
        "paired_null", "original_image", "private_latent", "embed_record",
        "stored_h", "truth", "prompt", "seed",
    ):
        assert forbidden not in source.lower()
    assert "b_low" not in source.lower()


def test_test_only_threshold_can_express_equality_without_default_zero() -> None:
    test_asset = build_test_only_threshold_asset(1.25)
    assert test_asset.test_only
    assert test_asset.tau_blind == 1.25
    assert not (1.25 > test_asset.tau_blind)
