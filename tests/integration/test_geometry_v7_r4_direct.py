from __future__ import annotations

from argparse import Namespace
import ast
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from experiments import run_geometry_v7_r4_direct as entry
from cegwm.geometry_v7.contracts import (
    CANONICAL_CORNERS_NORMALIZED,
    GeometryEstimate,
    GeometryStatus,
)
from cegwm.geometry_v7.r0 import ContentScore
from cegwm.geometry_v7.r2 import R2_CONDITION_IDS, R2_DEV_UNIT_IDS, R2_TEST_UNIT_IDS
from cegwm.geometry_v7.r3 import R3_B_LOW


ALL_UNITS = R2_DEV_UNIT_IDS + R2_TEST_UNIT_IDS
SAFE = {
    "core_rotation_neg15": set(range(8)),
    "core_rotation_pos15": set(range(8)),
    "core_fixed_canvas_zoom_0_8": {0, 2, 4, 5, 7},
    "core_fixed_canvas_zoom_1_2": {4},
    "core_translation_pos32_x": {0, 4},
    "core_translation_neg32_x": {0, 1, 4, 5},
    "core_translation_pos32_y": {0, 1, 2, 4, 5},
    "core_translation_neg32_y": {0, 4, 5},
    "core_offset_crop_rescale": {0, 1, 4},
    "core_composite_c0_85_t16_neg16_r10": {0},
}
FALSE_POSITIVE = {
    "core_fixed_canvas_zoom_1_2": {0, 5},
    "core_translation_pos32_x": {5},
}
DIRECT_NEGATIVE = {
    "core_fixed_canvas_zoom_0_8": {1, 3, 6},
    "core_fixed_canvas_zoom_1_2": {1, 6},
    "core_translation_pos32_x": {1, 6},
    "core_translation_neg32_x": {2},
    "core_translation_pos32_y": {3},
    "core_translation_neg32_y": {1},
    "core_offset_crop_rescale": {2, 5},
    "core_composite_c0_85_t16_neg16_r10": {1, 4},
}


def _paired(margin: float):
    return {
        "gate_a_margin": margin, "gate_b_margin": margin,
        "margin": margin, "positive": margin > 0.0,
    }


def _fixture_payloads():
    pre, real, features, outcomes = [], [], [], []
    dev, test = [], []
    for condition in R2_CONDITION_IDS:
        for index, unit in enumerate(ALL_UNITS):
            split = "dev" if index < 4 else "test"
            direct_negative = index in DIRECT_NEGATIVE.get(condition, set())
            safe = index in SAFE[condition]
            false_positive = index in FALSE_POSITIVE.get(condition, set())
            s0 = R3_B_LOW - 1.0 if direct_negative else -1.0
            identity = {"condition_id": condition, "unit_id": unit}
            pre.append({
                **identity,
                "membership_from_old_r1b": "N_recovery_negative",
                "pre_scores_from_old_r1b": {
                    "positive_cg_vs_g": _paired(s0),
                    "negative_g_vs_u": _paired(-1.0),
                },
            })
            real.append({
                **identity,
                "errors": [],
                "positive_score_delta": 2.0 if safe else -1.0,
                "recovered_negative": safe,
                "scores": {
                    "positive_cg_vs_g": _paired(1.0 if safe or false_positive else -1.0),
                    "negative_g_vs_u": _paired(1.0 if false_positive else -1.0),
                },
            })
            features.append({**identity, "mandatory_valid": True, "area_ratio": 1.0})
            outcomes.append({**identity, "complete": True, "safe": safe})
            rotation = condition in ("core_rotation_neg15", "core_rotation_pos15")
            zoom = condition == "core_fixed_canvas_zoom_0_8"
            row = {
                "split": split, **identity,
                "route": "DIRECT_NEGATIVE" if direct_negative else "BOUNDARY",
                "r2_selector_accepted": rotation or (zoom and not direct_negative),
                "old_cycle_score_px": 4.0 if zoom and not direct_negative else None,
                "decision": {
                    "pure_rotation_gate": rotation,
                    "regime": {
                        "valid": True,
                        "angle_degrees": -15.0 if condition == "core_rotation_neg15" else 15.0 if rotation else 0.0,
                        "scale": 1.25 if zoom else 1.0,
                        "translation": 0.0, "perspective": 0.0,
                    },
                },
            }
            (dev if split == "dev" else test).append(row)
    repair = {
        "schema": entry.reliable_runner.REPAIR_SCHEMA,
        "exact": entry.reliable_runner.REPAIR_EXACT,
        "status": "R1B_REPAIR_REAL_H_NOT_END_TO_END_READY",
        "frozen_old_membership_records": pre,
        "real_h_records": real,
    }
    r2 = {
        "schema": entry.reliable_runner.R2_SCHEMA,
        "exact": entry.reliable_runner.R2_EXACT,
        "status": "R2_SELECTIVE_RISK_FAILED",
        "feature_rows": features, "outcome_rows": outcomes,
    }
    advanced = {
        "schema": entry.reliable_runner.ADVANCED_SCHEMA,
        "exact": entry.reliable_runner.ADVANCED_EXACT,
        "status": "R3_ADVANCED_ENGINEERING_TEST40_RECORDED",
        "inputs": {
            "r1a": {"producer_exact": entry.reliable_runner.R1A_EXACT},
            "r1b_repair": {"producer_exact": entry.reliable_runner.REPAIR_EXACT},
            "r2": {"producer_exact": entry.reliable_runner.R2_EXACT},
            "old_r3": {"producer_exact": entry.reliable_runner.OLD_R3_EXACT},
        },
        "development_decisions": dev,
        "existing_test40_decisions": test,
    }
    return repair, r2, advanced


def _write_result(root: Path, payload):
    root.mkdir()
    (root / "result.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _content_score(weighted_joint: float):
    wrong = (-10.0,) * 16
    return ContentScore(0.0, 0.0, weighted_joint, wrong, wrong, wrong)


def _geometry():
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return GeometryEstimate(
        GeometryStatus.UNRELIABLE, 0.0,
        CANONICAL_CORNERS_NORMALIZED, CANONICAL_CORNERS_NORMALIZED,
        identity, True, True, None,
    )


def test_replay_mechanically_reconstructs_all_80_and_frozen_metrics():
    repair, r2, advanced = _fixture_payloads()
    rows = entry._replay_rows(repair, r2, advanced)
    aggregates = entry._aggregate_replay(rows)
    assert len(rows) == 80
    assert aggregates["global"] == {
        "fixed_denominator": 80,
        "direct_positive_count": 0,
        "direct_negative_count": 14,
        "boundary_count": 66,
        "recovered_count": 66,
        "final_positive_count": 43,
        "paired_negative_false_positive_count": 3,
        "safe_recovered_count": 40,
        "recovery_failure_count": 0,
        "content_only_final_positive_count": 0,
    }
    assert aggregates["seven_main_conditions"]["safe_recovered_count"] == 36
    assert aggregates["seven_main_conditions"]["dev"]["safe_recovered_count"] == 18
    assert aggregates["seven_main_conditions"]["test"]["safe_recovered_count"] == 18
    assert aggregates["reliable_ablation"]["accepted_count"] == 21
    assert aggregates["reliable_ablation"]["safe_recovered_count"] == 21
    assert all("raw" in row and "errors" in row for row in rows)


def test_replay_loader_validates_real_shape_order_and_bindings(tmp_path):
    payloads = _fixture_payloads()
    roots = tuple(tmp_path / name for name in ("repair", "r2", "advanced"))
    for root, payload in zip(roots, payloads, strict=True):
        _write_result(root, payload)
    repair, r2, advanced = entry.reliable_runner._validate_inputs(*roots)
    assert len(entry._replay_rows(repair, r2, advanced)) == 80

    advanced["existing_test40_decisions"][0], advanced["existing_test40_decisions"][1] = (
        advanced["existing_test40_decisions"][1],
        advanced["existing_test40_decisions"][0],
    )
    (roots[2] / "result.json").write_text(json.dumps(advanced), encoding="utf-8")
    with pytest.raises(ValueError, match="split identity/order"):
        entry.reliable_runner._validate_inputs(*roots)


def test_callback_fixed_roster_same_scores_and_call_counts():
    image = Image.new("RGB", (512, 512), (128, 128, 128))
    inputs = tuple(
        entry.CallbackInput(condition, unit, image, image, image)
        for condition, unit in entry.CALLBACK_ROSTER
    )
    content_calls = 0

    def scorer(_image):
        nonlocal content_calls
        phase = content_calls % 6
        content_calls += 1
        return _content_score(-1.0 if phase == 2 else 1.0 if phase == 5 else 0.0)

    records = entry._callback_records(inputs, scorer=scorer, detector=lambda _image: _geometry())
    assert len(records) == 7 and content_calls == 42
    assert all(record["route"] == "BOUNDARY" for record in records)
    assert all(record["runtime"]["recovered"] is True for record in records)
    assert all(record["runtime"]["final_positive"] is True for record in records)
    assert all(record["final_negative_false_positive"] is False for record in records)
    assert all(record["call_counts"] == {
        "score_rgb": 2, "content_scorer": 6, "detect_geometry": 1,
        "paired_null_rectifications": 2, "candidate_rectifications": 1,
    } for record in records)


def test_callback_r1a_validation_is_identity_only(tmp_path):
    raw = [
        {"condition_id": condition, "unit_id": unit, "geometry": {"ignored": True}}
        for condition in R2_CONDITION_IDS for unit in ALL_UNITS
    ]
    raw.extend({"condition_id": "sanity", "unit_id": f"sanity-{index}"} for index in range(24))
    result = {
        "exact": entry.R1A_EXACT,
        "status": "R1A_BLOCKING_METHOD_CANARY_PASSED",
        "blocking_method_canary_passed": True,
        "r0_input": {
            "producer_exact": entry.R0_EXACT,
            "selected_residual_strength_multiplier": 0.75,
            "ordered_evaluation_cg_inputs": [{"unit_id": unit} for unit in ALL_UNITS],
        },
        "fixed_counts": {"core_conditions": 10, "units_per_condition": 8, "records": 104},
        "raw_records": raw,
    }
    root = tmp_path / "r1a"
    _write_result(root, result)
    entry._validate_r1a_identity(tmp_path, root, ALL_UNITS)

    raw[0], raw[1] = raw[1], raw[0]
    (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="identity/order"):
        entry._validate_r1a_identity(tmp_path, root, ALL_UNITS)


def test_callback_errors_keep_fixed_seven_and_never_retry():
    image = Image.new("RGB", (512, 512), (128, 128, 128))
    inputs = tuple(
        entry.CallbackInput(condition, unit, image, image, image)
        for condition, unit in entry.CALLBACK_ROSTER
    )
    detect_calls = 0

    def detector(_image):
        nonlocal detect_calls
        detect_calls += 1
        raise RuntimeError("detector interrupted")

    records = entry._callback_records(
        inputs, scorer=lambda _image: _content_score(-1.0), detector=detector
    )
    assert len(records) == 7 and detect_calls == 7
    assert all(record["operational_interruption"] for record in records)
    assert all(record["call_counts"]["score_rgb"] == 1 for record in records)
    assert all(record["call_counts"]["detect_geometry"] == 1 for record in records)

    scorer_calls = 0

    def interrupted_scorer(_image):
        nonlocal scorer_calls
        scorer_calls += 1
        if scorer_calls == 2:
            raise RuntimeError("content scorer interrupted")
        return _content_score(0.0)

    first = entry._callback_record(
        inputs[0], scorer=interrupted_scorer, detector=lambda _image: _geometry()
    )
    assert first["operational_interruption"] is True
    assert first["call_counts"]["score_rgb"] == 1
    assert first["call_counts"]["content_scorer"] == 2
    assert first["call_counts"]["detect_geometry"] == 0

    setup_rows = entry._operational_callback_rows(RuntimeError("setup"))
    assert len(setup_rows) == 7
    assert [(row["condition_id"], row["unit_id"]) for row in setup_rows] == list(entry.CALLBACK_ROSTER)
    assert all(not row["attempted"] and row["operational_interruption"] for row in setup_rows)


def test_boundary_post_score_failure_has_null_final_fp_and_completed_recovery():
    image = Image.new("RGB", (512, 512), (128, 128, 128))
    item = entry.CallbackInput(*entry.CALLBACK_ROSTER[0], image, image, image)
    scorer_calls = 0

    def scorer(_image):
        nonlocal scorer_calls
        scorer_calls += 1
        if scorer_calls == 4:
            raise RuntimeError("post score interrupted")
        phase = (scorer_calls - 1) % 6
        return _content_score(-1.0 if phase == 2 else 0.0)

    record = entry._callback_record(item, scorer=scorer, detector=lambda _image: _geometry())
    assert record["route"] == "BOUNDARY"
    assert record["runtime"]["recovered"] is True
    assert record["runtime"]["post_score"] is None
    assert record["final_negative_false_positive"] is None
    assert record["call_counts"] == {
        "score_rgb": 2, "content_scorer": 4, "detect_geometry": 1,
        "paired_null_rectifications": 2, "candidate_rectifications": 1,
    }
    assert record["operational_interruption"] is True
    assert len(record["score_snapshots"]) == 1


def test_run_callback_preserves_render_failure_and_processes_unaffected(monkeypatch, tmp_path):
    from experiments import run_geometry_v7_r1b as repair_runner

    image = Image.new("RGB", (512, 512), (128, 128, 128))
    rendered = tuple(
        entry.CallbackInput(
            condition, unit,
            None if index == 0 else image,
            None if index == 0 else image,
            None if index == 0 else image,
            ("attack_render:OSError",) if index == 0 else (),
        )
        for index, (condition, unit) in enumerate(entry.CALLBACK_ROSTER)
    )
    monkeypatch.setattr(
        repair_runner, "_load_r0_inputs",
        lambda *_args: tuple(SimpleNamespace(unit_id=unit) for unit in ALL_UNITS),
    )
    monkeypatch.setattr(entry, "_validate_r1a_identity", lambda *_args: None)
    monkeypatch.setattr(entry, "_render_callback_inputs", lambda _inputs: rendered)
    content_calls = 0

    def scorer(_image):
        nonlocal content_calls
        phase = content_calls % 6
        content_calls += 1
        return _content_score(-1.0 if phase == 2 else 1.0 if phase == 5 else 0.0)

    monkeypatch.setattr(
        entry, "_setup_real_callbacks",
        lambda *_args: (scorer, lambda _image: _geometry()),
    )
    args = Namespace(
        repo_root=str(tmp_path), r0_artifact_root=str(tmp_path / "r0"),
        r1a_artifact_root=str(tmp_path / "r1a"),
        syncseal_checkpoint=str(tmp_path / "syncseal.pt"),
    )
    payload = entry._run_callback(args, "1" * 40)
    assert payload["status"] == entry.OPERATIONAL_STATUS
    assert len(payload["rows"]) == 7
    assert payload["rows"][0]["condition_id"] == entry.CALLBACK_ROSTER[0][0]
    assert payload["rows"][0]["unit_id"] == entry.CALLBACK_ROSTER[0][1]
    assert payload["rows"][0]["errors"] == ["attack_render:OSError"]
    assert payload["rows"][0]["attempted"] is False
    assert all(row["attempted"] for row in payload["rows"][1:])
    assert all(row["runtime"]["final_positive"] for row in payload["rows"][1:])
    assert content_calls == 36


def test_callback_runtime_call_has_no_outer_metadata_arguments():
    tree = ast.parse(Path(entry.__file__).read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "detect_direct_and_recover"
    ]
    assert len(calls) == 1
    assert len(calls[0].args) == 1
    assert {keyword.arg for keyword in calls[0].keywords} == {"score_rgb", "detect_geometry"}
    assert all(name not in Path(entry.__file__).read_text(encoding="utf-8").split("def _callback_record", 1)[1].split("def _callback_records", 1)[0]
               for name in ("truth", "attack_label", "membership", "post_outcome"))


def test_create_only_result_and_cli_mode_contract(tmp_path, monkeypatch):
    result = tmp_path / "result"
    payload = entry._replay_payload(
        exact="1" * 40,
        roots=(tmp_path / "repair", tmp_path / "r2", tmp_path / "advanced"),
        rows=(), aggregates={},
    )
    entry._write_result(result, payload)
    assert json.loads((result / "result.json").read_text())["status"] == entry.REPLAY_STATUS
    with pytest.raises(FileExistsError):
        entry._write_result(result, payload)

    execution_result = tmp_path / "execution-result"
    monkeypatch.setattr(entry.reliable_runner, "_git_exact", lambda *_args: "1" * 40)
    monkeypatch.setattr(entry, "_run_replay", lambda _args, exact: {
        **payload, "exact": exact,
    })
    args = Namespace(
        mode="replay", repo_root=str(tmp_path), expected_exact="1" * 40,
        result_dir=str(execution_result), r1b_repair_root="repair",
        r2_root="r2", advanced_r3_root="advanced",
        r0_artifact_root=None, r1a_artifact_root=None, syncseal_checkpoint=None,
    )
    stored = entry.execute(args)
    assert stored["status"] == entry.REPLAY_STATUS
    assert (execution_result / "result.json").is_file()
