from __future__ import annotations

import ast
import argparse
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from experiments import run_geometry_v7_r1b as runner
from cegwm.geometry_v7.contracts import (
    CANONICAL_CORNERS_NORMALIZED,
    homography_observed_to_canonical,
)
from cegwm.geometry_v7.r0 import ContentScore, R0Arm
from cegwm.geometry_v7.r1a import (
    R1A_CORE_CONDITIONS,
    R1A_SANITY_CONDITIONS,
    apply_homography,
    corner_rmse,
    truth_correspondences,
)
from cegwm.geometry_v7.r1b import (
    R1B_OPERATIONAL_FAILURE,
    R1B_REPAIR_METHOD_NOT_READY,
    R1B_REPAIR_PIXEL_GRID,
    evaluate_lambda_unit,
    freeze_pre_recovery_record,
    scored_triplet,
)
from cegwm.protocol.content_chain import load_content_chain_contract


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _score(weighted_joint: float, wrong: float = -2.0) -> ContentScore:
    return ContentScore(
        0.0,
        0.0,
        weighted_joint,
        (0.0,) * 16,
        (0.0,) * 16,
        (wrong,) * 16,
    )


def _content_payload(score: ContentScore, *, gate_a: bool) -> dict[str, object]:
    payload = {
        "lf": score.lf,
        "hf": score.hf,
        "weighted_joint": score.weighted_joint,
        "wrong_key_lf": list(score.wrong_key_lf),
        "wrong_key_hf": list(score.wrong_key_hf),
        "wrong_key_weighted_joint": list(score.wrong_key_weighted_joint),
    }
    if gate_a:
        payload["gate_a_margin"] = score.gate_a_margin
    return payload


def _decision_payload(decision) -> dict[str, object]:
    return {
        "paired_null_arm": decision.paired_null_arm.value,
        "gate_a_margin": decision.gate_a_margin,
        "gate_b_margin": decision.gate_b_margin,
        "margin": decision.margin,
        "positive": decision.positive,
    }


def _triplet_payload(scores) -> dict[str, object]:
    return {
        "u": _content_payload(scores.u, gate_a=False),
        "g": _content_payload(scores.g, gate_a=False),
        "cg": _content_payload(scores.cg, gate_a=False),
        "positive_cg_vs_g": _decision_payload(scores.positive_cg_vs_g),
        "negative_g_vs_u": _decision_payload(scores.negative_g_vs_u),
    }


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 512), "gray").save(path, format="PNG")


def _write_fake_r0(root: Path) -> tuple[str, ...]:
    roster = tuple(
        unit.unit_id
        for unit in load_content_chain_contract(_REPO_ROOT).evaluation_roster
    )
    raw = []
    for unit_id in roster:
        u, g, cg = _score(0.1), _score(0.0), _score(1.0)
        scores = scored_triplet(u=u, g=g, cg=cg)
        arms = []
        for arm, score, decision in (
            (R0Arm.U, u, None),
            (R0Arm.G, g, scores.negative_g_vs_u),
            (R0Arm.C, None, None),
            (R0Arm.CG, cg, scores.positive_cg_vs_g),
        ):
            relative = Path("images") / unit_id / f"{arm.name}.png"
            if arm is not R0Arm.C:
                _write_png(root / relative)
            arms.append(
                {
                    "arm": arm.value,
                    "content": None
                    if score is None
                    else _content_payload(score, gate_a=True),
                    "paired_content_decision": None
                    if decision is None
                    else _decision_payload(decision),
                    "errors": [] if arm is not R0Arm.C else ["unread fixture C"],
                    "image_file": relative.as_posix(),
                }
            )
        raw.append(
            {
                "unit_id": unit_id,
                "stage": "evaluation",
                "residual_strength_multiplier": 0.75,
                "arms": arms,
            }
        )
    result = {
        "exact": runner.R0_PRODUCER_EXACT,
        "status": runner.R0_REQUIRED_STATUS,
        "selection": {"selected_residual_strength_multiplier": 0.75},
        "rosters": {"evaluation": list(roster)},
        "evaluation_aggregate": {
            "stage": "evaluation_fixed_8",
            "roster": list(roster),
            "residual_strength_multiplier": 0.75,
            "carrier_compatibility_passed": True,
        },
        "raw_unit_records": raw,
    }
    (root / "result.json").write_text(
        json.dumps(result, sort_keys=True), encoding="utf-8"
    )
    return roster


def _prediction_payload(spec, *, invalid_h: bool = False) -> dict[str, object]:
    truth = truth_correspondences(spec)
    predicted = tuple((x + 0.02, y) for x, y in truth)
    homography = homography_observed_to_canonical(predicted)
    if invalid_h:
        homography = ((1.0, 0.0, 0.0),) * 3
    return {
        "condition_kind": "core_nonidentity",
        "truth_observed_corners_in_canonical_normalized": truth,
        "geometry": {
            "status": "RELIABLE",
            "raw_syncseal_corners": tuple(axis for point in predicted for axis in point),
            "observed_corners_in_canonical_normalized": predicted,
            "homography_observed_to_canonical": homography,
            "legal": True,
            "error": None,
        },
        "prediction_rmse": corner_rmse(predicted, truth),
    }


def _write_fake_r1a(
    root: Path, roster: tuple[str, ...], *, invalid_identity=None
) -> None:
    raw = []
    for spec in (*R1A_SANITY_CONDITIONS, *R1A_CORE_CONDITIONS):
        for unit_id in roster:
            item = {
                "condition_id": spec.condition_id,
                "condition_kind": spec.kind.value,
                "unit_id": unit_id,
            }
            if spec in R1A_CORE_CONDITIONS:
                item.update(
                    _prediction_payload(
                        spec,
                        invalid_h=(spec.condition_id, unit_id) == invalid_identity,
                    )
                )
            raw.append(item)
    result = {
        "exact": runner.R1A_PRODUCER_EXACT,
        "status": runner.R1A_REQUIRED_STATUS,
        "blocking_method_canary_passed": True,
        "r0_input": {
            "producer_exact": runner.R0_PRODUCER_EXACT,
            "selected_residual_strength_multiplier": 0.75,
            "ordered_evaluation_cg_inputs": [
                {"unit_id": unit_id, "path": f"unused/{unit_id}.png"}
                for unit_id in roster
            ],
        },
        "fixed_counts": {
            "core_conditions": 10,
            "units_per_condition": 8,
            "records": 104,
        },
        "condition_specs": [
            {"condition_id": spec.condition_id, "kind": spec.kind.value}
            for spec in (*R1A_SANITY_CONDITIONS, *R1A_CORE_CONDITIONS)
        ],
        "condition_aggregates": [
            {
                "condition_id": spec.condition_id,
                "condition_kind": spec.kind.value,
                "roster": list(roster),
                "denominator": 8,
                "passed": True,
            }
            for spec in (*R1A_SANITY_CONDITIONS, *R1A_CORE_CONDITIONS)
        ],
        "raw_records": raw,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(
        json.dumps(result, sort_keys=True), encoding="utf-8"
    )


def _write_fake_old_r1b(
    root: Path,
    roster: tuple[str, ...],
    *, missing_zero: bool = False,
    operational: bool = False,
) -> None:
    pre_payload = []
    lambda_payload = []
    evaluations = []
    for spec in R1A_CORE_CONDITIONS:
        pre = []
        for unit_id in roster:
            scores = scored_triplet(
                u=_score(0.1), g=_score(0.0), cg=_score(0.1)
            )
            record = freeze_pre_recovery_record(
                unit_id=unit_id, spec=spec, clean_score=1.0, scores=scores
            )
            pre.append(record)
            pre_payload.append(
                {
                    "unit_id": unit_id,
                    "condition_id": spec.condition_id,
                    "clean_score_from_accepted_r0": 1.0,
                    "pre_scores": _triplet_payload(scores),
                    "membership": record.membership.value,
                    "errors": [],
                }
            )
        evaluations.append(
            {
                "condition_id": spec.condition_id,
                "roster": list(roster),
                "eligible_roster": list(roster),
                "damage_only_roster": [],
            }
        )
        if missing_zero and spec is R1A_CORE_CONDITIONS[0]:
            continue
        for record in pre:
            recovered = scored_triplet(
                u=_score(0.1), g=_score(0.0), cg=_score(0.4)
            )
            zero = evaluate_lambda_unit(
                pre_record=record,
                spec=spec,
                lambda_value=0.0,
                scores=recovered,
            )
            lambda_payload.append(
                {
                    "unit_id": record.unit_id,
                    "condition_id": spec.condition_id,
                    "lambda": 0.0,
                    "epsilon_normalized": zero.epsilon_normalized,
                    "epsilon_pixels": zero.epsilon_pixels,
                    "scores": _triplet_payload(recovered),
                    "positive_gate_a_delta": zero.positive_gate_a_delta,
                    "positive_gate_b_delta": zero.positive_gate_b_delta,
                    "positive_score_delta": zero.positive_score_delta,
                    "gain": zero.gain,
                    "improved": zero.improved,
                    "recovered_negative": zero.recovered_negative,
                    "decision_harm": zero.decision_harm,
                    "observed_negative_false_positive": (
                        zero.observed_negative_false_positive
                    ),
                    "errors": [],
                }
            )
    result = {
        "schema": runner.OLD_R1B_SCHEMA,
        "exact": runner.OLD_R1B_PRODUCER_EXACT,
        "status": R1B_OPERATIONAL_FAILURE if operational else "OLD_METHOD_RESULT",
        "blocking_method_canary_passed": None if operational else False,
        "pre_recovery_partition_frozen_before_rectification": pre_payload,
        "lambda_records": lambda_payload,
        "condition_evaluations": evaluations,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(
        json.dumps(result, sort_keys=True), encoding="utf-8"
    )


@pytest.mark.integration
def test_sorted_json_artifacts_load_complete_fixed_route(tmp_path: Path) -> None:
    r0_root = tmp_path / "r0"
    r0_root.mkdir()
    roster = _write_fake_r0(r0_root)
    r1a_root = tmp_path / "r1a"
    old_root = tmp_path / "old"
    _write_fake_r1a(r1a_root, roster)
    _write_fake_old_r1b(old_root, roster)

    inputs = runner._load_r0_inputs(_REPO_ROOT, r0_root)
    predictions = runner._validate_r1a_artifact(_REPO_ROOT, r1a_root, roster)
    old = runner._load_old_r1b_artifact(old_root, roster)
    assert tuple(item.unit_id for item in inputs) == roster
    assert len(predictions) == 80
    assert sum(map(len, old.pre_by_condition.values())) == 80
    assert sum(map(len, old.zero_by_condition.values())) == 80
    assert not any((r0_root / "images" / unit / "C.png").exists() for unit in roster)


@pytest.mark.integration
def test_invalid_stored_prediction_is_unit_failure_not_artifact_drift(
    tmp_path: Path,
) -> None:
    r0_root = tmp_path / "r0"
    r0_root.mkdir()
    roster = _write_fake_r0(r0_root)
    r1a_root = tmp_path / "r1a"
    identity = (R1A_CORE_CONDITIONS[0].condition_id, roster[0])
    _write_fake_r1a(r1a_root, roster, invalid_identity=identity)
    predictions = runner._validate_r1a_artifact(_REPO_ROOT, r1a_root, roster)
    assert "stored_prediction:invalid_homography" in predictions[identity].errors
    assert predictions[identity].predicted_correspondences is not None


@pytest.mark.integration
def test_matching_h_bow_tie_prediction_is_unit_invalid_correspondence(
    tmp_path: Path,
) -> None:
    r0_root = tmp_path / "r0"
    r0_root.mkdir()
    roster = _write_fake_r0(r0_root)
    r1a_root = tmp_path / "r1a"
    _write_fake_r1a(r1a_root, roster)
    path = r1a_root / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = R1A_CORE_CONDITIONS[0]
    identity = (spec.condition_id, roster[0])
    item = next(
        record
        for record in payload["raw_records"]
        if (record["condition_id"], record["unit_id"]) == identity
    )
    matching_h = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.5, 0.0, 1.0))
    bow_tie = apply_homography(matching_h, CANONICAL_CORNERS_NORMALIZED)
    item["geometry"]["observed_corners_in_canonical_normalized"] = bow_tie
    item["geometry"]["homography_observed_to_canonical"] = matching_h
    item["prediction_rmse"] = corner_rmse(
        bow_tie, truth_correspondences(spec)
    )
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    predictions = runner._validate_r1a_artifact(_REPO_ROOT, r1a_root, roster)
    prediction = predictions[identity]
    assert prediction.predicted_correspondences == bow_tie
    assert prediction.predicted_h_observed_to_canonical == matching_h
    assert "stored_prediction:invalid_correspondences" in prediction.errors


@pytest.mark.integration
def test_r1a_truth_or_order_drift_is_rejected_before_method(tmp_path: Path) -> None:
    r0_root = tmp_path / "r0"
    r0_root.mkdir()
    roster = _write_fake_r0(r0_root)
    r1a_root = tmp_path / "r1a"
    _write_fake_r1a(r1a_root, roster)
    path = r1a_root / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    first_core = next(
        item
        for item in payload["raw_records"]
        if item["condition_id"] == R1A_CORE_CONDITIONS[0].condition_id
    )
    first_core["truth_observed_corners_in_canonical_normalized"][0][0] += 0.01
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen truth correspondences"):
        runner._validate_r1a_artifact(_REPO_ROOT, r1a_root, roster)


@pytest.mark.integration
@pytest.mark.parametrize("missing_zero,operational", ((True, False), (False, True)))
def test_old_r1b_incomplete_or_operational_artifact_cannot_enter_method(
    tmp_path: Path, missing_zero: bool, operational: bool
) -> None:
    roster = tuple(f"evaluation-{index:02d}" for index in range(8))
    root = tmp_path / "old"
    _write_fake_old_r1b(
        root, roster, missing_zero=missing_zero, operational=operational
    )
    with pytest.raises(ValueError, match="old (?:R1B|applicable)"):
        runner._load_old_r1b_artifact(root, roster)


@pytest.mark.integration
def test_real_h_is_one_cg_derived_matrix_shared_across_u_g_cg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    matrix = homography_observed_to_canonical(CANONICAL_CORNERS_NORMALIZED)

    def tracked(image, homography):
        observed.append((image, homography))
        return image

    monkeypatch.setattr(runner, "rectify_attacked_rgb", tracked)
    attacked = runner.AttackedTriplet(
        "unit", "condition", *(Image.new("RGB", (512, 512)) for _ in range(3))
    )
    rectified = runner._rectify_shared_h(attacked, matrix)
    assert len(observed) == 3
    assert all(item[1] is matrix for item in observed)
    assert (rectified.u, rectified.g, rectified.cg) == (
        attacked.u, attacked.g, attacked.cg
    )


@pytest.mark.integration
def test_fine_grid_reuses_zero_and_scores_every_nonzero_point() -> None:
    roster = tuple(f"evaluation-{index:02d}" for index in range(8))
    pre = {}
    zero = {}
    predictions = {}
    rendered = {}
    for spec in R1A_CORE_CONDITIONS:
        pre_records = []
        zero_records = []
        for unit_id in roster:
            pre_scores = scored_triplet(
                u=_score(0.1), g=_score(0.0), cg=_score(0.1)
            )
            pre_record = freeze_pre_recovery_record(
                unit_id=unit_id, spec=spec, clean_score=1.0, scores=pre_scores
            )
            pre_records.append(pre_record)
            zero_scores = scored_triplet(
                u=_score(0.1), g=_score(0.0), cg=_score(0.4)
            )
            zero_records.append(
                evaluate_lambda_unit(
                    pre_record=pre_record,
                    spec=spec,
                    lambda_value=0.0,
                    scores=zero_scores,
                )
            )
            truth = truth_correspondences(spec)
            predicted = tuple((x + 0.02, y) for x, y in truth)
            rmse = corner_rmse(predicted, truth)
            predictions[(spec.condition_id, unit_id)] = runner.R1BStoredPrediction(
                unit_id,
                spec.condition_id,
                truth,
                predicted,
                homography_observed_to_canonical(predicted),
                rmse,
                rmse * 511.0 / 2.0,
                (),
            )
            rendered[(spec.condition_id, unit_id)] = runner.AttackedTriplet(
                unit_id,
                spec.condition_id,
                *(Image.new("RGB", (512, 512)) for _ in range(3)),
            )
        pre[spec.condition_id] = tuple(pre_records)
        zero[spec.condition_id] = tuple(zero_records)
    calls = 0

    def scorer(_image):
        nonlocal calls
        calls += 1
        return _score(0.5)

    grid = runner._fine_grid_records(
        pre=pre,
        old_zero=zero,
        predictions=predictions,
        rendered=rendered,
        render_failures={},
        scorer=scorer,
    )
    assert calls == 10 * 5 * 8 * 3
    assert all(tuple(condition_grid) == R1B_REPAIR_PIXEL_GRID for condition_grid in grid.values())
    assert all(
        record.scores is zero[condition_id][index].scores
        for condition_id, condition_grid in grid.items()
        for index, record in enumerate(condition_grid[0])
    )


@pytest.mark.integration
def test_operational_payload_has_null_method_axes_and_fixed_records(tmp_path: Path) -> None:
    roster = tuple(f"evaluation-{index:02d}" for index in range(8))
    old_root = tmp_path / "old"
    _write_fake_old_r1b(old_root, roster)
    old = runner._load_old_r1b_artifact(old_root, roster)
    predictions = {}
    for spec in R1A_CORE_CONDITIONS:
        for unit_id in roster:
            truth = truth_correspondences(spec)
            predictions[(spec.condition_id, unit_id)] = runner.R1BStoredPrediction(
                unit_id, spec.condition_id, truth, None, None, None, None,
                ("stored_prediction:missing",),
            )
    real, fine = runner._operational_records(
        pre=old.pre_by_condition,
        old_zero=old.zero_by_condition,
        error=RuntimeError("setup"),
    )
    inputs = tuple(
        runner.R0R1BInput(
            unit_id,
            Path("U.png"), Path("G.png"), Path("CG.png"),
            "U.png", "G.png", "CG.png",
        )
        for unit_id in roster
    )
    payload = runner._result_payload(
        exact="1" * 40,
        r0_root=tmp_path / "r0",
        r1a_root=tmp_path / "r1a",
        old_r1b_root=old_root,
        inputs=inputs,
        old=old,
        predictions=predictions,
        real=real,
        fine=fine,
        evaluation=None,
        setup_error=RuntimeError("setup"),
    )
    assert payload["status"] == R1B_OPERATIONAL_FAILURE
    assert payload["real_h_status"] is None
    assert payload["fine_grid_status"] is None
    assert payload["r2_candidate"] is None
    assert len(payload["real_h_records"]) == 80
    assert len(payload["fine_grid_records"]) == 480


def _run_with_injected_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scorer,
    *,
    rectification_failure: bool = False,
) -> dict[str, object]:
    r0_root = tmp_path / "r0"
    r0_root.mkdir()
    roster = _write_fake_r0(r0_root)
    r1a_root = tmp_path / "r1a"
    old_root = tmp_path / "old"
    _write_fake_r1a(r1a_root, roster)
    _write_fake_old_r1b(old_root, roster)

    class FakeCalibrationAssets:
        pass

    monkeypatch.setenv(runner.engine.KEY_ENV, "injected-key")
    monkeypatch.setenv(runner.engine.TOKEN_ENV, "injected-token")
    monkeypatch.setattr(runner, "_git_exact", lambda _root, exact: exact)
    monkeypatch.setattr(runner, "normalize_detection_key", lambda _key: b"k")
    monkeypatch.setattr(
        runner, "public_key_digest", lambda _key: runner.CONTENT_CHAIN_PUBLIC_KEY_DIGEST
    )
    monkeypatch.setattr(runner, "derive_stability_wrong_keys", lambda _key: ())
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner, "ContentCalibrationAssets", FakeCalibrationAssets)
    monkeypatch.setattr(
        runner.content_chain_runner,
        "_load_pipeline_and_assets",
        lambda _model_id, _token: (object(), FakeCalibrationAssets()),
    )
    monkeypatch.setattr(
        runner.r0_runner, "_content_scorer", lambda **_kwargs: scorer
    )
    rendered = {
        (spec.condition_id, unit_id): runner.AttackedTriplet(
            unit_id,
            spec.condition_id,
            *(Image.new("RGB", (512, 512)) for _ in range(3)),
        )
        for spec in R1A_CORE_CONDITIONS
        for unit_id in roster
    }
    monkeypatch.setattr(
        runner, "_render_attacks", lambda _inputs: (rendered, {})
    )
    if rectification_failure:
        monkeypatch.setattr(
            runner,
            "rectify_attacked_rgb",
            lambda _image, _homography: (_ for _ in ()).throw(
                ValueError("injected rectification failure")
            ),
        )
    return runner._run(
        argparse.Namespace(
            repo_root=str(_REPO_ROOT),
            expected_exact="1" * 40,
            r0_artifact_root=str(r0_root),
            r1a_artifact_root=str(r1a_root),
            old_r1b_artifact_root=str(old_root),
            result_dir=str(tmp_path / "result"),
        )
    )


@pytest.mark.integration
@pytest.mark.parametrize("raise_after", (0, 240), ids=("part_a", "part_b"))
def test_scorer_runtime_interruption_is_global_operational_on_real_runner_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raise_after: int,
) -> None:
    calls = 0

    def scorer(_image):
        nonlocal calls
        calls += 1
        if calls > raise_after:
            raise RuntimeError("injected content runtime interruption")
        return _score(0.5)

    payload = _run_with_injected_runtime(tmp_path, monkeypatch, scorer)
    assert calls == raise_after + 1
    assert payload["status"] == R1B_OPERATIONAL_FAILURE
    assert payload["real_h_status"] is None
    assert payload["fine_grid_status"] is None
    assert payload["r2_candidate"] is None
    assert len(payload["real_h_records"]) == 80
    assert len(payload["fine_grid_records"]) == 480
    assert any(
        "content_scorer_runtime:RuntimeError" in failure["errors"]
        for failure in payload["failures"]
    )


@pytest.mark.integration
def test_none_scorer_setup_is_operational_without_evaluation_dereference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _run_with_injected_runtime(tmp_path, monkeypatch, None)
    assert payload["status"] == R1B_OPERATIONAL_FAILURE
    assert payload["real_h_status"] is None
    assert payload["fine_grid_status"] is None
    assert payload["r2_candidate"] is None
    assert len(payload["real_h_records"]) == 80
    assert len(payload["fine_grid_records"]) == 480
    assert payload["provenance"]["operational_setup_error_class"] == "RuntimeError"


@pytest.mark.integration
def test_rectification_failure_remains_fixed_denominator_method_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _run_with_injected_runtime(
        tmp_path, monkeypatch, lambda _image: _score(0.5),
        rectification_failure=True,
    )
    assert payload["status"] == R1B_REPAIR_METHOD_NOT_READY
    assert payload["real_h_status"] == "NO_CORE_PASSED"
    assert payload["fine_grid_status"] == "ZERO_ONLY_ALL_CORE"
    assert payload["r2_candidate"] is False
    assert len(payload["real_h_records"]) == 80
    assert len(payload["fine_grid_records"]) == 480
    assert any(
        "real_h_recovery:ValueError" in failure["errors"]
        for failure in payload["failures"]
    )


@pytest.mark.integration
def test_runner_source_is_scoring_only_u_g_cg_and_method_failures_are_local() -> None:
    source = inspect.getsource(runner)
    ast.parse(source)
    assert "r0_runner._content_scorer" in source
    assert "content_chain_runner._load_pipeline_and_assets" in source
    assert "run_content_iss_evaluation_pair" not in source
    assert "SyncSealTorchScript" not in source
    assert '"generation_invoked": False' in source
    assert "R0Arm.C.value" not in inspect.getsource(runner._load_r0_inputs)
    assert "_fine_grid_records(" in inspect.getsource(runner._run)
    assert R1B_REPAIR_METHOD_NOT_READY in inspect.getsource(
        __import__("cegwm.geometry_v7.r1b", fromlist=["evaluate_r1b_repair"])
    )


@pytest.mark.integration
def test_phase_a_repair_notebook_guards_are_exact() -> None:
    path = _REPO_ROOT / "notebooks" / "geometry_v7_r1b.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code[0]["source"] == [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')",
    ]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code)
    for index, cell in enumerate(code):
        ast.parse("".join(cell["source"]), filename=f"{path}:code-cell-{index}")
    source = "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"])
    assert "APPROVED_EXACT = 'PENDING_AFTER_GEOMETRY_V7_R1B_REPAIR_PUSH'" in source
    assert "re.fullmatch(r'[0-9a-f]{40}', APPROVED_EXACT)" in source
    assert "'checkout', '--detach', APPROVED_EXACT" in source
    assert "assert torch.cuda.is_available()" in source
    assert runner.R0_PRODUCER_EXACT in source
    assert runner.R1A_PRODUCER_EXACT in source
    assert runner.OLD_R1B_PRODUCER_EXACT in source
    assert source.count("'experiments.run_geometry_v7_r1b'") == 1
    assert "--old-r1b-artifact-root" in source
    assert "/ 'r1b-repair'" in source
    assert source.count("if DRIVE_RESULT_DIR.exists():") == 2
    assert "shutil.copytree(LOCAL_RESULT_DIR, DRIVE_RESULT_DIR)" in source
    assert "userdata.get('HF_TOKEN')" in source
    assert "userdata.get('CEG_WM_ROOT_KEY')" in source
    assert "force_remount" not in source
    assert "sha256" not in source.lower()
    assert "git', 'pull'" not in source


@pytest.mark.integration
def test_runner_cli_help_imports_without_model_execution() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(_REPO_ROOT / "src"), environment.get("PYTHONPATH", "")))
    )
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.run_geometry_v7_r1b", "--help"],
        cwd=_REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--r0-artifact-root" in completed.stdout
    assert "--r1a-artifact-root" in completed.stdout
    assert "--old-r1b-artifact-root" in completed.stdout
    assert "--result-dir" in completed.stdout
