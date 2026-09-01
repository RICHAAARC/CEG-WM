from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from PIL import Image
import pytest

from experiments import run_geometry_v7_r1b as runner
from cegwm.geometry_v7.r0 import ContentScore, R0Arm
from cegwm.geometry_v7.r1a import R1A_CORE_CONDITIONS, R1A_SANITY_CONDITIONS
from cegwm.geometry_v7.r1b import (
    R1B_TRUTH_UTILITY_FAILED,
    R1BEvaluation,
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


def _content_payload(score: ContentScore) -> dict[str, object]:
    return {
        "lf": score.lf,
        "hf": score.hf,
        "weighted_joint": score.weighted_joint,
        "wrong_key_lf": list(score.wrong_key_lf),
        "wrong_key_hf": list(score.wrong_key_hf),
        "wrong_key_weighted_joint": list(score.wrong_key_weighted_joint),
        "gate_a_margin": score.gate_a_margin,
    }


def _decision_payload(decision) -> dict[str, object]:
    return {
        "paired_null_arm": decision.paired_null_arm.value,
        "gate_a_margin": decision.gate_a_margin,
        "gate_b_margin": decision.gate_b_margin,
        "margin": decision.margin,
        "positive": decision.positive,
    }


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 512), "gray").save(path, format="PNG")


def _write_fake_r0(root: Path) -> tuple[str, ...]:
    contract = load_content_chain_contract(_REPO_ROOT)
    roster = tuple(unit.unit_id for unit in contract.evaluation_roster)
    records = []
    for unit_id in roster:
        u = _score(0.1)
        g = _score(0.0)
        cg = _score(1.0)
        clean = scored_triplet(u=u, g=g, cg=cg)
        arm_payloads = []
        for arm, score, decision in (
            (R0Arm.U, u, None),
            (R0Arm.G, g, clean.negative_g_vs_u),
            (R0Arm.C, None, None),
            (R0Arm.CG, cg, clean.positive_cg_vs_g),
        ):
            relative = Path("images") / unit_id / f"{arm.name}.png"
            if arm is not R0Arm.C:
                _write_png(root / relative)
            arm_payloads.append(
                {
                    "arm": arm.value,
                    "content": None if score is None else _content_payload(score),
                    "paired_content_decision": None
                    if decision is None
                    else _decision_payload(decision),
                    "errors": ["C deliberately unavailable"] if arm is R0Arm.C else [],
                    "image_file": relative.as_posix(),
                }
            )
        records.append(
            {
                "unit_id": unit_id,
                "stage": "evaluation",
                "residual_strength_multiplier": 0.75,
                "arms": arm_payloads,
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
        "raw_unit_records": records,
    }
    (root / "result.json").write_text(
        json.dumps(result, sort_keys=True), encoding="utf-8"
    )
    return roster


def _write_fake_r1a(root: Path, roster: tuple[str, ...]) -> None:
    all_specs = (*R1A_SANITY_CONDITIONS, *R1A_CORE_CONDITIONS)
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
            {
                "condition_id": spec.condition_id,
                "kind": spec.kind.value,
            }
            for spec in all_specs
        ],
        "condition_aggregates": [
            {
                "condition_id": spec.condition_id,
                "condition_kind": spec.kind.value,
                "roster": list(roster),
                "denominator": 8,
                "passed": True,
            }
            for spec in all_specs
        ],
        "raw_records": [
            {"condition_id": spec.condition_id, "unit_id": unit_id}
            for spec in all_specs
            for unit_id in roster
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(json.dumps(result), encoding="utf-8")


@pytest.mark.integration
def test_artifact_loaders_bind_r0_u_g_cg_and_r1a_fixed_identity(
    tmp_path: Path,
) -> None:
    r0_root = tmp_path / "r0"
    r0_root.mkdir()
    roster = _write_fake_r0(r0_root)
    r1a_root = tmp_path / "r1a"
    _write_fake_r1a(r1a_root, roster)
    inputs = runner._load_r0_inputs(_REPO_ROOT, r0_root)
    assert tuple(item.unit_id for item in inputs) == roster
    assert all(item.clean_score == 1.0 for item in inputs)
    assert all(item.u_path.name == "U.png" for item in inputs)
    assert all(item.g_path.name == "G.png" for item in inputs)
    assert all(item.cg_path.name == "CG.png" for item in inputs)
    assert not any((r0_root / "images" / unit / "C.png").exists() for unit in roster)
    runner._validate_r1a_artifact(_REPO_ROOT, r1a_root, roster)


@pytest.mark.integration
@pytest.mark.parametrize("artifact", ("r0", "r1a"))
def test_artifact_loaders_reject_exact_or_core_identity_drift(
    tmp_path: Path, artifact: str
) -> None:
    r0_root = tmp_path / "r0"
    r0_root.mkdir()
    roster = _write_fake_r0(r0_root)
    r1a_root = tmp_path / "r1a"
    _write_fake_r1a(r1a_root, roster)
    target = (r0_root if artifact == "r0" else r1a_root) / "result.json"
    result = json.loads(target.read_text(encoding="utf-8"))
    result["exact"] = "0" * 40
    target.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact identity"):
        if artifact == "r0":
            runner._load_r0_inputs(_REPO_ROOT, r0_root)
        else:
            runner._validate_r1a_artifact(_REPO_ROOT, r1a_root, roster)


def _input_fixture() -> tuple[runner.R0R1BInput, ...]:
    clean = scored_triplet(u=_score(0.1), g=_score(0.0), cg=_score(1.0))
    return tuple(
        runner.R0R1BInput(
            unit_id,
            Path(f"/unused/{unit_id}/U.png"),
            Path(f"/unused/{unit_id}/G.png"),
            Path(f"/unused/{unit_id}/CG.png"),
            f"images/{unit_id}/U.png",
            f"images/{unit_id}/G.png",
            f"images/{unit_id}/CG.png",
            clean,
            1.0,
        )
        for unit_id in tuple(f"evaluation-{index:02d}" for index in range(8))
    )


@pytest.mark.integration
def test_setup_failure_publishes_fixed_80_pre_records_and_method_null(
    tmp_path: Path,
) -> None:
    inputs = _input_fixture()
    pre = runner._setup_failure_pre_records(inputs, RuntimeError("setup stopped"))
    payload = runner._result_payload(
        exact="1" * 40,
        r0_root=tmp_path / "r0",
        r1a_root=tmp_path / "r1a",
        inputs=inputs,
        pre=pre,
        lambdas={},
        evaluation=None,
        setup_error=RuntimeError("setup stopped"),
    )
    assert payload["status"] == runner.R1B_OPERATIONAL_FAILURE
    assert payload["blocking_method_canary_passed"] is None
    assert payload["fixed_counts"]["pre_records"] == 80
    assert len(payload["pre_recovery_partition_frozen_before_rectification"]) == 80
    assert len(payload["failures"]) == 80
    assert payload["lambda_records"] == []


@pytest.mark.integration
def test_finite_gate_miss_remains_method_status_not_operational(tmp_path: Path) -> None:
    inputs = _input_fixture()
    pre = {
        spec.condition_id: tuple(
            freeze_pre_recovery_record(
                unit_id=item.unit_id,
                spec=spec,
                clean_score=1.0,
                scores=scored_triplet(u=_score(0.1), g=_score(0.0), cg=_score(0.8)),
            )
            for item in inputs
        )
        for spec in R1A_CORE_CONDITIONS
    }
    evaluation = R1BEvaluation(R1B_TRUTH_UTILITY_FAILED, (), 1, False)
    payload = runner._result_payload(
        exact="1" * 40,
        r0_root=tmp_path / "r0",
        r1a_root=tmp_path / "r1a",
        inputs=inputs,
        pre=pre,
        lambdas={},
        evaluation=evaluation,
        setup_error=None,
    )
    assert payload["status"] == R1B_TRUTH_UTILITY_FAILED
    assert payload["blocking_method_canary_passed"] is False
    assert payload["failures"] == []


@pytest.mark.integration
def test_lambda_one_reuses_pre_scores_without_rescoring() -> None:
    inputs = _input_fixture()
    first = R1A_CORE_CONDITIONS[0]
    pre = {}
    rendered = {}
    for spec in R1A_CORE_CONDITIONS:
        score_value = 0.1 if spec is first else 0.8
        pre[spec.condition_id] = tuple(
            freeze_pre_recovery_record(
                unit_id=item.unit_id,
                spec=spec,
                clean_score=1.0,
                scores=scored_triplet(
                    u=_score(0.1), g=_score(0.0), cg=_score(score_value)
                ),
            )
            for item in inputs
        )
    for item in inputs:
        rendered[(first.condition_id, item.unit_id)] = runner.AttackedTriplet(
            item.unit_id,
            first.condition_id,
            Image.new("RGB", (512, 512)),
            Image.new("RGB", (512, 512)),
            Image.new("RGB", (512, 512)),
        )
    calls = 0

    def scorer(_image):
        nonlocal calls
        calls += 1
        return _score(0.5)

    grid = runner._lambda_score_all(
        rendered=rendered,
        pre_by_condition=pre,
        scorer=scorer,
    )
    assert tuple(grid) == (first.condition_id,)
    assert tuple(grid[first.condition_id]) == runner.R1B_LAMBDA_GRID
    assert calls == 4 * 8 * 3
    assert all(
        record.scores is pre_record.scores
        for record, pre_record in zip(
            grid[first.condition_id][1.0], pre[first.condition_id], strict=True
        )
    )


@pytest.mark.integration
def test_all_pre_scores_finish_before_first_membership_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _input_fixture()
    rendered = {
        (spec.condition_id, item.unit_id): runner.AttackedTriplet(
            item.unit_id,
            spec.condition_id,
            Image.new("RGB", (512, 512)),
            Image.new("RGB", (512, 512)),
            Image.new("RGB", (512, 512)),
        )
        for spec in R1A_CORE_CONDITIONS
        for item in inputs
    }
    events: list[str] = []

    def scorer(_image):
        events.append("score")
        return _score(0.5)

    original_freeze = runner.freeze_pre_recovery_record

    def tracked_freeze(**kwargs):
        events.append("freeze")
        return original_freeze(**kwargs)

    monkeypatch.setattr(runner, "freeze_pre_recovery_record", tracked_freeze)
    pre = runner._pre_score_all(
        inputs=inputs,
        rendered=rendered,
        render_failures={},
        scorer=scorer,
    )
    assert len(pre) == 10
    assert events[: 10 * 8 * 3] == ["score"] * (10 * 8 * 3)
    assert events[10 * 8 * 3 :] == ["freeze"] * (10 * 8)


@pytest.mark.integration
def test_runner_source_uses_scoring_assets_only_and_never_generation_or_c() -> None:
    source = inspect.getsource(runner)
    ast.parse(source)
    assert "r0_runner._content_scorer" in source
    assert "content_chain_runner._load_pipeline_and_assets" in source
    assert all(
        forbidden not in source
        for forbidden in (
            "run_content_iss_evaluation_pair",
            "run_content_chain_unit",
            "run_sd35_content_adaptive",
            "SyncSealTorchScript",
            "download_official_syncseal",
        )
    )
    loader_source = inspect.getsource(runner._load_r0_inputs)
    assert "R0Arm.C.value" not in loader_source
    assert "R0Arm.U.value" in loader_source
    assert "R0Arm.G.value" in loader_source
    assert "R0Arm.CG.value" in loader_source
    assert '"generation_invoked": False' in source


@pytest.mark.integration
def test_phase_a_r1b_notebook_guards_are_exact() -> None:
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
    assert "APPROVED_EXACT = 'PENDING_AFTER_GEOMETRY_V7_R1B_PUSH'" in source
    assert "re.fullmatch(r'[0-9a-f]{40}', APPROVED_EXACT)" in source
    assert "'checkout', '--detach', APPROVED_EXACT" in source
    assert "assert torch.cuda.is_available()" in source
    assert runner.R0_PRODUCER_EXACT in source
    assert runner.R1A_PRODUCER_EXACT in source
    assert source.count("'experiments.run_geometry_v7_r1b'") == 1
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
    assert "--result-dir" in completed.stdout


@pytest.mark.integration
def test_not_applicable_status_is_explicit_in_method_payload_contract() -> None:
    source = inspect.getsource(
        __import__("cegwm.geometry_v7.r1b", fromlist=["evaluate_condition"])
    )
    assert re.search(r'"NOT_APPLICABLE/INSUFFICIENT_ELIGIBLE"', source)
