from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from experiments import run_geometry_v7_r1a as runner
from cegwm.geometry_v7.contracts import (
    CANONICAL_CORNERS_NORMALIZED,
    GeometryEstimate,
    estimate_geometry,
)
from cegwm.geometry_v7.r1a import (
    R1A_ALL_CONDITIONS,
    R1A_BLOCKING_METHOD_CANARY_FAILED,
    evaluate_r1a,
)
from cegwm.protocol.content_chain import load_content_chain_contract


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _rendered_fixture() -> tuple[runner.RenderedAttack, ...]:
    return tuple(
        runner.RenderedAttack(
            f"evaluation-{unit_index:02d}",
            spec.condition_id,
            Image.new("RGB", (512, 512)),
            f"attacked/{spec.condition_id}/evaluation-{unit_index:02d}.png",
            "0" * 64,
        )
        for spec in R1A_ALL_CONDITIONS
        for unit_index in range(8)
    )


def _input_fixture() -> tuple[runner.R0CGInput, ...]:
    return tuple(
        runner.R0CGInput(
            f"evaluation-{index:02d}",
            Path(f"/unused/evaluation-{index:02d}.png"),
            f"images/evaluation-{index:02d}.png",
        )
        for index in range(8)
    )


def _write_fake_r0_artifact(root: Path) -> tuple[str, ...]:
    contract = load_content_chain_contract(_REPO_ROOT)
    roster = tuple(unit.unit_id for unit in contract.evaluation_roster)
    records = []
    for unit_id in roster:
        relative = Path("images") / "evaluation" / unit_id / "CG.png"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (512, 512), "gray").save(path, format="PNG")
        records.append(
            {
                "unit_id": unit_id,
                "stage": "evaluation",
                "residual_strength_multiplier": 0.75,
                "arms": [
                    {
                        "arm": "C_with_content_no_sync",
                        "image_file": "unused/missing-c.png",
                        "errors": [],
                    },
                    {
                        "arm": "CG_with_content_with_sync",
                        "image_file": relative.as_posix(),
                        "errors": [],
                    },
                ],
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
    (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return roster


@pytest.mark.integration
def test_r0_input_loader_selects_only_fixed_evaluation_cg_without_hash_gate(
    tmp_path: Path,
) -> None:
    roster = _write_fake_r0_artifact(tmp_path)
    inputs = runner._load_r0_cg_inputs(_REPO_ROOT, tmp_path)
    assert tuple(item.unit_id for item in inputs) == roster
    assert all(item.path.name == "CG.png" for item in inputs)
    assert not (tmp_path / "result.json.sha256").exists()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("exact", "0" * 40),
        ("status", "PAIRED_COMPATIBILITY_CANARY_FAILED"),
    ),
)
def test_r0_input_loader_rejects_identity_or_status_drift(
    tmp_path: Path, field: str, value: str
) -> None:
    _write_fake_r0_artifact(tmp_path)
    path = tmp_path / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result[field] = value
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="identity, status, selection, or roster"):
        runner._load_r0_cg_inputs(_REPO_ROOT, tmp_path)


@pytest.mark.integration
def test_syncseal_setup_failure_projects_all_104_fixed_records() -> None:
    rendered = _rendered_fixture()
    records = runner._records_after_setup_failure(
        rendered, RuntimeError("model load stopped")
    )
    assert len(records) == 13
    assert sum(len(item.records) for item in records) == 104
    evaluation = evaluate_r1a(
        condition_records=records,
        ordered_roster=tuple(f"evaluation-{index:02d}" for index in range(8)),
    )
    assert all(aggregate.denominator == 8 for aggregate in evaluation.aggregates)
    assert all(not aggregate.passed for aggregate in evaluation.aggregates)
    assert all(
        record.errors == ("syncseal_runtime_setup:RuntimeError",)
        for item in records
        for record in item.records
    )


@pytest.mark.integration
@pytest.mark.parametrize("reported", (False, True))
def test_detector_operational_failure_controls_complete_top_level_payload(
    tmp_path: Path, reported: bool
) -> None:
    rendered = _rendered_fixture()

    def detector(_image):
        if reported:
            return GeometryEstimate.error_record("detector reported error")
        raise RuntimeError("detector threw")

    records = runner._records_after_detection(rendered, detector)
    roster = tuple(item.unit_id for item in _input_fixture())
    evaluation = evaluate_r1a(
        condition_records=records,
        ordered_roster=roster,
    )
    payload = runner._result_payload(
        exact="1" * 40,
        artifact_root=tmp_path,
        inputs=_input_fixture(),
        rendered=rendered,
        records=records,
        evaluation=evaluation,
        setup_error=None,
        checkpoint=None,
    )
    assert payload["status"] == runner.OPERATIONAL_FAILURE_STATUS
    assert payload["blocking_method_canary_passed"] is None
    assert (
        payload["fixed_denominator_evaluation_status"]
        == R1A_BLOCKING_METHOD_CANARY_FAILED
    )
    assert len(payload["raw_records"]) == len(payload["failures"]) == 104
    assert len(payload["condition_aggregates"]) == 13
    assert all(item["denominator"] == 8 for item in payload["condition_aggregates"])


@pytest.mark.integration
def test_finite_legal_gate_failure_remains_method_failure_not_operational(
    tmp_path: Path,
) -> None:
    rendered = _rendered_fixture()
    records = runner._records_after_detection(
        rendered,
        lambda _image: estimate_geometry(0.0, CANONICAL_CORNERS_NORMALIZED),
    )
    evaluation = evaluate_r1a(
        condition_records=records,
        ordered_roster=tuple(item.unit_id for item in _input_fixture()),
    )
    payload = runner._result_payload(
        exact="1" * 40,
        artifact_root=tmp_path,
        inputs=_input_fixture(),
        rendered=rendered,
        records=records,
        evaluation=evaluation,
        setup_error=None,
        checkpoint=None,
    )
    assert payload["status"] == R1A_BLOCKING_METHOD_CANARY_FAILED
    assert payload["blocking_method_canary_passed"] is False
    assert payload["failures"] == []


@pytest.mark.integration
def test_finite_unsupported_points_remain_method_failure_with_full_payload(
    tmp_path: Path,
) -> None:
    raw = (
        (-1.0, -1.0),
        (127.0 / 128.0, 127.0 / 128.0),
        (127.0 / 128.0, -1.0),
        (-1.0, 127.0 / 128.0),
    )
    converted = (
        (-1.0, -1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
    )
    geometry = estimate_geometry(
        0.0,
        converted,
        raw_syncseal_corners=raw,
    )
    rendered = _rendered_fixture()
    records = runner._records_after_detection(rendered, lambda _image: geometry)
    evaluation = evaluate_r1a(
        condition_records=records,
        ordered_roster=tuple(item.unit_id for item in _input_fixture()),
    )
    payload = runner._result_payload(
        exact="1" * 40,
        artifact_root=tmp_path,
        inputs=_input_fixture(),
        rendered=rendered,
        records=records,
        evaluation=evaluation,
        setup_error=None,
        checkpoint=None,
    )
    assert payload["status"] == R1A_BLOCKING_METHOD_CANARY_FAILED
    assert payload["blocking_method_canary_passed"] is False
    assert len(payload["raw_records"]) == len(payload["failures"]) == 104
    assert len(payload["condition_aggregates"]) == 13
    assert all(item["denominator"] == 8 for item in payload["condition_aggregates"])
    assert all(item["errors"] == ("geometry_invalid",) for item in payload["raw_records"])
    retained = payload["raw_records"][0]["geometry"]
    assert retained["raw_syncseal_corners"] == raw
    assert retained["observed_corners_in_canonical_normalized"] == converted
    assert retained["homography_observed_to_canonical"] is None


@pytest.mark.integration
def test_runner_source_keeps_truth_out_of_detector_and_content_chain_out_of_r1a() -> None:
    source = inspect.getsource(runner)
    ast.parse(source)
    detector_source = inspect.getsource(runner._records_after_detection)
    assert "detect_attacked_rgb(detector, item.image)" in detector_source
    assert all(
        forbidden not in detector_source
        for forbidden in ("truth", "matrix", "prompt", "key", "latent", "original")
    )
    assert "run_content_iss_evaluation_pair" not in source
    assert "blind_weighted_scores" not in source
    assert "CEG_WM_ROOT_KEY" not in source and "HF_TOKEN" not in source
    assert "download_official_syncseal_torchscript" in source
    assert "SyncSealTorchScript.from_file" in source
    assert runner.R0_PRODUCER_EXACT in source


@pytest.mark.integration
def test_phase_a_r1a_notebook_is_unexecuted_and_fail_closed() -> None:
    path = _REPO_ROOT / "notebooks" / "geometry_v7_r1a.ipynb"
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
    assert "APPROVED_EXACT = 'PENDING_AFTER_GEOMETRY_V7_R1A_PUSH'" in source
    assert "re.fullmatch(r'[0-9a-f]{40}', APPROVED_EXACT)" in source
    assert "'checkout', '--detach', APPROVED_EXACT" in source
    assert "assert torch.cuda.is_available()" in source
    assert runner.R0_PRODUCER_EXACT in source
    assert source.count("'experiments.run_geometry_v7_r1a'") == 1
    assert "if LOCAL_RESULT_DIR.exists() or SYNCSEAL_CHECKPOINT.exists():" in source
    assert source.count("if DRIVE_RESULT_DIR.exists():") == 2
    assert "shutil.copytree(LOCAL_RESULT_DIR, DRIVE_RESULT_DIR)" in source
    assert "force_remount" not in source
    assert "userdata" not in source
    assert "sha256" not in source.lower()
    assert "git', 'pull'" not in source


@pytest.mark.integration
def test_runner_cli_help_imports_without_model_execution() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(_REPO_ROOT / "src"), environment.get("PYTHONPATH", "")))
    )
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.run_geometry_v7_r1a", "--help"],
        cwd=_REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--r0-artifact-root" in completed.stdout
    assert "--syncseal-checkpoint" in completed.stdout
