from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

import cegwm.geometry_v3.confirmation as P1
import cegwm.geometry_v3.mechanism_audit as P1M0


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[2]
RUNNER = _load(
    ROOT / "experiments" / "run_geometry_v3_qk_active_writer_p1m0_mechanism_audit.py",
    "geometry_v3_p1m0_runner",
)
P1_TEST = _load(
    ROOT / "tests" / "integration" / "test_geometry_v3_qk_active_writer_p1_confirmation.py",
    "geometry_v3_p1_source_fixture",
)


def _p1_records() -> tuple[dict[str, object], ...]:
    raw = {
        ("identity", "q"): (0.002291210228577256, 0.0007208778406493366, 0.0022876462899148464),
        ("identity", "k"): (-0.0019550323486328125, 0.0031769266352057457, -0.001948195043951273),
        ("rotate270", "q"): (0.003234180621802807, -0.003264396684244275, 0.0032398172188550234),
        ("rotate270", "k"): (-0.005187239497900009, -0.0005108852055855095, -0.005191202275454998),
        ("similarity", "q"): (0.001084530958905816, 0.00018784429994411767, 0.001094376901164651),
        ("similarity", "k"): (0.0011492747580632567, 0.003148894291371107, 0.0011397396447136998),
        ("crop_rescale", "q"): (0.0021522999741137028, 0.0013334894319996238, 0.0021502317395061255),
        ("crop_rescale", "k"): (-0.001911961124278605, 0.00017432268941774964, -0.0019050919217988849),
    }
    records = []
    for attack in P1.P1_ATTACK_IDS:
        for kind in P1.P1_KIND_IDS:
            values = raw[(attack, kind)]
            margin = values[0] - max(values[1:])
            for control, score in zip(P1.P1_CONTROL_IDS, values, strict=True):
                records.append({
                    "config_id": P1.P1_CONFIG_ID,
                    "attack_id": attack,
                    "feature_kind": kind,
                    "control": control,
                    "status": "calculated",
                    "error_class": None,
                    "score": score,
                    "margin": margin,
                })
    return tuple(records)


def _sources(tmp_path: Path) -> tuple[Path, Path, P1M0.ValidatedSources]:
    p0 = P1_TEST._source_fixture(tmp_path / "p0")
    p0_identity = P1.validate_p0_source(p0)
    p1 = tmp_path / "p1"
    result = P1.P1ExecutionResult(
        P1.P1_STATUS_UNRESOLVED,
        _p1_records(),
        -1.784181222319603e-6,
        -0.003381319053005427,
        (), (), (), None,
    )
    P1.package_p1_artifacts(
        p1, exact=P1M0.P1_EXECUTION_EXACT,
        source_identity=p0_identity, result=result,
    )
    sources = P1M0.validate_sources(p0, p1)
    return p0, p1, sources


def _plan(tmp_path: Path, p0: Path, p1: Path) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "expected_exact": "a" * 40,
        "execution_exact": "a" * 40,
        "p0_source_directory": str(p0),
        "p1_source_directory": str(p1),
        "output_directory": "/content/drive/MyDrive/CEG-WM/Geometry-V3/P1M0/Geometry-V3-P1M0-test",
    }), encoding="utf-8")
    return path


def _run_main(plan: Path, monkeypatch: pytest.MonkeyPatch, execute) -> tuple[int, dict[str, object]]:
    monkeypatch.setenv(RUNNER.TOKEN_ENV, "token")
    monkeypatch.setenv(RUNNER.KEY_ENV, "key")
    monkeypatch.setattr(RUNNER, "_git_exact", lambda expected: expected)
    monkeypatch.setattr(RUNNER, "execute_plan", execute)
    read_fd, write_fd = os.pipe()
    rc = RUNNER._main(["--plan", str(plan), "--control-fd", str(write_fd)])
    payload = os.read(read_fd, RUNNER.MAX_CONTROL_BYTES + 1)
    os.close(read_fd)
    return rc, json.loads(payload)


@pytest.mark.integration
def test_protocol_freezes_identity_only_single_config_mechanism_audit() -> None:
    plan = P1M0.public_plan()
    assert plan["protocol"] == P1M0.P1M0_PROTOCOL_ID
    assert plan["fixed_config_id"] == "block12-qk-rms0p0025"
    assert plan["writer_step_index"] == 18
    assert plan["relative_rms_budget"] == 0.0025
    assert plan["generation_roles"] == ["no_writer", "writer"]
    assert plan["stages"] == list(P1M0.P1M0_STAGES)
    assert plan["fixed_unit_count"] == 24
    assert plan["science_denominator"] == 0
    assert "retry" not in json.dumps(plan).lower()
    assert "fallback" not in json.dumps(plan).lower()


@pytest.mark.integration
def test_real_structured_sources_bind_raw_components_and_two_instance_displacement(
    tmp_path: Path,
) -> None:
    _, _, sources = _sources(tmp_path)
    assert len(sources.p0_selected_scores) == 24
    assert len(sources.p1_scores) == 24
    assert len(sources.two_instance_displacement) == 6
    assert {item["control"] for item in sources.two_instance_displacement} == set(P1.P1_CONTROL_IDS)
    assert all("two_instance_displacement" in item for item in sources.two_instance_displacement)
    wrong = [
        item for item in sources.two_instance_displacement
        if item["control"] == "wrong_key_anchor"
    ]
    assert len(wrong) == 2
    assert all(
        item["interpretation"] == "wrong_key_domain_confounded_two_instance_displacement"
        for item in wrong
    )
    assert sources.p1_identity["status"] == "P1_UNRESOLVED"


@pytest.mark.integration
def test_main_validates_both_real_structured_sources_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    p0, p1, expected_sources = _sources(tmp_path)

    def execute(plan, *, geometry_key, hf_token, sources, preloader):
        del plan, preloader
        assert geometry_key == "key" and hf_token == "token"
        assert sources == expected_sources
        return {
            "run_id": "geometry-v3-qk-p1m0-test",
            "status": P1M0.P1M0_STATUS_INCONCLUSIVE,
            "artifact_status": "complete",
            "fixed_config_id": P1.P1_CONFIG_ID,
            "science_denominator": 0,
        }

    rc, control = _run_main(_plan(tmp_path, p0, p1), monkeypatch, execute)
    assert rc == 0
    assert control["status"] == "success"
    assert control["p1m0_status"] == P1M0.P1M0_STATUS_INCONCLUSIVE


@pytest.mark.integration
@pytest.mark.parametrize("source_name", ("p0", "p1"))
def test_main_fails_source_validation_before_model_for_tampered_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_name: str,
) -> None:
    p0, p1, _ = _sources(tmp_path)
    target = p0 if source_name == "p0" else p1
    with (target / "metrics.jsonl").open("ab") as stream:
        stream.write(b"{}\n")

    rc, control = _run_main(
        _plan(tmp_path, p0, p1), monkeypatch,
        lambda *args, **kwargs: pytest.fail("execution reached"),
    )
    assert rc == 1
    assert control == {
        "status": "failure", "failure_point": "source_validation",
        "error_class": "validation_error", "science_denominator": 0,
    }


def _classification_fixture(
    *, contract_pass: bool = True, q_lift: bool = True, k_lift: bool = True,
    q_rgb_separation: float = -0.002, k_rgb_separation: float = -0.003,
):
    contracts = []
    for kind, lift in (("q", q_lift), ("k", k_lift)):
        contracts.append({
            "feature_kind": kind,
            "contract_pass": contract_pass,
            "axis_contract_pass": contract_pass,
            "token_contract_pass": contract_pass,
            "channel_contract_pass": contract_pass,
            "normalization_contract_pass": contract_pass,
            "positive_injection_sign_consistent": contract_pass,
            "normalized_correct_correlation_lift_positive": lift,
        })
    stages = []
    for kind, rgb_separation in (("q", q_rgb_separation), ("k", k_rgb_separation)):
        for index, stage in enumerate(P1M0.P1M0_STAGES):
            separation = rgb_separation if stage == "final_rgb_reencode" else 0.004 - 0.001 * index
            stages.append({
                "feature_kind": kind,
                "stage": stage,
                "writer_correct_score": 0.008 - 0.001 * index,
                "writer_wrong_score": 0.001,
                "no_writer_correct_score": 0.002,
                "no_writer_wrong_score": -0.001,
                "correct_score_change_from_previous_stage": -0.001,
                "writer_separation": separation,
            })
    return contracts, stages


@pytest.mark.integration
@pytest.mark.parametrize(
    ("fixture_kwargs", "expected"),
    (
        ({"contract_pass": False}, P1M0.P1M0_STATUS_MISMATCH),
        ({}, P1M0.P1M0_STATUS_INSUFFICIENT),
        ({"k_lift": False}, P1M0.P1M0_STATUS_INCONCLUSIVE),
        ({"q_rgb_separation": 0.001}, P1M0.P1M0_STATUS_INCONCLUSIVE),
    ),
)
def test_dynamic_status_classification_uses_complete_low_sensitivity_scalar_fixture(
    fixture_kwargs: dict[str, object], expected: str,
) -> None:
    contracts, stages = _classification_fixture(**fixture_kwargs)
    assert P1M0.classify_p1m0(contracts, stages) == expected


@pytest.mark.integration
def test_status_rule_is_frozen_without_method_promotion() -> None:
    plan = P1M0.public_plan()["decision_rule"]
    assert plan["implementation_mismatch"].startswith("any_public_contract")
    assert "qk_hook_lifts_positive" in plan["observability_insufficiency"]
    assert P1M0.P1M0_SCIENCE_DENOMINATOR == 0
