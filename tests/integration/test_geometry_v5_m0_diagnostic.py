from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from cegwm.protocol.geometry_v5_m0 import GeometryV5M0RawRecord, load_geometry_v5_m0_contract
from experiments import geometry_v5_m0_diagnostic as diagnostic


_ROOT = Path(__file__).resolve().parents[2]


class _FakeRGB:
    mode = "RGB"
    size = (512, 512)


def _output(images: object) -> object:
    return type("FakeDiffusersOutput", (), {"images": images})()


def _passing_raw(case: dict[str, object]) -> GeometryV5M0RawRecord:
    truth = diagnostic._truth_h(case)
    return GeometryV5M0RawRecord(
        "ESTIMATE_AVAILABLE", math.degrees(math.atan2(truth[1][0], truth[0][0])), math.hypot(truth[0][0], truth[1][0]),
        truth[0][2], truth[1][2], truth, {},
    )


def _fake_bindings(calls: dict[str, list[object]], *, fail_load: bool = False, fail_generation: bool = False, fail_case: str | None = None, generation_output: object | None = None) -> diagnostic._Bindings:
    contract = load_geometry_v5_m0_contract(_ROOT)
    cases = [case for case in contract.config["development"]["attacks"] if case["attack_id"] in {"identity", "rotation_+10", "scale_1.1", "translation_x_+0.08"}]

    def load() -> object:
        calls["load"].append("load")
        if fail_load:
            raise RuntimeError("secret model load")
        return object()

    def initial(_pipeline: object, seed: int) -> int:
        calls["initial"].append(seed)
        return seed

    def generate(_pipeline: object, _prompt: str, initial_z_t: int) -> object:
        calls["generate"].append(initial_z_t)
        if fail_generation:
            raise RuntimeError("secret generation")
        return _output([_FakeRGB()]) if generation_output is None else generation_output

    def attack(final_rgb: object, case: dict[str, object]) -> int:
        assert isinstance(final_rgb, _FakeRGB) and not hasattr(final_rgb, "images")
        calls["attack_image"].append(final_rgb)
        calls["attack"].append(case["attack_id"])
        if case["attack_id"] == fail_case:
            raise ValueError("secret attack")
        return cases.index(case)

    def detect(_pipeline: object, attacked_rgb: int) -> GeometryV5M0RawRecord:
        calls["detect"].append(attacked_rgb)
        return _passing_raw(cases[attacked_rgb])

    calls.setdefault("attack_image", [])
    return diagnostic._Bindings(load, initial, generate, attack, detect)


@pytest.mark.integration
def test_fake_diagnostic_runs_one_generation_four_independent_cases_and_freezes_raw_before_truth(tmp_path: Path) -> None:
    calls: dict[str, list[object]] = {name: [] for name in ("load", "initial", "generate", "attack", "detect")}
    events: list[str] = []
    output = tmp_path / "diagnostic.json"
    result, complete = diagnostic.run_diagnostic(_ROOT, output, _fake_bindings(calls), events.append)
    assert complete and calls["load"] == ["load"] and calls["initial"] == calls["generate"] == [7501]
    assert calls["attack"] == ["identity", "rotation_+10", "scale_1.1", "translation_x_+0.08"] and len(calls["attack_image"]) == len(calls["detect"]) == 4
    assert all(events[index:index + 2] == ["raw_frozen", "truth_evaluated"] for index in range(0, 8, 2))
    assert result["diagnostic_denominator"] == 4 and result["science_denominator"] == 0 and result["claim"] == "nonformal_colab_method_diagnostic_only"
    text = output.read_text(encoding="utf-8")
    assert text == diagnostic._canonical_json(json.loads(text)).decode("utf-8") and "secret" not in text
    with pytest.raises(FileExistsError):
        diagnostic.run_diagnostic(_ROOT, output, _fake_bindings(calls))


@pytest.mark.integration
def test_diagnostic_retains_four_failed_cases_for_model_generation_and_single_case_failures(tmp_path: Path) -> None:
    for name, kwargs, expected_stage in (("load", {"fail_load": True}, "model_load"), ("generation", {"fail_generation": True}, "generation")):
        calls: dict[str, list[object]] = {key: [] for key in ("load", "initial", "generate", "attack", "detect")}
        result, complete = diagnostic.run_diagnostic(_ROOT, tmp_path / f"{name}.json", _fake_bindings(calls, **kwargs))
        assert not complete and len(result["cases"]) == 4 and all(case["failure_stage"] == expected_stage and case["error_class"] == "RuntimeError" for case in result["cases"])
    calls = {key: [] for key in ("load", "initial", "generate", "attack", "detect")}
    result, complete = diagnostic.run_diagnostic(_ROOT, tmp_path / "case.json", _fake_bindings(calls, fail_case="scale_1.1"))
    failed = [case for case in result["cases"] if case["attack_id"] == "scale_1.1"]
    assert not complete and len(result["cases"]) == 4 and failed[0]["failure_stage"] == "attack" and len(calls["detect"]) == 3


@pytest.mark.integration
def test_diagnostic_rejects_invalid_generation_output_containers_as_four_generation_failures(tmp_path: Path) -> None:
    invalid_outputs = (object(), _output([]), _output([_FakeRGB(), _FakeRGB()]), _output([type("BadMode", (), {"mode": "RGBA", "size": (512, 512)})()]), _output([type("BadSize", (), {"mode": "RGB", "size": (256, 256)})()]))
    for index, output in enumerate(invalid_outputs):
        calls: dict[str, list[object]] = {key: [] for key in ("load", "initial", "generate", "attack", "detect")}
        result, complete = diagnostic.run_diagnostic(_ROOT, tmp_path / f"invalid-{index}.json", _fake_bindings(calls, generation_output=output))
        assert not complete and len(result["cases"]) == 4
        assert all(case["failure_stage"] == "generation" and case["error_class"] == "ValueError" for case in result["cases"])
        assert calls["attack"] == calls["detect"] == []


@pytest.mark.integration
def test_diagnostic_forward_truth_is_strict_inverse_in_centered_unit_coordinates() -> None:
    contract = load_geometry_v5_m0_contract(_ROOT)
    cases = {case["attack_id"]: case for case in contract.config["development"]["attacks"]}
    identity, rotation, scale, translation = (diagnostic._truth_h(cases[name]) for name in ("identity", "rotation_+10", "scale_1.1", "translation_x_+0.08"))
    assert identity == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert math.degrees(math.atan2(rotation[1][0], rotation[0][0])) == pytest.approx(-10.0)
    assert math.hypot(scale[0][0], scale[1][0]) == pytest.approx(1 / 1.1)
    assert translation[0][2] == pytest.approx(-0.08) and translation[1][2] == pytest.approx(0.0)


@pytest.mark.integration
def test_diagnostic_module_is_lazy_concrete_and_has_no_formal_gate() -> None:
    source = (_ROOT / "experiments/geometry_v5_m0_diagnostic.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not {"torch", "diffusers", "PIL"} & imported
    assert all(name in source for name in ("load_bound_sd21_pipeline", "generate_bound_sd21", "recover_and_estimate_bound_sd21"))
    assert "aggregate" not in source and "GeometryV5Observation" not in source and "RELIABLE" not in source
    assert "diagnostic_only_not_a_gate" in source and "fake" not in diagnostic._DIAGNOSTIC_ID
