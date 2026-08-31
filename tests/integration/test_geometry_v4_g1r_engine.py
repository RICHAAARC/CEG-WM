from __future__ import annotations

import inspect
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments import geometry_v4_g1r_engine as engine
from cegwm.method import geometry_v4_g1r as method
from cegwm.method.geometry_v4_g1r import G1RFinalRGBObservability
from cegwm.protocol.geometry_v4_g1r import ATTACKS, PLACEMENT, SEARCH_TOP_K, WRITER_ID, derive_g1r_keys
from cegwm.runtime.geometry_v4_g1r_sd35 import G1RGeneratedPair

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_keyed_joint_search_is_deterministic_bounded_and_contains_five_attack_truths() -> None:
    key = b"geometry-v4-g1r-cpu-key-v1"
    search_key = derive_g1r_keys(key)["search"]
    wrong_search_key = derive_g1r_keys(b"geometry-v4-g1r-cpu-wrong-key-v1")["search"]
    field = method.g1r_anchor_fields((96, 96), key).search
    plane = np.clip(.5 + .02 * field * np.sqrt(field.size), 0.0, 1.0)
    rgb = np.repeat(plane[..., None], 3, axis=2)
    identities = []
    for attack in ATTACKS:
        attacked = engine._apply_attack(rgb, attack)
        candidates = method._search_candidates(attacked, search_key)
        repeated = method._search_candidates(attacked, search_key)
        assert len(candidates) == len(repeated) == SEARCH_TOP_K
        assert [tuple(item["rank"]) for item in candidates] == [tuple(item["rank"]) for item in repeated]
        assert all(-10.0 <= item["angle"] <= 10.0 and .84 <= item["scale"] <= 1.16 for item in candidates)
        for item in candidates:
            relative_translation = np.linalg.inv(method._similarity_h(item["angle"], item["scale"])) @ np.asarray(item["canonical_to_attacked"])
            assert abs(relative_translation[0, 2]) <= .12 and abs(relative_translation[1, 2]) <= .12
        truth = np.linalg.inv(engine._truth_for_attack(attack))
        truth_points = engine._points(truth)
        errors = [float(np.max(np.linalg.norm(engine._points(np.asarray(item["canonical_to_attacked"])) - truth_points, axis=1) / np.sqrt(2.0))) for item in candidates]
        assert min(errors) <= .025
        identities.append(tuple((item["angle"], item["scale"], tuple(np.asarray(item["canonical_to_attacked"]).reshape(-1)), item["ncc"], item["translation_psr"]) for item in candidates))
    wrong = method._search_candidates(rgb, wrong_search_key)
    assert identities[0] != tuple((item["angle"], item["scale"], tuple(np.asarray(item["canonical_to_attacked"]).reshape(-1)), item["ncc"], item["translation_psr"]) for item in wrong)
    assert len({round(float(item["translation_psr"]), 8) for item in method._search_candidates(rgb, search_key)}) > 1


@pytest.mark.integration
def test_host_dominant_search_separates_correct_from_negative_and_wrong() -> None:
    ordinary = engine._carrier("colored_texture", 96)
    marked, _ = method.write_g1r_rgb(ordinary, engine.CPU_KEY)
    correct_key = derive_g1r_keys(engine.CPU_KEY)["search"]
    wrong_key = derive_g1r_keys(engine.CPU_WRONG_KEY)["search"]
    correct = method._search_candidates(marked, correct_key)[0]
    negative = method._search_candidates(ordinary, correct_key)[0]
    wrong = method._search_candidates(marked, wrong_key)[0]
    assert float(correct["component_consensus"]) > float(negative["component_consensus"])
    assert float(correct["component_consensus"]) > float(wrong["component_consensus"])
    assert float(correct["ncc"]) > float(negative["ncc"])
    assert float(correct["ncc"]) > float(wrong["ncc"])


@pytest.mark.integration
def test_real_rosters_are_complete_disjoint_and_have_no_subset_interface() -> None:
    development = engine.build_real_roster(ROOT, "development")
    confirmation = engine.build_real_roster(ROOT, "confirmation")
    assert len(development) == len(confirmation) == 20
    assert {item[0] for item in development} == {6201, 6202, 6203, 6204}
    assert {item[0] for item in confirmation} == {6301, 6302, 6303, 6304}
    assert not {item[0] for item in development} & {item[0] for item in confirmation}
    assert tuple(item[2] for item in development) == ATTACKS * 4
    assert tuple(inspect.signature(engine.build_real_roster).parameters) == ("repo_root", "split")


@pytest.mark.integration
def test_truth_is_attached_only_after_blind_three_arm_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tuple(inspect.signature(engine._blind_arms).parameters) == ("attacked_marked", "attacked_negative")
    assert tuple(inspect.signature(engine._apply_attack).parameters) == ("rgb", "attack")
    assert "truth" not in inspect.getsource(engine._apply_attack)
    assert "truth_transform" not in inspect.getsource(engine._blind_arms)
    assert tuple(inspect.signature(engine._evaluate_frozen_arms).parameters) == ("arms", "truth_attacked_to_canonical")

    events: list[str] = []
    unreliable = {"H_hat": None, "corners_hat": (), "support": 0, "reliability": 0.0, "status": "UNRELIABLE"}
    monkeypatch.setattr(engine, "_blind_arms", lambda marked, negative: events.append("blind") or engine.BlindArms(unreliable, unreliable, unreliable))
    monkeypatch.setattr(engine, "_truth_for_attack", lambda attack: events.append("truth") or np.eye(3))
    monkeypatch.setattr(engine, "_evaluate_frozen_arms", lambda arms, truth: events.append("evaluate") or {name: {"status": "UNRELIABLE", "support": 0, "truth_metrics": {}, "unsafe": False} for name in ("correct", "wrong", "negative")})
    records = engine.run_cpu_canary()
    assert len(records) == 20 and events == [item for _ in range(20) for item in ("blind", "truth", "evaluate")]


@pytest.mark.integration
def test_truth_metric_is_attacked_to_canonical_and_normalized_diagonal() -> None:
    truth = engine._truth_for_attack("translation_0.08_0")
    reliable = {"status": "RELIABLE", "H_hat": tuple(float(value) for value in truth.reshape(-1))}
    metrics = engine._truth_metrics(reliable, truth)
    assert metrics == pytest.approx({"mapped_corner_error": 0.0, "center_reprojection_error": 0.0, "rotation_abs_error_degrees": 0.0, "log_scale_abs_error": 0.0})


@pytest.mark.integration
def test_truth_probe_is_sanitized_record_only_evidence_at_declared_h() -> None:
    ordinary = engine._carrier("gradient_shapes", 96)
    marked, _ = method.write_g1r_rgb(ordinary, engine.CPU_KEY)
    probe = engine._truth_probe(marked, engine.CPU_KEY, engine._truth_for_attack("identity"))
    assert set(probe) == {"search_at_truth", "search_best_translation_at_truth_rs", "fit_at_truth", "holdout_at_truth", "holdout_after_fit"}
    assert probe["fit_at_truth"]["support"] >= 6
    assert len(probe["fit_at_truth"]["prethreshold_tiles"]) == 8
    serialized = json.dumps(probe, allow_nan=False, sort_keys=True)
    assert "key" not in serialized.lower() and "phase" not in serialized.lower()
    assert engine.CPU_KEY.hex() not in serialized and engine.CPU_KEY.decode("ascii") not in serialized


@pytest.mark.integration
def test_fixed_cpu_three_arm_canary_reaches_engineering_exit() -> None:
    records = engine.run_cpu_canary()
    summary = engine.summarize_cpu_canary(records)
    assert len(records) == 20 and all(record["failure"] is None for record in records)
    assert summary["formal_denominator"] == 0 and summary["units"] == 20
    assert summary["correct_safe_reliable"] >= 18
    assert summary["correct_unsafe"] == summary["wrong_unsafe"] == summary["negative_unsafe"] == 0
    assert all(value >= 3 for value in summary["correct_safe_by_attack"].values())
    assert summary["status"] == "CPU_ENGINEERING_EXIT"


class _FakePipeline:
    def to(self, device: str):
        assert device == "cuda"
        return self


class _FakeContentDetector:
    def __call__(self, image, key):
        return 0.0

    def identities(self):
        return {"adapter_id": "fake-content-detector"}


def _unreliable_arm() -> dict[str, object]:
    return {"H_hat": None, "corners_hat": (), "support": 0, "reliability": 0.0, "status": "UNRELIABLE"}


@pytest.mark.integration
def test_real_three_arms_are_independent_and_use_the_required_current_rgb(monkeypatch: pytest.MonkeyPatch) -> None:
    marked = np.full((32, 32, 3), .6)
    negative = np.full((32, 32, 3), .4)
    correct_key, wrong_key = b"0123456789abcdef", b"wrong-key-0123456789"
    calls: list[tuple[np.ndarray, object]] = []

    def fake_detect(rgb, key):
        calls.append((rgb, key))
        return _unreliable_arm(), {"search_top_k": (), "selected_fit": {"support": 0}, "holdout": {}}

    monkeypatch.setattr(engine, "_detect_g1r_engineering", fake_detect)
    arms = engine._blind_arms_for_keys(marked, negative, correct_key, wrong_key)
    assert arms.correct is not arms.wrong and arms.wrong is not arms.negative
    assert calls[0][0] is marked and calls[0][1] == correct_key
    assert calls[1][0] is marked and calls[1][1] == wrong_key
    assert calls[2][0] is negative and calls[2][1] == correct_key
    assert all("engineering_diagnostics" in getattr(arms, name) for name in ("correct", "wrong", "negative"))
    assert json.dumps(arms.correct, allow_nan=False)


@pytest.mark.integration
def test_real_development_generates_each_source_once_retains_20_units_and_orders_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    generated: list[int] = []
    events: list[str] = []

    monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(engine, "_load_real_pipeline_and_assets", lambda token: (_FakePipeline(), object()))
    monkeypatch.setattr(engine, "build_reused_weighted_joint_content_adapter", lambda assets, root: _FakeContentDetector())
    monkeypatch.setattr(engine, "_cuda_generator", lambda seed: seed)

    def fake_pair(pipeline, prompt, key, *, height, width, generator):
        del pipeline, prompt, key
        assert (height, width) == (512, 512)
        generated.append(generator)
        image = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8), mode="RGB")
        return G1RGeneratedPair(image, image)

    monkeypatch.setattr(engine, "run_g1r_sd35_pair", fake_pair)
    monkeypatch.setattr(engine, "measure_g1r_final_rgb", lambda *args: G1RFinalRGBObservability(50.0, .99, 0.0, 0.0, 0.0, 0.0, 0.0, {name: 1.0 for name in ("search", "fit", "validate")}, {name: 0.0 for name in ("search", "fit", "validate")}))
    monkeypatch.setattr(engine, "_blind_arms_for_keys", lambda *args: events.append("blind") or engine.BlindArms(_unreliable_arm(), _unreliable_arm(), _unreliable_arm()))
    monkeypatch.setattr(engine, "_truth_for_attack", lambda attack: events.append("truth") or np.eye(3))
    monkeypatch.setattr(engine, "_truth_probe", lambda *args: events.append("probe") or {"record_only": True})
    sources, records, identity = engine.run_real_development(b"0123456789abcdef", b"wrong-key-0123456789", repo_root=ROOT, hf_token="test-token")
    assert generated == [6201, 6202, 6203, 6204]
    assert len(sources) == 4 and len(records) == 20 and all(record["failure"] is None for record in records)
    assert tuple(record["attack"] for record in records) == ATTACKS * 4
    assert all(set(record["arms"]) == {"correct", "wrong", "negative"} for record in records)
    assert events == [item for _ in range(20) for item in ("blind", "truth", "probe")]
    assert all(record["truth_probe"] == {"record_only": True} for record in records)
    assert identity == {"adapter_id": "fake-content-detector"}


@pytest.mark.integration
def test_truth_probe_runs_post_freeze_and_cannot_change_arm_or_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    quiet = engine.BlindArms(_unreliable_arm(), _unreliable_arm(), _unreliable_arm())
    monkeypatch.setattr(engine, "_blind_arms_for_keys", lambda *args: events.append("blind") or quiet)
    monkeypatch.setattr(engine, "_truth_for_attack", lambda attack: events.append("truth") or np.eye(3))
    monkeypatch.setattr(engine, "_truth_probe", lambda *args: (_ for _ in ()).throw(RuntimeError("diagnostic only")))
    arms = engine._blind_arms_for_keys(np.zeros((32, 32, 3)), np.zeros((32, 32, 3)), b"correct", b"wrong")
    truth = engine._truth_for_attack("identity")
    evaluation = engine._evaluate_frozen_arms(arms, truth)
    try:
        engine._truth_probe(np.zeros((32, 32, 3)), b"correct", truth)
    except RuntimeError:
        pass
    assert events == ["blind", "truth"]
    assert all(evaluation[name]["status"] == "UNRELIABLE" and not evaluation[name]["unsafe"] for name in ("correct", "wrong", "negative"))


@pytest.mark.integration
def test_real_development_retains_source_failure_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(engine, "_load_real_pipeline_and_assets", lambda token: (_FakePipeline(), object()))
    monkeypatch.setattr(engine, "build_reused_weighted_joint_content_adapter", lambda assets, root: _FakeContentDetector())
    monkeypatch.setattr(engine, "_cuda_generator", lambda seed: seed)

    def fail_pairs(pipeline, prompt, key, *, height, width, generator):
        del pipeline, prompt, key, height, width
        calls.append(generator)
        raise RuntimeError("retained failure")

    monkeypatch.setattr(engine, "run_g1r_sd35_pair", fail_pairs)
    sources, records, _ = engine.run_real_development(b"0123456789abcdef", b"wrong-key-0123456789", repo_root=ROOT, hf_token="test-token")
    assert calls == [6201, 6202, 6203, 6204]
    assert len(sources) == 4 and len(records) == 20
    assert all(source["failure"] == {"scope": "source_generation", "type": "RuntimeError"} for source in sources)
    assert all(record["failure"] == {"scope": "source_generation", "type": "RuntimeError"} and record["arms"] is None for record in records)


@pytest.mark.integration
def test_real_summary_cannot_be_passed_by_final_rgb_only() -> None:
    sources = tuple({"seed": seed, "prompt": "p", "failure": None, "final_rgb": {"passed": True}} for seed in (6201, 6202, 6203, 6204))
    arms = {name: {**_unreliable_arm(), "truth_metrics": {}, "unsafe": False} for name in ("correct", "wrong", "negative")}
    records = tuple({"seed": seed, "prompt": "p", "attack": attack, "failure": None, "arms": arms} for seed in (6201, 6202, 6203, 6204) for attack in ATTACKS)
    summary = engine.summarize_real_development(sources, records)
    assert summary["source_observability_passed"] == 4 and summary["correct_safe_reliable"] == 0
    assert summary["failures"] == 0 and summary["status"] == "GATE_FAILED"


@pytest.mark.integration
def test_real_summary_requires_all_20_safe_correct_and_zero_unsafe() -> None:
    sources = tuple({"seed": seed, "prompt": "p", "failure": None, "final_rgb": {"passed": True}} for seed in (6201, 6202, 6203, 6204))
    correct = {"H_hat": tuple(np.eye(3).reshape(-1)), "corners_hat": (), "support": 8, "reliability": 1.0, "status": "RELIABLE", "truth_metrics": {}, "unsafe": False}
    quiet = {**_unreliable_arm(), "truth_metrics": {}, "unsafe": False}
    records = tuple({"seed": seed, "prompt": "p", "attack": attack, "failure": None, "arms": {"correct": correct, "wrong": quiet, "negative": quiet}} for seed in (6201, 6202, 6203, 6204) for attack in ATTACKS)
    assert engine.summarize_real_development(sources, records)["status"] == "PASS"
    changed = [dict(record) for record in records]
    changed[0] = {**changed[0], "arms": {**changed[0]["arms"], "wrong": {**quiet, "status": "RELIABLE", "unsafe": True}}}
    assert engine.summarize_real_development(sources, tuple(changed))["status"] == "GATE_FAILED"


@pytest.mark.integration
def test_artifacts_are_create_only_hashed_and_secret_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "_environment_metadata", lambda: {"cuda_available": True, "gpu": {"name": "fake", "vram_bytes": 1}})
    sources = tuple({"seed": seed, "prompt": "p", "failure": None, "final_rgb": {"passed": False}} for seed in (6201, 6202, 6203, 6204))
    records = tuple({"seed": seed, "prompt": "p", "attack": attack, "failure": {"scope": "unit", "type": "FakeError"}, "arms": None} for seed in (6201, 6202, 6203, 6204) for attack in ATTACKS)
    summary = engine.summarize_real_development(sources, records)
    root = tmp_path / "artifact"
    engine.write_development_artifacts(root, source_exact="a" * 40, repo_root=ROOT, sources=sources, records=records, summary=summary, content_detector={})
    for name in ("g1r-development-records.json", "g1r-development-summary.json", "g1r-development-manifest.json"):
        payload = (root / name).read_bytes()
        assert (root / f"{name}.sha256").read_text(encoding="ascii") == f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
        json.loads(payload)
    manifest = json.loads((root / "g1r-development-manifest.json").read_text(encoding="ascii"))
    assert manifest["source_exact"] == "a" * 40 and manifest["stage"] == "development" and manifest["units"] == 20
    assert manifest["seeds"] == [6201, 6202, 6203, 6204] and manifest["attacks"] == list(ATTACKS)
    assert manifest["config_sha256"] and manifest["notebook_identity"] == "geometry_v4_g0_g1_colab_v4_g1r_development_v1"
    assert manifest["placement"] == PLACEMENT and manifest["writer_id"] == WRITER_ID
    joined = b"".join(path.read_bytes() for path in root.iterdir())
    assert b"root-secret-value" not in joined and b"hf-secret-value" not in joined
    with pytest.raises(FileExistsError):
        engine.write_development_artifacts(root, source_exact="a" * 40, repo_root=ROOT, sources=sources, records=records, summary=summary, content_detector={})


@pytest.mark.integration
def test_cli_rejects_confirmation_and_has_no_subset_surface(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = engine.main(["--stage", "confirmation", "--repo-root", str(ROOT), "--artifact-root", str(tmp_path / "artifact"), "--expected-exact", "a" * 40], environ={})
    assert result == 2 and json.loads(capsys.readouterr().out)["status"] == "STOPPED"
    source = inspect.getsource(engine.main)
    assert "--subset" not in source and "--resume" not in source and "--retry" not in source


@pytest.mark.integration
def test_development_cli_pops_secrets_and_stdout_is_compact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exact = "a" * 40
    sources = tuple({"seed": seed, "prompt": "p", "failure": None, "final_rgb": {"passed": False}} for seed in (6201, 6202, 6203, 6204))
    arms = {name: {**_unreliable_arm(), "truth_metrics": {}, "unsafe": False} for name in ("correct", "wrong", "negative")}
    records = tuple({"seed": seed, "prompt": "p", "attack": attack, "failure": None, "arms": arms} for seed in (6201, 6202, 6203, 6204) for attack in ATTACKS)
    monkeypatch.setattr(engine, "_checkout_state", lambda root: (exact, "", True))
    monkeypatch.setattr(engine, "run_real_development", lambda *args, **kwargs: (sources, records, {}))
    monkeypatch.setattr(engine, "write_development_artifacts", lambda *args, **kwargs: {})
    environ = {"CEG_WM_ROOT_KEY": "root-secret-value-0123456789", "HF_TOKEN": "hf-secret-value"}
    result = engine.main(["--stage", "development", "--repo-root", str(ROOT), "--artifact-root", str(tmp_path / "artifact"), "--expected-exact", exact], environ=environ)
    output = capsys.readouterr().out
    assert result == 2 and environ == {}
    assert "root-secret-value" not in output and "hf-secret-value" not in output
    assert json.loads(output)["status"] == "GATE_FAILED"
