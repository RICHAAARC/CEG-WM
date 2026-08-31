from pathlib import Path

from PIL import Image

from cegwm.baselines.t2smark_canary import CONDITIONS, RUN_SCHEMA, atomic_json, atomic_png, establish_contract, pending_observations, run_canary, sha256_file, valid_observation


def config() -> dict:
    return {"schema": RUN_SCHEMA, "project_exact": "a" * 40, "official_exact": "b" * 40, "model_id": "m", "model_revision": "r", "prompt": "p", "generation_seed": 1, "watermark_seed": 2, "parameters": {"x": 1}}


def prepare_generation(path: Path) -> None:
    image = Image.new("RGB", (3, 3)); atomic_png(path / "clean.png", image); atomic_png(path / "watermarked.png", image)
    atomic_json(path / "generation_checkpoint.json", {"identity": {key: config()[key] for key in ("schema","project_exact","official_exact","model_id","model_revision","prompt","generation_seed","watermark_seed","parameters")}, "files": {"clean.png": sha256_file(path / "clean.png"), "watermarked.png": sha256_file(path / "watermarked.png")}})


def test_contract_drift_fails_closed_and_atomic_json(tmp_path: Path) -> None:
    atomic_json(tmp_path / "x.json", {"a": 1}); assert (tmp_path / "x.json").read_text()
    establish_contract(tmp_path, config()); changed = config(); changed["prompt"] = "other"
    try: establish_contract(tmp_path, changed)
    except RuntimeError as exc: assert "identity drift" in str(exc)
    else: raise AssertionError("drift accepted")


def test_valid_observation_reused_and_corruption_retried(tmp_path: Path) -> None:
    prepare_generation(tmp_path)
    calls = []
    def execute(condition, role): calls.append((condition, role)); return Image.new("RGB", (3, 3)), 1.0
    run_canary(tmp_path, config(), execute); assert len(calls) == 12 and valid_observation(tmp_path, config(), CONDITIONS[0], "clean_negative")
    calls.clear(); run_canary(tmp_path, config(), execute); assert calls == []
    (tmp_path / "images" / f"{CONDITIONS[0]}__clean_negative.png").write_bytes(b"bad")
    calls.clear(); run_canary(tmp_path, config(), execute); assert calls == [(CONDITIONS[0], "clean_negative")]


def test_failed_records_retry_and_final_requires_all_valid(tmp_path: Path) -> None:
    prepare_generation(tmp_path)
    def fail(condition, role):
        if condition == CONDITIONS[0]: raise RuntimeError("retry")
        return Image.new("RGB", (3, 3)), 1.0
    run_canary(tmp_path, config(), fail)
    assert not (tmp_path / "canary_result.json").exists() and len(pending_observations(tmp_path, config())) == 2
    run_canary(tmp_path, config(), lambda c, r: (Image.new("RGB", (3, 3)), 1.0))
    assert (tmp_path / "canary_result.json").is_file() and (tmp_path / "scores.csv").is_file()


def test_stale_final_is_quarantined_when_repair_fails(tmp_path: Path) -> None:
    prepare_generation(tmp_path)
    run_canary(tmp_path, config(), lambda c, r: (Image.new("RGB", (3, 3)), 1.0))
    (tmp_path / "images" / f"{CONDITIONS[0]}__clean_negative.png").write_bytes(b"bad")
    run_canary(tmp_path, config(), lambda c, r: (_ for _ in ()).throw(RuntimeError("repair failed")))
    assert not (tmp_path / "canary_result.json").exists()
    assert list(tmp_path.glob("canary_result.stale.*.json"))


def test_force_recomputes_all_observations_after_generation_rebuild(tmp_path: Path) -> None:
    prepare_generation(tmp_path)
    calls = []
    run_canary(tmp_path, config(), lambda c, r: (Image.new("RGB", (3, 3)), 1.0))
    run_canary(tmp_path, config(), lambda c, r: (calls.append((c, r)) or Image.new("RGB", (3, 3)), 1.0), force=True)
    assert len(calls) == 12
