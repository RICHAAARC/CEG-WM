from pathlib import Path

from PIL import Image

from cegwm.baselines.t2smark_canary import CONDITIONS, RUN_SCHEMA, RunLock, atomic_json, atomic_png, clear_stale_lock, establish_contract, generation_digest, load_t2smark_sd35_pipeline, pending_observations, run_canary, run_transaction, sha256_file, valid_generation, valid_observation, validate_final_publication


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
    assert list((tmp_path / "quarantine").glob("canary_result.json.*"))


def test_force_recomputes_all_observations_after_generation_rebuild(tmp_path: Path) -> None:
    prepare_generation(tmp_path)
    calls = []
    run_canary(tmp_path, config(), lambda c, r: (Image.new("RGB", (3, 3)), 1.0))
    run_canary(tmp_path, config(), lambda c, r: (calls.append((c, r)) or Image.new("RGB", (3, 3)), 1.0), force=True)
    assert len(calls) == 12


def test_lock_rejects_second_runner_and_releases_own_token(tmp_path: Path) -> None:
    with RunLock(tmp_path):
        try:
            with RunLock(tmp_path): pass
        except RuntimeError as exc: assert "locked" in str(exc)
        else: raise AssertionError("second runner acquired lock")
    assert not (tmp_path / ".run.lock").exists()


def test_persisted_records_exclude_secret_and_raw_key_fixture_values(tmp_path: Path) -> None:
    prepare_generation(tmp_path)
    run_canary(tmp_path, config(), lambda c, r: (Image.new("RGB", (3, 3)), 1.0))
    text = "\n".join(path.read_text() for path in tmp_path.rglob("*.json"))
    assert "HF_TOKEN" not in text and "master_key" not in text and "session_key" not in text and "message_bits" not in text


def test_final_manifest_is_required_and_rejects_hash_mismatch(tmp_path: Path) -> None:
    prepare_generation(tmp_path)
    run_canary(tmp_path, config(), lambda c, r: (Image.new("RGB", (3, 3)), 1.0))
    assert validate_final_publication(tmp_path)
    (tmp_path / "scores.csv").write_text("corrupt")
    assert not validate_final_publication(tmp_path)


def test_malformed_contract_fails_closed_and_malformed_records_retry(tmp_path: Path) -> None:
    atomic_json(tmp_path / "run_config.json", {"schema": RUN_SCHEMA})
    try: establish_contract(tmp_path, config())
    except RuntimeError: pass
    else: raise AssertionError("malformed contract accepted")
    (tmp_path / "generation_checkpoint.json").write_text("[]")
    assert not valid_observation(tmp_path, config(), CONDITIONS[0], "clean_negative")


def test_transaction_locks_generation_and_quarantines_final_on_failure(tmp_path: Path) -> None:
    prepare_generation(tmp_path); run_canary(tmp_path, config(), lambda c, r: (Image.new("RGB", (3, 3)), 1.0))
    (tmp_path / "generation_checkpoint.json").write_text("{}")
    def generate():
        assert (tmp_path / ".run.lock").exists()
        try:
            with RunLock(tmp_path): pass
        except RuntimeError: pass
        else: raise AssertionError("nested lock acquired")
        raise RuntimeError("generation failed")
    try: run_transaction(tmp_path, config(), generate, lambda c, r: (Image.new("RGB", (3, 3)), 1.0))
    except RuntimeError: pass
    assert not (tmp_path / ".run.lock").exists()
    assert list((tmp_path / "quarantine").glob("canary_result.json.*"))


def test_malformed_generation_returns_false_without_exception(tmp_path: Path) -> None:
    for payload in ([], {}, {"identity": {}}, {"identity": config(), "files": {}}, {"identity": config(), "files": {"clean.png": 3}}):
        atomic_json(tmp_path / "generation_checkpoint.json", payload)
        assert not valid_generation(tmp_path, config()) and generation_digest(tmp_path) is None


def test_clear_stale_lock_returns_owner(tmp_path: Path) -> None:
    atomic_json(tmp_path / ".run.lock", {"pid": 1, "token": "old"})
    assert clear_stale_lock(tmp_path)["token"] == "old" and not (tmp_path / ".run.lock").exists()


def test_main_uses_transaction_not_lock_outside_generation() -> None:
    source = Path("src/cegwm/baselines/t2smark_canary.py").read_text()
    main_source = source[source.index("def main()"):]
    assert "run_transaction(root, config, generate, execute, force=args.force_rerun_all)" in main_source
    assert "run_canary(root,config" not in main_source and "establish_contract(root,config)" not in main_source


def test_shared_pipeline_loader_requires_token(tmp_path: Path) -> None:
    try:
        load_t2smark_sd35_pipeline(
            tmp_path,
            model_id="model",
            model_revision="revision",
            hf_token="",
        )
    except RuntimeError as error:
        assert str(error) == "HF_TOKEN is required"
    else:
        raise AssertionError("empty token reached the official runtime import")
