from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from PIL import Image

from experiments.stage_a import run_hf_a2_colab as runner

_ROOT = Path(__file__).resolve().parents[2]
_RAW_KEY = "stage-a-colab-detection-key-0001"
_HF_TOKEN = "hf_stage_a_test_token"


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    shutil.copytree(_ROOT / "configs", repo / "configs")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Stage A Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "configs"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "fixture"], check=True)
    exact = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, exact


class _Generator:
    def __init__(self, device: str) -> None:
        assert device == "cuda"
        self.seed = 0

    def manual_seed(self, seed: int) -> _Generator:
        self.seed = seed
        return self


def _args(
    repo: Path,
    exact: str,
    output_root: Path,
    run_store_root: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(repo),
        output_root=str(output_root),
        expected_exact=exact,
        run_store_root=str(run_store_root),
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_hf_calls: frozenset[int] = frozenset(),
    interrupt_hf_call: int | None = None,
) -> dict[str, int]:
    calls = {"load": 0, "hf": 0, "plain": 0, "score": 0}
    assets = SimpleNamespace()

    def fake_load(model_id: str, hf_token: str) -> tuple[object, object]:
        assert model_id == "stabilityai/stable-diffusion-3.5-medium"
        assert hf_token == _HF_TOKEN
        calls["load"] += 1
        return object(), assets

    monkeypatch.setattr(runner, "_load_pipeline_and_assets", fake_load)
    monkeypatch.setattr(runner.torch, "Generator", _Generator)

    def fake_hf(pipeline: object, prompt: str, key: bytes, public_assets: object, **kwargs: object) -> SimpleNamespace:
        del pipeline, prompt, public_assets
        calls["hf"] += 1
        if calls["hf"] == interrupt_hf_call:
            raise KeyboardInterrupt
        if calls["hf"] in fail_hf_calls:
            raise RuntimeError("private detail that must not be exported")
        seed = kwargs["generator"].seed
        pixels = np.full((8, 8, 3), (seed + key[0]) % 200 + 30, dtype=np.uint8)
        return SimpleNamespace(
            image=Image.fromarray(pixels),
            injection_budget=SimpleNamespace(relative_l2=0.01),
        )

    def fake_plain(pipeline: object, prompt: str, **kwargs: object) -> Image.Image:
        del pipeline, prompt
        calls["plain"] += 1
        seed = kwargs["generator"].seed
        return Image.fromarray(np.full((8, 8, 3), seed % 200, dtype=np.uint8))

    def fake_score(image: Image.Image, key: bytes, public_assets: object) -> float:
        del public_assets
        calls["score"] += 1
        return float(np.asarray(image).mean() / 255.0 + key[0] / 1024.0)

    monkeypatch.setattr(runner, "run_sd35_hf", fake_hf)
    monkeypatch.setattr(runner, "run_sd35_plain", fake_plain)
    monkeypatch.setattr(runner, "score_hf_image", fake_score)
    return calls


def _payloads(output_root: Path, run_id: str) -> tuple[dict[str, object], dict[str, object], Path]:
    local_run_dir = output_root / run_id
    receipt = json.loads((local_run_dir / "receipt.json").read_text(encoding="utf-8"))
    result = json.loads((local_run_dir / "result.json").read_text(encoding="utf-8"))
    return receipt, result, local_run_dir / f"{run_id}.zip"


def _only_run_id(root: Path) -> str:
    directories = [path.name for path in root.iterdir() if path.is_dir()]
    assert len(directories) == 1
    return directories[0]


@pytest.mark.integration
def test_runner_uses_fixed_production_calls_and_exports_only_public_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    output_root = tmp_path / "output"
    run_store_root = tmp_path / "run-store"
    calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)

    rc = runner.execute(_args(repo, exact, output_root, run_store_root))
    run_id = _only_run_id(output_root)
    receipt, result, zip_path = _payloads(output_root, run_id)

    assert rc == receipt["rc"] == result["rc"] == 0
    assert receipt["resolved_exact"] == result["resolved_exact"] == exact
    assert receipt["scientific_status"] == result["scientific_status"] == "not_adjudicated"
    assert result["completeness"] == "complete_for_hf_v2_clean_confirmation_execution"
    assert receipt["evaluated_candidate_id"] == result["evaluated_candidate_id"] == (
        "hf_tail_rademacher_v1_rankgate_v2"
    )
    assert receipt["carrier_method_id"] == result["carrier_method_id"] == (
        "hf_tail_rademacher_v1"
    )
    assert len(result["records"]) == 16
    assert calls == {"load": 1, "hf": 8, "plain": 8, "score": 8 * 2 * 17}
    assert {record["arm"] for record in result["records"]} == {"hf_anchor", "primary_null"}
    assert Counter(record["unit_id"] for record in result["records"]) == {
        f"confirmation-{index:04d}": 2 for index in range(1, 9)
    }
    evidence = result["clean_confirmation_evidence"]
    assert evidence["candidate_outcome_allowed"] is True
    assert evidence["evaluation_status"] == "candidate_outcome"
    assert evidence["candidate_outcome"] in {"pass", "fail"}
    assert evidence["gate_a_required_units"] == evidence["gate_b_required_units"] == 7
    assert evidence["median_margin_is_gate"] is False
    assert evidence["mean_margin_is_gate"] is False
    assert evidence["min_margin_is_gate"] is False
    assert evidence["primary_null_cutoff_is_gate"] is False
    assert evidence["formal_fpr_claim"] is False
    assert all(len(record["scores"]) == 17 for record in result["records"])
    assert result["records"][0]["scores"] != result["records"][1]["scores"]
    assert runner.KEY_ENV not in os.environ
    assert runner.TOKEN_ENV not in os.environ
    assert receipt["checkpoint_interval_hours"] == runner.CHECKPOINT_INTERVAL_HOURS == 2.0
    assert receipt["model_id"] == "stabilityai/stable-diffusion-3.5-medium"
    assert not ({"model_revision", "vae_weight_digest", "full_weight_digest"} & set(receipt))
    assert zip_path.is_file()
    stored = run_store_root / run_id
    stored_zip = stored / f"{run_id}.zip"
    stored_checksum = stored / f"{run_id}.zip.sha256"
    assert stored_zip.is_file() and stored_checksum.is_file()
    assert not list(stored.glob("checkpoint-*.zip"))
    checksum_parts = stored_checksum.read_text(encoding="utf-8").strip().split()
    assert checksum_parts == [hashlib.sha256(stored_zip.read_bytes()).hexdigest(), stored_zip.name]
    with zipfile.ZipFile(stored_zip) as archive:
        assert set(archive.namelist()) == {"receipt.json", "result.json"}
        archived = b"".join(archive.read(name) for name in archive.namelist())
    exported = archived + stored_checksum.read_bytes()
    assert _RAW_KEY.encode() not in exported
    assert _HF_TOKEN.encode() not in exported
    assert b"A red ceramic teapot" not in exported
    for forbidden in (
        b"private_latent",
        b'"carrier":',
        b'"mask":',
        b"cached_qk",
        b"traceback",
    ):
        assert forbidden not in exported


@pytest.mark.integration
def test_timed_checkpoint_resume_skips_persisted_failure_and_retries_uncommitted_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, exact = _repo(tmp_path)
    run_store_root = tmp_path / "run-store"
    first_calls = _install_fakes(
        monkeypatch,
        fail_hf_calls=frozenset({1}),
        interrupt_hf_call=5,
    )
    clock = iter([0.0, 100.0, 7201.0, 7300.0, 14402.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)

    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(repo, exact, tmp_path / "first-output", run_store_root))

    run_id = _only_run_id(tmp_path / "first-output")
    stored = run_store_root / run_id
    checkpoint_files = sorted(stored.iterdir())
    assert len(checkpoint_files) == 4
    checkpoint_zip = sorted(stored.glob("checkpoint-*.zip"))[-1]
    with zipfile.ZipFile(checkpoint_zip) as archive:
        checkpoint_bytes = archive.read("state.json")
    assert _RAW_KEY.encode() not in checkpoint_bytes
    assert b"A red ceramic teapot" not in checkpoint_bytes
    assert b"private detail" not in checkpoint_bytes
    assert first_calls["hf"] == 5
    assert first_calls["plain"] == 3

    resumed_calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    resumed_output = tmp_path / "resumed-output"
    rc = runner.execute(_args(repo, exact, resumed_output, run_store_root))
    receipt, result, zip_path = _payloads(resumed_output, run_id)
    failed = [record for record in result["records"] if record["status"] != "success"]

    assert rc == receipt["rc"] == result["rc"] == 1
    assert len(result["records"]) == 16
    assert len(failed) == 2
    assert {record["unit_id"] for record in failed} == {"confirmation-0001"}
    assert result["clean_confirmation_evidence"]["evaluation_status"] == (
        "not_evaluable_operational"
    )
    assert result["clean_confirmation_evidence"]["candidate_outcome"] is None
    assert result["clean_confirmation_evidence"]["candidate_outcome_allowed"] is False
    assert receipt["completeness"] == result["completeness"] == (
        "incomplete_operational_execution"
    )
    assert all(record["failure_reason"] == "unit_execution_failure" for record in failed)
    assert resumed_calls == {"load": 1, "hf": 4, "plain": 4, "score": 4 * 2 * 17}
    assert zip_path.is_file()
    assert len(list(stored.glob("checkpoint-*.zip"))) == 2
    with zipfile.ZipFile(zip_path) as archive:
        assert b"private detail" not in b"".join(archive.read(name) for name in archive.namelist())

    verified_calls = _install_fakes(monkeypatch)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    assert runner.execute(_args(repo, exact, tmp_path / "verified-output", run_store_root)) == 0
    assert verified_calls == {"load": 0, "hf": 0, "plain": 0, "score": 0}

    highest_zip = stored / "checkpoint-0002-units-0004.zip"
    with zipfile.ZipFile(highest_zip) as archive:
        ambiguous_state = json.loads(archive.read("state.json"))
    ambiguous_state["checkpoint_sequence"] = 1
    ambiguous_zip = stored / "checkpoint-0001-units-0004.zip"
    with zipfile.ZipFile(ambiguous_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("state.json", json.dumps(ambiguous_state))
    ambiguous_checksum = stored / f"{ambiguous_zip.name}.sha256"
    ambiguous_checksum.write_text(
        f"{hashlib.sha256(ambiguous_zip.read_bytes()).hexdigest()}  {ambiguous_zip.name}\n",
        encoding="utf-8",
    )
    protocol = runner._load_protocol(repo)
    expected = runner._new_state(
        run_id=run_id,
        resolved_exact=exact,
        protocol=protocol,
        model_id=protocol.config["generation_runtime"]["model_id"],
        key_digest=runner.public_key_digest(_RAW_KEY),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        runner._discover_checkpoint(stored, expected)


@pytest.mark.integration
def test_due_checkpoint_at_final_boundary_and_verified_final_do_not_create_empty_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, exact = _repo(tmp_path)
    run_store_root = tmp_path / "run-store"
    calls = _install_fakes(monkeypatch)
    clock = iter([0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 7201.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)

    assert runner.execute(_args(repo, exact, tmp_path / "first-output", run_store_root)) == 0
    assert calls["hf"] == 8
    run_id = _only_run_id(tmp_path / "first-output")
    stored = run_store_root / run_id
    checkpoint_files = sorted(stored.glob("checkpoint-*.zip*"))
    assert len(checkpoint_files) == 2

    resumed_calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 999999.0)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    assert runner.execute(_args(repo, exact, tmp_path / "verified-output", run_store_root)) == 0
    assert resumed_calls == {"load": 0, "hf": 0, "plain": 0, "score": 0}
    assert sorted(stored.glob("checkpoint-*.zip*")) == checkpoint_files

    final_checksum = stored / f"{run_id}.zip.sha256"
    final_checksum.write_text(f"{'0' * 64}  {run_id}.zip\n", encoding="utf-8")
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    assert runner.execute(
        _args(repo, exact, tmp_path / "bad-final-output", run_store_root)
    ) == 0


@pytest.mark.integration
def test_all_success_checkpoint_publication_failure_is_rc1_without_candidate_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, exact = _repo(tmp_path)
    output_root = tmp_path / "output"
    run_store_root = tmp_path / "run-store"
    _install_fakes(monkeypatch)
    clock = iter([0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 7201.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    original_publish_pair = runner._publish_pair_create_only

    def fail_checkpoint_publication(
        zip_path: Path,
        checksum_path: Path,
        sink: Path,
        *,
        artifact_kind: str,
    ) -> None:
        if artifact_kind == "checkpoint":
            raise RuntimeError("simulated checkpoint publication failure")
        original_publish_pair(
            zip_path,
            checksum_path,
            sink,
            artifact_kind=artifact_kind,
        )

    monkeypatch.setattr(runner, "_publish_pair_create_only", fail_checkpoint_publication)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)

    assert runner.execute(_args(repo, exact, output_root, run_store_root)) == 1
    run_id = _only_run_id(output_root)
    receipt, result, _ = _payloads(output_root, run_id)
    evidence = result["clean_confirmation_evidence"]

    assert receipt["rc"] == result["rc"] == 1
    assert all(record["status"] == "success" for record in result["records"])
    assert evidence["successful_pair_count"] == 8
    assert evidence["candidate_outcome_allowed"] is False
    assert evidence["candidate_outcome"] is None
    assert evidence["gate_a_pass"] is evidence["gate_b_pass"] is None
    stored = run_store_root / run_id
    assert not list(stored.glob("checkpoint-*.zip*"))
    assert (stored / f"{run_id}.zip").is_file()
    assert (stored / f"{run_id}.zip.sha256").is_file()


@pytest.mark.integration
def test_all_success_final_publication_failure_preserves_local_rc0_outcome_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, exact = _repo(tmp_path)
    output_root = tmp_path / "output"
    run_store_root = tmp_path / "run-store"
    _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    original_publish_pair = runner._publish_terminal_pair_create_only

    def fail_final_publication(
        zip_path: Path,
        checksum_path: Path,
        sink: Path,
        *,
        artifact_kind: str,
    ) -> None:
        if artifact_kind == "final":
            raise RuntimeError("simulated final publication failure")
        original_publish_pair(
            zip_path,
            checksum_path,
            sink,
            artifact_kind=artifact_kind,
        )

    monkeypatch.setattr(runner, "_publish_terminal_pair_create_only", fail_final_publication)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    monkeypatch.setattr(sys, "argv", [
        "run_hf_a2_colab",
        "--repo-root", str(repo),
        "--output-root", str(output_root),
        "--expected-exact", exact,
        "--run-store-root", str(run_store_root),
    ])

    with pytest.raises(SystemExit) as exit_info:
        runner.main()

    assert exit_info.value.code == 2
    fatal_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("CEGWM_FATAL ")
    ]
    assert len(fatal_lines) == 1
    fatal_event = json.loads(fatal_lines[0].removeprefix("CEGWM_FATAL "))
    assert fatal_event["error_class"] == "final_export_failure"
    assert fatal_event["export_status"] == "local_only"
    run_id = fatal_event["run_id"]
    receipt, result, _ = _payloads(output_root, run_id)
    evidence = result["clean_confirmation_evidence"]
    assert receipt["rc"] == result["rc"] == 0
    assert all(record["status"] == "success" for record in result["records"])
    assert evidence["successful_pair_count"] == 8
    assert evidence["candidate_outcome_allowed"] is True
    assert evidence["candidate_outcome"] in {"pass", "fail"}
    local_final = output_root / run_id / f"{run_id}.zip"
    local_checksum = output_root / run_id / f"{run_id}.zip.sha256"
    assert local_final.is_file() and local_checksum.is_file()
    assert not (output_root / run_id / "failure-final_export_failure.zip").exists()
    assert not list((run_store_root / run_id).glob("*.zip*"))


@pytest.mark.integration
def test_terminal_package_is_copied_without_production_validation_or_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, exact = _repo(tmp_path)
    output_root = tmp_path / "output"
    run_store_root = tmp_path / "run-store"
    _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    events: list[str] = []
    original_publish = runner._publish_final

    def record_publish(
        zip_path: Path,
        checksum_path: Path,
        run_store: Path,
    ) -> None:
        events.append("publish")
        original_publish(zip_path, checksum_path, run_store)

    def reject_checksum_validation(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise AssertionError("terminal publication must not verify its completed pair")

    original_read_bytes = Path.read_bytes

    def reject_drive_readback(path: Path) -> bytes:
        if run_store_root in path.parents:
            raise AssertionError("terminal publication must not read back Drive bytes")
        return original_read_bytes(path)

    monkeypatch.setattr(runner, "_verify_checksum", reject_checksum_validation)
    monkeypatch.setattr(Path, "read_bytes", reject_drive_readback)
    monkeypatch.setattr(runner, "_publish_final", record_publish)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)

    assert runner.execute(_args(repo, exact, output_root, run_store_root)) == 0
    assert events == ["publish"]
    run_id = _only_run_id(output_root)
    stored = run_store_root / run_id
    assert (stored / f"{run_id}.zip").is_file()
    assert (stored / f"{run_id}.zip.sha256").is_file()


@pytest.mark.integration
def test_partial_terminal_copy_is_never_deleted_or_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_zip = tmp_path / "run.zip"
    local_checksum = tmp_path / "run.zip.sha256"
    sink = tmp_path / "drive"
    sink.mkdir()
    local_zip.write_bytes(b"complete-local-zip")
    local_checksum.write_text("digest  run.zip\n", encoding="utf-8")
    original_copy = runner.shutil.copyfileobj
    copy_count = 0

    def fail_second_copy(source: object, target: object) -> None:
        nonlocal copy_count
        copy_count += 1
        if copy_count == 2:
            target.write(b"partial")
            raise OSError("simulated Drive interruption")
        original_copy(source, target)

    monkeypatch.setattr(runner.shutil, "copyfileobj", fail_second_copy)
    with pytest.raises(OSError, match="Drive interruption"):
        runner._publish_terminal_pair_create_only(
            local_zip,
            local_checksum,
            sink,
            artifact_kind="final",
        )

    assert (sink / local_zip.name).read_bytes() == local_zip.read_bytes()
    assert (sink / local_checksum.name).read_bytes() == b"partial"
    with pytest.raises(RuntimeError, match="refuses overwrite"):
        runner._publish_terminal_pair_create_only(
            local_zip,
            local_checksum,
            sink,
            artifact_kind="final",
        )


@pytest.mark.integration
def test_run_identity_is_deterministic_and_cli_keeps_fixed_interval_internal(tmp_path: Path) -> None:
    repo, exact = _repo(tmp_path)
    protocol = runner._load_protocol(repo)
    model_id = protocol.config["generation_runtime"]["model_id"]
    key_digest = runner.public_key_digest(_RAW_KEY)
    first = runner._deterministic_run_id(exact, protocol, model_id, key_digest)

    assert first == runner._deterministic_run_id(exact, protocol, model_id, key_digest)
    assert first != runner._deterministic_run_id("f" * 40, protocol, model_id, key_digest)
    assert first != runner._deterministic_run_id(
        exact,
        protocol,
        model_id,
        runner.public_key_digest("different-stage-a-root-key-0002"),
    )
    parsed = runner._parser().parse_args([
        "--repo-root", str(repo),
        "--output-root", str(tmp_path / "parsed-output"),
        "--expected-exact", exact,
        "--run-store-root", str(tmp_path / "run-store"),
    ])
    assert set(vars(parsed)) == {"repo_root", "output_root", "expected_exact", "run_store_root"}
    assert runner.CHECKPOINT_INTERVAL_HOURS == 2.0

    empty_state = runner._new_state(
        run_id=first,
        resolved_exact=exact,
        protocol=protocol,
        model_id=model_id,
        key_digest=key_digest,
    )
    empty_state["checkpoint_sequence"] = 1
    empty_zip = tmp_path / "checkpoint-0001-units-0000.zip"
    with zipfile.ZipFile(empty_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("state.json", json.dumps(empty_state))
    empty_checksum = tmp_path / f"{empty_zip.name}.sha256"
    empty_checksum.write_text(
        f"{hashlib.sha256(empty_zip.read_bytes()).hexdigest()}  {empty_zip.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        runner._resume_state(empty_zip, empty_checksum, empty_state | {"checkpoint_sequence": 0})


@pytest.mark.integration
def test_rank_gate_uses_all_units_strict_ties_and_no_absolute_margin() -> None:
    protocol = runner._load_protocol(_ROOT)
    exact = "e" * 40
    key_digest = runner.public_key_digest(_RAW_KEY)
    run_id = runner._deterministic_run_id(
        exact,
        protocol,
        protocol.config["generation_runtime"]["model_id"],
        key_digest,
    )
    expected = runner._new_state(
        run_id=run_id,
        resolved_exact=exact,
        protocol=protocol,
        model_id=protocol.config["generation_runtime"]["model_id"],
        key_digest=key_digest,
    )
    def evidence_for(gate_a_units: int, gate_b_units: int) -> dict[str, object]:
        wrong_scores = {f"wrong_{index:02d}": 0.0 for index in range(16)}
        records = []
        for index, unit in enumerate(protocol.untouched_confirmation):
            registered = 1.0e-9 if index < gate_a_units else 0.0
            primary_null_registered = (
                registered - 1.0 if index < gate_b_units else registered + 1.0
            )
            common = dict(
                run_id=run_id,
                unit_id=unit.unit_id,
                source_cluster_id=unit.source_id,
                condition="identity",
                code_revision=exact,
                config_digest=protocol.protocol_digest,
                key_public_digest=key_digest,
                status="success",
            )
            records.extend([
                runner.StageARecord(
                    arm="hf_anchor",
                    scores={"registered": registered, **wrong_scores},
                    **common,
                ),
                runner.StageARecord(
                    arm="primary_null",
                    scores={"registered": primary_null_registered, **wrong_scores},
                    **common,
                ),
            ])
        return runner._clean_confirmation_evidence(
            records,
            expected,
            candidate_outcome_allowed=True,
        )

    evidence = evidence_for(7, 7)

    assert evidence["evaluation_status"] == "candidate_outcome"
    assert evidence["candidate_outcome_allowed"] is True
    assert evidence["gate_a_registered_top_rank_units"] == 7
    assert evidence["gate_b_paired_hf_gt_primary_null_units"] == 7
    assert evidence["gate_a_pass"] is evidence["gate_b_pass"] is True
    assert evidence["candidate_outcome"] == "pass"
    assert evidence["median_correct_minus_wrong_key_max_effect_size"] == pytest.approx(1.0e-9)
    assert evidence["mean_correct_minus_wrong_key_max_effect_size"] == pytest.approx(
        0.875e-9
    )
    assert evidence["min_correct_minus_wrong_key_max_effect_size"] == 0.0
    assert evidence["median_margin_is_gate"] is False
    assert evidence["mean_margin_is_gate"] is False
    assert evidence["min_margin_is_gate"] is False
    assert evidence["unit_evidence"][-1]["registered_top_rank"] is False

    gate_a_fail = evidence_for(6, 7)
    assert gate_a_fail["successful_pair_count"] == 8
    assert gate_a_fail["gate_a_pass"] is False
    assert gate_a_fail["gate_b_pass"] is True
    assert gate_a_fail["candidate_outcome"] == "fail"

    gate_b_fail = evidence_for(7, 6)
    assert gate_b_fail["successful_pair_count"] == 8
    assert gate_b_fail["gate_a_pass"] is True
    assert gate_b_fail["gate_b_pass"] is False
    assert gate_b_fail["candidate_outcome"] == "fail"


@pytest.mark.integration
def test_runner_requires_nonempty_root_key_and_explicit_hf_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, exact = _repo(tmp_path)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, "   ")
    with pytest.raises(RuntimeError, match="hugging_face_token"):
        runner.execute(_args(repo, exact, tmp_path / "token-output", tmp_path / "token-store"))
    assert runner.KEY_ENV not in os.environ and runner.TOKEN_ENV not in os.environ

    monkeypatch.setenv(runner.KEY_ENV, "")
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    with pytest.raises(RuntimeError, match="root_key"):
        runner.execute(_args(repo, exact, tmp_path / "key-output", tmp_path / "key-store"))
    assert runner.KEY_ENV not in os.environ and runner.TOKEN_ENV not in os.environ


@pytest.mark.integration
def test_top_level_resume_failure_exports_only_sanitized_partial_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, exact = _repo(tmp_path)
    output_root = tmp_path / "output"
    run_store_root = tmp_path / "run-store"
    protocol = runner._load_protocol(repo)
    model_id = protocol.config["generation_runtime"]["model_id"]
    run_id = runner._deterministic_run_id(
        exact,
        protocol,
        model_id,
        runner.public_key_digest(_RAW_KEY),
    )
    stored = run_store_root / run_id
    stored.mkdir(parents=True)
    resume_zip = stored / "checkpoint-0001-units-0002.zip"
    resume_checksum = stored / "checkpoint-0001-units-0002.zip.sha256"
    private_detail = "private checkpoint detail that must not be exported"
    with zipfile.ZipFile(resume_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("state.json", json.dumps({"private": private_detail}))
    resume_checksum.write_text(
        f"{hashlib.sha256(resume_zip.read_bytes()).hexdigest()}  {resume_zip.name}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    monkeypatch.setenv("CEGWM_PRIVATE_DETAIL", private_detail)
    monkeypatch.setattr(sys, "argv", [
        "run_hf_a2_colab",
        "--repo-root", str(repo),
        "--output-root", str(output_root),
        "--expected-exact", exact,
        "--run-store-root", str(run_store_root),
    ])

    with pytest.raises(SystemExit) as exit_info:
        runner.main()

    assert exit_info.value.code == 2
    stdout = capsys.readouterr().out
    fatal_lines = [line for line in stdout.splitlines() if line.startswith("CEGWM_FATAL ")]
    assert len(fatal_lines) == 1
    fatal_event = json.loads(fatal_lines[0].removeprefix("CEGWM_FATAL "))
    assert fatal_event == {
        "run_id": run_id,
        "error_class": "resume_validation_failure",
        "export_status": "published",
    }
    assert _RAW_KEY not in stdout and private_detail not in stdout
    assert _HF_TOKEN not in stdout
    local_run_dir = output_root / run_id
    receipt = json.loads((local_run_dir / "receipt.json").read_text(encoding="utf-8"))
    result = json.loads((local_run_dir / "result.json").read_text(encoding="utf-8"))
    zip_path = local_run_dir / "failure-resume_validation_failure.zip"
    checksum_path = local_run_dir / f"{zip_path.name}.sha256"
    assert receipt["rc"] == result["rc"] == 2
    assert receipt["completeness"] == result["completeness"] == (
        "incomplete_operational_execution"
    )
    assert receipt["error_class"] == result["error_class"] == "resume_validation_failure"
    assert receipt["result_kind"] == result["result_kind"] == "operational_failure_not_scientific"
    assert receipt["resume_status"] == result["resume_status"] == "rejected"
    assert receipt["approved_execution_exact"] == result["approved_execution_exact"] == exact
    assert receipt["resolved_exact"] == result["resolved_exact"] == exact
    assert receipt["protocol_digest"] == result["protocol_digest"]
    assert receipt["model_id"] == result["model_id"] == model_id
    assert len(receipt["ordered_roster_unit_ids"]) == len(result["ordered_roster_unit_ids"]) == 8
    assert receipt["committed_unit_count"] == result["committed_unit_count"] == 0
    assert receipt["committed_unit_ids"] == result["committed_unit_ids"] == []
    assert result["records"] == []
    assert result["clean_confirmation_evidence"]["candidate_outcome_allowed"] is False
    assert result["fixed_unit_count"] == 8 and result["fixed_record_count"] == 16
    with zipfile.ZipFile(zip_path) as archive:
        exported = b"".join(archive.read(name) for name in archive.namelist())
    exported += b"".join(
        path.read_bytes()
        for path in (zip_path.parent / "receipt.json", zip_path.parent / "result.json", checksum_path)
    )
    assert _RAW_KEY.encode() not in exported
    assert _HF_TOKEN.encode() not in exported
    assert private_detail.encode() not in exported
    assert b"traceback" not in exported
    stored_zip = stored / zip_path.name
    stored_checksum = stored / checksum_path.name
    assert stored_zip.read_bytes() == zip_path.read_bytes()
    assert stored_checksum.read_bytes() == checksum_path.read_bytes()
    assert not (stored / f"{run_id}.zip").exists()

    second_output = tmp_path / "second-output"
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    monkeypatch.setattr(sys, "argv", [
        "run_hf_a2_colab",
        "--repo-root", str(repo),
        "--output-root", str(second_output),
        "--expected-exact", exact,
        "--run-store-root", str(run_store_root),
    ])
    with pytest.raises(SystemExit) as second_exit:
        runner.main()
    assert second_exit.value.code == 2
    second_event = json.loads(capsys.readouterr().out.strip().removeprefix("CEGWM_FATAL "))
    assert second_event == {
        "run_id": run_id,
        "error_class": "resume_validation_failure",
        "export_status": "present_for_external_validation",
    }
    assert not second_output.exists()

    stored_zip.write_bytes(b"corrupt")
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    monkeypatch.setattr(sys, "argv", [
        "run_hf_a2_colab",
        "--repo-root", str(repo),
        "--output-root", str(tmp_path / "third-output"),
        "--expected-exact", exact,
        "--run-store-root", str(run_store_root),
    ])
    with pytest.raises(SystemExit) as third_exit:
        runner.main()
    assert third_exit.value.code == 2
    third_event = json.loads(capsys.readouterr().out.strip().removeprefix("CEGWM_FATAL "))
    assert third_event["export_status"] == "present_for_external_validation"


@pytest.mark.integration
def test_runtime_fatal_preserves_only_the_real_committed_pair_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, exact = _repo(tmp_path)
    output_root = tmp_path / "output"
    run_store_root = tmp_path / "run-store"
    _install_fakes(monkeypatch, interrupt_hf_call=2)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        runner,
        "_verify_checksum",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("terminal failure publication must not verify its pair")
        ),
    )
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    monkeypatch.setattr(sys, "argv", [
        "run_hf_a2_colab",
        "--repo-root", str(repo),
        "--output-root", str(output_root),
        "--expected-exact", exact,
        "--run-store-root", str(run_store_root),
    ])

    with pytest.raises(SystemExit) as exit_info:
        runner.main()

    assert exit_info.value.code == 2
    lines = capsys.readouterr().out.splitlines()
    fatal_event = json.loads(
        next(line for line in lines if line.startswith("CEGWM_FATAL ")).removeprefix("CEGWM_FATAL ")
    )
    assert fatal_event["error_class"] == "runtime_execution_failure"
    assert fatal_event["export_status"] == "published"
    run_id = fatal_event["run_id"]
    local_run_dir = output_root / run_id
    result = json.loads((local_run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["rc"] == 2
    assert result["committed_unit_count"] == 1
    assert result["committed_unit_ids"] == ["confirmation-0001"]
    assert len(result["records"]) == 2
    assert [record["unit_id"] for record in result["records"]] == [
        "confirmation-0001"
    ] * 2
    assert result["clean_confirmation_evidence"]["evaluation_status"] == (
        "not_evaluable_operational"
    )
    assert result["clean_confirmation_evidence"]["candidate_outcome_allowed"] is False
    assert [record["arm"] for record in result["records"]] == ["hf_anchor", "primary_null"]
    stored = run_store_root / run_id
    assert (stored / "failure-runtime_execution_failure.zip").is_file()
    protocol = runner._load_protocol(repo)
    expected = runner._new_state(
        run_id=run_id,
        resolved_exact=exact,
        protocol=protocol,
        model_id=protocol.config["generation_runtime"]["model_id"],
        key_digest=runner.public_key_digest(_RAW_KEY),
    )
    assert runner._discover_checkpoint(stored, expected) is None
    assert not (stored / f"{run_id}.zip").exists()
    exported = b"".join(path.read_bytes() for path in local_run_dir.iterdir() if path.is_file())
    assert _RAW_KEY.encode() not in exported and _HF_TOKEN.encode() not in exported
