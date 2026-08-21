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

    def fake_load(model_id: str) -> tuple[object, object]:
        assert model_id == "stabilityai/stable-diffusion-3.5-medium"
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

    rc = runner.execute(_args(repo, exact, output_root, run_store_root))
    run_id = _only_run_id(output_root)
    receipt, result, zip_path = _payloads(output_root, run_id)

    assert rc == receipt["rc"] == result["rc"] == 0
    assert receipt["resolved_exact"] == result["resolved_exact"] == exact
    assert receipt["scientific_status"] == result["scientific_status"] == "not_evaluated"
    assert result["completeness"] == "incomplete_for_hf_anchor"
    assert len(result["records"]) == 16
    assert calls == {"load": 1, "hf": 8, "plain": 8, "score": 8 * 2 * 17}
    assert {record["arm"] for record in result["records"]} == {"hf_anchor", "primary_null"}
    assert Counter(record["unit_id"] for record in result["records"]) == {
        f"selection-{index:04d}": 2 for index in range(1, 9)
    }
    assert all(len(record["scores"]) == 17 for record in result["records"])
    assert result["records"][0]["scores"] != result["records"][1]["scores"]
    assert runner.KEY_ENV not in os.environ
    assert receipt["checkpoint_interval_hours"] == runner.CHECKPOINT_INTERVAL_HOURS == 2.0
    assert receipt["model_id"] == "stabilityai/stable-diffusion-3.5-medium"
    assert not ({"model_revision", "vae_weight_digest", "full_weight_digest"} & set(receipt))
    assert zip_path.is_file()
    stored = run_store_root / run_id
    assert (stored / f"{run_id}.zip").is_file()
    assert (stored / f"{run_id}.zip.sha256").is_file()
    assert not list(stored.glob("checkpoint-*.zip"))
    with zipfile.ZipFile(zip_path) as archive:
        assert set(archive.namelist()) == {"receipt.json", "result.json"}
        archived = b"".join(archive.read(name) for name in archive.namelist())
    exported = b"".join(path.read_bytes() for path in (zip_path.parent / "receipt.json", zip_path.parent / "result.json")) + archived
    assert _RAW_KEY.encode() not in exported
    assert b"A red ceramic teapot" not in exported
    for forbidden in (b"private_latent", b"carrier", b"cached_qk", b"traceback"):
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
    resumed_output = tmp_path / "resumed-output"
    rc = runner.execute(_args(repo, exact, resumed_output, run_store_root))
    receipt, result, zip_path = _payloads(resumed_output, run_id)
    failed = [record for record in result["records"] if record["status"] != "success"]

    assert rc == receipt["rc"] == result["rc"] == 1
    assert len(result["records"]) == 16
    assert len(failed) == 2
    assert {record["unit_id"] for record in failed} == {"selection-0001"}
    assert all(record["failure_reason"] == "unit_execution_failure" for record in failed)
    assert resumed_calls == {"load": 1, "hf": 4, "plain": 4, "score": 4 * 2 * 17}
    assert zip_path.is_file()
    assert len(list(stored.glob("checkpoint-*.zip"))) == 2
    with zipfile.ZipFile(zip_path) as archive:
        assert b"private detail" not in b"".join(archive.read(name) for name in archive.namelist())

    verified_calls = _install_fakes(monkeypatch)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    assert runner.execute(_args(repo, exact, tmp_path / "verified-output", run_store_root)) == 1
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

    assert runner.execute(_args(repo, exact, tmp_path / "first-output", run_store_root)) == 0
    assert calls["hf"] == 8
    run_id = _only_run_id(tmp_path / "first-output")
    stored = run_store_root / run_id
    checkpoint_files = sorted(stored.glob("checkpoint-*.zip*"))
    assert len(checkpoint_files) == 2

    resumed_calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 999999.0)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    assert runner.execute(_args(repo, exact, tmp_path / "verified-output", run_store_root)) == 0
    assert resumed_calls == {"load": 0, "hf": 0, "plain": 0, "score": 0}
    assert sorted(stored.glob("checkpoint-*.zip*")) == checkpoint_files

    final_checksum = stored / f"{run_id}.zip.sha256"
    final_checksum.write_text(f"{'0' * 64}  {run_id}.zip\n", encoding="utf-8")
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.execute(_args(repo, exact, tmp_path / "bad-final-output", run_store_root))


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
    assert "resume_validation_failure" in stdout
    assert _RAW_KEY not in stdout and private_detail not in stdout
    receipt, result, zip_path = _payloads(output_root, run_id)
    assert receipt["rc"] == result["rc"] == 2
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
    assert result["fixed_unit_count"] == 8 and result["fixed_record_count"] == 16
    with zipfile.ZipFile(zip_path) as archive:
        exported = b"".join(archive.read(name) for name in archive.namelist())
    exported += b"".join(
        path.read_bytes() for path in (zip_path.parent / "receipt.json", zip_path.parent / "result.json")
    )
    assert _RAW_KEY.encode() not in exported
    assert private_detail.encode() not in exported
    assert b"traceback" not in exported
