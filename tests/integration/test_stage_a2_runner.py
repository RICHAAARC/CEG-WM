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
    run_id: str,
    checkpoint_sink: Path,
    *,
    interval: float = 2.0,
    resume_zip: Path | None = None,
    resume_checksum: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(repo),
        output_root=str(output_root),
        expected_exact=exact,
        model_revision="c" * 40,
        run_id=run_id,
        checkpoint_sink=str(checkpoint_sink),
        checkpoint_interval_hours=interval,
        resume_zip=str(resume_zip) if resume_zip else None,
        resume_checksum=str(resume_checksum) if resume_checksum else None,
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_hf_calls: frozenset[int] = frozenset(),
    interrupt_hf_call: int | None = None,
) -> dict[str, int]:
    calls = {"hf": 0, "plain": 0, "score": 0}
    assets = SimpleNamespace(vae_weight_digest="d" * 64)
    monkeypatch.setattr(runner, "_load_pipeline_and_assets", lambda model_id, revision: (object(), assets))
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


@pytest.mark.integration
def test_runner_uses_fixed_production_calls_and_exports_only_public_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    output_root = tmp_path / "output"
    checkpoint_sink = tmp_path / "checkpoint-sink"
    checkpoint_sink.mkdir()
    calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)

    rc = runner.execute(_args(repo, exact, output_root, "a2-run-success", checkpoint_sink))
    receipt, result, zip_path = _payloads(output_root, "a2-run-success")

    assert rc == receipt["rc"] == result["rc"] == 0
    assert receipt["resolved_exact"] == result["resolved_exact"] == exact
    assert receipt["scientific_status"] == result["scientific_status"] == "not_evaluated"
    assert result["completeness"] == "incomplete_for_hf_anchor"
    assert len(result["records"]) == 16
    assert calls == {"hf": 8, "plain": 8, "score": 8 * 2 * 17}
    assert {record["arm"] for record in result["records"]} == {"hf_anchor", "primary_null"}
    assert Counter(record["unit_id"] for record in result["records"]) == {
        f"selection-{index:04d}": 2 for index in range(1, 9)
    }
    assert all(len(record["scores"]) == 17 for record in result["records"])
    assert result["records"][0]["scores"] != result["records"][1]["scores"]
    assert runner.KEY_ENV not in os.environ
    assert receipt["checkpoint_interval_hours"] == 2.0
    assert not list(checkpoint_sink.iterdir())
    assert zip_path.is_file()
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
    checkpoint_sink = tmp_path / "checkpoint-sink"
    checkpoint_sink.mkdir()
    first_calls = _install_fakes(
        monkeypatch,
        fail_hf_calls=frozenset({1}),
        interrupt_hf_call=3,
    )
    clock = iter([0.0, 100.0, 3601.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)

    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(
            repo,
            exact,
            tmp_path / "first-output",
            "a2-run-resume",
            checkpoint_sink,
            interval=1.0,
        ))

    checkpoint_files = sorted(checkpoint_sink.iterdir())
    assert {path.suffix for path in checkpoint_files} == {".sha256", ".zip"}
    checkpoint_zip = next(path for path in checkpoint_files if path.suffix == ".zip")
    checkpoint_checksum = next(path for path in checkpoint_files if path.suffix == ".sha256")
    with zipfile.ZipFile(checkpoint_zip) as archive:
        checkpoint_bytes = archive.read("state.json")
    assert _RAW_KEY.encode() not in checkpoint_bytes
    assert b"A red ceramic teapot" not in checkpoint_bytes
    assert b"private detail" not in checkpoint_bytes
    assert first_calls["hf"] == 3
    assert first_calls["plain"] == 1

    resumed_calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    resumed_output = tmp_path / "resumed-output"
    rc = runner.execute(_args(
        repo,
        exact,
        resumed_output,
        "a2-run-resume",
        checkpoint_sink,
        interval=1.0,
        resume_zip=checkpoint_zip,
        resume_checksum=checkpoint_checksum,
    ))
    receipt, result, zip_path = _payloads(resumed_output, "a2-run-resume")
    failed = [record for record in result["records"] if record["status"] != "success"]

    assert rc == receipt["rc"] == result["rc"] == 1
    assert len(result["records"]) == 16
    assert len(failed) == 2
    assert {record["unit_id"] for record in failed} == {"selection-0001"}
    assert all(record["failure_reason"] == "unit_execution_failure" for record in failed)
    assert resumed_calls["hf"] == 6
    assert resumed_calls["plain"] == 6
    assert zip_path.is_file()
    assert sorted(checkpoint_sink.iterdir()) == checkpoint_files
    with zipfile.ZipFile(zip_path) as archive:
        assert b"private detail" not in b"".join(archive.read(name) for name in archive.namelist())

    monkeypatch.setenv(runner.KEY_ENV, "different-stage-a-detection-key-0002")
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.execute(_args(
            repo,
            exact,
            tmp_path / "mismatch-output",
            "a2-run-resume",
            checkpoint_sink,
            interval=1.0,
            resume_zip=checkpoint_zip,
            resume_checksum=checkpoint_checksum,
        ))


@pytest.mark.integration
def test_due_checkpoint_after_final_safe_boundary_resumes_without_empty_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, exact = _repo(tmp_path)
    checkpoint_sink = tmp_path / "checkpoint-sink"
    checkpoint_sink.mkdir()
    calls = _install_fakes(monkeypatch)
    clock = iter([0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 3601.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)

    assert runner.execute(_args(
        repo,
        exact,
        tmp_path / "first-output",
        "a2-run-complete-checkpoint",
        checkpoint_sink,
        interval=1.0,
    )) == 0
    assert calls["hf"] == 8
    checkpoint_files = sorted(checkpoint_sink.iterdir())
    assert len(checkpoint_files) == 2
    checkpoint_zip = next(path for path in checkpoint_files if path.suffix == ".zip")
    checkpoint_checksum = next(path for path in checkpoint_files if path.suffix == ".sha256")

    resumed_calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 999999.0)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    resumed_output = tmp_path / "resumed-output"
    assert runner.execute(_args(
        repo,
        exact,
        resumed_output,
        "a2-run-complete-checkpoint",
        checkpoint_sink,
        interval=1.0,
        resume_zip=checkpoint_zip,
        resume_checksum=checkpoint_checksum,
    )) == 0
    _, result, _ = _payloads(resumed_output, "a2-run-complete-checkpoint")
    assert len(result["records"]) == 16
    assert resumed_calls == {"hf": 0, "plain": 0, "score": 0}
    assert sorted(checkpoint_sink.iterdir()) == checkpoint_files


@pytest.mark.integration
def test_checkpoint_interval_range_is_fail_closed(tmp_path: Path) -> None:
    repo, exact = _repo(tmp_path)
    checkpoint_sink = tmp_path / "checkpoint-sink"
    checkpoint_sink.mkdir()
    parsed = runner._parser().parse_args([
        "--repo-root", str(repo),
        "--output-root", str(tmp_path / "parsed-output"),
        "--expected-exact", exact,
        "--model-revision", "c" * 40,
        "--run-id", "a2-run-default-interval",
        "--checkpoint-sink", str(checkpoint_sink),
    ])
    assert parsed.checkpoint_interval_hours == 2.0

    with pytest.raises(ValueError, match=r"\[1.0, 2.0\]"):
        runner.execute(_args(
            repo,
            exact,
            tmp_path / "output",
            "a2-run-bad-interval",
            checkpoint_sink,
            interval=0.5,
        ))


@pytest.mark.integration
def test_top_level_resume_failure_exports_only_sanitized_partial_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, exact = _repo(tmp_path)
    output_root = tmp_path / "output"
    checkpoint_sink = tmp_path / "checkpoint-sink"
    checkpoint_sink.mkdir()
    resume_zip = tmp_path / "resume.zip"
    resume_checksum = tmp_path / "resume.zip.sha256"
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
        "--model-revision", "c" * 40,
        "--run-id", "a2-run-fatal-package",
        "--checkpoint-sink", str(checkpoint_sink),
        "--resume-zip", str(resume_zip),
        "--resume-checksum", str(resume_checksum),
    ])

    with pytest.raises(SystemExit) as exit_info:
        runner.main()

    assert exit_info.value.code == 2
    stdout = capsys.readouterr().out
    assert "resume_validation_failure" in stdout
    assert _RAW_KEY not in stdout and private_detail not in stdout
    receipt, result, zip_path = _payloads(output_root, "a2-run-fatal-package")
    assert receipt["rc"] == result["rc"] == 2
    assert receipt["error_class"] == result["error_class"] == "resume_validation_failure"
    assert receipt["result_kind"] == result["result_kind"] == "operational_failure_not_scientific"
    assert receipt["resume_status"] == result["resume_status"] == "rejected"
    assert receipt["approved_execution_exact"] == result["approved_execution_exact"] == exact
    assert receipt["resolved_exact"] == result["resolved_exact"] == exact
    assert receipt["protocol_digest"] == result["protocol_digest"]
    assert receipt["model_revision"] == result["model_revision"] == "c" * 40
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
