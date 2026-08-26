from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from PIL import Image

from cegwm.protocol.records import StageARecord
from cegwm.shared.keys import normalize_detection_key
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


def _args(repo: Path, exact: str, output_root: Path, run_store_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(repo),
        output_root=str(output_root),
        expected_exact=exact,
        run_store_root=str(run_store_root),
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_lf_calls: frozenset[int] = frozenset(),
    interrupt_lf_call: int | None = None,
    nonfinite_scores: bool = False,
) -> dict[str, int]:
    calls = {"load": 0, "lf": 0, "plain": 0, "score": 0}
    registered_key = normalize_detection_key(_RAW_KEY)
    assets = SimpleNamespace(
        candidate_id=runner.LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        evaluated_candidate_id=runner.LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
        detector_statistic_id=runner.LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    )

    def fake_load(model_id: str, hf_token: str) -> tuple[object, object]:
        assert model_id == "stabilityai/stable-diffusion-3.5-medium"
        assert hf_token == _HF_TOKEN
        calls["load"] += 1
        return object(), assets

    monkeypatch.setattr(runner, "_load_pipeline_and_assets", fake_load)
    monkeypatch.setattr(runner.torch, "Generator", _Generator)

    def fake_lf(
        pipeline: object,
        prompt: str,
        key: bytes,
        public_assets: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        del pipeline, prompt
        calls["lf"] += 1
        if calls["lf"] == interrupt_lf_call:
            raise KeyboardInterrupt
        if calls["lf"] in fail_lf_calls:
            raise RuntimeError("private detail that must not be exported")
        seed = kwargs["generator"].seed
        assert public_assets.candidate_id == runner.LF_BALANCED_BLOCKS_CARRIER_METHOD_ID
        pixels = np.full((8, 8, 3), (seed % 20) + 130, dtype=np.uint8)
        return SimpleNamespace(
            image=Image.fromarray(pixels),
            injection_budget=SimpleNamespace(relative_l2=0.011999),
        )

    def fake_plain(pipeline: object, prompt: str, **kwargs: object) -> Image.Image:
        del pipeline, prompt
        calls["plain"] += 1
        seed = kwargs["generator"].seed
        return Image.fromarray(np.full((8, 8, 3), (seed % 20) + 20, dtype=np.uint8))

    def fake_score(image: Image.Image, key: bytes, public_assets: object) -> float:
        calls["score"] += 1
        if nonfinite_scores:
            return float("nan")
        image_value = float(np.asarray(image).mean() / 255.0)
        assert public_assets.detector_statistic_id == runner.LF_BLOCKNORM_DETECTOR_STATISTIC_ID
        return image_value + (0.25 if key == registered_key else key[0] / 4096.0)

    monkeypatch.setattr(runner, "run_sd35_lf", fake_lf)
    monkeypatch.setattr(runner, "run_sd35_plain", fake_plain)
    monkeypatch.setattr(runner, "score_lf_image", fake_score)
    return calls


def _payloads(output_root: Path, run_id: str) -> tuple[dict[str, object], dict[str, object], Path]:
    local = output_root / run_id
    receipt = json.loads((local / "receipt.json").read_text(encoding="utf-8"))
    result = json.loads((local / "result.json").read_text(encoding="utf-8"))
    return receipt, result, local / f"{run_id}.zip"


def _only_run_id(root: Path) -> str:
    names = [path.name for path in root.iterdir() if path.is_dir()]
    assert len(names) == 1
    return names[0]


@pytest.mark.integration
def test_runner_executes_fixed_lf_confirmation_transaction_and_exports_public_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, exact = _repo(tmp_path)
    output_root = tmp_path / "output"
    store_root = tmp_path / "store"
    calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)

    rc = runner.execute(_args(repo, exact, output_root, store_root))
    run_id = _only_run_id(output_root)
    receipt, result, local_zip = _payloads(output_root, run_id)

    assert rc == receipt["rc"] == result["rc"] == 0
    assert result["execution_scope_id"] == runner.EXECUTION_SCOPE_ID
    assert result["completeness"] == (
        "complete_for_lf_balanced_blocks_untouched_confirmation_execution"
    )
    assert result["carrier_method_id"] == runner.LF_BALANCED_BLOCKS_CARRIER_METHOD_ID
    assert result["evaluated_candidate_id"] == runner.LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID
    assert result["detector_statistic_id"] == runner.LF_BLOCKNORM_DETECTOR_STATISTIC_ID
    assert result["record_arms_in_exact_unit_order"] == list(runner.RECORD_ARMS)
    assert len(result["records"]) == 16
    assert calls == {"load": 1, "lf": 8, "plain": 8, "score": 8 * 2 * 17}
    assert Counter(record["unit_id"] for record in result["records"]) == {
        f"lfbb-confirmation-{index:04d}": 2 for index in range(1, 9)
    }
    for index in range(8):
        assert [record["arm"] for record in result["records"][index * 2 : index * 2 + 2]] == list(
            runner.RECORD_ARMS
        )
    evidence = result["clean_confirmation_evidence"]
    assert evidence["confirmation_outcome_allowed"] is True
    assert evidence["evaluation_status"] == "confirmation_outcome"
    assert evidence["confirmation_outcome"] == (
        "confirmation_pass_candidate_for_agent5_adjudication"
    )
    assert evidence["confirmation_pass_candidate_id"] == runner.LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID
    assert evidence["fixed_unit_count"] == 8
    assert evidence["fixed_record_count"] == 16
    assert evidence["median_margin_is_gate"] is False
    assert evidence["primary_null_cutoff_is_gate"] is False
    assert evidence["formal_fpr_claim"] is False
    assert all(record["status"] == "success" for record in result["records"])
    assert all(len(record["scores"]) == 17 for record in result["records"])
    assert runner.KEY_ENV not in os.environ and runner.TOKEN_ENV not in os.environ
    assert receipt["checkpoint_interval_hours"] == runner.CHECKPOINT_INTERVAL_HOURS == 2.0
    stored_zip = store_root / run_id / f"{run_id}.zip"
    stored_sha = store_root / run_id / f"{run_id}.zip.sha256"
    assert local_zip.is_file() and stored_zip.is_file() and stored_sha.is_file()
    with zipfile.ZipFile(stored_zip) as archive:
        assert archive.namelist() == ["receipt.json", "result.json"]
        exported = b"".join(archive.read(name) for name in archive.namelist()) + stored_sha.read_bytes()
    assert _RAW_KEY.encode() not in exported
    assert _HF_TOKEN.encode() not in exported
    assert b"A glassblower" not in exported
    for forbidden in (b"private_latent", b'"carrier":', b'"mask":', b"traceback"):
        assert forbidden not in exported


@pytest.mark.integration
def test_checkpoint_resume_skips_full_committed_transactions_and_reruns_interrupted_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, exact = _repo(tmp_path)
    store_root = tmp_path / "store"
    first_calls = _install_fakes(
        monkeypatch,
        fail_lf_calls=frozenset({1}),
        interrupt_lf_call=3,
    )
    clock = iter([0.0, 100.0, 7201.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)

    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(repo, exact, tmp_path / "first", store_root))
    run_id = _only_run_id(tmp_path / "first")
    checkpoint_zip = next((store_root / run_id).glob("checkpoint-*.zip"))
    with zipfile.ZipFile(checkpoint_zip) as archive:
        state = json.loads(archive.read("state.json"))
    assert state["execution_scope_id"] == runner.EXECUTION_SCOPE_ID
    assert state["committed_unit_ids"] == [
        "lfbb-confirmation-0001",
        "lfbb-confirmation-0002",
    ]
    assert len(state["records"]) == 4
    assert [record["status"] for record in state["records"][:2]] == ["operational_failure"] * 2
    assert first_calls["plain"] == 3
    assert first_calls["lf"] == 3

    resumed_calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    rc = runner.execute(_args(repo, exact, tmp_path / "resumed", store_root))
    receipt, result, _ = _payloads(tmp_path / "resumed", run_id)

    assert rc == receipt["rc"] == 1
    assert resumed_calls["plain"] == 6
    assert resumed_calls["lf"] == 6
    assert len(result["records"]) == 16
    assert result["records"][:2] == state["records"][:2]
    evidence = result["clean_confirmation_evidence"]
    assert evidence["evaluation_status"] == "not_evaluable_operational"
    assert evidence["confirmation_pass_candidate_id"] is None
    assert evidence["confirmation_outcome"] is None
    assert all(
        facts["gate_a_pass"] is None and facts["gate_b_pass"] is None and facts["eligible"] is None
        for facts in evidence["candidate_evidence"].values()
    )


def _record(
    expected: dict[str, object],
    unit_id: str,
    arm: str,
    registered: float,
    wrong: float,
    *,
    psnr: float = 40.0,
) -> StageARecord:
    metrics = {"paired_rgb_psnr": psnr}
    if not arm.startswith("primary_null__"):
        metrics["actual_dtype_relative_l2"] = 0.012
    return StageARecord(
        run_id=str(expected["run_id"]),
        unit_id=unit_id,
        source_cluster_id=f"source-{unit_id}",
        arm=arm,
        condition="identity",
        code_revision=str(expected["resolved_exact"]),
        config_digest=str(expected["protocol_digest"]),
        key_public_digest=str(expected["key_public_digest"]),
        status="success",
        scores={"registered": registered, **{f"wrong_{index:02d}": wrong for index in range(16)}},
        metrics=metrics,
    )


def _expected() -> dict[str, object]:
    return {
        "run_id": "lfbbconf-test",
        "execution_scope_id": runner.EXECUTION_SCOPE_ID,
        "resolved_exact": "1" * 40,
        "protocol_digest": "2" * 64,
        "key_public_digest": "3" * 64,
        "ordered_roster_unit_ids": [
            f"lfbb-confirmation-{index:04d}" for index in range(1, 9)
        ],
        "carrier_method_id": runner.LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        "evaluated_candidate_id": runner.LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
        "detector_statistic_id": runner.LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        "rank_gate_a_min_units": 7,
        "rank_gate_b_min_units": 7,
    }


def _confirmation_records(
    expected: dict[str, object],
    *,
    gate_a: int = 8,
    gate_b: int = 8,
    margin: float = 0.1,
    psnr: float = 40.0,
) -> list[StageARecord]:
    records: list[StageARecord] = []
    for index, unit_id in enumerate(expected["ordered_roster_unit_ids"]):
        registered = margin if index < gate_a else 0.0
        null = registered - 0.05 if index < gate_b else registered
        records.extend([
            _record(expected, unit_id, runner.RECORD_ARMS[0], registered, 0.0, psnr=psnr),
            _record(expected, unit_id, runner.RECORD_ARMS[1], null, -0.2, psnr=psnr),
        ])
    return records


@pytest.mark.integration
def test_scale_free_gates_use_strict_ties_and_complete_denominator() -> None:
    expected = _expected()
    failure = runner._clean_confirmation_evidence(
        _confirmation_records(expected, gate_a=6),
        expected,
        confirmation_outcome_allowed=True,
    )
    assert failure["confirmation_outcome"] == "SCIENTIFIC_NEGATIVE_AND_STOP"
    assert failure["confirmation_pass_candidate_id"] is None

    passed = runner._clean_confirmation_evidence(
        _confirmation_records(expected, gate_a=7, gate_b=7),
        expected,
        confirmation_outcome_allowed=True,
    )
    assert passed["confirmation_pass_candidate_id"] == runner.LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID
    facts = passed["candidate_evidence"][runner.LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID]
    assert facts["gate_a_pass"] is True and facts["gate_b_pass"] is True
    assert "winner_ranking_order" not in passed

    tie_failure = runner._clean_confirmation_evidence(
        _confirmation_records(expected, gate_a=6, gate_b=6),
        expected,
        confirmation_outcome_allowed=True,
    )
    assert tie_failure["confirmation_outcome"] == "SCIENTIFIC_NEGATIVE_AND_STOP"
    partial = runner._clean_confirmation_evidence(
        _confirmation_records(expected)[:-2],
        expected,
        confirmation_outcome_allowed=False,
    )
    assert partial["evaluation_status"] == "not_evaluable_operational"
    assert partial["confirmation_outcome"] is None
    assert partial["confirmation_pass_candidate_id"] is None


@pytest.mark.integration
def test_resume_rejects_four_arm_order_or_identity_drift(tmp_path: Path) -> None:
    expected = {
        **_expected(),
        "record_arms_in_exact_unit_order": list(runner.RECORD_ARMS),
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "checkpoint_interval_hours": 2.0,
    }
    transaction = [record.to_dict() for record in _confirmation_records(expected)[:2]]
    transaction[0]["arm"], transaction[1]["arm"] = transaction[1]["arm"], transaction[0]["arm"]
    state = {
        **expected,
        "checkpoint_sequence": 1,
        "committed_unit_count": 1,
        "committed_unit_ids": ["lfbb-confirmation-0001"],
        "records": transaction,
    }
    zip_path = tmp_path / "checkpoint-0001-units-0001.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("state.json", json.dumps(state))
    sha_path = tmp_path / f"{zip_path.name}.sha256"
    sha_path.write_text(
        f"{hashlib.sha256(zip_path.read_bytes()).hexdigest()}  {zip_path.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="paired transaction"):
        runner._resume_state(zip_path, sha_path, expected)

    correct_transaction = [
        record.to_dict() for record in _confirmation_records(expected)[:2]
    ]
    wrong_scope_state = {
        **state,
        "execution_scope_id": "lf_balanced_blocks_selection_v1",
        "records": correct_transaction,
    }
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("state.json", json.dumps(wrong_scope_state))
    sha_path.write_text(
        f"{hashlib.sha256(zip_path.read_bytes()).hexdigest()}  {zip_path.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        runner._resume_state(zip_path, sha_path, expected)


@pytest.mark.integration
def test_nonfinite_scores_become_retained_operational_failures_without_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, exact = _repo(tmp_path)
    output_root = tmp_path / "output"
    _install_fakes(monkeypatch, nonfinite_scores=True)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)

    rc = runner.execute(_args(repo, exact, output_root, tmp_path / "store"))
    receipt, result, _ = _payloads(output_root, _only_run_id(output_root))

    assert rc == receipt["rc"] == result["rc"] == 1
    assert len(result["records"]) == 16
    assert all(record["status"] == "operational_failure" for record in result["records"])
    evidence = result["clean_confirmation_evidence"]
    assert evidence["evaluation_status"] == "not_evaluable_operational"
    assert evidence["confirmation_outcome"] is None
    assert evidence["confirmation_pass_candidate_id"] is None
    facts = evidence["candidate_evidence"][runner.LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID]
    assert facts["gate_a_pass"] is None
    assert facts["gate_b_pass"] is None
    assert facts["eligible"] is None


@pytest.mark.integration
def test_terminal_publication_is_create_only_without_readback_or_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_zip = tmp_path / "result.zip"
    source_sha = tmp_path / "result.zip.sha256"
    source_zip.write_bytes(b"zip-bytes")
    source_sha.write_text("not-revalidated  result.zip\n", encoding="utf-8")
    sink = tmp_path / "sink"
    sink.mkdir()

    runner._publish_terminal_pair_create_only(source_zip, source_sha, sink, artifact_kind="final")
    assert (sink / source_zip.name).read_bytes() == b"zip-bytes"
    assert (sink / source_sha.name).is_file()
    with pytest.raises(RuntimeError, match="refuses overwrite"):
        runner._publish_terminal_pair_create_only(source_zip, source_sha, sink, artifact_kind="final")

    partial_sink = tmp_path / "partial"
    partial_sink.mkdir()
    original_copy = shutil.copyfileobj
    calls = 0

    def fail_second(source: object, target: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated Drive copy failure")
        original_copy(source, target)

    monkeypatch.setattr(runner.shutil, "copyfileobj", fail_second)
    with pytest.raises(OSError, match="Drive copy"):
        runner._publish_terminal_pair_create_only(source_zip, source_sha, partial_sink, artifact_kind="final")
    assert (partial_sink / source_zip.name).is_file()
    assert (partial_sink / source_sha.name).exists()


@pytest.mark.integration
def test_missing_secrets_fail_closed_without_leaking_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch)
    monkeypatch.delenv(runner.KEY_ENV, raising=False)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    with pytest.raises(RuntimeError, match="root_key_environment_input_required"):
        runner.execute(_args(repo, exact, tmp_path / "missing-key", tmp_path / "store-a"))

    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, "")
    context: dict[str, object] = {}
    with pytest.raises(RuntimeError, match="hugging_face_token_environment_input_required"):
        runner.execute(_args(repo, exact, tmp_path / "missing-token", tmp_path / "store-b"), fatal_context=context)
    serialized_context = repr(context)
    assert _RAW_KEY not in serialized_context
    assert _HF_TOKEN not in serialized_context
    fatal_zip, _, published = runner._export_fatal(
        _args(repo, exact, tmp_path / "missing-token", tmp_path / "store-b"),
        context,
        "initialization_failure",
    )
    assert published is True
    with zipfile.ZipFile(fatal_zip) as archive:
        payload = b"".join(archive.read(name) for name in archive.namelist())
        result = json.loads(archive.read("result.json"))
    assert _RAW_KEY.encode() not in payload and _HF_TOKEN.encode() not in payload
    assert result["execution_scope_id"] == runner.EXECUTION_SCOPE_ID
    evidence = result["clean_confirmation_evidence"]
    assert evidence["confirmation_outcome_allowed"] is False
    assert evidence["confirmation_outcome"] is None
    assert evidence["confirmation_pass_candidate_id"] is None
    assert result["records"] == []


@pytest.mark.integration
def test_run_identity_is_deterministic_and_cli_has_no_mode_or_interval() -> None:
    protocol = runner._load_protocol(_ROOT)
    first = runner._deterministic_run_id("a" * 40, protocol, "stabilityai/stable-diffusion-3.5-medium", "b" * 64)
    second = runner._deterministic_run_id("a" * 40, protocol, "stabilityai/stable-diffusion-3.5-medium", "b" * 64)
    assert first == second and first.startswith("lfbbconf-")
    actions = runner._parser()._actions
    option_strings = {option for action in actions for option in action.option_strings}
    assert "--run-mode" not in option_strings
    assert "--checkpoint-interval" not in option_strings
    assert runner.CHECKPOINT_INTERVAL_HOURS == 2.0
