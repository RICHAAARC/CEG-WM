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
from PIL import Image
import pytest

from cegwm.protocol.records import StageARecord
from cegwm.shared.keys import normalize_detection_key
from experiments.stage_a import run_hf_a2_colab as runner

_ROOT = Path(__file__).resolve().parents[2]
_RAW_KEY = "stage-a-colab-detection-key-attack-comparison"
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


def _args(repo: Path, exact: str, output_root: Path, store_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(repo),
        output_root=str(output_root),
        expected_exact=exact,
        run_store_root=str(store_root),
    )


def _pattern(seed: int, offset: int) -> Image.Image:
    yy, xx = np.mgrid[:32, :32]
    seed_term = seed % 13
    pixels = np.stack(
        (
            (xx * 3 + yy * 5 + seed_term) % 120 + 20 + offset,
            (xx * 7 + yy * 2 + seed_term) % 120 + 20 + offset,
            (xx + yy * 11 + seed_term) % 120 + 20 + offset,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_hf_calls: frozenset[int] = frozenset(),
    interrupt_hf_call: int | None = None,
) -> dict[str, object]:
    calls: dict[str, object] = {
        "load": 0, "hf": 0, "lf": 0, "plain": 0, "score": 0,
        "seeds": [], "score_images": [],
    }
    registered_key = normalize_detection_key(_RAW_KEY)
    hf_assets = SimpleNamespace(method="hf")
    lf_assets = SimpleNamespace(method="lf")

    def fake_load(model_id: str, hf_token: str) -> tuple[object, object, object]:
        assert model_id == "stabilityai/stable-diffusion-3.5-medium"
        assert hf_token == _HF_TOKEN
        calls["load"] = int(calls["load"]) + 1
        return object(), hf_assets, lf_assets

    def fake_hf(
        pipeline: object, prompt: str, key: bytes, public_assets: object, **kwargs: object
    ) -> SimpleNamespace:
        del pipeline, prompt, key
        calls["hf"] = int(calls["hf"]) + 1
        if int(calls["hf"]) == interrupt_hf_call:
            raise KeyboardInterrupt
        if int(calls["hf"]) in fail_hf_calls:
            raise RuntimeError("private failure detail")
        seed = kwargs["generator"].seed
        calls["seeds"].append(("hf", seed))
        assert public_assets is hf_assets
        return SimpleNamespace(
            image=_pattern(seed, 18),
            injection_budget=SimpleNamespace(relative_l2=0.011998),
        )

    def fake_lf(
        pipeline: object, prompt: str, key: bytes, public_assets: object, **kwargs: object
    ) -> SimpleNamespace:
        del pipeline, prompt, key
        calls["lf"] = int(calls["lf"]) + 1
        seed = kwargs["generator"].seed
        calls["seeds"].append(("lf", seed))
        assert public_assets is lf_assets
        return SimpleNamespace(
            image=_pattern(seed, 28),
            injection_budget=SimpleNamespace(relative_l2=0.011999),
        )

    def fake_plain(pipeline: object, prompt: str, **kwargs: object) -> Image.Image:
        del pipeline, prompt
        calls["plain"] = int(calls["plain"]) + 1
        seed = kwargs["generator"].seed
        calls["seeds"].append(("plain", seed))
        return _pattern(seed, 0)

    def fake_scores(
        image: Image.Image,
        detection_key: bytes,
        wrong_keys: tuple[bytes, ...],
        assets: object,
    ) -> dict[str, float]:
        calls["score"] = int(calls["score"]) + 1
        calls["score_images"].append((assets.method, id(image)))
        mean = float(np.asarray(image, dtype=np.float64).mean() / 255.0)
        values = {"registered": mean + (0.5 if detection_key == registered_key else 0.0)}
        values.update({
            f"wrong_{index:02d}": mean + wrong_key[0] / 8192.0
            for index, wrong_key in enumerate(wrong_keys)
        })
        return values

    monkeypatch.setattr(runner, "_load_pipeline_and_assets", fake_load)
    monkeypatch.setattr(runner.torch, "Generator", _Generator)
    monkeypatch.setattr(runner, "run_sd35_hf", fake_hf)
    monkeypatch.setattr(runner, "run_sd35_lf", fake_lf)
    monkeypatch.setattr(runner, "run_sd35_plain", fake_plain)
    monkeypatch.setattr(runner, "_scores", fake_scores)
    return calls


def _payloads(output_root: Path, run_id: str) -> tuple[dict[str, object], dict[str, object], Path]:
    local = output_root / run_id
    return (
        json.loads((local / "receipt.json").read_text(encoding="utf-8")),
        json.loads((local / "result.json").read_text(encoding="utf-8")),
        local / f"{run_id}.zip",
    )


def _only_run_id(root: Path) -> str:
    names = [path.name for path in root.iterdir() if path.is_dir()]
    assert len(names) == 1
    return names[0]


@pytest.mark.integration
def test_runner_executes_three_real_paths_fixed_128_records_and_safe_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, exact = _repo(tmp_path)
    output_root, store_root = tmp_path / "output", tmp_path / "store"
    calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    rc = runner.execute(_args(repo, exact, output_root, store_root))
    run_id = _only_run_id(output_root)
    receipt, result, local_zip = _payloads(output_root, run_id)

    assert rc == receipt["rc"] == result["rc"] == 0
    assert result["execution_scope_id"] == runner.EXECUTION_SCOPE_ID
    assert result["completeness"] == runner.COMPLETE_EXECUTION
    assert runner.COMPLETE_EXECUTION == (
        "complete_for_hf_lf_paired_clean_reference_and_"
        "attack_complementarity_execution"
    )
    assert result["condition_order"] == list(runner.CONDITION_ORDER)
    assert result["attack_ids"] == list(runner.ATTACK_IDS)
    assert len(result["records"]) == 128
    assert calls["load"] == 1 and calls["hf"] == calls["lf"] == calls["plain"] == 8
    assert calls["score"] == 128
    assert Counter(record["unit_id"] for record in result["records"]) == {
        f"attack-comp-{index:04d}": 16 for index in range(1, 9)
    }
    expected_pairs = [
        (condition, arm)
        for condition in runner.CONDITION_ORDER
        for arm in runner.RECORD_ARMS
    ]
    for index in range(8):
        transaction = result["records"][index * 16 : index * 16 + 16]
        assert [(record["condition"], record["arm"]) for record in transaction] == expected_pairs
    seeds = calls["seeds"]
    for index in range(8):
        triple = seeds[index * 3 : index * 3 + 3]
        assert [name for name, _ in triple] == ["hf", "lf", "plain"]
        assert len({seed for _, seed in triple}) == 1
    score_images = calls["score_images"]
    for group_index in range(8 * 4):
        group = score_images[group_index * 4 : group_index * 4 + 4]
        assert group[0][0] == "hf" and group[1][0] == "lf"
        assert group[0][1] == group[1][1]
    evidence = result["attack_complementarity_evidence"]
    assert set(evidence) == {
        "scientific_outcome_allowed",
        "evaluation_status",
        "fixed_unit_count",
        "fixed_condition_count",
        "fixed_attack_count",
        "fixed_record_count",
        "unit_transaction_record_count",
        "paired_clean_prerequisite",
        "attack_conditions",
        "attack_complementarity_pass",
        "complementary_attack_ids",
        "attack_complementarity_outcome",
        "median_margin_is_gate",
        "primary_null_cutoff_is_gate",
        "score_retention_ratio_is_gate",
        "cross_detector_raw_score_comparison",
        "formal_fpr_claim",
    }
    assert evidence["scientific_outcome_allowed"] is True
    assert evidence["paired_clean_prerequisite"]["both_methods_pass"] is True
    assert evidence["fixed_record_count"] == 128
    assert evidence["paired_clean_prerequisite"]["hf"][
        "median_attacked_vs_pre_attack_psnr"
    ] is None
    assert set(evidence["attack_conditions"]) == set(runner.ATTACK_IDS)
    assert "identity_reference" not in evidence["attack_conditions"]
    assert set(evidence["paired_clean_prerequisite"]) == {
        "condition_id",
        "identity_reference_is_attack",
        "hf",
        "lf",
        "both_methods_pass",
        "unit_evidence",
    }
    assert evidence["paired_clean_prerequisite"]["condition_id"] == "identity_reference"
    assert evidence["paired_clean_prerequisite"]["identity_reference_is_attack"] is False
    assert all(
        set(evidence["attack_conditions"][attack_id])
        == {"hf", "lf", "complementarity_condition", "unit_evidence"}
        for attack_id in runner.ATTACK_IDS
    )
    assert all(len(record["scores"]) == 17 for record in result["records"])
    assert runner.KEY_ENV not in os.environ and runner.TOKEN_ENV not in os.environ
    stored_zip = store_root / run_id / f"{run_id}.zip"
    stored_sha = store_root / run_id / f"{run_id}.zip.sha256"
    assert local_zip.is_file() and stored_zip.is_file() and stored_sha.is_file()
    with zipfile.ZipFile(stored_zip) as archive:
        assert archive.namelist() == ["receipt.json", "result.json"]
        exported = b"".join(archive.read(name) for name in archive.namelist())
    assert _RAW_KEY.encode() not in exported and _HF_TOKEN.encode() not in exported
    assert b"private_latent" not in exported and b"traceback" not in exported


def _expected() -> dict[str, object]:
    return {
        "run_id": "hlfac-test",
        "resolved_exact": "1" * 40,
        "protocol_digest": "2" * 64,
        "key_public_digest": "3" * 64,
        "ordered_roster_unit_ids": [f"attack-comp-{index:04d}" for index in range(1, 9)],
        "ordered_roster_source_ids": [
            f"source-attack-comp-{index:04d}" for index in range(1, 9)
        ],
        "method_identities": {
            "hf": {"evaluated_candidate_id": runner.HF_EVALUATED_CANDIDATE_ID},
            "lf": {"evaluated_candidate_id": runner.LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID},
        },
        "rank_gate_a_min_units": 7,
        "rank_gate_b_min_units": 7,
    }


def _record(
    expected: dict[str, object], unit_id: str, condition: str, arm: str,
    registered: float, wrong: float,
) -> StageARecord:
    metrics = {"paired_rgb_psnr": 40.0}
    if not arm.startswith("primary_null__"):
        metrics["actual_dtype_relative_l2"] = 0.012
        if condition in runner.ATTACK_IDS:
            metrics["attacked_vs_pre_attack_psnr"] = 30.0
    return StageARecord(
        run_id=str(expected["run_id"]), unit_id=unit_id,
        source_cluster_id=f"source-{unit_id}", arm=arm, condition=condition,
        code_revision=str(expected["resolved_exact"]),
        config_digest=str(expected["protocol_digest"]),
        key_public_digest=str(expected["key_public_digest"]), status="success",
        scores={"registered": registered, **{f"wrong_{index:02d}": wrong for index in range(16)}},
        metrics=metrics,
    )


def _records(
    expected: dict[str, object],
    gate_counts: dict[tuple[str, str], tuple[int, int]],
) -> list[StageARecord]:
    records: list[StageARecord] = []
    for index, unit_id in enumerate(expected["ordered_roster_unit_ids"]):
        for condition in runner.CONDITION_ORDER:
            for method, candidate_arm, null_arm in (
                ("hf", runner.RECORD_ARMS[0], runner.RECORD_ARMS[1]),
                ("lf", runner.RECORD_ARMS[2], runner.RECORD_ARMS[3]),
            ):
                gate_a, gate_b = gate_counts.get((condition, method), (8, 8))
                registered = 0.2 if index < gate_a else 0.0
                null = registered - 0.1 if index < gate_b else registered
                records.extend([
                    _record(expected, unit_id, condition, candidate_arm, registered, 0.0),
                    _record(expected, unit_id, condition, null_arm, null, -0.2),
                ])
    return records


@pytest.mark.integration
def test_clean_prerequisite_failure_nulls_all_attack_decisions() -> None:
    expected = _expected()
    evidence = runner._attack_complementarity_evidence(
        _records(expected, {("identity_reference", "hf"): (6, 8)}),
        expected,
        scientific_outcome_allowed=True,
    )
    assert evidence["evaluation_status"] == "paired_clean_prerequisite_failed"
    assert evidence["attack_complementarity_outcome"] == (
        "SCIENTIFIC_NEGATIVE_FOR_PAIRED_CLEAN_PREREQUISITE_"
        "ATTACK_COMPLEMENTARITY_NOT_EVALUABLE_AND_STOP"
    )
    assert evidence["paired_clean_prerequisite"]["both_methods_pass"] is False
    assert evidence["attack_complementarity_pass"] is None
    assert evidence["complementary_attack_ids"] is None
    for attack_id in runner.ATTACK_IDS:
        for method in ("hf", "lf"):
            facts = evidence["attack_conditions"][attack_id][method]
            assert facts["gate_a_pass"] is None
            assert facts["gate_b_pass"] is None
            assert facts["method_survives_attack"] is None
        assert evidence["attack_conditions"][attack_id]["complementarity_condition"] is None


@pytest.mark.integration
def test_complementarity_pass_no_complement_and_strict_tie_cases() -> None:
    expected = _expected()
    complement = runner._attack_complementarity_evidence(
        _records(expected, {("jpeg_q75", "hf"): (6, 8)}),
        expected,
        scientific_outcome_allowed=True,
    )
    assert complement["attack_complementarity_pass"] is True
    assert complement["complementary_attack_ids"] == ["jpeg_q75"]
    assert complement["attack_complementarity_outcome"] == "attack_complementarity_pass_candidate_for_agent5_adjudication"
    assert "identity_reference" not in complement["complementary_attack_ids"]
    assert complement["attack_conditions"]["jpeg_q75"]["hf"][
        "method_survives_attack"
    ] is False
    assert complement["attack_conditions"]["jpeg_q75"]["lf"][
        "method_survives_attack"
    ] is True
    assert complement["attack_conditions"]["jpeg_q75"][
        "complementarity_condition"
    ] is True
    negative = runner._attack_complementarity_evidence(
        _records(expected, {}), expected, scientific_outcome_allowed=True
    )
    assert negative["attack_complementarity_pass"] is False
    assert negative["complementary_attack_ids"] == []
    assert negative["attack_complementarity_outcome"] == "SCIENTIFIC_NEGATIVE_FOR_COMPLEMENTARITY_AND_STOP"
    tie = runner._attack_complementarity_evidence(
        _records(expected, {("identity_reference", "lf"): (6, 6)}),
        expected,
        scientific_outcome_allowed=True,
    )
    assert tie["paired_clean_prerequisite"]["both_methods_pass"] is False


@pytest.mark.integration
def test_non_rc0_or_partial_records_have_no_scientific_booleans() -> None:
    expected = _expected()
    full = _records(expected, {("jpeg_q75", "hf"): (6, 8)})
    for records in (full, full[:-16]):
        evidence = runner._attack_complementarity_evidence(
            records, expected, scientific_outcome_allowed=False
        )
        assert evidence["evaluation_status"] == "not_evaluable_operational"
        assert evidence["attack_complementarity_outcome"] is None
        assert evidence["paired_clean_prerequisite"]["both_methods_pass"] is None
        assert evidence["attack_complementarity_pass"] is None
        assert evidence["complementary_attack_ids"] is None
        for method in ("hf", "lf"):
            clean = evidence["paired_clean_prerequisite"][method]
            assert clean["gate_a_pass"] is None and clean["gate_b_pass"] is None
        for attack_id in runner.ATTACK_IDS:
            attack = evidence["attack_conditions"][attack_id]
            assert attack["complementarity_condition"] is None
            for method in ("hf", "lf"):
                facts = attack[method]
                assert facts["gate_a_pass"] is None
                assert facts["gate_b_pass"] is None
                assert facts["method_survives_attack"] is None


@pytest.mark.integration
def test_resume_binds_complete_16_record_transactions_and_rejects_drift(tmp_path: Path) -> None:
    expected = {
        **_expected(),
        "execution_scope_id": runner.EXECUTION_SCOPE_ID,
        "condition_order": list(runner.CONDITION_ORDER),
        "attack_ids": list(runner.ATTACK_IDS),
        "record_arms_in_exact_condition_order": list(runner.RECORD_ARMS),
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "checkpoint_interval_hours": 2.0,
    }
    transaction = [record.to_dict() for record in _records(expected, {})[:16]]
    state = {**expected, "checkpoint_sequence": 1, "committed_unit_count": 1,
             "committed_unit_ids": ["attack-comp-0001"], "records": transaction}
    zip_path = tmp_path / "checkpoint-0001-units-0001.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("state.json", json.dumps(state))
    sha_path = tmp_path / f"{zip_path.name}.sha256"
    sha_path.write_text(f"{hashlib.sha256(zip_path.read_bytes()).hexdigest()}  {zip_path.name}\n", encoding="utf-8")
    assert runner._resume_state(zip_path, sha_path, expected)["committed_unit_count"] == 1
    transaction[0]["condition"] = "jpeg_q75"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("state.json", json.dumps({**state, "records": transaction}))
    sha_path.write_text(f"{hashlib.sha256(zip_path.read_bytes()).hexdigest()}  {zip_path.name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="16-record transaction"):
        runner._resume_state(zip_path, sha_path, expected)


@pytest.mark.integration
def test_checkpoint_resume_skips_committed_unit_and_reruns_interrupted_whole_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, exact = _repo(tmp_path)
    store = tmp_path / "store"
    first_calls = _install_fakes(monkeypatch, interrupt_hf_call=2)
    clock = iter([0.0, 7201.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(repo, exact, tmp_path / "first", store))
    run_id = _only_run_id(tmp_path / "first")
    checkpoint_zip = next((store / run_id).glob("checkpoint-*.zip"))
    with zipfile.ZipFile(checkpoint_zip) as archive:
        state = json.loads(archive.read("state.json"))
    assert state["committed_unit_ids"] == ["attack-comp-0001"]
    assert len(state["records"]) == 16
    assert first_calls["hf"] == 2 and first_calls["lf"] == first_calls["plain"] == 1

    resumed_calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    rc = runner.execute(_args(repo, exact, tmp_path / "resumed", store))
    _, result, _ = _payloads(tmp_path / "resumed", run_id)
    assert rc == 0 and len(result["records"]) == 128
    assert resumed_calls["hf"] == resumed_calls["lf"] == resumed_calls["plain"] == 7


@pytest.mark.integration
def test_operational_failure_retains_full_unit_and_nulls_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch, fail_hf_calls=frozenset({1}))
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    output = tmp_path / "output"
    rc = runner.execute(_args(repo, exact, output, tmp_path / "store"))
    _, result, _ = _payloads(output, _only_run_id(output))
    assert rc == 1 and len(result["records"]) == 128
    assert all(record["status"] == "operational_failure" for record in result["records"][:16])
    evidence = result["attack_complementarity_evidence"]
    assert evidence["evaluation_status"] == "not_evaluable_operational"
    assert evidence["attack_complementarity_outcome"] is None


@pytest.mark.integration
def test_checkpoint_publication_failure_makes_complete_records_rc1_not_evaluable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch)
    clock = iter([0.0, *([7201.0] * 8)])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        runner,
        "_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("Drive checkpoint failure")),
    )
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _HF_TOKEN)
    output = tmp_path / "output"
    rc = runner.execute(_args(repo, exact, output, tmp_path / "store"))
    _, result, _ = _payloads(output, _only_run_id(output))
    assert rc == 1 and len(result["records"]) == 128
    assert all(record["status"] == "success" for record in result["records"])
    evidence = result["attack_complementarity_evidence"]
    assert evidence["evaluation_status"] == "not_evaluable_operational"
    assert evidence["attack_complementarity_outcome"] is None
    assert evidence["paired_clean_prerequisite"]["both_methods_pass"] is None
    assert evidence["attack_complementarity_pass"] is None


@pytest.mark.integration
def test_missing_token_fatal_package_has_no_secret_or_scientific_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch)
    monkeypatch.setenv(runner.KEY_ENV, _RAW_KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, "")
    context: dict[str, object] = {}
    args = _args(repo, exact, tmp_path / "output", tmp_path / "store")
    with pytest.raises(RuntimeError, match="hugging_face_token"):
        runner.execute(args, fatal_context=context)
    fatal_zip, _, published = runner._export_fatal(args, context, "initialization_failure")
    assert published is True
    with zipfile.ZipFile(fatal_zip) as archive:
        payload = b"".join(archive.read(name) for name in archive.namelist())
        result = json.loads(archive.read("result.json"))
    assert _RAW_KEY.encode() not in payload and _HF_TOKEN.encode() not in payload
    evidence = result["attack_complementarity_evidence"]
    assert evidence["scientific_outcome_allowed"] is False
    assert evidence["attack_complementarity_outcome"] is None


@pytest.mark.integration
def test_run_identity_is_deterministic_and_cli_has_no_mode_or_interval() -> None:
    protocol = runner._load_protocol(_ROOT)
    first = runner._deterministic_run_id("a" * 40, protocol, "stabilityai/stable-diffusion-3.5-medium", "b" * 64)
    second = runner._deterministic_run_id("a" * 40, protocol, "stabilityai/stable-diffusion-3.5-medium", "b" * 64)
    assert first == second and first.startswith("hlfac-")
    options = {option for action in runner._parser()._actions for option in action.option_strings}
    assert "--run-mode" not in options and "--checkpoint-interval" not in options
    assert runner.CHECKPOINT_INTERVAL_HOURS == 2.0
