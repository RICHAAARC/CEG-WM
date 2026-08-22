from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest

from cegwm.shared.keys import normalize_detection_key, public_key_digest
from experiments.stage_a_frequency_response import run_colab as runner
from experiments.stage_a_frequency_response.protocol import load_plan

_ROOT = Path(__file__).resolve().parents[2]
_KEY = "frequency-response-integration-detection-key"
_TOKEN = "hf_frequency_response_test_token"


class _Generator:
    def __init__(self, device: str) -> None:
        assert device == "cuda"
        self.seed = 0

    def manual_seed(self, seed: int) -> _Generator:
        self.seed = seed
        return self


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    shutil.copytree(_ROOT / "configs", repo / "configs")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Frequency Response Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "configs"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "fixture"], check=True)
    exact = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return repo, exact


def _pattern(seed: int, offset: int) -> Image.Image:
    yy, xx = np.mgrid[:24, :24]
    pixels = np.stack((
        (xx * 3 + yy * 5 + seed) % 100 + 20 + offset,
        (xx * 7 + yy * 2 + seed) % 100 + 20 + offset,
        (xx + yy * 11 + seed) % 100 + 20 + offset,
    ), axis=-1).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch, *, fail_hf_call: int | None = None,
    interrupt_hf_call: int | None = None,
) -> dict[str, object]:
    calls: dict[str, object] = {"load": 0, "hf": 0, "lf": 0, "plain": 0, "seeds": [], "scores": 0}
    registered = normalize_detection_key(_KEY)
    hf_assets, lf_assets = SimpleNamespace(method="hf"), SimpleNamespace(method="lf")

    def load(
        model_id: str, token: str, lf_method_identity: dict[str, str],
    ) -> tuple[object, object, object]:
        calls["load"] = int(calls["load"]) + 1
        assert model_id == "stabilityai/stable-diffusion-3.5-medium" and token == _TOKEN
        assert lf_method_identity == {
            "carrier_method_id": "lf_shell_balanced_blocks_v2",
            "detector_statistic_id": "lf_block_centered_normalized_median_corr_v2",
            "evaluated_candidate_id": "lf_shell_balanced_blocks_v2_blocknorm_median_v1",
        }
        return object(), hf_assets, lf_assets

    def hf(_: object, __: str, ___: bytes, assets: object, **kwargs: object) -> SimpleNamespace:
        calls["hf"] = int(calls["hf"]) + 1
        if calls["hf"] == interrupt_hf_call:
            raise KeyboardInterrupt
        if calls["hf"] == fail_hf_call:
            raise RuntimeError("sensitive runtime detail")
        assert assets is hf_assets
        seed = kwargs["generator"].seed
        calls["seeds"].append(("hf", seed))
        return SimpleNamespace(image=_pattern(seed, 14), injection_budget=SimpleNamespace(relative_l2=0.01199))

    def lf(_: object, __: str, ___: bytes, assets: object, **kwargs: object) -> SimpleNamespace:
        calls["lf"] = int(calls["lf"]) + 1
        assert assets is lf_assets
        seed = kwargs["generator"].seed
        calls["seeds"].append(("lf", seed))
        return SimpleNamespace(image=_pattern(seed, 24), injection_budget=SimpleNamespace(relative_l2=0.01198))

    def plain(_: object, __: str, **kwargs: object) -> Image.Image:
        calls["plain"] = int(calls["plain"]) + 1
        seed = kwargs["generator"].seed
        calls["seeds"].append(("plain", seed))
        return _pattern(seed, 0)

    def scores(image: Image.Image, key: bytes, wrong_keys: tuple[bytes, ...], assets: object) -> dict[str, float]:
        calls["scores"] = int(calls["scores"]) + 1
        mean = float(np.asarray(image, dtype=np.float64).mean() / 255.0)
        return {
            "registered": mean + (0.5 if key == registered else 0.0),
            **{f"wrong_{index:02d}": mean + wrong[0] / 8192.0 for index, wrong in enumerate(wrong_keys)},
        }

    monkeypatch.setattr(runner, "_load_pipeline_and_assets", load)
    monkeypatch.setattr(runner.torch, "Generator", _Generator)
    monkeypatch.setattr(runner, "run_sd35_hf", hf)
    monkeypatch.setattr(runner, "run_sd35_lf", lf)
    monkeypatch.setattr(runner, "run_sd35_plain", plain)
    monkeypatch.setattr(runner, "_scores", scores)
    return calls


def _args(repo: Path, exact: str, output_root: Path, store: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(repo), expected_exact=exact,
        output_root=str(output_root), run_store_root=str(store),
    )


def _run_id(root: Path) -> str:
    children = [path.name for path in root.iterdir() if path.is_dir()]
    assert len(children) == 1
    return children[0]


def _terminal_payload(store: Path, run_id: str) -> tuple[dict[str, object], bytes]:
    zip_path = store / run_id / f"{run_id}.zip"
    with zipfile.ZipFile(zip_path) as archive:
        raw = b"".join(archive.read(name) for name in archive.namelist())
        return json.loads(archive.read("result.json")), raw


def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runner.KEY_ENV, _KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _TOKEN)


def _expected(repo: Path, exact: str) -> dict[str, object]:
    plan = load_plan(
        repo / "configs/stage_a_frequency_response/standalone_lf_hf_frequency_response_v1.json",
        repo / "configs/stage_a_frequency_response/standalone_lf_hf_frequency_response_v1.jsonl",
    )
    return runner._new_state(exact, plan, public_key_digest(normalize_detection_key(_KEY)))


@pytest.mark.integration
def test_real_lf_assets_constructor_receives_exact_protocol_identity_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = load_plan(
        _ROOT / "configs/stage_a_frequency_response/standalone_lf_hf_frequency_response_v1.json",
        _ROOT / "configs/stage_a_frequency_response/standalone_lf_hf_frequency_response_v1.jsonl",
    )

    class VAE:
        def encode(self, _: object) -> None:
            return None

    class Processor:
        def preprocess(self, _: object) -> None:
            return None

    moved: list[str] = []
    pipeline = SimpleNamespace(
        vae=VAE(), image_processor=Processor(), to=lambda device: moved.append(device),
    )
    loads: list[tuple[str, object, str]] = []

    def load(model_id: str, *, torch_dtype: object, token: str) -> object:
        loads.append((model_id, torch_dtype, token))
        return pipeline

    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner, "load_sd35_pipeline", load)
    loaded, _, lf_assets = runner._load_pipeline_and_assets(
        plan.model_id, _TOKEN, plan.method_identities["lf"],
    )
    assert loaded is pipeline and moved == ["cuda"]
    assert loads == [(plan.model_id, runner.torch.float16, _TOKEN)]
    assert {
        "carrier_method_id": lf_assets.candidate_id,
        "detector_statistic_id": lf_assets.detector_statistic_id,
        "evaluated_candidate_id": lf_assets.evaluated_candidate_id,
    } == plan.method_identities["lf"]

    missing = dict(plan.method_identities["lf"])
    missing.pop("evaluated_candidate_id")
    with pytest.raises(ValueError, match="identity fields"):
        runner._load_pipeline_and_assets(plan.model_id, _TOKEN, missing)
    mismatched = dict(plan.method_identities["lf"])
    mismatched["evaluated_candidate_id"] = "lf_shell_rademacher_v1_blocknorm_median_v2"
    with pytest.raises(ValueError, match="evaluated identity"):
        runner._load_pipeline_and_assets(plan.model_id, _TOKEN, mismatched)


@pytest.mark.integration
def test_fresh_success_exports_complete_fixed_320_descriptive_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    _env(monkeypatch)
    output, store = tmp_path / "output", tmp_path / "store"
    assert runner.execute(_args(repo, exact, output, store)) == 0
    run_id = _run_id(output)
    result, _ = _terminal_payload(store, run_id)
    assert result["evidence_contract"] == "STANDALONE_LF_HF_FREQUENCY_RESPONSE_EVIDENCE"
    assert result["complete"] is True and result["rc"] == 0
    assert result["committed_unit_count"] == 8 and len(result["records"]) == 320
    assert [tuple((record["condition"], record["arm"])) for record in result["records"][:40]] == list(runner.expected_pairs())
    assert calls["hf"] == calls["lf"] == calls["plain"] == 8 and calls["scores"] == 320
    assert not list((store / run_id).glob("checkpoint-*.zip"))


@pytest.mark.integration
def test_operational_failure_is_a_durable_complete_40_record_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch, fail_hf_call=3)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    _env(monkeypatch)
    output, store = tmp_path / "output", tmp_path / "store"
    assert runner.execute(_args(repo, exact, output, store)) == 2
    result, _ = _terminal_payload(store, _run_id(output))
    failed = [record for record in result["records"] if record["status"] == "operational_failure"]
    assert result["complete"] is False and len(result["records"]) == 320
    assert len(failed) == 40 and {record["unit_id"] for record in failed} == {"frequency-response-0003"}
    assert {record["failure_reason"] for record in failed} == {"unit_execution_failure"}
    assert result["scientific_evaluation_allowed"] is False


@pytest.mark.integration
def test_keyboard_interrupt_does_not_commit_partial_unit_and_local_resume_reruns_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    first = _install_fakes(monkeypatch, interrupt_hf_call=2)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    _env(monkeypatch)
    output, store = tmp_path / "output", tmp_path / "store"
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(repo, exact, output, store))
    run_id = _run_id(output)
    state = json.loads((output / run_id / "state.json").read_text(encoding="utf-8"))
    assert state["committed_unit_ids"] == ["frequency-response-0001"]
    assert len(state["records"]) == 40 and first["hf"] == 2
    resumed = _install_fakes(monkeypatch)
    _env(monkeypatch)
    assert runner.execute(_args(repo, exact, output, store)) == 0
    assert resumed["hf"] == resumed["lf"] == resumed["plain"] == 7


@pytest.mark.integration
def test_two_hour_checkpoint_enables_sink_resume_without_recomputing_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch, interrupt_hf_call=2)
    clock = iter([0.0, 7201.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    _env(monkeypatch)
    first_output, store = tmp_path / "first", tmp_path / "store"
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(repo, exact, first_output, store))
    run_id = _run_id(first_output)
    checkpoint = next((store / run_id).glob("checkpoint-*.zip"))
    assert checkpoint.name == "checkpoint-0001-units-0001.zip"
    resumed = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    _env(monkeypatch)
    assert runner.execute(_args(repo, exact, tmp_path / "second", store)) == 0
    assert resumed["hf"] == resumed["lf"] == resumed["plain"] == 7


@pytest.mark.integration
def test_resume_rejects_identity_drift_and_39_or_41_record_transactions(tmp_path: Path) -> None:
    repo, exact = _repo(tmp_path)
    expected = _expected(repo, exact)
    drifted = dict(expected)
    drifted["roster_digest"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        runner._validate_state(drifted, expected)
    for count in (39, 41):
        state = dict(expected)
        state["committed_unit_ids"] = [expected["ordered_unit_ids"][0]]
        state["committed_unit_count"] = 1
        state["records"] = [{} for _ in range(count)]
        with pytest.raises(ValueError, match="40-record"):
            runner._validate_state(state, expected)


@pytest.mark.integration
def test_sink_orphan_bad_sha_and_divergence_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    expected = _expected(repo, exact)
    run_store = tmp_path / "sink" / expected["run_id"]
    run_store.mkdir(parents=True)
    orphan = run_store / "checkpoint-0001-units-0001.zip.sha256"
    orphan.write_text("0" * 64 + "  checkpoint-0001-units-0001.zip\n", encoding="utf-8")
    with pytest.raises(ValueError, match="orphan"):
        runner._discover_sink(run_store, expected)
    orphan.unlink()
    bad_zip = run_store / "checkpoint-0001-units-0001.zip"
    bad_zip.write_bytes(b"bad")
    (run_store / f"{bad_zip.name}.sha256").write_text(f"{'0' * 64}  {bad_zip.name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        runner._discover_sink(run_store, expected)

    local = dict(expected)
    sink = dict(expected)
    local["committed_unit_count"] = sink["committed_unit_count"] = 1
    local["committed_unit_ids"] = sink["committed_unit_ids"] = [expected["ordered_unit_ids"][0]]
    local["records"] = [{"value": "left"}] * 40
    sink["records"] = [{"value": "right"}] * 40
    with pytest.raises(ValueError, match="diverge"):
        runner._select_resume_state(local, sink)
    local = dict(expected)
    local["checkpoint_sequence"] = 1
    with pytest.raises(ValueError, match="no sink history"):
        runner._select_resume_state(local, None)


@pytest.mark.integration
def test_create_only_publication_and_valid_terminal_pair_prevent_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, sink = tmp_path / "source", tmp_path / "sink"
    source.mkdir()
    sink.mkdir()
    zip_path, sha_path = runner._write_zip_pair(source, "sample", {"state.json": {"safe": True}})
    (sink / zip_path.name).write_bytes(b"occupied")
    with pytest.raises(RuntimeError, match="overwrite"):
        runner._publish_pair_create_only(zip_path, sha_path, sink)
    assert (sink / zip_path.name).read_bytes() == b"occupied"

    repo, exact = _repo(tmp_path)
    initial = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    _env(monkeypatch)
    output, store = tmp_path / "output", tmp_path / "store"
    assert runner.execute(_args(repo, exact, output, store)) == 0
    assert initial["load"] == 1
    no_rerun = _install_fakes(monkeypatch)
    _env(monkeypatch)
    assert runner.execute(_args(repo, exact, tmp_path / "other-output", store)) == 0
    assert no_rerun["load"] == no_rerun["scores"] == 0


@pytest.mark.integration
def test_create_only_partial_keyboard_interrupt_leaves_no_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, sink = tmp_path / "source", tmp_path / "sink"
    source.mkdir()
    sink.mkdir()
    zip_path, sha_path = runner._write_zip_pair(
        source, "sample", {"state.json": {"safe": True}},
    )

    def partial_interrupt(source_stream: object, target: object) -> None:
        del source_stream
        target.write(b"partial")
        target.flush()
        raise KeyboardInterrupt

    monkeypatch.setattr(runner.shutil, "copyfileobj", partial_interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner._publish_pair_create_only(zip_path, sha_path, sink)
    assert list(sink.iterdir()) == []


@pytest.mark.integration
def test_terminal_fast_path_rejects_nonexact_or_rewritten_public_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    _env(monkeypatch)
    output, store = tmp_path / "output", tmp_path / "store"
    assert runner.execute(_args(repo, exact, output, store)) == 0
    expected = _expected(repo, exact)
    run_id = expected["run_id"]
    final_zip = store / run_id / f"{run_id}.zip"
    with zipfile.ZipFile(final_zip) as archive:
        baseline = {
            "receipt.json": json.loads(archive.read("receipt.json")),
            "result.json": json.loads(archive.read("result.json")),
        }

    mutations = [
        ("missing_limitations", None),
        ("altered_aggregate", None),
        ("receipt_disagreement", None),
        *((f"extra_{name}", name) for name in (
            "winner", "complementarity", "joint", "fpr", "threshold",
            "prompt", "secret", "private_latent",
        )),
    ]
    for index, (label, extra_name) in enumerate(mutations):
        payloads = json.loads(json.dumps(baseline))
        result = payloads["result.json"]
        if label == "missing_limitations":
            result.pop("limitations")
        elif label == "altered_aggregate":
            result["descriptive_per_method_response"]["hf"]["identity"][
                "successful_candidate_records"
            ] = 9
        elif label == "receipt_disagreement":
            payloads["receipt.json"]["status"] = "rewritten"
        else:
            result[extra_name] = "forbidden"
        directory = tmp_path / f"tampered-{index:02d}"
        directory.mkdir()
        zip_path, checksum_path = runner._write_zip_pair(directory, run_id, payloads)
        with pytest.raises(ValueError, match="terminal"):
            runner._validate_terminal_pair(
                ("final", zip_path, checksum_path), expected, None,
            )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mutation", "message"),
    (("lower", "latest verified checkpoint"),
     ("higher", "latest verified checkpoint"),
     ("rewritten", "diverges")),
)
def test_terminal_rejects_lower_higher_or_rewritten_checkpoint_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, message: str,
) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch, interrupt_hf_call=2)
    clock = iter([0.0, 7201.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    _env(monkeypatch)
    output, store = tmp_path / "first", tmp_path / "store"
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(repo, exact, output, store))
    _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    _env(monkeypatch)
    assert runner.execute(_args(repo, exact, tmp_path / "second", store)) == 0

    expected = _expected(repo, exact)
    checkpoint, terminal = runner._discover_sink(store / expected["run_id"], expected)
    assert checkpoint is not None and terminal is not None
    with zipfile.ZipFile(terminal[1]) as archive:
        result = json.loads(archive.read("result.json"))
    state = {key: result[key] for key in runner._STATE_KEYS}
    if mutation == "lower":
        state["checkpoint_sequence"] = checkpoint["checkpoint_sequence"] - 1
    elif mutation == "higher":
        state["checkpoint_sequence"] = checkpoint["checkpoint_sequence"] + 1
    else:
        state["records"][0]["scores"]["registered"] += 0.125
    records = [runner.StageARecord(**payload) for payload in state["records"]]
    forged_result = runner._result_payload(state, records, 0)
    payloads = {
        "receipt.json": runner._receipt_payload(forged_result, failure=False),
        "result.json": forged_result,
    }
    forged = tmp_path / "forged"
    forged.mkdir()
    zip_path, checksum_path = runner._write_zip_pair(
        forged, str(expected["run_id"]), payloads,
    )
    with pytest.raises(ValueError, match=message):
        runner._validate_terminal_pair(
            ("final", zip_path, checksum_path), expected, checkpoint,
        )


@pytest.mark.integration
def test_terminal_without_checkpoint_rejects_nonzero_checkpoint_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    _env(monkeypatch)
    output, store = tmp_path / "output", tmp_path / "store"
    assert runner.execute(_args(repo, exact, output, store)) == 0
    expected = _expected(repo, exact)
    result, _ = _terminal_payload(store, str(expected["run_id"]))
    state = {key: result[key] for key in runner._STATE_KEYS}
    state["checkpoint_sequence"] = 1
    records = [runner.StageARecord(**payload) for payload in state["records"]]
    forged_result = runner._result_payload(state, records, 0)
    forged = tmp_path / "forged"
    forged.mkdir()
    zip_path, checksum_path = runner._write_zip_pair(
        forged, str(expected["run_id"]), {
            "receipt.json": runner._receipt_payload(forged_result, failure=False),
            "result.json": forged_result,
        },
    )
    with pytest.raises(ValueError, match="latest verified checkpoint"):
        runner._validate_terminal_pair(
            ("final", zip_path, checksum_path), expected, None,
        )


@pytest.mark.integration
def test_missing_token_publishes_sanitized_failure_pair_and_prevents_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    calls = _install_fakes(monkeypatch)
    monkeypatch.setenv(runner.KEY_ENV, _KEY)
    monkeypatch.delenv(runner.TOKEN_ENV, raising=False)
    output, store = tmp_path / "output", tmp_path / "store"
    assert runner.execute(_args(repo, exact, output, store)) == 2
    run_id = _run_id(output)
    failure_zip = next((store / run_id).glob("failure-*.zip"))
    with zipfile.ZipFile(failure_zip) as archive:
        raw = b"".join(archive.read(name) for name in archive.namelist())
        result = json.loads(archive.read("result.json"))
    assert result["result_kind"] == "operational_failure_not_scientific"
    assert result["committed_unit_count"] == 0 and result["records"] == []
    assert _KEY.encode() not in raw and _TOKEN.encode() not in raw
    assert calls["load"] == calls["scores"] == 0
    _env(monkeypatch)
    assert runner.execute(_args(repo, exact, tmp_path / "other-output", store)) == 2
    assert calls["load"] == calls["scores"] == 0

    expected = _expected(repo, exact)
    with zipfile.ZipFile(failure_zip) as archive:
        payloads = {
            "receipt.json": json.loads(archive.read("receipt.json")),
            "result.json": json.loads(archive.read("result.json")),
        }
    payloads["result.json"]["descriptive_per_method_response"] = {}
    tampered = tmp_path / "tampered-failure"
    tampered.mkdir()
    zip_path, checksum_path = runner._write_zip_pair(
        tampered, "failure-hugging_face_token_missing", payloads,
    )
    with pytest.raises(ValueError, match="failure terminal result schema"):
        runner._validate_terminal_pair(
            ("failure", zip_path, checksum_path), expected, None,
        )


@pytest.mark.integration
def test_checkpointed_token_missing_failure_is_valid_and_prevents_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch, interrupt_hf_call=2)
    clock = iter([0.0, 7201.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    _env(monkeypatch)
    store = tmp_path / "store"
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(repo, exact, tmp_path / "first", store))

    calls = _install_fakes(monkeypatch)
    monkeypatch.setenv(runner.KEY_ENV, _KEY)
    monkeypatch.delenv(runner.TOKEN_ENV, raising=False)
    assert runner.execute(_args(repo, exact, tmp_path / "second", store)) == 2
    run_id = _run_id(store)
    failure_zip = next((store / run_id).glob("failure-*.zip"))
    with zipfile.ZipFile(failure_zip) as archive:
        result = json.loads(archive.read("result.json"))
    assert result["checkpoint_sequence"] == 1
    assert result["committed_unit_count"] == 1 and len(result["records"]) == 40
    assert calls["load"] == calls["scores"] == 0

    no_rerun = _install_fakes(monkeypatch)
    _env(monkeypatch)
    assert runner.execute(_args(repo, exact, tmp_path / "third", store)) == 2
    assert no_rerun["load"] == no_rerun["scores"] == 0


@pytest.mark.integration
def test_token_missing_failure_rejects_complete_fixed_roster_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    _env(monkeypatch)
    output, store = tmp_path / "output", tmp_path / "store"
    assert runner.execute(_args(repo, exact, output, store)) == 0
    expected = _expected(repo, exact)
    result, _ = _terminal_payload(store, str(expected["run_id"]))
    complete_state = {key: result[key] for key in runner._STATE_KEYS}
    with pytest.raises(ValueError, match="pending frozen roster"):
        runner._failure_result_payload(
            complete_state, "hugging_face_token_missing",
        )

    forged_result = {
        **complete_state,
        "evidence_contract": runner.EVIDENCE_CONTRACT,
        "result_kind": "operational_failure_not_scientific",
        "error_class": "hugging_face_token_missing",
        "rc": 2,
        "complete": False,
        "status": "operational_failure",
        "scientific_evaluation_allowed": False,
        "claim_ceiling": "descriptive_per_method_response_only",
        "limitations": list(runner._LIMITATIONS),
    }
    forged = tmp_path / "forged"
    forged.mkdir()
    zip_path, checksum_path = runner._write_zip_pair(
        forged, "failure-hugging_face_token_missing", {
            "receipt.json": runner._receipt_payload(forged_result, failure=True),
            "result.json": forged_result,
        },
    )
    with pytest.raises(ValueError, match="pending frozen roster"):
        runner._validate_terminal_pair(
            ("failure", zip_path, checksum_path), expected, None,
        )


@pytest.mark.integration
def test_artifacts_exclude_secrets_private_inputs_and_preserve_claim_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    _env(monkeypatch)
    output, store = tmp_path / "output", tmp_path / "store"
    assert runner.execute(_args(repo, exact, output, store)) == 0
    result, raw = _terminal_payload(store, _run_id(output))
    assert _KEY.encode() not in raw and _TOKEN.encode() not in raw
    roster_path = repo / "configs/stage_a_frequency_response/standalone_lf_hf_frequency_response_v1.jsonl"
    for line in roster_path.read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["prompt"].encode() not in raw
    serialized = json.dumps(result, sort_keys=True)
    assert not any(name in serialized for name in ("private_latent", "embedding_latent", "embed_side_route", "cached_qk"))
    assert result["claim_ceiling"] == "descriptive_per_method_response_only"
    assert set(result["descriptive_per_method_response"]) == {"hf", "lf"}
    assert not any(key in result for key in ("winner", "complementarity", "joint", "fpr", "threshold"))


@pytest.mark.integration
def test_cli_is_automatic_with_fixed_interval() -> None:
    options = {option for action in runner._parser()._actions for option in action.option_strings}
    assert "--run-mode" not in options and "--checkpoint-interval" not in options
    assert {"--output-root", "--run-store-root"}.issubset(options)
    assert runner.CHECKPOINT_INTERVAL_HOURS == 2.0
