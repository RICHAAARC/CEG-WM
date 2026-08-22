from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest

from cegwm.shared.keys import normalize_detection_key
from experiments.stage_a_frequency_response import run_colab as runner

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
    exact = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
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

    def load(model_id: str, token: str) -> tuple[object, object, object]:
        calls["load"] = int(calls["load"]) + 1
        assert model_id == "stabilityai/stable-diffusion-3.5-medium" and token == _TOKEN
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


def _args(repo: Path, exact: str, local: Path, sink: Path) -> argparse.Namespace:
    return argparse.Namespace(repo_root=str(repo), expected_exact=exact, local_work_root=str(local), artifact_sink=str(sink))


def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runner.KEY_ENV, _KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _TOKEN)


def _run_id(local: Path) -> str:
    run_dirs = [path for path in local.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1 and run_dirs[0].name.startswith("slhfr-")
    return run_dirs[0].name


def _final(sink: Path, run_id: str) -> tuple[dict[str, object], dict[str, object], bytes]:
    zip_path = sink / run_id / f"{run_id}.zip"
    checksum_path = sink / run_id / f"{run_id}.zip.sha256"
    raw = zip_path.read_bytes()
    declared = checksum_path.read_text(encoding="utf-8").split()
    assert declared == [hashlib.sha256(raw).hexdigest(), zip_path.name]
    with zipfile.ZipFile(zip_path) as archive:
        assert set(archive.namelist()) == {"receipt.json", "result.json"}
        return json.loads(archive.read("receipt.json")), json.loads(archive.read("result.json")), raw


def _state(local: Path, run_id: str) -> dict[str, object]:
    return json.loads((local / run_id / "state.json").read_text(encoding="utf-8"))


@pytest.mark.integration
def test_fresh_run_exports_fixed_320_descriptive_only_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    calls = _install_fakes(monkeypatch)
    _env(monkeypatch)
    local, sink = tmp_path / "local", tmp_path / "sink"
    assert runner.execute(_args(repo, exact, local, sink)) == 0
    run_id = _run_id(local)
    receipt, result, raw = _final(sink, run_id)
    assert receipt["evidence_contract"] == "STANDALONE_LF_HF_FREQUENCY_RESPONSE_EVIDENCE"
    assert result["complete"] is True and result["rc"] == 0 and len(result["records"]) == 320
    assert [tuple((record["condition"], record["arm"])) for record in result["records"][:40]] == list(runner.expected_pairs())
    assert calls["hf"] == calls["lf"] == calls["plain"] == 8 and calls["scores"] == 320
    assert set(result["descriptive_per_method_response"]) == {"hf", "lf"}
    assert not any(term in result for term in ("winner", "complementarity", "joint", "fpr", "threshold"))
    assert b"sensitive runtime detail" not in raw and _KEY.encode() not in raw and _TOKEN.encode() not in raw
    first_prompt = json.loads((_ROOT / "configs/stage_a_frequency_response/standalone_lf_hf_frequency_response_v1.jsonl").read_text(encoding="utf-8").splitlines()[0])["prompt"]
    assert first_prompt.encode() not in raw
    assert not any(name in raw for name in (b'"private_latent"', b'"carrier"', b'"mask"', b'"route"'))
    assert not list((sink / run_id).glob("*.json")) and not list((sink / run_id).glob("state*"))


@pytest.mark.integration
def test_operational_failure_is_atomic_retained_and_terminal_does_not_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    calls = _install_fakes(monkeypatch, fail_hf_call=3)
    _env(monkeypatch)
    local, sink = tmp_path / "local", tmp_path / "sink"
    assert runner.execute(_args(repo, exact, local, sink)) == 2
    run_id = _run_id(local)
    _, result, raw = _final(sink, run_id)
    failures = [record for record in result["records"] if record["status"] == "operational_failure"]
    assert len(failures) == 40 and {record["unit_id"] for record in failures} == {"frequency-response-0003"}
    assert {record["failure_reason"] for record in failures} == {"unit_execution_failure"}
    assert b"sensitive runtime detail" not in raw
    second_calls = _install_fakes(monkeypatch)
    _env(monkeypatch)
    assert runner.execute(_args(repo, exact, local, sink)) == 2
    assert second_calls["load"] == second_calls["hf"] == second_calls["scores"] == 0
    assert calls["hf"] == 8


@pytest.mark.integration
def test_keyboard_interrupt_leaves_prefix_and_reruns_whole_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    calls = _install_fakes(monkeypatch, interrupt_hf_call=3)
    _env(monkeypatch)
    local, sink = tmp_path / "local", tmp_path / "sink"
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(repo, exact, local, sink))
    run_id = _run_id(local)
    state = _state(local, run_id)
    assert state["committed_unit_count"] == 2 and len(state["records"]) == 80
    resumed = _install_fakes(monkeypatch)
    _env(monkeypatch)
    assert runner.execute(_args(repo, exact, local, sink)) == 0
    assert resumed["hf"] == resumed["lf"] == resumed["plain"] == 6
    assert calls["hf"] == 3 and calls["lf"] == calls["plain"] == 2


@pytest.mark.integration
def test_two_hour_checkpoint_restores_new_runtime_and_short_run_is_final_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch, interrupt_hf_call=3)
    clock_values = iter([0.0, 100.0, 7300.0])
    monkeypatch.setattr(runner.time, "time", lambda: next(clock_values))
    _env(monkeypatch)
    local, sink = tmp_path / "local", tmp_path / "sink"
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(repo, exact, local, sink))
    run_id = _run_id(local)
    checkpoint_pairs = sorted((sink / run_id).glob("*checkpoint*.zip"))
    assert len(checkpoint_pairs) == 1 and checkpoint_pairs[0].with_suffix(".zip.sha256").is_file()
    shutil.rmtree(local / run_id)
    resumed = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "time", lambda: 7400.0)
    _env(monkeypatch)
    assert runner.execute(_args(repo, exact, local, sink)) == 0
    assert resumed["hf"] == 6

    short_repo, short_exact = _repo(tmp_path / "short")
    _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "time", lambda: 1.0)
    _env(monkeypatch)
    short_local, short_sink = tmp_path / "short-local", tmp_path / "short-sink"
    assert runner.execute(_args(short_repo, short_exact, short_local, short_sink)) == 0
    short_id = _run_id(short_local)
    assert not list((short_sink / short_id).glob("*checkpoint*"))


def _interrupted_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, units: int = 1) -> tuple[Path, str, Path, Path, str]:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch, interrupt_hf_call=units + 1)
    _env(monkeypatch)
    local, sink = tmp_path / "local", tmp_path / "sink"
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(repo, exact, local, sink))
    return repo, exact, local, sink, _run_id(local)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda state: state.__setitem__("protocol_digest", "0" * 64), "protocol_digest"),
        (lambda state: state.__setitem__("roster_digest", "0" * 64), "roster_digest"),
        (lambda state: state.__setitem__("key_public_digest", "0" * 64), "key_public_digest"),
        (lambda state: state.__setitem__("ordered_unit_ids", list(reversed(state["ordered_unit_ids"]))), "ordered_unit_ids"),
        (lambda state: state.__setitem__("condition_order", list(reversed(state["condition_order"]))), "condition_order"),
        (lambda state: state.__setitem__("record_arms_in_exact_condition_order", list(reversed(state["record_arms_in_exact_condition_order"]))), "record_arms"),
        (lambda state: state.__setitem__("fixed_record_count", 319), "fixed_record_count"),
    ],
)
def test_resume_rejects_identity_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutator: object, match: str) -> None:
    repo, exact, local, sink, run_id = _interrupted_state(tmp_path, monkeypatch)
    state_path = local / run_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    mutator(state)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _install_fakes(monkeypatch)
    _env(monkeypatch)
    with pytest.raises(ValueError, match=match):
        runner.execute(_args(repo, exact, local, sink))


@pytest.mark.integration
@pytest.mark.parametrize("record_count", [39, 41])
def test_resume_rejects_non_atomic_record_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_count: int) -> None:
    repo, exact, local, sink, run_id = _interrupted_state(tmp_path, monkeypatch)
    state_path = local / run_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    records = list(state["records"])
    state["records"] = (records * 2)[:record_count]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _install_fakes(monkeypatch)
    _env(monkeypatch)
    with pytest.raises(ValueError, match="record count"):
        runner.execute(_args(repo, exact, local, sink))


@pytest.mark.integration
@pytest.mark.parametrize(("map_name", "value"), [("scores", math.nan), ("metrics", math.inf), ("scores", True)])
def test_resume_rejects_nonfinite_or_bool_public_numbers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, map_name: str, value: object) -> None:
    repo, exact, local, sink, run_id = _interrupted_state(tmp_path, monkeypatch)
    state_path = local / run_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    record = next(item for item in state["records"] if item[map_name])
    record[map_name][next(iter(record[map_name]))] = value
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _install_fakes(monkeypatch)
    _env(monkeypatch)
    with pytest.raises(ValueError, match="finite public numbers"):
        runner.execute(_args(repo, exact, local, sink))


@pytest.mark.integration
def test_resume_rejects_extra_record_or_score_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact, local, sink, run_id = _interrupted_state(tmp_path, monkeypatch)
    state_path = local / run_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["records"][0]["scores"]["extra"] = 0.0
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _install_fakes(monkeypatch)
    _env(monkeypatch)
    with pytest.raises(ValueError, match="score or failure shape"):
        runner.execute(_args(repo, exact, local, sink))


@pytest.mark.integration
def test_local_history_may_extend_but_may_not_diverge_from_sink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch, interrupt_hf_call=3)
    clock_values = iter([0.0, 7300.0, 7400.0])
    monkeypatch.setattr(runner.time, "time", lambda: next(clock_values))
    _env(monkeypatch)
    local, sink = tmp_path / "local", tmp_path / "sink"
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(repo, exact, local, sink))
    run_id = _run_id(local)
    state_path = local / run_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["records"][0]["scores"]["registered"] += 0.125
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "time", lambda: 7500.0)
    _env(monkeypatch)
    with pytest.raises(RuntimeError, match="diverge"):
        runner.execute(_args(repo, exact, local, sink))


@pytest.mark.integration
def test_checkpoint_orphan_fails_closed_before_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch, interrupt_hf_call=2)
    clock_values = iter([0.0, 7300.0])
    monkeypatch.setattr(runner.time, "time", lambda: next(clock_values))
    _env(monkeypatch)
    local, sink = tmp_path / "local", tmp_path / "sink"
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(repo, exact, local, sink))
    run_id = _run_id(local)
    checkpoint_sha = next((sink / run_id).glob("*checkpoint*.zip.sha256"))
    checkpoint_sha.unlink()
    _install_fakes(monkeypatch)
    monkeypatch.setattr(runner.time, "time", lambda: 7400.0)
    _env(monkeypatch)
    with pytest.raises(RuntimeError, match="orphan"):
        runner.execute(_args(repo, exact, local, sink))


@pytest.mark.integration
def test_create_only_publication_cleans_partial_current_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_zip = tmp_path / "source.zip"
    source_sha = tmp_path / "source.zip.sha256"
    source_zip.write_bytes(b"complete zip bytes")
    source_sha.write_text("digest  source.zip\n", encoding="utf-8")
    sink_zip, sink_sha = tmp_path / "sink.zip", tmp_path / "sink.zip.sha256"
    original = shutil.copyfileobj
    calls = 0

    def interrupted(reader: object, writer: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            writer.write(b"partial")
            raise KeyboardInterrupt
        original(reader, writer)

    monkeypatch.setattr(runner.shutil, "copyfileobj", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner._publish_pair(source_zip, source_sha, sink_zip, sink_sha)
    assert not sink_zip.exists() and not sink_sha.exists()

    sink_zip.write_bytes(b"pre-existing")
    with pytest.raises(FileExistsError):
        runner._publish_pair(source_zip, source_sha, sink_zip, sink_sha)
    assert sink_zip.read_bytes() == b"pre-existing" and not sink_sha.exists()
