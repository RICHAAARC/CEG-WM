from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from cegwm.method.content_adaptive_v2 import COUNTERFACTUAL_EFFECT_FIELDS
from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
)
from cegwm.protocol.content_chain_v2 import load_content_adaptive_dual_branch_v2_clean_protocol
from experiments import run_content_adaptive_dual_branch_v2_clean as runner

_ROOT = Path(__file__).resolve().parents[2]
_EXACT = "a" * 40
_KEY = "runner-key-value-01"
_TOKEN = "hf_test_secret"
_OLD_PROTOCOL_DIGEST = "bfd9b7464195107f7dc57a43ab3042501500f5e2c07a322269859bb908a3dbb8"
_FIXED_PUBLIC_KEY_DIGEST = "805bc21e173a" + "0" * 52
_OLD_RUN_ID = "content-adaptive-v2-bfd9b7464195-805bc21e173a"


class _Generator:
    def __init__(self, device: str) -> None:
        assert device == "cuda"
        self.seed = 0

    def manual_seed(self, seed: int) -> _Generator:
        self.seed = seed
        return self


def _image(seed: int, offset: int) -> Image.Image:
    yy, xx = np.mgrid[:32, :32]
    pixels = np.stack((xx * 3 + yy, yy * 4 + xx, xx * 2 + yy * 2), axis=-1)
    return Image.fromarray((pixels + seed % 7 + offset + 30).astype(np.uint8), mode="RGB")


def _protocol():
    root = _ROOT / "configs" / "content_chain"
    return load_content_adaptive_dual_branch_v2_clean_protocol(
        root / "content_adaptive_dual_branch_v2_clean_v1.json",
        root / "content_adaptive_dual_branch_v2_clean.jsonl",
    )


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(_ROOT), expected_exact=_EXACT,
        local_work_root=str(tmp_path / "local"), artifact_sink=str(tmp_path / "sink"),
    )


def _set_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runner.KEY_ENV, _KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _TOKEN)


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    interrupt_at: int | None = None,
    system_exit_at: int | None = None,
    fail_all: bool = False,
) -> list[int]:
    protocol = _protocol()
    assets = SimpleNamespace(hf_public_assets=object(), lf_public_assets=object())
    monkeypatch.setattr(runner, "_git_exact", lambda repo, exact: exact)
    monkeypatch.setattr(runner, "_load_protocol", lambda repo: protocol)
    monkeypatch.setattr(runner, "_load_pipeline_and_assets", lambda model, token: (object(), assets))
    monkeypatch.setattr(runner.torch, "Generator", _Generator)
    calls: list[int] = []

    def adaptive(
        pipeline: object, prompt: str, key: bytes, received: object, **kwargs: object,
    ) -> SimpleNamespace:
        del pipeline, prompt, key
        assert received is assets
        call_index = len(calls)
        calls.append(call_index)
        if fail_all:
            raise RuntimeError(f"private {_KEY} {_TOKEN}")
        if interrupt_at is not None and call_index == interrupt_at:
            raise KeyboardInterrupt("private interruption")
        if system_exit_at is not None and call_index == system_exit_at:
            raise SystemExit("private interruption")
        unit_index = next(
            index for index, unit in enumerate(protocol.roster)
            if unit.seed == kwargs["generator"].seed
        )
        lf_share = 0.20 + 0.05 * unit_index
        effects = tuple(0.01 * effect + 0.001 * unit_index for effect in range(1, 7))
        measurement = SimpleNamespace(
            combined_budget=SimpleNamespace(relative_l2=0.0119),
            lf_effective_relative_l2=0.006, hf_effective_relative_l2=0.006,
            lf_branch_share=lf_share, hf_branch_share=1.0 - lf_share,
            **dict(zip(COUNTERFACTUAL_EFFECT_FIELDS, effects, strict=True)),
            minimum_counterfactual_effect=min(effects), probe_evaluation_count=64,
        )
        return SimpleNamespace(image=_image(kwargs["generator"].seed, 2), measurement=measurement)

    def plain(pipeline: object, prompt: str, **kwargs: object) -> Image.Image:
        del pipeline, prompt
        return _image(kwargs["generator"].seed, 0)

    score_calls = 0

    def scores(
        image: Image.Image, key: bytes, wrong: tuple[bytes, ...],
        hf_assets: object, lf_assets: object,
    ) -> dict[str, dict[str, float]]:
        nonlocal score_calls
        del image, key
        assert hf_assets is assets.hf_public_assets and lf_assets is assets.lf_public_assets
        registered = 0.9 if score_calls % 2 == 0 else 0.2
        score_calls += 1
        values = {"registered": registered, **{
            f"wrong_{index:02d}": 0.1 for index in range(len(wrong))
        }}
        return {"lf": dict(values), "hf": dict(values), "joint": dict(values)}

    monkeypatch.setattr(runner, "run_sd35_content_adaptive", adaptive)
    monkeypatch.setattr(runner, "run_sd35_plain", plain)
    monkeypatch.setattr(runner, "_blind_scores", scores)
    return calls


def _run_id(protocol=None) -> str:
    protocol = protocol or _protocol()
    key_digest = runner.public_key_digest(runner.normalize_detection_key(_KEY))
    return f"content-adaptive-v2-{protocol.protocol_digest[:12]}-{key_digest[:12]}"


def _terminal(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], bytes]:
    run_id = _run_id()
    archive_path = tmp_path / "sink" / run_id / f"{run_id}.zip"
    payload = archive_path.read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == ["receipt.json", "result.json"]
        receipt = json.loads(archive.read("receipt.json"))
        result = json.loads(archive.read("result.json"))
    checksum = archive_path.with_name(f"{archive_path.name}.sha256").read_text(encoding="ascii")
    assert checksum.split() == [runner.hashlib.sha256(payload).hexdigest(), archive_path.name]
    return receipt, result, payload


def _identity() -> tuple[object, dict[str, object]]:
    protocol = _protocol()
    key_digest = runner.public_key_digest(runner.normalize_detection_key(_KEY))
    return protocol, runner._public_identity(
        protocol, exact=_EXACT, key_digest=key_digest, run_id=_run_id(protocol)
    )


def _failure_transaction(protocol, identity, unit_index: int = 0) -> list[dict[str, object]]:
    unit = protocol.roster[unit_index]
    return [
        runner._content_v2_record(
            run_id=identity["run_id"], unit_id=unit.unit_id,
            source_cluster_id=unit.source_id, arm=arm, condition="clean",
            code_revision=identity["exact"], config_digest=identity["protocol_digest"],
            key_public_digest=identity["public_key_digest"], status="operational_failure",
            failure_reason="RuntimeError",
        )
        for arm in runner.ARMS
    ]


@pytest.mark.integration
def test_runtime_asset_contract_changes_run_identity_and_rejects_old_resume(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    assert protocol.protocol_digest == (
        "af4434590c12c882808279f331e1e987e2031719b9076c8d6ca2bd2d5f66d51f"
    )
    new_run_id = "content-adaptive-v2-af4434590c12-805bc21e173a"
    identity = runner._public_identity(
        protocol,
        exact=_EXACT,
        key_digest=_FIXED_PUBLIC_KEY_DIGEST,
        run_id=new_run_id,
    )
    assert identity["public_key_digest"][:12] == "805bc21e173a"
    assert identity["run_id"] == new_run_id
    assert identity["run_id"] != _OLD_RUN_ID
    old_identity = dict(identity)
    old_identity["run_id"] = _OLD_RUN_ID
    old_identity["protocol_id"] = "cegwm-stage-a-content-adaptive-dual-branch-v2-semantic-gate-v1"
    old_identity["protocol_digest"] = _OLD_PROTOCOL_DIGEST
    local_root = tmp_path / "local" / identity["run_id"]
    local_root.mkdir(parents=True)
    runner._write_local_state(local_root / "state.json", runner._new_state(old_identity, 1.0))
    with pytest.raises(ValueError, match="public identity differs"):
        runner._resolve_state(
            local_state_path=local_root / "state.json",
            sink_run_root=tmp_path / "sink" / identity["run_id"],
            identity=identity,
            protocol=protocol,
            now=2.0,
        )


def _prepare_complete_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_all: bool = False,
    checkpoint_at_end: bool = False,
) -> dict[str, object]:
    _install_fakes(monkeypatch, fail_all=fail_all)
    original_terminal = runner._publish_terminal
    original_now = runner._now
    monkeypatch.setattr(
        runner,
        "_publish_terminal",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    if checkpoint_at_end:
        times = iter([*([0.0] * 8), 7201.0])
        monkeypatch.setattr(runner, "_now", lambda: next(times))
    else:
        monkeypatch.setattr(runner, "_now", lambda: 1.0)
    _set_secrets(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(tmp_path))
    monkeypatch.setattr(runner, "_publish_terminal", original_terminal)
    monkeypatch.setattr(runner, "_now", original_now)
    state_path = tmp_path / "local" / _run_id() / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["committed_unit_count"] == 8 and len(state["records"]) == 16
    return state


def _forbid_completion_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("complete prefix must not call runtime or scoring helpers")

    for name in (
        "_wrong_keys", "_load_pipeline_and_assets", "load_dino_content_assets",
        "load_sd35_pipeline", "run_sd35_content_adaptive", "run_sd35_plain",
        "_blind_scores", "score_lf_image", "score_hf_image", "_unit_transaction",
    ):
        monkeypatch.setattr(runner, name, forbidden)


@pytest.mark.integration
def test_fresh_short_run_is_final_only_record_derived_and_secret_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fakes(monkeypatch)
    monkeypatch.setattr(runner, "_now", lambda: 100.0)
    _set_secrets(monkeypatch)
    assert runner.execute(_args(tmp_path)) == 0
    receipt, result, archive_bytes = _terminal(tmp_path)
    assert receipt["committed_unit_count"] == 8 and receipt["external_validation_required"] is True
    assert set(receipt).isdisjoint({"gate_evidence", "scientific_outcome_allowed", "records"})
    assert result["rc"] == 0 and result["scientific_outcome_allowed"] is True
    assert len(result["records"]) == 16 and len(result["unit_aggregate_metrics"]) == 8
    assert all(tuple(record) == runner.RECORD_FIELDS for record in result["records"])
    assert all(len(record["scores"]) == 51 for record in result["records"])
    expected_lf = np.asarray([0.20 + 0.05 * index for index in range(8)])
    assert result["lf_branch_share_population_std"] == pytest.approx(np.std(expected_lf, ddof=0))
    assert result["hf_branch_share_population_std"] == pytest.approx(np.std(1.0 - expected_lf, ddof=0))
    assert all(gate["gate_a_pass_units"] == 8 for gate in result["gate_evidence"]["branches"].values())
    assert not list((tmp_path / "sink" / _run_id()).glob("*.checkpoint-*.zip"))
    all_bytes = archive_bytes + (tmp_path / "local" / _run_id() / "state.json").read_bytes()
    assert _KEY.encode() not in all_bytes and _TOKEN.encode() not in all_bytes
    assert runner.KEY_ENV not in runner.os.environ and runner.TOKEN_ENV not in runner.os.environ
    lines = capsys.readouterr().out.splitlines()
    labels = [line.split(" ", 1)[0] for line in lines]
    events = [json.loads(line.split(" ", 1)[1]) for line in lines]
    assert labels == ["CEGWM_PROGRESS", "CEGWM_PROGRESS", *(["CEGWM_PROGRESS"] * 8), "CEGWM_SUMMARY"]
    assert [event["phase"] for event in events] == [
        "identity_ready", "resume_ready", *(["unit_committed"] * 8), "terminal",
    ]
    assert [event["committed"] for event in events] == [0, 0, *range(1, 9), 8]
    assert all(tuple(event) == ("run_id", "committed", "fixed_total", "phase") for event in events[:-1])
    assert tuple(events[-1]) == ("run_id", "committed", "fixed_total", "rc", "phase")


@pytest.mark.integration
def test_complete_local_prefix_terminalizes_without_token_or_runtime_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _prepare_complete_prefix(tmp_path, monkeypatch)
    protocol, identity = _identity()
    expected = runner._derive_result(state["records"], protocol, identity)
    _forbid_completion_helpers(monkeypatch)
    monkeypatch.setenv(runner.KEY_ENV, _KEY)
    monkeypatch.delenv(runner.TOKEN_ENV, raising=False)
    assert runner.execute(_args(tmp_path)) == 0
    _, result, payload = _terminal(tmp_path)
    assert result == expected
    assert result["records"] == state["records"]
    assert result["gate_evidence"]["all_predeclared_gates_pass"] is True
    assert result["lf_branch_share_population_std"] is not None
    assert _KEY.encode() not in payload


@pytest.mark.integration
def test_complete_sink_prefix_restores_and_terminalizes_without_token_or_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _prepare_complete_prefix(
        tmp_path, monkeypatch, checkpoint_at_end=True
    )
    local_state = tmp_path / "local" / _run_id() / "state.json"
    local_state.unlink()
    _forbid_completion_helpers(monkeypatch)
    monkeypatch.setenv(runner.KEY_ENV, _KEY)
    monkeypatch.delenv(runner.TOKEN_ENV, raising=False)
    assert runner.execute(_args(tmp_path)) == 0
    restored = json.loads(local_state.read_text(encoding="utf-8"))
    assert restored["records"] == state["records"]
    _, result, _ = _terminal(tmp_path)
    assert result["rc"] == 0 and result["records"] == state["records"]


@pytest.mark.integration
def test_complete_failure_prefix_terminalizes_exact_rc2_without_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _prepare_complete_prefix(tmp_path, monkeypatch, fail_all=True)
    _forbid_completion_helpers(monkeypatch)
    monkeypatch.setenv(runner.KEY_ENV, _KEY)
    monkeypatch.delenv(runner.TOKEN_ENV, raising=False)
    assert runner.execute(_args(tmp_path)) == 2
    _, result, _ = _terminal(tmp_path)
    assert result["rc"] == 2
    assert result["records"] == state["records"]
    assert len(result["failed_units"]) == 8
    assert result["gate_evidence"] is None
    assert result["scientific_outcome_allowed"] is False
    assert result["scientific_status"] == "not_evaluable"


@pytest.mark.integration
def test_incomplete_prefix_requires_token_and_model_starts_only_pending_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch, interrupt_at=2)
    monkeypatch.setattr(runner, "_now", lambda: 1.0)
    _set_secrets(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(tmp_path))
    state_path = tmp_path / "local" / _run_id() / "state.json"
    before = state_path.read_bytes()
    model_calls: list[tuple[str, str]] = []

    def model(model_id: str, token: str):
        model_calls.append((model_id, token))
        return object(), object()

    monkeypatch.setattr(runner, "_load_pipeline_and_assets", model)
    monkeypatch.setenv(runner.KEY_ENV, _KEY)
    monkeypatch.delenv(runner.TOKEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN_is_required_for_incomplete_execution"):
        runner.execute(_args(tmp_path))
    assert model_calls == [] and state_path.read_bytes() == before

    pending: list[str] = []

    def stop_pending(**kwargs: object) -> list[dict[str, object]]:
        pending.append(kwargs["unit"].unit_id)
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "_unit_transaction", stop_pending)
    _set_secrets(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(tmp_path))
    assert model_calls == [(_protocol().config["generation_runtime"]["model_id"], _TOKEN)]
    assert pending == [_protocol().roster[2].unit_id]
    assert state_path.read_bytes() == before


@pytest.mark.integration
def test_unit_exceptions_commit_two_failure_records_and_rc2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch, fail_all=True)
    monkeypatch.setattr(runner, "_now", lambda: 1.0)
    _set_secrets(monkeypatch)
    assert runner.execute(_args(tmp_path)) == 2
    _, result, payload = _terminal(tmp_path)
    assert result["scientific_outcome_allowed"] is False
    assert len(result["records"]) == 16 and len(result["failed_units"]) == 8
    assert all(record["status"] == "operational_failure" for record in result["records"])
    assert all(record["failure_reason"] == "RuntimeError" for record in result["records"])
    assert result["gate_evidence"] is None
    assert _KEY.encode() not in payload and _TOKEN.encode() not in payload


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["keyboard", "system_exit"])
def test_incomplete_unit_is_uncommitted_and_reruns_whole_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    kwargs = {"interrupt_at": 2} if kind == "keyboard" else {"system_exit_at": 2}
    _install_fakes(monkeypatch, **kwargs)
    monkeypatch.setattr(runner, "_now", lambda: 10.0)
    _set_secrets(monkeypatch)
    expected = KeyboardInterrupt if kind == "keyboard" else SystemExit
    with pytest.raises(expected):
        runner.execute(_args(tmp_path))
    state_path = tmp_path / "local" / _run_id() / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["committed_unit_count"] == 2 and len(state["records"]) == 4
    assert not (tmp_path / "sink" / _run_id() / f"{_run_id()}.zip").exists()
    resumed_calls = _install_fakes(monkeypatch)
    _set_secrets(monkeypatch)
    assert runner.execute(_args(tmp_path)) == 0
    assert len(resumed_calls) == 6


@pytest.mark.integration
def test_two_hour_checkpoint_follows_local_commit_and_sink_only_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fakes(monkeypatch, interrupt_at=1)
    times = iter([0.0, 7201.0])
    monkeypatch.setattr(runner, "_now", lambda: next(times))
    _set_secrets(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(tmp_path))
    assert len(calls) == 2
    local_state_path = tmp_path / "local" / _run_id() / "state.json"
    state = json.loads(local_state_path.read_text(encoding="utf-8"))
    assert state["committed_unit_count"] == 1 and state["checkpoint_sequence"] == 1
    sink_root = tmp_path / "sink" / _run_id()
    checkpoint = sink_root / f"{_run_id()}.checkpoint-0000.zip"
    assert checkpoint.exists() and checkpoint.with_name(f"{checkpoint.name}.sha256").exists()
    with zipfile.ZipFile(checkpoint) as archive:
        assert archive.namelist() == ["state.json"]
    local_state_path.unlink()
    resumed_calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(runner, "_now", lambda: 7201.0)
    _set_secrets(monkeypatch)
    assert runner.execute(_args(tmp_path)) == 0
    assert len(resumed_calls) == 7


@pytest.mark.integration
def test_sequence_lag_reconciliation_requires_identical_records(tmp_path: Path) -> None:
    protocol, identity = _identity()
    local_root = tmp_path / "local" / identity["run_id"]
    sink_root = tmp_path / "sink" / identity["run_id"]
    local_root.mkdir(parents=True)
    records = _failure_transaction(protocol, identity)
    local = runner._new_state(identity, 100.0)
    local["records"] = records
    local["committed_unit_count"] = 1
    runner._write_local_state(local_root / "state.json", local)
    checkpoint = dict(local)
    checkpoint["checkpoint_sequence"] = 1
    checkpoint["checkpoint_time_anchor_unix_seconds"] = 200.0
    runner._publish_pair(
        local_run_root=local_root, sink_run_root=sink_root,
        archive_name=f"{identity['run_id']}.checkpoint-0000.zip",
        members=(("state.json", runner._json_bytes(checkpoint)),),
    )
    resolved = runner._resolve_state(
        local_state_path=local_root / "state.json", sink_run_root=sink_root,
        identity=identity, protocol=protocol, now=300.0,
    )
    assert resolved["records"] == records and resolved["checkpoint_sequence"] == 1
    assert resolved["checkpoint_time_anchor_unix_seconds"] == 200.0


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["fewer", "more", "different", "identity", "rollback"])
def test_resume_rejects_divergence_identity_and_rollback(tmp_path: Path, variant: str) -> None:
    protocol, identity = _identity()
    local_root = tmp_path / "local" / identity["run_id"]
    sink_root = tmp_path / "sink" / identity["run_id"]
    local_root.mkdir(parents=True)
    local = runner._new_state(identity, 100.0)
    local["records"] = _failure_transaction(protocol, identity)
    local["committed_unit_count"] = 1
    if variant == "identity":
        local["identity"] = dict(identity)
        local["identity"]["record_contract_id"] = "drift"
        runner._write_local_state(local_root / "state.json", local)
    elif variant == "rollback":
        local["checkpoint_sequence"] = 1
        runner._write_local_state(local_root / "state.json", local)
    else:
        runner._write_local_state(local_root / "state.json", local)
        checkpoint = dict(local)
        checkpoint["checkpoint_sequence"] = 1
        checkpoint["checkpoint_time_anchor_unix_seconds"] = 200.0
        if variant == "fewer":
            checkpoint["records"] = []
            checkpoint["committed_unit_count"] = 0
        elif variant == "more":
            checkpoint["records"] = [*checkpoint["records"], *_failure_transaction(protocol, identity, 1)]
            checkpoint["committed_unit_count"] = 2
        else:
            checkpoint["records"] = [dict(record) for record in checkpoint["records"]]
            checkpoint["records"][0]["failure_reason"] = "ValueError"
            checkpoint["records"][1]["failure_reason"] = "ValueError"
        runner._publish_pair(
            local_run_root=local_root, sink_run_root=sink_root,
            archive_name=f"{identity['run_id']}.checkpoint-0000.zip",
            members=(("state.json", runner._json_bytes(checkpoint)),),
        )
    with pytest.raises(ValueError):
        runner._resolve_state(
            local_state_path=local_root / "state.json", sink_run_root=sink_root,
            identity=identity, protocol=protocol, now=300.0,
        )


@pytest.mark.integration
def test_state_rejects_partial_nonfinite_and_private_fields() -> None:
    protocol, identity = _identity()
    records = _failure_transaction(protocol, identity)
    for invalid_records in (records[:1], [*records, records[0]]):
        state = runner._new_state(identity, 1.0)
        state["records"] = invalid_records
        state["committed_unit_count"] = 1
        with pytest.raises(ValueError, match="two-record"):
            runner._validate_state(state, identity, protocol)
    nonfinite = runner._new_state(identity, 1.0)
    nonfinite["checkpoint_time_anchor_unix_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        runner._validate_state(nonfinite, identity, protocol)
    private = runner._new_state(identity, 1.0)
    private["records"] = [dict(record) for record in records]
    private["records"][0]["private_route"] = "forbidden"
    private["committed_unit_count"] = 1
    with pytest.raises(ValueError, match="fields or order"):
        runner._validate_state(private, identity, protocol)
    identity_type_drift = runner._new_state(identity, 1.0)
    identity_type_drift["identity"] = dict(identity)
    identity_type_drift["identity"]["fixed_unit_count"] = True
    with pytest.raises(ValueError, match="public identity differs"):
        runner._validate_state(identity_type_drift, identity, protocol)
    assert not runner._same_json_bytes({"records": [{"value": 1}]}, {"records": [{"value": 1.0}]})
    assert not runner._same_json_bytes({"records": [{"value": True}]}, {"records": [{"value": 1}]})


@pytest.mark.integration
@pytest.mark.parametrize("failure", [RuntimeError("copy"), KeyboardInterrupt(), SystemExit()])
def test_publication_cleans_only_current_partial_and_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException,
) -> None:
    local_root, sink_root = tmp_path / "local-run", tmp_path / "sink-run"
    local_root.mkdir()
    sink_root.mkdir()
    existing = sink_root / "preexisting.zip"
    existing.write_bytes(b"keep")
    def partial_then_fail(incoming, outgoing) -> None:
        outgoing.write(incoming.read(3))
        outgoing.flush()
        raise failure

    monkeypatch.setattr(runner.shutil, "copyfileobj", partial_then_fail)
    with pytest.raises(type(failure)):
        runner._publish_pair(
            local_run_root=local_root, sink_run_root=sink_root,
            archive_name="attempt.zip", members=(("state.json", b"{}\n"),),
        )
    assert existing.read_bytes() == b"keep"
    assert not (sink_root / "attempt.zip").exists()
    assert not (sink_root / "attempt.zip.sha256").exists()
    assert not list(local_root.iterdir())


@pytest.mark.integration
def test_zero_commit_init_fatal_and_fatal_after_prefix_are_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    monkeypatch.setattr(
        runner, "_load_pipeline_and_assets",
        lambda model, token: (_ for _ in ()).throw(RuntimeError(f"private {_KEY} {_TOKEN}")),
    )
    monkeypatch.setattr(runner, "_now", lambda: 1.0)
    _set_secrets(monkeypatch)
    assert runner.execute(_args(tmp_path)) == 2
    _, result, payload = _terminal(tmp_path)
    assert result["committed_unit_count"] == 0 and result["record_count"] == 0
    assert result["records"] == [] and result["operational_error_class"] == "RuntimeError"
    assert "gate_evidence" not in result and "scientific_outcome_allowed" not in result
    assert _KEY.encode() not in payload and _TOKEN.encode() not in payload

    second = tmp_path / "after-prefix"
    _install_fakes(monkeypatch)
    original_publish = runner._publish_pair

    def fail_checkpoint(**kwargs: object) -> None:
        if ".checkpoint-" in str(kwargs["archive_name"]):
            raise RuntimeError("private checkpoint failure")
        original_publish(**kwargs)

    monkeypatch.setattr(runner, "_publish_pair", fail_checkpoint)
    times = iter([0.0, 7201.0])
    monkeypatch.setattr(runner, "_now", lambda: next(times))
    _set_secrets(monkeypatch)
    assert runner.execute(_args(second)) == 2
    _, result, _ = _terminal(second)
    state = json.loads((second / "local" / _run_id() / "state.json").read_text(encoding="utf-8"))
    assert result["committed_unit_count"] == 1 and result["record_count"] == 2
    assert result["records"] == state["records"]


@pytest.mark.integration
def test_checksum_create_only_and_terminal_no_fast_path(tmp_path: Path) -> None:
    protocol, identity = _identity()
    local_root = tmp_path / "local" / identity["run_id"]
    sink_root = tmp_path / "sink" / identity["run_id"]
    local_root.mkdir(parents=True)
    state = runner._new_state(identity, 1.0)
    checkpoint = dict(state)
    checkpoint["checkpoint_sequence"] = 1
    name = f"{identity['run_id']}.checkpoint-0000.zip"
    runner._publish_pair(
        local_run_root=local_root, sink_run_root=sink_root,
        archive_name=name, members=(("state.json", runner._json_bytes(checkpoint)),),
    )
    with pytest.raises(FileExistsError, match="create-only"):
        runner._publish_pair(
            local_run_root=local_root, sink_run_root=sink_root,
            archive_name=name, members=(("state.json", runner._json_bytes(checkpoint)),),
        )
    (sink_root / f"{name}.sha256").write_text("0" * 64 + f"  {name}\n", encoding="ascii")
    with pytest.raises(ValueError, match="checksum"):
        runner._load_sink_checkpoint(sink_root, identity, protocol)
    (sink_root / f"{identity['run_id']}.zip").write_bytes(b"preexisting")
    with pytest.raises(FileExistsError, match="no terminal reconstruction"):
        runner._terminal_pair_presence(sink_root, identity["run_id"])


@pytest.mark.integration
def test_population_std_is_null_for_non_rc0_nonfinite_or_identity_invalid() -> None:
    unit_ids = tuple(f"unit-{index}" for index in range(8))
    metrics = [
        {"unit_id": unit_id, "lf_branch_share": 0.2 + index * 0.05,
         "hf_branch_share": 0.8 - index * 0.05}
        for index, unit_id in enumerate(unit_ids)
    ]
    for rc in (1, 2):
        assert runner._branch_share_population_summary(
            metrics, unit_ids, rc=rc, share_sum_absolute_tolerance=1e-12,
            population_std_absolute_tolerance=1e-12,
        ) == (None, None, False, False)
    nonfinite = [dict(metric) for metric in metrics]
    nonfinite[3]["lf_branch_share"] = float("nan")
    assert runner._branch_share_population_summary(
        nonfinite, unit_ids, rc=0, share_sum_absolute_tolerance=1e-12,
        population_std_absolute_tolerance=1e-12,
    ) == (None, None, False, False)
    duplicate = [dict(metric) for metric in metrics]
    duplicate[2]["unit_id"] = duplicate[1]["unit_id"]
    assert runner._branch_share_population_summary(
        duplicate, unit_ids, rc=0, share_sum_absolute_tolerance=1e-12,
        population_std_absolute_tolerance=1e-12,
    ) == (None, None, False, False)


class _BlindVAE(torch.nn.Module):
    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: pixels))


class _BlindProcessor:
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        del image
        return torch.zeros((1, 3, 2, 2))


@pytest.mark.integration
def test_recorded_score_helper_accepts_only_ordinary_image_keys_and_frozen_public_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tuple(inspect.signature(runner._blind_scores).parameters) == (
        "image", "key", "wrong_keys", "hf_public_assets", "lf_public_assets",
    )
    vae, processor = _BlindVAE(), _BlindProcessor()
    image_processor_id = "stabilityai/stable-diffusion-3.5-medium:image_processor"
    hf_assets = FrozenHFPublicAssets(vae, processor, image_processor_id)
    lf_assets = FrozenLFPublicAssets(
        vae, processor, image_processor_id, LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID, LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    wrong_keys = tuple(f"wrong-{index:02d}".encode() for index in range(16))
    monkeypatch.setattr(runner, "score_lf_image", lambda image, key, assets: float(len(key)))
    monkeypatch.setattr(runner, "score_hf_image", lambda image, key, assets: float(len(key) + 2))
    values = runner._blind_scores(
        Image.new("RGB", (4, 4)), b"registered", wrong_keys, hf_assets, lf_assets,
    )
    assert values["joint"]["registered"] == min(
        values["lf"]["registered"], values["hf"]["registered"]
    )
    with pytest.raises(ValueError, match="RGB"):
        runner._blind_scores(Image.new("L", (4, 4)), b"registered", wrong_keys, hf_assets, lf_assets)
    with pytest.raises(TypeError, match="FrozenHFPublicAssets"):
        runner._blind_scores(Image.new("RGB", (4, 4)), b"registered", wrong_keys, object(), lf_assets)
