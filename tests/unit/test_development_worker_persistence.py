"""CPU constraints for create-only stateless development workers."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from experiments.runners.development_persistence import (
    GPU_MIX_POLICY,
    HARD_SESSION_CAP_SECONDS,
    DIAGNOSTIC_SCHEMA_VERSION,
    SOFT_STOP_SECONDS,
    DevelopmentPersistenceError,
    DevelopmentPersistentStore,
    FrozenWorkerIdentity,
    SessionReceipt,
)


def _worker() -> FrozenWorkerIdentity:
    return FrozenWorkerIdentity(
        revision="1" * 40,
        protocol_digest="2" * 64,
        execution_intent_authority_digest="3" * 64,
        input_manifest_digest="4" * 64,
        candidate_config_digest="5" * 64,
        package_sha256="6" * 64,
        bootstrap_sha256="7" * 64,
    )


def _store(tmp_path: Path) -> DevelopmentPersistentStore:
    return DevelopmentPersistentStore(
        tmp_path.resolve(), run_id="development_run", worker_identity=_worker()
    )


@pytest.mark.quick
def test_create_only_commit_rebuilds_ledger_from_verified_markers(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = store.acquire_lease(
        session_id="colab_session", now_epoch_seconds=100, lease_duration_seconds=100
    )
    intent = store.create_intent(
        lease,
        shard_id="scientific_breadth",
        unit_id="development_scientific_unit_0000",
        unit_index=0,
        attempt_index=0,
        parent_attempt_intent_digest=None,
        now_epoch_seconds=101,
    )
    marker = store.commit_unit(
        lease,
        intent,
        members={"records/scientific.json": b'{"status":"success"}\n'},
        now_epoch_seconds=102,
    )
    recovered = store.recover()

    assert recovered.committed_units == (marker,)
    assert recovered.interrupted_attempts == ()
    assert recovered.next_attempt_by_unit == ()
    assert len(recovered.ledger_digest) == 64
    assert not (store.run_root / "ledger.json").exists()
    with pytest.raises(DevelopmentPersistenceError, match="committed unit"):
        store.next_attempt_index(intent.unit_id)


@pytest.mark.quick
def test_dangling_intent_becomes_interrupted_retry_lineage(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first_lease = store.acquire_lease(
        session_id="lost_session", now_epoch_seconds=100, lease_duration_seconds=10
    )
    first = store.create_intent(
        first_lease,
        shard_id="scientific_breadth",
        unit_id="development_scientific_unit_0001",
        unit_index=1,
        attempt_index=0,
        parent_attempt_intent_digest=None,
        now_epoch_seconds=101,
    )
    recovered = store.recover()
    interrupted = recovered.interrupted_attempts[0]

    assert interrupted.failure_class == "resource_failure"
    assert interrupted.failure_reason == "colab_session_interrupted"
    assert interrupted.retry_parent_intent_digest == first.digest()
    assert recovered.next_attempt_by_unit == ((first.unit_id, 1),)

    resumed = store.acquire_lease(
        session_id="resumed_session", now_epoch_seconds=111, lease_duration_seconds=20
    )
    second = store.create_intent(
        resumed,
        shard_id="scientific_breadth",
        unit_id=first.unit_id,
        unit_index=1,
        attempt_index=1,
        parent_attempt_intent_digest=first.digest(),
        now_epoch_seconds=112,
    )
    marker = store.commit_unit(
        resumed,
        second,
        members={"records/scientific.json": b'{"status":"success"}\n'},
        now_epoch_seconds=113,
    )
    assert marker.parent_attempt_intent_digest == first.digest()
    assert store.recover().committed_units == (marker,)


@pytest.mark.quick
def test_single_writer_lease_and_stale_fence_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.acquire_lease(
        session_id="first_session", now_epoch_seconds=100, lease_duration_seconds=10
    )
    with pytest.raises(DevelopmentPersistenceError, match="active writer"):
        store.acquire_lease(
            session_id="second_session", now_epoch_seconds=105, lease_duration_seconds=10
        )
    second = store.acquire_lease(
        session_id="second_session", now_epoch_seconds=110, lease_duration_seconds=10
    )
    assert second.fencing_token == first.fencing_token + 1
    with pytest.raises(DevelopmentPersistenceError, match="stale fencing"):
        store.create_intent(
            first,
            shard_id="scientific_breadth",
            unit_id="development_scientific_unit_0002",
            unit_index=2,
            attempt_index=0,
            parent_attempt_intent_digest=None,
            now_epoch_seconds=111,
        )


@pytest.mark.quick
@pytest.mark.parametrize(
    "member_path",
    ("../escape.json", "/absolute.json", "nested\\windows.json", "nested/../escape.json"),
)
def test_unsafe_bundle_member_paths_are_rejected(
    tmp_path: Path, member_path: str
) -> None:
    store = _store(tmp_path)
    lease = store.acquire_lease(
        session_id="colab_session", now_epoch_seconds=100, lease_duration_seconds=50
    )
    intent = store.create_intent(
        lease,
        shard_id="scientific_breadth",
        unit_id="development_scientific_unit_0003",
        unit_index=3,
        attempt_index=0,
        parent_attempt_intent_digest=None,
        now_epoch_seconds=101,
    )
    with pytest.raises(DevelopmentPersistenceError, match="path"):
        store.commit_unit(
            lease,
            intent,
            members={member_path: b"payload"},
            now_epoch_seconds=102,
        )


@pytest.mark.quick
def test_secret_sentinel_never_reaches_bundle_or_receipt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = store.acquire_lease(
        session_id="colab_session", now_epoch_seconds=100, lease_duration_seconds=50
    )
    intent = store.create_intent(
        lease,
        shard_id="scientific_breadth",
        unit_id="development_scientific_unit_0004",
        unit_index=4,
        attempt_index=0,
        parent_attempt_intent_digest=None,
        now_epoch_seconds=101,
    )
    with pytest.raises(DevelopmentPersistenceError, match="secret"):
        store.commit_unit(
            lease,
            intent,
            members={"records/scientific.json": b'{"token":"raw-secret-sentinel"}\n'},
            raw_secret_values=("raw-secret-sentinel",),
            now_epoch_seconds=102,
        )
    assert not tuple((store.run_root / "bundles").iterdir())
    assert not tuple((store.run_root / "markers").iterdir())


@pytest.mark.quick
def test_session_receipt_freezes_soft_stop_hard_cap_and_gpu_mix_policy(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    receipt = SessionReceipt(
        schema_version=DIAGNOSTIC_SCHEMA_VERSION,
        session_id="colab_session",
        run_id=store.run_id,
        started_at_utc="2026-08-03T00:00:00Z",
        ended_at_utc="2026-08-03T01:00:00Z",
        gpu_model="nvidia_l4",
        cuda_identity="cuda_12_8",
        environment_digest="8" * 64,
        revision=_worker().revision,
        package_sha256=_worker().package_sha256,
        walltime_seconds=3600.0,
        peak_vram_bytes=20_000_000_000,
        termination_reason="soft_stop_after_current_unit",
        soft_stop_seconds=SOFT_STOP_SECONDS,
        hard_session_cap_seconds=HARD_SESSION_CAP_SECONDS,
        gpu_mix_policy=GPU_MIX_POLICY,
        committed_unit_ids=("development_scientific_unit_0000",),
        public_secret_identity_digests=("9" * 64,),
    )
    path = store.write_session_receipt(receipt)

    assert json.loads(path.read_text(encoding="utf-8"))["soft_stop_seconds"] == 75600
    with pytest.raises(DevelopmentPersistenceError, match="hard cap"):
        store.write_session_receipt(
            replace(
                receipt,
                session_id="over_cap_session",
                walltime_seconds=float(HARD_SESSION_CAP_SECONDS),
            )
        )
