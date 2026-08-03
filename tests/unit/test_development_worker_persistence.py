"""CPU constraints for create-only stateless development workers."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.protocol.development_exploration import DevelopmentStudyUnit
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    derive_source_cluster_id,
)
from experiments.runners.development_persistence import (
    DIAGNOSTIC_SCHEMA_VERSION,
    GPU_MIX_POLICY,
    HARD_SESSION_CAP_SECONDS,
    SOFT_STOP_SECONDS,
    DevelopmentPersistenceError,
    DevelopmentPersistentStore,
    FrozenDevelopmentUnitBinding,
    FrozenWorkerIdentity,
    SessionReceipt,
    canonical_json_bytes,
    create_frozen_development_unit_binding,
)


def _utc(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _roster(count: int = 6) -> tuple[DevelopmentStudyUnit, ...]:
    return tuple(
        DevelopmentStudyUnit(
            unit_index=index,
            phase="scientific_breadth",
            responsibility_id="key_schedule",
            source_cluster_ordinal=index,
            content_branch_id="not_applicable_content_branch",
            geometry_case_id="not_applicable_geometry_case",
            maximum_record_attempts=3,
            maximum_duration_seconds=900,
        )
        for index in range(count)
    )


def _protocol_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _bindings(
    roster: tuple[DevelopmentStudyUnit, ...],
) -> tuple[FrozenDevelopmentUnitBinding, ...]:
    bindings = []
    for unit in roster:
        prompt_digest = f"{unit.unit_index + 20:064x}"
        image_lineage_digest = f"{unit.unit_index + 40:064x}"
        key_family_digest = f"{unit.unit_index + 60:064x}"
        generation_seed = 1000 + unit.unit_index
        analysis_identity = AnalysisUnitIdentity(
            unit_id=f"manifest_development_unit_{unit.unit_index:04d}",
            case_id="development_manifest_case",
            source_cluster_id=derive_source_cluster_id(
                prompt_digest=prompt_digest,
                generation_seed=generation_seed,
                image_lineage_digest=image_lineage_digest,
                registered_key_family_digest=key_family_digest,
            ),
            prompt_digest=prompt_digest,
            generation_seed=generation_seed,
            image_lineage_digest=image_lineage_digest,
            registered_key_family_digest=key_family_digest,
        )
        bindings.append(
            create_frozen_development_unit_binding(
                unit,
                analysis_unit_identity=analysis_identity,
                scientific_question_id="registered_identity_separation",
                development_case_id="development_key_identity_separation",
                candidate_identity="registered_key_identity_candidate",
                candidate_config_digest=f"{unit.unit_index + 80:064x}",
            )
        )
    return tuple(bindings)


def _store(tmp_path: Path) -> DevelopmentPersistentStore:
    package = tmp_path / "development_execution_package.zip"
    bootstrap = tmp_path / "development_bootstrap.py"
    package.write_bytes(b"frozen development package")
    bootstrap.write_bytes(b"frozen development bootstrap")
    roster = _roster()
    bindings = _bindings(roster)
    worker = FrozenWorkerIdentity(
        revision="1" * 40,
        protocol_digest="2" * 64,
        execution_intent_authority_digest="3" * 64,
        input_manifest_digest="4" * 64,
        candidate_config_digest="5" * 64,
        unit_roster_digest=_protocol_digest(tuple(asdict(item) for item in roster)),
        package_sha256=sha256(package.read_bytes()).hexdigest(),
        bootstrap_sha256=sha256(bootstrap.read_bytes()).hexdigest(),
    )
    return DevelopmentPersistentStore(
        (tmp_path / "persistent").resolve(),
        run_id="development_run",
        worker_identity=worker,
        package_path=package,
        bootstrap_path=bootstrap,
        registered_unit_bindings=bindings,
    )


def _lease(store: DevelopmentPersistentStore, *, session_id: str = "colab_session", start: int = 100, duration: int = 100):
    return store.acquire_lease(
        session_id=session_id,
        now_epoch_seconds=start,
        lease_duration_seconds=duration,
    )


def _intent(store: DevelopmentPersistentStore, lease, *, index: int = 0, now: int = 101, attempt: int = 0, parent: str | None = None):
    return store.create_intent(
        lease,
        unit_id=f"development_scientific_unit_{index:04d}",
        unit_index=index,
        attempt_index=attempt,
        parent_attempt_intent_digest=parent,
        now_epoch_seconds=now,
    )


def _commit(store: DevelopmentPersistentStore, lease, intent, *, now: int = 102):
    return store.commit_unit(
        lease,
        intent,
        members={"records/scientific.json": b'{"status":"success"}\n'},
        now_epoch_seconds=now,
    )


def _receipt(store: DevelopmentPersistentStore, *, session_id: str, start: int, end: int, committed_unit_ids: tuple[str, ...]) -> SessionReceipt:
    return SessionReceipt(
        schema_version=DIAGNOSTIC_SCHEMA_VERSION,
        session_id=session_id,
        run_id=store.run_id,
        started_at_utc=_utc(start),
        ended_at_utc=_utc(end),
        gpu_model="nvidia_l4",
        cuda_identity="cuda_12_8",
        environment_digest="8" * 64,
        revision=store.worker_identity.revision,
        package_sha256=store.worker_identity.package_sha256,
        walltime_seconds=float(end - start),
        peak_vram_bytes=20_000_000_000,
        termination_reason="soft_stop_after_current_unit",
        soft_stop_seconds=SOFT_STOP_SECONDS,
        hard_session_cap_seconds=HARD_SESSION_CAP_SECONDS,
        gpu_mix_policy=GPU_MIX_POLICY,
        committed_unit_ids=committed_unit_ids,
        public_secret_identity_digests=("9" * 64,),
    )


@pytest.mark.quick
def test_construction_and_recovery_recompute_package_and_bootstrap_paths(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease)
    _commit(store, lease, intent)
    assert len(store.recover(now_epoch_seconds=201).ledger_digest) == 64

    store.package_path.write_bytes(b"drifted package")
    with pytest.raises(DevelopmentPersistenceError, match="package SHA-256 drifted"):
        store.recover(now_epoch_seconds=201)

    store.package_path.write_bytes(b"frozen development package")
    store.bootstrap_path.write_bytes(b"drifted bootstrap")
    with pytest.raises(DevelopmentPersistenceError, match="bootstrap SHA-256 drifted"):
        store.recover(now_epoch_seconds=201)


@pytest.mark.quick
def test_constructor_rejects_supplied_digest_not_matching_actual_path(tmp_path: Path) -> None:
    package = tmp_path / "package.zip"
    bootstrap = tmp_path / "bootstrap.py"
    package.write_bytes(b"package")
    bootstrap.write_bytes(b"bootstrap")
    roster = _roster(1)
    bindings = _bindings(roster)
    worker = FrozenWorkerIdentity(
        revision="1" * 40,
        protocol_digest="2" * 64,
        execution_intent_authority_digest="3" * 64,
        input_manifest_digest="4" * 64,
        candidate_config_digest="5" * 64,
        unit_roster_digest=_protocol_digest(tuple(asdict(item) for item in roster)),
        package_sha256="6" * 64,
        bootstrap_sha256=sha256(bootstrap.read_bytes()).hexdigest(),
    )
    with pytest.raises(DevelopmentPersistenceError, match="package SHA-256 drifted"):
        DevelopmentPersistentStore(
            tmp_path / "persistent",
            run_id="development_run",
            worker_identity=worker,
            package_path=package,
            bootstrap_path=bootstrap,
            registered_unit_bindings=bindings,
        )


@pytest.mark.quick
def test_intent_contains_complete_registered_unit_binding_and_rejects_drift(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease, index=2)

    assert intent.unit_id == "development_scientific_unit_0002"
    assert intent.shard_id == intent.phase == "scientific_breadth"
    assert intent.responsibility_id == "key_schedule"
    assert intent.source_cluster_ordinal == 2
    assert intent.content_branch_id == "not_applicable_content_branch"
    assert intent.geometry_case_id == "not_applicable_geometry_case"
    assert intent.maximum_record_attempts == 3
    assert intent.maximum_duration_seconds == 900
    assert intent.unit_roster_digest == store.worker_identity.unit_roster_digest
    assert intent.analysis_unit_identity["unit_id"] == "manifest_development_unit_0002"
    assert len(intent.analysis_unit_identity_digest) == 64
    assert intent.scientific_question_id == "registered_identity_separation"
    assert intent.development_case_id == "development_key_identity_separation"
    assert intent.candidate_identity == "registered_key_identity_candidate"
    assert len(intent.candidate_config_digest) == 64
    assert len(intent.unit_descriptor_digest) == 64

    with pytest.raises(DevelopmentPersistenceError, match="outside frozen roster"):
        store.create_intent(
            lease,
            unit_id="development_scientific_unit_999999",
            unit_index=999999,
            attempt_index=0,
            parent_attempt_intent_digest=None,
            now_epoch_seconds=102,
        )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("shard", "descriptor digest|registered binding"),
        ("analysis_identity", "analysis identity"),
        ("scientific_question", "descriptor digest|registered binding"),
        ("candidate_config", "descriptor digest|registered binding"),
        ("unit_descriptor", "descriptor digest"),
    ),
)
def test_recovery_rejects_registered_scientific_binding_tamper(
    tmp_path: Path,
    mutation: str,
    reason: str,
) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease)
    path = store.run_root / "intents" / f"{intent.unit_id}__attempt_0.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "shard":
        payload["shard_id"] = "critical_pair_extension"
    elif mutation == "analysis_identity":
        payload["analysis_unit_identity"]["prompt_digest"] = "f" * 64
    elif mutation == "scientific_question":
        payload["scientific_question_id"] = "drifted_scientific_question"
    elif mutation == "candidate_config":
        payload["candidate_config_digest"] = "f" * 64
    else:
        payload["unit_descriptor_digest"] = "f" * 64
    path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(DevelopmentPersistenceError, match=reason):
        store.recover(now_epoch_seconds=201)


@pytest.mark.quick
def test_create_only_commit_rebuilds_ledger_from_verified_markers(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease)
    marker = _commit(store, lease, intent)
    recovered = store.recover(now_epoch_seconds=201)

    assert recovered.committed_units == (marker,)
    assert recovered.interrupted_attempts == ()
    assert recovered.next_attempt_by_unit == ()
    assert len(recovered.ledger_digest) == 64
    assert not (store.run_root / "ledger.json").exists()
    with pytest.raises(DevelopmentPersistenceError, match="committed unit"):
        store.next_attempt_index(intent.unit_id)


@pytest.mark.quick
def test_only_expired_or_closed_leaf_intent_becomes_interrupted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first_lease = _lease(store, session_id="lost_session", duration=10)
    first = _intent(store, first_lease, index=1)

    active = store.recover(now_epoch_seconds=105)
    assert active.interrupted_attempts == ()
    assert active.next_attempt_by_unit == ()

    recovered = store.recover(now_epoch_seconds=111)
    interrupted = recovered.interrupted_attempts[0]
    assert interrupted.failure_class == "resource_failure"
    assert interrupted.failure_reason == "colab_session_interrupted"
    assert interrupted.retry_parent_intent_digest == first.digest()
    assert recovered.next_attempt_by_unit == ((first.unit_id, 1),)

    resumed = _lease(store, session_id="resumed_session", start=111, duration=20)
    second = _intent(
        store,
        resumed,
        index=1,
        now=112,
        attempt=1,
        parent=first.digest(),
    )
    marker = _commit(store, resumed, second, now=113)
    assert marker.parent_attempt_intent_digest == first.digest()
    assert store.recover(now_epoch_seconds=132).committed_units == (marker,)
    assert store.recover(now_epoch_seconds=132).interrupted_attempts == ()


@pytest.mark.quick
def test_single_writer_lease_soft_stop_and_stale_fence_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _lease(store, session_id="first_session", duration=10)
    with pytest.raises(DevelopmentPersistenceError, match="active writer"):
        _lease(store, session_id="second_session", start=105, duration=10)
    second = _lease(store, session_id="second_session", start=110, duration=HARD_SESSION_CAP_SECONDS - 1)
    assert second.fencing_token == first.fencing_token + 1
    with pytest.raises(DevelopmentPersistenceError, match="stale fencing"):
        _intent(store, first, index=2, now=111)
    _intent(store, second, index=2, now=110 + SOFT_STOP_SECONDS - 1)
    with pytest.raises(DevelopmentPersistenceError, match="soft stop"):
        _intent(store, second, index=3, now=110 + SOFT_STOP_SECONDS)


@pytest.mark.quick
@pytest.mark.parametrize(
    "member_path",
    (
        "../escape.json",
        "/absolute.json",
        "nested\\windows.json",
        "nested/../escape.json",
        "payload.py",
        "payload.PYC",
        "payload.sh",
        "payload.so",
        "payload.exe",
        "payload.pkl",
    ),
)
def test_unsafe_or_executable_bundle_members_are_rejected(tmp_path: Path, member_path: str) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease, index=3)
    with pytest.raises(DevelopmentPersistenceError, match="path|deserialization"):
        store.commit_unit(
            lease,
            intent,
            members={member_path: b"payload"},
            now_epoch_seconds=102,
        )


@pytest.mark.quick
def test_bundle_member_identities_are_unique_under_casefold(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease, index=4)
    with pytest.raises(DevelopmentPersistenceError, match="casefold"):
        store.commit_unit(
            lease,
            intent,
            members={"records/Result.json": b"one", "records/result.JSON": b"two"},
            now_epoch_seconds=102,
        )


@pytest.mark.quick
def test_secret_sentinel_never_reaches_bundle_or_receipt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease, index=4)
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
def test_session_receipt_is_strict_and_matches_lease_and_committed_markers(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    marker = _commit(store, lease, _intent(store, lease), now=102)
    receipt = _receipt(
        store,
        session_id=lease.session_id,
        start=100,
        end=102,
        committed_unit_ids=(marker.unit_id,),
    )
    path = store.write_session_receipt(receipt)
    assert json.loads(path.read_text(encoding="utf-8"))["soft_stop_seconds"] == 75600
    assert store.recover(now_epoch_seconds=102).committed_units == (marker,)

    mutations = (
        (replace(receipt, session_id="missing_session"), "lease lineage"),
        (replace(receipt, started_at_utc=_utc(99)), "start does not match lease"),
        (replace(receipt, ended_at_utc=_utc(99), walltime_seconds=-1.0), "UTC order|hard cap"),
        (replace(receipt, walltime_seconds=1.0), "walltime and UTC interval"),
        (replace(receipt, gpu_model=""), "GPU model"),
        (replace(receipt, cuda_identity=""), "CUDA identity"),
        (replace(receipt, peak_vram_bytes=0), "peak VRAM"),
        (replace(receipt, termination_reason=""), "termination reason"),
        (replace(receipt, committed_unit_ids=()), "differ from markers"),
    )
    for mutated, reason in mutations:
        with pytest.raises(DevelopmentPersistenceError, match=reason):
            store.write_session_receipt(mutated)


@pytest.mark.quick
def test_recovery_rejects_deleted_lease_and_marker_lineage_drift(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    _commit(store, lease, _intent(store, lease))
    lease_path = store.run_root / "leases" / "fence_00000001.json"
    lease_path.unlink()
    with pytest.raises(DevelopmentPersistenceError, match="lease/session/fence lineage"):
        store.recover(now_epoch_seconds=201)
