"""Frozen CPU tests for the salient-local-LF mask/write pilot."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from struct import pack, unpack

import pytest
import torch

from experiments.methods import CegWmExperimentAdapter, load_ceg_wm_experiment_adapter_configuration
from experiments.metrics.salient_local_lf_mask_write_validation import (
    SalientLocalLfMaskWriteMetricError,
    SalientLocalLfTerminalFailure,
    aggregate_salient_local_lf_mask_write_validation,
    create_mask_write_observation,
    observe_public_rgb8_quality,
)
from experiments.protocol.development_records import DevelopmentScientificRecord
from experiments.protocol.salient_local_lf_mask_write_validation import (
    CANONICAL_CONTENT_RELATIVE_L2_LIMIT,
    SCIENTIFIC_ROSTER_AUTHORITY_DIGEST,
    SalientLocalLfMaskWriteProtocolError,
    canonical_digest,
    load_salient_local_lf_mask_write_validation_protocol,
)
from experiments.runners.salient_local_lf_mask_write_validation import (
    SalientLocalLfMaskWriteIdentityError,
    SalientLocalLfMaskWriteIntegrityError,
    SalientLocalLfMaskWriteRunnerError,
    SalientLocalLfMaskWriteValidationRunner,
    aggregate_supports_scientific_claim,
    _actual_dtype_budget_pass,
)
from experiments.runners.development_persistence import (
    DevelopmentPersistentStore,
    FrozenWorkerIdentity,
)
from main import identify_root_key, rgb8_image_digest
from runtime import InspyrenetSaliencyRuntime, Sd35RuntimeAdapter
from scripts.experiment_execution.salient_local_lf_mask_write_validation_entrypoint import (
    _classify_scientific_failure,
    _safe_failure,
)
from scripts.experiment_execution.salient_local_lf_mask_write_validation_server import _verify_locked_dependencies


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/salient_local_lf_mask_write_validation.json"
COMPONENTS = ROOT / "configs/experiments/internal_execution_components.json"


def struct_binary32(value: float) -> float:
    return unpack(">f", pack(">f", value))[0]


def next_binary32(value: float) -> float:
    bits = unpack(">I", pack(">f", value))[0]
    return unpack(">f", pack(">I", bits + 1))[0]


def _protocol():
    return load_salient_local_lf_mask_write_validation_protocol(CONFIG, repository_root=ROOT)


def _quality(*, over_limit: bool = False):
    clean = torch.full((1, 3, 512, 512), 100, dtype=torch.uint8)
    marked = torch.full_like(clean, 101)
    if over_limit:
        marked.reshape(-1)[0] = 102
    return observe_public_rgb8_quality(
        clean,
        marked,
        clean_image_digest=rgb8_image_digest(clean),
        marked_image_digest=rgb8_image_digest(marked),
    )


def _observation(
    cluster: int,
    *,
    mechanism: bool = True,
    quality_pass: bool = True,
    source_cluster_id: str | None = None,
):
    quality = _quality(over_limit=not quality_pass)
    return create_mask_write_observation(
        cluster_ordinal=cluster,
        source_cluster_id=source_cluster_id or f"{cluster + 1:064x}",
        clean_image_digest=quality.clean_image_digest,
        marked_image_digest=quality.marked_image_digest,
        embed_saliency_observation_identity=f"{cluster + 11:064x}",
        detect_saliency_observation_identity=f"{cluster + 21:064x}",
        embed_mask_identity=f"{cluster + 31:064x}",
        detect_mask_identity=f"{cluster + 41:064x}",
        embed_mask_coverage=256 if mechanism else 32,
        detect_mask_coverage=256,
        mask_intersection_over_union=0.75,
        nominal_masked_lf_outside_bitwise_zero=True,
        nominal_masked_lf_inside_nonzero=True,
        nominal_masked_lf_consumed_by_materialization=True,
        accepted_materialization_replay_identity=f"{cluster + 51:064x}",
        realized_relative_l2=0.01,
        actual_dtype_budget_pass=True,
        identity_pass=True,
        integrity_pass=True,
        quality=quality,
    )


def _runner() -> SalientLocalLfMaskWriteValidationRunner:
    protocol = _protocol()
    runtime = object.__new__(Sd35RuntimeAdapter)
    saliency = object.__new__(InspyrenetSaliencyRuntime)
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS),
        runtime_adapter=runtime,
    )
    return SalientLocalLfMaskWriteValidationRunner(
        protocol=protocol, adapter=adapter, runtime_adapter=runtime,
        saliency_runtime=saliency, method_code_revision="1" * 40,
        registered_root_key="salient-mask-write-test-root",
        protocol_digest=protocol.digest(), execution_intent_authority_digest="2" * 64,
        candidate_config_digest="3" * 64, package_identity="4" * 64,
    )


def test_authored_roster_and_historical_producer_authorities_are_exact() -> None:
    protocol = _protocol()
    assert protocol.manifest.scientific_roster_authority_digest == SCIENTIFIC_ROSTER_AUTHORITY_DIGEST
    assert canonical_digest(protocol.manifest.authority_payload()) == SCIENTIFIC_ROSTER_AUTHORITY_DIGEST
    assert len(protocol.unit_roster) == 10
    assert tuple(item.generation_seed for item in protocol.manifest.entries) == tuple(range(202608150100, 202608150108))
    assert tuple(item.source_cluster_id for item in protocol.manifest.entries) == (
        "dd32b622fef8f72ec34ab75821f07d4f6aac09357e3e9ae64ba1dd3b088841b9",
        "134b101f00c67ff3f7a572599f48b9e08375ea4b99bd0758f0cbdda6372dcace",
        "9e80a52b9811bd818d4fc54e737b5d14352f07ab28fcf1ffb4e58b9a43efa3cb",
        "05ece93f211d8bfe08ccccfe58afc75318f780e32b833a5432a0e819a2384882",
        "cf5e80a5f458747470b6f3b31432eb06113650790a56204796f286e3c7ac26f7",
        "5650dc2956ed60e33f1fca9126f57dd8d1cd5c83f01847a4ddc0fb620917ebf1",
        "47a08bd4378ae98c781671f1494b16dd6793c47212327f3b5335a1ad0395d2e8",
        "e13d99fe6a37a22328de208908e81fc53a4ffaa8bd1d49b8bb8261ef9a298d91",
    )
    assert tuple(item.producer_revision for item in protocol.historical_prior_authorities) == (
        "925c2cbc727e3b18e91c0b3981eeed1b470a955a",
        "7c0d86d6eac5ffcfc4a30f2f5fb22884aaa848da",
    )
    assert tuple(len(item.paths) for item in protocol.historical_prior_authorities) == (3, 3)
    assert protocol.current_experiment_authority.tracked_path_count == 27
    assert protocol.current_experiment_authority.current_unique_prompt_digest_count == 1724
    assert protocol.manifest.future_split_deny_authority.exclusion_roles == (
        "masked_lf_whitening_fit", "independent_confirmation", "candidate_selection",
        "calibration", "evaluation",
    )
    assert canonical_digest(asdict(protocol.manifest.future_split_deny_authority)) == (
        protocol.manifest.future_split_deny_authority_digest
    )


def test_roster_authority_and_derived_identity_tamper_fail_closed(tmp_path: Path) -> None:
    manifest = json.loads((ROOT / "configs/experiments/salient_local_lf_mask_write_validation_manifest.json").read_text())
    manifest["entries"][0]["generation_seed"] += 1
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    config = json.loads(CONFIG.read_text())
    config["manifest_path"] = str(target.relative_to(ROOT)) if ROOT in target.parents else str(target)
    config["manifest_file_sha256"] = __import__("hashlib").sha256(target.read_bytes()).hexdigest()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SalientLocalLfMaskWriteProtocolError):
        load_salient_local_lf_mask_write_validation_protocol(config_path, repository_root=ROOT)


def test_future_split_deny_axis_and_digest_tamper_fail_closed(tmp_path: Path) -> None:
    manifest = json.loads((ROOT / "configs/experiments/salient_local_lf_mask_write_validation_manifest.json").read_text())
    manifest["future_split_deny_authority"]["key_lineage_digests"] = manifest[
        "future_split_deny_authority"
    ]["key_lineage_digests"][:-1]
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    config = json.loads(CONFIG.read_text())
    config["manifest_path"] = str(target)
    config["manifest_file_sha256"] = sha256(target.read_bytes()).hexdigest()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SalientLocalLfMaskWriteProtocolError):
        load_salient_local_lf_mask_write_validation_protocol(config_path, repository_root=ROOT)


def test_current_authority_inventory_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["current_experiment_authority"]["paths"][0]["raw_sha256"] = "0" * 64
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SalientLocalLfMaskWriteProtocolError):
        load_salient_local_lf_mask_write_validation_protocol(config_path, repository_root=ROOT)


def test_actual_dtype_budget_uses_exact_binary32_boundary() -> None:
    assert CANONICAL_CONTENT_RELATIVE_L2_LIMIT == struct_binary32(3 / 250)
    assert _actual_dtype_budget_pass(
        budget_status="accepted",
        realized_relative_l2=CANONICAL_CONTENT_RELATIVE_L2_LIMIT,
    ) is True
    assert _actual_dtype_budget_pass(
        budget_status="accepted",
        realized_relative_l2=next_binary32(CANONICAL_CONTENT_RELATIVE_L2_LIMIT),
    ) is False


def test_signed_integer_quality_accepts_exact_boundary_and_rejects_next() -> None:
    accepted = _quality()
    rejected = _quality(over_limit=True)
    assert accepted.squared_code_delta_sum == 786432
    assert accepted.quality_pass is True
    assert rejected.squared_code_delta_sum == 786435
    assert rejected.quality_pass is False
    assert accepted.normalized_mean_squared_error == 1 / 65025
    assert accepted.root_mean_squared_code_delta == 1.0
    with pytest.raises(SalientLocalLfMaskWriteMetricError):
        observe_public_rgb8_quality(
            torch.zeros((1, 3, 512, 512), dtype=torch.uint8),
            torch.zeros((1, 3, 512, 512), dtype=torch.uint8),
            clean_image_digest="0" * 64,
            marked_image_digest="0" * 64,
        )


def test_quality_violation_is_complete_scientific_negative_not_failure() -> None:
    observations = [_observation(index, quality_pass=index != 7) for index in range(8)]
    aggregate = aggregate_salient_local_lf_mask_write_validation(observations, ())
    assert aggregate.successful_observation_count == 8
    assert aggregate.quality_success_count == 7
    assert aggregate.module_outcome == "mechanism_signal_not_observed"
    assert aggregate.candidate_recommendation == "candidate_not_recommended"
    assert aggregate.allow_request_for_independent_masked_lf_null_fit is False


def test_mechanism_requires_seven_of_eight_and_quality_requires_eight_of_eight() -> None:
    passing = [_observation(index, mechanism=index != 7) for index in range(8)]
    assert aggregate_salient_local_lf_mask_write_validation(passing, ()).allow_request_for_independent_masked_lf_null_fit
    failing = [_observation(index, mechanism=index not in {6, 7}) for index in range(8)]
    assert not aggregate_salient_local_lf_mask_write_validation(failing, ()).allow_request_for_independent_masked_lf_null_fit
    with pytest.raises(SalientLocalLfMaskWriteMetricError):
        aggregate_salient_local_lf_mask_write_validation(passing[:7], ())


def test_failure_priority_and_fixed_denominator_are_stable() -> None:
    observations = [_observation(index) for index in range(6)]
    failures = (
        SalientLocalLfTerminalFailure(6, "resource_failure", "resource_failure"),
        SalientLocalLfTerminalFailure(7, "implementation_failure", "implementation_failure"),
    )
    aggregate = aggregate_salient_local_lf_mask_write_validation(observations, failures)
    assert aggregate.module_outcome == "implementation_blocked"
    assert aggregate.scientific_denominator == 8


def test_scientific_failure_classification_is_exact_and_prioritized() -> None:
    assert _classify_scientific_failure(SalientLocalLfMaskWriteIdentityError("private")) == (
        "identity_failure", "salient_local_lf_public_observation_identity_drift",
    )
    assert _classify_scientific_failure(SalientLocalLfMaskWriteIntegrityError("private")) == (
        "integrity_failure", "salient_local_lf_public_materialization_integrity_drift",
    )
    assert _classify_scientific_failure(RuntimeError("private"))[0] == "implementation_failure"
    assert _classify_scientific_failure(MemoryError("private"))[0] == "resource_failure"
    assert _classify_scientific_failure(OSError("private"))[0] == "environment_failure"


def test_scientific_record_roundtrip_preserves_typed_observation() -> None:
    runner = _runner()
    record = runner._scientific_record(
        unit_index=2, attempt_index=0, elapsed=0.25, observation=_observation(0),
    )
    replay = DevelopmentScientificRecord.from_payload(json.loads(json.dumps(record.payload())))
    assert replay == record
    assert replay.operation_result_payload["mask_write_observation"]["quality"]["squared_code_delta_sum"] == 786432
    assert replay.metric_observation["source_cluster_id"] == runner.protocol.analysis_identity(2).source_cluster_id


def test_real_persistent_store_commits_recovers_and_replays_fixed_ten(
    tmp_path: Path,
) -> None:
    runner = _runner()
    worker = FrozenWorkerIdentity(
        revision=runner.method_code_revision,
        protocol_digest=runner.protocol_digest,
        execution_intent_authority_digest=runner.execution_intent_authority_digest,
        input_manifest_digest=runner.protocol.manifest.digest(),
        candidate_config_digest=runner.candidate_config_digest,
        unit_roster_digest=runner.protocol.unit_roster_digest,
    )
    store = DevelopmentPersistentStore(
        tmp_path, run_id=runner.protocol.run_id, worker_identity=worker,
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    lease = store.acquire_lease(
        session_id="salient_local_lf_persistence_session",
        now_epoch_seconds=100, lease_duration_seconds=1000,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=100)
    for unit_index in range(10):
        intent = store.create_session_intent(cursor, lease, now_epoch_seconds=101 + unit_index * 2)
        if unit_index < 2:
            record = runner._operational_record(
                unit_index=unit_index,
                operation={
                    "operational_role": "environment_runtime_throughput_preflight",
                    "case_ids": [f"salient_local_lf_operational_preflight_{unit_index}"],
                    "responsibility_result_digests": [["content_embedder", f"{unit_index + 80:064x}"]],
                    "runtime_config_digest": f"{unit_index + 90:064x}",
                    "counts_as_scientific_coverage": False,
                    "scientific_claims_supported": False,
                },
                elapsed=0.25, attempt_index=0,
            )
        else:
            entry = runner.protocol.manifest.entries[unit_index - 2]
            record = runner._scientific_record(
                unit_index=unit_index, attempt_index=0, elapsed=0.25,
                observation=_observation(
                    unit_index - 2, source_cluster_id=entry.source_cluster_id,
                ),
            )
        marker = store.commit_session_unit(
            cursor, lease, intent, record=record,
            raw_secret_values=("salient-mask-write-test-root", runner.registered_root_key),
            now_epoch_seconds=102 + unit_index * 2,
        )
        assert marker.unit_index == unit_index
        assert marker.attempt_index == 0
    reopened = DevelopmentPersistentStore(
        tmp_path, run_id=runner.protocol.run_id, worker_identity=worker,
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    recovery = reopened.recover(now_epoch_seconds=500)
    evidence = reopened.verified_terminal_scientific_evidence_for_unit_indexes(
        tuple(range(2, 10)), now_epoch_seconds=500,
    )
    aggregate = runner.replay_aggregate(evidence)
    assert len(recovery.committed_units) == 10
    assert tuple(item.unit_index for item in recovery.committed_units) == tuple(range(10))
    assert aggregate.successful_observation_count == 8
    assert aggregate_supports_scientific_claim(aggregate) is True
    first_record, first_marker = evidence[0]
    with pytest.raises(SalientLocalLfMaskWriteRunnerError):
        runner.replay_aggregate(((first_record, replace(first_marker, unit_id="development_unit_0003")), *evidence[1:]))


def test_safe_failure_is_package_relative_bounded_and_secret_free() -> None:
    secret = "registered-root-secret-value"
    try:
        raise RuntimeError(secret + " /content/drive/private/checkpoint")
    except RuntimeError as exc:
        diagnostic = _safe_failure(exc, repository=ROOT,
                                   operation_identity="salient_local_lf_test_execution", unit_index=2)
    encoded = json.dumps(diagnostic)
    assert secret not in encoded
    assert "/content/" not in encoded
    assert len(diagnostic["failure_message_redacted"].encode()) <= 512
    assert len(diagnostic["package_relative_frames"]) <= 8


def test_overlay_remains_unqualified_until_cumulative_delivery() -> None:
    state = __import__("yaml").safe_load((ROOT / ".codex/research_state/salient_local_lf_candidate_readiness.yaml").read_text())
    assert state["source_cpu_api_implementation_ready"] is True
    assert state["rgb_quality_gate_defined"] is False
    assert state["experiment_protocol_admitted"] is False
    assert state["candidate_runtime_qualified"] is False
    assert state["scientific_mechanism_validated"] is False


def test_sixty_seven_distribution_lock_is_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    lock = (ROOT / "requirements_inspyrenet_salient_local_lf_gpu_execution.txt").read_text().splitlines()
    versions = dict(line.split("==", 1) for line in lock if line and not line.startswith("#"))
    monkeypatch.setattr(
        "scripts.experiment_execution.salient_local_lf_mask_write_validation_server.metadata.version",
        lambda name: versions[name],
    )
    assert len(versions) == 67
    assert _verify_locked_dependencies(ROOT) == "855f73f7cb79cc9b9ec5f4d5a62b17cafc336866836601360882bd1cbaa3568b"
