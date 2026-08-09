"""Frozen identity checks for LF whitened directional validation."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.metrics.lf_whitened_directional_validation import (
    aggregate_lf_whitened_direction,
    create_lf_whitened_directional_observation,
)
from experiments.metrics.lf_whitened_score_screening import (
    fit_lf_null_whitening_asset,
)
from experiments.protocol.development_records import DevelopmentOperationalRecord

from experiments.protocol.hf_only_detector_directional_validation import (
    load_authority_deny_axes,
)
from experiments.protocol.lf_whitened_directional_validation import (
    FUTURE_SPLIT_EXCLUSION_ROLES,
    LF_DIRECTIONAL_COMPONENT_IDS,
    PRACTICAL_MARGIN_FLOOR,
    LfWhitenedDirectionalProtocolError,
    canonical_digest,
    derive_lf_whitened_directional_analysis_identity,
    load_lf_whitened_directional_validation_protocol,
)
from experiments.protocol.lf_whitened_score_screening import (
    load_lf_whitened_score_screening_protocol,
)
from experiments.runners.development_persistence import (
    DevelopmentPersistentStore,
    FrozenWorkerIdentity,
)
from experiments.runners.lf_whitened_directional_validation import (
    LfWhitenedDirectionalValidationRunner,
)
from main import LfNullWhiteningAsset, identify_root_key
from runtime import create_runtime_adapter
from scripts.experiment_execution.lf_whitened_directional_validation_entrypoint import (
    _derive_registered_experiment_root,
)
from scripts.experiment_execution.component_source_closure import (
    build_component_source_closure,
)
from tests.unit.test_runtime_content_write_and_vae import FakeContentBackend


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/lf_whitened_directional_validation.json"
READINESS = ROOT / ".codex/research_state/method_readiness.yaml"
SCREENING_CONFIG = ROOT / "configs/experiments/lf_whitened_score_screening.json"
COMPONENTS = ROOT / "configs/experiments/internal_execution_components.json"
ROOT_KEY = "ceg-wm-lf-whitened-directional-test-key"


def _load_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text("utf-8"))


def _write_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@pytest.mark.unit
def test_lf_whitened_directional_protocol_freezes_budget_controls_and_gate() -> None:
    protocol, manifest = load_lf_whitened_directional_validation_protocol(
        CONFIG, repository_root=ROOT
    )

    assert protocol.operational_unit_count == 1
    assert protocol.scientific_cluster_count == 32
    assert protocol.maximum_total_units == 33
    assert tuple(unit.unit_index for unit in protocol.unit_roster) == tuple(
        range(33)
    )
    assert protocol.unit_roster[0].responsibility_id == (
        "lf_whitened_detector_runtime_preflight"
    )
    assert {
        unit.responsibility_id for unit in protocol.unit_roster[1:]
    } == {"lf_detector"}
    assert protocol.wrong_key_roster_size == 4
    assert protocol.practical_margin_floor == PRACTICAL_MARGIN_FLOOR
    assert (
        protocol.content_relative_l2_numerator,
        protocol.content_relative_l2_denominator,
    ) == (3, 250)
    assert protocol.minimum_registered_minus_null_success_count == 28
    assert protocol.minimum_registered_minus_max_wrong_success_count == 28
    assert protocol.confidence_level == 0.95
    assert protocol.confidence_lower_bound_requirement == (
        "strictly_greater_than_one_half"
    )
    assert protocol.passing_module_outcome == "mechanism_signal_observed"
    assert protocol.passing_candidate_recommendation == (
        "candidate_worth_further_selection"
    )
    assert "no_threshold" in protocol.claim_boundary
    assert "no_fpr" in protocol.claim_boundary
    assert "no_promotion" in protocol.claim_boundary
    assert len(manifest.entries) == 32
    assert {item.split for item in manifest.entries} == {"development"}
    assert {item.role_id for item in manifest.entries} == {
        "lf_whitened_directional_validation"
    }
    assert protocol.future_split_exclusion_roles == FUTURE_SPLIT_EXCLUSION_ROLES


@pytest.mark.unit
def test_lf_whitened_directional_manifest_is_disjoint_from_prior_authorities() -> None:
    protocol, manifest = load_lf_whitened_directional_validation_protocol(
        CONFIG, repository_root=ROOT
    )
    prior = load_authority_deny_axes(
        protocol.prior_development_manifests, ROOT
    )

    assert {item.prompt_digest for item in manifest.entries}.isdisjoint(
        prior.prompt_digests
    )
    assert {
        manifest.source_cluster_namespace,
        *(item.cluster_identity for item in manifest.entries),
    }.isdisjoint(prior.source_cluster_identities)
    assert {manifest.seed_namespace}.isdisjoint(prior.seed_namespaces)
    assert {item.generation_seed for item in manifest.entries}.isdisjoint(
        prior.generation_seeds
    )
    assert {
        manifest.image_lineage_namespace,
        *(item.image_lineage_identity for item in manifest.entries),
        *(item.image_lineage_digest for item in manifest.entries),
    }.isdisjoint(prior.image_lineage_identities)
    assert {
        manifest.key_family_namespace,
        protocol.registered_key_derivation_identity,
        protocol.wrong_key_control_identity,
    }.isdisjoint(prior.key_control_identities)


@pytest.mark.unit
def test_lf_whitened_directional_component_authority_replays_reviewed_sources() -> None:
    protocol, _manifest = load_lf_whitened_directional_validation_protocol(
        CONFIG, repository_root=ROOT
    )
    readiness = json.loads(READINESS.read_text("utf-8"))
    closure = build_component_source_closure(
        LF_DIRECTIONAL_COMPONENT_IDS,
        readiness["components"],
        ROOT,
    )

    assert protocol.ordered_component_ids == closure.ordered_component_ids
    assert tuple(asdict(item) for item in protocol.component_source_bindings) == (
        tuple(asdict(item) for item in closure.source_bindings)
    )
    assert (
        protocol.component_implementation_digest
        == closure.component_implementation_digest
    )
    assert protocol.candidate_specification_sha256 == sha256(
        (ROOT / "docs/design/candidate_specifications.md").read_bytes()
    ).hexdigest()
    assert protocol.method_review_reference == (
        readiness["independent_semantic_review"]["review_reference"]
    )
    assert protocol.method_reviewed_revision == (
        readiness["independent_semantic_review"][
            "reviewed_repository_revision"
        ]
    )


@pytest.mark.unit
def test_lf_whitening_asset_fit_authority_remains_bound_to_completed_producer() -> None:
    protocol, _manifest = load_lf_whitened_directional_validation_protocol(
        CONFIG, repository_root=ROOT
    )
    screening, _fit, _screen = load_lf_whitened_score_screening_protocol(
        SCREENING_CONFIG,
        repository_root=ROOT,
    )

    assert protocol.whitening_asset_fit_producer_revision == (
        "a78c47184cf83ad351bb4442ebd31c218726de25"
    )
    assert protocol.whitening_asset_fit_identity == screening.protocol_id
    assert protocol.whitening_asset_fit_run_id == screening.run_id
    assert protocol.whitening_asset_fit_protocol_digest == screening.digest()
    assert protocol.whitening_null_fit_manifest_file_sha256 == sha256(
        (ROOT / screening.null_fit_manifest_path).read_bytes()
    ).hexdigest()


@pytest.mark.unit
def test_lf_whitened_directional_analysis_identities_cover_unique_clusters() -> None:
    _protocol, manifest = load_lf_whitened_directional_validation_protocol(
        CONFIG, repository_root=ROOT
    )
    identities = tuple(
        derive_lf_whitened_directional_analysis_identity(
            entry,
            manifest,
            key_family_digest="f" * 64,
        )
        for entry in manifest.entries
    )

    assert len({item.unit_id for item in identities}) == 32
    assert len({item.source_cluster_id for item in identities}) == 32
    assert {item.generation_seed for item in identities} == {
        item.generation_seed for item in manifest.entries
    }


@pytest.mark.unit
def test_lf_whitened_directional_protocol_rejects_gate_or_component_drift(
    tmp_path: Path,
) -> None:
    payload = _load_config()
    payload["minimum_registered_minus_null_success_count"] = 27
    gate_path = tmp_path / "gate.json"
    _write_config(gate_path, payload)
    with pytest.raises(
        LfWhitenedDirectionalProtocolError, match="scientific gate"
    ):
        load_lf_whitened_directional_validation_protocol(
            gate_path, repository_root=ROOT
        )

    payload = _load_config()
    bindings = payload["component_source_bindings"]
    assert type(bindings) is list
    detector_binding = bindings[3]
    assert type(detector_binding) is dict
    detector_binding["source_sha256"] = "0" * 64
    closure_path = tmp_path / "closure.json"
    _write_config(closure_path, payload)
    with pytest.raises(
        LfWhitenedDirectionalProtocolError, match="implementation digest"
    ):
        load_lf_whitened_directional_validation_protocol(
            closure_path, repository_root=ROOT
        )


@pytest.mark.unit
def test_lf_whitened_directional_protocol_rejects_prior_prompt_reuse(
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        (ROOT / "configs/experiments/lf_whitened_directional_validation_manifest.json").read_text(
            "utf-8"
        )
    )
    prior = json.loads(
        (ROOT / "configs/experiments/lf_transmission_diagnostic_manifest.json").read_text(
            "utf-8"
        )
    )
    prior_prompt = prior["entries"][0]["prompt"]
    manifest["entries"][0]["prompt"] = prior_prompt
    manifest["entries"][0]["prompt_digest"] = sha256(
        prior_prompt.encode("utf-8")
    ).hexdigest()
    manifest_path = tmp_path / "overlap_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    payload = _load_config()
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_file_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    config_path = tmp_path / "overlap_config.json"
    _write_config(config_path, payload)

    with pytest.raises(LfWhitenedDirectionalProtocolError, match="overlaps"):
        load_lf_whitened_directional_validation_protocol(
            config_path, repository_root=ROOT
        )


def _asset() -> LfNullWhiteningAsset:
    fit = fit_lf_null_whitening_asset(
        tuple(
            tuple(float((cluster + 1) * (index + 1)) for index in range(96))
            for cluster in range(32)
        ),
        fit_manifest_sha256="a" * 64,
    )
    return LfNullWhiteningAsset.from_canonical_payload(
        fit.canonical_payload,
        whitening_asset_digest=fit.whitening_asset_digest,
    )


def _lf_base_latent() -> torch.Tensor:
    return torch.linspace(
        -1.0,
        1.0,
        steps=16 * 64 * 64,
        dtype=torch.float32,
    ).reshape(1, 16, 64, 64).to(torch.float16)


def _directional_runner() -> tuple[
    LfWhitenedDirectionalValidationRunner,
    object,
]:
    protocol, manifest = load_lf_whitened_directional_validation_protocol(
        CONFIG, repository_root=ROOT
    )
    runtime = create_runtime_adapter(
        FakeContentBackend(
            callback_sequences=tuple(tuple(range(20)) for _ in range(8))
        )  # type: ignore[arg-type]
    )
    runtime.initialize("cpu")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS)
    )
    registered_root = _derive_registered_experiment_root(
        ROOT_KEY,
        protocol_digest=protocol.digest(),
        manifest_digest=manifest.digest(),
        key_family_namespace=manifest.key_family_namespace,
    )
    public_root = identify_root_key(registered_root).root_key_public_digest
    return (
        LfWhitenedDirectionalValidationRunner(
            protocol=protocol,
            manifest=manifest,
            adapter=adapter,
            runtime_adapter=runtime,
            whitening_asset=_asset(),
            method_code_revision="a" * 40,
            run_id=protocol.run_id,
            registered_root_key=registered_root,
            root_key_public_digest=public_root,
            protocol_digest=protocol.digest(),
            execution_intent_authority_digest="b" * 64,
            candidate_config_digest="c" * 64,
        ),
        runtime,
    )


@pytest.mark.unit
def test_lf_whitened_directional_runner_uses_public_detector_and_four_wrong_controls() -> None:
    runner, runtime = _directional_runner()
    operational = runner.execute_operational_smoke(
        base_latent=_lf_base_latent(),
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
    )
    scientific = runner.execute_scientific_cluster(
        cluster_ordinal=0,
        base_latent=_lf_base_latent(),
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
    )
    runtime.close()

    assert type(operational) is DevelopmentOperationalRecord
    assert operational.counts_as_scientific_coverage is False
    result = scientific.operation_result_payload["directional_observation"]
    assert len(result["wrong_key_scores"]) == 4
    assert scientific.detector_trace["public_callable"] == (
        "main.lf_null_whitened_matched_detector"
    )
    assert scientific.detector_trace["same_image_registered_four_wrong_reuse"] is True
    assert scientific.detector_trace["paired_clean_primary_null"] is True
    assert scientific.detector_trace["reference_image_used"] is False
    assert scientific.detector_trace["embed_record_used"] is False
    assert scientific.detector_trace["private_latent_used_by_detector"] is False
    assert scientific.threshold_trace["raw_threshold_identity"] is None
    assert scientific.module_outcome is None
    assert scientific.candidate_recommendation is None


def _metric_observation(index: int, *, passed: bool):
    registered = 0.4 if passed else 0.0
    return create_lf_whitened_directional_observation(
        cluster_ordinal=index,
        registered_score=registered,
        primary_null_score=0.0,
        wrong_key_scores=(0.1, 0.09, 0.08, 0.07),
        candidate_observation_digest=canonical_digest({"candidate": index}),
        clean_observation_digest=canonical_digest({"clean": index}),
        registered_detector_identity="d" * 64,
        primary_null_detector_identity="d" * 64,
        wrong_key_detector_identities=("d" * 64,) * 4,
        detector_config_digest="e" * 64,
        observation_protocol="final_image_vae_posterior_mode",
        whitening_asset_digest="f" * 64,
        registered_template_digest=canonical_digest({"registered": index}),
        primary_null_template_digest=canonical_digest({"registered": index}),
        wrong_key_template_digests=tuple(
            canonical_digest({"wrong": wrong, "cluster": index})
            for wrong in range(4)
        ),
        registered_root_key_public_digest="1" * 64,
        wrong_key_indexes=(0, 1, 2, 3),
        materialization_integrity_status="passed",
        materialization_budget_status="accepted",
        realized_relative_l2=0.01,
        content_relative_l2_limit=3 / 250,
        actual_runtime_dtype="torch.float16",
    )


@pytest.mark.unit
def test_lf_whitened_directional_metric_keeps_failures_in_frozen_denominator() -> None:
    passing = aggregate_lf_whitened_direction(
        tuple(_metric_observation(index, passed=True) for index in range(32)),
        failed_cluster_count=0,
    )
    failed = aggregate_lf_whitened_direction(
        tuple(_metric_observation(index, passed=True) for index in range(27)),
        failed_cluster_count=5,
    )

    assert passing.directional_validation_passed is True
    assert passing.module_outcome == "mechanism_signal_observed"
    assert passing.candidate_recommendation == "candidate_worth_further_selection"
    assert passing.registered_minus_max_wrong.exact_one_sided_confidence_lower_bound > 0.5
    assert failed.directional_validation_passed is False
    assert failed.expected_cluster_count == 32
    assert failed.failed_cluster_count == 5
    assert failed.registered_minus_max_wrong.observation_count == 32


@pytest.mark.unit
def test_lf_whitened_directional_persistence_commits_recovers_and_preserves_retry_lineage(
    tmp_path: Path,
) -> None:
    runner, runtime = _directional_runner()
    identity = FrozenWorkerIdentity(
        revision=runner.method_code_revision,
        protocol_digest=runner.protocol_digest,
        execution_intent_authority_digest=runner.execution_intent_authority_digest,
        input_manifest_digest=runner.manifest.digest(),
        candidate_config_digest=runner.candidate_config_digest,
        unit_roster_digest=runner.protocol.unit_roster_digest,
    )
    store = DevelopmentPersistentStore(
        tmp_path,
        run_id=runner.run_id,
        worker_identity=identity,
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    lease = store.acquire_lease(
        session_id="lf_directional_persistence_session",
        now_epoch_seconds=100,
        lease_duration_seconds=10000,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=100)
    intent = store.create_session_intent(cursor, lease, now_epoch_seconds=101)
    record = runner.execute_operational_smoke(
        base_latent=_lf_base_latent(),
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
    )
    store.commit_session_unit(
        cursor,
        lease,
        intent,
        record=record,
        raw_secret_values=(ROOT_KEY, runner.registered_root_key),
        now_epoch_seconds=102,
    )
    success_intent = store.create_session_intent(cursor, lease, now_epoch_seconds=103)
    success = runner.execute_scientific_cluster(
        cluster_ordinal=0,
        base_latent=_lf_base_latent(),
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
    )
    store.commit_session_unit(
        cursor,
        lease,
        success_intent,
        record=success,
        raw_secret_values=(ROOT_KEY, runner.registered_root_key),
        now_epoch_seconds=104,
    )
    retry_intent = store.create_session_intent(cursor, lease, now_epoch_seconds=105)
    retry_record = runner.create_failed_scientific_record(
        cluster_ordinal=1,
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
        actual_elapsed_seconds=1.0,
        failure_type="builtins.MemoryError",
        resource_failure=True,
        failure_category="resource_failure",
    )
    marker = store.commit_session_unit(
        cursor,
        lease,
        retry_intent,
        record=retry_record,
        raw_secret_values=(ROOT_KEY, runner.registered_root_key),
        now_epoch_seconds=106,
    )
    assert marker.attempt_disposition == "retryable_resource_failure"
    second_intent = store.create_session_intent(cursor, lease, now_epoch_seconds=107)
    assert second_intent.attempt_index == 1
    assert second_intent.parent_attempt_intent_digest == retry_intent.digest()
    terminal = runner.create_failed_scientific_record(
        cluster_ordinal=1,
        attempt_index=1,
        retry_parent_intent_digest=retry_intent.digest(),
        maximum_duration_seconds=2700,
        actual_elapsed_seconds=1.0,
        failure_type="builtins.MemoryError",
        resource_failure=True,
        failure_category="resource_failure",
    )
    store.commit_session_unit(
        cursor,
        lease,
        second_intent,
        record=terminal,
        raw_secret_values=(ROOT_KEY, runner.registered_root_key),
        now_epoch_seconds=108,
    )
    recovery = store.recover(now_epoch_seconds=109)
    runtime.close()

    assert cursor.next_unit_index == 3
    assert len(recovery.committed_units) == 4
    assert recovery.committed_units[-1].attempt_disposition == "final_failure"
