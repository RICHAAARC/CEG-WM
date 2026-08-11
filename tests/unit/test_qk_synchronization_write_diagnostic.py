"""Frozen protocol and metric tests for Q/K synchronization-write diagnosis."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import inspect
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional
from torch.utils.checkpoint import (
    CheckpointError,
    checkpoint as activation_checkpoint,
)

from experiments.methods import (
    CegWmExperimentAdapter,
    CegWmExperimentAdapterError,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.metrics.qk_synchronization_write_diagnostic import (
    QkSynchronizationWriteMetricError,
    QkTerminalFailure,
    aggregate_qk_ratio_probes,
    aggregate_qk_synchronization_diagnosis,
    create_qk_ratio_probe_observation,
    create_qk_rgb8_quality_delta,
    create_qk_transform_dependency_blocked_terminal,
    create_qk_transformed_relation_observation,
)
from experiments.protocol.hf_only_detector_directional_validation import (
    load_authority_deny_axes,
)
from experiments.protocol.development_records import (
    DEVELOPMENT_CLAIM_BOUNDARY,
    OPERATIONAL_RECORD_COLLECTION_ROLE,
    OPERATIONAL_RECORD_KIND,
    OPERATIONAL_RECORD_SCHEMA,
    DevelopmentOperationalRecord,
    DevelopmentRecordError,
    canonical_development_value_digest,
)
from experiments.protocol.qk_synchronization_write_diagnostic import (
    CLAIM_BOUNDARY,
    GEOMETRY_RATIO_ROSTER,
    QkSynchronizationWriteProtocolError,
    TRANSFORM_PROBE_ROSTER,
    derive_qk_synchronization_analysis_identity,
    load_qk_synchronization_write_protocol,
)
from experiments.runners.qk_synchronization_write_diagnostic import (
    QkSynchronizationWriteDiagnosticRunner,
    RGB8_MEMBER_PATH,
)
from experiments.runners.development_persistence import (
    DevelopmentPersistenceError,
    DevelopmentPersistentStore,
    FrozenWorkerIdentity,
)
from runtime import RuntimeAdapterError, Sd35RuntimeAdapter, create_runtime_adapter
from runtime import sd35_backend as sd35_backend_module
from scripts.experiment_execution.qk_synchronization_write_diagnostic_entrypoint import (
    QkSynchronizationWriteEntrypointError,
    _authorized_persistence_bindings,
    _failure_diagnostic,
    _is_resource_failure,
    _qualified_exception_type_chain,
    _runtime_failure_safe_attribution,
    _selected_rgb8,
    execute_qk_synchronization_write_diagnostic_session,
)
from tests.unit.test_runtime_qk_observation import (
    FakeGeometrySynchronizationBackend,
    FakePosterior,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/qk_synchronization_write_diagnostic.json"
COMPONENTS = ROOT / "configs/experiments/internal_execution_components.json"


class _SpatialGeometrySynchronizationBackend(FakeGeometrySynchronizationBackend):
    def __init__(self) -> None:
        super().__init__()
        self.generation_call_count = 0
        self.public_detection_images: list[torch.Tensor] = []

    @staticmethod
    def _decoded_image(latent: torch.Tensor) -> torch.Tensor:
        spatial = latent.to(dtype=torch.float32)[:, :3]
        return torch.sigmoid(
            functional.interpolate(
                spatial,
                size=(512, 512),
                mode="bilinear",
                align_corners=True,
            )
        )

    @staticmethod
    def _encoded_latent(image: torch.Tensor) -> torch.Tensor:
        return functional.adaptive_avg_pool2d(
            image.to(dtype=torch.float32)[:, :2],
            (4, 4),
        )

    def run_generation(self, initial_latent, callback):
        self.generation_call_count += 1
        return super().run_generation(initial_latent, callback)

    def vae_encode(self, image: torch.Tensor) -> FakePosterior:
        self.public_detection_images.append(image.detach().cpu().clone())
        return FakePosterior(self._encoded_latent(image))

    def vae_encode_differentiable(
        self, image: torch.Tensor
    ) -> FakePosterior:
        return FakePosterior(
            self._encoded_latent(image),
            preserve_gradient=True,
        )


def _public_chain_runner():
    protocol, manifest = load_qk_synchronization_write_protocol(
        CONFIG, repository_root=ROOT
    )
    backend = _SpatialGeometrySynchronizationBackend()
    runtime = create_runtime_adapter(backend)
    runtime.initialize("cpu")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS),
        runtime_adapter=runtime,
    )
    runner = QkSynchronizationWriteDiagnosticRunner(
        protocol=protocol,
        manifest=manifest,
        adapter=adapter,
        runtime_adapter=runtime,
        method_code_revision="1" * 40,
        run_id=protocol.run_id,
        content_registered_root_key="qk-public-chain-content-root",
        geometry_registered_root_key="qk-public-chain-geometry-root",
        protocol_digest=protocol.digest(),
        execution_intent_authority_digest="2" * 64,
        candidate_config_digest="3" * 64,
        package_identity="4" * 64,
    )
    return runner, runtime, backend


def _runner() -> QkSynchronizationWriteDiagnosticRunner:
    protocol, manifest = load_qk_synchronization_write_protocol(
        CONFIG, repository_root=ROOT
    )
    runtime = object.__new__(Sd35RuntimeAdapter)
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS),
        runtime_adapter=runtime,
    )
    return QkSynchronizationWriteDiagnosticRunner(
        protocol=protocol,
        manifest=manifest,
        adapter=adapter,
        runtime_adapter=runtime,
        method_code_revision="1" * 40,
        run_id=protocol.run_id,
        content_registered_root_key="qk-diagnosis-content-test-root",
        geometry_registered_root_key="qk-diagnosis-geometry-test-root",
        protocol_digest=protocol.digest(),
        execution_intent_authority_digest="2" * 64,
        candidate_config_digest="3" * 64,
        package_identity="4" * 64,
    )


def _valid_qk_operational_record() -> DevelopmentOperationalRecord:
    operation = {
        "operational_role": "public_qk_synchronization_write_smoke",
        "source_cluster_ordinal": 0,
        "case_ids": ["qk_synchronization_write_public_runtime_smoke"],
        "responsibility_result_digests": [["qk_geometry_sync", "4" * 64]],
        "elapsed_seconds": 1.0,
        "runtime_config_digest": "5" * 64,
        "counts_as_scientific_coverage": False,
        "scientific_claims_supported": False,
    }
    record = DevelopmentOperationalRecord(
        schema_version=OPERATIONAL_RECORD_SCHEMA,
        collection_role=OPERATIONAL_RECORD_COLLECTION_ROLE,
        record_kind=OPERATIONAL_RECORD_KIND,
        record_id="0" * 64,
        run_id="qk_synchronization_write_operational_record_test",
        protocol_digest="1" * 64,
        method_code_revision="2" * 40,
        unit_index=0,
        phase="development_environment_preflight",
        source_cluster_ordinal=0,
        candidate_config_digest="3" * 64,
        attempt_index=0,
        retry_parent_intent_digest=None,
        actual_elapsed_seconds=1.0,
        maximum_duration_seconds=2700,
        operation_result_payload=operation,
        counts_as_scientific_coverage=False,
        scientific_claims_supported=False,
        scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
    )
    return replace(
        record,
        record_id=canonical_development_value_digest(
            record.payload_without_record_id()
        ),
    )


def _accepted_ratio(
    cluster: int,
    ratio_identity: str,
    ratio: float,
    *,
    eligible: bool = True,
    content_only_rgb8_digest: str | None = None,
    geometry_written_rgb8_digest: str | None = None,
):
    registered_post = 0.14 if eligible else 0.105
    content_digest = (
        f"content-{cluster}"
        if content_only_rgb8_digest is None
        else content_only_rgb8_digest
    )
    geometry_digest = (
        f"geometry-{cluster}-{ratio_identity}"
        if geometry_written_rgb8_digest is None
        else geometry_written_rgb8_digest
    )
    return create_qk_ratio_probe_observation(
        cluster_ordinal=cluster,
        ratio_identity=ratio_identity,
        geometry_ratio=ratio,
        write_accepted=True,
        line_search_factor=0.5,
        ste_acceptance_baseline_score=0.2,
        ste_acceptance_score=0.21,
        public_pre_registered_score=0.1,
        public_pre_wrong_key_scores=(0.01, 0.02, 0.03, 0.04),
        public_post_registered_score=registered_post,
        public_post_wrong_key_scores=(0.015, 0.025, 0.035, 0.045),
        actual_geometry_relative_l2=0.0005,
        actual_total_relative_l2=0.012,
        content_span_projection_relative=0.00001,
        rgb8_quality_delta=create_qk_rgb8_quality_delta(
            relative_l2=0.004,
            mean_squared_error=3.0,
            content_only_rgb8_digest=content_digest,
            geometry_written_rgb8_digest=geometry_digest,
        ),
        public_pre_observation_identity=(
            f"public_rgb8_vae_qk_pre_{cluster}_{ratio_identity}"
        ),
        public_post_observation_identity=(
            f"public_rgb8_vae_qk_post_{cluster}_{ratio_identity}"
        ),
        content_only_rgb8_digest=content_digest,
        geometry_written_rgb8_digest=geometry_digest,
        geometry_key_family_digest="a" * 64,
        registered_template_digest="b" * 64,
        wrong_key_template_digests=("c" * 64, "d" * 64, "e" * 64, "f" * 64),
        wrong_key_indexes=(0, 1, 2, 3),
        method_identity="main.geometry_synchronization_write_and_qk_geometry_sync",
        runtime_identity="runtime.public_suffix_and_image_only_qk_observation",
        runtime_config_digest="1" * 64,
        model_revision="2" * 40,
        package_identity="3" * 64,
        identity_violation_count=0,
        budget_violation_count=0,
        integrity_violation_count=0,
        nonfinite_violation_count=0,
    )


def _ratio_matrix(*, first_eligible_ratio_index: int | None):
    return tuple(
        _accepted_ratio(
            cluster,
            ratio_identity,
            ratio,
            eligible=(
                first_eligible_ratio_index is not None
                and ratio_index >= first_eligible_ratio_index
            ),
        )
        for ratio_index, (ratio_identity, ratio) in enumerate(
            GEOMETRY_RATIO_ROSTER
        )
        for cluster in range(4)
    )


def _transforms(
    selected_ratio_identity: str,
    *,
    registered_score: float = 0.2,
    wrong_key_scores: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04),
):
    return tuple(
        create_qk_transformed_relation_observation(
            cluster_ordinal=cluster,
            transform_identity=transform_identity,
            selected_ratio_identity=selected_ratio_identity,
            source_geometry_written_rgb8_digest=f"source-{cluster}",
            transformed_rgb8_digest=f"transformed-{cluster}-{transform_identity}",
            registered_score=registered_score,
            wrong_key_scores=wrong_key_scores,
            public_observation_identity="public_image_only_qk_observation",
            method_identity="main.qk_geometry_sync",
            runtime_identity="runtime.public_rgb8_vae_qk_observation",
            identity_violation_count=0,
            integrity_violation_count=0,
            nonfinite_violation_count=0,
        )
        for transform_identity, *_ in TRANSFORM_PROBE_ROSTER
        for cluster in range(4)
    )


def _dependency_blocked_terminals():
    return tuple(
        create_qk_transform_dependency_blocked_terminal(
            cluster_ordinal=cluster,
            transform_identity=transform_identity,
        )
        for transform_identity, *_ in TRANSFORM_PROBE_ROSTER
        for cluster in range(4)
    )


@pytest.mark.unit
def test_qk_diagnosis_protocol_freezes_roster_order_controls_and_boundary(
    tmp_path: Path,
) -> None:
    protocol, manifest = load_qk_synchronization_write_protocol(
        CONFIG, repository_root=ROOT
    )
    config_payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert protocol.run_id == (
        "ceg_wm_qk_differentiable_vae_checkpoint_resource_qualification"
    )
    assert config_payload["run_id"] == protocol.run_id
    assert protocol.schema_version == (
        "ceg_wm_qk_synchronization_write_diagnosis_protocol"
    )
    assert protocol.protocol_id == "ceg_wm_qk_synchronization_write_diagnosis"
    assert protocol.protocol_version == "1.0.0"
    assert protocol.role_id == "qk_synchronization_write_diagnosis"
    assert protocol.candidate_identity == "qk_relation_similarity"
    assert protocol.routing_mode == "routing_disabled"
    assert protocol.content_branch_id == "hf_only"
    assert protocol.operational_unit_count == 1
    assert protocol.ratio_probe_unit_count == 12
    assert protocol.transform_probe_unit_count == 16
    assert protocol.scientific_unit_count == 28
    assert protocol.maximum_total_units == 29
    assert tuple(unit.unit_index for unit in protocol.unit_roster) == tuple(range(29))
    assert tuple(
        unit.geometry_case_id for unit in protocol.unit_roster[1:13]
    ) == tuple(name for name, _ in GEOMETRY_RATIO_ROSTER for _ in range(4))
    assert tuple(
        unit.geometry_case_id for unit in protocol.unit_roster[13:]
    ) == tuple(name for name, *_ in TRANSFORM_PROBE_ROSTER for _ in range(4))
    assert protocol.wrong_key_indexes == (0, 1, 2, 3)
    assert protocol.geometry_ratio_roster[0].ratio == 1.0 / 16.0
    assert protocol.geometry_ratio_roster[-1].ratio == 1.0 / 4.0
    assert tuple(
        (
            item.transform_identity,
            item.crop_fraction,
            item.scale_factor,
            item.rotation_degrees,
        )
        for item in protocol.transform_probe_roster
    ) == TRANSFORM_PROBE_ROSTER
    assert protocol.line_search_factors == tuple(
        1.0 / (2**index) for index in range(8)
    )
    assert (
        protocol.content_relative_l2_numerator,
        protocol.content_relative_l2_denominator,
    ) == (3, 250)
    assert protocol.content_projection_relative_limit == 1.0e-4
    assert protocol.callback_index == 18
    assert protocol.qk_observation_schedule_index == 7
    assert protocol.maximum_attempts_per_unit == 2
    assert protocol.authorized_operational_unit_count == 1
    assert protocol.authorized_scientific_unit_count == 0
    assert protocol.authorized_total_unit_count == 1
    assert protocol.authorized_maximum_attempts_per_unit == 1
    assert tuple(unit.unit_index for unit in protocol.authorized_unit_roster) == (0,)
    assert protocol.authorized_unit_roster[0].maximum_record_attempts == 1
    assert protocol.authorized_unit_roster_digest == (
        "a1edd1bdfb2c337c1ade319e1972a51bf19a647b0983932d947bf9e031502e3c"
    )
    assert protocol.maximum_duration_seconds_per_unit == 2700
    assert protocol.ratio_eligibility_rule == (
        "after_all_twelve_ratio_probe_units_are_terminal_choose_the_first_ratio_"
        "in_ascending_frozen_order_with_four_of_four_write_accepted_positive_"
        "actual_registered_gain_positive_keyed_gain_margin_and_zero_identity_"
        "budget_integrity_or_nonfinite_violation"
    )
    assert protocol.claim_boundary == CLAIM_BOUNDARY
    assert protocol.unit_roster_digest == (
        "e5eea4590b4dfaa0494aef483fce8f2ce89f10be66b92db8d776a3fe6c9ac448"
    )
    assert "no_ratio_selection" in CLAIM_BOUNDARY
    assert "no_estimator" in CLAIM_BOUNDARY
    assert len(manifest.entries) == 4

    legacy_payload = dict(config_payload)
    legacy_payload["run_id"] = "ceg_wm_qk_synchronization_write_diagnosis"
    legacy_config = tmp_path / "qk_legacy_run_identity.json"
    legacy_config.write_text(
        json.dumps(legacy_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        QkSynchronizationWriteProtocolError,
        match="protocol identity drifted",
    ):
        load_qk_synchronization_write_protocol(
            legacy_config,
            repository_root=ROOT,
        )


@pytest.mark.unit
def test_qk_failure_localization_entrypoint_cannot_execute_scientific_units() -> None:
    source = inspect.getsource(execute_qk_synchronization_write_diagnostic_session)

    assert "protocol.authorized_unit_roster" in source
    assert "execute_scientific_unit" not in source
    assert "replay_synchronization_diagnosis_aggregate" not in source
    assert "frozen_roster_complete" not in source
    assert "operational_failure_localization_complete" in source
    assert "operational_failure_localization_failed" in source

    bindings = _authorized_persistence_bindings(_runner())
    assert len(bindings) == 1
    assert bindings[0].unit_index == 0
    assert bindings[0].maximum_record_attempts == 1
    assert bindings[0].study_unit() == _runner().protocol.authorized_unit_roster[0]


@pytest.mark.unit
def test_qk_diagnosis_manifest_is_disjoint_on_all_five_axes() -> None:
    protocol, manifest = load_qk_synchronization_write_protocol(
        CONFIG, repository_root=ROOT
    )
    prior = load_authority_deny_axes(protocol.prior_development_manifests, ROOT)

    assert {item.prompt_digest for item in manifest.entries}.isdisjoint(
        prior.prompt_digests
    )
    assert {
        manifest.source_cluster_namespace,
        *(item.cluster_identity for item in manifest.entries),
    }.isdisjoint(prior.source_cluster_identities)
    assert {manifest.seed_namespace, *(item.generation_seed for item in manifest.entries)}.isdisjoint(
        {*prior.seed_namespaces, *prior.generation_seeds}
    )
    assert {
        manifest.image_lineage_namespace,
        *(item.image_lineage_digest for item in manifest.entries),
    }.isdisjoint(prior.image_lineage_identities)
    assert {
        manifest.content_key_family_namespace,
        manifest.geometry_key_family_namespace,
        protocol.content_registered_key_derivation_identity,
        protocol.geometry_registered_key_derivation_identity,
        protocol.wrong_key_control_identity,
    }.isdisjoint(prior.key_control_identities)


@pytest.mark.unit
def test_qk_diagnosis_reuses_cluster_identity_without_merging_scientific_units() -> None:
    protocol, manifest = load_qk_synchronization_write_protocol(
        CONFIG, repository_root=ROOT
    )
    units = tuple(
        unit for unit in protocol.unit_roster[1:] if unit.source_cluster_ordinal == 0
    )
    identities = tuple(
        derive_qk_synchronization_analysis_identity(
            manifest.entries[0],
            unit,
            content_key_family_digest="4" * 64,
            geometry_key_family_digest="5" * 64,
        )
        for unit in units
    )

    assert len(units) == 7
    assert len({item.unit_id for item in identities}) == 7
    assert len({item.source_cluster_id for item in identities}) == 1


@pytest.mark.unit
def test_qk_diagnosis_protocol_rejects_routing_or_budget_drift() -> None:
    protocol, _ = load_qk_synchronization_write_protocol(
        CONFIG, repository_root=ROOT
    )

    with pytest.raises(QkSynchronizationWriteProtocolError):
        replace(protocol, routing_mode="content_router").validate()
    with pytest.raises(QkSynchronizationWriteProtocolError):
        replace(protocol, scientific_unit_count=27).validate()


@pytest.mark.unit
def test_qk_ratio_observation_separates_ste_acceptance_from_public_rgb8_gains() -> None:
    observation = _accepted_ratio(0, *GEOMETRY_RATIO_ROSTER[0])

    assert observation.ste_acceptance_score == 0.21
    assert observation.registered_gain == pytest.approx(0.04)
    assert observation.maximum_wrong_gain == pytest.approx(0.005)
    assert observation.keyed_gain_margin == pytest.approx(0.035)
    assert observation.ratio_eligible
    assert observation.rgb8_quality_delta.content_only_rgb8_digest == (
        observation.content_only_rgb8_digest
    )
    assert observation.rgb8_quality_delta.geometry_written_rgb8_digest == (
        observation.geometry_written_rgb8_digest
    )
    with pytest.raises(QkSynchronizationWriteMetricError):
        replace(observation, wrong_key_indexes=(0, 1, 2)).validate()
    with pytest.raises(QkSynchronizationWriteMetricError):
        replace(observation, content_only_rgb8_digest="unpaired-content").validate()
    with pytest.raises(QkSynchronizationWriteMetricError):
        replace(observation, paired_public_evidence_digest="tampered-pair").validate()


@pytest.mark.unit
def test_qk_quality_and_rejected_write_enforce_exact_pairing_biconditional() -> None:
    with pytest.raises(QkSynchronizationWriteMetricError):
        create_qk_rgb8_quality_delta(
            relative_l2=0.1,
            mean_squared_error=1.0,
            content_only_rgb8_digest="same",
            geometry_written_rgb8_digest="same",
        )
    with pytest.raises(QkSynchronizationWriteMetricError):
        create_qk_rgb8_quality_delta(
            relative_l2=0.0,
            mean_squared_error=0.0,
            content_only_rgb8_digest="content",
            geometry_written_rgb8_digest="geometry",
        )
    identical = create_qk_rgb8_quality_delta(
        relative_l2=0.0,
        mean_squared_error=0.0,
        content_only_rgb8_digest="same",
        geometry_written_rgb8_digest="same",
    )
    assert identical.relative_l2 == identical.mean_squared_error == 0.0

    accepted = _accepted_ratio(0, *GEOMETRY_RATIO_ROSTER[0])
    rejected = create_qk_ratio_probe_observation(
        **{
            key: value
            for key, value in asdict(accepted).items()
            if key
            not in {
                "observation_identity",
                "paired_public_evidence_digest",
                "ratio_eligible",
                "write_accepted",
                "registered_gain",
                "wrong_key_gains",
                "maximum_wrong_gain",
                "keyed_gain_margin",
            }
        },
        write_accepted=False,
    )
    assert rejected.public_post_observation_identity is None
    assert rejected.geometry_written_rgb8_digest is None
    assert rejected.paired_public_evidence_digest is None
    with pytest.raises(QkSynchronizationWriteMetricError):
        replace(
            rejected,
            public_post_observation_identity="forbidden-post",
        ).validate()


@pytest.mark.unit
def test_qk_ratio_selection_waits_for_all_units_and_chooses_first_eligible_ratio() -> None:
    observations = _ratio_matrix(first_eligible_ratio_index=1)
    aggregate = aggregate_qk_ratio_probes(observations)

    assert aggregate.successful_unit_count == 12
    assert aggregate.eligible_counts_by_ratio == (
        (GEOMETRY_RATIO_ROSTER[0][0], 0),
        (GEOMETRY_RATIO_ROSTER[1][0], 4),
        (GEOMETRY_RATIO_ROSTER[2][0], 4),
    )
    assert aggregate.selected_ratio_identity == GEOMETRY_RATIO_ROSTER[1][0]
    assert aggregate.selected_geometry_ratio == 1.0 / 8.0
    with pytest.raises(QkSynchronizationWriteMetricError):
        aggregate_qk_ratio_probes(observations[:-1])


@pytest.mark.unit
def test_qk_ratio_failure_class_and_no_eligible_ratio_preserve_scientific_boundary() -> None:
    negative = aggregate_qk_ratio_probes(
        _ratio_matrix(first_eligible_ratio_index=None)
    )
    final = aggregate_qk_synchronization_diagnosis(
        negative, dependency_blocked_terminals=_dependency_blocked_terminals()
    )

    assert negative.ratio_probe_outcome == "mechanism_signal_not_observed"
    assert final.module_outcome == "mechanism_signal_not_observed"
    assert final.transform_excluded_count == 16
    assert final.candidate_recommendation == "candidate_not_recommended_for_selection"
    terminals = _dependency_blocked_terminals()
    with pytest.raises(QkSynchronizationWriteMetricError):
        aggregate_qk_synchronization_diagnosis(
            negative, dependency_blocked_terminals=terminals[:-1]
        )
    with pytest.raises(QkSynchronizationWriteMetricError):
        aggregate_qk_synchronization_diagnosis(
            negative,
            dependency_blocked_terminals=(*terminals[:-1], terminals[0]),
        )
    with pytest.raises(QkSynchronizationWriteMetricError):
        replace(terminals[0], terminal_identity="tampered-terminal").validate()
    with pytest.raises(TypeError):
        aggregate_qk_synchronization_diagnosis(
            negative, dependency_blocked_excluded_count=16
        )

    failures = tuple(
        QkTerminalFailure(
            cluster_ordinal=cluster,
            case_identity=ratio_identity,
            failure_class=(
                "implementation_failure"
                if cluster == 0 and ratio_index == 0
                else "resource_failure"
            ),
        )
        for ratio_index, (ratio_identity, _ratio) in enumerate(GEOMETRY_RATIO_ROSTER)
        for cluster in range(4)
    )
    blocked = aggregate_qk_ratio_probes((), failures)
    blocked_final = aggregate_qk_synchronization_diagnosis(
        blocked, dependency_blocked_terminals=_dependency_blocked_terminals()
    )
    assert blocked.ratio_probe_outcome == "implementation_blocked"
    assert blocked_final.module_outcome == "implementation_blocked"

    resource_only = tuple(
        replace(item, failure_class="resource_failure") for item in failures
    )
    resource_blocked = aggregate_qk_ratio_probes((), resource_only)
    assert resource_blocked.ratio_probe_outcome == "resource_blocked"
    with pytest.raises(QkSynchronizationWriteMetricError):
        replace(resource_only[0], failure_class="scientific_failure").validate()


@pytest.mark.unit
def test_qk_transform_probe_uses_selected_ratio_and_fixed_sixteen_unit_denominator() -> None:
    ratio = aggregate_qk_ratio_probes(
        _ratio_matrix(first_eligible_ratio_index=0)
    )
    transformed = _transforms(ratio.selected_ratio_identity)
    aggregate = aggregate_qk_synchronization_diagnosis(ratio, transformed)

    assert aggregate.transform_observation_count == 16
    assert aggregate.module_outcome == "mechanism_signal_observed"
    assert aggregate.candidate_recommendation == (
        "candidate_worth_further_selection"
    )
    assert aggregate.transform_margin_minimum == pytest.approx(0.16)
    with pytest.raises(QkSynchronizationWriteMetricError):
        aggregate_qk_synchronization_diagnosis(ratio, transformed[:-1])
    with pytest.raises(QkSynchronizationWriteMetricError):
        aggregate_qk_synchronization_diagnosis(
            ratio,
            transformed,
            dependency_blocked_terminals=_dependency_blocked_terminals(),
        )
    with pytest.raises(QkSynchronizationWriteMetricError):
        replace(transformed[0], identity_violation_count=1).validate()


@pytest.mark.unit
def test_qk_negative_transform_margins_remain_diagnostic_not_transform_robustness_gate() -> None:
    ratio = aggregate_qk_ratio_probes(
        _ratio_matrix(first_eligible_ratio_index=0)
    )
    transformed = _transforms(
        ratio.selected_ratio_identity,
        registered_score=-0.2,
        wrong_key_scores=(0.01, 0.02, 0.03, 0.04),
    )
    aggregate = aggregate_qk_synchronization_diagnosis(ratio, transformed)

    assert aggregate.transform_observation_count == 16
    assert aggregate.transform_margin_minimum == pytest.approx(-0.24)
    assert aggregate.transform_margin_mean == pytest.approx(-0.24)
    assert aggregate.transform_margin_median == pytest.approx(-0.24)
    assert aggregate.module_outcome == ratio.ratio_probe_outcome
    assert aggregate.module_outcome == "mechanism_signal_observed"
    assert aggregate.candidate_recommendation == (
        "candidate_worth_further_selection"
    )


@pytest.mark.unit
def test_qk_runner_registers_exact_roster_and_commits_dependency_terminal_shape() -> None:
    runner = _runner()
    bindings = runner.create_persistence_unit_bindings()
    record = runner.create_dependency_blocked_record(
        unit_index=13,
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
    )

    assert len(bindings) == 29
    assert tuple(item.unit_index for item in bindings) == tuple(range(29))
    assert record.execution_status == "excluded"
    assert record.failure_class == "dependency_blocked"
    assert record.decision_trace["decision_role"] == "dependency_blocked_excluded"
    assert record.operation_result_payload["dependency_blocked_terminal"][
        "dependency_identity"
    ] == "no_eligible_geometry_write_ratio"
    assert record.scientific_claim_boundary == (
        "preliminary_development_signal_only_no_promotion_or_scientific_claim"
    )


@pytest.mark.unit
def test_qk_runner_resource_retry_and_implementation_terminal_are_explicit() -> None:
    runner = _runner()
    retry = runner.create_failed_record(
        unit_index=1,
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
        actual_elapsed_seconds=1.0,
        failure_type="builtins.MemoryError",
        resource_failure=True,
    )
    terminal = runner.create_failed_record(
        unit_index=1,
        attempt_index=1,
        retry_parent_intent_digest="5" * 64,
        maximum_duration_seconds=2700,
        actual_elapsed_seconds=2.0,
        failure_type="builtins.MemoryError",
        resource_failure=True,
    )
    implementation = runner.create_failed_record(
        unit_index=2,
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
        actual_elapsed_seconds=1.0,
        failure_type="builtins.RuntimeError",
        resource_failure=False,
    )

    assert retry.execution_status == "retry"
    assert retry.attempt_disposition() == "retryable_resource_failure"
    assert terminal.execution_status == "failed"
    assert terminal.attempt_disposition() == "final_failure"
    assert implementation.failure_class == "implementation_failure"


@pytest.mark.unit
def test_qk_operational_smoke_commits_and_recovers_exact_non_scientific_record(
    tmp_path: Path,
) -> None:
    runner, runtime, _backend = _public_chain_runner()
    bindings = _authorized_persistence_bindings(runner)
    store = DevelopmentPersistentStore(
        tmp_path,
        run_id=runner.run_id,
        worker_identity=FrozenWorkerIdentity(
            revision=runner.method_code_revision,
            protocol_digest=runner.protocol_digest,
            execution_intent_authority_digest=(
                runner.execution_intent_authority_digest
            ),
            input_manifest_digest=runner.manifest.digest(),
            candidate_config_digest=runner.candidate_config_digest,
            unit_roster_digest=runner.protocol.authorized_unit_roster_digest,
        ),
        registered_unit_bindings=bindings,
    )
    lease = store.acquire_lease(
        session_id="qk_public_runtime_smoke_session",
        now_epoch_seconds=100,
        lease_duration_seconds=10000,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=100)
    intent = store.create_session_intent(cursor, lease, now_epoch_seconds=101)
    record = runner.execute_operational_smoke(
        base_latent=torch.randn(
            (1, 16, 64, 64),
            generator=torch.Generator().manual_seed(2026084200),
            dtype=torch.float16,
        ),
        attempt_index=intent.attempt_index,
        retry_parent_intent_digest=intent.parent_attempt_intent_digest,
        maximum_duration_seconds=intent.maximum_duration_seconds,
    )
    record.validate()
    marker = store.commit_session_unit(
        cursor,
        lease,
        intent,
        record=record,
        raw_secret_values=(runner.content_root, runner.geometry_root),
        now_epoch_seconds=102,
    )
    recovery = store.recover(now_epoch_seconds=103)
    recovered_cursor = store.open_session_cursor(lease, now_epoch_seconds=103)
    runtime.close()

    assert marker.attempt_disposition == "success"
    assert len(bindings) == 1
    assert intent.unit_index == 0
    assert intent.maximum_record_attempts == 1
    assert marker.record_id == record.record_id
    assert tuple(item.unit_index for item in recovery.committed_units) == (0,)
    assert recovered_cursor.operational_records == (record,)
    assert recovered_cursor.next_unit_index == 1
    assert record.counts_as_scientific_coverage is False
    assert record.scientific_claims_supported is False
    assert record.operation_result_payload == {
        "operational_role": "public_qk_synchronization_write_smoke",
        "source_cluster_ordinal": 0,
        "case_ids": ["qk_synchronization_write_public_runtime_smoke"],
        "responsibility_result_digests": [
            [
                "qk_geometry_sync",
                record.operation_result_payload["responsibility_result_digests"][0][1],
            ]
        ],
        "elapsed_seconds": record.actual_elapsed_seconds,
        "runtime_config_digest": record.operation_result_payload[
            "runtime_config_digest"
        ],
        "counts_as_scientific_coverage": False,
        "scientific_claims_supported": False,
    }
    assert len(
        record.operation_result_payload["responsibility_result_digests"][0][1]
    ) == 64


@pytest.mark.unit
def test_qk_operational_failure_exhausts_only_authorized_unit_zero(
    tmp_path: Path,
) -> None:
    runner = _runner()
    bindings = _authorized_persistence_bindings(runner)
    store = DevelopmentPersistentStore(
        tmp_path,
        run_id=runner.run_id,
        worker_identity=FrozenWorkerIdentity(
            revision=runner.method_code_revision,
            protocol_digest=runner.protocol_digest,
            execution_intent_authority_digest=(
                runner.execution_intent_authority_digest
            ),
            input_manifest_digest=runner.manifest.digest(),
            candidate_config_digest=runner.candidate_config_digest,
            unit_roster_digest=runner.protocol.authorized_unit_roster_digest,
        ),
        registered_unit_bindings=bindings,
    )
    lease = store.acquire_lease(
        session_id="qk_failure_localization_session",
        now_epoch_seconds=100,
        lease_duration_seconds=10,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=100)
    intent = store.create_session_intent(cursor, lease, now_epoch_seconds=101)

    assert intent.unit_index == 0
    assert intent.attempt_index == 0
    assert intent.maximum_record_attempts == 1
    assert cursor.committed_units == ()
    assert tuple((store.run_root / "markers").glob("*.json")) == ()
    assert tuple((store.run_root / "bundles").glob("*.zip")) == ()
    with pytest.raises(
        DevelopmentPersistenceError,
        match="interrupted unit exhausted frozen attempts",
    ):
        store.recover(now_epoch_seconds=111)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "replacement", "error_match"),
    (
        (
            "operational_role",
            "qk_runtime_smoke_alias",
            "identity drifted",
        ),
        (
            "case_ids",
            ["qk_synchronization_write_wrong_runtime_smoke"],
            "case identity drifted",
        ),
        (
            "responsibility_result_digests",
            [["content_embedder", "a" * 64]],
            "responsibility coverage drifted",
        ),
    ),
)
def test_qk_operational_smoke_rejects_identity_drift(
    field: str,
    replacement: object,
    error_match: str,
) -> None:
    record = _valid_qk_operational_record()
    operation = {**record.operation_result_payload, field: replacement}
    with pytest.raises(DevelopmentRecordError, match=error_match):
        replace(record, operation_result_payload=operation).validate()


@pytest.mark.unit
def test_qk_operational_smoke_rejects_extra_result_field() -> None:
    record = _valid_qk_operational_record()
    operation = {**record.operation_result_payload, "write_status": "accepted"}
    with pytest.raises(DevelopmentRecordError, match="result schema drifted"):
        replace(record, operation_result_payload=operation).validate()


@pytest.mark.unit
@pytest.mark.parametrize(
    "resource_type",
    tuple(
        dict.fromkeys(
            (
                MemoryError,
                getattr(torch, "OutOfMemoryError", MemoryError),
                getattr(torch.cuda, "OutOfMemoryError", MemoryError),
            )
        )
    ),
)
@pytest.mark.parametrize("link_name", ("__cause__", "__context__"))
def test_qk_resource_failure_follows_wrapped_type_chain(
    resource_type: type[BaseException],
    link_name: str,
) -> None:
    secret = "must-not-enter-diagnostic"
    resource = resource_type(secret)
    runtime_error = RuntimeAdapterError(secret)
    setattr(runtime_error, link_name, resource)
    adapter_error = CegWmExperimentAdapterError(secret)
    setattr(adapter_error, link_name, runtime_error)

    assert _is_resource_failure(adapter_error) is True
    assert _qualified_exception_type_chain(adapter_error) == (
        "experiments.methods.ceg_wm.CegWmExperimentAdapterError",
        "runtime.adapter.RuntimeAdapterError",
        f"{resource_type.__module__}.{resource_type.__qualname__}",
    )


@pytest.mark.unit
def test_qk_failure_diagnostic_is_cycle_safe_and_excludes_sensitive_state() -> None:
    secret = "qk-secret-must-not-be-persisted"
    outer = CegWmExperimentAdapterError(secret)
    resource = MemoryError(secret)
    outer.__cause__ = resource
    resource.__cause__ = outer
    binding = _runner().create_persistence_unit_bindings()[0]

    diagnostic = _failure_diagnostic(outer, active_binding=binding)
    serialized = json.dumps(diagnostic, sort_keys=True)

    assert diagnostic["failure_type_chain"] == [
        "experiments.methods.ceg_wm.CegWmExperimentAdapterError",
        "builtins.MemoryError",
    ]
    assert diagnostic["failure_class"] == "resource_failure"
    assert diagnostic["unit_index"] == 0
    assert diagnostic["unit_id"] == "development_unit_0000"
    assert diagnostic["operation_identity"] == (
        "qk_synchronization_write_public_runtime_smoke"
    )
    assert diagnostic["phase"] == "development_environment_preflight"
    assert diagnostic["counts_as_scientific_coverage"] is False
    assert diagnostic["scientific_claims_supported"] is False
    assert secret not in serialized
    for forbidden in (
        "message",
        "repr",
        "traceback",
        "locals",
        "record_id",
        "module_outcome",
        "candidate_recommendation",
    ):
        assert forbidden not in diagnostic


@pytest.mark.unit
def test_qk_failure_diagnostic_does_not_classify_message_as_resource_failure() -> None:
    error = RuntimeError("torch.cuda.OutOfMemoryError and MemoryError text only")
    diagnostic = _failure_diagnostic(error, active_binding=None)

    assert _is_resource_failure(error) is False
    assert diagnostic["failure_type_chain"] == ["builtins.RuntimeError"]
    assert diagnostic["failure_class"] == "implementation_failure"
    assert diagnostic["unit_index"] is None
    assert "module_outcome" not in diagnostic


@pytest.mark.unit
def test_qk_failure_diagnostic_records_only_launch_blocking_identity() -> None:
    secret = "launch-blocking-message-must-not-be-persisted"
    diagnostic = _failure_diagnostic(
        RuntimeError(secret),
        active_binding=None,
        cuda_launch_blocking_enabled=True,
    )

    assert diagnostic["cuda_launch_blocking_identity"] == (
        "cuda_launch_blocking_enabled"
    )
    assert secret not in json.dumps(diagnostic, sort_keys=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    "stage_error_type",
    (
        sd35_backend_module.Sd35BackendGenerationSuffixTransformerForwardError,
        sd35_backend_module.Sd35BackendGenerationSuffixSchedulerStepError,
        sd35_backend_module.Sd35BackendDifferentiableVaeInitialDecodeForwardError,
        sd35_backend_module.Sd35BackendDifferentiableVaeCheckpointRecomputationError,
        sd35_backend_module.Sd35BackendDifferentiableVaeCheckpointExecutionError,
        sd35_backend_module.Sd35BackendDifferentiableVaeEncodeError,
        sd35_backend_module.Sd35BackendDifferentiableDetectionNoiseSchedulingError,
        sd35_backend_module.Sd35BackendDifferentiableQkTransformerForwardError,
    ),
)
def test_qk_failure_diagnostic_preserves_safe_sd35_backend_failure_type(
    stage_error_type: type[BaseException],
) -> None:
    secret = "stage-message-must-not-be-persisted"
    leaf = RuntimeError(secret)
    stage = stage_error_type()
    stage.__cause__ = leaf
    outer = CegWmExperimentAdapterError(secret)
    outer.__cause__ = stage

    diagnostic = _failure_diagnostic(outer, active_binding=None)
    serialized = json.dumps(diagnostic, sort_keys=True)

    assert diagnostic["failure_type_chain"] == [
        "experiments.methods.ceg_wm.CegWmExperimentAdapterError",
        f"{stage_error_type.__module__}.{stage_error_type.__qualname__}",
        "builtins.RuntimeError",
    ]
    assert diagnostic["failure_class"] == "implementation_failure"
    assert secret not in serialized


@pytest.mark.unit
def test_qk_failure_diagnostic_preserves_only_bounded_cuda_memory_facts() -> None:
    secret = "cuda-fact-secret-must-not-be-persisted"
    facts = (
        ("before_allocated_bytes", 10),
        ("before_reserved_bytes", 20),
        ("before_max_allocated_bytes", 30),
        ("before_max_reserved_bytes", 40),
        ("after_allocated_bytes", 50),
        ("after_reserved_bytes", 60),
        ("after_max_allocated_bytes", 70),
        ("after_max_reserved_bytes", 80),
        ("total_device_bytes", 100),
    )
    stage = sd35_backend_module.Sd35BackendDifferentiableVaeInitialDecodeForwardError(
        cuda_memory_facts=facts,
        runtime_reason_identity="runtime_reported_memory_allocation_failure",
    )
    stage.__cause__ = RuntimeError(secret)
    outer = CegWmExperimentAdapterError(secret)
    outer.__cause__ = stage

    assert _runtime_failure_safe_attribution(outer) == (
        "differentiable_vae_initial_decode_forward",
        "runtime_reported_memory_allocation_failure",
        dict(facts),
    )
    diagnostic = _failure_diagnostic(outer, active_binding=None)
    serialized = json.dumps(diagnostic, sort_keys=True)

    assert diagnostic["runtime_failure_operation_identity"] == (
        "differentiable_vae_initial_decode_forward"
    )
    assert diagnostic["runtime_failure_cuda_memory_facts"] == dict(facts)
    assert diagnostic["runtime_failure_reason_identity"] == (
        "runtime_reported_memory_allocation_failure"
    )
    assert diagnostic["failure_class"] == "implementation_failure"
    assert secret not in serialized
    for forbidden in ("message", "traceback", "tensor", "latent", "repr"):
        assert forbidden not in serialized

    no_cuda_stage = (
        sd35_backend_module.Sd35BackendDifferentiableVaeInitialDecodeForwardError()
    )
    no_cuda_outer = CegWmExperimentAdapterError(secret)
    no_cuda_outer.__cause__ = no_cuda_stage
    assert _runtime_failure_safe_attribution(no_cuda_outer) == (
        "differentiable_vae_initial_decode_forward",
        "unclassified_runtime_failure",
        None,
    )
    no_cuda_diagnostic = _failure_diagnostic(
        no_cuda_outer,
        active_binding=None,
    )
    assert no_cuda_diagnostic["runtime_failure_operation_identity"] == (
        "differentiable_vae_initial_decode_forward"
    )
    assert "runtime_failure_cuda_memory_facts" not in no_cuda_diagnostic
    assert no_cuda_diagnostic["runtime_failure_reason_identity"] == (
        "unclassified_runtime_failure"
    )

    class ForeignFailure(RuntimeError):
        operation_identity = "unsafe_path_or_secret_sentinel"

        def __init__(self) -> None:
            super().__init__()
            self.cuda_memory_facts = facts

    foreign_outer = CegWmExperimentAdapterError(secret)
    foreign_outer.__cause__ = ForeignFailure()
    assert _runtime_failure_safe_attribution(foreign_outer) is None
    foreign_diagnostic = _failure_diagnostic(
        foreign_outer,
        active_binding=None,
    )
    foreign_serialized = json.dumps(foreign_diagnostic, sort_keys=True)
    assert "runtime_failure_operation_identity" not in foreign_diagnostic
    assert "runtime_failure_cuda_memory_facts" not in foreign_diagnostic
    assert "runtime_failure_reason_identity" not in foreign_diagnostic
    assert "unsafe_path_or_secret_sentinel" not in foreign_serialized

    class SpoofedTrustedFailure(
        sd35_backend_module.Sd35BackendDifferentiableVaeInitialDecodeForwardError
    ):
        operation_identity = "spoofed_trusted_failure_identity"

    spoofed_outer = CegWmExperimentAdapterError(secret)
    spoofed_outer.__cause__ = SpoofedTrustedFailure(cuda_memory_facts=facts)
    assert _runtime_failure_safe_attribution(spoofed_outer) is None
    spoofed_diagnostic = _failure_diagnostic(
        spoofed_outer,
        active_binding=None,
    )
    spoofed_serialized = json.dumps(spoofed_diagnostic, sort_keys=True)
    assert "runtime_failure_operation_identity" not in spoofed_diagnostic
    assert "runtime_failure_cuda_memory_facts" not in spoofed_diagnostic
    assert "runtime_failure_reason_identity" not in spoofed_diagnostic
    assert "spoofed_trusted_failure_identity" not in spoofed_serialized

    string_reported_memory_stage = (
        sd35_backend_module.Sd35BackendDifferentiableVaeInitialDecodeForwardError(
            cuda_memory_facts=facts,
            runtime_reason_identity=(
                "runtime_reported_memory_allocation_failure"
            ),
        )
    )
    string_reported_memory_stage.__cause__ = RuntimeError(
        "CUDA out of memory: secret allocation detail"
    )
    string_reported_outer = CegWmExperimentAdapterError(secret)
    string_reported_outer.__cause__ = string_reported_memory_stage
    string_reported_diagnostic = _failure_diagnostic(
        string_reported_outer,
        active_binding=None,
    )
    string_reported_serialized = json.dumps(
        string_reported_diagnostic,
        sort_keys=True,
    )
    assert string_reported_diagnostic["failure_class"] == (
        "implementation_failure"
    )
    assert string_reported_diagnostic["runtime_failure_reason_identity"] == (
        "runtime_reported_memory_allocation_failure"
    )
    assert "secret allocation detail" not in string_reported_serialized


@pytest.mark.unit
@pytest.mark.parametrize(
    ("trusted_failure_type", "operation_identity"),
    (
        (
            sd35_backend_module.Sd35BackendDifferentiableVaeInitialDecodeForwardError,
            "differentiable_vae_initial_decode_forward",
        ),
        (
            sd35_backend_module.Sd35BackendDifferentiableVaeCheckpointRecomputationError,
            "differentiable_vae_checkpoint_recomputation",
        ),
        (
            sd35_backend_module.Sd35BackendDifferentiableVaeCheckpointExecutionError,
            "differentiable_vae_checkpoint_execution",
        ),
    ),
)
def test_qk_runtime_failure_attribution_requires_exact_checkpoint_failure_type(
    trusted_failure_type: type[BaseException],
    operation_identity: str,
) -> None:
    secret = "checkpoint-failure-secret-must-not-be-persisted"
    facts = (
        ("before_allocated_bytes", 10),
        ("before_reserved_bytes", 20),
        ("before_max_allocated_bytes", 30),
        ("before_max_reserved_bytes", 40),
        ("after_allocated_bytes", 50),
        ("after_reserved_bytes", 60),
        ("after_max_allocated_bytes", 70),
        ("after_max_reserved_bytes", 80),
        ("total_device_bytes", 100),
    )
    exact_failure = trusted_failure_type(
        cuda_memory_facts=facts,
        runtime_reason_identity="runtime_reported_memory_allocation_failure",
    )
    exact_failure.__cause__ = MemoryError(secret)
    exact_outer = CegWmExperimentAdapterError(secret)
    exact_outer.__cause__ = exact_failure
    exact_diagnostic = _failure_diagnostic(exact_outer, active_binding=None)

    assert _runtime_failure_safe_attribution(exact_outer) == (
        operation_identity,
        "runtime_reported_memory_allocation_failure",
        dict(facts),
    )
    assert exact_diagnostic["failure_class"] == "resource_failure"
    assert exact_diagnostic["runtime_failure_operation_identity"] == (
        operation_identity
    )
    assert secret not in json.dumps(exact_diagnostic, sort_keys=True)

    class DerivedCheckpointFailure(trusted_failure_type):
        pass

    derived_failure = DerivedCheckpointFailure(
        cuda_memory_facts=facts,
        runtime_reason_identity="runtime_reported_memory_allocation_failure",
    )
    derived_outer = CegWmExperimentAdapterError(secret)
    derived_outer.__cause__ = derived_failure
    derived_diagnostic = _failure_diagnostic(derived_outer, active_binding=None)

    assert _runtime_failure_safe_attribution(derived_outer) is None
    assert "runtime_failure_operation_identity" not in derived_diagnostic
    assert "runtime_failure_reason_identity" not in derived_diagnostic
    assert "runtime_failure_cuda_memory_facts" not in derived_diagnostic
    assert secret not in json.dumps(derived_diagnostic, sort_keys=True)


@pytest.mark.unit
def test_qk_runtime_failure_attribution_identifies_real_checkpoint_backward_metadata_mismatch(
) -> None:
    invocation_count = 0

    def checkpointed_sine(value: torch.Tensor) -> torch.Tensor:
        nonlocal invocation_count
        invocation_count += 1
        active = value if invocation_count == 1 else value[:1]
        return torch.sin(active)

    checkpoint_input = torch.arange(4.0, requires_grad=True)
    checkpoint_output = activation_checkpoint(
        checkpointed_sine,
        checkpoint_input,
        use_reentrant=False,
        preserve_rng_state=True,
    )
    with pytest.raises(CheckpointError) as checkpoint_failure:
        torch.autograd.grad(checkpoint_output.sum(), checkpoint_input)

    secret = "checkpoint-framework-message-must-not-be-persisted"
    outer = CegWmExperimentAdapterError(secret)
    outer.__cause__ = checkpoint_failure.value
    diagnostic = _failure_diagnostic(outer, active_binding=None)
    serialized = json.dumps(diagnostic, sort_keys=True)

    assert type(checkpoint_failure.value) is CheckpointError
    assert invocation_count == 2
    assert diagnostic["failure_type_chain"] == [
        "experiments.methods.ceg_wm.CegWmExperimentAdapterError",
        "torch.utils.checkpoint.CheckpointError",
    ]
    assert diagnostic["failure_class"] == "implementation_failure"
    assert diagnostic["runtime_failure_operation_identity"] == (
        "differentiable_vae_checkpoint_execution"
    )
    assert diagnostic["runtime_failure_reason_identity"] == (
        "checkpoint_recomputation_metadata_mismatch"
    )
    assert "runtime_failure_cuda_memory_facts" not in diagnostic
    assert secret not in serialized
    for forbidden in ("message", "traceback", "tensor", "path", "repr"):
        assert forbidden not in serialized

    class DerivedCheckpointError(CheckpointError):
        pass

    derived_outer = CegWmExperimentAdapterError(secret)
    derived_outer.__cause__ = DerivedCheckpointError(secret)
    derived_diagnostic = _failure_diagnostic(derived_outer, active_binding=None)
    derived_serialized = json.dumps(derived_diagnostic, sort_keys=True)

    assert _runtime_failure_safe_attribution(derived_outer) is None
    assert "runtime_failure_operation_identity" not in derived_diagnostic
    assert "runtime_failure_reason_identity" not in derived_diagnostic
    assert "runtime_failure_cuda_memory_facts" not in derived_diagnostic
    assert secret not in derived_serialized


@pytest.mark.unit
def test_qk_runner_success_record_binds_ratio_metric_to_registered_responsibility() -> None:
    runner = _runner()
    observation = _accepted_ratio(0, *GEOMETRY_RATIO_ROSTER[0])
    record = runner._record(
        unit_index=1,
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
        actual_elapsed_seconds=1.0,
        operation={"ratio_probe_observation": asdict(observation)},
        observation_identity=observation.observation_identity,
    )

    assert record.execution_status == "success"
    assert record.responsibility_id == "geometry_write_ratio_probe"
    assert record.metric_observation["responsibility_id"] == record.responsibility_id
    assert tuple(record.metric_observation["registered_metric_ids"]) == record.metric_ids


@pytest.mark.unit
def test_qk_runner_crosses_real_public_write_and_blind_observation_chain() -> None:
    runner, runtime, backend = _public_chain_runner()

    observation, operation, members = runner.execute_ratio_probe(
        unit_index=1,
        base_latent=torch.randn(
            (1, 16, 64, 64),
            generator=torch.Generator().manual_seed(2026084199),
            dtype=torch.float16,
        ),
    )

    assert operation["routing_used"] is False
    assert operation["content_branch_id"] == "hf_only"
    assert len(observation.public_pre_wrong_key_scores) == 4
    assert observation.public_pre_observation_identity
    assert operation["private_qk_or_latent_persisted"] is False
    assert set(members) <= {"diagnostics/geometry_written_rgb8.bin"}
    assert backend.suffix_replay_modes
    assert True in backend.suffix_replay_modes
    public_pre_image = backend.public_detection_images[-1]
    public_pre_rgb8 = torch.floor(public_pre_image * 255.0).to(torch.uint8)
    assert sha256(public_pre_rgb8.contiguous().numpy().tobytes()).hexdigest() == (
        observation.content_only_rgb8_digest
    )
    assert observation.rgb8_quality_delta is None
    runtime.close()


@pytest.mark.integration
@pytest.mark.slow
def test_qk_ratio_record_recovers_exact_rgb8_for_transform_without_regeneration(
    tmp_path: Path,
) -> None:
    runner, runtime, backend = _public_chain_runner()
    protocol = runner.protocol
    manifest = runner.manifest
    store = DevelopmentPersistentStore(
        tmp_path,
        run_id=runner.run_id,
        worker_identity=FrozenWorkerIdentity(
            revision=runner.method_code_revision,
            protocol_digest=runner.protocol_digest,
            execution_intent_authority_digest=(
                runner.execution_intent_authority_digest
            ),
            input_manifest_digest=manifest.digest(),
            candidate_config_digest=runner.candidate_config_digest,
            unit_roster_digest=protocol.unit_roster_digest,
        ),
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    lease = store.acquire_lease(
        session_id="qk_ratio_member_recovery_session",
        now_epoch_seconds=100,
        lease_duration_seconds=10000,
    )
    ratio_binding = runner.create_persistence_unit_bindings()[1]
    ratio_intent = store.create_intent(
        lease,
        unit_id=ratio_binding.unit_id,
        unit_index=ratio_binding.unit_index,
        attempt_index=0,
        parent_attempt_intent_digest=None,
        now_epoch_seconds=101,
    )
    content_rgb8 = torch.zeros((1, 3, 512, 512), dtype=torch.uint8)
    source_rgb8 = torch.arange(
        1 * 3 * 512 * 512,
        dtype=torch.int64,
    ).remainder(251).to(torch.uint8).reshape(1, 3, 512, 512)
    content_digest = sha256(content_rgb8.numpy().tobytes()).hexdigest()
    source_digest = sha256(source_rgb8.numpy().tobytes()).hexdigest()
    selected_ratio_identity = protocol.geometry_ratio_roster[0].ratio_identity
    ratio_observation = _accepted_ratio(
        0,
        selected_ratio_identity,
        protocol.geometry_ratio_roster[0].ratio,
        content_only_rgb8_digest=content_digest,
        geometry_written_rgb8_digest=source_digest,
    )
    source_bytes = source_rgb8.numpy().tobytes()
    operation = {
        "routing_used": False,
        "content_branch_id": "hf_only",
        "hf_carrier_identity": "integration_fixture_carrier",
        "ratio_probe_observation": asdict(ratio_observation),
        "accepted_rgb8_member": {
            "path": RGB8_MEMBER_PATH,
            "shape": tuple(source_rgb8.shape),
            "dtype": "torch.uint8",
            "size_bytes": len(source_bytes),
            "sha256": source_digest,
        },
        "private_qk_or_latent_persisted": False,
    }
    ratio_record = runner._record(
        unit_index=1,
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
        actual_elapsed_seconds=1.0,
        operation=operation,
        observation_identity=ratio_observation.observation_identity,
    )
    ratio_record.validate()
    marker = store.commit_unit(
        lease,
        ratio_intent,
        record=ratio_record,
        diagnostic_members={RGB8_MEMBER_PATH: source_bytes},
        now_epoch_seconds=102,
    )
    recovery = store.recover(now_epoch_seconds=103)
    evidence = store.verified_terminal_scientific_evidence(
        now_epoch_seconds=103
    )
    recovered_rgb8 = _selected_rgb8(
        tmp_path,
        run_id=runner.run_id,
        cluster_ordinal=0,
        selected_ratio_identity=selected_ratio_identity,
        evidence=evidence,
        protocol=protocol,
        manifest=manifest,
    )
    assert marker.attempt_disposition == "success"
    assert tuple(item.unit_index for item in recovery.committed_units) == (1,)
    assert len(tuple((store.run_root / "markers").glob("*.json"))) == 1
    assert sha256(recovered_rgb8.numpy().tobytes()).hexdigest() == (
        ratio_observation.geometry_written_rgb8_digest
    )

    generation_count_before_transform = backend.generation_call_count
    transformed_record, transformed_members = runner.execute_scientific_unit(
        unit_index=13,
        base_latent=None,
        selected_ratio_identity=selected_ratio_identity,
        source_rgb8=recovered_rgb8,
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
    )
    transformed_record.validate()
    assert transformed_members == {}
    assert backend.generation_call_count == generation_count_before_transform
    assert transformed_record.operation_result_payload[
        "transformed_relation_observation"
    ]["source_geometry_written_rgb8_digest"] == (
        ratio_observation.geometry_written_rgb8_digest
    )

    for cluster_ordinal, ratio_identity in (
        (1, selected_ratio_identity),
        (0, protocol.geometry_ratio_roster[1].ratio_identity),
    ):
        with pytest.raises(QkSynchronizationWriteEntrypointError):
            _selected_rgb8(
                tmp_path,
                run_id=runner.run_id,
                cluster_ordinal=cluster_ordinal,
                selected_ratio_identity=ratio_identity,
                evidence=evidence,
                protocol=protocol,
                manifest=manifest,
            )

    bundle_path = (
        store.run_root / "bundles" / f"sha256_{marker.bundle_sha256}.zip"
    )
    bundle_path.write_bytes(bundle_path.read_bytes() + b"tampered")
    with pytest.raises(
        QkSynchronizationWriteEntrypointError,
        match="bundle bytes drifted",
    ):
        _selected_rgb8(
            tmp_path,
            run_id=runner.run_id,
            cluster_ordinal=0,
            selected_ratio_identity=selected_ratio_identity,
            evidence=evidence,
            protocol=protocol,
            manifest=manifest,
        )
    runtime.close()


@pytest.mark.unit
@pytest.mark.parametrize(
    "violation_field",
    (
        "identity_violation_count",
        "integrity_violation_count",
        "nonfinite_violation_count",
    ),
)
def test_qk_transform_probe_rejects_any_recorded_violation(
    violation_field: str,
) -> None:
    observation = _transforms(GEOMETRY_RATIO_ROSTER[0][0])[0]

    with pytest.raises(QkSynchronizationWriteMetricError):
        replace(observation, **{violation_field: 1}).validate()


@pytest.mark.unit
def test_qk_diagnosis_has_no_threshold_fpr_promotion_or_estimator_claim() -> None:
    protocol, _ = load_qk_synchronization_write_protocol(
        CONFIG, repository_root=ROOT
    )
    frozen = asdict(protocol)

    assert "threshold" not in frozen
    assert "fpr" not in frozen
    assert "estimator" not in frozen
    assert "promotion" not in frozen
    assert protocol.passing_module_outcome == "mechanism_signal_observed"
    assert protocol.passing_candidate_recommendation == (
        "candidate_worth_further_selection"
    )
