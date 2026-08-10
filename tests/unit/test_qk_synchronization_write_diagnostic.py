"""Frozen protocol and metric tests for Q/K synchronization-write diagnosis."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest

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
from experiments.protocol.qk_synchronization_write_diagnostic import (
    CLAIM_BOUNDARY,
    GEOMETRY_RATIO_ROSTER,
    QkSynchronizationWriteProtocolError,
    TRANSFORM_PROBE_ROSTER,
    derive_qk_synchronization_analysis_identity,
    load_qk_synchronization_write_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/qk_synchronization_write_diagnostic.json"


def _accepted_ratio(
    cluster: int,
    ratio_identity: str,
    ratio: float,
    *,
    eligible: bool = True,
):
    registered_post = 0.14 if eligible else 0.105
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
            content_only_rgb8_digest=f"content-{cluster}",
            geometry_written_rgb8_digest=f"geometry-{cluster}-{ratio_identity}",
        ),
        public_pre_observation_identity=(
            f"public_rgb8_vae_qk_pre_{cluster}_{ratio_identity}"
        ),
        public_post_observation_identity=(
            f"public_rgb8_vae_qk_post_{cluster}_{ratio_identity}"
        ),
        content_only_rgb8_digest=f"content-{cluster}",
        geometry_written_rgb8_digest=f"geometry-{cluster}-{ratio_identity}",
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
def test_qk_diagnosis_protocol_freezes_roster_order_controls_and_boundary() -> None:
    protocol, manifest = load_qk_synchronization_write_protocol(
        CONFIG, repository_root=ROOT
    )

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
    assert "no_ratio_selection" in CLAIM_BOUNDARY
    assert "no_estimator" in CLAIM_BOUNDARY
    assert len(manifest.entries) == 4


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
