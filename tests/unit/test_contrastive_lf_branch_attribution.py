"""Frozen Stage-A contrastive LF attribution protocol tests."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import nextafter
from pathlib import Path
import re

import pytest

from experiments.metrics.binomial import clopper_pearson_lower
from experiments.protocol.contrastive_lf_branch_attribution import (
    ATTACKS,
    BLIND_DETECTOR_INPUTS,
    CANDIDATE_IDS,
    CANDIDATE_ROLE_LABELS,
    CLUSTER_COUNT,
    CONFIG_DIGEST,
    CONFIG_PATH,
    CONFIRMATION_ROLE,
    ContrastiveLfProtocolError,
    ContrastiveLfProtocolResult,
    ContrastiveLfRecord,
    DENOMINATORS_BY_ROLE,
    DenominatorReport,
    ENTRIES_DIGESTS,
    EXTERNAL_WRONG_KEY_INDEXES,
    FORBIDDEN_DETECTOR_INPUTS,
    GATE_ORDER,
    GateReport,
    INTERNAL_DECOY_INDEXES,
    JPEG_GOLDEN_DIGESTS,
    MANIFEST_DIGESTS,
    MANIFEST_PATHS,
    MULTISCALE_CANDIDATE_ID,
    NULL_FIT_ROLE,
    PROMPT_ROSTER_DIGEST,
    PROMPT_ROSTER_PATH,
    PROTOCOL_ID,
    SELECTION_ROLE,
    SINGLE_SCALE_CANDIDATE_ID,
    SOURCE_ROSTER_ROWS_DIGEST,
    SOURCE_SNAPSHOT_SHA256,
    SOURCE_SNAPSHOT_PATH,
    ValidatedContrastiveLfRecordCollection,
    authenticate_selection_artifact,
    blur_complement_passes,
    branch_key_margin,
    build_record_templates,
    canonical_digest,
    choose_selection_winner,
    classify_result,
    condition_false_positive_gate_passes,
    identity_attribution_passes,
    identity_attribution_gate_passes,
    load_configuration,
    load_manifest,
    load_prompt_roster,
    population_standardize,
    provisional_tau,
    quality_gate_passes,
    validate_failure_tail,
    validate_record_collection,
    validate_split_disjointness,
)


ROOT = Path(__file__).resolve().parents[2]
NEW_CONFIG_PATHS = frozenset(
    {
        CONFIG_PATH,
        PROMPT_ROSTER_PATH,
        *MANIFEST_PATHS.values(),
    }
)


def _manifests():
    return tuple(
        load_manifest(ROOT / MANIFEST_PATHS[role_id], expected_role=role_id)
        for role_id in (NULL_FIT_ROLE, SELECTION_ROLE, CONFIRMATION_ROLE)
    )


def _walk_prior_identities(
    value: object,
    *,
    prompt_digests: set[str],
    source_clusters: set[str],
    image_lineages: set[str],
) -> None:
    if isinstance(value, dict):
        for field_name, field_value in value.items():
            if (
                field_name == "prompt_digest"
                and isinstance(field_value, str)
                and len(field_value) == 64
            ):
                prompt_digests.add(field_value)
            if (
                field_name == "source_cluster_id"
                and isinstance(field_value, str)
                and len(field_value) == 64
            ):
                source_clusters.add(field_value)
            if (
                field_name
                in {"image_lineage_identity", "image_lineage_digest"}
                and isinstance(field_value, str)
                and len(field_value) == 64
            ):
                image_lineages.add(field_value)
            if (
                field_name
                in {"prompt", "prompt_text", "operational_smoke_prompt"}
                and isinstance(field_value, str)
                and field_value
            ):
                prompt_digests.add(
                    sha256(field_value.encode("utf-8")).hexdigest()
                )
            _walk_prior_identities(
                field_value,
                prompt_digests=prompt_digests,
                source_clusters=source_clusters,
                image_lineages=image_lineages,
            )
    elif isinstance(value, list):
        for item in value:
            _walk_prior_identities(
                item,
                prompt_digests=prompt_digests,
                source_clusters=source_clusters,
                image_lineages=image_lineages,
            )


def _record(template, **overrides: object) -> ContrastiveLfRecord:
    payload = {
        "template": template,
        "attempt_index": 0,
        "execution_status": "completed",
        "method_config_digest": CONFIG_DIGEST,
        "implementation_revision": "1" * 40,
        "model_identity": "stable_diffusion_medium_pipeline",
        "runtime_identity": "registered_sd35_runtime_identity",
        "codec_identity": "pillow_rgb8_jpeg_exact_capability",
        "raw_score": None,
        "internal_decoy_scores": (),
        "registered_score": None,
        "wrong_key_score": None,
        "primary_null_score": None,
        "population_mean": None,
        "population_variance": None,
        "population_sigma": None,
        "null_asset_digest": None,
        "provisional_threshold_digest": None,
        "z_score": None,
        "key_margin": None,
        "budget_status": None,
        "materialization_replay_identity": None,
        "replay_digest": None,
        "nonfinite_detected": False,
        "paired_rgb8_mse": None,
        "failure_class": None,
        "failure_reason": None,
    }
    payload.update(overrides)
    return ContrastiveLfRecord(**payload)


def _completed_record(template) -> ContrastiveLfRecord:
    evidence: dict[str, object] = {}
    if template.record_kind in {"null_statistic", "detector"}:
        evidence["raw_score"] = 1.0
        evidence["internal_decoy_scores"] = (0.0,) * (
            template.internal_decoy_score_count
        )
        if template.control_identity == "registered_attribution":
            evidence["registered_score"] = 1.0
        elif template.control_identity == "external_wrong_key":
            evidence["wrong_key_score"] = 1.0
        elif template.control_identity == "paired_primary_null":
            evidence["primary_null_score"] = 1.0
    elif template.record_kind == "budget":
        evidence.update(
            budget_status="accepted",
            materialization_replay_identity="ordinary_rgb8_materialization",
            replay_digest="a" * 64,
        )
    elif template.record_kind == "quality":
        evidence["paired_rgb8_mse"] = 0.0
    return _record(template, **evidence)


def _validated_collection(
    role_id: str,
    *,
    selected_candidate_id: str | None = None,
    failure_index: int | None = None,
    failure_class: str = "operation_failure",
) -> ValidatedContrastiveLfRecordCollection:
    manifest = load_manifest(
        ROOT / MANIFEST_PATHS[role_id], expected_role=role_id
    )
    templates = build_record_templates(
        manifest, selected_candidate_id=selected_candidate_id
    )
    if failure_index is None:
        records = tuple(_completed_record(template) for template in templates)
    else:
        records = tuple(
            _completed_record(template)
            if ordinal < failure_index
            else _record(
                template,
                execution_status="failed",
                failure_class=failure_class,
                failure_reason="bounded_record_failure",
            )
            if ordinal == failure_index
            else _record(template, execution_status="unstarted")
            for ordinal, template in enumerate(templates)
        )
    return validate_record_collection(
        records,
        role_id=role_id,
        selected_candidate_id=selected_candidate_id,
    )


def _result(
    collection: ValidatedContrastiveLfRecordCollection,
    *,
    gate_reports: tuple[GateReport, ...],
    result_classification: str,
    selected_candidate_id: str | None,
) -> ContrastiveLfProtocolResult:
    role_id = collection.role_id
    return ContrastiveLfProtocolResult(
        schema_version=1,
        protocol_id=PROTOCOL_ID,
        role_id=role_id,
        sample_manifest_digest=MANIFEST_DIGESTS[role_id],
        manifest_entries_digest=ENTRIES_DIGESTS[role_id],
        record_collection_digest=collection.record_collection_digest,
        denominator_reports=collection.denominator_reports,
        gate_reports=gate_reports,
        first_failed_gate=next(
            (
                report.gate_id
                for report in gate_reports
                if report.gate_status == "failed"
            ),
            None,
        ),
        result_classification=result_classification,
        candidate_selection_passed=(
            role_id == CONFIRMATION_ROLE or result_classification == "success"
        ),
        confirmation_passed=(
            role_id == CONFIRMATION_ROLE and result_classification == "success"
        ),
        selected_candidate_id=selected_candidate_id,
        candidate_promoted=False,
        formal_tau_created=False,
        formal_fpr_created=False,
        full_ceg_wm_eligible=False,
    )


@pytest.mark.unit
def test_config_roster_and_manifests_replay_exact_frozen_digests() -> None:
    configuration = load_configuration(ROOT / CONFIG_PATH)
    roster = load_prompt_roster(ROOT / PROMPT_ROSTER_PATH)
    manifests = _manifests()
    validate_split_disjointness(manifests)

    assert configuration["config_digest"] == CONFIG_DIGEST
    for capability_role, digest in JPEG_GOLDEN_DIGESTS.items():
        assert configuration[f"{capability_role}_sha256"] == digest
    assert roster["prompt_roster_digest"] == PROMPT_ROSTER_DIGEST
    assert roster["rows_digest"] == SOURCE_ROSTER_ROWS_DIGEST
    assert roster["source_snapshot_path"] == SOURCE_SNAPSHOT_PATH
    assert roster["source_snapshot_sha256"] == SOURCE_SNAPSHOT_SHA256
    assert sha256((ROOT / SOURCE_SNAPSHOT_PATH).read_bytes()).hexdigest() == (
        SOURCE_SNAPSHOT_SHA256
    )
    assert tuple(manifest.role_id for manifest in manifests) == (
        NULL_FIT_ROLE,
        SELECTION_ROLE,
        CONFIRMATION_ROLE,
    )
    for manifest in manifests:
        assert manifest.entries_digest == ENTRIES_DIGESTS[manifest.role_id]
        assert manifest.manifest_digest == MANIFEST_DIGESTS[manifest.role_id]
        assert len(manifest.entries) == CLUSTER_COUNT
    frozen_file_sha256 = {
        CONFIG_PATH: (
            "75f58f28a5991de906611573d2d6d9133ff47e806357e630ca47dc7de7464a8c"
        ),
        PROMPT_ROSTER_PATH: (
            "352e2762c4828af3c536ff3a6e0dd5a78a00cdd13a839656d59d311e7418bdc5"
        ),
        MANIFEST_PATHS[NULL_FIT_ROLE]: (
            "e598c3c7aa952dc87317f4f4f5a14cbff56645d5d991dded419d2d16830e7710"
        ),
        MANIFEST_PATHS[SELECTION_ROLE]: (
            "73099470250254bd8930e8873a48752a6290de4bfa0523cf0776dea56d4d0562"
        ),
        MANIFEST_PATHS[CONFIRMATION_ROLE]: (
            "f44f87d3aa8c888154927c384cbc6814cfdf3b75c4fcbfd685ff7812e66ab15b"
        ),
    }
    assert {
        path: sha256((ROOT / path).read_bytes()).hexdigest()
        for path in frozen_file_sha256
    } == frozen_file_sha256


@pytest.mark.unit
def test_prompt_roster_is_fresh_against_every_frozen_prior_binding() -> None:
    roster = load_prompt_roster(ROOT / PROMPT_ROSTER_PATH)
    binding_paths = {
        binding["relative_path"]
        for binding in roster["exclusion_source_bindings"]
    }
    assert len(binding_paths) == 30
    assert not binding_paths & NEW_CONFIG_PATHS
    assert (
        "configs/experiments/semantic_texture_soft_route_candidate_selection_manifest.json"
        in binding_paths
    )
    assert (
        "configs/experiments/semantic_texture_soft_route_untouched_confirmation_manifest.json"
        in binding_paths
    )

    prior_prompts: set[str] = set()
    prior_clusters: set[str] = set()
    prior_lineages: set[str] = set()
    source_universe = (
        "configs/experiments/hf_only_reference_prompt_roster.json"
    )
    for binding in roster["exclusion_source_bindings"]:
        path = ROOT / binding["relative_path"]
        assert sha256(path.read_bytes()).hexdigest() == binding["file_sha256"]
        if binding["relative_path"] == source_universe:
            continue
        _walk_prior_identities(
            json.loads(path.read_text(encoding="utf-8")),
            prompt_digests=prior_prompts,
            source_clusters=prior_clusters,
            image_lineages=prior_lineages,
        )
    assert len(prior_prompts) == roster["excluded_prompt_digest_count"] == 324
    assert canonical_digest(sorted(prior_prompts)) == (
        roster["excluded_prompt_digests_digest"]
    )

    rows = roster["rows"]
    assert [row["source_row"] for row in rows] == list(range(132, 228))
    assert len({row["prompt_digest"] for row in rows}) == 96
    assert not {row["prompt_digest"] for row in rows} & prior_prompts

    snapshot_lines = (ROOT / SOURCE_SNAPSHOT_PATH).read_text(
        encoding="utf-8"
    ).splitlines()
    assert snapshot_lines[0] == "Prompt\tCategory\tChallenge\tNote"
    eligible = []
    for source_row, line in enumerate(snapshot_lines[1:], start=1):
        prompt_text, category, challenge, _note = line.split("\t")
        prompt_digest = sha256(prompt_text.encode("utf-8")).hexdigest()
        if prompt_text and prompt_digest not in prior_prompts:
            eligible.append(
                (source_row, prompt_text, prompt_digest, category, challenge)
            )
    assert [
        (
            row["source_row"],
            row["prompt_text"],
            row["prompt_digest"],
            row["category"],
            row["challenge"],
        )
        for row in rows
    ] == eligible[:96]
    manifests = _manifests()
    assert not {
        entry.source_cluster_id for manifest in manifests for entry in manifest.entries
    } & prior_clusters
    assert not {
        identity
        for manifest in manifests
        for entry in manifest.entries
        for identity in (
            entry.image_lineage_identity,
            entry.image_lineage_digest,
        )
    } & prior_lineages


@pytest.mark.unit
def test_three_splits_are_disjoint_on_every_frozen_identity_axis() -> None:
    manifests = _manifests()
    validate_split_disjointness(manifests)
    for field_name in (
        "source_row",
        "prompt_digest",
        "generation_seed",
        "source_cluster_id",
        "image_lineage_identity",
        "image_lineage_digest",
    ):
        axes = [
            {getattr(entry, field_name) for entry in manifest.entries}
            for manifest in manifests
        ]
        assert not axes[0] & axes[1]
        assert not axes[0] & axes[2]
        assert not axes[1] & axes[2]
    assert [entry.generation_seed for entry in manifests[0].entries] == list(
        range(202608210000, 202608210032)
    )
    assert [entry.generation_seed for entry in manifests[1].entries] == list(
        range(202608210100, 202608210132)
    )
    assert [entry.generation_seed for entry in manifests[2].entries] == list(
        range(202608210200, 202608210232)
    )


@pytest.mark.unit
def test_record_templates_preserve_all_fixed_denominators_and_roles() -> None:
    null_manifest, selection_manifest, confirmation_manifest = _manifests()
    null_records = build_record_templates(null_manifest)
    selection_records = build_record_templates(selection_manifest)
    confirmation_records = build_record_templates(
        confirmation_manifest,
        selected_candidate_id=MULTISCALE_CANDIDATE_ID,
    )

    assert len(null_records) == 32 + 96
    assert len(selection_records) == 128 + 512 + 3840 + 96 + 384
    assert len(confirmation_records) == 96 + 384 + 2560 + 64 + 256
    for records, expected in (
        (null_records, DENOMINATORS_BY_ROLE[NULL_FIT_ROLE]),
        (selection_records, DENOMINATORS_BY_ROLE[SELECTION_ROLE]),
        (confirmation_records, DENOMINATORS_BY_ROLE[CONFIRMATION_ROLE]),
    ):
        assert len({record.record_id for record in records}) == len(records)
        assert sum(record.record_kind == "null_statistic" for record in records) == (
            expected.raw_null_statistic_count
        )
        assert sum(
            record.record_kind == "clean_base_observation" for record in records
        ) == expected.clean_base_observation_count
        assert sum(record.record_kind == "base_generation" for record in records) == (
            expected.base_generation_count
        )
        assert sum(
            record.record_kind == "attacked_observation" for record in records
        ) == expected.attacked_observation_slot_count
        assert sum(record.record_kind == "detector" for record in records) == (
            expected.detector_record_count
        )
        assert sum(record.record_kind == "budget" for record in records) == (
            expected.budget_record_count
        )
        assert sum(record.record_kind == "quality" for record in records) == (
            expected.quality_record_count
        )
        assert {record.prompt_roster_digest for record in records} == {
            PROMPT_ROSTER_DIGEST
        }
        assert {record.source_roster_rows_digest for record in records} == {
            SOURCE_ROSTER_ROWS_DIGEST
        }

    selection_detector = [
        record for record in selection_records if record.record_kind == "detector"
    ]
    for candidate_id in ("hf_sparse_tail", *CANDIDATE_IDS):
        candidate_records = [
            record
            for record in selection_detector
            if record.candidate_id == candidate_id
        ]
        assert len(candidate_records) == 32 * 4 * 10
        assert sum(record.key_role == "wrong" for record in candidate_records) == (
            32 * 4 * 8
        )
        assert sum(
            record.control_identity == "paired_primary_null"
            for record in candidate_records
        ) == 32 * 4
        expected_decoys = 0 if candidate_id == "hf_sparse_tail" else 8
        assert {
            record.internal_decoy_score_count for record in candidate_records
        } == {expected_decoys}
    assert {record.attack_id for record in selection_detector} == set(ATTACKS)


@pytest.mark.unit
def test_population_threshold_margin_quality_and_exact_binomial_gates() -> None:
    values = tuple(float(value) for value in range(32))
    mean, variance, sigma, z_score = population_standardize(values, 31.0)
    assert mean == 15.5
    assert variance == 85.25
    assert sigma * sigma == pytest.approx(variance)
    assert z_score == pytest.approx((31.0 - mean) / sigma)
    assert provisional_tau(values) == nextafter(28.0, float("inf"))
    tied = tuple(float(min(value, 28)) for value in range(32))
    assert provisional_tau(tied) == nextafter(28.0, float("inf"))
    with pytest.raises(ContrastiveLfProtocolError):
        population_standardize((1.0,) * 32, 1.0)
    with pytest.raises(ContrastiveLfProtocolError):
        provisional_tau((*values[:-1], float("nan")))

    wrong = (0.0, 0.5, 0.25, -0.5, -1.0, 0.1, 0.2, 0.3)
    assert branch_key_margin(1.0, wrong) == 0.5
    assert identity_attribution_passes(
        registered_z=1.0,
        primary_null_z=0.75,
        external_wrong_z=wrong,
    )
    assert identity_attribution_gate_passes((True,) * 28 + (False,) * 4)
    assert not identity_attribution_gate_passes((True,) * 27 + (False,) * 5)
    assert condition_false_positive_gate_passes(
        primary_null_positive_count=3,
        external_wrong_positive_counts=(3,) * 8,
    )
    assert not condition_false_positive_gate_passes(
        primary_null_positive_count=4,
        external_wrong_positive_counts=(0,) * 8,
    )
    assert not condition_false_positive_gate_passes(
        primary_null_positive_count=0,
        external_wrong_positive_counts=(0, 0, 0, 0, 0, 0, 0, 4),
    )

    epsilon = float.fromhex("0x1.0203040506070p-16")
    assert quality_gate_passes((epsilon,) * 32, (0.0,) * 32)
    assert not quality_gate_passes(
        (nextafter(epsilon, float("inf")),) * 32,
        (0.0,) * 32,
    )
    lower = clopper_pearson_lower(24, 32, confidence_level=0.95)
    assert lower > 0.5
    assert blur_complement_passes(
        success_count=24, clopper_pearson_lower_value=lower
    )
    assert not blur_complement_passes(
        success_count=23,
        clopper_pearson_lower_value=clopper_pearson_lower(
            23, 32, confidence_level=0.95
        ),
    )


@pytest.mark.unit
def test_hierarchy_classification_and_confirmation_admission_are_frozen() -> None:
    assert choose_selection_winner(
        multiscale_passed=True, single_scale_passed=True
    ) == MULTISCALE_CANDIDATE_ID
    assert choose_selection_winner(
        multiscale_passed=True, single_scale_passed=False
    ) == MULTISCALE_CANDIDATE_ID
    assert choose_selection_winner(
        multiscale_passed=False, single_scale_passed=True
    ) == SINGLE_SCALE_CANDIDATE_ID
    assert choose_selection_winner(
        multiscale_passed=False, single_scale_passed=False
    ) is None

    complete_collection = _validated_collection(SELECTION_ROLE)
    operational_collection = _validated_collection(
        SELECTION_ROLE, failure_index=0, failure_class="runtime_failure"
    )
    incomplete_collection = _validated_collection(
        SELECTION_ROLE, failure_index=0, failure_class="operation_failure"
    )
    assert classify_result(
        validated_collection=operational_collection,
        scientific_gates_passed=False,
    ) == "operational_failure"
    assert classify_result(
        validated_collection=incomplete_collection,
        scientific_gates_passed=False,
    ) == "insufficient_evidence"
    assert classify_result(
        validated_collection=complete_collection,
        scientific_gates_passed=False,
    ) == "scientific_failure"
    assert classify_result(
        validated_collection=complete_collection,
        scientific_gates_passed=True,
    ) == "success"

    artifact = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "selection_manifest_digest": MANIFEST_DIGESTS[SELECTION_ROLE],
        "candidate_selection_passed": True,
        "selected_candidate_id": MULTISCALE_CANDIDATE_ID,
        "candidate_null_asset_digest": "2" * 64,
        "provisional_threshold_digest": "3" * 64,
        "diagnostic_only": True,
        "formal_tau_created": False,
        "formal_fpr_created": False,
        "candidate_promoted": False,
        "full_ceg_wm_eligible": False,
    }
    artifact_digest = canonical_digest(artifact)
    assert authenticate_selection_artifact(
        artifact, expected_artifact_digest=artifact_digest
    ) == MULTISCALE_CANDIDATE_ID
    drifted = {**artifact, "candidate_selection_passed": False}
    with pytest.raises(ContrastiveLfProtocolError):
        authenticate_selection_artifact(
            drifted, expected_artifact_digest=canonical_digest(drifted)
        )
    confirmation = _manifests()[2]
    with pytest.raises(ContrastiveLfProtocolError):
        build_record_templates(confirmation)


@pytest.mark.unit
def test_failure_tail_and_bounded_record_schema_fail_closed() -> None:
    templates = build_record_templates(_manifests()[1])[:3]
    records = (
        _record(templates[0]),
        _record(
            templates[1],
            execution_status="failed",
            failure_class="operation_failure",
            failure_reason="bounded_generation_failure",
        ),
        _record(templates[2], execution_status="unstarted"),
    )
    validate_failure_tail(records)
    with pytest.raises(ContrastiveLfProtocolError):
        validate_failure_tail(
            (
                records[0],
                records[1],
                _record(
                    templates[2],
                    execution_status="failed",
                    failure_class="operation_failure",
                    failure_reason="second_failure",
                ),
            )
        )
    with pytest.raises(ContrastiveLfProtocolError):
        _record(
            templates[1],
            execution_status="failed",
            failure_class="runtime_failure",
            failure_reason="Traceback includes /home/private secret token",
        ).validate()
    with pytest.raises(ContrastiveLfProtocolError):
        _record(
            templates[1],
            execution_status="failed",
            failure_class="runtime_failure",
            failure_reason="bounded failure at /content/private/location",
        ).validate()
    with pytest.raises(TypeError):
        ContrastiveLfRecord(**{**asdict(records[0]), "unexpected_field": 1})

    forbidden_evidence = {
        "raw_score": 1.0,
        "internal_decoy_scores": (0.0,),
        "registered_score": 1.0,
        "wrong_key_score": 1.0,
        "primary_null_score": 1.0,
        "population_mean": 0.0,
        "population_variance": 1.0,
        "population_sigma": 1.0,
        "null_asset_digest": "a" * 64,
        "provisional_threshold_digest": "b" * 64,
        "z_score": 0.0,
        "key_margin": 0.0,
        "budget_status": "accepted",
        "materialization_replay_identity": "ordinary_rgb8_materialization",
        "replay_digest": "c" * 64,
        "paired_rgb8_mse": 0.0,
    }
    for field_name, field_value in forbidden_evidence.items():
        with pytest.raises(ContrastiveLfProtocolError, match="has_evidence"):
            _record(
                templates[2],
                execution_status="unstarted",
                **{field_name: field_value},
            ).validate()
        with pytest.raises(ContrastiveLfProtocolError, match="has_evidence"):
            _record(
                templates[1],
                execution_status="failed",
                failure_class="operation_failure",
                failure_reason="bounded_generation_failure",
                **{field_name: field_value},
            ).validate()
    with pytest.raises(ContrastiveLfProtocolError, match="has_evidence"):
        _record(
            templates[2],
            execution_status="unstarted",
            nonfinite_detected=True,
        ).validate()
    for field_name, field_value in (
        ("failure_class", "operation_failure"),
        ("failure_reason", "bounded_generation_failure"),
    ):
        with pytest.raises(ContrastiveLfProtocolError, match="has_evidence"):
            _record(
                templates[2],
                execution_status="unstarted",
                **{field_name: field_value},
            ).validate()

    lf_detector_template = next(
        template
        for template in build_record_templates(_manifests()[1])
        if template.record_kind == "detector"
        and template.candidate_id == MULTISCALE_CANDIDATE_ID
        and template.control_identity == "registered_attribution"
    )
    valid_lf_record = _record(
        lf_detector_template,
        raw_score=1.0,
        registered_score=1.0,
        internal_decoy_scores=tuple(float(value) for value in range(8)),
    )
    valid_lf_record.validate()
    with pytest.raises(ContrastiveLfProtocolError):
        replace(valid_lf_record, internal_decoy_scores=(0.0,) * 7).validate()
    with pytest.raises(ContrastiveLfProtocolError):
        replace(valid_lf_record, raw_score=float("nan")).validate()


@pytest.mark.unit
def test_protocol_result_keeps_all_gate_reports_and_independent_statuses() -> None:
    passing_gates = tuple(GateReport(gate_id, "passed") for gate_id in GATE_ORDER)
    selection_collection = _validated_collection(SELECTION_ROLE)
    selection_success = _result(
        selection_collection,
        gate_reports=passing_gates,
        result_classification="success",
        selected_candidate_id=MULTISCALE_CANDIDATE_ID,
    )
    selection_success.validate(selection_collection)

    failed_gates = tuple(
        GateReport(
            gate_id,
            "failed" if gate_id == "candidate_attribution_null_wrong" else "passed",
        )
        for gate_id in GATE_ORDER
    )
    scientific_failure = replace(
        selection_success,
        gate_reports=failed_gates,
        first_failed_gate="candidate_attribution_null_wrong",
        result_classification="scientific_failure",
        candidate_selection_passed=False,
        selected_candidate_id=None,
    )
    scientific_failure.validate(selection_collection)
    assert len(scientific_failure.gate_reports) == len(GATE_ORDER)

    operational_collection = _validated_collection(
        SELECTION_ROLE, failure_index=4, failure_class="runtime_failure"
    )
    operational = _result(
        operational_collection,
        gate_reports=tuple(
            GateReport(gate_id, "not_evaluable") for gate_id in GATE_ORDER
        ),
        result_classification="operational_failure",
        selected_candidate_id=None,
    )
    operational.validate(operational_collection)

    confirmation_collection = _validated_collection(
        CONFIRMATION_ROLE,
        selected_candidate_id=MULTISCALE_CANDIDATE_ID,
    )
    confirmation_success = _result(
        confirmation_collection,
        gate_reports=passing_gates,
        result_classification="success",
        selected_candidate_id=MULTISCALE_CANDIDATE_ID,
    )
    confirmation_success.validate(confirmation_collection)
    confirmation_failure = replace(
        confirmation_success,
        gate_reports=failed_gates,
        first_failed_gate="candidate_attribution_null_wrong",
        result_classification="scientific_failure",
        confirmation_passed=False,
    )
    confirmation_failure.validate(confirmation_collection)
    assert confirmation_failure.selected_candidate_id == MULTISCALE_CANDIDATE_ID

    with pytest.raises(ContrastiveLfProtocolError):
        replace(
            selection_success, record_collection_digest="f" * 64
        ).validate(selection_collection)
    denominator_drift = replace(
        selection_collection.denominator_reports[0],
        completed_record_count=(
            selection_collection.denominator_reports[0].completed_record_count - 1
        ),
        unstarted_record_count=1,
    )
    with pytest.raises(ContrastiveLfProtocolError):
        replace(
            selection_success,
            denominator_reports=(
                denominator_drift,
                *selection_collection.denominator_reports[1:],
            ),
        ).validate(selection_collection)
    with pytest.raises(ContrastiveLfProtocolError):
        selection_success.validate(
            {"record_collection_validated": True}  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_record_collection_rejects_order_identity_and_global_failure_tamper() -> None:
    collection = _validated_collection(SELECTION_ROLE)
    records = collection.records
    with pytest.raises(ContrastiveLfProtocolError):
        validate_record_collection(
            (records[1], records[0], *records[2:]), role_id=SELECTION_ROLE
        )
    with pytest.raises(ContrastiveLfProtocolError):
        validate_record_collection(records[:-1], role_id=SELECTION_ROLE)
    with pytest.raises(ContrastiveLfProtocolError):
        validate_record_collection((*records, records[-1]), role_id=SELECTION_ROLE)
    with pytest.raises(ContrastiveLfProtocolError):
        validate_record_collection(records, role_id=CONFIRMATION_ROLE)
    with pytest.raises(ContrastiveLfProtocolError):
        validate_record_collection(
            (
                replace(
                    records[0],
                    template=replace(records[0].template, record_id="f" * 64),
                ),
                *records[1:],
            ),
            role_id=SELECTION_ROLE,
        )

    failed_indexes = tuple(
        next(
            ordinal
            for ordinal, record in enumerate(records)
            if record.template.record_kind == record_kind
        )
        for record_kind in (
            "base_generation",
            "attacked_observation",
            "detector",
            "budget",
            "quality",
        )
    )
    five_failures = tuple(
        _record(
            record.template,
            execution_status="failed",
            failure_class="operation_failure",
            failure_reason="bounded_record_failure",
        )
        if ordinal in failed_indexes
        else record
        for ordinal, record in enumerate(records)
    )
    with pytest.raises(ContrastiveLfProtocolError):
        validate_record_collection(five_failures, role_id=SELECTION_ROLE)

    forged_reports = tuple(
        DenominatorReport(
            record_kind=report.record_kind,
            expected_record_count=report.expected_record_count,
            completed_record_count=report.expected_record_count - 1,
            failed_record_count=1,
            unstarted_record_count=0,
        )
        for report in collection.denominator_reports
    )
    with pytest.raises(ContrastiveLfProtocolError):
        replace(
            _result(
                collection,
                gate_reports=tuple(
                    GateReport(gate_id, "not_evaluable")
                    for gate_id in GATE_ORDER
                ),
                result_classification="insufficient_evidence",
                selected_candidate_id=None,
            ),
            denominator_reports=forged_reports,
        ).validate(collection)


@pytest.mark.unit
def test_tamper_extra_missing_and_blind_private_inputs_are_rejected(
    tmp_path: Path,
) -> None:
    configuration = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    configuration["attacks"].append({"attack_id": "unregistered_attack"})
    payload = dict(configuration)
    payload.pop("config_digest")
    configuration["config_digest"] = canonical_digest(payload)
    config_path = tmp_path / "configuration.json"
    config_path.write_text(
        json.dumps(configuration, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ContrastiveLfProtocolError):
        load_configuration(config_path)

    manifest = json.loads(
        (ROOT / MANIFEST_PATHS[NULL_FIT_ROLE]).read_text(encoding="utf-8")
    )
    manifest["entries"][0]["source_row"] = 999
    manifest["entries_digest"] = canonical_digest(manifest["entries"])
    manifest_payload = dict(manifest)
    manifest_payload.pop("manifest_digest")
    manifest["manifest_digest"] = canonical_digest(manifest_payload)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ContrastiveLfProtocolError):
        load_manifest(manifest_path, expected_role=NULL_FIT_ROLE)

    roster = json.loads(
        (ROOT / PROMPT_ROSTER_PATH).read_text(encoding="utf-8")
    )
    roster["extra_field"] = "forbidden"
    roster_path = tmp_path / "roster.json"
    roster_path.write_text(json.dumps(roster), encoding="utf-8")
    with pytest.raises(ContrastiveLfProtocolError):
        load_prompt_roster(roster_path)

    assert BLIND_DETECTOR_INPUTS == (
        "current_rgb8",
        "detection_key",
        "public_frozen_assets",
    )
    assert set(FORBIDDEN_DETECTOR_INPUTS) == {
        "reference_image",
        "prompt",
        "embed_record",
        "private_latent",
        "embed_route",
        "qk_cache",
    }


@pytest.mark.unit
def test_field_registry_covers_contrastive_lf_persisted_protocol_surface() -> None:
    registry = (ROOT / "docs/reference/field_registry.md").read_text(
        encoding="utf-8"
    )
    for field_name in (
        "slot_ordinal",
        "manifest_entries_digest",
        "candidate_role_label",
        "arm_id",
        "internal_decoy_scores",
        "internal_decoy_roster_identity",
        "internal_decoy_roster_digest",
        "external_wrong_key_roster_identity",
        "external_wrong_key_roster_digest",
        "registered_score",
        "wrong_key_score",
        "population_mean",
        "population_variance",
        "population_sigma",
        "null_asset_digest",
        "provisional_threshold_digest",
        "record_collection_digest",
        "nonfinite_detected",
        "expected_record_count",
        "completed_record_count",
        "failed_record_count",
        "unstarted_record_count",
        "denominator_reports",
        "gate_reports",
        "first_failed_gate",
        "result_classification",
        "candidate_selection_passed",
        "confirmation_passed",
        "selected_candidate_id",
        "candidate_promoted",
        "formal_tau_created",
        "formal_fpr_created",
    ):
        assert f"| {field_name} |" in registry

    registered_fields = {
        line.split("|")[1].strip()
        for line in registry.splitlines()
        if line.startswith("|") and len(line.split("|")) >= 3
    }
    persisted_keys: set[str] = set()

    def collect_mapping_keys(value: object) -> None:
        if isinstance(value, dict):
            persisted_keys.update(value)
            for child in value.values():
                collect_mapping_keys(child)
        elif isinstance(value, list):
            for child in value:
                collect_mapping_keys(child)

    for path in NEW_CONFIG_PATHS:
        collect_mapping_keys(json.loads((ROOT / path).read_text(encoding="utf-8")))
    assert not persisted_keys - registered_fields


@pytest.mark.unit
def test_protocol_uses_only_semantic_candidate_and_role_identities() -> None:
    assert PROTOCOL_ID == "contrastive_lf_branch_attribution"
    assert CANDIDATE_IDS == (
        "lf_multiscale_lowpass_contrastive",
        "lf_five_by_five_lowpass_contrastive",
    )
    assert CANDIDATE_ROLE_LABELS == (
        "multiscale_primary_candidate",
        "single_scale_fallback_candidate",
    )
    source = (
        ROOT / "experiments/protocol/contrastive_lf_branch_attribution.py"
    ).read_text(encoding="utf-8")
    assert re.search(r"\b(?:A1|A2)\b", source) is None
    for forbidden_identity in ("protocol_v1", "gate_1", "phase_1"):
        assert forbidden_identity not in source
