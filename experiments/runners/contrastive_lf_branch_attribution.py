"""Fixed-denominator Stage-A null-fit and candidate-selection runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import time
from typing import Callable, Protocol, Sequence

import torch

from experiments.attacks.contrastive_lf_branch_attribution import (
    apply_contrastive_lf_attack,
    validate_jpeg_capability,
)
from experiments.metrics.contrastive_lf_branch_attribution import (
    StageABranchCase,
    StageAQualityCase,
    evaluate_stage_a_candidate_gates,
    evaluate_stage_a_hf_anchor,
)
from experiments.runners.development_persistence import (
    SOFT_STOP_SECONDS,
    STAGE_A_SNAPSHOT_INTERVAL_SECONDS,
    StageACommittedUnitStore,
    canonical_json_bytes,
)
from experiments.protocol.contrastive_lf_branch_attribution import (
    ATTACKS,
    CANDIDATE_IDS,
    CONFIG_DIGEST,
    GATE_ORDER,
    HF_CANDIDATE_ID,
    MANIFEST_DIGESTS,
    MULTISCALE_CANDIDATE_ID,
    NULL_FIT_ROLE,
    PROTOCOL_ID,
    SELECTION_ROLE,
    SINGLE_SCALE_CANDIDATE_ID,
    ContrastiveLfManifest,
    ContrastiveLfProtocolResult,
    ContrastiveLfRecord,
    ContrastiveLfRecordTemplate,
    GateReport,
    build_record_templates,
    canonical_digest,
    choose_selection_winner,
    load_manifest,
    validate_failure_tail,
    validate_record_collection,
)
from main import (
    ContrastiveLfDetectionResult,
    ContrastiveLfNullAsset,
    ContrastiveLfRawObservation,
    DerivedWrongKeyMaterial,
    HfDetectionResult,
    HfPopulationNullAsset,
    fit_contrastive_lf_null_asset,
    fit_hf_population_null_asset,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


class ContrastiveLfRunnerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StageAGeneration:
    arm_id: str
    image_rgb8: torch.Tensor
    clean_image_rgb8: torch.Tensor
    materialization_replay_identity: str | None
    replay_digest: str | None
    budget_status: str | None
    paired_rgb8_mse: float


@dataclass(frozen=True, slots=True)
class StageADetection:
    raw_score: float
    standardized_score: float
    internal_decoy_scores: tuple[float, ...]
    null_asset_digest: str
    provisional_threshold_digest: str
    detector_identity: str


@dataclass(frozen=True, slots=True)
class StageANullFitArtifact:
    schema_version: int
    protocol_id: str
    role_id: str
    null_manifest_digest: str
    implementation_revision: str
    method_config_digest: str
    model_identity: str
    runtime_identity: str
    codec_identity: str
    record_collection_digest: str
    records: tuple[ContrastiveLfRecord, ...]
    hf_null_asset: HfPopulationNullAsset
    multiscale_null_asset: ContrastiveLfNullAsset
    single_scale_null_asset: ContrastiveLfNullAsset
    candidate_promoted: bool = False
    formal_tau_created: bool = False
    formal_fpr_created: bool = False
    full_ceg_wm_eligible: bool = False
    diagnostic_only: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)

    def validate(
        self,
        manifest: ContrastiveLfManifest,
        *,
        operations: StageAOperations | None = None,
    ) -> None:
        """Authenticate the full null population and its exact record replay."""

        manifest.validate()
        expected_templates = build_record_templates(manifest)
        if (
            self.schema_version != 1
            or self.protocol_id != PROTOCOL_ID
            or self.role_id != NULL_FIT_ROLE
            or manifest.role_id != NULL_FIT_ROLE
            or self.null_manifest_digest != manifest.manifest_digest
            or self.null_manifest_digest != MANIFEST_DIGESTS[NULL_FIT_ROLE]
            or self.method_config_digest != CONFIG_DIGEST
            or len(self.records) != len(expected_templates)
            or any(
                record.template != template
                or record.execution_status != "completed"
                for record, template in zip(self.records, expected_templates, strict=True)
            )
            or self.record_collection_digest
            != canonical_digest([record.canonical_payload() for record in self.records])
            or any(
                value is not False
                for value in (
                    self.candidate_promoted,
                    self.formal_tau_created,
                    self.formal_fpr_created,
                    self.full_ceg_wm_eligible,
                )
            )
            or self.diagnostic_only is not True
        ):
            raise ContrastiveLfRunnerError("null-fit artifact authority is invalid")
        for record in self.records:
            record.validate()
        try:
            self.hf_null_asset.validate()
            self.multiscale_null_asset.validate()
            self.single_scale_null_asset.validate()
        except Exception as exc:
            raise ContrastiveLfRunnerError("null-fit asset digest drifted") from exc
        if (
            self.hf_null_asset.null_manifest_digest != self.null_manifest_digest
            or self.multiscale_null_asset.null_manifest_digest != self.null_manifest_digest
            or self.single_scale_null_asset.null_manifest_digest != self.null_manifest_digest
            or self.multiscale_null_asset.candidate_id != MULTISCALE_CANDIDATE_ID
            or self.single_scale_null_asset.candidate_id != SINGLE_SCALE_CANDIDATE_ID
        ):
            raise ContrastiveLfRunnerError("null-fit asset binding drifted")
        if operations is not None and any(
            observed != expected
            for observed, expected in (
                (self.method_config_digest, operations.method_config_digest),
                (self.model_identity, operations.model_identity),
                (self.runtime_identity, operations.runtime_identity),
                (self.codec_identity, operations.codec_identity),
            )
        ):
            raise ContrastiveLfRunnerError("null-fit execution identity drifted")

    @property
    def artifact_digest(self) -> str:
        return canonical_digest(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class StageAExecutionResult:
    null_fit_records: tuple[ContrastiveLfRecord, ...]
    null_fit_artifact: StageANullFitArtifact | None
    selection_records: tuple[ContrastiveLfRecord, ...]
    selection_result: ContrastiveLfProtocolResult | None
    result_classification: str
    failure_reason: str | None
    science_started: bool = False
    scientific_unit_count: int = 0
    candidate_promoted: bool = False
    formal_tau_created: bool = False
    formal_fpr_created: bool = False

    def validate_for_delivery(self) -> None:
        """Bind delivery to exact frozen records without creating new authority."""

        if (
            self.result_classification
            not in {"success", "scientific_failure", "insufficient_evidence", "operational_failure"}
            or self.science_started is not False
            or self.scientific_unit_count != 0
            or self.candidate_promoted is not False
            or self.formal_tau_created is not False
            or self.formal_fpr_created is not False
            or (self.failure_reason is not None and len(self.failure_reason) > 120)
        ):
            raise ContrastiveLfRunnerError("Stage-A delivery boundary is invalid")
        if not self.null_fit_records and not self.selection_records:
            if self.result_classification != "operational_failure" or not self.failure_reason:
                raise ContrastiveLfRunnerError("empty Stage-A delivery is not a pre-execution failure")
            return
        null_manifest = load_manifest(
            PACKAGE_ROOT / "configs/experiments/contrastive_lf_null_fit_manifest.json",
            expected_role=NULL_FIT_ROLE,
        )
        _validate_role_records(self.null_fit_records, null_manifest)
        if self.null_fit_artifact is None:
            if self.selection_records or self.selection_result is not None:
                raise ContrastiveLfRunnerError("selection exists without null authority")
            return
        self.null_fit_artifact.validate(null_manifest)
        if self.null_fit_records != self.null_fit_artifact.records:
            raise ContrastiveLfRunnerError("null-fit result records drifted from artifact")
        if not self.selection_records:
            if self.selection_result is not None:
                raise ContrastiveLfRunnerError("selection result lacks records")
            return
        selection_collection = validate_record_collection(
            self.selection_records, role_id=SELECTION_ROLE
        )
        if self.selection_result is None:
            if self.result_classification != "operational_failure":
                raise ContrastiveLfRunnerError("selection records lack governed result")
            return
        self.selection_result.validate(selection_collection)
        if self.result_classification != self.selection_result.result_classification:
            raise ContrastiveLfRunnerError("Stage-A result classification drifted")


@dataclass(frozen=True, slots=True)
class StageAResumableOutcome:
    run_id: str
    run_root: str
    session_id: str
    session_status: str
    execution_result: StageAExecutionResult | None
    completed_null_fit_units: int
    completed_selection_units: int
    producer_revisions: tuple[str, ...]
    cache_diagnostics: dict[str, int]
    most_recent_snapshot_path: str | None


class StageAOperations(Protocol):
    implementation_revision: str
    method_config_digest: str
    model_identity: str
    runtime_identity: str
    codec_identity: str
    root_key: str

    def clean(self, entry: object) -> StageAGeneration: ...
    def write(self, entry: object, arm_id: str) -> StageAGeneration: ...
    def attack(self, entry: object, generation: StageAGeneration, attack_id: str) -> torch.Tensor: ...
    def prepare_public_observation(self, image_rgb8: torch.Tensor) -> object: ...
    def observe_hf_raw(self, prepared: object, key: str | DerivedWrongKeyMaterial) -> HfDetectionResult: ...
    def observe_lf_raw(self, prepared: object, key: str | DerivedWrongKeyMaterial, candidate_id: str) -> ContrastiveLfRawObservation: ...
    def observe_hf(self, prepared: object, key: str | DerivedWrongKeyMaterial, asset: HfPopulationNullAsset) -> StageADetection: ...
    def observe_lf(self, prepared: object, key: str | DerivedWrongKeyMaterial, asset: ContrastiveLfNullAsset) -> StageADetection: ...
    def score_lf_raw(self, raw: ContrastiveLfRawObservation, asset: ContrastiveLfNullAsset) -> ContrastiveLfDetectionResult: ...
    def cache_diagnostics(self) -> dict[str, int]: ...
    def wrong_key(self, index: int) -> DerivedWrongKeyMaterial: ...
    def close(self) -> None: ...


def _validate_role_records(
    records: Sequence[ContrastiveLfRecord], manifest: ContrastiveLfManifest
) -> None:
    expected = build_record_templates(manifest)
    if len(records) != len(expected):
        raise ContrastiveLfRunnerError("Stage-A record denominator drifted")
    for record, template in zip(records, expected, strict=True):
        if type(record) is not ContrastiveLfRecord or record.template != template:
            raise ContrastiveLfRunnerError("Stage-A record template drifted")
        record.validate()
    validate_failure_tail(records)


def _empty_record(template: ContrastiveLfRecordTemplate, operations: StageAOperations) -> ContrastiveLfRecord:
    return ContrastiveLfRecord(
        template=template,
        attempt_index=0,
        execution_status="unstarted",
        method_config_digest=CONFIG_DIGEST,
        implementation_revision=operations.implementation_revision,
        model_identity=operations.model_identity,
        runtime_identity=operations.runtime_identity,
        codec_identity=operations.codec_identity,
        raw_score=None,
        internal_decoy_scores=(),
        registered_score=None,
        wrong_key_score=None,
        primary_null_score=None,
        population_mean=None,
        population_variance=None,
        population_sigma=None,
        null_asset_digest=None,
        provisional_threshold_digest=None,
        z_score=None,
        key_margin=None,
        budget_status=None,
        materialization_replay_identity=None,
        replay_digest=None,
        nonfinite_detected=False,
        paired_rgb8_mse=None,
        failure_class=None,
        failure_reason=None,
    )


def _failure_class(error: BaseException) -> str:
    name = type(error).__name__.lower()
    if "memory" in name or "resource" in name:
        return "resource_failure"
    if "dependency" in name or "import" in name:
        return "dependency_failure"
    if "codec" in name or "jpeg" in name:
        return "codec_failure"
    if "runtime" in name or "backend" in name:
        return "runtime_failure"
    return "operation_failure"


def _fail(record: ContrastiveLfRecord, error: BaseException) -> ContrastiveLfRecord:
    return replace(
        record,
        execution_status="failed",
        failure_class=_failure_class(error),
        failure_reason=type(error).__name__[:120],
        nonfinite_detected=False,
    )


def _complete(record: ContrastiveLfRecord, **values: object) -> ContrastiveLfRecord:
    result = replace(record, execution_status="completed", **values)
    result.validate()
    return result


def _threshold_digest(candidate_id: str, asset_digest: str, tau: float) -> str:
    return canonical_digest(
        {
            "candidate_id": candidate_id,
            "null_asset_digest": asset_digest,
            "rule": "nextafter_fourth_largest_z_toward_positive_infinity",
            "tau_float64_hex": float.hex(tau),
        }
    )


def _record_from_payload(payload: object) -> ContrastiveLfRecord:
    if type(payload) is not dict:
        raise ContrastiveLfRunnerError("persisted Stage-A record is invalid")
    template_names = {field.name for field in fields(ContrastiveLfRecordTemplate)}
    if not template_names <= set(payload):
        raise ContrastiveLfRunnerError("persisted Stage-A template is incomplete")
    template_values = {name: payload[name] for name in template_names}
    template = ContrastiveLfRecordTemplate(**template_values)
    record_values = {
        field.name: payload[field.name]
        for field in fields(ContrastiveLfRecord)
        if field.name != "template"
    }
    record_values["internal_decoy_scores"] = tuple(
        record_values["internal_decoy_scores"]
    )
    record = ContrastiveLfRecord(template=template, **record_values)
    record.validate()
    return record


def _execute_null_cluster(
    manifest: ContrastiveLfManifest,
    operations: StageAOperations,
    records: list[ContrastiveLfRecord],
    cluster_ordinal: int,
    hf_raw: dict[str, HfDetectionResult],
    lf_raw: dict[tuple[str, str], ContrastiveLfRawObservation],
) -> tuple[bool, str | None]:
    entry = manifest.entries[cluster_ordinal]
    cluster = entry.source_cluster_id
    image: torch.Tensor | None = None
    public: object | None = None
    for index, record in enumerate(records):
        template = record.template
        if template.source_cluster_id != cluster:
            continue
        try:
            if template.record_kind == "clean_base_observation":
                generation = operations.clean(entry)
                image = generation.image_rgb8
                records[index] = _complete(record)
                continue
            if image is None:
                raise ContrastiveLfRunnerError("null-fit cluster image is unavailable")
            if public is None:
                public = operations.prepare_public_observation(image)
            if template.candidate_id == HF_CANDIDATE_ID:
                value = operations.observe_hf_raw(public, operations.root_key)
                hf_raw[cluster] = value
                records[index] = _complete(record, raw_score=float(value.hf_score))
            else:
                value = operations.observe_lf_raw(
                    public, operations.root_key, template.candidate_id
                )
                lf_raw[(cluster, template.candidate_id)] = value
                records[index] = _complete(
                    record,
                    raw_score=float(value.raw_feature[0]),
                    internal_decoy_scores=tuple(
                        feature[0] for feature in value.internal_decoy_features
                    ),
                )
        except Exception as exc:
            records[index] = _fail(record, exc)
            return False, type(exc).__name__[:120]
    return True, None


def _fit_null_artifact(
    manifest: ContrastiveLfManifest,
    operations: StageAOperations,
    records: list[ContrastiveLfRecord],
    hf_raw: dict[str, HfDetectionResult],
    lf_raw: dict[tuple[str, str], ContrastiveLfRawObservation],
) -> StageANullFitArtifact:
    ordered_clusters = tuple(entry.source_cluster_id for entry in manifest.entries)
    hf_asset = fit_hf_population_null_asset(
        tuple(float(hf_raw[cluster].hf_score) for cluster in ordered_clusters),
        null_manifest_digest=manifest.manifest_digest,
    )
    multiscale = fit_contrastive_lf_null_asset(
        tuple(
            lf_raw[(cluster, MULTISCALE_CANDIDATE_ID)]
            for cluster in ordered_clusters
        ),
        candidate_id=MULTISCALE_CANDIDATE_ID,
        null_manifest_digest=manifest.manifest_digest,
    )
    single = fit_contrastive_lf_null_asset(
        tuple(
            lf_raw[(cluster, SINGLE_SCALE_CANDIDATE_ID)]
            for cluster in ordered_clusters
        ),
        candidate_id=SINGLE_SCALE_CANDIDATE_ID,
        null_manifest_digest=manifest.manifest_digest,
    )
    assets = {
        HF_CANDIDATE_ID: hf_asset,
        MULTISCALE_CANDIDATE_ID: multiscale,
        SINGLE_SCALE_CANDIDATE_ID: single,
    }
    for index, record in enumerate(records):
        if record.template.record_kind != "null_statistic":
            continue
        candidate = record.template.candidate_id
        asset = assets[candidate]
        if candidate == HF_CANDIDATE_ID:
            raw = float(hf_raw[record.template.source_cluster_id].hf_score)
            z = (raw - hf_asset.population_mean) / hf_asset.population_sigma
            records[index] = replace(
                record,
                population_mean=hf_asset.population_mean,
                population_variance=hf_asset.population_variance,
                population_sigma=hf_asset.population_sigma,
                null_asset_digest=hf_asset.asset_digest,
                provisional_threshold_digest=_threshold_digest(
                    candidate, hf_asset.asset_digest, hf_asset.provisional_tau
                ),
                z_score=z,
            )
        else:
            raw_observation = lf_raw[
                (record.template.source_cluster_id, candidate)
            ]
            detected = operations.score_lf_raw(raw_observation, asset)
            records[index] = replace(
                record,
                raw_score=detected.contrastive_score,
                internal_decoy_scores=detected.internal_decoy_scores,
                population_mean=asset.contrastive_population_mean,
                population_variance=asset.contrastive_population_variance,
                population_sigma=asset.contrastive_population_sigma,
                null_asset_digest=asset.asset_digest,
                provisional_threshold_digest=_threshold_digest(
                    candidate, asset.asset_digest, asset.provisional_tau
                ),
                z_score=detected.standardized_score,
            )
        records[index].validate()
    digest = canonical_digest([record.canonical_payload() for record in records])
    artifact = StageANullFitArtifact(
        schema_version=1,
        protocol_id=PROTOCOL_ID,
        role_id=NULL_FIT_ROLE,
        null_manifest_digest=manifest.manifest_digest,
        implementation_revision=operations.implementation_revision,
        method_config_digest=operations.method_config_digest,
        model_identity=operations.model_identity,
        runtime_identity=operations.runtime_identity,
        codec_identity=operations.codec_identity,
        record_collection_digest=digest,
        records=tuple(records),
        hf_null_asset=hf_asset,
        multiscale_null_asset=multiscale,
        single_scale_null_asset=single,
    )
    artifact.validate(manifest, operations=operations)
    return artifact


def execute_null_fit(
    manifest: ContrastiveLfManifest, operations: StageAOperations
) -> tuple[tuple[ContrastiveLfRecord, ...], StageANullFitArtifact | None, str | None]:
    manifest.validate()
    if manifest.role_id != NULL_FIT_ROLE:
        raise ContrastiveLfRunnerError("null-fit manifest role mismatch")
    templates = build_record_templates(manifest)
    records = [_empty_record(template, operations) for template in templates]
    hf_raw: dict[str, HfDetectionResult] = {}
    lf_raw: dict[tuple[str, str], ContrastiveLfRawObservation] = {}
    for cluster_ordinal in range(len(manifest.entries)):
        completed, failure = _execute_null_cluster(
            manifest, operations, records, cluster_ordinal, hf_raw, lf_raw
        )
        if not completed:
            return tuple(records), None, failure
    try:
        artifact = _fit_null_artifact(
            manifest, operations, records, hf_raw, lf_raw
        )
        return tuple(records), artifact, None
    except Exception as exc:
        records[-1] = _fail(_empty_record(records[-1].template, operations), exc)
        return tuple(records), None, type(exc).__name__[:120]


def _selection_result(
    records: Sequence[ContrastiveLfRecord],
    *,
    candidate_results: dict[str, object] | None,
    hf_anchor: bool | None,
) -> ContrastiveLfProtocolResult:
    collection = validate_record_collection(records, role_id=SELECTION_ROLE)
    complete = collection.denominator_complete
    if not complete:
        gate_reports = tuple(GateReport(gate, "not_evaluable") for gate in GATE_ORDER)
        winner = None
    else:
        assert candidate_results is not None and hf_anchor is not None
        multiscale = candidate_results[MULTISCALE_CANDIDATE_ID]
        single = candidate_results[SINGLE_SCALE_CANDIDATE_ID]
        budget = all(
            record.budget_status == "accepted"
            for record in records
            if record.template.record_kind == "budget"
        )
        attribution_candidates = {
            candidate
            for candidate, value in candidate_results.items()
            if value.identity_attribution_passed
            and value.condition_false_positive_passed
        }
        blur_candidates = {
            candidate
            for candidate in attribution_candidates
            if candidate_results[candidate].blur_complement_passed
        }
        quality_candidates = {
            candidate
            for candidate in blur_candidates
            if candidate_results[candidate].quality_passed
        }
        candidate_attr = bool(attribution_candidates)
        blur = bool(blur_candidates)
        quality = bool(quality_candidates)
        statuses = (True, budget, hf_anchor, candidate_attr, blur, quality)
        gate_reports = tuple(
            GateReport(gate, "passed" if status else "failed")
            for gate, status in zip(GATE_ORDER, statuses, strict=True)
        )
        winner = choose_selection_winner(
            multiscale_passed=MULTISCALE_CANDIDATE_ID in quality_candidates,
            single_scale_passed=SINGLE_SCALE_CANDIDATE_ID in quality_candidates,
        ) if all(statuses[:3]) and quality else None
    gates_pass = all(report.gate_status == "passed" for report in gate_reports)
    result = ContrastiveLfProtocolResult(
        schema_version=1,
        protocol_id=PROTOCOL_ID,
        role_id=SELECTION_ROLE,
        sample_manifest_digest=MANIFEST_DIGESTS[SELECTION_ROLE],
        manifest_entries_digest=records[0].template.manifest_entries_digest,
        record_collection_digest=collection.record_collection_digest,
        denominator_reports=collection.denominator_reports,
        gate_reports=gate_reports,
        first_failed_gate=next((report.gate_id for report in gate_reports if report.gate_status == "failed"), None),
        result_classification=(
            "operational_failure"
            if collection.operational_failure_observed
            else "insufficient_evidence"
            if not complete
            else "success"
            if gates_pass and winner is not None
            else "scientific_failure"
        ),
        candidate_selection_passed=gates_pass and winner is not None,
        confirmation_passed=False,
        selected_candidate_id=winner if gates_pass else None,
        candidate_promoted=False,
        formal_tau_created=False,
        formal_fpr_created=False,
        full_ceg_wm_eligible=False,
    )
    result.validate(collection)
    return result


def _execute_selection_cluster(
    manifest: ContrastiveLfManifest,
    operations: StageAOperations,
    records: list[ContrastiveLfRecord],
    cluster_ordinal: int,
    assets: dict[str, object],
    detections: dict[tuple[str, str, str, str, int | None], StageADetection],
) -> tuple[bool, str | None]:
    entry = manifest.entries[cluster_ordinal]
    cluster = entry.source_cluster_id
    generations: dict[str, StageAGeneration] = {}
    attacked: dict[tuple[str, str], torch.Tensor] = {}
    prepared: dict[tuple[str, str], object] = {}
    for index, record in enumerate(records):
        template = record.template
        if template.source_cluster_id != cluster:
            continue
        try:
            if template.record_kind == "base_generation":
                generation = (
                    operations.clean(entry)
                    if template.arm_id == "clean_unwatermarked"
                    else operations.write(entry, template.arm_id)
                )
                if generation.arm_id != template.arm_id or not isfinite(
                    generation.paired_rgb8_mse
                ):
                    raise ContrastiveLfRunnerError("generation identity drifted")
                generations[template.arm_id] = generation
                records[index] = _complete(record)
            elif template.record_kind == "attacked_observation":
                generation = generations[template.arm_id]
                image = operations.attack(entry, generation, template.attack_id)
                attacked[(template.arm_id, template.attack_id)] = image
                records[index] = _complete(record)
            elif template.record_kind == "detector":
                image_key = (template.arm_id, template.attack_id)
                image = attacked[image_key]
                public = prepared.get(image_key)
                if public is None:
                    public = operations.prepare_public_observation(image)
                    prepared[image_key] = public
                key = (
                    operations.root_key
                    if template.key_role == "registered"
                    else operations.wrong_key(template.wrong_key_index)
                )
                asset = assets[template.candidate_id]
                detection = (
                    operations.observe_hf(public, key, asset)
                    if template.candidate_id == HF_CANDIDATE_ID
                    else operations.observe_lf(public, key, asset)
                )
                detections[
                    (
                        cluster,
                        template.candidate_id,
                        template.attack_id,
                        template.control_identity,
                        template.wrong_key_index,
                    )
                ] = detection
                bound: dict[str, float] = {}
                if template.control_identity == "registered_attribution":
                    bound["registered_score"] = detection.raw_score
                elif template.control_identity == "external_wrong_key":
                    bound["wrong_key_score"] = detection.raw_score
                else:
                    bound["primary_null_score"] = detection.raw_score
                records[index] = _complete(
                    record,
                    raw_score=detection.raw_score,
                    internal_decoy_scores=detection.internal_decoy_scores,
                    null_asset_digest=detection.null_asset_digest,
                    provisional_threshold_digest=detection.provisional_threshold_digest,
                    z_score=detection.standardized_score,
                    **bound,
                )
            elif template.record_kind == "budget":
                generation = generations[template.arm_id]
                records[index] = _complete(
                    record,
                    budget_status=generation.budget_status,
                    materialization_replay_identity=(
                        generation.materialization_replay_identity
                    ),
                    replay_digest=generation.replay_digest,
                )
            else:
                candidate_image = attacked[(template.arm_id, template.attack_id)]
                clean_image = attacked[("clean_unwatermarked", template.attack_id)]
                mse = float(
                    torch.mean(
                        (
                            (candidate_image.to(torch.float32) - clean_image.to(torch.float32))
                            / 255.0
                        )
                        ** 2
                    ).item()
                )
                if not isfinite(mse):
                    raise ContrastiveLfRunnerError("paired RGB8 MSE is non-finite")
                records[index] = _complete(record, paired_rgb8_mse=mse)
        except Exception as exc:
            records[index] = _fail(record, exc)
            return False, type(exc).__name__[:120]
    return True, None


def _evaluate_completed_selection(
    records: Sequence[ContrastiveLfRecord],
    assets: dict[str, object],
) -> ContrastiveLfProtocolResult:
    def detection_for(
        cluster: str,
        candidate: str,
        attack_id: str,
        control: str,
        wrong_index: int | None = None,
    ) -> ContrastiveLfRecord:
        return next(
            record
            for record in records
            if record.template.record_kind == "detector"
            and record.template.source_cluster_id == cluster
            and record.template.candidate_id == candidate
            and record.template.attack_id == attack_id
            and record.template.control_identity == control
            and record.template.wrong_key_index == wrong_index
        )

    branch_by_candidate: dict[str, list[StageABranchCase]] = {
        candidate: [] for candidate in CANDIDATE_IDS
    }
    quality_by_candidate: dict[str, list[StageAQualityCase]] = {
        candidate: [] for candidate in CANDIDATE_IDS
    }
    clusters = tuple(
        dict.fromkeys(record.template.source_cluster_id for record in records)
    )
    for cluster in clusters:
        for attack_id in ATTACKS:
            hf_registered = detection_for(
                cluster, HF_CANDIDATE_ID, attack_id, "registered_attribution"
            )
            hf_null = detection_for(
                cluster, HF_CANDIDATE_ID, attack_id, "paired_primary_null"
            )
            hf_wrong = tuple(
                detection_for(
                    cluster,
                    HF_CANDIDATE_ID,
                    attack_id,
                    "external_wrong_key",
                    index,
                ).z_score
                for index in range(8)
            )
            for candidate in CANDIDATE_IDS:
                registered = detection_for(
                    cluster, candidate, attack_id, "registered_attribution"
                )
                primary = detection_for(
                    cluster, candidate, attack_id, "paired_primary_null"
                )
                wrong = tuple(
                    detection_for(
                        cluster,
                        candidate,
                        attack_id,
                        "external_wrong_key",
                        index,
                    ).z_score
                    for index in range(8)
                )
                assert (
                    registered.z_score is not None
                    and primary.z_score is not None
                    and hf_registered.z_score is not None
                    and hf_null.z_score is not None
                    and all(value is not None for value in (*wrong, *hf_wrong))
                )
                branch_by_candidate[candidate].append(
                    StageABranchCase(
                        cluster,
                        attack_id,
                        registered.z_score,
                        primary.z_score,
                        tuple(float(value) for value in wrong),
                        hf_registered.z_score,
                        hf_null.z_score,
                        tuple(float(value) for value in hf_wrong),
                    )
                )
                candidate_quality = next(
                    record.paired_rgb8_mse
                    for record in records
                    if record.template.record_kind == "quality"
                    and record.template.source_cluster_id == cluster
                    and record.template.candidate_id == candidate
                    and record.template.attack_id == attack_id
                )
                hf_quality = next(
                    record.paired_rgb8_mse
                    for record in records
                    if record.template.record_kind == "quality"
                    and record.template.source_cluster_id == cluster
                    and record.template.candidate_id == HF_CANDIDATE_ID
                    and record.template.attack_id == attack_id
                )
                assert candidate_quality is not None and hf_quality is not None
                quality_by_candidate[candidate].append(
                    StageAQualityCase(
                        cluster, attack_id, candidate_quality, hf_quality
                    )
                )
    candidate_results = {
        candidate: evaluate_stage_a_candidate_gates(
            candidate,
            branch_by_candidate[candidate],
            quality_by_candidate[candidate],
            candidate_tau=assets[candidate].provisional_tau,
            hf_tau=assets[HF_CANDIDATE_ID].provisional_tau,
        )
        for candidate in CANDIDATE_IDS
    }
    hf_anchor = evaluate_stage_a_hf_anchor(
        branch_by_candidate[MULTISCALE_CANDIDATE_ID],
        hf_tau=assets[HF_CANDIDATE_ID].provisional_tau,
    )
    return _selection_result(
        records, candidate_results=candidate_results, hf_anchor=hf_anchor
    )


def execute_selection(
    manifest: ContrastiveLfManifest,
    operations: StageAOperations,
    null_artifact: StageANullFitArtifact,
) -> tuple[tuple[ContrastiveLfRecord, ...], ContrastiveLfProtocolResult]:
    manifest.validate()
    if manifest.role_id != SELECTION_ROLE:
        raise ContrastiveLfRunnerError("selection manifest role mismatch")
    if type(null_artifact) is not StageANullFitArtifact:
        raise ContrastiveLfRunnerError("authenticated null-fit artifact is required")
    null_manifest = load_manifest(
        PACKAGE_ROOT / "configs/experiments/contrastive_lf_null_fit_manifest.json",
        expected_role=NULL_FIT_ROLE,
    )
    null_artifact.validate(null_manifest, operations=operations)
    templates = build_record_templates(manifest)
    records = [_empty_record(template, operations) for template in templates]
    detections: dict[tuple[str, str, str, str, int | None], StageADetection] = {}
    assets = {
        HF_CANDIDATE_ID: null_artifact.hf_null_asset,
        MULTISCALE_CANDIDATE_ID: null_artifact.multiscale_null_asset,
        SINGLE_SCALE_CANDIDATE_ID: null_artifact.single_scale_null_asset,
    }
    for cluster_ordinal in range(len(manifest.entries)):
        completed, _failure = _execute_selection_cluster(
            manifest,
            operations,
            records,
            cluster_ordinal,
            assets,
            detections,
        )
        if not completed:
            return tuple(records), _selection_result(
                records, candidate_results=None, hf_anchor=None
            )
    return tuple(records), _evaluate_completed_selection(records, assets)


def execute_stage_a_null_fit_and_selection(
    null_manifest: ContrastiveLfManifest,
    selection_manifest: ContrastiveLfManifest,
    operations: StageAOperations,
) -> StageAExecutionResult:
    null_records, artifact, failure = execute_null_fit(null_manifest, operations)
    if artifact is None:
        failed = next(
            (record for record in null_records if record.execution_status == "failed"),
            None,
        )
        classification = (
            "operational_failure"
            if failed is not None
            and failed.failure_class
            in {"dependency_failure", "runtime_failure", "codec_failure", "resource_failure"}
            else "insufficient_evidence"
        )
        return StageAExecutionResult(
            null_records,
            None,
            (),
            None,
            classification,
            failure,
        )
    selection_records, selection_result = execute_selection(selection_manifest, operations, artifact)
    return StageAExecutionResult(
        null_fit_records=null_records,
        null_fit_artifact=artifact,
        selection_records=selection_records,
        selection_result=selection_result,
        result_classification=selection_result.result_classification,
        failure_reason=(selection_result.result_classification if selection_result.result_classification in {"operational_failure", "insufficient_evidence"} else None),
    )


def _lf_raw_from_payload(payload: object) -> ContrastiveLfRawObservation:
    if type(payload) is not dict:
        raise ContrastiveLfRunnerError("persisted LF raw observation is invalid")
    values = dict(payload)
    values["raw_feature"] = tuple(values["raw_feature"])
    values["internal_decoy_features"] = tuple(
        tuple(item) for item in values["internal_decoy_features"]
    )
    result = ContrastiveLfRawObservation(**values)
    return result


def _hf_raw_from_payload(payload: object) -> HfDetectionResult:
    if type(payload) is not dict:
        raise ContrastiveLfRunnerError("persisted HF raw observation is invalid")
    result = HfDetectionResult(**payload)
    return result


def _cluster_records(
    records: Sequence[ContrastiveLfRecord], source_cluster_id: str
) -> tuple[ContrastiveLfRecord, ...]:
    return tuple(
        record
        for record in records
        if record.template.source_cluster_id == source_cluster_id
    )


def execute_stage_a_resumable(
    null_manifest: ContrastiveLfManifest,
    selection_manifest: ContrastiveLfManifest,
    operations: StageAOperations,
    *,
    runs_root: str | Path,
    new_run_id: str,
    session_id: str,
    package_sha256: str,
    stop_requested: Callable[[], bool] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    resolved_run_callback: Callable[[StageACommittedUnitStore], None] | None = None,
) -> StageAResumableOutcome:
    """Execute Stage A as 64 authenticated cluster units with safe resume."""

    from datetime import datetime, timezone

    null_manifest.validate()
    selection_manifest.validate()
    if (
        null_manifest.role_id != NULL_FIT_ROLE
        or selection_manifest.role_id != SELECTION_ROLE
    ):
        raise ContrastiveLfRunnerError("Stage-A resumable manifest role drifted")
    root_digest = getattr(operations, "root_key_public_digest", None)
    observation_identity = getattr(operations, "observation_behavior_identity", None)
    if not isinstance(root_digest, str) or not isinstance(observation_identity, str):
        raise ContrastiveLfRunnerError("Stage-A resumable public identity is unavailable")
    behavior_identity = {
        "candidate_ids": list(CANDIDATE_IDS),
        "codec_identity": operations.codec_identity,
        "method_config_digest": operations.method_config_digest,
        "model_identity": operations.model_identity,
        "null_manifest_digest": null_manifest.manifest_digest,
        "observation_behavior_identity": observation_identity,
        "protocol_id": PROTOCOL_ID,
        "public_root_key_digest": root_digest,
        "runtime_identity": operations.runtime_identity,
        "selection_manifest_digest": selection_manifest.manifest_digest,
    }
    utc_now = lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    store = StageACommittedUnitStore.discover_or_create(
        runs_root,
        behavior_identity=behavior_identity,
        new_run_id=new_run_id,
        created_at_utc=utc_now(),
        initial_producer_revision=operations.implementation_revision,
    )
    if resolved_run_callback is not None:
        resolved_run_callback(store)
    started = monotonic()
    last_snapshot = started
    snapshot_index = len(tuple((store.run_root / "snapshots").glob("*.zip")))
    latest_snapshot: str | None = None

    def units_by_phase() -> dict[tuple[str, int], dict[str, object]]:
        return {
            (unit["phase"], unit["cluster_ordinal"]): unit
            for unit in store.committed_units()
        }

    def progress_event(phase: str, recent: str | None) -> None:
        units = units_by_phase()
        elapsed = monotonic() - started
        cache = operations.cache_diagnostics()
        completed_null = sum(key[0] == "null_fit" for key in units)
        completed_selection = sum(key[0] == "candidate_selection" for key in units)
        completed = completed_null + completed_selection
        eta = None if completed == 0 else (elapsed / completed) * (64 - completed)
        event = {
            "cache_diagnostics": cache,
            "completed": completed_null if phase == "null_fit" else completed_selection,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "most_recent_snapshot_path": latest_snapshot,
            "phase": phase,
            "recent_unit": recent,
            "run_id": store.run_root.name,
            "session_id": session_id,
            "total": 32,
        }
        store.append_heartbeat(event)
        print(json.dumps(event, sort_keys=True), flush=True)

    def safe_snapshot(reason: str) -> None:
        nonlocal snapshot_index, latest_snapshot, last_snapshot
        units = store.committed_units()
        revisions = sorted({unit["producer_revision"] for unit in units})
        receipt = store.write_snapshot(
            session_id=session_id,
            snapshot_index=snapshot_index,
            payload={
                "behavior_identity_digest": store.behavior_identity_digest,
                "cache_diagnostics": operations.cache_diagnostics(),
                "committed_unit_count": len(units),
                "producer_revisions": revisions,
                "reason": reason,
                "run_id": store.run_root.name,
                "session_id": session_id,
            },
        )
        snapshot_index += 1
        latest_snapshot = receipt["archive_path"]
        last_snapshot = monotonic()

    def should_stop() -> bool:
        return (
            monotonic() - started >= SOFT_STOP_SECONDS
            or (stop_requested is not None and bool(stop_requested()))
        )

    def resumable(reason: str) -> StageAResumableOutcome:
        safe_snapshot(reason)
        units = store.committed_units()
        revisions = tuple(sorted({unit["producer_revision"] for unit in units}))
        receipt_payload = {
            "behavior_identity_digest": store.behavior_identity_digest,
            "cache_diagnostics": operations.cache_diagnostics(),
            "committed_unit_count": len(units),
            "ended_at_utc": utc_now(),
            "heterogeneous_revisions": len(revisions) > 1,
            "most_recent_snapshot_path": latest_snapshot,
            "producer_revision": operations.implementation_revision,
            "producer_revisions": list(revisions),
            "run_id": store.run_root.name,
            "session_id": session_id,
            "session_status": "interrupted_resumable",
            "termination_reason": reason,
        }
        store.write_session_receipt(session_id, receipt_payload)
        phase_units = units_by_phase()
        return StageAResumableOutcome(
            run_id=store.run_root.name,
            run_root=str(store.run_root),
            session_id=session_id,
            session_status="interrupted_resumable",
            execution_result=None,
            completed_null_fit_units=sum(key[0] == "null_fit" for key in phase_units),
            completed_selection_units=sum(
                key[0] == "candidate_selection" for key in phase_units
            ),
            producer_revisions=revisions,
            cache_diagnostics=operations.cache_diagnostics(),
            most_recent_snapshot_path=latest_snapshot,
        )

    null_records = [
        _empty_record(template, operations) for template in build_record_templates(null_manifest)
    ]
    hf_raw: dict[str, HfDetectionResult] = {}
    lf_raw: dict[tuple[str, str], ContrastiveLfRawObservation] = {}
    committed = units_by_phase()
    resumed_null_failure: StageAExecutionResult | None = None
    for ordinal, entry in enumerate(null_manifest.entries):
        prior = committed.get(("null_fit", ordinal))
        if prior is not None:
            observed_records = tuple(
                _record_from_payload(payload) for payload in prior["records"]
            )
            expected = _cluster_records(null_records, entry.source_cluster_id)
            if tuple(record.template for record in observed_records) != tuple(
                record.template for record in expected
            ):
                raise ContrastiveLfRunnerError("resumed null-fit templates drifted")
            for observed in observed_records:
                null_records[observed.template.slot_ordinal] = observed
            if prior["status"] == "failed":
                failed_record = next(
                    record
                    for record in observed_records
                    if record.execution_status == "failed"
                )
                classification = (
                    "operational_failure"
                    if failed_record.failure_class
                    in {
                        "dependency_failure",
                        "runtime_failure",
                        "codec_failure",
                        "resource_failure",
                    }
                    else "insufficient_evidence"
                )
                resumed_null_failure = StageAExecutionResult(
                    tuple(null_records),
                    None,
                    (),
                    None,
                    classification,
                    failed_record.failure_reason,
                )
                break
            evidence = prior["evidence"]
            hf_raw[entry.source_cluster_id] = _hf_raw_from_payload(evidence["hf_raw"])
            for candidate in CANDIDATE_IDS:
                lf_raw[(entry.source_cluster_id, candidate)] = _lf_raw_from_payload(
                    evidence[candidate]
                )
            continue
        if should_stop():
            return resumable("soft_stop_before_new_unit")
        try:
            completed, failure = _execute_null_cluster(
                null_manifest, operations, null_records, ordinal, hf_raw, lf_raw
            )
        except KeyboardInterrupt:
            return resumable("session_interrupted_during_uncommitted_unit")
        cluster_records = _cluster_records(null_records, entry.source_cluster_id)
        evidence = {}
        if completed:
            evidence = {
                "hf_raw": asdict(hf_raw[entry.source_cluster_id]),
                MULTISCALE_CANDIDATE_ID: asdict(
                    lf_raw[(entry.source_cluster_id, MULTISCALE_CANDIDATE_ID)]
                ),
                SINGLE_SCALE_CANDIDATE_ID: asdict(
                    lf_raw[(entry.source_cluster_id, SINGLE_SCALE_CANDIDATE_ID)]
                ),
            }
        store.commit_unit(
            phase="null_fit",
            cluster_ordinal=ordinal,
            source_cluster_id=entry.source_cluster_id,
            producer_revision=operations.implementation_revision,
            session_id=session_id,
            committed_at_utc=utc_now(),
            records=[record.canonical_payload() for record in cluster_records],
            evidence=evidence,
            status="completed" if completed else "failed",
            cache_diagnostics=operations.cache_diagnostics(),
            package_sha256=package_sha256,
        )
        progress_event("null_fit", entry.source_cluster_id)
        if monotonic() - last_snapshot >= STAGE_A_SNAPSHOT_INTERVAL_SECONDS:
            safe_snapshot("periodic_safe_point")
        if not completed:
            failed_record = next(
                record
                for record in cluster_records
                if record.execution_status == "failed"
            )
            classification = (
                "operational_failure"
                if failed_record.failure_class
                in {
                    "dependency_failure",
                    "runtime_failure",
                    "codec_failure",
                    "resource_failure",
                }
                else "insufficient_evidence"
            )
            result = StageAExecutionResult(
                tuple(null_records), None, (), None, classification, failure
            )
            break
    else:
        result = None
    if resumed_null_failure is not None:
        result = resumed_null_failure
    if result is not None:
        final_result = result
    else:
        artifact = _fit_null_artifact(
            null_manifest, operations, null_records, hf_raw, lf_raw
        )
        authority_payload = {
            "behavior_identity_digest": store.behavior_identity_digest,
            "hf_null_asset": asdict(artifact.hf_null_asset),
            "multiscale_null_asset": asdict(artifact.multiscale_null_asset),
            "null_manifest_digest": artifact.null_manifest_digest,
            "single_scale_null_asset": asdict(artifact.single_scale_null_asset),
        }
        authority_path = store.run_root / "null_fit_asset_authority.json"
        authority_blob = canonical_json_bytes(authority_payload)
        if authority_path.exists():
            if authority_path.is_symlink() or authority_path.read_bytes() != authority_blob:
                raise ContrastiveLfRunnerError("persisted null asset authority drifted")
        else:
            with authority_path.open("xb") as handle:
                handle.write(authority_blob)

        selection_records = [
            _empty_record(template, operations)
            for template in build_record_templates(selection_manifest)
        ]
        assets = {
            HF_CANDIDATE_ID: artifact.hf_null_asset,
            MULTISCALE_CANDIDATE_ID: artifact.multiscale_null_asset,
            SINGLE_SCALE_CANDIDATE_ID: artifact.single_scale_null_asset,
        }
        committed = units_by_phase()
        selection_failed = False
        for ordinal, entry in enumerate(selection_manifest.entries):
            prior = committed.get(("candidate_selection", ordinal))
            if prior is not None:
                observed_records = tuple(
                    _record_from_payload(payload) for payload in prior["records"]
                )
                expected = _cluster_records(selection_records, entry.source_cluster_id)
                if tuple(record.template for record in observed_records) != tuple(
                    record.template for record in expected
                ):
                    raise ContrastiveLfRunnerError("resumed selection templates drifted")
                for observed in observed_records:
                    selection_records[observed.template.slot_ordinal] = observed
                if prior["status"] == "failed":
                    selection_failed = True
                    break
                continue
            if should_stop():
                return resumable("soft_stop_before_new_unit")
            detections: dict[
                tuple[str, str, str, str, int | None], StageADetection
            ] = {}
            try:
                completed, failure = _execute_selection_cluster(
                    selection_manifest,
                    operations,
                    selection_records,
                    ordinal,
                    assets,
                    detections,
                )
            except KeyboardInterrupt:
                return resumable("session_interrupted_during_uncommitted_unit")
            cluster_records = _cluster_records(
                selection_records, entry.source_cluster_id
            )
            store.commit_unit(
                phase="candidate_selection",
                cluster_ordinal=ordinal,
                source_cluster_id=entry.source_cluster_id,
                producer_revision=operations.implementation_revision,
                session_id=session_id,
                committed_at_utc=utc_now(),
                records=[record.canonical_payload() for record in cluster_records],
                evidence={},
                status="completed" if completed else "failed",
                cache_diagnostics=operations.cache_diagnostics(),
                package_sha256=package_sha256,
            )
            progress_event("candidate_selection", entry.source_cluster_id)
            if monotonic() - last_snapshot >= STAGE_A_SNAPSHOT_INTERVAL_SECONDS:
                safe_snapshot("periodic_safe_point")
            if not completed:
                selection_failed = True
                break
        selection_result = (
            _selection_result(
                selection_records, candidate_results=None, hf_anchor=None
            )
            if selection_failed
            else _evaluate_completed_selection(selection_records, assets)
        )
        final_result = StageAExecutionResult(
            null_fit_records=tuple(null_records),
            null_fit_artifact=artifact,
            selection_records=tuple(selection_records),
            selection_result=selection_result,
            result_classification=selection_result.result_classification,
            failure_reason=(
                selection_result.result_classification
                if selection_result.result_classification
                in {"operational_failure", "insufficient_evidence"}
                else None
            ),
        )

    safe_snapshot("final_safe_point")
    units = store.committed_units()
    revisions = tuple(sorted({unit["producer_revision"] for unit in units}))
    session_payload = {
        "behavior_identity_digest": store.behavior_identity_digest,
        "cache_diagnostics": operations.cache_diagnostics(),
        "committed_unit_count": len(units),
        "ended_at_utc": utc_now(),
        "heterogeneous_revisions": len(revisions) > 1,
        "most_recent_snapshot_path": latest_snapshot,
        "producer_revision": operations.implementation_revision,
        "producer_revisions": list(revisions),
        "result_classification": final_result.result_classification,
        "run_id": store.run_root.name,
        "session_id": session_id,
        "session_status": "completed",
    }
    store.write_session_receipt(session_id, session_payload)
    phase_units = units_by_phase()
    return StageAResumableOutcome(
        run_id=store.run_root.name,
        run_root=str(store.run_root),
        session_id=session_id,
        session_status="completed",
        execution_result=final_result,
        completed_null_fit_units=sum(key[0] == "null_fit" for key in phase_units),
        completed_selection_units=sum(
            key[0] == "candidate_selection" for key in phase_units
        ),
        producer_revisions=revisions,
        cache_diagnostics=operations.cache_diagnostics(),
        most_recent_snapshot_path=latest_snapshot,
    )


class AdapterBackedStageAOperations:
    def __init__(self, *, backend: object, runtime_adapter: object, session: object, adapter: object, root_key: str, implementation_revision: str) -> None:
        from main import identify_root_key
        self._backend, self._runtime, self._session, self._adapter = backend, runtime_adapter, session, adapter
        self.root_key = root_key
        self._root_digest = identify_root_key(root_key).root_key_public_digest
        self.implementation_revision = implementation_revision
        self.method_config_digest = CONFIG_DIGEST
        self.model_identity = canonical_digest({"model_id": session.model_id, "model_revision": session.model_revision})
        self.runtime_identity = canonical_digest({"runtime_config_digest": session.runtime_config_digest, "backend": session.runtime_backend_name, "device": session.selected_device})
        self.codec_identity = "pillow_rgb8_jpeg_exact_capability"
        self._public_observation_cache: dict[str, object] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._vae_encode_count = 0
        self._observation_behavior_identity = canonical_digest(
            {
                "runtime_config_digest": session.runtime_config_digest,
                "vae_encode_protocol": session.vae_encode_protocol,
                "public_preprocess": "ordinary_contiguous_rgb8_nchw_to_posterior_mode_binary32",
            }
        )

    @property
    def root_key_public_digest(self) -> str:
        return self._root_digest

    @property
    def observation_behavior_identity(self) -> str:
        return self._observation_behavior_identity

    def _latent(self, entry: object) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(entry.generation_seed)
        return torch.randn((1, 16, self._session.image_height // 8, self._session.image_width // 8), dtype=torch.float32, generator=generator).to(device=self._session.selected_device, dtype=torch.float16)

    def clean(self, entry: object) -> StageAGeneration:
        from runtime import materialize_ordinary_rgb8_snapshot
        self._backend.set_development_generation_prompts(entry.prompt_text, "")
        result = self._runtime.execute_clean_image_and_vae_observation(self._latent(entry))
        image = materialize_ordinary_rgb8_snapshot(result.clean_image)
        return StageAGeneration("clean_unwatermarked", image, image.clone(), None, None, None, 0.0)

    def write(self, entry: object, arm_id: str) -> StageAGeneration:
        from main import ContentMaterializationResult
        from runtime import materialize_ordinary_rgb8_snapshot
        self._backend.set_development_generation_prompts(entry.prompt_text, "")
        observed = self._adapter.execute_contrastive_lf_content_arm_write_and_vae(self._latent(entry), self.root_key, arm_id=arm_id)
        result = observed.result.content_write_result
        authority = result.content_materialization_result
        if type(authority) is not ContentMaterializationResult or authority.budget_status != "accepted" or authority.integrity_status != "passed":
            raise ContrastiveLfRunnerError("content budget authority is invalid")
        clean = materialize_ordinary_rgb8_snapshot(result.clean_image)
        written = materialize_ordinary_rgb8_snapshot(result.watermarked_image)
        mse = float(torch.mean(((written.to(torch.float32) - clean.to(torch.float32)) / 255.0) ** 2).item())
        replay = authority.observation.materialization_replay_identity
        return StageAGeneration(arm_id, written, clean, replay, canonical_digest({"materialization_replay_identity": replay, "arm_id": arm_id}), authority.budget_status, mse)

    def attack(self, entry: object, generation: StageAGeneration, attack_id: str) -> torch.Tensor:
        return apply_contrastive_lf_attack(generation.image_rgb8, attack_id, source_cluster_id=entry.source_cluster_id, generation_seed=entry.generation_seed).image_rgb8

    def prepare_public_observation(self, image_rgb8: torch.Tensor) -> object:
        from experiments.methods import ContrastiveLfPublicImageVaeObservation
        from runtime import materialize_ordinary_rgb8_snapshot

        image = materialize_ordinary_rgb8_snapshot(image_rgb8)
        image_digest = sha256(
            str(image.dtype).encode("ascii")
            + repr(tuple(int(size) for size in image.shape)).encode("ascii")
            + image.cpu().numpy().tobytes(order="C")
        ).hexdigest()
        cache_key = canonical_digest(
            {
                "image_rgb8_digest": image_digest,
                "shape": tuple(int(size) for size in image.shape),
                "dtype": str(image.dtype),
                "observation_behavior_identity": self._observation_behavior_identity,
            }
        )
        cached = self._public_observation_cache.get(cache_key)
        if cached is not None:
            if (
                type(cached) is not ContrastiveLfPublicImageVaeObservation
                or cached.input_rgb8_digest != image_digest
            ):
                raise ContrastiveLfRunnerError("public observation cache identity drifted")
            self._cache_hits += 1
            return cached
        observed = self._adapter.prepare_stage_a_public_rgb8_observation(image)
        if (
            type(observed) is not ContrastiveLfPublicImageVaeObservation
            or observed.input_rgb8_digest != image_digest
        ):
            raise ContrastiveLfRunnerError("public observation input identity drifted")
        self._public_observation_cache[cache_key] = observed
        self._cache_misses += 1
        self._vae_encode_count += 1
        return observed

    def observe_hf_raw(self, prepared: object, key: str | DerivedWrongKeyMaterial) -> HfDetectionResult:
        from main import HfDetectionResult
        result = self._adapter.score_stage_a_hf_prepared_observation(prepared, key)
        if type(result) is not HfDetectionResult:
            raise ContrastiveLfRunnerError("prepared HF raw result drifted")
        return result

    def observe_lf_raw(self, prepared: object, key: str | DerivedWrongKeyMaterial, candidate_id: str) -> ContrastiveLfRawObservation:
        result = self._adapter.score_contrastive_lf_prepared_observation(
            prepared, key, candidate_id=candidate_id
        )
        if type(result) is not ContrastiveLfRawObservation:
            raise ContrastiveLfRunnerError("prepared LF raw result drifted")
        return result

    def score_lf_raw(self, raw: ContrastiveLfRawObservation, asset: ContrastiveLfNullAsset) -> ContrastiveLfDetectionResult:
        from main import contrastive_lf_detector
        return contrastive_lf_detector(raw, asset)

    def observe_hf(self, prepared: object, key: str | DerivedWrongKeyMaterial, asset: HfPopulationNullAsset) -> StageADetection:
        from main import HfPopulationStandardizedResult
        result = self._adapter.score_stage_a_hf_prepared_observation(
            prepared, key, null_asset=asset
        )
        if type(result) is not HfPopulationStandardizedResult:
            raise ContrastiveLfRunnerError("prepared HF result drifted")
        return StageADetection(result.raw_score, result.standardized_score, (), asset.asset_digest, _threshold_digest(HF_CANDIDATE_ID, asset.asset_digest, asset.provisional_tau), result.detector_identity)

    def observe_lf(self, prepared: object, key: str | DerivedWrongKeyMaterial, asset: ContrastiveLfNullAsset) -> StageADetection:
        result = self._adapter.score_contrastive_lf_prepared_observation(
            prepared,
            key,
            candidate_id=asset.candidate_id,
            null_asset=asset,
        )
        if type(result) is not ContrastiveLfDetectionResult:
            raise ContrastiveLfRunnerError("prepared LF result drifted")
        return StageADetection(result.contrastive_score, result.standardized_score, result.internal_decoy_scores, asset.asset_digest, _threshold_digest(asset.candidate_id, asset.asset_digest, asset.provisional_tau), result.detector_identity)

    def cache_diagnostics(self) -> dict[str, int]:
        return {
            "cache_entry_count": len(self._public_observation_cache),
            "cache_hit_count": self._cache_hits,
            "cache_miss_count": self._cache_misses,
            "vae_encode_count": self._vae_encode_count,
        }

    def wrong_key(self, index: int) -> DerivedWrongKeyMaterial:
        from main import derive_wrong_key_material
        return derive_wrong_key_material(self._root_digest, index)

    def close(self) -> None:
        self._runtime.close()


def create_adapter_backed_stage_a_operations(*, implementation_revision: str) -> AdapterBackedStageAOperations:
    from experiments.methods import CegWmExperimentAdapter, load_ceg_wm_experiment_adapter_configuration
    from runtime import Sd35PipelineBackend, create_runtime_adapter
    hf_token, root_key = os.environ.get("HF_TOKEN"), os.environ.get("CEG_WM_ROOT_KEY")
    cache_root, persistent_root = os.environ.get("CEG_WM_CACHE_ROOT"), os.environ.get("CEG_WM_PERSISTENT_ROOT")
    if not all((hf_token, root_key, cache_root, persistent_root)):
        raise ContrastiveLfRunnerError("required production environment is incomplete")
    validate_jpeg_capability()
    backend = Sd35PipelineBackend(cache_root=Path(cache_root), persistent_root=Path(persistent_root), hf_token=hf_token, prompt="contrastive_lf_branch_attribution", negative_prompt="")
    runtime_adapter = create_runtime_adapter(backend, PACKAGE_ROOT / "configs/runtime/runtime_sd35_flowmatch.json")
    session = runtime_adapter.initialize("cuda")
    adapter = CegWmExperimentAdapter(load_ceg_wm_experiment_adapter_configuration(PACKAGE_ROOT / "configs/experiments/internal_execution_components.json"), runtime_adapter)
    return AdapterBackedStageAOperations(backend=backend, runtime_adapter=runtime_adapter, session=session, adapter=adapter, root_key=root_key, implementation_revision=implementation_revision)


__all__ = [
    "AdapterBackedStageAOperations",
    "StageADetection",
    "StageAExecutionResult",
    "StageAGeneration",
    "StageANullFitArtifact",
    "StageAOperations",
    "StageAResumableOutcome",
    "create_adapter_backed_stage_a_operations",
    "execute_null_fit",
    "execute_selection",
    "execute_stage_a_resumable",
    "execute_stage_a_null_fit_and_selection",
]
