"""Fixed-denominator Stage-A null-fit and candidate-selection runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import isfinite
import os
from pathlib import Path
from typing import Protocol, Sequence

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
                (self.implementation_revision, operations.implementation_revision),
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
    def observe_hf_raw(self, image_rgb8: torch.Tensor, key: str | DerivedWrongKeyMaterial) -> HfDetectionResult: ...
    def observe_lf_raw(self, image_rgb8: torch.Tensor, key: str | DerivedWrongKeyMaterial, candidate_id: str) -> ContrastiveLfRawObservation: ...
    def observe_hf(self, image_rgb8: torch.Tensor, key: str | DerivedWrongKeyMaterial, asset: HfPopulationNullAsset) -> StageADetection: ...
    def observe_lf(self, image_rgb8: torch.Tensor, key: str | DerivedWrongKeyMaterial, asset: ContrastiveLfNullAsset) -> StageADetection: ...
    def score_lf_raw(self, raw: ContrastiveLfRawObservation, asset: ContrastiveLfNullAsset) -> ContrastiveLfDetectionResult: ...
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


def execute_null_fit(
    manifest: ContrastiveLfManifest, operations: StageAOperations
) -> tuple[tuple[ContrastiveLfRecord, ...], StageANullFitArtifact | None, str | None]:
    manifest.validate()
    if manifest.role_id != NULL_FIT_ROLE:
        raise ContrastiveLfRunnerError("null-fit manifest role mismatch")
    templates = build_record_templates(manifest)
    records = [_empty_record(template, operations) for template in templates]
    images: dict[str, torch.Tensor] = {}
    hf_raw: dict[str, HfDetectionResult] = {}
    lf_raw: dict[tuple[str, str], ContrastiveLfRawObservation] = {}
    for index, record in enumerate(records):
        template = record.template
        entry = next(item for item in manifest.entries if item.source_cluster_id == template.source_cluster_id)
        try:
            if template.record_kind == "clean_base_observation":
                generation = operations.clean(entry)
                images[template.source_cluster_id] = generation.image_rgb8
                records[index] = _complete(record)
            elif template.candidate_id == HF_CANDIDATE_ID:
                value = operations.observe_hf_raw(images[template.source_cluster_id], operations.root_key)
                hf_raw[template.source_cluster_id] = value
                records[index] = _complete(record, raw_score=float(value.hf_score))
            else:
                value = operations.observe_lf_raw(
                    images[template.source_cluster_id], operations.root_key, template.candidate_id
                )
                lf_raw[(template.source_cluster_id, template.candidate_id)] = value
                records[index] = _complete(record, raw_score=float(value.raw_feature[0]), internal_decoy_scores=tuple(feature[0] for feature in value.internal_decoy_features))
        except Exception as exc:
            records[index] = _fail(record, exc)
            return tuple(records), None, type(exc).__name__[:120]
    try:
        ordered_clusters = tuple(entry.source_cluster_id for entry in manifest.entries)
        hf_asset = fit_hf_population_null_asset(
            tuple(float(hf_raw[cluster].hf_score) for cluster in ordered_clusters),
            null_manifest_digest=manifest.manifest_digest,
        )
        multiscale = fit_contrastive_lf_null_asset(
            tuple(lf_raw[(cluster, MULTISCALE_CANDIDATE_ID)] for cluster in ordered_clusters),
            candidate_id=MULTISCALE_CANDIDATE_ID,
            null_manifest_digest=manifest.manifest_digest,
        )
        single = fit_contrastive_lf_null_asset(
            tuple(lf_raw[(cluster, SINGLE_SCALE_CANDIDATE_ID)] for cluster in ordered_clusters),
            candidate_id=SINGLE_SCALE_CANDIDATE_ID,
            null_manifest_digest=manifest.manifest_digest,
        )
        assets = {HF_CANDIDATE_ID: hf_asset, MULTISCALE_CANDIDATE_ID: multiscale, SINGLE_SCALE_CANDIDATE_ID: single}
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
                    provisional_threshold_digest=_threshold_digest(candidate, hf_asset.asset_digest, hf_asset.provisional_tau),
                    z_score=z,
                )
            else:
                raw_observation = lf_raw[(record.template.source_cluster_id, candidate)]
                detected = operations.score_lf_raw(raw_observation, asset)
                records[index] = replace(
                    record,
                    raw_score=detected.contrastive_score,
                    internal_decoy_scores=detected.internal_decoy_scores,
                    population_mean=asset.contrastive_population_mean,
                    population_variance=asset.contrastive_population_variance,
                    population_sigma=asset.contrastive_population_sigma,
                    null_asset_digest=asset.asset_digest,
                    provisional_threshold_digest=_threshold_digest(candidate, asset.asset_digest, asset.provisional_tau),
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
    generations: dict[tuple[str, str], StageAGeneration] = {}
    attacked: dict[tuple[str, str, str], torch.Tensor] = {}
    detections: dict[tuple[str, str, str, str, int | None], StageADetection] = {}
    assets = {
        HF_CANDIDATE_ID: null_artifact.hf_null_asset,
        MULTISCALE_CANDIDATE_ID: null_artifact.multiscale_null_asset,
        SINGLE_SCALE_CANDIDATE_ID: null_artifact.single_scale_null_asset,
    }
    for index, record in enumerate(records):
        template = record.template
        entry = next(item for item in manifest.entries if item.source_cluster_id == template.source_cluster_id)
        try:
            if template.record_kind == "base_generation":
                generation = operations.clean(entry) if template.arm_id == "clean_unwatermarked" else operations.write(entry, template.arm_id)
                if generation.arm_id != template.arm_id or not isfinite(generation.paired_rgb8_mse):
                    raise ContrastiveLfRunnerError("generation identity drifted")
                generations[(template.source_cluster_id, template.arm_id)] = generation
                records[index] = _complete(record)
            elif template.record_kind == "attacked_observation":
                generation = generations[(template.source_cluster_id, template.arm_id)]
                image = operations.attack(entry, generation, template.attack_id)
                attacked[(template.source_cluster_id, template.arm_id, template.attack_id)] = image
                records[index] = _complete(record)
            elif template.record_kind == "detector":
                image = attacked[(template.source_cluster_id, template.arm_id, template.attack_id)]
                key = operations.root_key if template.key_role == "registered" else operations.wrong_key(template.wrong_key_index)
                asset = assets[template.candidate_id]
                detection = operations.observe_hf(image, key, asset) if template.candidate_id == HF_CANDIDATE_ID else operations.observe_lf(image, key, asset)
                detections[(template.source_cluster_id, template.candidate_id, template.attack_id, template.control_identity, template.wrong_key_index)] = detection
                bound = {}
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
                generation = generations[(template.source_cluster_id, template.arm_id)]
                records[index] = _complete(
                    record,
                    budget_status=generation.budget_status,
                    materialization_replay_identity=generation.materialization_replay_identity,
                    replay_digest=generation.replay_digest,
                )
            else:
                candidate_image = attacked[(template.source_cluster_id, template.arm_id, template.attack_id)]
                clean_image = attacked[(template.source_cluster_id, "clean_unwatermarked", template.attack_id)]
                mse = float(torch.mean(((candidate_image.to(torch.float32) - clean_image.to(torch.float32)) / 255.0) ** 2).item())
                if not isfinite(mse):
                    raise ContrastiveLfRunnerError("paired RGB8 MSE is non-finite")
                records[index] = _complete(record, paired_rgb8_mse=mse)
        except Exception as exc:
            records[index] = _fail(record, exc)
            return tuple(records), _selection_result(records, candidate_results=None, hf_anchor=None)
    branch_by_candidate: dict[str, list[StageABranchCase]] = {candidate: [] for candidate in CANDIDATE_IDS}
    quality_by_candidate: dict[str, list[StageAQualityCase]] = {candidate: [] for candidate in CANDIDATE_IDS}
    arm_by_candidate = {MULTISCALE_CANDIDATE_ID: "multiscale_low_frequency_only", SINGLE_SCALE_CANDIDATE_ID: "single_scale_low_frequency_only"}
    for entry in manifest.entries:
        cluster = entry.source_cluster_id
        for attack_id in ATTACKS:
            hf_registered = detections[(cluster, HF_CANDIDATE_ID, attack_id, "registered_attribution", None)]
            hf_null = detections[(cluster, HF_CANDIDATE_ID, attack_id, "paired_primary_null", None)]
            hf_wrong = tuple(detections[(cluster, HF_CANDIDATE_ID, attack_id, "external_wrong_key", index)].standardized_score for index in range(8))
            for candidate in CANDIDATE_IDS:
                registered = detections[(cluster, candidate, attack_id, "registered_attribution", None)]
                primary = detections[(cluster, candidate, attack_id, "paired_primary_null", None)]
                wrong = tuple(detections[(cluster, candidate, attack_id, "external_wrong_key", index)].standardized_score for index in range(8))
                branch_by_candidate[candidate].append(StageABranchCase(cluster, attack_id, registered.standardized_score, primary.standardized_score, wrong, hf_registered.standardized_score, hf_null.standardized_score, hf_wrong))
                candidate_quality = next(record.paired_rgb8_mse for record in records if record.template.record_kind == "quality" and record.template.source_cluster_id == cluster and record.template.candidate_id == candidate and record.template.attack_id == attack_id)
                hf_quality = next(record.paired_rgb8_mse for record in records if record.template.record_kind == "quality" and record.template.source_cluster_id == cluster and record.template.candidate_id == HF_CANDIDATE_ID and record.template.attack_id == attack_id)
                quality_by_candidate[candidate].append(StageAQualityCase(cluster, attack_id, candidate_quality, hf_quality))
    candidate_results = {
        candidate: evaluate_stage_a_candidate_gates(
            candidate,
            branch_by_candidate[candidate],
            quality_by_candidate[candidate],
            candidate_tau=assets[candidate].provisional_tau,
            hf_tau=null_artifact.hf_null_asset.provisional_tau,
        )
        for candidate in CANDIDATE_IDS
    }
    hf_anchor = evaluate_stage_a_hf_anchor(
        branch_by_candidate[MULTISCALE_CANDIDATE_ID],
        hf_tau=null_artifact.hf_null_asset.provisional_tau,
    )
    return tuple(records), _selection_result(records, candidate_results=candidate_results, hf_anchor=hf_anchor)


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

    def observe_hf_raw(self, image_rgb8: torch.Tensor, key: str | DerivedWrongKeyMaterial) -> HfDetectionResult:
        return self._adapter.observe_stage_a_hf_raw(image_rgb8, key)

    def observe_lf_raw(self, image_rgb8: torch.Tensor, key: str | DerivedWrongKeyMaterial, candidate_id: str) -> ContrastiveLfRawObservation:
        return self._adapter.observe_contrastive_lf_raw(image_rgb8, key, candidate_id=candidate_id)

    def score_lf_raw(self, raw: ContrastiveLfRawObservation, asset: ContrastiveLfNullAsset) -> ContrastiveLfDetectionResult:
        from main import contrastive_lf_detector
        return contrastive_lf_detector(raw, asset)

    def observe_hf(self, image_rgb8: torch.Tensor, key: str | DerivedWrongKeyMaterial, asset: HfPopulationNullAsset) -> StageADetection:
        result = self._adapter.observe_stage_a_hf(image_rgb8, key, asset).result
        return StageADetection(result.raw_score, result.standardized_score, (), asset.asset_digest, _threshold_digest(HF_CANDIDATE_ID, asset.asset_digest, asset.provisional_tau), result.detector_identity)

    def observe_lf(self, image_rgb8: torch.Tensor, key: str | DerivedWrongKeyMaterial, asset: ContrastiveLfNullAsset) -> StageADetection:
        result = self._adapter.observe_contrastive_lf_candidate(image_rgb8, key, asset).result
        return StageADetection(result.contrastive_score, result.standardized_score, result.internal_decoy_scores, asset.asset_digest, _threshold_digest(asset.candidate_id, asset.asset_digest, asset.provisional_tau), result.detector_identity)

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
    "create_adapter_backed_stage_a_operations",
    "execute_null_fit",
    "execute_selection",
    "execute_stage_a_null_fit_and_selection",
]
