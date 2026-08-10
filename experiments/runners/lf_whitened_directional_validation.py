"""Real public-blind runner for LF whitened directional validation."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import isfinite
from struct import pack
from time import monotonic
from typing import Sequence

import torch

from experiments.methods import CegWmExperimentAdapter
from experiments.metrics.lf_whitened_directional_validation import (
    LfWhitenedDirectionalAggregate,
    LfWhitenedDirectionalObservation,
    aggregate_lf_whitened_direction,
    create_lf_whitened_directional_observation,
)
from experiments.protocol.development_records import (
    DEVELOPMENT_CLAIM_BOUNDARY,
    DEVELOPMENT_RECORD_COLLECTION_ROLE,
    METRIC_SCHEMA_VERSION,
    OPERATIONAL_RECORD_COLLECTION_ROLE,
    OPERATIONAL_RECORD_KIND,
    OPERATIONAL_RECORD_SCHEMA,
    RECORD_SCHEMA_VERSION,
    DevelopmentRecordError,
    DevelopmentOperationalRecord,
    DevelopmentScientificRecord,
    canonical_development_value_digest,
)
from experiments.protocol.internal_splits import AnalysisUnitIdentity, derive_source_cluster_id
from experiments.protocol.lf_whitened_directional_validation import (
    OPERATIONAL_UNIT_COUNT,
    SCIENTIFIC_CLUSTER_COUNT,
    LfWhitenedDirectionalProtocol,
    canonical_digest,
    derive_lf_whitened_directional_analysis_identity,
)
from experiments.protocol.lf_whitened_score_screening import LfWhiteningManifest
from experiments.runners.development_persistence import (
    CommittedUnit,
    FrozenDevelopmentUnitBinding,
    create_frozen_development_unit_binding,
)
from main import (
    LfDetectionObservation,
    LfNullWhiteningAsset,
    PreparedLfWhitenedTemplate,
    derive_wrong_key_material,
    prepare_lf_null_whitened_observation,
    prepare_lf_null_whitened_template,
)
from runtime import ContentWriteVaeResult, Sd35RuntimeAdapter


class LfWhitenedDirectionalRunnerError(RuntimeError):
    """The LF directional execution violated its frozen contract."""


class LfWhitenedDirectionalEvidenceViolation(LfWhitenedDirectionalRunnerError):
    """A completed runtime operation violated a scientific evidence boundary."""

    def __init__(self, category: str, diagnostics: dict[str, object]) -> None:
        if category not in {
            "identity_violation",
            "budget_violation",
            "integrity_violation",
            "nonfinite_violation",
        }:
            raise ValueError("directional violation category is invalid")
        self.category = category
        self.diagnostics = diagnostics
        super().__init__(category)


def _observation(tensor: torch.Tensor) -> LfDetectionObservation:
    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != (1, 16, 64, 64)
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise LfWhitenedDirectionalRunnerError("public LF observation tensor is invalid")
    return LfDetectionObservation.from_public_image_encoding(
        tuple(float(item) for item in tensor.detach().cpu().float().reshape(-1)),
        tuple(int(size) for size in tensor.shape),
    )


class LfWhitenedDirectionalValidationRunner:
    """Runs paired LF writes and registered/null/four-wrong public detections."""

    def __init__(
        self,
        *,
        protocol: LfWhitenedDirectionalProtocol,
        manifest: LfWhiteningManifest,
        adapter: CegWmExperimentAdapter,
        runtime_adapter: Sd35RuntimeAdapter,
        whitening_asset: LfNullWhiteningAsset,
        method_code_revision: str,
        run_id: str,
        registered_root_key: str,
        root_key_public_digest: str,
        protocol_digest: str,
        execution_intent_authority_digest: str,
        candidate_config_digest: str,
    ) -> None:
        protocol.validate()
        manifest.validate(expected_role="lf_whitened_directional_validation", count=32)
        if type(adapter) is not CegWmExperimentAdapter or type(runtime_adapter) is not Sd35RuntimeAdapter:
            raise LfWhitenedDirectionalRunnerError("exact method and runtime adapters are required")
        if type(whitening_asset) is not LfNullWhiteningAsset:
            raise LfWhitenedDirectionalRunnerError("frozen whitening asset exact type is required")
        if type(method_code_revision) is not str or len(method_code_revision) != 40:
            raise LfWhitenedDirectionalRunnerError("method revision is invalid")
        if protocol_digest != protocol.digest() or run_id != protocol.run_id:
            raise LfWhitenedDirectionalRunnerError("protocol or run identity drifted")
        self.protocol = protocol
        self.manifest = manifest
        self.adapter = adapter
        self.runtime = runtime_adapter
        self.whitening_asset = whitening_asset
        self.method_code_revision = method_code_revision
        self.run_id = run_id
        self.registered_root_key = registered_root_key
        self.root_key_public_digest = root_key_public_digest
        self.protocol_digest = protocol_digest
        self.execution_intent_authority_digest = execution_intent_authority_digest
        self.candidate_config_digest = candidate_config_digest
        self._registered_prepared_template = prepare_lf_null_whitened_template(
            self.registered_root_key,
            self.whitening_asset,
        )
        self._wrong_prepared_templates: tuple[
            PreparedLfWhitenedTemplate, ...
        ] = tuple(
            prepare_lf_null_whitened_template(
                derive_wrong_key_material(self.root_key_public_digest, index),
                self.whitening_asset,
            )
            for index in range(self.protocol.wrong_key_roster_size)
        )

    def _key_family_digest(self) -> str:
        return canonical_digest({
            "root_key_public_digest": self.root_key_public_digest,
            "seed_namespace": self.manifest.seed_namespace,
            "role": "registered_lf_whitened_directional_key_family",
        })

    def _analysis_identity(self, unit_index: int) -> AnalysisUnitIdentity:
        if unit_index == 0:
            return AnalysisUnitIdentity(
                unit_id="lf_whitened_detector_public_runtime_preflight",
                case_id="paired_clean_lf_public_detector_smoke",
                source_cluster_id=derive_source_cluster_id(
                    prompt_digest=self.protocol.operational_smoke_prompt_digest,
                    generation_seed=self.protocol.operational_smoke_generation_seed,
                    image_lineage_digest=self.protocol.operational_smoke_image_lineage_digest,
                    registered_key_family_digest=self._key_family_digest(),
                ),
                prompt_digest=self.protocol.operational_smoke_prompt_digest,
                generation_seed=self.protocol.operational_smoke_generation_seed,
                image_lineage_digest=self.protocol.operational_smoke_image_lineage_digest,
                registered_key_family_digest=self._key_family_digest(),
            )
        return derive_lf_whitened_directional_analysis_identity(
            self.manifest.entries[unit_index - OPERATIONAL_UNIT_COUNT],
            self.manifest,
            key_family_digest=self._key_family_digest(),
        )

    def create_persistence_unit_bindings(self) -> tuple[FrozenDevelopmentUnitBinding, ...]:
        bindings = []
        for unit in self.protocol.unit_roster:
            operational = unit.unit_index == 0
            bindings.append(create_frozen_development_unit_binding(
                unit,
                analysis_unit_identity=self._analysis_identity(unit.unit_index),
                scientific_question_id=(
                    "lf_whitened_public_detector_execution_smoke"
                    if operational
                    else "lf_whitened_registered_score_direction_against_paired_controls"
                ),
                development_case_id=(
                    "paired_clean_lf_public_detector_smoke"
                    if operational
                    else "paired_clean_lf_whitened_blind_directional_detection"
                ),
                candidate_identity=self.protocol.candidate_identity,
                candidate_config_digest=self.candidate_config_digest,
            ))
        return tuple(bindings)

    def _execute_paired_runtime(self, base_latent: torch.Tensor) -> ContentWriteVaeResult:
        carrier = self.adapter.build_lf_carrier(
            self.registered_root_key, tuple(int(size) for size in base_latent.shape)
        ).result

        def embed(values: tuple[float, ...]):
            return self.adapter.embed_content(
                values,
                None,
                lf_carrier_result=carrier,
            ).result

        result = self.runtime.execute_content_write_and_vae(base_latent, embed)
        if type(result) is not ContentWriteVaeResult:
            raise LfWhitenedDirectionalRunnerError("paired runtime result exact type is required")
        return result

    def _detect_public_pair(
        self, result: ContentWriteVaeResult, *, cluster_ordinal: int
    ) -> tuple[LfWhitenedDirectionalObservation, dict[str, object]]:
        candidate = _observation(result.watermarked_detection_latent)
        clean = _observation(result.clean_detection_latent)
        prepared_candidate = prepare_lf_null_whitened_observation(
            candidate,
            self.whitening_asset,
        )
        prepared_clean = prepare_lf_null_whitened_observation(
            clean,
            self.whitening_asset,
        )
        registered = self.adapter.detect_lf_null_whitened(
            candidate,
            self.registered_root_key,
            self.whitening_asset,
            prepared_observation=prepared_candidate,
            prepared_template=self._registered_prepared_template,
        ).result
        primary_null = self.adapter.detect_lf_null_whitened(
            clean,
            self.registered_root_key,
            self.whitening_asset,
            prepared_observation=prepared_clean,
            prepared_template=self._registered_prepared_template,
        ).result
        wrong = tuple(
            self.adapter.detect_lf_null_whitened(
                candidate,
                derive_wrong_key_material(self.root_key_public_digest, index),
                self.whitening_asset,
                prepared_observation=prepared_candidate,
                prepared_template=self._wrong_prepared_templates[index],
            ).result
            for index in range(self.protocol.wrong_key_roster_size)
        )
        detections = (registered, primary_null, *wrong)
        if (
            len({item.detector_config_digest for item in detections}) != 1
            or len({item.detector_identity for item in detections}) != 1
            or len({item.whitening_asset_digest for item in detections}) != 1
            or any(item.whitening_asset_digest != self.whitening_asset.whitening_asset_digest for item in detections)
            or registered.observation_digest != candidate.observation_digest
            or primary_null.observation_digest != clean.observation_digest
            or any(item.observation_digest != candidate.observation_digest for item in wrong)
        ):
            raise LfWhitenedDirectionalEvidenceViolation(
                "identity_violation", {"result_available": False}
            )
        measurement = result.content_materialization
        materialization = result.content_materialization_result
        diagnostics = {
            "realized_relative_l2": measurement.realized_relative_l2,
            "content_relative_l2_limit": materialization.content_relative_l2_limit,
            "budget_status": materialization.budget_status,
            "integrity_status": measurement.integrity_status,
        }
        if (
            pack(">f", materialization.content_relative_l2_nominal) != pack(">f", 3 / 250)
            or pack(">f", materialization.content_relative_l2_limit) != pack(">f", 3 / 250)
            or materialization.budget_status != "accepted"
            or measurement.realized_relative_l2 > materialization.content_relative_l2_limit
        ):
            raise LfWhitenedDirectionalEvidenceViolation("budget_violation", diagnostics)
        if measurement.integrity_status != "passed" or materialization.integrity_status != "passed":
            raise LfWhitenedDirectionalEvidenceViolation("integrity_violation", diagnostics)
        observation = create_lf_whitened_directional_observation(
            cluster_ordinal=cluster_ordinal,
            registered_score=registered.lf_score,
            primary_null_score=primary_null.lf_score,
            wrong_key_scores=tuple(item.lf_score for item in wrong),
            candidate_observation_digest=candidate.observation_digest,
            clean_observation_digest=clean.observation_digest,
            registered_detector_identity=registered.detector_identity,
            primary_null_detector_identity=primary_null.detector_identity,
            wrong_key_detector_identities=tuple(item.detector_identity for item in wrong),
            detector_config_digest=registered.detector_config_digest,
            observation_protocol=candidate.observation_protocol,
            whitening_asset_digest=registered.whitening_asset_digest,
            registered_template_digest=registered.template_digest,
            primary_null_template_digest=primary_null.template_digest,
            wrong_key_template_digests=tuple(item.template_digest for item in wrong),
            registered_root_key_public_digest=registered.root_key_public_digest,
            wrong_key_indexes=tuple(range(self.protocol.wrong_key_roster_size)),
            materialization_integrity_status=measurement.integrity_status,
            materialization_budget_status=materialization.budget_status,
            realized_relative_l2=measurement.realized_relative_l2,
            content_relative_l2_limit=materialization.content_relative_l2_limit,
            actual_runtime_dtype=str(measurement.written_latent_actual.dtype),
        )
        return observation, {
            "registered": asdict(registered),
            "primary_null": asdict(primary_null),
            "wrong_keys": tuple(asdict(item) for item in wrong),
        }

    def execute_operational_smoke(
        self,
        *,
        base_latent: torch.Tensor,
        attempt_index: int,
        retry_parent_intent_digest: str | None,
        maximum_duration_seconds: int,
        started_monotonic: float | None = None,
    ) -> DevelopmentOperationalRecord:
        started = monotonic() if started_monotonic is None else started_monotonic
        result = self._execute_paired_runtime(base_latent)
        observation, _ = self._detect_public_pair(result, cluster_ordinal=0)
        elapsed = float(monotonic() - started)
        payload = {
            "operational_role": "environment_runtime_throughput_preflight",
            "source_cluster_ordinal": 0,
            "case_ids": ["paired_clean_lf_public_detector_smoke"],
            "responsibility_result_digests": [["content_embedder", observation.observation_identity]],
            "elapsed_seconds": elapsed,
            "runtime_config_digest": result.runtime_config_digest,
            "counts_as_scientific_coverage": False,
            "scientific_claims_supported": False,
        }
        record = DevelopmentOperationalRecord(
            schema_version=OPERATIONAL_RECORD_SCHEMA,
            collection_role=OPERATIONAL_RECORD_COLLECTION_ROLE,
            record_kind=OPERATIONAL_RECORD_KIND,
            record_id="0" * 64,
            run_id=self.run_id,
            protocol_digest=self.protocol_digest,
            method_code_revision=self.method_code_revision,
            unit_index=0,
            phase="development_environment_preflight",
            source_cluster_ordinal=0,
            candidate_config_digest=self.candidate_config_digest,
            attempt_index=attempt_index,
            retry_parent_intent_digest=retry_parent_intent_digest,
            actual_elapsed_seconds=elapsed,
            maximum_duration_seconds=maximum_duration_seconds,
            operation_result_payload=payload,
            counts_as_scientific_coverage=False,
            scientific_claims_supported=False,
            scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
        )
        record = replace(record, record_id=canonical_development_value_digest(record.payload_without_record_id()))
        record.validate()
        return record

    def _scientific_record(
        self,
        *,
        cluster_ordinal: int,
        attempt_index: int,
        retry_parent_intent_digest: str | None,
        maximum_duration_seconds: int,
        actual_elapsed_seconds: float,
        observation: LfWhitenedDirectionalObservation | None,
        operation_payload: dict[str, object],
        resource_failure: bool = False,
    ) -> DevelopmentScientificRecord:
        success = observation is not None
        retry = resource_failure and attempt_index + 1 < self.protocol.maximum_attempts_per_unit
        identity = self._analysis_identity(OPERATIONAL_UNIT_COUNT + cluster_ordinal)
        metric_ids = (
            "registered_primary_null_directional_margin",
            "registered_max_four_wrong_directional_margin",
            "content_write_budget",
        )
        metric: dict[str, object] = {}
        if observation is not None:
            metric = {
                "schema_version": METRIC_SCHEMA_VERSION,
                "metric_role": "development_exploratory_cluster_level",
                "responsibility_id": "lf_detector",
                "source_cluster_id": identity.source_cluster_id,
                "registered_metric_ids": metric_ids,
                "candidate_config_digest": self.candidate_config_digest,
                "paired_ablation_identity": "paired_clean_lf_same_prompt_seed_rendered_rgb8",
                "content_branch_id": "lf_only",
                "geometry_case_id": "not_applicable",
                "sufficient_statistics": (
                    ("registered_minus_primary_null", observation.registered_minus_primary_null),
                    ("registered_minus_max_wrong", observation.registered_minus_max_wrong),
                    ("realized_relative_l2", observation.realized_relative_l2),
                ),
                "result_identity_digests": (observation.observation_identity,),
                "threshold_role": "not_fitted_directional_validation",
                "threshold_identity": None,
                "threshold_fit_source_cluster_digest": None,
            }
            metric["observation_digest"] = canonical_development_value_digest(metric)
        entry = self.manifest.entries[cluster_ordinal]
        payload = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "collection_role": DEVELOPMENT_RECORD_COLLECTION_ROLE,
            "record_id": "0" * 64,
            "run_id": self.run_id,
            "protocol_id": self.protocol.protocol_id,
            "protocol_version": self.protocol.protocol_version,
            "protocol_digest": self.protocol_digest,
            "execution_intent_authority_digest": self.execution_intent_authority_digest,
            "method_code_revision": self.method_code_revision,
            "unit_index": OPERATIONAL_UNIT_COUNT + cluster_ordinal,
            "phase": "development_scientific_breadth",
            "analysis_unit_identity": asdict(identity),
            "responsibility_id": "lf_detector",
            "scientific_question_id": "lf_whitened_registered_score_direction_against_paired_controls",
            "development_case_id": "paired_clean_lf_whitened_blind_directional_detection",
            "candidate_identity": self.protocol.candidate_identity,
            "candidate_config_digest": self.candidate_config_digest,
            "paired_ablation_identity": "paired_clean_lf_same_prompt_seed_rendered_rgb8",
            "negative_control_case_ids": ("same_image_four_wrong_keys", "paired_clean_primary_null"),
            "metric_ids": metric_ids,
            "content_branch_id": "lf_only",
            "geometry_case_id": "not_applicable",
            "attempt_index": attempt_index,
            "execution_status": "success" if success else ("retry" if retry else "failed"),
            "failure_class": None if success else ("resource_failure" if resource_failure else "implementation_failure"),
            "failure_reason": None if success else "lf_whitened_directional_operation_failed",
            "retry_parent_intent_digest": retry_parent_intent_digest,
            "actual_elapsed_seconds": actual_elapsed_seconds,
            "maximum_duration_seconds": maximum_duration_seconds,
            "duration_limit_exceeded": actual_elapsed_seconds > maximum_duration_seconds,
            "operation_result_payload": operation_payload,
            "operation_result_digest": canonical_development_value_digest(operation_payload),
            "metric_observation": metric,
            "routing_trace": {"routing_used": False},
            "branch_score_trace": ({
                "registered_score": observation.registered_score,
                "primary_null_score": observation.primary_null_score,
                "wrong_key_scores": observation.wrong_key_scores,
                "registered_minus_primary_null": observation.registered_minus_primary_null,
                "registered_minus_max_wrong": observation.registered_minus_max_wrong,
            } if observation is not None else {}),
            "detector_trace": ({
                "public_callable": self.protocol.public_callable,
                "detector_config_digest": observation.detector_config_digest,
                "whitening_asset_digest": observation.whitening_asset_digest,
                "raw_detector_identity": observation.registered_detector_identity,
                "rectified_detector_identity": observation.registered_detector_identity,
                "raw_detector_config_digest": observation.detector_config_digest,
                "rectified_detector_config_digest": observation.detector_config_digest,
                "raw_preprocessing_identity": observation.observation_protocol,
                "rectified_preprocessing_identity": observation.observation_protocol,
                "same_image_registered_four_wrong_reuse": True,
                "paired_clean_primary_null": True,
                "reference_image_used": False,
                "embed_record_used": False,
                "private_latent_used_by_detector": False,
            } if observation is not None else {"formal_detector_completed": False}),
            "geometry_trace": {"geometry_attempted": False},
            "threshold_trace": {"threshold_role": "not_fitted_directional_validation", "raw_threshold_identity": None, "rectified_threshold_identity": None},
            "key_control_trace": {"root_key_public_digest": self.root_key_public_digest, "wrong_key_indexes": tuple(range(self.protocol.wrong_key_roster_size)), "raw_secret_persisted": False},
            "decision_trace": {"positive_source": None, "decision_role": "threshold_free_directional_observation_only" if success else ("retryable_resource_failure" if retry else "terminal_failed_directional_observation")},
            "provenance_trace": {
                "protocol_digest": self.protocol_digest,
                "execution_intent_authority_digest": self.execution_intent_authority_digest,
                "method_code_revision": self.method_code_revision,
                "candidate_config_digest": self.candidate_config_digest,
                "manifest_digest": self.manifest.digest(),
                "cluster_identity": entry.cluster_identity,
                "whitening_asset_fit_producer_revision": self.protocol.whitening_asset_fit_producer_revision,
            },
            "module_outcome": None,
            "candidate_recommendation": None,
            "scientific_claim_boundary": DEVELOPMENT_CLAIM_BOUNDARY,
        }
        provisional = DevelopmentScientificRecord(**payload)
        record = DevelopmentScientificRecord(**{**payload, "record_id": canonical_development_value_digest(provisional.payload_without_record_id())})
        record.validate()
        return record

    def execute_scientific_cluster(self, *, cluster_ordinal: int, base_latent: torch.Tensor, attempt_index: int, retry_parent_intent_digest: str | None, maximum_duration_seconds: int, started_monotonic: float | None = None) -> DevelopmentScientificRecord:
        if type(cluster_ordinal) is not int or not 0 <= cluster_ordinal < SCIENTIFIC_CLUSTER_COUNT:
            raise LfWhitenedDirectionalRunnerError("scientific cluster is outside frozen manifest")
        started = monotonic() if started_monotonic is None else started_monotonic
        result = self._execute_paired_runtime(base_latent)
        observation, detections = self._detect_public_pair(result, cluster_ordinal=cluster_ordinal)
        elapsed = float(monotonic() - started)
        if not isfinite(elapsed) or elapsed < 0.0:
            raise LfWhitenedDirectionalRunnerError("scientific elapsed time is invalid")
        materialization = result.content_materialization
        operation = {
            "candidate_id": result.candidate_id,
            "runtime_config_digest": result.runtime_config_digest,
            "paired_base_latent_digest": result.paired_base_latent_digest,
            "materialization_replay_identity": materialization.materialization_replay_identity,
            "materialization_integrity_status": materialization.integrity_status,
            "realized_total_l2": materialization.realized_total_l2,
            "realized_relative_l2": materialization.realized_relative_l2,
            "content_relative_l2_limit": result.content_materialization_result.content_relative_l2_limit,
            "actual_runtime_dtype": str(materialization.written_latent_actual.dtype),
            "public_lf_detection_results": detections,
            "directional_observation": asdict(observation),
        }
        return self._scientific_record(cluster_ordinal=cluster_ordinal, attempt_index=attempt_index, retry_parent_intent_digest=retry_parent_intent_digest, maximum_duration_seconds=maximum_duration_seconds, actual_elapsed_seconds=elapsed, observation=observation, operation_payload=operation)

    def create_failed_scientific_record(self, *, cluster_ordinal: int, attempt_index: int, retry_parent_intent_digest: str | None, maximum_duration_seconds: int, actual_elapsed_seconds: float, failure_type: str, resource_failure: bool, failure_category: str, failure_diagnostics: dict[str, object] | None = None) -> DevelopmentScientificRecord:
        operation = {"failure_stage": "lf_whitened_directional_runtime_operation", "failure_type": failure_type, "result_available": False, "failure_category": failure_category}
        if failure_diagnostics:
            operation.update(failure_diagnostics)
        return self._scientific_record(cluster_ordinal=cluster_ordinal, attempt_index=attempt_index, retry_parent_intent_digest=retry_parent_intent_digest, maximum_duration_seconds=maximum_duration_seconds, actual_elapsed_seconds=actual_elapsed_seconds, observation=None, operation_payload=operation, resource_failure=resource_failure)

    def replay_directional_aggregate(self, verified_evidence: Sequence[tuple[DevelopmentScientificRecord, CommittedUnit]]) -> LfWhitenedDirectionalAggregate:
        evidence = tuple(verified_evidence)
        if len(evidence) != SCIENTIFIC_CLUSTER_COUNT or tuple(record.unit_index for record, _ in evidence) != tuple(range(1, 33)):
            raise LfWhitenedDirectionalRunnerError("verified directional coverage is incomplete")
        observations: list[LfWhitenedDirectionalObservation] = []
        failure_counts = {
            "implementation_failure": 0,
            "resource_failure": 0,
        }
        violation_counts = {key: 0 for key in ("identity_violation", "budget_violation", "integrity_violation", "nonfinite_violation")}
        sources: set[str] = set()
        for record, marker in evidence:
            try:
                record.validate()
            except DevelopmentRecordError as exc:
                raise LfWhitenedDirectionalRunnerError(
                    "verified directional record is invalid"
                ) from exc
            ordinal = record.unit_index - 1
            identity = AnalysisUnitIdentity(**record.analysis_unit_identity)
            digest = sha256((json.dumps(record.payload(), ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()
            if (
                marker.protocol_digest != self.protocol_digest or marker.run_id != self.run_id
                or marker.revision != self.method_code_revision or marker.record_digest != digest
                or record.protocol_digest != self.protocol_digest
                or record.execution_intent_authority_digest != self.execution_intent_authority_digest
                or record.candidate_config_digest != self.candidate_config_digest
                or record.provenance_trace.get("manifest_digest") != self.manifest.digest()
                or record.provenance_trace.get("whitening_asset_fit_producer_revision") != self.protocol.whitening_asset_fit_producer_revision
                or identity != self._analysis_identity(record.unit_index)
                or identity.source_cluster_id in sources
            ):
                raise LfWhitenedDirectionalRunnerError("verified directional evidence binding drifted")
            sources.add(identity.source_cluster_id)
            if record.execution_status == "success":
                if (
                    record.failure_class is not None
                    or marker.attempt_disposition != "success"
                ):
                    raise LfWhitenedDirectionalRunnerError(
                        "verified successful directional record carries failure"
                    )
            else:
                if (
                    record.execution_status != "failed"
                    or marker.attempt_disposition != "final_failure"
                    or record.failure_class not in failure_counts
                ):
                    raise LfWhitenedDirectionalRunnerError(
                        "verified directional terminal failure class is invalid"
                    )
                failure_counts[record.failure_class] += 1
                category = record.operation_result_payload.get("failure_category")
                if category in violation_counts:
                    violation_counts[str(category)] += 1
                continue
            raw = record.operation_result_payload.get("directional_observation")
            if type(raw) is not dict:
                raise LfWhitenedDirectionalRunnerError("verified directional observation is missing")
            try:
                observation = LfWhitenedDirectionalObservation(**{
                    **raw,
                    "wrong_key_scores": tuple(raw["wrong_key_scores"]),
                    "wrong_key_detector_identities": tuple(raw["wrong_key_detector_identities"]),
                    "wrong_key_template_digests": tuple(raw["wrong_key_template_digests"]),
                    "wrong_key_indexes": tuple(raw["wrong_key_indexes"]),
                })
            except (KeyError, TypeError) as exc:
                raise LfWhitenedDirectionalRunnerError("verified observation schema drifted") from exc
            observation.validate()
            if observation.cluster_ordinal != ordinal:
                raise LfWhitenedDirectionalRunnerError("verified observation ordinal drifted")
            observations.append(observation)
        return aggregate_lf_whitened_direction(
            observations,
            implementation_failure_count=failure_counts[
                "implementation_failure"
            ],
            resource_failure_count=failure_counts["resource_failure"],
            identity_violation_count=violation_counts["identity_violation"],
            budget_violation_count=violation_counts["budget_violation"],
            integrity_violation_count=violation_counts["integrity_violation"],
            nonfinite_violation_count=violation_counts["nonfinite_violation"],
        )


__all__ = [
    "LfWhitenedDirectionalEvidenceViolation",
    "LfWhitenedDirectionalRunnerError",
    "LfWhitenedDirectionalValidationRunner",
]
