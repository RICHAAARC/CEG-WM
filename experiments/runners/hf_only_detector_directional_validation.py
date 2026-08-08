"""Real method/runtime runner for HF-only detector directional validation."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import isfinite
from time import monotonic
from struct import pack
from typing import Sequence

import torch

from experiments.methods import CegWmExperimentAdapter
from experiments.metrics.hf_only_detector_directional_validation import (
    HfDetectorDirectionalAggregate,
    HfDetectorDirectionalObservation,
    aggregate_hf_detector_direction,
    create_hf_detector_directional_observation,
    paired_rgb8_quality,
)
from experiments.protocol.development_records import (
    DEVELOPMENT_CLAIM_BOUNDARY,
    DEVELOPMENT_RECORD_COLLECTION_ROLE,
    METRIC_SCHEMA_VERSION,
    OPERATIONAL_RECORD_COLLECTION_ROLE,
    OPERATIONAL_RECORD_KIND,
    OPERATIONAL_RECORD_SCHEMA,
    RECORD_SCHEMA_VERSION,
    DevelopmentOperationalRecord,
    DevelopmentScientificRecord,
    canonical_development_value_digest,
)
from experiments.protocol.hf_only_detector_directional_validation import (
    HfDetectorDirectionalManifest,
    HfDetectorDirectionalManifestEntry,
    HfOnlyDetectorDirectionalProtocol,
    OPERATIONAL_UNIT_COUNT,
    SCIENTIFIC_CLUSTER_COUNT,
    canonical_digest,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    derive_source_cluster_id,
)
from experiments.runners.development_persistence import (
    CommittedUnit,
    FrozenDevelopmentUnitBinding,
    create_frozen_development_unit_binding,
)
from main import HfDetectionObservation, derive_wrong_key_material
from runtime import ContentWriteVaeResult, Sd35RuntimeAdapter


class HfDetectorDirectionalRunnerError(RuntimeError):
    """The HF detector directional execution violated its frozen contract."""


class HfDetectorDirectionalEvidenceViolation(HfDetectorDirectionalRunnerError):
    """A successful runtime return violated a frozen evidence boundary."""

    def __init__(self, category: str) -> None:
        if category not in {
            "identity_violation",
            "budget_violation",
            "integrity_violation",
            "nonfinite_violation",
        }:
            raise ValueError("directional violation category is invalid")
        self.category = category
        super().__init__(category)


def _tensor_values(tensor: torch.Tensor) -> tuple[float, ...]:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 4
        or tensor.shape[0] != 1
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise HfDetectorDirectionalRunnerError("public detector tensor is invalid")
    return tuple(
        float(item)
        for item in tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    )


def _public_observation(tensor: torch.Tensor) -> HfDetectionObservation:
    return HfDetectionObservation.from_public_image_encoding(
        _tensor_values(tensor), tuple(int(size) for size in tensor.shape)
    )


def _rgb8_values(image: torch.Tensor) -> tuple[int, ...]:
    if (
        not isinstance(image, torch.Tensor)
        or image.ndim != 4
        or tuple(image.shape[:2]) != (1, 3)
        or not bool(torch.isfinite(image).all().item())
    ):
        raise HfDetectorDirectionalRunnerError("final RGB tensor is invalid")
    rgb8 = image.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    rgb8 = torch.round(rgb8 * 255.0).to(dtype=torch.uint8)
    return tuple(int(item) for item in rgb8.reshape(-1).tolist())


class HfOnlyDetectorDirectionalRunner:
    """Runs paired generation followed by three public blind HF detections."""

    def __init__(
        self,
        *,
        protocol: HfOnlyDetectorDirectionalProtocol,
        manifest: HfDetectorDirectionalManifest,
        adapter: CegWmExperimentAdapter,
        runtime_adapter: Sd35RuntimeAdapter,
        method_code_revision: str,
        run_id: str,
        registered_root_key: str,
        root_key_public_digest: str,
        protocol_digest: str,
        execution_intent_authority_digest: str,
        candidate_config_digest: str,
    ) -> None:
        protocol.validate()
        manifest.validate()
        if type(adapter) is not CegWmExperimentAdapter:
            raise HfDetectorDirectionalRunnerError(
                "method adapter exact type is required"
            )
        if type(runtime_adapter) is not Sd35RuntimeAdapter:
            raise HfDetectorDirectionalRunnerError(
                "runtime adapter exact type is required"
            )
        if type(method_code_revision) is not str or len(method_code_revision) != 40:
            raise HfDetectorDirectionalRunnerError("method revision is invalid")
        self.protocol = protocol
        self.manifest = manifest
        self.adapter = adapter
        self.runtime = runtime_adapter
        self.method_code_revision = method_code_revision
        self.run_id = run_id
        self.registered_root_key = registered_root_key
        self.root_key_public_digest = root_key_public_digest
        self.protocol_digest = protocol_digest
        self.execution_intent_authority_digest = execution_intent_authority_digest
        self.candidate_config_digest = candidate_config_digest

    def _entry_for_unit(self, unit_index: int) -> HfDetectorDirectionalManifestEntry:
        if unit_index < OPERATIONAL_UNIT_COUNT:
            return self.manifest.operational_entries[unit_index]
        return self.manifest.scientific_entries[unit_index - OPERATIONAL_UNIT_COUNT]

    def _analysis_identity(
        self, entry: HfDetectorDirectionalManifestEntry
    ) -> AnalysisUnitIdentity:
        key_family = canonical_digest(
            {
                "root_key_public_digest": self.root_key_public_digest,
                "seed_namespace": self.manifest.seed_namespace,
                "role": "registered_hf_detector_directional_key_family",
            }
        )
        source_cluster_id = derive_source_cluster_id(
            prompt_digest=entry.prompt_digest,
            generation_seed=entry.generation_seed,
            image_lineage_digest=entry.image_lineage_digest,
            registered_key_family_digest=key_family,
        )
        return AnalysisUnitIdentity(
            unit_id=f"hf_detector_{entry.cluster_identity}",
            case_id=(
                "paired_clean_hf_operational_smoke"
                if entry.entry_role == "operational_smoke"
                else "paired_clean_hf_blind_directional_detection"
            ),
            source_cluster_id=source_cluster_id,
            prompt_digest=entry.prompt_digest,
            generation_seed=entry.generation_seed,
            image_lineage_digest=entry.image_lineage_digest,
            registered_key_family_digest=key_family,
        )

    def create_persistence_unit_bindings(
        self,
    ) -> tuple[FrozenDevelopmentUnitBinding, ...]:
        bindings = []
        for unit in self.protocol.unit_roster:
            entry = self._entry_for_unit(unit.unit_index)
            bindings.append(
                create_frozen_development_unit_binding(
                    unit,
                    analysis_unit_identity=self._analysis_identity(entry),
                    scientific_question_id=(
                        "hf_public_detector_execution_smoke"
                        if unit.unit_index < OPERATIONAL_UNIT_COUNT
                        else "hf_registered_score_direction_against_paired_controls"
                    ),
                    development_case_id=(
                        "paired_clean_hf_operational_smoke"
                        if unit.unit_index < OPERATIONAL_UNIT_COUNT
                        else "paired_clean_hf_blind_directional_detection"
                    ),
                    candidate_identity=self.protocol.candidate_identity,
                    candidate_config_digest=self.candidate_config_digest,
                )
            )
        return tuple(bindings)

    def _execute_paired_runtime(self, base_latent: torch.Tensor) -> ContentWriteVaeResult:
        shape = tuple(int(size) for size in base_latent.shape)
        carrier = self.adapter.build_hf_carrier(
            self.registered_root_key, shape
        ).result

        def embed(values: tuple[float, ...]):
            return self.adapter.embed_content(values, carrier).result

        result = self.runtime.execute_content_write_and_vae(base_latent, embed)
        if type(result) is not ContentWriteVaeResult:
            raise HfDetectorDirectionalRunnerError(
                "paired runtime result exact type is required"
            )
        return result

    def _detect_public_rgb_pair(
        self,
        runtime_result: ContentWriteVaeResult,
        *,
        cluster_ordinal: int,
    ) -> tuple[HfDetectorDirectionalObservation, dict[str, object]]:
        candidate_observation = _public_observation(
            runtime_result.watermarked_detection_latent
        )
        clean_observation = _public_observation(runtime_result.clean_detection_latent)
        wrong_key_index = cluster_ordinal % self.protocol.wrong_key_roster_size
        wrong_key = derive_wrong_key_material(
            self.root_key_public_digest, wrong_key_index
        )
        registered = self.adapter.detect_hf(
            candidate_observation, self.registered_root_key
        ).result
        wrong = self.adapter.detect_hf(candidate_observation, wrong_key).result
        primary_null = self.adapter.detect_hf(
            clean_observation, self.registered_root_key
        ).result
        if not (
            registered.detector_config_digest
            == wrong.detector_config_digest
            == primary_null.detector_config_digest
        ):
            raise HfDetectorDirectionalRunnerError(
                "formal HF detector configuration drifted"
            )
        preprocessing_identity = candidate_observation.observation_protocol
        if clean_observation.observation_protocol != preprocessing_identity:
            raise HfDetectorDirectionalRunnerError(
                "formal HF detector preprocessing drifted"
            )
        statistic_identity = "hf_direct_score_centered_normalized_correlation"
        detector_identity = canonical_digest(
            {
                "candidate_identity": self.protocol.candidate_identity,
                "detector_operation_identity": self.protocol.detector_operation_identity,
                "detector_config_digest": registered.detector_config_digest,
                "preprocessing_identity": preprocessing_identity,
                "statistic_identity": statistic_identity,
            }
        )
        materialization = runtime_result.content_materialization
        materialization_result = runtime_result.content_materialization_result
        if (
            pack(">f", materialization_result.content_relative_l2_nominal)
            != pack(">f", 3 / 250)
            or pack(">f", materialization_result.content_relative_l2_limit)
            != pack(">f", 3 / 250)
            or materialization_result.budget_status != "accepted"
            or materialization.realized_relative_l2 > 3 / 250
        ):
            raise HfDetectorDirectionalEvidenceViolation("budget_violation")
        if (
            materialization.integrity_status != "passed"
            or materialization_result.integrity_status != "passed"
        ):
            raise HfDetectorDirectionalEvidenceViolation("integrity_violation")
        rgb_relative_l2, rgb_mse = paired_rgb8_quality(
            _rgb8_values(runtime_result.watermarked_image),
            _rgb8_values(runtime_result.clean_image),
        )
        observation = create_hf_detector_directional_observation(
            cluster_ordinal=cluster_ordinal,
            wrong_key_index=wrong_key_index,
            registered_score=registered.hf_score,
            wrong_key_score=wrong.hf_score,
            primary_null_score=primary_null.hf_score,
            candidate_observation_digest=registered.observation_digest,
            clean_observation_digest=primary_null.observation_digest,
            registered_detector_identity=detector_identity,
            wrong_key_detector_identity=detector_identity,
            primary_null_detector_identity=detector_identity,
            detector_config_digest=registered.detector_config_digest,
            observation_protocol=candidate_observation.observation_protocol,
            detector_statistic_identity=statistic_identity,
            registered_template_digest=registered.template_digest,
            wrong_key_template_digest=wrong.template_digest,
            primary_null_template_digest=primary_null.template_digest,
            registered_root_key_public_digest=registered.root_key_public_digest,
            wrong_key_root_key_public_digest=wrong.root_key_public_digest,
            primary_null_root_key_public_digest=primary_null.root_key_public_digest,
            materialization_integrity_status=materialization.integrity_status,
            realized_relative_l2=materialization.realized_relative_l2,
            content_relative_l2_limit=materialization_result.content_relative_l2_limit,
            rgb_paired_relative_l2=rgb_relative_l2,
            rgb_paired_mse=rgb_mse,
            rgb_quality_dtype="torch.uint8",
            actual_runtime_dtype=str(materialization.written_latent_actual.dtype),
        )
        return observation, {
            "registered": asdict(registered),
            "wrong_key": asdict(wrong),
            "primary_null": asdict(primary_null),
        }

    def execute_operational_smoke(
        self,
        *,
        unit_index: int,
        base_latent: torch.Tensor,
        attempt_index: int,
        retry_parent_intent_digest: str | None,
        maximum_duration_seconds: int,
        started_monotonic: float | None = None,
    ) -> DevelopmentOperationalRecord:
        if not 0 <= unit_index < OPERATIONAL_UNIT_COUNT:
            raise HfDetectorDirectionalRunnerError(
                "operational unit is outside frozen roster"
            )
        started = monotonic() if started_monotonic is None else started_monotonic
        result = self._execute_paired_runtime(base_latent)
        observation, detector_results = self._detect_public_rgb_pair(
            result, cluster_ordinal=unit_index
        )
        elapsed = float(monotonic() - started)
        payload = {
            "operational_role": "environment_runtime_throughput_preflight",
            "source_cluster_ordinal": unit_index,
            "case_ids": ["paired_clean_hf_public_detector_smoke"],
            "responsibility_result_digests": [
                [
                    "content_embedder",
                    canonical_digest(
                        {
                            "candidate_id": result.candidate_id,
                            "detector_config_digest": observation.detector_config_digest,
                            "observation_identity": observation.observation_identity,
                            "registered_detector_identity": detector_results["registered"]["detector_identity"],
                        }
                    ),
                ]
            ],
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
            unit_index=unit_index,
            phase="development_environment_preflight",
            source_cluster_ordinal=unit_index,
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
        record = replace(
            record,
            record_id=canonical_development_value_digest(
                record.payload_without_record_id()
            ),
        )
        record.validate()
        return record

    def execute_scientific_cluster(
        self,
        *,
        cluster_ordinal: int,
        base_latent: torch.Tensor,
        attempt_index: int,
        retry_parent_intent_digest: str | None,
        maximum_duration_seconds: int,
        started_monotonic: float | None = None,
    ) -> DevelopmentScientificRecord:
        if not 0 <= cluster_ordinal < SCIENTIFIC_CLUSTER_COUNT:
            raise HfDetectorDirectionalRunnerError(
                "scientific cluster is outside frozen manifest"
            )
        started = monotonic() if started_monotonic is None else started_monotonic
        result = self._execute_paired_runtime(base_latent)
        observation, detector_results = self._detect_public_rgb_pair(
            result, cluster_ordinal=cluster_ordinal
        )
        elapsed = float(monotonic() - started)
        if not isfinite(elapsed) or elapsed < 0.0:
            raise HfDetectorDirectionalRunnerError("scientific elapsed time is invalid")
        entry = self.manifest.scientific_entries[cluster_ordinal]
        identity = self._analysis_identity(entry)
        materialization = result.content_materialization
        operation_payload = {
            "candidate_id": result.candidate_id,
            "runtime_config_digest": result.runtime_config_digest,
            "paired_base_latent_digest": result.paired_base_latent_digest,
            "materialization_replay_identity": materialization.materialization_replay_identity,
            "materialization_integrity_status": materialization.integrity_status,
            "realized_total_l2": materialization.realized_total_l2,
            "realized_relative_l2": materialization.realized_relative_l2,
            "content_relative_l2_nominal": result.content_materialization_result.content_relative_l2_nominal,
            "content_relative_l2_limit": result.content_materialization_result.content_relative_l2_limit,
            "rgb_paired_relative_l2": observation.rgb_paired_relative_l2,
            "rgb_paired_mse": observation.rgb_paired_mse,
            "rgb_quality_dtype": observation.rgb_quality_dtype,
            "actual_runtime_dtype": observation.actual_runtime_dtype,
            "formal_hf_detection_results": detector_results,
            "directional_observation": asdict(observation),
        }
        metric_payload = {
            "schema_version": METRIC_SCHEMA_VERSION,
            "metric_role": "development_exploratory_cluster_level",
            "responsibility_id": "hf_detector",
            "source_cluster_id": identity.source_cluster_id,
            "registered_metric_ids": (
                "registered_primary_null_directional_margin",
                "registered_wrong_key_directional_margin",
                "paired_rgb_quality",
            ),
            "candidate_config_digest": self.candidate_config_digest,
            "paired_ablation_identity": "paired_clean_hf_same_prompt_seed_final_rgb",
            "content_branch_id": "hf_only",
            "geometry_case_id": "not_applicable",
            "sufficient_statistics": (
                (
                    "registered_minus_primary_null",
                    observation.registered_minus_primary_null,
                ),
                (
                    "registered_minus_wrong_key",
                    observation.registered_minus_wrong_key,
                ),
                ("realized_relative_l2", observation.realized_relative_l2),
                ("rgb_paired_relative_l2", observation.rgb_paired_relative_l2),
            ),
            "result_identity_digests": (observation.observation_identity,),
            "threshold_role": "not_fitted_directional_validation",
            "threshold_identity": None,
            "threshold_fit_source_cluster_digest": None,
        }
        metric_payload["observation_digest"] = canonical_development_value_digest(
            metric_payload
        )
        record_payload = {
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
            "responsibility_id": "hf_detector",
            "scientific_question_id": "hf_registered_score_direction_against_paired_controls",
            "development_case_id": "paired_clean_hf_blind_directional_detection",
            "candidate_identity": self.protocol.candidate_identity,
            "candidate_config_digest": self.candidate_config_digest,
            "paired_ablation_identity": "paired_clean_hf_same_prompt_seed_final_rgb",
            "negative_control_case_ids": (
                "same_image_wrong_key",
                "paired_clean_primary_null",
            ),
            "metric_ids": (
                "registered_primary_null_directional_margin",
                "registered_wrong_key_directional_margin",
                "paired_rgb_quality",
            ),
            "content_branch_id": "hf_only",
            "geometry_case_id": "not_applicable",
            "attempt_index": attempt_index,
            "execution_status": "success",
            "failure_class": None,
            "failure_reason": None,
            "retry_parent_intent_digest": retry_parent_intent_digest,
            "actual_elapsed_seconds": elapsed,
            "maximum_duration_seconds": maximum_duration_seconds,
            "duration_limit_exceeded": elapsed > maximum_duration_seconds,
            "operation_result_payload": operation_payload,
            "operation_result_digest": canonical_development_value_digest(
                operation_payload
            ),
            "metric_observation": metric_payload,
            "routing_trace": {"routing_used": False},
            "branch_score_trace": {
                "registered_score": observation.registered_score,
                "wrong_key_score": observation.wrong_key_score,
                "primary_null_score": observation.primary_null_score,
                "registered_minus_wrong_key": observation.registered_minus_wrong_key,
                "registered_minus_primary_null": observation.registered_minus_primary_null,
            },
            "detector_trace": {
                "detector_operation_identity": self.protocol.detector_operation_identity,
                "detector_config_digest": observation.detector_config_digest,
                "preprocessing_identity": observation.observation_protocol,
                "statistic_identity": "hf_direct_score_centered_normalized_correlation",
                "registered_detector_identity": observation.registered_detector_identity,
                "wrong_key_detector_identity": observation.wrong_key_detector_identity,
                "primary_null_detector_identity": observation.primary_null_detector_identity,
                "candidate_observation_digest": observation.candidate_observation_digest,
                "clean_observation_digest": observation.clean_observation_digest,
                "same_image_registered_wrong_reuse": True,
                "paired_clean_primary_null": True,
                "reference_image_used": False,
                "embed_record_used": False,
                "private_latent_used_by_detector": False,
            },
            "geometry_trace": {"geometry_attempted": False},
            "threshold_trace": {
                "threshold_role": "not_fitted_directional_validation",
                "raw_threshold_identity": None,
                "rectified_threshold_identity": None,
            },
            "key_control_trace": {
                "root_key_public_digest": self.root_key_public_digest,
                "wrong_key_index": observation.wrong_key_index,
                "wrong_key_roster_size": self.protocol.wrong_key_roster_size,
                "raw_secret_persisted": False,
            },
            "decision_trace": {
                "positive_source": None,
                "decision_role": "threshold_free_directional_observation_only",
                "practical_margin_floor": self.protocol.practical_margin_floor,
            },
            "provenance_trace": {
                "protocol_digest": self.protocol_digest,
                "execution_intent_authority_digest": self.execution_intent_authority_digest,
                "method_code_revision": self.method_code_revision,
                "candidate_config_digest": self.candidate_config_digest,
                "manifest_digest": self.manifest.digest(),
                "cluster_identity": entry.cluster_identity,
            },
            "module_outcome": None,
            "candidate_recommendation": None,
            "scientific_claim_boundary": DEVELOPMENT_CLAIM_BOUNDARY,
        }
        provisional = DevelopmentScientificRecord(**record_payload)
        record = DevelopmentScientificRecord(
            **{
                **record_payload,
                "record_id": canonical_development_value_digest(
                    provisional.payload_without_record_id()
                ),
            }
        )
        record.validate()
        return record

    def create_failed_scientific_record(
        self,
        *,
        cluster_ordinal: int,
        attempt_index: int,
        retry_parent_intent_digest: str | None,
        maximum_duration_seconds: int,
        actual_elapsed_seconds: float,
        failure_type: str,
        resource_failure: bool,
        failure_category: str,
    ) -> DevelopmentScientificRecord:
        if not 0 <= cluster_ordinal < SCIENTIFIC_CLUSTER_COUNT:
            raise HfDetectorDirectionalRunnerError(
                "failed cluster is outside frozen manifest"
            )
        entry = self.manifest.scientific_entries[cluster_ordinal]
        identity = self._analysis_identity(entry)
        operation_payload = {
            "failure_stage": "hf_detector_directional_runtime_operation",
            "failure_type": failure_type,
            "result_available": False,
            "failure_category": failure_category,
        }
        record_payload = {
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
            "responsibility_id": "hf_detector",
            "scientific_question_id": "hf_registered_score_direction_against_paired_controls",
            "development_case_id": "paired_clean_hf_blind_directional_detection",
            "candidate_identity": self.protocol.candidate_identity,
            "candidate_config_digest": self.candidate_config_digest,
            "paired_ablation_identity": "paired_clean_hf_same_prompt_seed_final_rgb",
            "negative_control_case_ids": (
                "same_image_wrong_key",
                "paired_clean_primary_null",
            ),
            "metric_ids": (
                "registered_primary_null_directional_margin",
                "registered_wrong_key_directional_margin",
                "paired_rgb_quality",
            ),
            "content_branch_id": "hf_only",
            "geometry_case_id": "not_applicable",
            "attempt_index": attempt_index,
            "execution_status": "failed",
            "failure_class": "resource_failure" if resource_failure else "implementation_failure",
            "failure_reason": "hf_detector_directional_operation_failed",
            "retry_parent_intent_digest": retry_parent_intent_digest,
            "actual_elapsed_seconds": actual_elapsed_seconds,
            "maximum_duration_seconds": maximum_duration_seconds,
            "duration_limit_exceeded": actual_elapsed_seconds > maximum_duration_seconds,
            "operation_result_payload": operation_payload,
            "operation_result_digest": canonical_development_value_digest(
                operation_payload
            ),
            "metric_observation": {},
            "routing_trace": {"routing_used": False},
            "branch_score_trace": {},
            "detector_trace": {"formal_detector_completed": False},
            "geometry_trace": {"geometry_attempted": False},
            "threshold_trace": {
                "threshold_role": "not_fitted_directional_validation",
                "raw_threshold_identity": None,
                "rectified_threshold_identity": None,
            },
            "key_control_trace": {
                "root_key_public_digest": self.root_key_public_digest,
                "wrong_key_index": cluster_ordinal % self.protocol.wrong_key_roster_size,
                "raw_secret_persisted": False,
            },
            "decision_trace": {
                "positive_source": None,
                "decision_role": "failed_directional_observation",
            },
            "provenance_trace": {
                "protocol_digest": self.protocol_digest,
                "execution_intent_authority_digest": self.execution_intent_authority_digest,
                "method_code_revision": self.method_code_revision,
                "candidate_config_digest": self.candidate_config_digest,
                "manifest_digest": self.manifest.digest(),
                "cluster_identity": entry.cluster_identity,
            },
            "module_outcome": None,
            "candidate_recommendation": None,
            "scientific_claim_boundary": DEVELOPMENT_CLAIM_BOUNDARY,
        }
        provisional = DevelopmentScientificRecord(**record_payload)
        record = DevelopmentScientificRecord(
            **{
                **record_payload,
                "record_id": canonical_development_value_digest(
                    provisional.payload_without_record_id()
                ),
            }
        )
        record.validate()
        return record

    def replay_directional_aggregate(
        self,
        verified_evidence: Sequence[
            tuple[DevelopmentScientificRecord, CommittedUnit]
        ],
    ) -> HfDetectorDirectionalAggregate:
        evidence = tuple(verified_evidence)
        if len(evidence) != SCIENTIFIC_CLUSTER_COUNT:
            raise HfDetectorDirectionalRunnerError(
                "verified directional coverage is incomplete"
            )
        indexes = tuple(record.unit_index for record, _marker in evidence)
        expected_indexes = tuple(
            range(OPERATIONAL_UNIT_COUNT, OPERATIONAL_UNIT_COUNT + SCIENTIFIC_CLUSTER_COUNT)
        )
        if indexes != expected_indexes or len(set(indexes)) != len(indexes):
            raise HfDetectorDirectionalRunnerError(
                "verified directional unit indexes drifted"
            )
        observations = []
        failure_count = 0
        violation_counts = {
            "identity_violation": 0,
            "budget_violation": 0,
            "integrity_violation": 0,
            "nonfinite_violation": 0,
        }
        source_ids: set[str] = set()
        for record, marker in evidence:
            record.validate()
            cluster_ordinal = record.unit_index - OPERATIONAL_UNIT_COUNT
            entry = self.manifest.scientific_entries[cluster_ordinal]
            identity = AnalysisUnitIdentity(**record.analysis_unit_identity)
            marker_record_digest = sha256(
                (
                    json.dumps(
                        record.payload(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            ).hexdigest()
            if (
                marker.protocol_digest != self.protocol_digest
                or marker.run_id != self.run_id
                or marker.revision != self.method_code_revision
                or marker.record_digest != marker_record_digest
                or record.run_id != self.run_id
                or record.protocol_digest != self.protocol_digest
                or record.execution_intent_authority_digest
                != self.execution_intent_authority_digest
                or record.candidate_config_digest != self.candidate_config_digest
                or record.provenance_trace.get("manifest_digest")
                != self.manifest.digest()
                or identity != self._analysis_identity(entry)
                or identity.source_cluster_id in source_ids
            ):
                raise HfDetectorDirectionalRunnerError(
                    "verified directional evidence binding drifted"
                )
            source_ids.add(identity.source_cluster_id)
            if record.execution_status != "success":
                failure_count += 1
                category = record.operation_result_payload.get("failure_category")
                if category in violation_counts:
                    violation_counts[category] += 1
                continue
            raw = record.operation_result_payload.get("directional_observation")
            if type(raw) is not dict:
                raise HfDetectorDirectionalRunnerError(
                    "verified directional observation is missing"
                )
            try:
                observation = HfDetectorDirectionalObservation(**raw)
            except TypeError as exc:
                raise HfDetectorDirectionalRunnerError(
                    "verified directional observation schema drifted"
                ) from exc
            observation.validate()
            observations.append(observation)
        return aggregate_hf_detector_direction(
            observations,
            failed_cluster_count=failure_count,
            identity_violation_count=violation_counts["identity_violation"],
            budget_violation_count=violation_counts["budget_violation"],
            integrity_violation_count=violation_counts["integrity_violation"],
            nonfinite_violation_count=violation_counts["nonfinite_violation"],
        )


__all__ = [
    "HfDetectorDirectionalRunnerError",
    "HfDetectorDirectionalEvidenceViolation",
    "HfOnlyDetectorDirectionalRunner",
]
