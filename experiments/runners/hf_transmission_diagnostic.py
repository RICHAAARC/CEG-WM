"""Real method/runtime runner for the eight-cluster HF transport diagnostic."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite
from time import monotonic
from typing import Sequence

import torch

from experiments.methods import CegWmExperimentAdapter
from experiments.metrics.hf_transmission_diagnostic import (
    HfSignalPositionObservation,
    HfTransmissionDirectionalDecision,
    create_hf_signal_position_observation,
    diagnostic_latent_template_projection,
    evaluate_hf_transmission_direction,
)
from experiments.protocol.development_records import (
    DEVELOPMENT_CLAIM_BOUNDARY,
    DEVELOPMENT_RECORD_COLLECTION_ROLE,
    METRIC_SCHEMA_VERSION,
    RECORD_SCHEMA_VERSION,
    DevelopmentScientificRecord,
    canonical_development_value_digest,
)
from experiments.protocol.hf_transmission_diagnostic import (
    HfTransmissionDiagnosticProtocol,
    HfTransmissionManifest,
    HfTransmissionManifestEntry,
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


class HfTransmissionRunnerError(RuntimeError):
    """The real HF transport diagnostic violated its frozen contract."""


def _tensor_values(tensor: torch.Tensor) -> tuple[float, ...]:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 4
        or tensor.shape[0] != 1
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise HfTransmissionRunnerError("HF signal tensor is invalid")
    return tuple(
        float(item)
        for item in tensor.detach().to(
            device="cpu", dtype=torch.float32
        ).reshape(-1)
    )


def _observation(tensor: torch.Tensor) -> HfDetectionObservation:
    return HfDetectionObservation.from_public_image_encoding(
        _tensor_values(tensor),
        tuple(int(size) for size in tensor.shape),
    )


class HfTransmissionDiagnosticRunner:
    """Calls the registered HF carrier, embedder, runtime and blind detector."""

    def __init__(
        self,
        *,
        protocol: HfTransmissionDiagnosticProtocol,
        manifest: HfTransmissionManifest,
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
            raise HfTransmissionRunnerError("method adapter exact type is required")
        if type(runtime_adapter) is not Sd35RuntimeAdapter:
            raise HfTransmissionRunnerError("runtime adapter exact type is required")
        if type(method_code_revision) is not str or len(method_code_revision) != 40:
            raise HfTransmissionRunnerError("method revision is invalid")
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

    def _analysis_identity(
        self, entry: HfTransmissionManifestEntry
    ) -> AnalysisUnitIdentity:
        key_family = canonical_digest(
            {
                "root_key_public_digest": self.root_key_public_digest,
                "seed_namespace": self.manifest.seed_namespace,
                "role": "registered_hf_transmission_detection_key_family",
            }
        )
        source_cluster_id = derive_source_cluster_id(
            prompt_digest=entry.prompt_digest,
            generation_seed=entry.generation_seed,
            image_lineage_digest=entry.image_lineage_digest,
            registered_key_family_digest=key_family,
        )
        return AnalysisUnitIdentity(
            unit_id=f"hf_transmission_cluster_{entry.cluster_ordinal:02d}",
            case_id="paired_clean_hf_transport_observation",
            source_cluster_id=source_cluster_id,
            prompt_digest=entry.prompt_digest,
            generation_seed=entry.generation_seed,
            image_lineage_digest=entry.image_lineage_digest,
            registered_key_family_digest=key_family,
        )

    def create_persistence_unit_bindings(
        self,
    ) -> tuple[FrozenDevelopmentUnitBinding, ...]:
        bindings: list[FrozenDevelopmentUnitBinding] = []
        for unit in self.protocol.unit_roster:
            entry = self.manifest.entries[unit.source_cluster_ordinal]
            operational = unit.unit_index < self.protocol.operational_unit_count
            bindings.append(
                create_frozen_development_unit_binding(
                    unit,
                    analysis_unit_identity=self._analysis_identity(entry),
                    scientific_question_id=(
                        "hf_transmission_runtime_smoke"
                        if operational
                        else "hf_signal_survival_across_generation_boundaries"
                    ),
                    development_case_id=(
                        "hf_transmission_operational_smoke"
                        if operational
                        else "paired_clean_hf_transport_observation"
                    ),
                    candidate_identity=(
                        "hf_transmission_execution_environment"
                        if operational
                        else self.protocol.candidate_identity
                    ),
                    candidate_config_digest=self.candidate_config_digest,
                )
            )
        return tuple(bindings)

    def _score_position(
        self,
        position_id: str,
        candidate_tensor: torch.Tensor,
        clean_tensor: torch.Tensor,
    ) -> HfSignalPositionObservation:
        if position_id == "rgb_vae_reencoded":
            raise HfTransmissionRunnerError(
                "final public-image position requires the formal detector path"
            )
        candidate_values = _tensor_values(candidate_tensor)
        clean_values = _tensor_values(clean_tensor)
        wrong_key = derive_wrong_key_material(self.root_key_public_digest, 0)
        shape = tuple(int(size) for size in candidate_tensor.shape)
        registered_carrier = self.adapter.build_hf_carrier(
            self.registered_root_key, shape
        ).result
        wrong_carrier = self.adapter.build_hf_carrier(wrong_key, shape).result
        registered_score = diagnostic_latent_template_projection(
            candidate_values, registered_carrier.template
        )
        wrong_score = diagnostic_latent_template_projection(
            candidate_values, wrong_carrier.template
        )
        primary_null_score = diagnostic_latent_template_projection(
            clean_values, registered_carrier.template
        )
        return create_hf_signal_position_observation(
            position_id=position_id,
            statistic_role="diagnostic_latent_template_projection",
            registered_score=registered_score,
            wrong_key_score=wrong_score,
            primary_null_score=primary_null_score,
            registered_observation_digest=sha256(
                json.dumps(candidate_values).encode("utf-8")
            ).hexdigest(),
            primary_null_observation_digest=sha256(
                json.dumps(clean_values).encode("utf-8")
            ).hexdigest(),
            registered_statistic_identity=canonical_digest(
                {
                    "metric": "diagnostic_latent_template_projection",
                    "carrier_config_digest": (
                        registered_carrier.carrier_config_digest
                    ),
                    "template_digest": registered_carrier.template_digest,
                    "key_role": registered_carrier.key_role,
                    "position": position_id,
                }
            ),
            wrong_key_statistic_identity=canonical_digest(
                {
                    "metric": "diagnostic_latent_template_projection",
                    "carrier_config_digest": wrong_carrier.carrier_config_digest,
                    "template_digest": wrong_carrier.template_digest,
                    "key_role": wrong_carrier.key_role,
                    "wrong_key_index": wrong_carrier.wrong_key_index,
                    "position": position_id,
                }
            ),
            primary_null_statistic_identity=canonical_digest(
                {
                    "metric": "diagnostic_latent_template_projection",
                    "carrier_config_digest": (
                        registered_carrier.carrier_config_digest
                    ),
                    "template_digest": registered_carrier.template_digest,
                    "key_role": registered_carrier.key_role,
                    "position": position_id,
                    "control": "paired_clean_primary_null",
                }
            ),
            registered_template_digest=registered_carrier.template_digest,
            wrong_key_template_digest=wrong_carrier.template_digest,
            primary_null_template_digest=registered_carrier.template_digest,
            registered_root_key_public_digest=(
                registered_carrier.root_key_public_digest
            ),
            wrong_key_root_key_public_digest=wrong_carrier.root_key_public_digest,
            primary_null_root_key_public_digest=(
                registered_carrier.root_key_public_digest
            ),
            registered_key_role=registered_carrier.key_role,
            wrong_key_key_role=wrong_carrier.key_role,
            primary_null_key_role=registered_carrier.key_role,
            registered_wrong_key_index=registered_carrier.wrong_key_index,
            wrong_key_index=wrong_carrier.wrong_key_index,
            primary_null_wrong_key_index=registered_carrier.wrong_key_index,
        )

    def _score_final_public_image_position(
        self,
        candidate_tensor: torch.Tensor,
        clean_tensor: torch.Tensor,
    ) -> tuple[HfSignalPositionObservation, dict[str, object]]:
        candidate_observation = _observation(candidate_tensor)
        clean_observation = _observation(clean_tensor)
        wrong_key = derive_wrong_key_material(self.root_key_public_digest, 0)
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
            raise HfTransmissionRunnerError("HF detector configuration drifted")
        observation = create_hf_signal_position_observation(
            position_id="rgb_vae_reencoded",
            statistic_role="formal_hf_detector_operation",
            registered_score=registered.hf_score,
            wrong_key_score=wrong.hf_score,
            primary_null_score=primary_null.hf_score,
            registered_observation_digest=registered.observation_digest,
            primary_null_observation_digest=primary_null.observation_digest,
            registered_statistic_identity=registered.detector_identity,
            wrong_key_statistic_identity=wrong.detector_identity,
            primary_null_statistic_identity=primary_null.detector_identity,
            registered_template_digest=registered.template_digest,
            wrong_key_template_digest=wrong.template_digest,
            primary_null_template_digest=primary_null.template_digest,
            registered_root_key_public_digest=registered.root_key_public_digest,
            wrong_key_root_key_public_digest=wrong.root_key_public_digest,
            primary_null_root_key_public_digest=(
                primary_null.root_key_public_digest
            ),
            registered_key_role=registered.key_role,
            wrong_key_key_role=wrong.key_role,
            primary_null_key_role=primary_null.key_role,
            registered_wrong_key_index=registered.wrong_key_index,
            wrong_key_index=wrong.wrong_key_index,
            primary_null_wrong_key_index=primary_null.wrong_key_index,
        )
        return observation, {
            "registered": asdict(registered),
            "wrong_key": asdict(wrong),
            "primary_null": asdict(primary_null),
        }

    def execute_operational_smoke(
        self,
        *,
        base_latent: torch.Tensor,
    ) -> tuple[ContentWriteVaeResult, float]:
        """Run one real paired carrier/embedder/runtime operation without science."""

        started = monotonic()
        shape = tuple(int(size) for size in base_latent.shape)
        carrier = self.adapter.build_hf_carrier(
            self.registered_root_key, shape
        ).result

        def embed(values: tuple[float, ...]):
            return self.adapter.embed_content(values, carrier).result

        result = self.runtime.execute_content_write_and_vae(base_latent, embed)
        if type(result) is not ContentWriteVaeResult:
            raise HfTransmissionRunnerError("runtime result exact type is required")
        elapsed = float(monotonic() - started)
        if not isfinite(elapsed) or elapsed <= 0.0:
            raise HfTransmissionRunnerError("operational elapsed time was not measured")
        return result, elapsed

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
        if type(cluster_ordinal) is not int or not 0 <= cluster_ordinal < 8:
            raise HfTransmissionRunnerError("cluster ordinal is outside frozen manifest")
        started = monotonic() if started_monotonic is None else started_monotonic
        runtime_result, _runtime_elapsed = self.execute_operational_smoke(
            base_latent=base_latent
        )
        materialization = runtime_result.content_materialization
        final_position, formal_detector_results = (
            self._score_final_public_image_position(
                runtime_result.watermarked_detection_latent,
                runtime_result.clean_detection_latent,
            )
        )
        positions = (
            self._score_position(
                "callback_pre_write",
                materialization.baseline_latent_actual,
                materialization.baseline_latent_actual,
            ),
            self._score_position(
                "actual_dtype_post_write",
                materialization.written_latent_actual,
                materialization.baseline_latent_actual,
            ),
            self._score_position(
                "scheduler_suffix_final",
                runtime_result.watermarked_generation_terminal_latent,
                runtime_result.clean_generation_terminal_latent,
            ),
            final_position,
        )
        elapsed = float(monotonic() - started)
        if not isfinite(elapsed) or elapsed < 0.0:
            raise HfTransmissionRunnerError("cluster elapsed time is invalid")
        entry = self.manifest.entries[cluster_ordinal]
        identity = self._analysis_identity(entry)
        operation_payload = {
            "candidate_id": runtime_result.candidate_id,
            "runtime_config_digest": runtime_result.runtime_config_digest,
            "paired_base_latent_digest": runtime_result.paired_base_latent_digest,
            "materialization_replay_identity": (
                materialization.materialization_replay_identity
            ),
            "materialization_integrity_status": materialization.integrity_status,
            "realized_total_l2": materialization.realized_total_l2,
            "realized_relative_l2": materialization.realized_relative_l2,
            "signal_positions": tuple(asdict(item) for item in positions),
            "formal_hf_detection_results": formal_detector_results,
        }
        final = positions[-1]
        metric_payload = {
            "schema_version": METRIC_SCHEMA_VERSION,
            "metric_role": "development_exploratory_cluster_level",
            "responsibility_id": "hf_detector",
            "source_cluster_id": identity.source_cluster_id,
            "registered_metric_ids": (
                "wrong_key_attribution",
                "matched_budget_quality",
            ),
            "candidate_config_digest": self.candidate_config_digest,
            "paired_ablation_identity": "paired_clean_hf_same_generation",
            "content_branch_id": "hf_only",
            "geometry_case_id": "not_applicable",
            "sufficient_statistics": (
                ("registered_minus_primary_null", final.registered_minus_primary_null),
                ("registered_minus_wrong_key", final.registered_minus_wrong_key),
                ("realized_relative_l2", materialization.realized_relative_l2),
            ),
            "result_identity_digests": tuple(
                item.observation_identity for item in positions
            ),
            "threshold_role": "not_fitted_hf_transmission_diagnostic",
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
            "unit_index": self.protocol.operational_unit_count + cluster_ordinal,
            "phase": "development_scientific_breadth",
            "responsibility_id": "hf_detector",
            "analysis_unit_identity": asdict(identity),
            "scientific_question_id": "hf_signal_survival_across_generation_boundaries",
            "development_case_id": "paired_clean_hf_transport_observation",
            "candidate_identity": self.protocol.candidate_identity,
            "candidate_config_digest": self.candidate_config_digest,
            "paired_ablation_identity": "paired_clean_hf_same_generation",
            "negative_control_case_ids": (
                "same_image_wrong_key",
                "paired_clean_primary_null",
            ),
            "metric_ids": ("wrong_key_attribution", "matched_budget_quality"),
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
                item.position_id: {
                    "registered_score": item.registered_score,
                    "wrong_key_score": item.wrong_key_score,
                    "primary_null_score": item.primary_null_score,
                    "registered_minus_wrong_key": item.registered_minus_wrong_key,
                    "registered_minus_primary_null": item.registered_minus_primary_null,
                }
                for item in positions
            },
            "detector_trace": {
                "detector_identity": final.registered_statistic_identity,
                "same_image_registered_wrong_reuse": True,
                "paired_clean_primary_null": True,
            },
            "geometry_trace": {"geometry_attempted": False},
            "threshold_trace": {
                "threshold_role": "not_fitted_hf_transmission_diagnostic",
                "raw_threshold_identity": None,
                "rectified_threshold_identity": None,
            },
            "key_control_trace": {
                "root_key_public_digest": self.root_key_public_digest,
                "wrong_key_index": 0,
                "raw_secret_persisted": False,
            },
            "decision_trace": {
                "positive_source": None,
                "decision_role": "directional_transport_observation_only",
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
        record = DevelopmentScientificRecord(**record_payload)
        record = DevelopmentScientificRecord(
            **{
                **record_payload,
                "record_id": canonical_development_value_digest(
                    record.payload_without_record_id()
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
        retryable_resource_failure: bool,
    ) -> DevelopmentScientificRecord:
        """Persist an honest terminal or retryable failure after intent claim."""

        if not 0 <= cluster_ordinal < 8:
            raise HfTransmissionRunnerError("failed cluster is outside frozen manifest")
        if (
            type(failure_type) is not str
            or not failure_type
            or not isfinite(actual_elapsed_seconds)
            or actual_elapsed_seconds < 0.0
        ):
            raise HfTransmissionRunnerError("failed cluster facts are invalid")
        entry = self.manifest.entries[cluster_ordinal]
        identity = self._analysis_identity(entry)
        operation_payload = {
            "failure_stage": "hf_transmission_runtime_operation",
            "failure_type": failure_type,
            "result_available": False,
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
            "unit_index": self.protocol.operational_unit_count + cluster_ordinal,
            "phase": "development_scientific_breadth",
            "responsibility_id": "hf_detector",
            "analysis_unit_identity": asdict(identity),
            "scientific_question_id": "hf_signal_survival_across_generation_boundaries",
            "development_case_id": "paired_clean_hf_transport_observation",
            "candidate_identity": self.protocol.candidate_identity,
            "candidate_config_digest": self.candidate_config_digest,
            "paired_ablation_identity": "paired_clean_hf_same_generation",
            "negative_control_case_ids": (
                "same_image_wrong_key",
                "paired_clean_primary_null",
            ),
            "metric_ids": ("wrong_key_attribution", "matched_budget_quality"),
            "content_branch_id": "hf_only",
            "geometry_case_id": "not_applicable",
            "attempt_index": attempt_index,
            "execution_status": (
                "retry" if retryable_resource_failure else "failed"
            ),
            "failure_class": (
                "resource_failure"
                if resource_failure
                else "implementation_failure"
            ),
            "failure_reason": "hf_transmission_runtime_operation_failed",
            "retry_parent_intent_digest": retry_parent_intent_digest,
            "actual_elapsed_seconds": actual_elapsed_seconds,
            "maximum_duration_seconds": maximum_duration_seconds,
            "duration_limit_exceeded": (
                actual_elapsed_seconds > maximum_duration_seconds
            ),
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
                "threshold_role": "not_fitted_hf_transmission_diagnostic",
                "raw_threshold_identity": None,
                "rectified_threshold_identity": None,
            },
            "key_control_trace": {
                "root_key_public_digest": self.root_key_public_digest,
                "wrong_key_index": 0,
                "raw_secret_persisted": False,
            },
            "decision_trace": {
                "positive_source": None,
                "decision_role": "failed_transport_observation",
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

    def replay_directional_decision(
        self,
        verified_evidence: Sequence[
            tuple[DevelopmentScientificRecord, CommittedUnit]
        ],
    ) -> HfTransmissionDirectionalDecision:
        """Derive go/no-go only from eight verified COMMITTED terminal records."""

        evidence = tuple(verified_evidence)
        if len(evidence) != 8:
            raise HfTransmissionRunnerError("verified HF cluster coverage is incomplete")
        indexes = tuple(record.unit_index for record, _marker in evidence)
        if indexes != tuple(range(2, 10)) or len(set(indexes)) != 8:
            raise HfTransmissionRunnerError("verified HF unit indexes drifted")
        source_ids: set[str] = set()
        final_observations: list[HfSignalPositionObservation] = []
        failure_count = 0
        for record, marker in evidence:
            record.validate()
            if (
                marker.protocol_digest != self.protocol_digest
                or marker.run_id != self.run_id
                or marker.revision != self.method_code_revision
                or record.method_code_revision != self.method_code_revision
                or record.run_id != self.run_id
                or record.protocol_digest != self.protocol_digest
                or record.execution_intent_authority_digest
                != self.execution_intent_authority_digest
                or record.candidate_config_digest != self.candidate_config_digest
                or record.provenance_trace.get("manifest_digest")
                != self.manifest.digest()
                or record.responsibility_id != "hf_detector"
                or marker.record_digest
                != sha256(
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
            ):
                raise HfTransmissionRunnerError("verified HF evidence binding drifted")
            identity = AnalysisUnitIdentity(**record.analysis_unit_identity)
            if identity.source_cluster_id in source_ids:
                raise HfTransmissionRunnerError("verified HF source cluster duplicated")
            source_ids.add(identity.source_cluster_id)
            expected_entry = self.manifest.entries[record.unit_index - 2]
            if identity != self._analysis_identity(expected_entry):
                raise HfTransmissionRunnerError("verified HF source cluster is foreign")
            if record.execution_status != "success":
                failure_count += 1
                continue
            raw_positions = record.operation_result_payload.get("signal_positions")
            if type(raw_positions) not in {tuple, list} or len(raw_positions) != 4:
                raise HfTransmissionRunnerError("verified HF positions are incomplete")
            try:
                positions = tuple(
                    HfSignalPositionObservation(**item) for item in raw_positions
                )
            except (TypeError, ValueError) as exc:
                raise HfTransmissionRunnerError(
                    "verified HF position schema drifted"
                ) from exc
            for position in positions:
                position.validate()
            if tuple(item.position_id for item in positions) != self.protocol.signal_positions:
                raise HfTransmissionRunnerError("verified HF position order drifted")
            final_observations.append(positions[-1])
        return evaluate_hf_transmission_direction(
            final_observations,
            budget_integrity_nonfinite_failure_count=failure_count,
        )


__all__ = ["HfTransmissionDiagnosticRunner", "HfTransmissionRunnerError"]
