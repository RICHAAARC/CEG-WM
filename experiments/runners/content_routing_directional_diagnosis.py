"""Real public execution chain for the content-routing directional diagnosis."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from math import isfinite
from time import monotonic
from typing import Sequence

import torch

from experiments.methods import CegWmExperimentAdapter
from experiments.metrics.content_routing_directional_diagnosis import (
    ContentRoutingDirectionalObservation,
    ContentRoutingFoldReference,
    ContentRoutingReferenceMeasurement,
    ContentRoutingReferencePositiveSupportError,
    create_content_routing_blind_score_observation,
    create_content_routing_directional_observation,
    create_content_routing_reference_measurement,
    fit_content_routing_fold_reference,
)
from experiments.protocol.content_routing_directional_diagnosis import (
    CLAIM_BOUNDARY,
    OPERATIONAL_CASE_IDS,
    OPERATIONAL_ROLE,
    PUBLIC_CONTENT_OPERATION,
    ContentRoutingDirectionalProtocol,
    ContentRoutingManifest,
    ContentRoutingManifestEntry,
    canonical_digest,
)
from experiments.protocol.development_records import (
    DEVELOPMENT_CLAIM_BOUNDARY,
    DEVELOPMENT_RECORD_COLLECTION_ROLE,
    METRIC_SCHEMA_VERSION,
    OPERATIONAL_RECORD_COLLECTION_ROLE,
    OPERATIONAL_RECORD_KIND,
    OPERATIONAL_RECORD_SCHEMA,
    RECORD_SCHEMA_VERSION,
    ROUTING_REFERENCE_RECORD_COLLECTION_ROLE,
    ROUTING_REFERENCE_RECORD_KIND,
    ROUTING_REFERENCE_RECORD_SCHEMA,
    DevelopmentOperationalRecord,
    DevelopmentRoutingReferenceRecord,
    DevelopmentScientificRecord,
    canonical_development_value_digest,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    derive_source_cluster_id,
)
from experiments.runners.development_inputs import (
    DevelopmentSemanticObservationProducer,
)
from experiments.runners.development_persistence import (
    FrozenDevelopmentUnitBinding,
    UnitIntent,
    create_frozen_development_unit_binding,
)
from experiments.runners.formal_operations import (
    FormalHfContentDetectionOperation,
)
from main import (
    ContentDetectionResult,
    RoutingObservations,
    derive_wrong_key_material,
)
from runtime import Sd35RuntimeAdapter


class ContentRoutingDirectionalRunnerError(RuntimeError):
    """The routing diagnosis violated its frozen public execution contract."""


class ContentRoutingReferencePositiveSupportAbsentError(
    ContentRoutingDirectionalRunnerError
):
    """A frozen cross-fit fold lacks positive routing-reference support."""


def _rgb8(image: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(image, torch.Tensor)
        or image.ndim != 4
        or tuple(image.shape[:2]) != (1, 3)
        or min(image.shape[-2:]) <= 1
    ):
        raise ContentRoutingDirectionalRunnerError("routing public image is invalid")
    if image.dtype is torch.uint8:
        return image.detach().to(device="cpu").contiguous().clone()
    if not image.dtype.is_floating_point or not bool(torch.isfinite(image).all().item()):
        raise ContentRoutingDirectionalRunnerError("routing public image is invalid")
    return (
        torch.floor(
            image.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
            * 255.0
        )
        .to(torch.uint8)
        .contiguous()
    )


def _relative_l2(reference: torch.Tensor, observed: torch.Tensor) -> float:
    left = _rgb8(reference).to(torch.float64)
    right = _rgb8(observed).to(torch.float64)
    if left.shape != right.shape:
        raise ContentRoutingDirectionalRunnerError("routing image shapes drifted")
    denominator = float(torch.linalg.vector_norm(left).item())
    if denominator <= 0.0:
        raise ContentRoutingDirectionalRunnerError("routing clean image norm is zero")
    value = float(torch.linalg.vector_norm(right - left).item()) / denominator
    if not isfinite(value):
        raise ContentRoutingDirectionalRunnerError("routing image quality is nonfinite")
    return value


def _record_id(record: object) -> object:
    return replace(
        record,
        record_id=canonical_development_value_digest(
            record.payload_without_record_id()
        ),
    )


class ContentRoutingDirectionalDiagnosisRunner:
    """Execute reference fit and paired routed/uniform probes through public APIs."""

    def __init__(
        self,
        *,
        protocol: ContentRoutingDirectionalProtocol,
        reference_manifest: ContentRoutingManifest,
        probe_manifest: ContentRoutingManifest,
        adapter: CegWmExperimentAdapter,
        runtime_adapter: Sd35RuntimeAdapter,
        semantic_producer: DevelopmentSemanticObservationProducer,
        method_code_revision: str,
        registered_root_key: str,
        root_key_public_digest: str,
        protocol_digest: str,
        execution_intent_authority_digest: str,
        candidate_config_digest: str,
    ) -> None:
        protocol.validate()
        reference_manifest.validate(
            expected_role="content_routing_reference_fit",
            expected_count=32,
        )
        probe_manifest.validate(
            expected_role="content_routing_directional_probe",
            expected_count=8,
        )
        if type(adapter) is not CegWmExperimentAdapter:
            raise ContentRoutingDirectionalRunnerError("method adapter exact type required")
        if type(runtime_adapter) is not Sd35RuntimeAdapter:
            raise ContentRoutingDirectionalRunnerError("runtime adapter exact type required")
        if type(semantic_producer) is not DevelopmentSemanticObservationProducer:
            raise ContentRoutingDirectionalRunnerError("semantic producer exact type required")
        if type(method_code_revision) is not str or len(method_code_revision) != 40:
            raise ContentRoutingDirectionalRunnerError("method revision is invalid")
        self.protocol = protocol
        self.reference_manifest = reference_manifest
        self.probe_manifest = probe_manifest
        self.adapter = adapter
        self.runtime = runtime_adapter
        self.semantic = semantic_producer
        self.method_code_revision = method_code_revision
        self.registered_root_key = registered_root_key
        self.root_key_public_digest = root_key_public_digest
        self.protocol_digest = protocol_digest
        self.execution_intent_authority_digest = execution_intent_authority_digest
        self.candidate_config_digest = candidate_config_digest

    def _manifest_entry(self, unit_index: int) -> tuple[ContentRoutingManifestEntry, str]:
        if unit_index < 2:
            return self.reference_manifest.entries[unit_index], "development_environment_preflight"
        if unit_index < 34:
            return self.reference_manifest.entries[unit_index - 2], self.reference_manifest.role_id
        return self.probe_manifest.entries[unit_index - 34], self.probe_manifest.role_id

    def _analysis_identity(self, unit_index: int) -> AnalysisUnitIdentity:
        unit = self.protocol.unit_roster[unit_index]
        entry, role = self._manifest_entry(unit_index)
        image_lineage = entry.image_lineage_digest(role_id=role)
        key_family = canonical_digest(
            {
                "key_family_namespace": (
                    self.reference_manifest.key_family_namespace
                    if unit_index < 34
                    else self.probe_manifest.key_family_namespace
                ),
                "root_key_public_digest": self.root_key_public_digest,
            }
        )
        source_cluster = derive_source_cluster_id(
            prompt_digest=entry.prompt_digest,
            generation_seed=entry.generation_seed,
            image_lineage_digest=image_lineage,
            registered_key_family_digest=key_family,
        )
        return AnalysisUnitIdentity(
            unit_id=f"development_unit_{unit.unit_index:04d}",
            case_id=unit.phase,
            source_cluster_id=source_cluster,
            prompt_digest=entry.prompt_digest,
            generation_seed=entry.generation_seed,
            image_lineage_digest=image_lineage,
            registered_key_family_digest=key_family,
        )

    def create_persistence_unit_bindings(self) -> tuple[FrozenDevelopmentUnitBinding, ...]:
        return tuple(
            create_frozen_development_unit_binding(
                unit,
                analysis_unit_identity=self._analysis_identity(unit.unit_index),
                scientific_question_id="content_routing_directional_increment",
                development_case_id=(
                    "content_embedder_operational_preflight"
                    if unit.unit_index < 2
                    else "content_routing_reference_fit"
                    if unit.unit_index < 34
                    else "paired_routed_uniform_directional_probe"
                ),
                candidate_identity=self.protocol.routing_candidate_identity,
                candidate_config_digest=self.candidate_config_digest,
            )
            for unit in self.protocol.unit_roster
        )

    def execute_operational_unit(
        self,
        *,
        unit_index: int,
        base_latent: torch.Tensor,
        intent: UnitIntent,
    ) -> DevelopmentOperationalRecord:
        if unit_index not in {0, 1} or intent.unit_index != unit_index:
            raise ContentRoutingDirectionalRunnerError("operational unit identity drifted")
        started = monotonic()
        shape = tuple(int(value) for value in base_latent.shape)
        route = self.adapter.route_content(shape, mode="routing_uniform_control").result
        low = self.adapter.build_lf_carrier(
            self.registered_root_key, shape, routing_result=route
        ).result
        high = self.adapter.build_hf_carrier(
            self.registered_root_key, shape, routing_result=route
        ).result
        captured: list[object] = []

        def embed(values: tuple[float, ...]):
            result = self.adapter.embed_content(
                values,
                high,
                lf_carrier_result=low,
                mixing_coefficient=self.protocol.mixing_coefficient,
                routing_result=route,
            ).result
            captured.append(result)
            return result

        runtime_result = self.runtime.execute_content_write_and_vae(base_latent, embed)
        elapsed = float(monotonic() - started)
        if len(captured) != 1 or elapsed > intent.maximum_duration_seconds:
            raise ContentRoutingDirectionalRunnerError("operational content embedder failed")
        result_digest = canonical_development_value_digest(
            {
                "embedding_result_identity": captured[0].embedding_result_identity,
                "materialization_replay_identity": (
                    runtime_result.content_materialization.materialization_replay_identity
                ),
                "route_identity": route.route_identity,
            }
        )
        payload = {
            "operational_role": OPERATIONAL_ROLE,
            "source_cluster_ordinal": unit_index,
            "case_ids": list(OPERATIONAL_CASE_IDS),
            "responsibility_result_digests": [["content_embedder", result_digest]],
            "elapsed_seconds": elapsed,
            "runtime_config_digest": runtime_result.runtime_config_digest,
            "counts_as_scientific_coverage": False,
            "scientific_claims_supported": False,
        }
        record = DevelopmentOperationalRecord(
            schema_version=OPERATIONAL_RECORD_SCHEMA,
            collection_role=OPERATIONAL_RECORD_COLLECTION_ROLE,
            record_kind=OPERATIONAL_RECORD_KIND,
            record_id="0" * 64,
            run_id=self.protocol.run_id,
            protocol_digest=self.protocol_digest,
            method_code_revision=self.method_code_revision,
            unit_index=unit_index,
            phase="development_environment_preflight",
            source_cluster_ordinal=unit_index,
            candidate_config_digest=self.candidate_config_digest,
            attempt_index=intent.attempt_index,
            retry_parent_intent_digest=intent.parent_attempt_intent_digest,
            actual_elapsed_seconds=elapsed,
            maximum_duration_seconds=intent.maximum_duration_seconds,
            operation_result_payload=payload,
            counts_as_scientific_coverage=False,
            scientific_claims_supported=False,
            scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
        )
        record = _record_id(record)
        record.validate()
        return record

    def execute_reference_fit_unit(
        self,
        *,
        unit_index: int,
        base_latent: torch.Tensor,
        intent: UnitIntent,
    ) -> DevelopmentRoutingReferenceRecord:
        if not 2 <= unit_index < 34 or intent.unit_index != unit_index:
            raise ContentRoutingDirectionalRunnerError("reference unit identity drifted")
        ordinal = unit_index - 2
        started = monotonic()
        measurement = self.runtime.measure_generation_routing_reference_inputs(
            base_latent,
            sample_index=ordinal,
        )
        create_content_routing_reference_measurement(
            cluster_ordinal=ordinal,
            fold_index=ordinal % 4,
            texture_gradient_values=tuple(measurement.texture_gradient_values),
            texture_spatial_shape=tuple(measurement.texture_spatial_shape),
            response_ratio_values=tuple(measurement.response_ratio_values),
            response_spatial_shape=tuple(measurement.response_spatial_shape),
            sensitivity_ratio_values=tuple(measurement.sensitivity_ratio_values),
            sensitivity_spatial_shape=tuple(measurement.sensitivity_spatial_shape),
        )
        elapsed = float(monotonic() - started)
        if elapsed > intent.maximum_duration_seconds:
            raise ContentRoutingDirectionalRunnerError("reference unit exceeded duration")
        payload = {
            "candidate_id": measurement.candidate_id,
            "runtime_config_digest": measurement.runtime_config_digest,
            "model_id": measurement.model_id,
            "model_revision": measurement.model_revision,
            "callback_indices": list(measurement.callback_indices),
            "public_probe_domain_digest": measurement.public_probe_domain_digest,
            "public_probe_values_digest": measurement.public_probe_values_float32_be_sha256,
            "nominal_relative_probe_step": measurement.nominal_relative_probe_step,
            "actual_probe_step": measurement.actual_probe_step,
            "texture_gradient_values": list(measurement.texture_gradient_values),
            "texture_spatial_shape": list(measurement.texture_spatial_shape),
            "response_ratio_values": list(measurement.response_ratio_values),
            "response_spatial_shape": list(measurement.response_spatial_shape),
            "sensitivity_ratio_values": list(measurement.sensitivity_ratio_values),
            "sensitivity_spatial_shape": list(measurement.sensitivity_spatial_shape),
        }
        record = DevelopmentRoutingReferenceRecord(
            schema_version=ROUTING_REFERENCE_RECORD_SCHEMA,
            collection_role=ROUTING_REFERENCE_RECORD_COLLECTION_ROLE,
            record_kind=ROUTING_REFERENCE_RECORD_KIND,
            record_id="0" * 64,
            run_id=self.protocol.run_id,
            protocol_digest=self.protocol_digest,
            method_code_revision=self.method_code_revision,
            unit_index=unit_index,
            phase=ROUTING_REFERENCE_RECORD_KIND,
            source_cluster_ordinal=ordinal,
            fold_index=ordinal % 4,
            prompt_roster_digest=canonical_digest(asdict(self.reference_manifest)),
            candidate_config_digest=self.candidate_config_digest,
            attempt_index=intent.attempt_index,
            retry_parent_intent_digest=intent.parent_attempt_intent_digest,
            actual_elapsed_seconds=elapsed,
            maximum_duration_seconds=intent.maximum_duration_seconds,
            duration_limit_exceeded=False,
            execution_status="success",
            failure_class=None,
            failure_reason=None,
            measurement_payload=payload,
            counts_as_scientific_coverage=False,
            scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
        )
        record = _record_id(record)
        record.validate()
        return record

    def create_failed_reference_record(
        self,
        *,
        intent: UnitIntent,
        failure_class: str,
        failure_reason: str,
        elapsed_seconds: float,
    ) -> DevelopmentRoutingReferenceRecord:
        if (
            not 2 <= intent.unit_index < 34
            or failure_class not in {"implementation_failure", "resource_failure"}
        ):
            raise ContentRoutingDirectionalRunnerError("reference failure identity drifted")
        ordinal = intent.unit_index - 2
        record = DevelopmentRoutingReferenceRecord(
            schema_version=ROUTING_REFERENCE_RECORD_SCHEMA,
            collection_role=ROUTING_REFERENCE_RECORD_COLLECTION_ROLE,
            record_kind=ROUTING_REFERENCE_RECORD_KIND,
            record_id="0" * 64,
            run_id=self.protocol.run_id,
            protocol_digest=self.protocol_digest,
            method_code_revision=self.method_code_revision,
            unit_index=intent.unit_index,
            phase=ROUTING_REFERENCE_RECORD_KIND,
            source_cluster_ordinal=ordinal,
            fold_index=ordinal % 4,
            prompt_roster_digest=canonical_digest(asdict(self.reference_manifest)),
            candidate_config_digest=self.candidate_config_digest,
            attempt_index=intent.attempt_index,
            retry_parent_intent_digest=intent.parent_attempt_intent_digest,
            actual_elapsed_seconds=elapsed_seconds,
            maximum_duration_seconds=intent.maximum_duration_seconds,
            duration_limit_exceeded=elapsed_seconds > intent.maximum_duration_seconds,
            execution_status="failed",
            failure_class=failure_class,
            failure_reason=failure_reason,
            measurement_payload={},
            counts_as_scientific_coverage=False,
            scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
        )
        record = _record_id(record)
        record.validate()
        return record

    @staticmethod
    def reference_measurement_from_committed_record(
        record: DevelopmentRoutingReferenceRecord,
    ) -> ContentRoutingReferenceMeasurement:
        checked = DevelopmentRoutingReferenceRecord.from_payload(record.payload())
        if checked.execution_status != "success":
            raise ContentRoutingDirectionalRunnerError("reference record is not successful")
        payload = checked.measurement_payload
        return create_content_routing_reference_measurement(
            cluster_ordinal=checked.source_cluster_ordinal,
            fold_index=checked.fold_index,
            texture_gradient_values=tuple(payload["texture_gradient_values"]),
            texture_spatial_shape=tuple(payload["texture_spatial_shape"]),
            response_ratio_values=tuple(payload["response_ratio_values"]),
            response_spatial_shape=tuple(payload["response_spatial_shape"]),
            sensitivity_ratio_values=tuple(payload["sensitivity_ratio_values"]),
            sensitivity_spatial_shape=tuple(payload["sensitivity_spatial_shape"]),
        )

    @classmethod
    def validate_reference_positive_support(
        cls,
        records: Sequence[DevelopmentRoutingReferenceRecord],
    ) -> tuple[ContentRoutingFoldReference, ...]:
        """Budget all four frozen cross-fit references before any probe call."""

        measurements = tuple(
            cls.reference_measurement_from_committed_record(record)
            for record in records
        )
        try:
            references = tuple(
                fit_content_routing_fold_reference(
                    measurements,
                    probe_fold_index=fold_index,
                )
                for fold_index in range(4)
            )
        except ContentRoutingReferencePositiveSupportError as exc:
            raise ContentRoutingReferencePositiveSupportAbsentError(
                "routing_reference_positive_support_absent"
            ) from exc
        return references

    def _routing_observations(
        self,
        *,
        ordinal: int,
        prompt: str,
        base_latent: torch.Tensor,
        reference_records: Sequence[DevelopmentRoutingReferenceRecord],
    ) -> tuple[RoutingObservations, str]:
        measurements = tuple(
            self.reference_measurement_from_committed_record(record)
            for record in reference_records
        )
        reference = fit_content_routing_fold_reference(
            measurements,
            probe_fold_index=ordinal % 4,
        )
        raw = self.runtime.measure_generation_routing_reference_inputs(
            base_latent,
            sample_index=ordinal,
        )
        normalized = self.runtime.normalize_generation_routing_measurement(
            raw,
            reference_gradient=reference.texture_gradient_reference,
            reference_response=reference.latent_response_reference,
            reference_sensitivity=reference.local_sensitivity_reference,
        )
        semantic = self.semantic.observe(normalized.routing_rgb, prompt)
        return (
            RoutingObservations(
                semantic=semantic,
                texture=normalized.texture,
                response=normalized.response,
                sensitivity=normalized.sensitivity,
            ),
            reference.reference_identity,
        )

    def _score_rows(
        self,
        *,
        arm_id: str,
        candidate_image: torch.Tensor,
        clean_image: torch.Tensor,
        operation: FormalHfContentDetectionOperation,
    ) -> tuple[object, ...]:
        registered = operation(candidate_image, self.registered_root_key)
        primary_null = operation(clean_image, self.registered_root_key)
        wrong = tuple(
            operation(
                candidate_image,
                derive_wrong_key_material(self.root_key_public_digest, index),
            )
            for index in range(4)
        )
        results: tuple[tuple[str, int | None, str, ContentDetectionResult], ...] = (
            ("registered", None, "registered", registered),
            ("paired_clean_primary_null", None, "registered", primary_null),
            *(("wrong_key_control", index, "wrong", item) for index, item in enumerate(wrong)),
        )
        rows = []
        for control_role, wrong_index, key_role, result in results:
            if (
                type(result) is not ContentDetectionResult
                or result.formal_mode != "hf_only"
                or result.content_score != result.hf_score
                or any(
                    getattr(result, name) is not None
                    for name in self.protocol.public_score_required_null_result_fields
                )
            ):
                raise ContentRoutingDirectionalRunnerError("formal HF-only result drifted")
            hf = result.hf_result
            rows.append(
                create_content_routing_blind_score_observation(
                    arm_id=arm_id,
                    control_role=control_role,
                    wrong_key_index=wrong_index,
                    content_score=result.content_score,
                    hf_score=hf.hf_score,
                    formal_mode=result.formal_mode,
                    content_detector_identity=result.detector_identity,
                    content_config_digest=result.content_config_digest,
                    hf_detector_identity=hf.detector_identity,
                    hf_detector_config_digest=hf.detector_config_digest,
                    content_input_image_digest=result.content_input_image_digest,
                    hf_observation_digest=hf.observation_digest,
                    hf_template_digest=hf.template_digest,
                    root_key_public_digest=hf.root_key_public_digest,
                    key_role=key_role,
                )
            )
        return tuple(rows)

    def execute_probe_unit(
        self,
        *,
        unit_index: int,
        base_latent: torch.Tensor,
        intent: UnitIntent,
        reference_records: Sequence[DevelopmentRoutingReferenceRecord],
    ) -> DevelopmentScientificRecord:
        if not 34 <= unit_index < 42 or intent.unit_index != unit_index:
            raise ContentRoutingDirectionalRunnerError("probe unit identity drifted")
        ordinal = unit_index - 34
        entry = self.probe_manifest.entries[ordinal]
        started = monotonic()
        observations, reference_digest = self._routing_observations(
            ordinal=ordinal,
            prompt=entry.prompt,
            base_latent=base_latent,
            reference_records=reference_records,
        )
        shape = tuple(int(value) for value in base_latent.shape)
        routed = self.adapter.route_content(
            shape, mode="routing_stqr", observations=observations
        ).result
        uniform = self.adapter.route_content(shape, mode="routing_uniform_control").result

        def execute_arm(route):
            low = self.adapter.build_lf_carrier(
                self.registered_root_key, shape, routing_result=route
            ).result
            high = self.adapter.build_hf_carrier(
                self.registered_root_key, shape, routing_result=route
            ).result
            captured: list[object] = []

            def embed(values: tuple[float, ...]):
                result = self.adapter.embed_content(
                    values,
                    high,
                    lf_carrier_result=low,
                    mixing_coefficient=self.protocol.mixing_coefficient,
                    routing_result=route,
                ).result
                captured.append(result)
                return result

            runtime_result = self.runtime.execute_content_write_and_vae(base_latent, embed)
            if len(captured) != 1:
                raise ContentRoutingDirectionalRunnerError("embedder call count drifted")
            return runtime_result

        routed_runtime = execute_arm(routed)
        uniform_runtime = execute_arm(uniform)
        routed_clean = _rgb8(routed_runtime.clean_image)
        uniform_clean = _rgb8(uniform_runtime.clean_image)
        if not torch.equal(routed_clean, uniform_clean):
            raise ContentRoutingDirectionalRunnerError("paired clean controls drifted")
        routed_image = _rgb8(routed_runtime.watermarked_image)
        uniform_image = _rgb8(uniform_runtime.watermarked_image)
        operation = FormalHfContentDetectionOperation(self.adapter)
        rows = (
            *self._score_rows(
                arm_id="routed",
                candidate_image=routed_image,
                clean_image=routed_clean,
                operation=operation,
            ),
            *self._score_rows(
                arm_id="uniform",
                candidate_image=uniform_image,
                clean_image=routed_clean,
                operation=operation,
            ),
        )
        routed_materialization = routed_runtime.content_materialization
        uniform_materialization = uniform_runtime.content_materialization
        routed_result = routed_runtime.content_materialization_result
        uniform_result = uniform_runtime.content_materialization_result
        elapsed = float(monotonic() - started)
        if elapsed > intent.maximum_duration_seconds:
            raise ContentRoutingDirectionalRunnerError("probe unit exceeded duration")
        observation = create_content_routing_directional_observation(
            cluster_ordinal=ordinal,
            fold_index=ordinal % 4,
            blind_score_observations=rows,
            routed_mean_mask_lf=routed.mean_mask_lf,
            routed_mean_mask_hf=routed.mean_mask_hf,
            uniform_mean_mask_lf=uniform.mean_mask_lf,
            uniform_mean_mask_hf=uniform.mean_mask_hf,
            routed_clean_to_watermarked_rgb_relative_l2=_relative_l2(
                routed_clean, routed_image
            ),
            uniform_clean_to_watermarked_rgb_relative_l2=_relative_l2(
                routed_clean, uniform_image
            ),
            routed_realized_relative_l2=routed_materialization.realized_relative_l2,
            uniform_realized_relative_l2=uniform_materialization.realized_relative_l2,
            routed_materialization_integrity_status=routed_materialization.integrity_status,
            uniform_materialization_integrity_status=uniform_materialization.integrity_status,
            routed_materialization_budget_status=routed_result.budget_status,
            uniform_materialization_budget_status=uniform_result.budget_status,
            public_content_operation=PUBLIC_CONTENT_OPERATION,
            preprocessing_identity=operation.preprocessing_identity,
            routed_route_digest=routed.route_identity,
            uniform_route_digest=uniform.route_identity,
            cross_fit_reference_digest=reference_digest,
            routed_candidate_observation_digest=rows[0].hf_observation_digest,
            uniform_candidate_observation_digest=rows[6].hf_observation_digest,
            paired_clean_observation_digest=rows[1].hf_observation_digest,
            failure_class=None,
        )
        identity = self._analysis_identity(unit_index)
        operation_payload = {
            "routing_observation": asdict(observation),
            "routed_route_identity": routed.route_identity,
            "uniform_route_identity": uniform.route_identity,
            "routed_embedding_result_identity": (
                routed_result.embedding_result.embedding_result_identity
            ),
            "uniform_embedding_result_identity": (
                uniform_result.embedding_result.embedding_result_identity
            ),
        }
        metric = {
            "schema_version": METRIC_SCHEMA_VERSION,
            "metric_role": "development_exploratory_cluster_level",
            "responsibility_id": "content_router",
            "source_cluster_id": identity.source_cluster_id,
            "registered_metric_ids": (
                "routing_incremental_indicator",
                "routing_coverage",
                "matched_budget_quality",
            ),
            "candidate_config_digest": self.candidate_config_digest,
            "paired_ablation_identity": "paired_routed_uniform_same_generation",
            "content_branch_id": "paired_routed_uniform_content_embedding",
            "geometry_case_id": "geometry_case_not_applicable",
            "sufficient_statistics": (
                ("incremental_indicator", observation.incremental_indicator),
                ("routing_coverage", observation.routing_coverage),
                ("quality_relative_l2", observation.quality_relative_l2),
            ),
            "result_identity_digests": (observation.observation_identity,),
            "threshold_role": "not_fitted_routing_directional_diagnosis",
            "threshold_identity": None,
            "threshold_fit_source_cluster_digest": None,
        }
        metric["observation_digest"] = canonical_development_value_digest(metric)
        detector_identity = rows[0].content_detector_identity
        detector_config = rows[0].content_config_digest
        record = DevelopmentScientificRecord(
            schema_version=RECORD_SCHEMA_VERSION,
            collection_role=DEVELOPMENT_RECORD_COLLECTION_ROLE,
            record_id="0" * 64,
            run_id=self.protocol.run_id,
            protocol_id=self.protocol.protocol_id,
            protocol_version=self.protocol.protocol_version,
            protocol_digest=self.protocol_digest,
            execution_intent_authority_digest=self.execution_intent_authority_digest,
            method_code_revision=self.method_code_revision,
            unit_index=unit_index,
            phase="development_content_routing_directional_probe",
            analysis_unit_identity=asdict(identity),
            responsibility_id="content_router",
            scientific_question_id="content_routing_directional_increment",
            development_case_id="paired_routed_uniform_directional_probe",
            candidate_identity=self.protocol.routing_candidate_identity,
            candidate_config_digest=self.candidate_config_digest,
            paired_ablation_identity="paired_routed_uniform_same_generation",
            negative_control_case_ids=("paired_clean_primary_null", "wrong_key_control"),
            metric_ids=metric["registered_metric_ids"],
            content_branch_id="paired_routed_uniform_content_embedding",
            geometry_case_id="geometry_case_not_applicable",
            attempt_index=intent.attempt_index,
            execution_status="success",
            failure_class=None,
            failure_reason=None,
            retry_parent_intent_digest=intent.parent_attempt_intent_digest,
            actual_elapsed_seconds=elapsed,
            maximum_duration_seconds=intent.maximum_duration_seconds,
            duration_limit_exceeded=False,
            operation_result_payload=operation_payload,
            operation_result_digest=canonical_development_value_digest(operation_payload),
            metric_observation=metric,
            routing_trace={
                "routed_route_identity": routed.route_identity,
                "uniform_route_identity": uniform.route_identity,
                "cross_fit_reference_digest": reference_digest,
            },
            branch_score_trace={"blind_score_observations": [asdict(row) for row in rows]},
            detector_trace={
                "raw_detector_identity": detector_identity,
                "rectified_detector_identity": detector_identity,
                "raw_detector_config_digest": detector_config,
                "rectified_detector_config_digest": detector_config,
                "raw_preprocessing_identity": operation.preprocessing_identity,
                "rectified_preprocessing_identity": operation.preprocessing_identity,
            },
            geometry_trace={"geometry_case_id": "geometry_case_not_applicable"},
            threshold_trace={"raw_threshold_identity": None, "rectified_threshold_identity": None},
            key_control_trace={
                "root_key_public_digest": self.root_key_public_digest,
                "wrong_key_count": 4,
            },
            decision_trace={
                "positive_source": "raw_content",
                "incremental_indicator": observation.incremental_indicator,
                "formal_scientific_claims_supported": False,
            },
            provenance_trace={
                "protocol_digest": self.protocol_digest,
                "execution_intent_authority_digest": self.execution_intent_authority_digest,
                "method_code_revision": self.method_code_revision,
                "candidate_config_digest": self.candidate_config_digest,
            },
            module_outcome=None,
            candidate_recommendation=None,
            scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
        )
        record = _record_id(record)
        record.validate()
        return record

    def create_failed_probe_record(
        self,
        *,
        intent: UnitIntent,
        failure_class: str,
        failure_reason: str,
        elapsed_seconds: float,
    ) -> DevelopmentScientificRecord:
        if (
            not 34 <= intent.unit_index < 42
            or failure_class not in {"implementation_failure", "resource_failure"}
        ):
            raise ContentRoutingDirectionalRunnerError("probe failure identity drifted")
        identity = self._analysis_identity(intent.unit_index)
        empty_payload: dict[str, object] = {}
        record = DevelopmentScientificRecord(
            schema_version=RECORD_SCHEMA_VERSION,
            collection_role=DEVELOPMENT_RECORD_COLLECTION_ROLE,
            record_id="0" * 64,
            run_id=self.protocol.run_id,
            protocol_id=self.protocol.protocol_id,
            protocol_version=self.protocol.protocol_version,
            protocol_digest=self.protocol_digest,
            execution_intent_authority_digest=self.execution_intent_authority_digest,
            method_code_revision=self.method_code_revision,
            unit_index=intent.unit_index,
            phase="development_content_routing_directional_probe",
            analysis_unit_identity=asdict(identity),
            responsibility_id="content_router",
            scientific_question_id="content_routing_directional_increment",
            development_case_id="paired_routed_uniform_directional_probe",
            candidate_identity=self.protocol.routing_candidate_identity,
            candidate_config_digest=self.candidate_config_digest,
            paired_ablation_identity="paired_routed_uniform_same_generation",
            negative_control_case_ids=("paired_clean_primary_null", "wrong_key_control"),
            metric_ids=(
                "routing_incremental_indicator",
                "routing_coverage",
                "matched_budget_quality",
            ),
            content_branch_id="paired_routed_uniform_content_embedding",
            geometry_case_id="geometry_case_not_applicable",
            attempt_index=intent.attempt_index,
            execution_status="failed",
            failure_class=failure_class,
            failure_reason=failure_reason,
            retry_parent_intent_digest=intent.parent_attempt_intent_digest,
            actual_elapsed_seconds=elapsed_seconds,
            maximum_duration_seconds=intent.maximum_duration_seconds,
            duration_limit_exceeded=elapsed_seconds > intent.maximum_duration_seconds,
            operation_result_payload=empty_payload,
            operation_result_digest=canonical_development_value_digest(empty_payload),
            metric_observation={},
            routing_trace={},
            branch_score_trace={},
            detector_trace={},
            geometry_trace={},
            threshold_trace={},
            key_control_trace={},
            decision_trace={},
            provenance_trace={
                "protocol_digest": self.protocol_digest,
                "execution_intent_authority_digest": self.execution_intent_authority_digest,
                "method_code_revision": self.method_code_revision,
                "candidate_config_digest": self.candidate_config_digest,
            },
            module_outcome=None,
            candidate_recommendation=None,
            scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
        )
        record = _record_id(record)
        record.validate()
        return record


__all__ = [
    "ContentRoutingDirectionalDiagnosisRunner",
    "ContentRoutingDirectionalRunnerError",
]
