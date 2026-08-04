"""Build real per-unit inputs from runtime measurements and COMMITTED replay."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import ceil, isfinite
from pathlib import Path
import time
from typing import Literal

import torch

from experiments.attacks import AttackArtifact, GeometricAttackSpec
from experiments.metrics import load_metric_registry
from experiments.protocol.development_exploration import (
    DevelopmentProvisionalThreshold,
    DevelopmentStudyUnit,
    FrozenDevelopmentExecutionIntentAuthority,
    FrozenDevelopmentExplorationProtocol,
    build_development_cross_fit_plan,
)
from experiments.protocol.development_records import (
    DEVELOPMENT_CLAIM_BOUNDARY,
    ROUTING_REFERENCE_RECORD_COLLECTION_ROLE,
    ROUTING_REFERENCE_RECORD_KIND,
    ROUTING_REFERENCE_RECORD_SCHEMA,
    DevelopmentRoutingReferenceRecord,
    canonical_development_value_digest,
)
from experiments.protocol.internal_case import (
    INPUT_MANIFEST_SCHEMA_VERSION,
    FrozenCaseExecutionExpectation,
    FrozenCaseInputManifest,
    InternalCaseManifestEntry,
)
from experiments.protocol.internal_records import KeyControlTrace, RoutingTrace
from experiments.protocol.internal_validation import (
    load_frozen_internal_validation_protocol,
)
from experiments.runners import (
    DEVELOPMENT_ONLY_RECORD_SCOPE,
    FormalHfContentDetectionOperation,
    FormalRuntimeGeometryEstimationOperation,
    FrozenRecordBindings,
    GovernedRecordWriter,
    InternalCaseExecutionPayload,
    InternalRunnerContext,
    candidate_config_digest,
    create_formal_content_detector_binding,
    execution_config_digest,
    formal_operation_config_digest,
    geometry_reliability_config_digest,
)
from experiments.runners.development_exploration import (
    DevelopmentExplorationRunner,
    DevelopmentUnitInput,
)
from experiments.runners.development_inputs import (
    DevelopmentInputError,
    DevelopmentSemanticObservationProducer,
    FrozenDevelopmentPromptRoster,
    exact_positive_nearest_rank_p95,
    replay_branch_null_calibration,
)
from experiments.runners.development_persistence import (
    DevelopmentPersistentStore,
    DevelopmentSessionCursor,
    FrozenDevelopmentUnitBinding,
    PersistentLease,
    UnitIntent,
)
from main import (
    BranchNullCalibration,
    GeometryReliabilityThresholds,
    HfDetectionObservation,
    JointDecisionThresholds,
    LfDetectionObservation,
    RoutingObservations,
)
from main.content_chain.detector import NullScoreRecord
from runtime import (
    RuntimeAdapterError,
    RuntimeContentExecutionError,
    Sd35RuntimeAdapter,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INTERNAL_PROTOCOL_PATH = REPOSITORY_ROOT / "configs/experiments/internal_scientific_validation_protocol.json"
COMPONENT_REGISTRY_PATH = REPOSITORY_ROOT / "configs/experiments/internal_execution_components.json"

RoutingReferencePreparationStatus = Literal[
    "complete_success",
    "terminal_blocked",
    "retryable_stop",
    "soft_stop",
]
ROUTING_REFERENCE_SCHEDULER_READY = frozenset(
    {"complete_success", "terminal_blocked"}
)
ROUTING_REFERENCE_SESSION_STOP = frozenset(
    {"retryable_stop", "soft_stop"}
)


class DevelopmentDependencyInputBlocked(DevelopmentInputError):
    """Verified upstream records cannot yet authorize this frozen unit."""


def _canonical_digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _combination(cluster_ordinal: int) -> tuple[str, float]:
    cycle = cluster_ordinal % 5
    if cycle == 0:
        return "hf_only_standardized_score", 0.5
    if cycle in {1, 2, 3}:
        return "weighted_hf_lf_standardized_score", (0.25, 0.50, 0.75)[cycle - 1]
    return "maximum_hf_lf_standardized_score", 0.5


def _lower_nearest_rank(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise DevelopmentInputError("development geometry fit evidence is empty")
    return ordered[max(0, ceil(quantile * len(ordered)) - 1)]


def _development_rescue_threshold(provisional_threshold) -> tuple[float, str]:
    if type(provisional_threshold) is not DevelopmentProvisionalThreshold:
        raise DevelopmentInputError(
            "development rescue fit requires a verified provisional threshold"
        )
    tau = float(provisional_threshold.threshold)
    margins = sorted(
        tau - float(item.primary_null_score)
        for item in provisional_threshold.fit_inputs
        if float(item.primary_null_score) < tau
        and tau - float(item.primary_null_score) > 0.0
    )
    if not margins:
        raise DevelopmentInputError(
            "development rescue fit has no strictly positive fit-fold margin"
        )
    margin = margins[max(0, ceil(0.05 * len(margins)) - 1)]
    tau_rescue = tau - margin
    if not isfinite(tau_rescue) or not tau_rescue < tau:
        raise DevelopmentInputError("development rescue threshold fit is invalid")
    identity = _canonical_digest(
        {
            "fit_role": "development_exploratory_rescue_threshold_cross_fit",
            "fit_fold_index": provisional_threshold.fold_index,
            "fit_source_cluster_digest": provisional_threshold.fit_source_cluster_digest,
            "input_manifest_digest": provisional_threshold.input_manifest_digest,
            "invalid_for_splits": provisional_threshold.invalid_for_splits,
            "margin_quantile": "strictly_positive_exact_nearest_rank_p05",
            "primary_null_source_record_digests": tuple(
                item.source_record_digest for item in provisional_threshold.fit_inputs
            ),
            "provisional_threshold_identity": provisional_threshold.threshold_identity,
            "tau": tau,
            "tau_rescue": tau_rescue,
        }
    )
    return tau_rescue, identity


def _decoded_public_rgb8(image: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(image, torch.Tensor)
        or image.ndim != 4
        or tuple(image.shape[:2]) != (1, 3)
        or min(image.shape[-2:]) <= 1
        or not image.dtype.is_floating_point
        or not bool(torch.isfinite(image).all().item())
    ):
        raise DevelopmentInputError(
            "regenerated public image must be finite float RGB"
        )
    return torch.floor(
        image.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
        * 255.0
    ).to(torch.uint8)


class DevelopmentProductionInputBuilder:
    def __init__(
        self,
        *,
        cache_root: Path,
        prompt_roster: FrozenDevelopmentPromptRoster,
        protocol: FrozenDevelopmentExplorationProtocol,
        authority: FrozenDevelopmentExecutionIntentAuthority,
        registered_root_key: str,
        runtime_adapter: Sd35RuntimeAdapter,
        persistence_store: DevelopmentPersistentStore,
        session_cursor: DevelopmentSessionCursor,
        runner: DevelopmentExplorationRunner,
        hf_token: str,
    ) -> None:
        self.prompts = prompt_roster
        self.protocol = protocol
        self.authority = authority
        self.root_key = registered_root_key
        self.hf_token = hf_token
        self.runtime = runtime_adapter
        self.store = persistence_store
        self.session_cursor = session_cursor
        self.runner = runner
        self.cache_root = cache_root
        self._routing_observations_by_cluster: dict[int, RoutingObservations] = {}
        self.internal_protocol = load_frozen_internal_validation_protocol(
            INTERNAL_PROTOCOL_PATH
        )
        self.metric_registry = load_metric_registry(COMPONENT_REGISTRY_PATH)
        self.semantic = DevelopmentSemanticObservationProducer(
            cache_root=cache_root / "clip",
            hf_token=hf_token,
            device="cuda:0",
        )

    def _internal_record_scratch_root(
        self,
        *,
        unit_descriptor_digest: str,
        intent: UnitIntent,
    ) -> Path:
        """Keep the conditional runner's atomic scratch off the persistent worker root."""

        return (
            self.cache_root
            / "development_internal_record_scratch"
            / unit_descriptor_digest
            / intent.digest()
        )

    def _operational_routing_observations(
        self,
        base_latent: torch.Tensor,
        *,
        source_cluster_ordinal: int,
    ) -> RoutingObservations:
        measurement = self.runtime.measure_generation_routing_reference_inputs(
            base_latent,
            sample_index=source_cluster_ordinal,
        )
        normalized = self.runtime.normalize_generation_routing_measurement(
            measurement,
            reference_gradient=exact_positive_nearest_rank_p95(
                measurement.texture_gradient_values
            ),
            reference_response=exact_positive_nearest_rank_p95(
                measurement.response_ratio_values
            ),
            reference_sensitivity=exact_positive_nearest_rank_p95(
                measurement.sensitivity_ratio_values
            ),
        )
        semantic = self.semantic.observe(
            normalized.routing_rgb,
            self.prompts.entries[source_cluster_ordinal].prompt,
        )
        return RoutingObservations(
            semantic=semantic,
            texture=normalized.texture,
            response=normalized.response,
            sensitivity=normalized.sensitivity,
        )

    def build_operational_inputs(
        self,
        base_latent: torch.Tensor,
        *,
        source_cluster_ordinal: int,
    ) -> dict[str, DevelopmentUnitInput]:
        """Build real non-scientific inputs without provisional/formal thresholds."""

        routing = self._operational_routing_observations(
            base_latent,
            source_cluster_ordinal=source_cluster_ordinal,
        )
        shape = tuple(int(value) for value in base_latent.shape)
        values = tuple(
            float(value)
            for value in base_latent.detach().to(
                device="cpu", dtype=torch.float32
            ).reshape(-1)
        )
        high_frequency = self.runner.adapter.detect_hf(
            HfDetectionObservation.from_public_image_encoding(values, shape),
            self.root_key,
        ).result
        low_frequency = self.runner.adapter.detect_lf(
            LfDetectionObservation.from_public_image_encoding(values, shape),
            self.root_key,
        ).result
        high_frequency_null = BranchNullCalibration(
            branch="hf",
            detector_identity=high_frequency.detector_identity,
            partition_identity="wiring_only_non_scientific",
            records=(
                NullScoreRecord(
                    high_frequency.hf_score - 0.25,
                    "wiring_null_cluster_low",
                    "wiring_null_sample_low",
                ),
                NullScoreRecord(
                    high_frequency.hf_score + 0.25,
                    "wiring_null_cluster_high",
                    "wiring_null_sample_high",
                ),
            ),
        )
        low_frequency_null = BranchNullCalibration(
            branch="lf",
            detector_identity=low_frequency.detector_identity,
            partition_identity="wiring_only_non_scientific",
            records=(
                NullScoreRecord(
                    low_frequency.lf_score - 0.25,
                    "wiring_null_cluster_low",
                    "wiring_null_sample_low",
                ),
                NullScoreRecord(
                    low_frequency.lf_score + 0.25,
                    "wiring_null_cluster_high",
                    "wiring_null_sample_high",
                ),
            ),
        )
        reliability = GeometryReliabilityThresholds(
            gamma_coverage=0.0,
            gamma_uniqueness=0.0,
            gamma_gap=-1_000_000.0,
            gamma_key=-1_000_000.0,
            gamma_inlier=0.0,
            gamma_residual=1_000_000.0,
            gamma_identity=-1_000_000.0,
            epsilon_inlier=0.8,
            fit_identity="wiring_only_non_scientific",
        )
        combination_function_id, mixing_coefficient = _combination(
            source_cluster_ordinal
        )
        common = dict(
            registered_root_key=self.root_key,
            wrong_key_index=source_cluster_ordinal % 4,
            base_latent=base_latent,
            routing_observations=routing,
            mixing_coefficient=mixing_coefficient,
            combination_function_id=combination_function_id,
            hf_null=high_frequency_null,
            lf_null=low_frequency_null,
            epsilon_inlier=0.8,
            geometry_reliability_thresholds=reliability,
            provisional_threshold=None,
            cross_fit_plan=None,
            development_tau_rescue=None,
        )
        return {
            study.responsibility_id: DevelopmentUnitInput(**common)
            for study in self.protocol.module_matrix
        }

    def prepare_routing_reference_fit(
        self,
        backend,
        latent_factory,
        *,
        lease: PersistentLease,
        soft_stop_epoch_seconds: int,
    ) -> RoutingReferencePreparationStatus:
        successful = {
            item.source_cluster_ordinal: item
            for item in self.session_cursor.routing_reference_records
            if item.execution_status == "success"
        }
        terminal = {
            item.source_cluster_ordinal: item
            for item in self.session_cursor.terminal_routing_reference_records
        }
        for entry in self.prompts.entries:
            if entry.cluster_ordinal in terminal:
                continue
            if int(time.time()) >= soft_stop_epoch_seconds:
                return "soft_stop"
            unit = next(
                item
                for item in self.protocol.unit_roster
                if item.phase == ROUTING_REFERENCE_RECORD_KIND
                and item.source_cluster_ordinal == entry.cluster_ordinal
            )
            binding = self._registered_unit_binding(unit)
            if (
                unit.phase != ROUTING_REFERENCE_RECORD_KIND
                or unit.source_cluster_ordinal != entry.cluster_ordinal
            ):
                raise DevelopmentInputError(
                    "routing reference unit roster identity drifted"
                )
            now = int(time.time())
            if self.session_cursor.next_unit_index != binding.unit_index:
                raise DevelopmentInputError(
                    "routing reference cursor differs from the frozen roster"
                )
            started = time.monotonic()
            intent = self.store.create_session_intent(
                self.session_cursor,
                lease,
                now_epoch_seconds=now,
            )
            try:
                backend.set_development_generation_prompts(entry.prompt)
                measurement = (
                    self.runtime.measure_generation_routing_reference_inputs(
                        latent_factory(entry.generation_seed),
                        sample_index=entry.cluster_ordinal,
                    )
                )
                elapsed = time.monotonic() - started
                if elapsed > intent.maximum_duration_seconds:
                    record = self._routing_reference_attempt_record(
                        unit=unit,
                        binding=binding,
                        intent=intent,
                        source_cluster_ordinal=entry.cluster_ordinal,
                        actual_elapsed_seconds=elapsed,
                        failure_class="resource_failure",
                        failure_reason="routing_reference_duration_exceeded",
                    )
                else:
                    payload = {
                        "candidate_id": measurement.candidate_id,
                        "runtime_config_digest": measurement.runtime_config_digest,
                        "model_id": measurement.model_id,
                        "model_revision": measurement.model_revision,
                        "callback_indices": list(measurement.callback_indices),
                        "public_probe_domain_digest": (
                            measurement.public_probe_domain_digest
                        ),
                        "public_probe_values_digest": (
                            measurement.public_probe_values_float32_be_sha256
                        ),
                        "nominal_relative_probe_step": (
                            measurement.nominal_relative_probe_step
                        ),
                        "actual_probe_step": measurement.actual_probe_step,
                        "texture_gradient_values": list(
                            measurement.texture_gradient_values
                        ),
                        "texture_spatial_shape": list(
                            measurement.texture_spatial_shape
                        ),
                        "response_ratio_values": list(
                            measurement.response_ratio_values
                        ),
                        "response_spatial_shape": list(
                            measurement.response_spatial_shape
                        ),
                        "sensitivity_ratio_values": list(
                            measurement.sensitivity_ratio_values
                        ),
                        "sensitivity_spatial_shape": list(
                            measurement.sensitivity_spatial_shape
                        ),
                    }
                    record = self._routing_reference_attempt_record(
                        unit=unit,
                        binding=binding,
                        intent=intent,
                        source_cluster_ordinal=entry.cluster_ordinal,
                        actual_elapsed_seconds=elapsed,
                        measurement_payload=payload,
                    )
            except (
                MemoryError,
                OSError,
                RuntimeAdapterError,
                RuntimeContentExecutionError,
                torch.cuda.OutOfMemoryError,
            ):
                record = self._routing_reference_attempt_record(
                    unit=unit,
                    binding=binding,
                    intent=intent,
                    source_cluster_ordinal=entry.cluster_ordinal,
                    actual_elapsed_seconds=time.monotonic() - started,
                    failure_class="resource_failure",
                    failure_reason="routing_reference_resource_exhausted",
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                record = self._routing_reference_attempt_record(
                    unit=unit,
                    binding=binding,
                    intent=intent,
                    source_cluster_ordinal=entry.cluster_ordinal,
                    actual_elapsed_seconds=elapsed,
                    failure_class=(
                        "resource_failure"
                        if elapsed > intent.maximum_duration_seconds
                        else "implementation_failure"
                    ),
                    failure_reason=(
                        "routing_reference_duration_exceeded"
                        if elapsed > intent.maximum_duration_seconds
                        else f"{type(exc).__module__}.{type(exc).__qualname__}"
                    ),
                )
            self.store.commit_session_unit(
                self.session_cursor,
                lease,
                intent,
                record=record,
                raw_secret_values=(self.root_key, self.hf_token),
                now_epoch_seconds=max(now + 1, int(time.time())),
            )
            if record.execution_status == "retry":
                return "retryable_stop"
            if record.execution_status == "success":
                successful[entry.cluster_ordinal] = record
            terminal[entry.cluster_ordinal] = record
        if len(terminal) != 64:
            raise DevelopmentInputError(
                "routing reference terminal coverage is incomplete"
            )
        if len(successful) == 64:
            return "complete_success"
        return "terminal_blocked"

    def _routing_reference_attempt_record(
        self,
        *,
        unit: DevelopmentStudyUnit,
        binding: FrozenDevelopmentUnitBinding,
        intent: UnitIntent,
        source_cluster_ordinal: int,
        actual_elapsed_seconds: float,
        measurement_payload: dict[str, object] | None = None,
        failure_class: str | None = None,
        failure_reason: str | None = None,
    ) -> DevelopmentRoutingReferenceRecord:
        """Build one success, retryable, or final routing-reference record."""

        retryable = (
            failure_class == "resource_failure"
            and intent.attempt_index + 1 < unit.maximum_record_attempts
        )
        execution_status = (
            "success"
            if failure_class is None
            else "retry"
            if retryable
            else "failed"
        )
        if failure_class == "resource_failure" and not retryable:
            failure_reason = (
                "routing_reference_resource_blocked_after_attempt_exhaustion"
            )
        record = DevelopmentRoutingReferenceRecord(
            schema_version=ROUTING_REFERENCE_RECORD_SCHEMA,
            collection_role=ROUTING_REFERENCE_RECORD_COLLECTION_ROLE,
            record_kind=ROUTING_REFERENCE_RECORD_KIND,
            record_id="0" * 64,
            run_id=self.authority.run_id,
            protocol_digest=self.protocol.digest(),
            method_code_revision=self.runner.method_code_revision,
            unit_index=unit.unit_index,
            phase=unit.phase,
            source_cluster_ordinal=source_cluster_ordinal,
            fold_index=source_cluster_ordinal % 4,
            prompt_roster_digest=self.prompts.digest,
            candidate_config_digest=binding.candidate_config_digest,
            attempt_index=intent.attempt_index,
            retry_parent_intent_digest=intent.parent_attempt_intent_digest,
            actual_elapsed_seconds=actual_elapsed_seconds,
            maximum_duration_seconds=intent.maximum_duration_seconds,
            duration_limit_exceeded=(
                actual_elapsed_seconds > intent.maximum_duration_seconds
            ),
            execution_status=execution_status,
            failure_class=failure_class,
            failure_reason=failure_reason,
            measurement_payload=(
                {} if measurement_payload is None else measurement_payload
            ),
            counts_as_scientific_coverage=False,
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

    def _routing_observations(self, unit: DevelopmentStudyUnit, base_latent: torch.Tensor):
        fold_index = unit.source_cluster_ordinal % 4
        records = {
            item.source_cluster_ordinal: item
            for item in self.session_cursor.routing_reference_records
            if item.execution_status == "success"
        }
        if len(records) != 64:
            raise DevelopmentDependencyInputBlocked(
                "routing reference fit lacks complete successful evidence"
            )
        rows = [
            records[index].measurement_payload
            for index in range(64)
            if index % 4 != fold_index
        ]
        references = (
            exact_positive_nearest_rank_p95([value for row in rows for value in row["texture_gradient_values"]]),
            exact_positive_nearest_rank_p95([value for row in rows for value in row["response_ratio_values"]]),
            exact_positive_nearest_rank_p95([value for row in rows for value in row["sensitivity_ratio_values"]]),
        )
        measurement = self.runtime.measure_generation_routing_reference_inputs(
            base_latent,
            sample_index=unit.source_cluster_ordinal,
        )
        normalized = self.runtime.normalize_generation_routing_measurement(
            measurement,
            reference_gradient=references[0],
            reference_response=references[1],
            reference_sensitivity=references[2],
        )
        semantic = self.semantic.observe(
            normalized.routing_rgb,
            self.prompts.entries[unit.source_cluster_ordinal].prompt,
        )
        return RoutingObservations(
            semantic=semantic,
            texture=normalized.texture,
            response=normalized.response,
            sensitivity=normalized.sensitivity,
        )

    def _registered_unit_binding(self, unit: DevelopmentStudyUnit):
        matches = tuple(
            binding
            for binding in self.store.registered_unit_bindings
            if binding.unit_index == unit.unit_index
        )
        if len(matches) != 1 or matches[0].study_unit() != unit:
            raise DevelopmentInputError(
                "development unit differs from the registered persistence roster"
            )
        return matches[0]

    def _development_split_manifest(self, unit: DevelopmentStudyUnit):
        binding = self._registered_unit_binding(unit)
        assignments = tuple(
            assignment
            for assignment in self.authority.input_manifest.assignments
            if assignment.identity == binding.analysis_unit_identity
        )
        if len(assignments) != 1 or assignments[0].split != "development":
            raise DevelopmentInputError(
                "joint development unit lacks one registered development assignment"
            )
        return replace(
            self.authority.input_manifest,
            assignments=assignments,
        )

    def _regenerate_public_source_image(
        self,
        unit: DevelopmentStudyUnit,
        base_latent: torch.Tensor,
        routing_observations: RoutingObservations | None,
    ):
        shape = tuple(int(value) for value in base_latent.shape)
        routing = self.runner.adapter.route_content(
            shape,
            mode=(
                "routing_stqr"
                if unit.content_branch_id == "lf_hf_routed_combination"
                else "routing_uniform_control"
            ),
            observations=(
                routing_observations
                if unit.content_branch_id == "lf_hf_routed_combination"
                else None
            ),
        ).result
        lf_carrier = self.runner.adapter.build_lf_carrier(
            self.root_key,
            shape,
            routing_result=routing,
        ).result
        hf_carrier = self.runner.adapter.build_hf_carrier(
            self.root_key,
            shape,
            routing_result=routing,
        ).result
        captured_embeddings: list[object] = []

        def embedding_operation(latent_values: tuple[float, ...]):
            if unit.content_branch_id in {"clean_control", "hf_only"}:
                result = self.runner.adapter.embed_content(
                    latent_values,
                    hf_carrier,
                ).result
            elif unit.content_branch_id == "lf_only":
                result = self.runner.adapter.embed_content(
                    latent_values,
                    None,
                    lf_carrier_result=lf_carrier,
                ).result
            else:
                _combination_function_id, mixing_coefficient = _combination(
                    unit.source_cluster_ordinal
                )
                result = self.runner.adapter.embed_content(
                    latent_values,
                    hf_carrier,
                    lf_carrier_result=lf_carrier,
                    mixing_coefficient=mixing_coefficient,
                    routing_result=routing,
                ).result
            captured_embeddings.append(result)
            return result

        runtime_result = self.runtime.execute_content_write_and_vae(
            base_latent,
            embedding_operation,
        )
        if len(captured_embeddings) != 1:
            raise DevelopmentInputError(
                "joint public source regeneration did not embed exactly once"
            )
        decoded_image = (
            runtime_result.clean_image
            if unit.content_branch_id == "clean_control"
            else runtime_result.watermarked_image
        )
        return _decoded_public_rgb8(decoded_image), routing

    def _geometry_attack_specification(
        self,
        unit: DevelopmentStudyUnit,
    ) -> GeometricAttackSpec:
        try:
            case = self.protocol.geometry_study.case(unit.geometry_case_id)
        except ValueError as exc:
            raise DevelopmentInputError(
                "joint unit geometry case is outside the frozen protocol"
            ) from exc
        attack_id = {
            "identity": "identity",
            "crop": "crop",
            "scale": "scale",
            "rotation": "rotation",
            "compound": "crop_scale_rotation",
        }.get(case.operation_family)
        if attack_id is None:
            raise DevelopmentInputError(
                "joint unit geometry operation family is unregistered"
            )
        return GeometricAttackSpec(
            attack_id,
            crop_fraction=case.crop_fraction,
            scale_factor=case.scale_factor,
            rotation_degrees=case.rotation_degrees,
        )

    def _build_conditional_inputs(
        self,
        unit: DevelopmentStudyUnit,
        base_latent: torch.Tensor,
        routing_observations: RoutingObservations | None,
        *,
        provisional_threshold: DevelopmentProvisionalThreshold,
        development_tau_rescue: float,
        rescue_threshold_identity: str,
        reliability: GeometryReliabilityThresholds,
        intent: UnitIntent,
    ) -> tuple[InternalRunnerContext, InternalCaseExecutionPayload]:
        binding = self._registered_unit_binding(unit)
        split_manifest = self._development_split_manifest(unit)
        source_image, routing = self._regenerate_public_source_image(
            unit,
            base_latent,
            routing_observations,
        )
        source_artifact = AttackArtifact(
            binding.analysis_unit_identity,
            source_image,
        )
        attack_specification = self._geometry_attack_specification(unit)
        content_operation = FormalHfContentDetectionOperation(
            self.runner.adapter
        )
        content_binding, _prototype_score = (
            create_formal_content_detector_binding(
                content_operation,
                prototype_image=source_artifact.image,
                detection_key=self.root_key,
            )
        )
        thresholds = JointDecisionThresholds(
            tau=provisional_threshold.threshold,
            tau_rescue=development_tau_rescue,
            detector_binding_digest=content_binding.detector_binding_digest,
            calibration_identity=rescue_threshold_identity,
        )
        geometry_scope = _canonical_digest(
            {
                "execution_intent_authority_digest": self.authority.authority_digest,
                "reliability_fit_identity": reliability.fit_identity,
                "rescue_threshold_identity": rescue_threshold_identity,
                "unit_descriptor_digest": binding.unit_descriptor_digest,
            }
        )
        geometry_operation = FormalRuntimeGeometryEstimationOperation(
            runtime_adapter=self.runtime,
            adapter_configuration=self.runner.adapter.configuration,
            epsilon_inlier=reliability.epsilon_inlier,
            execution_scope=geometry_scope,
        )
        geometry_operation_identity = (
            "development_runtime_geometry_estimation_" + geometry_scope[:24]
        )
        payload = InternalCaseExecutionPayload(
            source_artifact=source_artifact,
            attack_specification=attack_specification,
            detection_key=self.root_key,
            content_detector_binding=content_binding,
            thresholds=thresholds,
            geometry_estimation_operation=geometry_operation,
            geometry_operation_identity=geometry_operation_identity,
            geometry_reliability_thresholds=reliability,
        )
        execution_expectation = FrozenCaseExecutionExpectation(
            content_detector_binding_digest=content_binding.detector_binding_digest,
            content_operation_config_digest=formal_operation_config_digest(
                content_operation,
                operation_role="content_detection",
            ),
            raw_detector_identity=content_binding.detector_identity,
            rectified_detector_identity=content_binding.detector_identity,
            raw_detector_config_digest=content_binding.content_config_digest,
            rectified_detector_config_digest=content_binding.content_config_digest,
            raw_preprocessing_identity=content_binding.preprocessing_identity,
            rectified_preprocessing_identity=content_binding.preprocessing_identity,
            raw_threshold_identity=thresholds.threshold_identity,
            rectified_threshold_identity=thresholds.threshold_identity,
            calibration_identity=thresholds.calibration_identity,
            tau=thresholds.tau,
            tau_rescue=thresholds.tau_rescue,
            geometry_operation_identity=geometry_operation_identity,
            geometry_operation_config_digest=formal_operation_config_digest(
                geometry_operation,
                operation_role="geometry_estimation",
            ),
            geometry_reliability_config_digest=(
                geometry_reliability_config_digest(reliability)
            ),
        )
        key_identity = self.runner.adapter.identify_key(self.root_key).result
        input_entry = InternalCaseManifestEntry(
            analysis_unit_identity=binding.analysis_unit_identity,
            split="development",
            input_artifact_digest=source_artifact.image_digest,
            attack_config_digest=attack_specification.attack_config_digest,
            metric_set_digest=self.metric_registry.registry_digest,
            routing_trace=RoutingTrace(
                routing_identity=routing.route_identity,
                routing_control=routing.mode,
                routing_observation_digest=_canonical_digest(
                    routing.observation_digests
                ),
                routing_mask_digest=_canonical_digest(
                    (routing.mask_lf_digest, routing.mask_hf_digest)
                ),
            ),
            key_control_trace=KeyControlTrace(
                registered_key_public_digest=(
                    key_identity.root_key_public_digest
                ),
                detection_key_public_digest=(
                    key_identity.root_key_public_digest
                ),
                key_role="registered",
                control_identity="registered_key_control",
            ),
            execution_expectation=execution_expectation,
        )
        input_manifest = FrozenCaseInputManifest(
            manifest_schema_version=INPUT_MANIFEST_SCHEMA_VERSION,
            manifest_id=(
                "development_joint_case_input_"
                + binding.unit_descriptor_digest[:24]
            ),
            manifest_revision=self.runner.method_code_revision,
            protocol_digest=self.internal_protocol.digest(),
            split_manifest_digest=split_manifest.digest(),
            entries=(input_entry,),
        )
        record_bindings = FrozenRecordBindings(
            run_id=self.authority.run_id,
            case_id=binding.analysis_unit_identity.case_id,
            input_manifest_digest=input_manifest.digest(),
            method_code_revision=self.runner.method_code_revision,
            candidate_config_digest=candidate_config_digest(
                adapter=self.runner.adapter,
                input_manifest=input_manifest,
                method_code_revision=self.runner.method_code_revision,
            ),
            method_config_digest=content_binding.content_config_digest,
            execution_config_digest=execution_config_digest(
                protocol=self.internal_protocol,
                adapter=self.runner.adapter,
                attack_registry=self.runner.attack_registry,
                metric_registry=self.metric_registry,
            ),
            model_revision=self.runtime.session.model_revision,
            environment_digest=self.runner.environment_digest,
            resource_identity_digest=self.runner.resource_identity_digest,
        )
        if intent.unit_index != unit.unit_index:
            raise DevelopmentInputError(
                "joint scratch record intent differs from frozen unit"
            )
        records_root = self._internal_record_scratch_root(
            unit_descriptor_digest=binding.unit_descriptor_digest,
            intent=intent,
        )
        writer = GovernedRecordWriter(
            records_root=records_root,
            frozen_protocol=self.internal_protocol,
            split_manifest=split_manifest,
            input_manifest=input_manifest,
            bindings=record_bindings,
            record_scope=DEVELOPMENT_ONLY_RECORD_SCOPE,
        )
        context = InternalRunnerContext(
            protocol=self.internal_protocol,
            split_manifest=split_manifest,
            input_manifest=input_manifest,
            adapter=self.runner.adapter,
            attack_registry=self.runner.attack_registry,
            metric_registry=self.metric_registry,
            writer=writer,
            bindings=record_bindings,
            record_scope=DEVELOPMENT_ONLY_RECORD_SCOPE,
        )
        return context, payload

    def _reliability_thresholds(
        self,
        unit: DevelopmentStudyUnit,
        now_epoch_seconds: int,
    ) -> GeometryReliabilityThresholds:
        current_identity = self._registered_unit_binding(
            unit
        ).analysis_unit_identity
        evidence = self.session_cursor.terminal_scientific_evidence
        fit_records = [
            record
            for record, _ in evidence
            if record.responsibility_id == "geometric_transform_estimator"
            and record.execution_status == "success"
            and record.analysis_unit_identity["source_cluster_id"]
            != current_identity.source_cluster_id
            and next(
                index
                for index, assignment in enumerate(self.authority.input_manifest.assignments)
                if assignment.identity.source_cluster_id
                == record.analysis_unit_identity["source_cluster_id"]
            ) % 4
            != unit.source_cluster_ordinal % 4
        ]
        payloads = [record.operation_result_payload for record in fit_records]
        residuals = [
            float(value)
            for payload in payloads
            for value in payload.get("anchor_residuals", ())
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(float(value))
            and float(value) >= 0.0
        ]
        epsilon = _lower_nearest_rank(residuals, 0.95)
        fields = {
            name: [float(payload[name]) for payload in payloads if isinstance(payload.get(name), (int, float)) and not isinstance(payload.get(name), bool) and isfinite(float(payload[name]))]
            for name in ("coverage", "uniqueness", "gap", "key_margin", "mean_residual", "identity_margin")
        }
        if any(not values for values in fields.values()):
            raise DevelopmentDependencyInputBlocked(
                "verified_estimator_evidence_incomplete"
            )
        declaration = {
            "fit_role": "development_exploratory_geometry_reliability_cross_fit",
            "protocol_digest": self.protocol.digest(),
            "record_ids": [record.record_id for record in fit_records],
            "epsilon_inlier": epsilon,
        }
        return GeometryReliabilityThresholds(
            gamma_coverage=max(0.45, _lower_nearest_rank(fields["coverage"], 0.05)),
            gamma_uniqueness=_lower_nearest_rank(fields["uniqueness"], 0.05),
            gamma_gap=_lower_nearest_rank(fields["gap"], 0.05),
            gamma_key=_lower_nearest_rank(fields["key_margin"], 0.05),
            gamma_inlier=_lower_nearest_rank(
                [
                    sum(float(value) <= epsilon for value in payload["anchor_residuals"])
                    / len(payload["anchor_residuals"])
                    for payload in payloads
                    if payload.get("anchor_residuals")
                ],
                0.05,
            ),
            gamma_residual=_lower_nearest_rank(fields["mean_residual"], 0.95),
            gamma_identity=_lower_nearest_rank(fields["identity_margin"], 0.05),
            epsilon_inlier=epsilon,
            fit_identity=_canonical_digest(declaration),
        )

    def build(
        self,
        unit: DevelopmentStudyUnit,
        base_latent: torch.Tensor,
        *,
        intent: UnitIntent,
        now_epoch_seconds: int,
    ) -> DevelopmentUnitInput:
        if intent.unit_index != unit.unit_index or intent.phase != unit.phase:
            raise DevelopmentInputError(
                "scientific input build requires the claimed unit intent"
            )
        combination_function_id, mixing_coefficient = _combination(unit.source_cluster_ordinal)
        routing_observations = None
        if (
            unit.responsibility_id == "content_router"
            or unit.content_branch_id == "lf_hf_routed_combination"
        ):
            routing_observations = self._routing_observations_by_cluster.get(
                unit.source_cluster_ordinal
            )
            if routing_observations is None:
                routing_observations = self._routing_observations(unit, base_latent)
                self._routing_observations_by_cluster[
                    unit.source_cluster_ordinal
                ] = routing_observations
        hf_null = lf_null = None
        if unit.responsibility_id == "content_detector":
            current_identity = self._registered_unit_binding(
                unit
            ).analysis_unit_identity
            source_cluster_ordinals = {
                assignment.identity.source_cluster_id: index
                for index, assignment in enumerate(
                    self.authority.input_manifest.assignments
                )
            }
            hf_null = replay_branch_null_calibration(
                self.session_cursor.terminal_scientific_evidence,
                branch="hf",
                current_source_cluster_id=current_identity.source_cluster_id,
                source_cluster_ordinals=source_cluster_ordinals,
            )
            lf_null = replay_branch_null_calibration(
                self.session_cursor.terminal_scientific_evidence,
                branch="lf",
                current_source_cluster_id=current_identity.source_cluster_id,
                source_cluster_ordinals=source_cluster_ordinals,
            )
        reliability = None
        if unit.responsibility_id in {"geometry_reliability", "image_rectifier", "conditional_recovery_decision"}:
            reliability = self._reliability_thresholds(unit, now_epoch_seconds)
        provisional_threshold = cross_fit_plan = development_tau_rescue = None
        internal_runner_context = internal_case_payload = None
        if unit.responsibility_id == "conditional_recovery_decision":
            cross_fit_plan = build_development_cross_fit_plan(
                responsibility_id="hf_detector",
                execution_intent_authority=self.authority,
                expected_execution_intent_authority_digest=self.authority.authority_digest,
                expected_source_cluster_count=len(
                    self.authority.input_manifest.assignments
                ),
            )
            thresholds = self.runner.replay_hf_provisional_thresholds_from_evidence(
                cross_fit_plan=cross_fit_plan,
                verified_evidence=(
                    self.session_cursor.terminal_scientific_evidence
                ),
            )
            source_cluster_id = self._registered_unit_binding(
                unit
            ).analysis_unit_identity.source_cluster_id
            provisional_threshold = next(
                threshold for threshold, fold in zip(thresholds, cross_fit_plan.folds, strict=True)
                if source_cluster_id in fold.recovery_probe_source_cluster_ids
            )
            (
                development_tau_rescue,
                rescue_threshold_identity,
            ) = _development_rescue_threshold(
                provisional_threshold
            )
            (
                internal_runner_context,
                internal_case_payload,
            ) = self._build_conditional_inputs(
                unit,
                base_latent,
                routing_observations,
                provisional_threshold=provisional_threshold,
                development_tau_rescue=development_tau_rescue,
                rescue_threshold_identity=rescue_threshold_identity,
                reliability=reliability,
                intent=intent,
            )
        return DevelopmentUnitInput(
            registered_root_key=self.root_key,
            wrong_key_index=unit.source_cluster_ordinal % 4,
            base_latent=base_latent,
            routing_observations=routing_observations,
            mixing_coefficient=mixing_coefficient,
            combination_function_id=combination_function_id,
            hf_null=hf_null,
            lf_null=lf_null,
            epsilon_inlier=(
                None
                if unit.responsibility_id == "geometric_transform_estimator"
                else reliability.epsilon_inlier
                if reliability is not None
                else None
            ),
            geometry_reliability_thresholds=reliability,
            provisional_threshold=provisional_threshold,
            cross_fit_plan=cross_fit_plan,
            development_tau_rescue=development_tau_rescue,
            internal_runner_context=internal_runner_context,
            internal_case_payload=internal_case_payload,
        )
