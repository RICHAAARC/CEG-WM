"""Public runtime runner for the salient-local-LF mask/write pilot."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import isfinite
import platform
from time import monotonic
from typing import Sequence

import torch

from experiments.methods import CegWmExperimentAdapter
from experiments.metrics.salient_local_lf_mask_write_validation import (
    SalientLocalLfMaskWriteObservation,
    SalientLocalLfTerminalFailure,
    aggregate_salient_local_lf_mask_write_validation,
    create_mask_write_observation,
    observe_public_rgb8_quality,
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
from experiments.protocol.salient_local_lf_mask_write_validation import (
    OPERATIONAL_UNIT_COUNT,
    SalientLocalLfMaskWriteProtocol,
    canonical_digest,
)
from experiments.runners.development_persistence import (
    CommittedUnit,
    FrozenDevelopmentUnitBinding,
    create_frozen_development_unit_binding,
)
from main import (
    SalientLocalLfEmbeddingResult,
    identify_root_key,
    salient_local_lf_content_embedder,
)
from runtime import InspyrenetSaliencyRuntime, Sd35RuntimeAdapter


class SalientLocalLfMaskWriteRunnerError(RuntimeError):
    """The public pilot execution or persistent replay drifted."""


def _metric_observation_payload(
    observation: SalientLocalLfMaskWriteObservation,
    *,
    source_cluster_id: str,
    candidate_config_digest: str,
    content_branch_id: str,
    geometry_case_id: str,
) -> dict[str, object]:
    payload = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "metric_role": "development_exploratory_cluster_level",
        "responsibility_id": "content_embedder",
        "source_cluster_id": source_cluster_id,
        "registered_metric_ids": (
            "salient_mask_iou", "masked_lf_causal_witness",
            "public_rgb8_signed_integer_quality",
        ),
        "candidate_config_digest": candidate_config_digest,
        "paired_ablation_identity": "clean_and_global_hf_local_lf_paired_public_rgb8",
        "content_branch_id": content_branch_id,
        "geometry_case_id": geometry_case_id,
        "sufficient_statistics": (
            ("embed_mask_coverage", float(observation.embed_mask_coverage)),
            ("detect_mask_coverage", float(observation.detect_mask_coverage)),
            ("mask_intersection_over_union", observation.mask_intersection_over_union),
            ("squared_code_delta_sum", float(observation.quality.squared_code_delta_sum)),
        ),
        "result_identity_digests": (observation.observation_identity, observation.quality.observation_identity),
        "threshold_role": "not_applicable_mask_write_validation",
        "threshold_identity": None,
        "threshold_fit_source_cluster_digest": None,
    }
    return {**payload, "observation_digest": canonical_development_value_digest(payload)}


class SalientLocalLfMaskWriteValidationRunner:
    def __init__(self, *, protocol: SalientLocalLfMaskWriteProtocol,
                 adapter: CegWmExperimentAdapter, runtime_adapter: Sd35RuntimeAdapter,
                 saliency_runtime: InspyrenetSaliencyRuntime,
                 method_code_revision: str, registered_root_key: str,
                 protocol_digest: str, execution_intent_authority_digest: str,
                 candidate_config_digest: str, package_identity: str) -> None:
        if (type(adapter) is not CegWmExperimentAdapter or type(runtime_adapter) is not Sd35RuntimeAdapter
                or type(saliency_runtime) is not InspyrenetSaliencyRuntime):
            raise SalientLocalLfMaskWriteRunnerError("exact public runtime identities are required")
        if protocol_digest != protocol.digest() or len(method_code_revision) != 40 or len(package_identity) != 64:
            raise SalientLocalLfMaskWriteRunnerError("execution authority drifted")
        self.protocol = protocol
        self.adapter = adapter
        self.runtime = runtime_adapter
        self.saliency_runtime = saliency_runtime
        self.method_code_revision = method_code_revision
        self.registered_root_key = registered_root_key
        self.root_key_public_digest = identify_root_key(registered_root_key).root_key_public_digest
        self.protocol_digest = protocol_digest
        self.execution_intent_authority_digest = execution_intent_authority_digest
        self.candidate_config_digest = candidate_config_digest
        self.package_identity = package_identity

    def create_persistence_unit_bindings(self) -> tuple[FrozenDevelopmentUnitBinding, ...]:
        bindings = []
        for unit in self.protocol.unit_roster:
            identity = self.protocol.analysis_identity(unit.unit_index)
            bindings.append(create_frozen_development_unit_binding(
                unit, analysis_unit_identity=identity,
                scientific_question_id=(
                    "salient_local_lf_checkpoint_runtime_qualification"
                    if unit.unit_index == 0 else
                    "salient_local_lf_public_runtime_wiring"
                    if unit.unit_index == 1 else
                    "salient_local_lf_mask_write_quality_and_causal_witness"
                ),
                development_case_id=unit.content_branch_id,
                candidate_identity=str(self.protocol.raw["candidate_identity"]),
                candidate_config_digest=self.candidate_config_digest,
            ))
        return tuple(bindings)

    def _operational_record(self, *, unit_index: int, operation: dict[str, object],
                            elapsed: float, attempt_index: int) -> DevelopmentOperationalRecord:
        record = DevelopmentOperationalRecord(
            schema_version=OPERATIONAL_RECORD_SCHEMA,
            collection_role=OPERATIONAL_RECORD_COLLECTION_ROLE,
            record_kind=OPERATIONAL_RECORD_KIND, record_id="0" * 64,
            run_id=self.protocol.run_id, protocol_digest=self.protocol_digest,
            method_code_revision=self.method_code_revision, unit_index=unit_index,
            phase=self.protocol.unit_roster[unit_index].phase,
            source_cluster_ordinal=unit_index, candidate_config_digest=self.candidate_config_digest,
            attempt_index=attempt_index, retry_parent_intent_digest=None,
            actual_elapsed_seconds=elapsed, maximum_duration_seconds=2700,
            operation_result_payload=operation, counts_as_scientific_coverage=False,
            scientific_claims_supported=False, scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
        )
        record = replace(record, record_id=canonical_development_value_digest(record.payload_without_record_id()))
        record.validate()
        return record

    def execute_checkpoint_runtime_preflight(self, *, attempt_index: int = 0) -> DevelopmentOperationalRecord:
        started = monotonic()
        qualification_rgb8 = torch.full((1, 3, 512, 512), 127, dtype=torch.uint8)
        observation = self.saliency_runtime.observe(
            qualification_rgb8,
            observation_role="detect_public_rgb8",
        )
        operation = {
            "operational_role": "inspyrenet_checkpoint_and_public_api_preflight",
            "case_ids": ["package_source_lock_runtime_checkpoint_public_saliency_preflight"],
            "responsibility_result_digests": [["content_router", canonical_digest({
                "checkpoint_asset_identity": self.protocol.raw["checkpoint_asset_identity"],
                "checkpoint_size_bytes": self.protocol.raw["checkpoint_size_bytes"],
                "checkpoint_sha256": self.protocol.raw["checkpoint_sha256"],
                "source_revision": self.protocol.raw["source_revision"],
                "model_revision": self.protocol.raw["model_revision"],
                "preprocess": "pil_bilinear_1024_imagenet_float32",
                "forward": "direct_forward_inspyre_raw_d0_sigmoid_once",
                "observation_identity": observation.observation_identity,
                "probability_digest": observation.probability_digest,
                "probability_spatial_shape": list(observation.spatial_shape),
                "gpu_model": torch.cuda.get_device_name(0),
                "python_runtime": platform.python_version(),
                "torch_runtime": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            })]],
            "counts_as_scientific_coverage": False,
            "scientific_claims_supported": False,
        }
        return self._operational_record(unit_index=0, operation=operation,
                                        elapsed=float(monotonic() - started), attempt_index=attempt_index)

    def execute_public_runtime_preflight(self, *, base_latent: torch.Tensor,
                                         attempt_index: int = 0) -> DevelopmentOperationalRecord:
        started = monotonic()
        result = self.adapter.execute_global_hf_local_lf_content_write(
            base_latent, self.saliency_runtime, self.registered_root_key,
        ).result
        detection = self.runtime.observe_salient_local_lf_detection_image(
            result.watermarked_image_rgb8, self.saliency_runtime,
        )
        operation = {
            "operational_role": "salient_local_lf_public_runtime_throughput_preflight",
            "case_ids": ["nonterminal_content_write_vae_floor_rgb8_saliency_materialization_detection_preflight"],
            "responsibility_result_digests": [["content_embedder", result.embedding_result_identity]],
            "runtime_config_digest": result.runtime_config_digest,
            "embed_observation_identity": result.embed_saliency_observation.observation_identity,
            "detect_observation_identity": detection.saliency_observation.observation_identity,
            "counts_as_scientific_coverage": False,
            "scientific_claims_supported": False,
        }
        return self._operational_record(unit_index=1, operation=operation,
                                        elapsed=float(monotonic() - started), attempt_index=attempt_index)

    def execute_scientific_observation(self, *, unit_index: int,
                                       base_latent: torch.Tensor) -> SalientLocalLfMaskWriteObservation:
        if not OPERATIONAL_UNIT_COUNT <= unit_index < len(self.protocol.unit_roster):
            raise SalientLocalLfMaskWriteRunnerError("scientific unit index is invalid")
        entry = self.protocol.manifest.entries[unit_index - OPERATIONAL_UNIT_COUNT]
        write = self.adapter.execute_global_hf_local_lf_content_write(
            base_latent, self.saliency_runtime, self.registered_root_key,
        ).result
        if write.clean_image_digest == write.watermarked_image_digest:
            raise SalientLocalLfMaskWriteRunnerError("salient content write disappeared")
        embed_route = self.adapter.route_inspyrenet_salient_local_lf(
            tuple(base_latent.shape), write.embed_saliency_observation,
        ).result
        hf = self.adapter.build_hf_carrier(self.registered_root_key, tuple(base_latent.shape)).result
        lf = self.adapter.build_lf_carrier(self.registered_root_key, tuple(base_latent.shape)).result
        reconstructed = salient_local_lf_content_embedder(
            tuple(float(value) for value in base_latent.detach().to(device="cpu", dtype=torch.float32).reshape(-1).tolist()),
            hf, lf, embed_route,
        )
        if type(reconstructed) is not SalientLocalLfEmbeddingResult or reconstructed.embedding_result_identity != write.embedding_result_identity:
            raise SalientLocalLfMaskWriteRunnerError("nominal embedding public replay drifted")
        detected = self.runtime.observe_salient_local_lf_detection_image(
            write.watermarked_image_rgb8, self.saliency_runtime,
        )
        detect_route = self.adapter.route_inspyrenet_salient_local_lf(
            tuple(base_latent.shape), detected.saliency_observation,
        ).result
        embed_mask = tuple(value > 0.5 for value in embed_route.spatial_mask)
        detect_mask = tuple(value > 0.5 for value in detect_route.spatial_mask)
        intersection = sum(left and right for left, right in zip(embed_mask, detect_mask, strict=True))
        union = sum(left or right for left, right in zip(embed_mask, detect_mask, strict=True))
        if union == 0:
            raise SalientLocalLfMaskWriteRunnerError("saliency mask union is empty")
        quality = observe_public_rgb8_quality(
            write.clean_image_rgb8,
            write.watermarked_image_rgb8,
            clean_image_digest=write.clean_image_digest,
            marked_image_digest=write.watermarked_image_digest,
        )
        return create_mask_write_observation(
            cluster_ordinal=entry.cluster_ordinal, source_cluster_id=entry.source_cluster_id,
            clean_image_digest=write.clean_image_digest, marked_image_digest=write.watermarked_image_digest,
            embed_saliency_observation_identity=write.embed_saliency_observation.observation_identity,
            detect_saliency_observation_identity=detected.saliency_observation.observation_identity,
            embed_mask_identity=embed_route.mask_lf_digest, detect_mask_identity=detect_route.mask_lf_digest,
            embed_mask_coverage=embed_route.coverage_spatial_pixels,
            detect_mask_coverage=detect_route.coverage_spatial_pixels,
            mask_intersection_over_union=intersection / float(union),
            nominal_masked_lf_outside_bitwise_zero=reconstructed.mask_outside_bitwise_zero,
            nominal_masked_lf_inside_nonzero=reconstructed.mask_inside_has_energy,
            nominal_masked_lf_consumed_by_materialization=(
                reconstructed.delta_content_digest == write.delta_content_digest
                and write.accepted_materialization.integrity_status == "passed"
            ),
            accepted_materialization_replay_identity=write.accepted_materialization.materialization_replay_identity,
            realized_relative_l2=write.realized_relative_l2,
            actual_dtype_budget_pass=(write.budget_status == "accepted" and write.realized_relative_l2 <= 3.0 / 250.0),
            identity_pass=(detected.input_image_digest == write.watermarked_image_digest),
            integrity_pass=(write.integrity_status == "passed"), quality=quality,
        )

    def _scientific_record(self, *, unit_index: int, attempt_index: int,
                           elapsed: float, observation: SalientLocalLfMaskWriteObservation | None,
                           failure_class: str | None = None, failure_reason: str | None = None) -> DevelopmentScientificRecord:
        unit = self.protocol.unit_roster[unit_index]
        identity = self.protocol.analysis_identity(unit_index)
        success = observation is not None
        operation = {"mask_write_observation": asdict(observation)} if success else {}
        metric = _metric_observation_payload(
            observation,
            source_cluster_id=identity.source_cluster_id,
            candidate_config_digest=self.candidate_config_digest,
            content_branch_id=unit.content_branch_id,
            geometry_case_id=unit.geometry_case_id,
        ) if success else {}
        payload = {
            "schema_version": RECORD_SCHEMA_VERSION, "collection_role": DEVELOPMENT_RECORD_COLLECTION_ROLE,
            "record_id": "0" * 64, "run_id": self.protocol.run_id,
            "protocol_id": self.protocol.protocol_id, "protocol_version": self.protocol.protocol_version,
            "protocol_digest": self.protocol_digest,
            "execution_intent_authority_digest": self.execution_intent_authority_digest,
            "method_code_revision": self.method_code_revision, "unit_index": unit_index,
            "phase": unit.phase, "analysis_unit_identity": asdict(identity),
            "responsibility_id": "content_embedder",
            "scientific_question_id": "salient_local_lf_mask_write_quality_and_causal_witness",
            "development_case_id": unit.content_branch_id,
            "candidate_identity": str(self.protocol.raw["candidate_identity"]),
            "candidate_config_digest": self.candidate_config_digest,
            "paired_ablation_identity": "clean_and_global_hf_local_lf_paired_public_rgb8",
            "negative_control_case_ids": ("clean", "hf_only", "masked_lf_causal", "lf_disabled"),
            "metric_ids": ("salient_mask_iou", "masked_lf_causal_witness", "public_rgb8_signed_integer_quality"),
            "content_branch_id": unit.content_branch_id, "geometry_case_id": unit.geometry_case_id,
            "attempt_index": attempt_index, "execution_status": "success" if success else "failed",
            "failure_class": failure_class, "failure_reason": failure_reason,
            "retry_parent_intent_digest": None, "actual_elapsed_seconds": elapsed,
            "maximum_duration_seconds": unit.maximum_duration_seconds,
            "duration_limit_exceeded": elapsed > unit.maximum_duration_seconds,
            "operation_result_payload": operation,
            "operation_result_digest": canonical_digest(operation),
            "metric_observation": metric, "routing_trace": {}, "branch_score_trace": {},
            "detector_trace": {"formal_detector_executed": False}, "geometry_trace": {}, "threshold_trace": {},
            "key_control_trace": {"root_key_public_digest": self.root_key_public_digest, "raw_secret_persisted": False},
            "decision_trace": {"content_is_only_positive_authority": True, "geometry_positive_authority": False},
            "provenance_trace": {
                "protocol_digest": self.protocol_digest,
                "execution_intent_authority_digest": self.execution_intent_authority_digest,
                "method_code_revision": self.method_code_revision,
                "candidate_config_digest": self.candidate_config_digest,
                "manifest_digest": self.protocol.manifest.digest(),
                "package_identity": self.package_identity,
            },
            "module_outcome": None, "candidate_recommendation": None,
            "scientific_claim_boundary": DEVELOPMENT_CLAIM_BOUNDARY,
        }
        provisional = DevelopmentScientificRecord(**payload)
        record = replace(provisional, record_id=canonical_development_value_digest(provisional.payload_without_record_id()))
        record.validate()
        return record

    def execute_scientific_unit(self, *, unit_index: int, base_latent: torch.Tensor,
                                attempt_index: int = 0) -> DevelopmentScientificRecord:
        started = monotonic()
        observation = self.execute_scientific_observation(unit_index=unit_index, base_latent=base_latent)
        return self._scientific_record(unit_index=unit_index, attempt_index=attempt_index,
                                       elapsed=float(monotonic() - started), observation=observation)

    def create_failed_scientific_record(self, *, unit_index: int, attempt_index: int,
                                        elapsed: float, failure_class: str, failure_reason: str) -> DevelopmentScientificRecord:
        return self._scientific_record(unit_index=unit_index, attempt_index=attempt_index,
                                       elapsed=elapsed, observation=None,
                                       failure_class=failure_class, failure_reason=failure_reason)

    def replay_aggregate(self, evidence: Sequence[tuple[DevelopmentScientificRecord, CommittedUnit]]):
        observations = []
        failures = []
        for record, marker in evidence:
            record.validate()
            if marker.attempt_disposition == "success":
                raw = record.operation_result_payload.get("mask_write_observation")
                if type(raw) is not dict or type(raw.get("quality")) is not dict:
                    raise SalientLocalLfMaskWriteRunnerError("scientific observation payload is missing")
                from experiments.metrics.salient_local_lf_mask_write_validation import PublicRgb8QualityObservation
                observations.append(SalientLocalLfMaskWriteObservation(**{
                    **raw, "quality": PublicRgb8QualityObservation(**raw["quality"]),
                }))
            else:
                failures.append(SalientLocalLfTerminalFailure(
                    record.analysis_unit_identity["source_cluster_id"] and record.unit_index - OPERATIONAL_UNIT_COUNT,
                    str(record.failure_class), str(record.failure_reason),
                ))
        return aggregate_salient_local_lf_mask_write_validation(observations, failures)


__all__ = ["SalientLocalLfMaskWriteValidationRunner", "SalientLocalLfMaskWriteRunnerError"]
