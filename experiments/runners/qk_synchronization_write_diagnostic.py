"""Real public-image runner for Q/K synchronization-write diagnosis."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import isfinite
from time import monotonic
from typing import Sequence

import torch

from experiments.attacks.geometric import (
    AttackArtifact,
    GeometricAttackSpec,
    apply_geometric_attack,
    load_attack_registry,
)
from experiments.methods import CegWmExperimentAdapter
from experiments.metrics.qk_synchronization_write_diagnostic import (
    QkRatioProbeAggregate,
    QkRatioProbeObservation,
    QkSynchronizationDiagnosisAggregate,
    QkTerminalFailure,
    QkTransformDependencyBlockedTerminal,
    QkTransformedRelationObservation,
    aggregate_qk_ratio_probes,
    aggregate_qk_synchronization_diagnosis,
    create_qk_ratio_probe_observation,
    create_qk_rgb8_quality_delta,
    create_qk_transform_dependency_blocked_terminal,
    create_qk_transformed_relation_observation,
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
from experiments.protocol.internal_splits import AnalysisUnitIdentity, derive_source_cluster_id
from experiments.protocol.qk_synchronization_write_diagnostic import (
    OPERATIONAL_UNIT_COUNT,
    RATIO_PROBE_UNIT_COUNT,
    QkSynchronizationManifest,
    QkSynchronizationWriteProtocol,
    canonical_digest,
    derive_qk_synchronization_analysis_identity,
)
from experiments.runners.development_persistence import (
    CommittedUnit,
    FrozenDevelopmentUnitBinding,
    create_frozen_development_unit_binding,
)
from main import derive_wrong_key_material, identify_root_key
from runtime import Sd35RuntimeAdapter


RGB8_MEMBER_PATH = "diagnostics/geometry_written_rgb8.bin"


class QkSynchronizationWriteRunnerError(RuntimeError):
    """The Q/K diagnosis violated its frozen public execution contract."""


def _rgb8_tensor(image: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(image, torch.Tensor)
        or image.ndim != 4
        or tuple(image.shape[:2]) != (1, 3)
        or not bool(torch.isfinite(image).all().item())
    ):
        raise QkSynchronizationWriteRunnerError("public RGB image is invalid")
    return torch.floor(image.detach().to(device="cpu", dtype=torch.float32).clamp(0, 1) * 255).to(torch.uint8).contiguous()


def _rgb8_bytes(image: torch.Tensor) -> bytes:
    return _rgb8_tensor(image).numpy().tobytes()


def _rgb8_digest(image: torch.Tensor) -> str:
    return sha256(_rgb8_bytes(image)).hexdigest()


def _quality(content: torch.Tensor, geometry: torch.Tensor):
    left = _rgb8_tensor(content).to(torch.float64)
    right = _rgb8_tensor(geometry).to(torch.float64)
    difference = right - left
    baseline = float(torch.linalg.vector_norm(left))
    relative = float(torch.linalg.vector_norm(difference)) / max(baseline, 1.0)
    mse = float(torch.mean(difference * difference))
    return create_qk_rgb8_quality_delta(
        relative_l2=relative,
        mean_squared_error=mse,
        content_only_rgb8_digest=_rgb8_digest(content),
        geometry_written_rgb8_digest=_rgb8_digest(geometry),
    )


class QkSynchronizationWriteDiagnosticRunner:
    """Compose HF-only content, real suffix replay and blind Q/K observations."""

    def __init__(
        self,
        *,
        protocol: QkSynchronizationWriteProtocol,
        manifest: QkSynchronizationManifest,
        adapter: CegWmExperimentAdapter,
        runtime_adapter: Sd35RuntimeAdapter,
        method_code_revision: str,
        run_id: str,
        content_registered_root_key: str,
        geometry_registered_root_key: str,
        protocol_digest: str,
        execution_intent_authority_digest: str,
        candidate_config_digest: str,
        package_identity: str,
    ) -> None:
        protocol.validate()
        manifest.validate()
        if type(adapter) is not CegWmExperimentAdapter or type(runtime_adapter) is not Sd35RuntimeAdapter:
            raise QkSynchronizationWriteRunnerError("exact public adapters are required")
        if run_id != protocol.run_id or protocol_digest != protocol.digest():
            raise QkSynchronizationWriteRunnerError("protocol identity drifted")
        if len(method_code_revision) != 40 or len(package_identity) != 64:
            raise QkSynchronizationWriteRunnerError("execution identity is invalid")
        self.protocol = protocol
        self.manifest = manifest
        self.adapter = adapter
        self.runtime = runtime_adapter
        self.method_code_revision = method_code_revision
        self.run_id = run_id
        self.content_root = content_registered_root_key
        self.geometry_root = geometry_registered_root_key
        self.content_public = identify_root_key(content_registered_root_key).root_key_public_digest
        self.geometry_public = identify_root_key(geometry_registered_root_key).root_key_public_digest
        self.protocol_digest = protocol_digest
        self.execution_intent_authority_digest = execution_intent_authority_digest
        self.candidate_config_digest = candidate_config_digest
        self.package_identity = package_identity
        self.attack_registry = load_attack_registry()

    def _key_family(self, role: str) -> str:
        return canonical_digest({"role": role, "protocol_digest": self.protocol_digest, "manifest_digest": self.manifest.digest()})

    def _identity(self, unit_index: int) -> AnalysisUnitIdentity:
        unit = self.protocol.unit_roster[unit_index]
        if unit_index == 0:
            return AnalysisUnitIdentity(
                unit_id="qk_synchronization_write_public_runtime_preflight",
                case_id="qk_synchronization_write_public_runtime_smoke",
                source_cluster_id=derive_source_cluster_id(
                    prompt_digest=self.protocol.operational_smoke_prompt_digest,
                    generation_seed=self.protocol.operational_smoke_generation_seed,
                    image_lineage_digest=self.protocol.operational_smoke_image_lineage_digest,
                    registered_key_family_digest=self._key_family("operational_qk_key_family"),
                ),
                prompt_digest=self.protocol.operational_smoke_prompt_digest,
                generation_seed=self.protocol.operational_smoke_generation_seed,
                image_lineage_digest=self.protocol.operational_smoke_image_lineage_digest,
                registered_key_family_digest=self._key_family("operational_qk_key_family"),
            )
        entry = self.manifest.entries[unit.source_cluster_ordinal]
        return derive_qk_synchronization_analysis_identity(
            entry,
            unit,
            content_key_family_digest=self._key_family("registered_hf_content_key_family"),
            geometry_key_family_digest=self._key_family("registered_geometry_key_family"),
        )

    def create_persistence_unit_bindings(self) -> tuple[FrozenDevelopmentUnitBinding, ...]:
        return tuple(
            create_frozen_development_unit_binding(
                unit,
                analysis_unit_identity=self._identity(unit.unit_index),
                scientific_question_id=(
                    "qk_synchronization_write_public_runtime_smoke"
                    if unit.unit_index == 0
                    else "registered_geometry_write_gain_against_four_wrong_keys"
                    if unit.unit_index <= RATIO_PROBE_UNIT_COUNT
                    else "transformed_public_image_qk_relation_probe"
                ),
                development_case_id=unit.responsibility_id,
                candidate_identity=self.protocol.candidate_identity,
                candidate_config_digest=self.candidate_config_digest,
            )
            for unit in self.protocol.unit_roster
        )

    def _hf_operation(self, base_latent: torch.Tensor):
        carrier = self.adapter.build_hf_carrier(
            self.content_root, tuple(int(size) for size in base_latent.shape), routing_result=None
        ).result

        def embed(values: tuple[float, ...]):
            return self.adapter.embed_content(values, carrier, lf_carrier_result=None, routing_result=None).result

        direction = torch.tensor(carrier.direction, dtype=torch.float32, device=base_latent.device).reshape(base_latent.shape)
        return embed, (direction,), carrier

    def _scores(self, runtime_result):
        registered = self.adapter.synchronize_qk_observation(runtime_result, self.geometry_root)
        wrong = tuple(
            self.adapter.synchronize_qk_observation(
                runtime_result, derive_wrong_key_material(self.geometry_public, index)
            )
            for index in self.protocol.wrong_key_indexes
        )
        results = (registered, *wrong)
        if len({item.upstream_runtime_identity for item in results}) != 1:
            raise QkSynchronizationWriteRunnerError("registered and wrong keys did not reuse one public observation")
        if len({item.result.geometry_config_digest for item in results}) != 1:
            raise QkSynchronizationWriteRunnerError("Q/K detector configuration drifted")
        return registered, wrong

    def execute_ratio_probe(self, *, unit_index: int, base_latent: torch.Tensor):
        unit = self.protocol.unit_roster[unit_index]
        if not 1 <= unit_index <= RATIO_PROBE_UNIT_COUNT:
            raise QkSynchronizationWriteRunnerError("ratio unit is outside frozen roster")
        ratio = next(item for item in self.protocol.geometry_ratio_roster if item.ratio_identity == unit.geometry_case_id)
        embed, directions, carrier = self._hf_operation(base_latent)
        call = self.adapter.execute_qk_synchronization_write(
            base_latent,
            embed,
            directions,
            geometry_ratio=ratio.ratio,
            detection_key=self.geometry_root,
        )
        result = call.result
        content_image = result.content_write_result.watermarked_image
        pre_runtime = self.runtime.observe_detection_qk(content_image)
        pre_registered, pre_wrong = self._scores(pre_runtime)
        write = result.geometry_write_result
        accepted_runtime = result.accepted_actual_runtime_result
        post_registered = None
        post_wrong = ()
        geometry_image = None
        quality = None
        if write.accepted:
            if accepted_runtime is None or result.accepted_post_write_observation is None:
                raise QkSynchronizationWriteRunnerError("accepted write lacks actual public suffix result")
            post_registered, post_wrong = self._scores(accepted_runtime.qk_observation)
            if post_registered.result != result.accepted_post_write_observation:
                raise QkSynchronizationWriteRunnerError("accepted post-write public observation drifted")
            geometry_image = accepted_runtime.rgb8_image
            quality = _quality(content_image, geometry_image)
        observation = create_qk_ratio_probe_observation(
            cluster_ordinal=unit.source_cluster_ordinal,
            ratio_identity=ratio.ratio_identity,
            geometry_ratio=ratio.ratio,
            write_accepted=write.accepted,
            line_search_factor=write.line_search_factor,
            ste_acceptance_baseline_score=write.baseline_score,
            ste_acceptance_score=write.accepted_score,
            public_pre_registered_score=pre_registered.result.relation_score,
            public_pre_wrong_key_scores=tuple(item.result.relation_score for item in pre_wrong),
            public_post_registered_score=(None if post_registered is None else post_registered.result.relation_score),
            public_post_wrong_key_scores=tuple(item.result.relation_score for item in post_wrong),
            actual_geometry_relative_l2=write.geometry_relative_l2_actual,
            actual_total_relative_l2=write.total_relative_l2_actual,
            content_span_projection_relative=write.content_projection_relative,
            rgb8_quality_delta=quality,
            public_pre_observation_identity=pre_registered.upstream_runtime_identity,
            public_post_observation_identity=(None if post_registered is None else post_registered.upstream_runtime_identity),
            content_only_rgb8_digest=_rgb8_digest(content_image),
            geometry_written_rgb8_digest=(None if geometry_image is None else _rgb8_digest(geometry_image)),
            geometry_key_family_digest=self._key_family("registered_geometry_key_family"),
            registered_template_digest=pre_registered.result.projection_digest,
            wrong_key_template_digests=tuple(item.result.projection_digest for item in pre_wrong),
            wrong_key_indexes=self.protocol.wrong_key_indexes,
            method_identity=self.protocol.public_method_callable,
            runtime_identity=self.protocol.public_runtime_chain,
            runtime_config_digest=result.runtime_config_digest,
            model_revision=pre_registered.result.model_revision,
            package_identity=self.package_identity,
            identity_violation_count=0,
            budget_violation_count=0,
            integrity_violation_count=0,
            nonfinite_violation_count=0,
        )
        operation = {
            "routing_used": False,
            "content_branch_id": "hf_only",
            "hf_carrier_identity": carrier.carrier_config_digest,
            "ratio_probe_observation": asdict(observation),
            "accepted_rgb8_member": (
                None
                if geometry_image is None
                else {
                    "path": RGB8_MEMBER_PATH,
                    "shape": tuple(_rgb8_tensor(geometry_image).shape),
                    "dtype": "torch.uint8",
                    "size_bytes": len(_rgb8_bytes(geometry_image)),
                    "sha256": _rgb8_digest(geometry_image),
                }
            ),
            "private_qk_or_latent_persisted": False,
        }
        members = {} if geometry_image is None else {RGB8_MEMBER_PATH: _rgb8_bytes(geometry_image)}
        return observation, operation, members

    def execute_transform_probe(
        self,
        *,
        unit_index: int,
        selected_ratio_identity: str,
        source_rgb8: torch.Tensor,
    ):
        unit = self.protocol.unit_roster[unit_index]
        if not RATIO_PROBE_UNIT_COUNT < unit_index < len(self.protocol.unit_roster):
            raise QkSynchronizationWriteRunnerError("transform unit is outside frozen roster")
        spec = next(item for item in self.protocol.transform_probe_roster if item.transform_identity == unit.geometry_case_id)
        artifact = AttackArtifact(self._identity(unit_index), _rgb8_tensor(source_rgb8))
        attacked = apply_geometric_attack(
            artifact,
            GeometricAttackSpec(
                attack_id=spec.transform_identity,
                crop_fraction=spec.crop_fraction,
                scale_factor=spec.scale_factor,
                rotation_degrees=spec.rotation_degrees,
            ),
            registry=self.attack_registry,
        )
        runtime_result = self.runtime.observe_detection_qk(attacked.image.to(torch.float32) / 255.0)
        registered, wrong = self._scores(runtime_result)
        observation = create_qk_transformed_relation_observation(
            cluster_ordinal=unit.source_cluster_ordinal,
            transform_identity=spec.transform_identity,
            selected_ratio_identity=selected_ratio_identity,
            source_geometry_written_rgb8_digest=_rgb8_digest(source_rgb8),
            transformed_rgb8_digest=sha256(attacked.image.cpu().contiguous().numpy().tobytes()).hexdigest(),
            registered_score=registered.result.relation_score,
            wrong_key_scores=tuple(item.result.relation_score for item in wrong),
            public_observation_identity=registered.upstream_runtime_identity,
            method_identity="main.qk_geometry_sync",
            runtime_identity="runtime.public_rgb8_vae_qk_observation",
            identity_violation_count=0,
            integrity_violation_count=0,
            nonfinite_violation_count=0,
        )
        return observation, {
            "transformed_relation_observation": asdict(observation),
            "source_member_verified": True,
            "diffusion_regenerated": False,
            "private_qk_or_latent_persisted": False,
        }

    def _record(
        self,
        *,
        unit_index: int,
        attempt_index: int,
        retry_parent_intent_digest: str | None,
        maximum_duration_seconds: int,
        actual_elapsed_seconds: float,
        operation: dict[str, object],
        observation_identity: str | None,
        resource_failure: bool = False,
        dependency_blocked: bool = False,
    ) -> DevelopmentScientificRecord:
        unit = self.protocol.unit_roster[unit_index]
        success = observation_identity is not None
        retry = resource_failure and attempt_index + 1 < unit.maximum_record_attempts
        identity = self._identity(unit_index)
        metric_ids = (
            ("registered_gain", "maximum_wrong_gain", "keyed_gain_margin", "rgb8_quality_delta")
            if unit_index <= RATIO_PROBE_UNIT_COUNT
            else ("registered_minus_max_wrong",)
        )
        metric = {}
        if observation_identity is not None:
            raw_observation = operation.get(
                "ratio_probe_observation"
                if unit_index <= RATIO_PROBE_UNIT_COUNT
                else "transformed_relation_observation"
            )
            if type(raw_observation) is not dict:
                raise QkSynchronizationWriteRunnerError(
                    "successful scientific operation lacks its metric payload"
                )
            sufficient_statistics = (
                (
                    ("registered_gain", raw_observation["registered_gain"]),
                    ("maximum_wrong_gain", raw_observation["maximum_wrong_gain"]),
                    ("keyed_gain_margin", raw_observation["keyed_gain_margin"]),
                )
                if unit_index <= RATIO_PROBE_UNIT_COUNT
                else (("registered_minus_max_wrong", raw_observation["registered_minus_max_wrong"]),)
            )
            metric = {
                "schema_version": METRIC_SCHEMA_VERSION,
                "metric_role": "development_exploratory_cluster_level",
                "responsibility_id": unit.responsibility_id,
                "source_cluster_id": identity.source_cluster_id,
                "registered_metric_ids": metric_ids,
                "candidate_config_digest": self.candidate_config_digest,
                "paired_ablation_identity": "public_image_qk_registered_and_four_wrong_keys",
                "content_branch_id": "hf_only",
                "geometry_case_id": unit.geometry_case_id,
                "sufficient_statistics": sufficient_statistics,
                "result_identity_digests": (observation_identity,),
                "threshold_role": "not_fitted_qk_diagnosis",
                "threshold_identity": None,
                "threshold_fit_source_cluster_digest": None,
            }
            metric["observation_digest"] = canonical_development_value_digest(metric)
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
            "unit_index": unit_index,
            "phase": unit.phase,
            "analysis_unit_identity": asdict(identity),
            "responsibility_id": unit.responsibility_id,
            "scientific_question_id": "registered_geometry_write_gain_against_four_wrong_keys" if unit_index <= RATIO_PROBE_UNIT_COUNT else "transformed_public_image_qk_relation_probe",
            "development_case_id": unit.responsibility_id,
            "candidate_identity": self.protocol.candidate_identity,
            "candidate_config_digest": self.candidate_config_digest,
            "paired_ablation_identity": "public_image_qk_registered_and_four_wrong_keys",
            "negative_control_case_ids": ("same_image_four_wrong_geometry_keys",),
            "metric_ids": metric_ids,
            "content_branch_id": "hf_only",
            "geometry_case_id": unit.geometry_case_id,
            "attempt_index": attempt_index,
            "execution_status": (
                "success"
                if success
                else "excluded"
                if dependency_blocked
                else "retry"
                if retry
                else "failed"
            ),
            "failure_class": (
                None
                if success
                else "dependency_blocked"
                if dependency_blocked
                else "resource_failure"
                if resource_failure
                else "implementation_failure"
            ),
            "failure_reason": (
                None
                if success
                else "qk_transform_probe_dependency_blocked"
                if dependency_blocked
                else "qk_synchronization_write_operation_failed"
            ),
            "retry_parent_intent_digest": retry_parent_intent_digest,
            "actual_elapsed_seconds": actual_elapsed_seconds,
            "maximum_duration_seconds": maximum_duration_seconds,
            "duration_limit_exceeded": actual_elapsed_seconds > maximum_duration_seconds,
            "operation_result_payload": operation,
            "operation_result_digest": canonical_development_value_digest(operation),
            "metric_observation": metric,
            "routing_trace": {"routing_used": False},
            "branch_score_trace": {},
            "detector_trace": {"formal_detector_completed": False},
            "geometry_trace": {
                "geometry_attempted": not dependency_blocked,
                "public_callable": self.protocol.public_method_callable,
                "geometry_is_positive_authority": False,
            },
            "threshold_trace": {"threshold_role": "not_fitted_qk_diagnosis", "raw_threshold_identity": None, "rectified_threshold_identity": None},
            "key_control_trace": {"root_key_public_digest": self.geometry_public, "wrong_key_indexes": self.protocol.wrong_key_indexes, "raw_secret_persisted": False},
            "decision_trace": {"positive_source": None, "decision_role": "dependency_blocked_excluded" if dependency_blocked else "threshold_free_qk_diagnosis"},
            "provenance_trace": {
                "protocol_digest": self.protocol_digest,
                "execution_intent_authority_digest": self.execution_intent_authority_digest,
                "method_code_revision": self.method_code_revision,
                "manifest_digest": self.manifest.digest(),
                "candidate_config_digest": self.candidate_config_digest,
                "package_identity": self.package_identity,
            },
            "module_outcome": None,
            "candidate_recommendation": None,
            "scientific_claim_boundary": DEVELOPMENT_CLAIM_BOUNDARY,
        }
        provisional = DevelopmentScientificRecord(**payload)
        record = DevelopmentScientificRecord(**{**payload, "record_id": canonical_development_value_digest(provisional.payload_without_record_id())})
        record.validate()
        return record

    def execute_scientific_unit(self, *, unit_index: int, base_latent: torch.Tensor | None, selected_ratio_identity: str | None = None, source_rgb8: torch.Tensor | None = None, attempt_index: int, retry_parent_intent_digest: str | None, maximum_duration_seconds: int):
        started = monotonic()
        if unit_index <= RATIO_PROBE_UNIT_COUNT:
            if base_latent is None:
                raise QkSynchronizationWriteRunnerError("ratio probe requires base latent")
            observation, operation, members = self.execute_ratio_probe(unit_index=unit_index, base_latent=base_latent)
        else:
            if selected_ratio_identity is None or source_rgb8 is None:
                raise QkSynchronizationWriteRunnerError("transform probe requires selected-ratio RGB8 evidence")
            observation, operation = self.execute_transform_probe(unit_index=unit_index, selected_ratio_identity=selected_ratio_identity, source_rgb8=source_rgb8)
            members = {}
        record = self._record(
            unit_index=unit_index,
            attempt_index=attempt_index,
            retry_parent_intent_digest=retry_parent_intent_digest,
            maximum_duration_seconds=maximum_duration_seconds,
            actual_elapsed_seconds=float(monotonic() - started),
            operation=operation,
            observation_identity=observation.observation_identity,
        )
        return record, members

    def create_dependency_blocked_record(self, *, unit_index: int, attempt_index: int, retry_parent_intent_digest: str | None, maximum_duration_seconds: int):
        unit = self.protocol.unit_roster[unit_index]
        terminal = create_qk_transform_dependency_blocked_terminal(cluster_ordinal=unit.source_cluster_ordinal, transform_identity=unit.geometry_case_id)
        operation = {"dependency_blocked_terminal": asdict(terminal), "result_available": False}
        return self._record(unit_index=unit_index, attempt_index=attempt_index, retry_parent_intent_digest=retry_parent_intent_digest, maximum_duration_seconds=maximum_duration_seconds, actual_elapsed_seconds=0.0, operation=operation, observation_identity=None, dependency_blocked=True)

    def create_failed_record(self, *, unit_index: int, attempt_index: int, retry_parent_intent_digest: str | None, maximum_duration_seconds: int, actual_elapsed_seconds: float, failure_type: str, resource_failure: bool):
        return self._record(unit_index=unit_index, attempt_index=attempt_index, retry_parent_intent_digest=retry_parent_intent_digest, maximum_duration_seconds=maximum_duration_seconds, actual_elapsed_seconds=actual_elapsed_seconds, operation={"failure_type": failure_type, "result_available": False}, observation_identity=None, resource_failure=resource_failure)

    def replay_ratio_aggregate(self, evidence: Sequence[tuple[DevelopmentScientificRecord, CommittedUnit]]) -> QkRatioProbeAggregate:
        observations = []
        failures = []
        for record, marker in evidence:
            if not 1 <= record.unit_index <= RATIO_PROBE_UNIT_COUNT:
                continue
            if marker.attempt_disposition == "success":
                raw = record.operation_result_payload.get("ratio_probe_observation")
                if type(raw) is not dict:
                    raise QkSynchronizationWriteRunnerError("ratio observation is missing")
                quality = raw.get("rgb8_quality_delta")
                if type(quality) is dict:
                    raw = {**raw, "rgb8_quality_delta": create_qk_rgb8_quality_delta(**{key: value for key, value in quality.items() if key != "quality_identity"})}
                observations.append(QkRatioProbeObservation(**{**raw, "public_pre_wrong_key_scores": tuple(raw["public_pre_wrong_key_scores"]), "public_post_wrong_key_scores": tuple(raw["public_post_wrong_key_scores"]), "wrong_key_gains": tuple(raw["wrong_key_gains"]), "wrong_key_template_digests": tuple(raw["wrong_key_template_digests"]), "wrong_key_indexes": tuple(raw["wrong_key_indexes"])}))
            else:
                failures.append(QkTerminalFailure(self.protocol.unit_roster[record.unit_index].source_cluster_ordinal, record.geometry_case_id, record.failure_class))
        return aggregate_qk_ratio_probes(observations, failures)

    def replay_synchronization_diagnosis_aggregate(self, evidence: Sequence[tuple[DevelopmentScientificRecord, CommittedUnit]]) -> QkSynchronizationDiagnosisAggregate:
        ratio = self.replay_ratio_aggregate(evidence)
        transformed = []
        failures = []
        blocked = []
        for record, marker in evidence:
            if record.unit_index <= RATIO_PROBE_UNIT_COUNT:
                continue
            blocked_raw = record.operation_result_payload.get("dependency_blocked_terminal")
            if record.execution_status == "excluded" and type(blocked_raw) is dict:
                blocked.append(QkTransformDependencyBlockedTerminal(**blocked_raw))
            elif marker.attempt_disposition == "success":
                raw = record.operation_result_payload.get("transformed_relation_observation")
                if type(raw) is dict:
                    transformed.append(QkTransformedRelationObservation(**{**raw, "wrong_key_scores": tuple(raw["wrong_key_scores"])}))
                else:
                    raise QkSynchronizationWriteRunnerError("transform terminal payload is missing")
            else:
                failures.append(QkTerminalFailure(self.protocol.unit_roster[record.unit_index].source_cluster_ordinal, record.geometry_case_id, record.failure_class))
        return aggregate_qk_synchronization_diagnosis(ratio, transformed, failures, blocked)

    def execute_operational_smoke(self, *, base_latent: torch.Tensor, attempt_index: int, retry_parent_intent_digest: str | None, maximum_duration_seconds: int):
        started = monotonic()
        embed, directions, _carrier = self._hf_operation(base_latent)
        ratio = self.protocol.geometry_ratio_roster[0]
        result = self.adapter.execute_qk_synchronization_write(base_latent, embed, directions, geometry_ratio=ratio.ratio, detection_key=self.geometry_root).result
        elapsed = float(monotonic() - started)
        operation = {"operational_role": "public_qk_synchronization_write_smoke", "write_status": result.geometry_write_result.status, "runtime_config_digest": result.runtime_config_digest, "counts_as_scientific_coverage": False, "scientific_claims_supported": False}
        record = DevelopmentOperationalRecord(schema_version=OPERATIONAL_RECORD_SCHEMA, collection_role=OPERATIONAL_RECORD_COLLECTION_ROLE, record_kind=OPERATIONAL_RECORD_KIND, record_id="0"*64, run_id=self.run_id, protocol_digest=self.protocol_digest, method_code_revision=self.method_code_revision, unit_index=0, phase="development_environment_preflight", source_cluster_ordinal=0, candidate_config_digest=self.candidate_config_digest, attempt_index=attempt_index, retry_parent_intent_digest=retry_parent_intent_digest, actual_elapsed_seconds=elapsed, maximum_duration_seconds=maximum_duration_seconds, operation_result_payload=operation, counts_as_scientific_coverage=False, scientific_claims_supported=False, scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY)
        record = replace(record, record_id=canonical_development_value_digest(record.payload_without_record_id()))
        record.validate()
        return record


__all__ = ["QkSynchronizationWriteDiagnosticRunner", "QkSynchronizationWriteRunnerError", "RGB8_MEMBER_PATH"]
