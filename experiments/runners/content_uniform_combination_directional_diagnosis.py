"""Real public runner for disabled-routing LF/HF combination diagnosis."""

from __future__ import annotations

from dataclasses import asdict, replace
from time import monotonic
from typing import Sequence

import torch

from experiments.methods import CegWmExperimentAdapter
from experiments.metrics.content_uniform_combination_directional_diagnosis import (
    ContentCombinationArmCanonicalBudgetExceededError,
    ContentCombinationArmImageDigestInvalidError,
    ContentCombinationArmMaterializationRejectedError,
    ContentCombinationArmMeasurementNonfiniteError,
    ContentCombinationArmObservationIdentityDriftError,
    ContentCombinationArmRoleInvalidError,
    ContentCombinationFoldReference,
    ContentCombinationReferenceMeasurement,
    ContentUniformCombinationDirectionalMetricError,
    ContentUniformCombinationDirectionalAggregate,
    ContentUniformCombinationDirectionalObservation,
    aggregate_content_uniform_combination_directional_diagnosis,
    create_content_combination_arm_observation,
    create_content_combination_reference_measurement,
    create_content_combination_score_row,
    create_content_uniform_combination_directional_observation,
    fit_content_combination_fold_reference,
)
from experiments.protocol.content_uniform_combination_directional_diagnosis import (
    COMBINATION_FUNCTIONS,
    COMBINATION_WEIGHTS,
    OPERATIONAL_CASE_IDS,
    OPERATIONAL_ROLE,
    ContentUniformCombinationDirectionalProtocol,
    ContentUniformCombinationManifest,
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
    DevelopmentOperationalRecord,
    DevelopmentScientificRecord,
    canonical_development_value_digest,
)
from experiments.protocol.internal_splits import AnalysisUnitIdentity, derive_source_cluster_id
from experiments.runners.development_persistence import (
    FrozenDevelopmentUnitBinding,
    UnitIntent,
    create_frozen_development_unit_binding,
)
from main import (
    BranchNullCalibration,
    HfDetectionObservation,
    LfDetectionObservation,
    LfNullWhiteningAsset,
    NullScoreRecord,
    derive_wrong_key_material,
    rgb8_image_digest,
)
from runtime import Sd35RuntimeAdapter


class ContentUniformCombinationDirectionalRunnerError(RuntimeError):
    """The combination runner violated its frozen execution contract."""


class ContentCombinationScoreRowConstructionError(
    ContentUniformCombinationDirectionalRunnerError
):
    """A validated content-combination score row could not be constructed."""


class ContentCombinationArmObservationConstructionError(
    ContentUniformCombinationDirectionalRunnerError
):
    """A validated content-combination arm observation could not be constructed."""


class ContentCombinationProbeObservationConstructionError(
    ContentUniformCombinationDirectionalRunnerError
):
    """A validated content-combination probe observation could not be constructed."""


class ContentCombinationArmRoleInvalidRunnerError(
    ContentUniformCombinationDirectionalRunnerError
):
    """The metric rejected the arm role or embedding coefficient."""


class ContentCombinationArmMeasurementNonfiniteRunnerError(
    ContentUniformCombinationDirectionalRunnerError
):
    """The metric rejected a nonfinite arm measurement."""


class ContentCombinationArmCanonicalBudgetExceededRunnerError(
    ContentUniformCombinationDirectionalRunnerError
):
    """The metric rejected an arm above the canonical content budget."""


class ContentCombinationArmMaterializationRejectedRunnerError(
    ContentUniformCombinationDirectionalRunnerError
):
    """The metric rejected an arm materialization status."""


class ContentCombinationArmImageDigestInvalidRunnerError(
    ContentUniformCombinationDirectionalRunnerError
):
    """The metric rejected an arm image digest."""


class ContentCombinationArmObservationIdentityDriftRunnerError(
    ContentUniformCombinationDirectionalRunnerError
):
    """The metric rejected a drifted arm observation identity."""


def _translate_arm_observation_metric_error(
    error: ContentUniformCombinationDirectionalMetricError,
) -> None:
    if type(error) is ContentCombinationArmRoleInvalidError:
        raise ContentCombinationArmRoleInvalidRunnerError(
            "content combination arm role is invalid"
        ) from error
    if type(error) is ContentCombinationArmMeasurementNonfiniteError:
        raise ContentCombinationArmMeasurementNonfiniteRunnerError(
            "content combination arm measurement is nonfinite"
        ) from error
    if type(error) is ContentCombinationArmCanonicalBudgetExceededError:
        raise ContentCombinationArmCanonicalBudgetExceededRunnerError(
            "content combination arm canonical budget was exceeded"
        ) from error
    if type(error) is ContentCombinationArmMaterializationRejectedError:
        raise ContentCombinationArmMaterializationRejectedRunnerError(
            "content combination arm materialization was rejected"
        ) from error
    if type(error) is ContentCombinationArmImageDigestInvalidError:
        raise ContentCombinationArmImageDigestInvalidRunnerError(
            "content combination arm image digest is invalid"
        ) from error
    if type(error) is ContentCombinationArmObservationIdentityDriftError:
        raise ContentCombinationArmObservationIdentityDriftRunnerError(
            "content combination arm observation identity drifted"
        ) from error


def _record_id(record: object) -> object:
    return replace(record, record_id=canonical_development_value_digest(record.payload_without_record_id()))


def _rgb8(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or tuple(image.shape) != (1, 3, 512, 512) or not bool(torch.isfinite(image).all().item()):
        raise ContentUniformCombinationDirectionalRunnerError("public RGB image is invalid")
    if image.dtype is torch.uint8:
        return image.detach().cpu().contiguous()
    return torch.floor(image.detach().cpu().float().clamp(0.0, 1.0) * 255.0).to(torch.uint8).contiguous()


def _relative_l2(reference: torch.Tensor, observed: torch.Tensor) -> float:
    left=reference.float(); right=observed.float(); denominator=torch.linalg.vector_norm(left)
    if float(denominator.item()) <= 0.0:
        raise ContentUniformCombinationDirectionalRunnerError("paired clean RGB has zero norm")
    return float((torch.linalg.vector_norm(right-left)/denominator).item())


def _public_observations(latent: torch.Tensor) -> tuple[HfDetectionObservation, LfDetectionObservation]:
    if not isinstance(latent, torch.Tensor) or tuple(latent.shape)!=(1,16,64,64) or not bool(torch.isfinite(latent).all().item()):
        raise ContentUniformCombinationDirectionalRunnerError("public RGB-to-VAE observation is invalid")
    values=tuple(float(item) for item in latent.detach().cpu().float().reshape(-1))
    shape=tuple(int(value) for value in latent.shape)
    hf=HfDetectionObservation.from_public_image_encoding(values,shape)
    lf=LfDetectionObservation.from_public_image_encoding(values,shape)
    if hf.observation_digest!=lf.observation_digest:
        raise ContentUniformCombinationDirectionalRunnerError("HF/LF public observation identity drifted")
    return hf,lf


class ContentUniformCombinationDirectionalDiagnosisRunner:
    """Run one preflight, clean CDF fit, and eight six-image probes."""

    def __init__(self, *, protocol: ContentUniformCombinationDirectionalProtocol,
        reference_manifest: ContentUniformCombinationManifest,
        probe_manifest: ContentUniformCombinationManifest,
        adapter: CegWmExperimentAdapter, runtime_adapter: Sd35RuntimeAdapter,
        whitening_asset: LfNullWhiteningAsset, method_code_revision: str,
        registered_root_key: str, root_key_public_digest: str,
        protocol_digest: str, execution_intent_authority_digest: str,
        candidate_config_digest: str) -> None:
        protocol.validate()
        reference_manifest.validate(expected_role="content_uniform_combination_reference_fit",expected_count=32)
        probe_manifest.validate(expected_role="content_uniform_combination_directional_probe",expected_count=8)
        if type(adapter) is not CegWmExperimentAdapter or type(runtime_adapter) is not Sd35RuntimeAdapter or type(whitening_asset) is not LfNullWhiteningAsset:
            raise ContentUniformCombinationDirectionalRunnerError("exact public adapters and whitening asset are required")
        if method_code_revision.__class__ is not str or len(method_code_revision)!=40 or protocol_digest!=protocol.digest() or whitening_asset.whitening_asset_digest!=protocol.whitening_asset_digest:
            raise ContentUniformCombinationDirectionalRunnerError("runner frozen authority drifted")
        self.protocol=protocol; self.reference_manifest=reference_manifest; self.probe_manifest=probe_manifest
        self.adapter=adapter; self.runtime=runtime_adapter; self.whitening_asset=whitening_asset
        self.method_code_revision=method_code_revision; self.registered_root_key=registered_root_key
        self.root_key_public_digest=root_key_public_digest; self.protocol_digest=protocol_digest
        self.execution_intent_authority_digest=execution_intent_authority_digest
        self.candidate_config_digest=candidate_config_digest

    def _entry(self, unit_index: int):
        if unit_index==0: return self.reference_manifest.entries[0],"development_environment_preflight"
        if unit_index<33: return self.reference_manifest.entries[unit_index-1],self.reference_manifest.role_id
        return self.probe_manifest.entries[unit_index-33],self.probe_manifest.role_id

    def _analysis_identity(self, unit_index: int) -> AnalysisUnitIdentity:
        unit=self.protocol.unit_roster[unit_index]; entry,role=self._entry(unit_index)
        lineage=entry.image_lineage_digest(role_id=role)
        namespace=self.reference_manifest.key_family_namespace if unit_index<33 else self.probe_manifest.key_family_namespace
        key_family=canonical_digest({"key_family_namespace":namespace,"root_key_public_digest":self.root_key_public_digest})
        cluster=derive_source_cluster_id(prompt_digest=entry.prompt_digest,generation_seed=entry.generation_seed,image_lineage_digest=lineage,registered_key_family_digest=key_family)
        return AnalysisUnitIdentity(unit_id=f"development_unit_{unit_index:04d}",case_id=unit.phase,source_cluster_id=cluster,prompt_digest=entry.prompt_digest,generation_seed=entry.generation_seed,image_lineage_digest=lineage,registered_key_family_digest=key_family)

    def create_persistence_unit_bindings(self) -> tuple[FrozenDevelopmentUnitBinding,...]:
        result=[]
        for unit in self.protocol.unit_roster:
            result.append(create_frozen_development_unit_binding(unit,analysis_unit_identity=self._analysis_identity(unit.unit_index),scientific_question_id="content_uniform_combination_directional_increment",development_case_id="content_embedder_operational_preflight" if unit.unit_index==0 else "content_combination_reference_fit" if unit.unit_index<33 else "six_image_uniform_combination_probe",candidate_identity="content_combination_calibrated",candidate_config_digest=self.candidate_config_digest))
        return tuple(result)

    def execute_operational_unit(self, *, unit_index: int, base_latent: torch.Tensor, intent: UnitIntent) -> DevelopmentOperationalRecord:
        if unit_index!=0 or intent.unit_index!=0: raise ContentUniformCombinationDirectionalRunnerError("operational identity drifted")
        started=monotonic(); shape=tuple(int(value) for value in base_latent.shape)
        route=self.adapter.route_content(shape,mode="routing_uniform_control").result
        low=self.adapter.build_lf_carrier(self.registered_root_key,shape,routing_result=route).result
        high=self.adapter.build_hf_carrier(self.registered_root_key,shape,routing_result=route).result
        captured=[]
        def embed(values):
            result=self.adapter.embed_content(values,high,lf_carrier_result=low,mixing_coefficient=0.50,routing_result=route).result; captured.append(result); return result
        runtime_result=self.runtime.execute_content_write_and_vae(base_latent,embed); elapsed=float(monotonic()-started)
        if len(captured)!=1 or elapsed>intent.maximum_duration_seconds: raise ContentUniformCombinationDirectionalRunnerError("operational execution drifted")
        digest=canonical_development_value_digest({"embedding_result_identity":captured[0].embedding_result_identity,"materialization_replay_identity":runtime_result.content_materialization.materialization_replay_identity,"route_identity":route.route_identity})
        payload={"operational_role":OPERATIONAL_ROLE,"source_cluster_ordinal":0,"case_ids":list(OPERATIONAL_CASE_IDS),"responsibility_result_digests":[["content_embedder",digest]],"elapsed_seconds":elapsed,"runtime_config_digest":runtime_result.runtime_config_digest,"counts_as_scientific_coverage":False,"scientific_claims_supported":False}
        record=DevelopmentOperationalRecord(schema_version=OPERATIONAL_RECORD_SCHEMA,collection_role=OPERATIONAL_RECORD_COLLECTION_ROLE,record_kind=OPERATIONAL_RECORD_KIND,record_id="0"*64,run_id=self.protocol.run_id,protocol_digest=self.protocol_digest,method_code_revision=self.method_code_revision,unit_index=0,phase="development_environment_preflight",source_cluster_ordinal=0,candidate_config_digest=self.candidate_config_digest,attempt_index=intent.attempt_index,retry_parent_intent_digest=intent.parent_attempt_intent_digest,actual_elapsed_seconds=elapsed,maximum_duration_seconds=intent.maximum_duration_seconds,operation_result_payload=payload,counts_as_scientific_coverage=False,scientific_claims_supported=False,scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY)
        record=_record_id(record); record.validate(); return record

    def _detect_branches(self, latent: torch.Tensor, key: object):
        hf_observation,lf_observation=_public_observations(latent)
        hf=self.adapter.detect_hf(hf_observation,key).result
        lf=self.adapter.detect_lf_null_whitened(lf_observation,key,self.whitening_asset).result
        if hf.observation_digest!=lf.observation_digest or lf.whitening_asset_digest!=self.whitening_asset.whitening_asset_digest:
            raise ContentUniformCombinationDirectionalRunnerError("public branch detection identity drifted")
        return hf,lf

    def _metric_observation(self, *, identity: AnalysisUnitIdentity, metric_ids: tuple[str,...], paired: str, branch: str, statistics: tuple[tuple[str,float],...], result_digests: tuple[str,...]) -> dict[str,object]:
        metric={"schema_version":METRIC_SCHEMA_VERSION,"metric_role":"development_exploratory_cluster_level","responsibility_id":"content_detector","source_cluster_id":identity.source_cluster_id,"registered_metric_ids":metric_ids,"candidate_config_digest":self.candidate_config_digest,"paired_ablation_identity":paired,"content_branch_id":branch,"geometry_case_id":"geometry_case_not_applicable","sufficient_statistics":statistics,"result_identity_digests":result_digests,"threshold_role":"not_fitted_content_combination_directional_diagnosis","threshold_identity":None,"threshold_fit_source_cluster_digest":None}
        metric["observation_digest"]=canonical_development_value_digest(metric); return metric

    def _scientific_record(self, *, intent: UnitIntent, identity: AnalysisUnitIdentity, phase: str, case: str, branch: str, paired: str, status: str, failure_class: str|None, failure_reason: str|None, elapsed: float, operation_payload: dict[str,object], metric: dict[str,object]) -> DevelopmentScientificRecord:
        expected = {
            "development_content_combination_reference_fit": (
                "content_combination_reference_fit",
                "paired_clean_branch_null_reference",
                "clean_primary_null_cross_fit_reference",
            ),
            "development_content_uniform_combination_directional_probe": (
                "six_image_uniform_combination_probe",
                "six_image_uniform_combination_probe",
                "same_generation_uniform_route_six_image_control",
            ),
        }
        if expected.get(phase) != (case, branch, paired):
            raise ContentUniformCombinationDirectionalRunnerError(
                "scientific record responsibility identity drifted"
            )
        if (
            intent.phase != phase
            or intent.development_case_id != case
            or intent.content_branch_id != branch
        ):
            raise ContentUniformCombinationDirectionalRunnerError(
                "scientific intent responsibility identity drifted"
            )
        if status == "success":
            if (
                metric.get("paired_ablation_identity") != paired
                or metric.get("content_branch_id") != branch
            ):
                raise ContentUniformCombinationDirectionalRunnerError(
                    "scientific metric responsibility identity drifted"
                )
        elif metric:
            raise ContentUniformCombinationDirectionalRunnerError(
                "failed scientific record carries a metric observation"
            )
        record=DevelopmentScientificRecord(schema_version=RECORD_SCHEMA_VERSION,collection_role=DEVELOPMENT_RECORD_COLLECTION_ROLE,record_id="0"*64,run_id=self.protocol.run_id,protocol_id=self.protocol.protocol_id,protocol_version=self.protocol.protocol_version,protocol_digest=self.protocol_digest,execution_intent_authority_digest=self.execution_intent_authority_digest,method_code_revision=self.method_code_revision,unit_index=intent.unit_index,phase=phase,analysis_unit_identity=asdict(identity),responsibility_id="content_detector",scientific_question_id="content_uniform_combination_directional_increment",development_case_id=case,candidate_identity="content_combination_calibrated",candidate_config_digest=self.candidate_config_digest,paired_ablation_identity=paired,negative_control_case_ids=("paired_clean_primary_null","wrong_key_control"),metric_ids=("content_combination_branch_scores",) if status=="success" else ("content_combination_failure",),content_branch_id=branch,geometry_case_id="geometry_case_not_applicable",attempt_index=intent.attempt_index,execution_status=status,failure_class=failure_class,failure_reason=failure_reason,retry_parent_intent_digest=intent.parent_attempt_intent_digest,actual_elapsed_seconds=elapsed,maximum_duration_seconds=intent.maximum_duration_seconds,duration_limit_exceeded=elapsed>intent.maximum_duration_seconds,operation_result_payload=operation_payload,operation_result_digest=canonical_development_value_digest(operation_payload),metric_observation=metric,routing_trace={"route_identity":"routing_uniform_control"} if status=="success" else {},branch_score_trace=operation_payload if status=="success" else {},detector_trace={"formal_detector_remains_hf_only":True,"diagnostic_candidate":"content_combination_calibrated"} if status=="success" else {},geometry_trace={"geometry_case_id":"geometry_case_not_applicable"} if status=="success" else {},threshold_trace={"formal_tau_created":False} if status=="success" else {},key_control_trace={"root_key_public_digest":self.root_key_public_digest,"wrong_key_count":4} if status=="success" else {},decision_trace={"candidate_promoted":False,"scientific_claims_supported":False} if status=="success" else {},provenance_trace={"protocol_digest":self.protocol_digest,"execution_intent_authority_digest":self.execution_intent_authority_digest,"method_code_revision":self.method_code_revision,"candidate_config_digest":self.candidate_config_digest},module_outcome=None,candidate_recommendation=None,scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY)
        record=_record_id(record); record.validate(); return record

    def execute_reference_fit_unit(self, *, unit_index: int, base_latent: torch.Tensor, intent: UnitIntent) -> DevelopmentScientificRecord:
        if not 1<=unit_index<33 or intent.unit_index!=unit_index: raise ContentUniformCombinationDirectionalRunnerError("reference identity drifted")
        ordinal=unit_index-1; started=monotonic(); result=self.runtime.execute_clean_image_and_vae_observation(base_latent)
        hf,lf=self._detect_branches(result.clean_detection_latent,self.registered_root_key)
        elapsed=float(monotonic()-started); identity=self._analysis_identity(unit_index)
        measurement=create_content_combination_reference_measurement(cluster_ordinal=ordinal,fold_index=ordinal%4,hf_score=hf.hf_score,lf_score=lf.lf_score,hf_detector_identity=hf.detector_identity,lf_detector_identity=lf.detector_identity,whitening_asset_digest=lf.whitening_asset_digest,observation_digest=hf.observation_digest)
        payload={"reference_measurement":asdict(measurement),"clean_image_digest":rgb8_image_digest(_rgb8(result.clean_image)),"runtime_config_digest":result.runtime_config_digest}
        metric=self._metric_observation(identity=identity,metric_ids=("content_combination_branch_scores",),paired="clean_primary_null_cross_fit_reference",branch="paired_clean_branch_null_reference",statistics=(("hf_score",measurement.hf_score),("lf_score",measurement.lf_score)),result_digests=(measurement.measurement_identity,))
        return self._scientific_record(intent=intent,identity=identity,phase="development_content_combination_reference_fit",case="content_combination_reference_fit",branch="paired_clean_branch_null_reference",paired="clean_primary_null_cross_fit_reference",status="success",failure_class=None,failure_reason=None,elapsed=elapsed,operation_payload=payload,metric=metric)

    def _validated_typed_record(
        self,
        record: DevelopmentScientificRecord,
        *,
        reference: bool,
    ) -> DevelopmentScientificRecord:
        if type(record) is not DevelopmentScientificRecord:
            raise ContentUniformCombinationDirectionalRunnerError(
                "exact persistent scientific record type is required"
            )
        record.validate()
        expected = (
            (
                range(1, 33),
                "development_content_combination_reference_fit",
                "content_combination_reference_fit",
                "clean_primary_null_cross_fit_reference",
                "paired_clean_branch_null_reference",
            )
            if reference
            else (
                range(33, 41),
                "development_content_uniform_combination_directional_probe",
                "six_image_uniform_combination_probe",
                "same_generation_uniform_route_six_image_control",
                "six_image_uniform_combination_probe",
            )
        )
        unit_range, phase, case, paired, branch = expected
        if (
            record.run_id != self.protocol.run_id
            or record.protocol_id != self.protocol.protocol_id
            or record.protocol_version != self.protocol.protocol_version
            or record.protocol_digest != self.protocol_digest
            or record.execution_intent_authority_digest
            != self.execution_intent_authority_digest
            or record.method_code_revision != self.method_code_revision
            or record.candidate_identity != "content_combination_calibrated"
            or record.candidate_config_digest != self.candidate_config_digest
            or record.responsibility_id != "content_detector"
            or record.scientific_question_id
            != "content_uniform_combination_directional_increment"
            or record.unit_index not in unit_range
            or record.phase != phase
            or record.development_case_id != case
            or record.paired_ablation_identity != paired
            or record.content_branch_id != branch
            or record.attempt_index != 0
        ):
            raise ContentUniformCombinationDirectionalRunnerError(
                "persistent scientific record authority drifted"
            )
        if record.execution_status == "success":
            expected_keys = (
                {"reference_measurement", "clean_image_digest", "runtime_config_digest"}
                if reference
                else {"combination_observation", "clean_image_digest"}
            )
            if (
                record.failure_class is not None
                or record.failure_reason is not None
                or set(record.operation_result_payload) != expected_keys
                or record.metric_observation.get("paired_ablation_identity") != paired
                or record.metric_observation.get("content_branch_id") != branch
            ):
                raise ContentUniformCombinationDirectionalRunnerError(
                    "successful persistent scientific record payload drifted"
                )
        elif (
            record.execution_status != "failed"
            or record.failure_class not in {"implementation_failure", "resource_failure"}
            or type(record.failure_reason) is not str
            or not record.failure_reason
            or record.operation_result_payload
            or record.metric_observation
        ):
            raise ContentUniformCombinationDirectionalRunnerError(
                "failed persistent scientific record payload drifted"
            )
        return record

    def reference_measurement_from_committed_record(self, record: DevelopmentScientificRecord) -> ContentCombinationReferenceMeasurement:
        checked=self._validated_typed_record(record, reference=True)
        if checked.execution_status!="success": raise ContentUniformCombinationDirectionalRunnerError("reference record is not successful")
        try: measurement=ContentCombinationReferenceMeasurement(**checked.operation_result_payload["reference_measurement"])
        except (KeyError,TypeError) as exc: raise ContentUniformCombinationDirectionalRunnerError("reference payload drifted") from exc
        measurement.validate(); return measurement

    def fit_fold_references(self, records: Sequence[DevelopmentScientificRecord]) -> tuple[ContentCombinationFoldReference,...]:
        measurements=tuple(self.reference_measurement_from_committed_record(record) for record in records)
        return tuple(fit_content_combination_fold_reference(measurements,probe_fold_index=index) for index in range(4))

    @staticmethod
    def _calibrations(reference: ContentCombinationFoldReference) -> tuple[BranchNullCalibration,BranchNullCalibration]:
        hf=BranchNullCalibration(branch="hf",detector_identity=reference.hf_detector_identity,partition_identity=reference.reference_identity,records=tuple(NullScoreRecord(score=score,source_cluster_id=f"reference_cluster_{ordinal:04d}",sample_id=f"hf_reference_{ordinal:04d}") for ordinal,score in zip(reference.source_cluster_ordinals,reference.hf_scores,strict=True)))
        lf=BranchNullCalibration(branch="lf",detector_identity=reference.lf_detector_identity,partition_identity=reference.reference_identity,records=tuple(NullScoreRecord(score=score,source_cluster_id=f"reference_cluster_{ordinal:04d}",sample_id=f"lf_reference_{ordinal:04d}") for ordinal,score in zip(reference.source_cluster_ordinals,reference.lf_scores,strict=True)))
        return hf,lf

    @staticmethod
    def _create_arm_observation(**values: object):
        try:
            return create_content_combination_arm_observation(**values)
        except ContentUniformCombinationDirectionalMetricError as exc:
            _translate_arm_observation_metric_error(exc)
            if type(exc) is not ContentUniformCombinationDirectionalMetricError:
                raise
            raise ContentCombinationArmObservationConstructionError(
                "content combination arm observation construction failed"
            ) from exc

    @staticmethod
    def _create_probe_observation(**values: object):
        try:
            return create_content_uniform_combination_directional_observation(**values)
        except ContentUniformCombinationDirectionalMetricError as exc:
            _translate_arm_observation_metric_error(exc)
            if type(exc) is not ContentUniformCombinationDirectionalMetricError:
                raise
            raise ContentCombinationProbeObservationConstructionError(
                "content combination probe observation construction failed"
            ) from exc

    def _score_rows(self, *, arm_id: str, coefficient: float|None, image: torch.Tensor, latent: torch.Tensor, clean_image: torch.Tensor, clean_latent: torch.Tensor, reference: ContentCombinationFoldReference):
        hf_null,lf_null=self._calibrations(reference)
        controls=[("registered",None,"registered",latent,image,self.registered_root_key),("paired_clean_primary_null",None,"registered",clean_latent,clean_image,self.registered_root_key)]
        controls.extend(("wrong_key_control",index,"wrong",latent,image,derive_wrong_key_material(self.root_key_public_digest,index)) for index in range(4))
        rows=[]
        for role,index,key_role,observed_latent,observed_image,key in controls:
            hf,lf=self._detect_branches(observed_latent,key)
            functions=(("hf_only_standardized_score",None),*(("weighted_hf_lf_standardized_score",weight) for weight in COMBINATION_WEIGHTS),("maximum_hf_lf_standardized_score",None))
            for function,weight in functions:
                if function=="hf_only_standardized_score": result=self.adapter.detect_content(hf,hf_null=hf_null,combination=function).result
                else: result=self.adapter.detect_content(hf,lf,hf_null=hf_null,lf_null=lf_null,combination=function,weight=weight).result
                diagnostic=result.diagnostic_combination
                if diagnostic is None or diagnostic.function_id!=function or diagnostic.weight!=weight or diagnostic.diagnostic_only is not True or diagnostic.promoted is not False:
                    raise ContentUniformCombinationDirectionalRunnerError("diagnostic combination drifted")
                try:
                    row=create_content_combination_score_row(arm_id=arm_id,embedding_coefficient=coefficient,control_role=role,wrong_key_index=index,key_role=key_role,combination_function=function,detector_weight=weight,hf_raw_score=hf.hf_score,lf_raw_score=None if function=="hf_only_standardized_score" else lf.lf_score,hf_standardized_score=diagnostic.hf_standardization.z_score,lf_standardized_score=None if diagnostic.lf_standardization is None else diagnostic.lf_standardization.z_score,content_score=diagnostic.combined_score,content_detector_identity=result.detector_identity,content_config_digest=result.content_config_digest,hf_detector_identity=hf.detector_identity,lf_detector_identity=None if function=="hf_only_standardized_score" else lf.detector_identity,whitening_asset_digest=None if function=="hf_only_standardized_score" else lf.whitening_asset_digest,input_image_digest=rgb8_image_digest(observed_image),hf_observation_digest=hf.observation_digest,lf_observation_digest=None if function=="hf_only_standardized_score" else lf.observation_digest,hf_template_digest=hf.template_digest,lf_template_digest=None if function=="hf_only_standardized_score" else lf.template_digest,root_key_public_digest=hf.root_key_public_digest)
                except ContentUniformCombinationDirectionalMetricError as exc:
                    if type(exc) is not ContentUniformCombinationDirectionalMetricError:
                        raise
                    raise ContentCombinationScoreRowConstructionError(
                        "content combination score row construction failed"
                    ) from exc
                rows.append(row)
        return tuple(rows)

    def execute_probe_unit(self, *, unit_index: int, base_latent: torch.Tensor, intent: UnitIntent, reference_records: Sequence[DevelopmentScientificRecord]) -> DevelopmentScientificRecord:
        if not 33<=unit_index<41 or intent.unit_index!=unit_index: raise ContentUniformCombinationDirectionalRunnerError("probe identity drifted")
        ordinal=unit_index-33; references=self.fit_fold_references(reference_records); reference=references[ordinal%4]
        started=monotonic(); shape=tuple(int(value) for value in base_latent.shape)
        route=self.adapter.route_content(shape,mode="routing_uniform_control").result
        low=self.adapter.build_lf_carrier(self.registered_root_key,shape,routing_result=route).result
        high=self.adapter.build_hf_carrier(self.registered_root_key,shape,routing_result=route).result
        specs=(("hf_only",None,high,None,None),("lf_only",None,None,low,None),("uniform_combined_quarter",0.25,high,low,route),("uniform_combined_half",0.50,high,low,route),("uniform_combined_three_quarters",0.75,high,low,route))
        arms=[]; runtime_results=[]
        for arm_id,coefficient,hf_carrier,lf_carrier,routing in specs:
            captured=[]
            def embed(values, *, hf_carrier=hf_carrier, lf_carrier=lf_carrier, coefficient=coefficient, routing=routing):
                result=self.adapter.embed_content(values,hf_carrier,lf_carrier_result=lf_carrier,mixing_coefficient=coefficient,routing_result=routing).result; captured.append(result); return result
            result=self.runtime.execute_content_write_and_vae(base_latent,embed)
            if len(captured)!=1: raise ContentUniformCombinationDirectionalRunnerError("embedder call count drifted")
            runtime_results.append(result)
        clean_image=_rgb8(runtime_results[0].clean_image); clean_latent=runtime_results[0].clean_detection_latent
        if any(not torch.equal(clean_image,_rgb8(result.clean_image)) or not torch.equal(clean_latent,result.clean_detection_latent) for result in runtime_results[1:]):
            raise ContentUniformCombinationDirectionalRunnerError("same-base clean controls drifted")
        all_rows=[]
        for (arm_id,coefficient,*_),result in zip(specs,runtime_results,strict=True):
            image=_rgb8(result.watermarked_image)
            all_rows.extend(self._score_rows(arm_id=arm_id,coefficient=coefficient,image=image,latent=result.watermarked_detection_latent,clean_image=clean_image,clean_latent=clean_latent,reference=reference))
            arm=self._create_arm_observation(arm_id=arm_id,embedding_coefficient=coefficient,clean_to_watermarked_rgb_relative_l2=_relative_l2(clean_image,image),realized_relative_l2=result.content_materialization.realized_relative_l2,materialization_integrity_status=result.content_materialization.integrity_status,materialization_budget_status=result.content_materialization_result.budget_status,image_digest=rgb8_image_digest(image))
            arms.append(arm)
        elapsed=float(monotonic()-started)
        observation=self._create_probe_observation(cluster_ordinal=ordinal,fold_index=ordinal%4,fold_reference_identity=reference.reference_identity,whitening_asset_digest=self.whitening_asset.whitening_asset_digest,score_rows=tuple(all_rows),arm_observations=tuple(arms),failure_class=None)
        identity=self._analysis_identity(unit_index); payload={"combination_observation":asdict(observation),"clean_image_digest":rgb8_image_digest(clean_image)}
        metric=self._metric_observation(identity=identity,metric_ids=("content_combination_branch_scores",),paired="same_generation_uniform_route_six_image_control",branch="six_image_uniform_combination_probe",statistics=(("score_row_count",float(len(all_rows))),),result_digests=(observation.observation_identity,))
        return self._scientific_record(intent=intent,identity=identity,phase="development_content_uniform_combination_directional_probe",case="six_image_uniform_combination_probe",branch="six_image_uniform_combination_probe",paired="same_generation_uniform_route_six_image_control",status="success",failure_class=None,failure_reason=None,elapsed=elapsed,operation_payload=payload,metric=metric)

    def create_failed_scientific_record(self, *, intent: UnitIntent, failure_class: str, failure_reason: str, elapsed_seconds: float) -> DevelopmentScientificRecord:
        if intent.unit_index not in range(1,41) or failure_class not in {"implementation_failure","resource_failure"}: raise ContentUniformCombinationDirectionalRunnerError("failure identity drifted")
        reference=intent.unit_index<33; identity=self._analysis_identity(intent.unit_index)
        return self._scientific_record(intent=intent,identity=identity,phase="development_content_combination_reference_fit" if reference else "development_content_uniform_combination_directional_probe",case="content_combination_reference_fit" if reference else "six_image_uniform_combination_probe",branch="paired_clean_branch_null_reference" if reference else "six_image_uniform_combination_probe",paired="clean_primary_null_cross_fit_reference" if reference else "same_generation_uniform_route_six_image_control",status="failed",failure_class=failure_class,failure_reason=failure_reason,elapsed=elapsed_seconds,operation_payload={},metric={})

    def observation_from_record(self, record: DevelopmentScientificRecord) -> ContentUniformCombinationDirectionalObservation:
        checked=self._validated_typed_record(record, reference=False); payload=checked.operation_result_payload.get("combination_observation")
        if type(payload) is not dict: raise ContentUniformCombinationDirectionalRunnerError("probe observation payload missing")
        from experiments.metrics.content_uniform_combination_directional_diagnosis import ContentCombinationArmObservation,ContentCombinationScoreRow
        observation=ContentUniformCombinationDirectionalObservation(**{**payload,"score_rows":tuple(ContentCombinationScoreRow(**item) for item in payload["score_rows"]),"arm_observations":tuple(ContentCombinationArmObservation(**item) for item in payload["arm_observations"])})
        observation.validate(); return observation

    def replay_aggregate(self, records: Sequence[DevelopmentScientificRecord], **violations: int) -> ContentUniformCombinationDirectionalAggregate:
        observations=[]
        for record in records:
            checked=self._validated_typed_record(record, reference=False)
            if checked.execution_status=="success": observations.append(self.observation_from_record(checked))
            else: observations.append(create_content_uniform_combination_directional_observation(cluster_ordinal=checked.unit_index-33,fold_index=(checked.unit_index-33)%4,fold_reference_identity="0"*64,whitening_asset_digest="0"*64,score_rows=(),arm_observations=(),failure_class=checked.failure_class))
        return aggregate_content_uniform_combination_directional_diagnosis(observations,**violations)


__all__=[
    "ContentCombinationArmObservationConstructionError",
    "ContentCombinationArmCanonicalBudgetExceededRunnerError",
    "ContentCombinationArmImageDigestInvalidRunnerError",
    "ContentCombinationArmMaterializationRejectedRunnerError",
    "ContentCombinationArmMeasurementNonfiniteRunnerError",
    "ContentCombinationArmObservationIdentityDriftRunnerError",
    "ContentCombinationArmRoleInvalidRunnerError",
    "ContentCombinationProbeObservationConstructionError",
    "ContentCombinationScoreRowConstructionError",
    "ContentUniformCombinationDirectionalDiagnosisRunner",
    "ContentUniformCombinationDirectionalRunnerError",
]
