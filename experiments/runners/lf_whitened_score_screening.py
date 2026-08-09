"""Real clean-null fit and public blind LF score screening runner."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import isfinite
from time import monotonic
from typing import Sequence

import torch

from experiments.methods import CegWmExperimentAdapter
from experiments.metrics.lf_whitened_score_screening import (
    LfWhitenedScreeningDecision,
    LfWhitenedScreeningObservation,
    clean_null_band_energy_sums,
    create_lf_whitened_screening_observation,
    evaluate_lf_whitened_screening,
    fit_lf_null_whitening_asset,
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
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    derive_source_cluster_id,
)
from experiments.protocol.lf_whitened_score_screening import (
    LfWhitenedScoreScreeningProtocol,
    LfWhiteningManifest,
    canonical_digest,
    derive_lf_whitening_analysis_identity,
)
from experiments.runners.development_persistence import (
    CommittedUnit,
    FrozenDevelopmentUnitBinding,
    create_frozen_development_unit_binding,
)
from main import LfDetectionObservation, LfNullWhiteningAsset, derive_wrong_key_material
from runtime import CleanImageVaeObservationResult, ContentWriteVaeResult, Sd35RuntimeAdapter


class LfWhitenedScoreRunnerError(RuntimeError):
    """The LF whitening fit or screening execution violated its frozen contract."""


def _tensor_values(tensor: torch.Tensor) -> tuple[float, ...]:
    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != (1, 16, 64, 64)
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise LfWhitenedScoreRunnerError("public LF observation tensor is invalid")
    return tuple(
        float(item)
        for item in tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    )


def _observation(tensor: torch.Tensor) -> LfDetectionObservation:
    return LfDetectionObservation.from_public_image_encoding(
        _tensor_values(tensor), (1, 16, 64, 64)
    )


class LfWhitenedScoreScreeningRunner:
    """Fit one public W asset, then compare raw and whitened public LF scores."""

    def __init__(
        self,
        *,
        protocol: LfWhitenedScoreScreeningProtocol,
        null_fit_manifest: LfWhiteningManifest,
        screening_manifest: LfWhiteningManifest,
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
        null_fit_manifest.validate(expected_role="lf_whitening_null_fit", count=32)
        screening_manifest.validate(
            expected_role="lf_whitened_score_screening", count=8
        )
        if type(adapter) is not CegWmExperimentAdapter or type(runtime_adapter) is not Sd35RuntimeAdapter:
            raise LfWhitenedScoreRunnerError("exact method and runtime adapters are required")
        if type(method_code_revision) is not str or len(method_code_revision) != 40:
            raise LfWhitenedScoreRunnerError("method revision is invalid")
        self.protocol = protocol
        self.null_fit_manifest = null_fit_manifest
        self.screening_manifest = screening_manifest
        self.adapter = adapter
        self.runtime = runtime_adapter
        self.method_code_revision = method_code_revision
        self.run_id = run_id
        self.registered_root_key = registered_root_key
        self.root_key_public_digest = root_key_public_digest
        self.protocol_digest = protocol_digest
        self.execution_intent_authority_digest = execution_intent_authority_digest
        self.candidate_config_digest = candidate_config_digest
        self.fit_key_family_digest = canonical_digest(
            {
                "manifest_digest": null_fit_manifest.digest(),
                "role": "key_free_clean_public_null_fit",
            }
        )
        self.screening_key_family_digest = canonical_digest(
            {
                "manifest_digest": screening_manifest.digest(),
                "root_key_public_digest": root_key_public_digest,
                "role": "registered_lf_whitened_screening_key_family",
            }
        )

    def _analysis_identity(self, unit_index: int) -> AnalysisUnitIdentity:
        if unit_index == 0:
            key_family_digest = canonical_digest(
                {
                    "root_key_public_digest": self.root_key_public_digest,
                    "role": "registered_lf_clean_runtime_preflight_key_family",
                    "run_id": self.run_id,
                }
            )
            return AnalysisUnitIdentity(
                unit_id="lf_clean_public_vae_runtime_preflight",
                case_id="clean_public_vae_runtime_preflight",
                source_cluster_id=derive_source_cluster_id(
                    prompt_digest=self.protocol.operational_smoke_prompt_digest,
                    generation_seed=(
                        self.protocol.operational_smoke_generation_seed
                    ),
                    image_lineage_digest=(
                        self.protocol.operational_smoke_image_lineage_digest
                    ),
                    registered_key_family_digest=key_family_digest,
                ),
                prompt_digest=self.protocol.operational_smoke_prompt_digest,
                generation_seed=self.protocol.operational_smoke_generation_seed,
                image_lineage_digest=(
                    self.protocol.operational_smoke_image_lineage_digest
                ),
                registered_key_family_digest=key_family_digest,
            )
        if unit_index <= 32:
            return derive_lf_whitening_analysis_identity(
                self.null_fit_manifest.entries[unit_index - 1],
                self.null_fit_manifest,
                key_family_digest=self.fit_key_family_digest,
            )
        ordinal = unit_index - 33
        return derive_lf_whitening_analysis_identity(
            self.screening_manifest.entries[ordinal],
            self.screening_manifest,
            key_family_digest=self.screening_key_family_digest,
        )

    def create_persistence_unit_bindings(
        self,
    ) -> tuple[FrozenDevelopmentUnitBinding, ...]:
        bindings: list[FrozenDevelopmentUnitBinding] = []
        for unit in self.protocol.unit_roster:
            operational = unit.unit_index == 0
            fit = 1 <= unit.unit_index <= 32
            bindings.append(
                create_frozen_development_unit_binding(
                    unit,
                    analysis_unit_identity=self._analysis_identity(unit.unit_index),
                    scientific_question_id=(
                        "lf_clean_public_vae_runtime_preflight"
                        if operational
                        else (
                            "lf_clean_null_whitening_asset_fit"
                            if fit
                            else "lf_raw_whitened_key_attribution_screening"
                        )
                    ),
                    development_case_id=(
                        "clean_public_vae_runtime_preflight"
                        if operational
                        else (
                            "clean_public_vae_null_fit"
                            if fit
                            else "paired_clean_lf_raw_whitened_screening"
                        )
                    ),
                    candidate_identity=self.protocol.candidate_identity,
                    candidate_config_digest=self.candidate_config_digest,
                )
            )
        return tuple(bindings)

    def _record(
        self,
        *,
        unit_index: int,
        attempt_index: int,
        retry_parent_intent_digest: str | None,
        maximum_duration_seconds: int,
        actual_elapsed_seconds: float,
        operation_payload: dict[str, object],
        metric_statistics: tuple[tuple[str, float], ...],
        result_identity_digests: tuple[str, ...],
        success: bool,
        failure_class: str | None = None,
        failure_reason: str | None = None,
        retry: bool = False,
    ) -> DevelopmentScientificRecord:
        fit = 1 <= unit_index <= 32
        identity = self._analysis_identity(unit_index)
        responsibility = (
            "lf_whitening_null_fit" if fit else "lf_whitened_score_screening"
        )
        scientific_question = (
            "lf_clean_null_whitening_asset_fit"
            if fit
            else "lf_raw_whitened_key_attribution_screening"
        )
        development_case = (
            "clean_public_vae_null_fit"
            if fit
            else "paired_clean_lf_raw_whitened_screening"
        )
        paired = (
            "independent_clean_public_null_observation"
            if fit
            else "paired_clean_lf_same_generation"
        )
        negative_controls = (
            ("key_free_clean_public_null",)
            if fit
            else ("same_image_four_wrong_keys", "paired_clean_primary_null")
        )
        metric_ids = (
            ("clean_null_band_energy",)
            if fit
            else ("raw_whitened_wrong_key_attribution", "paired_clean_primary_null")
        )
        content_branch = "clean_control" if fit else "lf_only"
        metric_payload: dict[str, object] = {}
        if success:
            metric_payload = {
                "schema_version": METRIC_SCHEMA_VERSION,
                "metric_role": "development_exploratory_cluster_level",
                "responsibility_id": responsibility,
                "source_cluster_id": identity.source_cluster_id,
                "registered_metric_ids": metric_ids,
                "candidate_config_digest": self.candidate_config_digest,
                "paired_ablation_identity": paired,
                "content_branch_id": content_branch,
                "geometry_case_id": "not_applicable",
                "sufficient_statistics": metric_statistics,
                "result_identity_digests": result_identity_digests,
                "threshold_role": "not_fitted_lf_whitened_screening",
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
            "unit_index": unit_index,
            "phase": "development_scientific_breadth",
            "analysis_unit_identity": asdict(identity),
            "responsibility_id": responsibility,
            "scientific_question_id": scientific_question,
            "development_case_id": development_case,
            "candidate_identity": self.protocol.candidate_identity,
            "candidate_config_digest": self.candidate_config_digest,
            "paired_ablation_identity": paired,
            "negative_control_case_ids": negative_controls,
            "metric_ids": metric_ids,
            "content_branch_id": content_branch,
            "geometry_case_id": "not_applicable",
            "attempt_index": attempt_index,
            "execution_status": (
                "success" if success else ("retry" if retry else "failed")
            ),
            "failure_class": None if success else failure_class,
            "failure_reason": None if success else failure_reason,
            "retry_parent_intent_digest": retry_parent_intent_digest,
            "actual_elapsed_seconds": actual_elapsed_seconds,
            "maximum_duration_seconds": maximum_duration_seconds,
            "duration_limit_exceeded": actual_elapsed_seconds > maximum_duration_seconds,
            "operation_result_payload": operation_payload,
            "operation_result_digest": canonical_development_value_digest(
                operation_payload
            ),
            "metric_observation": metric_payload,
            "routing_trace": {"routing_used": False},
            "branch_score_trace": (
                {} if fit else operation_payload.get("score_summary", {})
            ),
            "detector_trace": (
                {"formal_detector_completed": False, "fit_uses_detection_key": False}
                if fit
                else {
                    "public_callable": "main.lf_null_whitened_matched_detector",
                    "raw_control_public_callable": "main.lf_detector",
                    "same_image_registered_wrong_reuse": True,
                    "paired_clean_primary_null": True,
                }
            ),
            "geometry_trace": {"geometry_attempted": False},
            "threshold_trace": {
                "threshold_role": "not_fitted_lf_whitened_screening",
                "raw_threshold_identity": None,
                "rectified_threshold_identity": None,
            },
            "key_control_trace": {
                "root_key_public_digest": (
                    None if fit else self.root_key_public_digest
                ),
                "wrong_key_indexes": () if fit else (0, 1, 2, 3),
                "raw_secret_persisted": False,
            },
            "decision_trace": {
                "positive_source": None,
                "decision_role": (
                    "clean_null_fit_observation_only"
                    if fit
                    else "directional_screening_observation_only"
                ),
            },
            "provenance_trace": {
                "protocol_digest": self.protocol_digest,
                "execution_intent_authority_digest": self.execution_intent_authority_digest,
                "method_code_revision": self.method_code_revision,
                "candidate_config_digest": self.candidate_config_digest,
                "manifest_digest": (
                    self.null_fit_manifest.digest()
                    if fit
                    else self.screening_manifest.digest()
                ),
                "cluster_identity": (
                    self.null_fit_manifest.entries[unit_index - 1].cluster_identity
                    if fit
                    else self.screening_manifest.entries[unit_index - 33].cluster_identity
                ),
            },
            "module_outcome": None,
            "candidate_recommendation": None,
            "scientific_claim_boundary": DEVELOPMENT_CLAIM_BOUNDARY,
        }
        draft = DevelopmentScientificRecord(**record_payload)
        record = DevelopmentScientificRecord(
            **{
                **record_payload,
                "record_id": canonical_development_value_digest(
                    draft.payload_without_record_id()
                ),
            }
        )
        record.validate()
        return record

    def execute_operational_smoke(
        self,
        *,
        base_latent: torch.Tensor,
        attempt_index: int,
        retry_parent_intent_digest: str | None,
        maximum_duration_seconds: int,
        started_monotonic: float | None = None,
    ) -> DevelopmentOperationalRecord:
        """Exercise the clean public RGB-to-VAE API without scientific coverage."""

        started = monotonic() if started_monotonic is None else started_monotonic
        result = self.runtime.execute_clean_image_and_vae_observation(base_latent)
        if type(result) is not CleanImageVaeObservationResult:
            raise LfWhitenedScoreRunnerError(
                "clean operational runtime result exact type is required"
            )
        observation = _observation(result.clean_detection_latent)
        shape = tuple(int(size) for size in base_latent.shape)
        carrier = self.adapter.build_lf_carrier(
            self.registered_root_key, shape
        ).result
        embedding = self.adapter.embed_content(
            _tensor_values(base_latent), lf_carrier_result=carrier
        ).result
        elapsed = float(monotonic() - started)
        payload = {
            "operational_role": "environment_runtime_throughput_preflight",
            "source_cluster_ordinal": 0,
            "case_ids": ["clean_public_rgb_vae_observation_smoke"],
            "responsibility_result_digests": [
                [
                    "content_embedder",
                    canonical_digest(
                        {
                            "embedding_result_identity": (
                                embedding.embedding_result_identity
                            ),
                            "observation_digest": observation.observation_digest,
                            "observation_protocol": observation.observation_protocol,
                            "runtime_config_digest": result.runtime_config_digest,
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
        record = replace(
            record,
            record_id=canonical_development_value_digest(
                record.payload_without_record_id()
            ),
        )
        record.validate()
        return record

    def execute_null_fit_cluster(
        self,
        *,
        cluster_ordinal: int,
        base_latent: torch.Tensor,
        attempt_index: int,
        retry_parent_intent_digest: str | None,
        maximum_duration_seconds: int,
        prior_verified_fit_evidence: Sequence[
            tuple[DevelopmentScientificRecord, CommittedUnit]
        ] = (),
        started_monotonic: float | None = None,
    ) -> DevelopmentScientificRecord:
        if type(cluster_ordinal) is not int or not 0 <= cluster_ordinal < 32:
            raise LfWhitenedScoreRunnerError("null-fit cluster is outside manifest")
        if len(tuple(prior_verified_fit_evidence)) != cluster_ordinal:
            raise LfWhitenedScoreRunnerError("null-fit evidence is not a frozen prefix")
        started = monotonic() if started_monotonic is None else started_monotonic
        result = self.runtime.execute_clean_image_and_vae_observation(base_latent)
        if type(result) is not CleanImageVaeObservationResult:
            raise LfWhitenedScoreRunnerError("clean runtime result exact type is required")
        energy = clean_null_band_energy_sums(
            _tensor_values(result.clean_detection_latent)
        )
        operation_payload: dict[str, object] = {
            "runtime_candidate_id": result.candidate_id,
            "runtime_config_digest": result.runtime_config_digest,
            "selected_device": result.selected_device,
            "clean_base_latent_digest": result.clean_base_latent_digest,
            "clean_observation_protocol": "final_image_vae_posterior_mode",
            "clean_observation_digest": sha256(
                json.dumps(_tensor_values(result.clean_detection_latent)).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "clean_null_band_energy_sums": energy,
            "whitening_asset_payload": None,
            "whitening_asset_digest": None,
            "fit_manifest_file_sha256": self.protocol.null_fit_manifest_file_sha256,
        }
        if cluster_ordinal == 31:
            rows: list[tuple[float, ...]] = []
            for expected, (record, _marker) in enumerate(
                prior_verified_fit_evidence, start=1
            ):
                if (
                    record.unit_index != expected
                    or record.execution_status != "success"
                    or record.responsibility_id != "lf_whitening_null_fit"
                ):
                    raise LfWhitenedScoreRunnerError(
                        "verified null-fit evidence is incomplete"
                    )
                row = record.operation_result_payload.get(
                    "clean_null_band_energy_sums"
                )
                if type(row) not in {tuple, list}:
                    raise LfWhitenedScoreRunnerError(
                        "verified null-fit statistic is missing"
                    )
                rows.append(tuple(float(value) for value in row))
            fit_result = fit_lf_null_whitening_asset(
                (*rows, energy),
                fit_manifest_sha256=self.protocol.null_fit_manifest_file_sha256,
            )
            asset = LfNullWhiteningAsset.from_canonical_payload(
                fit_result.canonical_payload,
                whitening_asset_digest=fit_result.whitening_asset_digest,
            )
            asset.validate()
            operation_payload["whitening_asset_payload"] = asset.canonical_payload
            operation_payload["whitening_asset_digest"] = asset.whitening_asset_digest
        elapsed = float(monotonic() - started)
        return self._record(
            unit_index=1 + cluster_ordinal,
            attempt_index=attempt_index,
            retry_parent_intent_digest=retry_parent_intent_digest,
            maximum_duration_seconds=maximum_duration_seconds,
            actual_elapsed_seconds=elapsed,
            operation_payload=operation_payload,
            metric_statistics=tuple(
                (f"channel_band_energy_{index:02d}", value)
                for index, value in enumerate(energy)
            ),
            result_identity_digests=(operation_payload["clean_observation_digest"],),
            success=True,
        )

    def replay_whitening_asset(
        self,
        verified_evidence: Sequence[
            tuple[DevelopmentScientificRecord, CommittedUnit]
        ],
    ) -> LfNullWhiteningAsset:
        evidence = tuple(verified_evidence)
        fit = tuple(
            item for item in evidence if 1 <= item[0].unit_index <= 32
        )
        if len(fit) != 32 or tuple(
            record.unit_index for record, _ in fit
        ) != tuple(range(1, 33)):
            raise LfWhitenedScoreRunnerError("verified null-fit coverage is incomplete")
        rows: list[tuple[float, ...]] = []
        for record, marker in fit:
            record.validate()
            if record.execution_status != "success":
                raise LfWhitenedScoreRunnerError(
                    "verified null-fit contains a terminal failure"
                )
            if record.responsibility_id != "lf_whitening_null_fit":
                raise LfWhitenedScoreRunnerError(
                    "verified null-fit responsibility drifted"
                )
            if (
                record.protocol_digest != self.protocol_digest
                or marker.protocol_digest != self.protocol_digest
            ):
                raise LfWhitenedScoreRunnerError(
                    "verified null-fit protocol binding drifted"
                )
            if record.candidate_config_digest != self.candidate_config_digest:
                raise LfWhitenedScoreRunnerError(
                    "verified null-fit candidate binding drifted"
                )
            if marker.revision != self.method_code_revision:
                raise LfWhitenedScoreRunnerError(
                    "verified null-fit revision binding drifted"
                )
            values = record.operation_result_payload.get(
                "clean_null_band_energy_sums"
            )
            if type(values) not in {tuple, list}:
                raise LfWhitenedScoreRunnerError("verified null-fit statistic is missing")
            rows.append(tuple(float(value) for value in values))
        expected_fit = fit_lf_null_whitening_asset(
            rows,
            fit_manifest_sha256=self.protocol.null_fit_manifest_file_sha256,
        )
        expected_asset = LfNullWhiteningAsset.from_canonical_payload(
            expected_fit.canonical_payload,
            whitening_asset_digest=expected_fit.whitening_asset_digest,
        )
        expected_asset.validate()
        final_record = fit[-1][0]
        payload = final_record.operation_result_payload.get(
            "whitening_asset_payload"
        )
        digest = final_record.operation_result_payload.get(
            "whitening_asset_digest"
        )
        if (
            type(payload) is not dict
            or digest != expected_asset.whitening_asset_digest
        ):
            raise LfWhitenedScoreRunnerError("committed whitening asset is missing")
        replayed = LfNullWhiteningAsset.from_canonical_payload(
            payload, whitening_asset_digest=digest
        )
        replayed.validate()
        if replayed.canonical_payload != expected_asset.canonical_payload:
            raise LfWhitenedScoreRunnerError("committed whitening asset drifted")
        return replayed

    def _execute_paired_runtime(
        self, *, base_latent: torch.Tensor
    ) -> ContentWriteVaeResult:
        shape = tuple(int(size) for size in base_latent.shape)
        carrier = self.adapter.build_lf_carrier(
            self.registered_root_key, shape
        ).result

        def embed(values: tuple[float, ...]):
            return self.adapter.embed_content(
                values, lf_carrier_result=carrier
            ).result

        result = self.runtime.execute_content_write_and_vae(base_latent, embed)
        if type(result) is not ContentWriteVaeResult:
            raise LfWhitenedScoreRunnerError("paired runtime result exact type is required")
        return result

    def execute_screening_cluster(
        self,
        *,
        cluster_ordinal: int,
        base_latent: torch.Tensor,
        whitening_asset: LfNullWhiteningAsset,
        attempt_index: int,
        retry_parent_intent_digest: str | None,
        maximum_duration_seconds: int,
        started_monotonic: float | None = None,
    ) -> DevelopmentScientificRecord:
        if type(cluster_ordinal) is not int or not 0 <= cluster_ordinal < 8:
            raise LfWhitenedScoreRunnerError("screening cluster is outside manifest")
        started = monotonic() if started_monotonic is None else started_monotonic
        runtime_result = self._execute_paired_runtime(base_latent=base_latent)
        materialization = runtime_result.content_materialization_result
        if materialization.integrity_status != "passed" or materialization.budget_status != "accepted":
            raise LfWhitenedScoreRunnerError("LF screening materialization failed")
        candidate = _observation(runtime_result.watermarked_detection_latent)
        clean = _observation(runtime_result.clean_detection_latent)
        raw_registered = self.adapter.detect_lf(
            candidate, self.registered_root_key
        ).result
        raw_null = self.adapter.detect_lf(clean, self.registered_root_key).result
        whitened_registered = self.adapter.detect_lf_null_whitened(
            candidate, self.registered_root_key, whitening_asset
        ).result
        whitened_null = self.adapter.detect_lf_null_whitened(
            clean, self.registered_root_key, whitening_asset
        ).result
        wrong_keys = tuple(
            derive_wrong_key_material(self.root_key_public_digest, index)
            for index in range(4)
        )
        raw_wrong = tuple(
            self.adapter.detect_lf(candidate, key).result for key in wrong_keys
        )
        whitened_wrong = tuple(
            self.adapter.detect_lf_null_whitened(
                candidate, key, whitening_asset
            ).result
            for key in wrong_keys
        )
        if len({item.detector_config_digest for item in (raw_registered, raw_null, *raw_wrong)}) != 1:
            raise LfWhitenedScoreRunnerError("raw detector configuration drifted")
        if len({item.detector_config_digest for item in (whitened_registered, whitened_null, *whitened_wrong)}) != 1:
            raise LfWhitenedScoreRunnerError("whitened detector configuration drifted")
        observation = create_lf_whitened_screening_observation(
            cluster_ordinal=cluster_ordinal,
            raw_registered_score=raw_registered.lf_score,
            raw_primary_null_score=raw_null.lf_score,
            raw_wrong_key_scores=tuple(item.lf_score for item in raw_wrong),
            whitened_registered_score=whitened_registered.lf_score,
            whitened_primary_null_score=whitened_null.lf_score,
            whitened_wrong_key_scores=tuple(
                item.lf_score for item in whitened_wrong
            ),
            whitening_asset_digest=whitening_asset.whitening_asset_digest,
            raw_detector_config_digest=raw_registered.detector_config_digest,
            whitened_detector_config_digest=(
                whitened_registered.detector_config_digest
            ),
        )
        operation_payload = {
            "screening_observation": asdict(observation),
            "raw_detection_results": {
                "registered": asdict(raw_registered),
                "primary_null": asdict(raw_null),
                "wrong_keys": tuple(asdict(item) for item in raw_wrong),
            },
            "whitened_detection_results": {
                "registered": asdict(whitened_registered),
                "primary_null": asdict(whitened_null),
                "wrong_keys": tuple(asdict(item) for item in whitened_wrong),
            },
            "whitening_asset_digest": whitening_asset.whitening_asset_digest,
            "realized_total_l2": materialization.realized_total_l2,
            "realized_relative_l2": materialization.realized_relative_l2,
            "materialization_scale": materialization.materialization_scale,
            "materialization_attempt_count": materialization.attempt_count,
            "budget_status": materialization.budget_status,
            "integrity_status": materialization.integrity_status,
            "score_summary": {
                "whitened_registered_minus_primary_null": observation.whitened_registered_minus_primary_null,
                "whitened_registered_minus_max_wrong": observation.whitened_registered_minus_max_wrong,
                "raw_registered_minus_max_wrong": observation.raw_registered_minus_max_wrong,
                "raw_to_whitened_wrong_margin_improvement": observation.raw_to_whitened_wrong_margin_improvement,
            },
        }
        elapsed = float(monotonic() - started)
        return self._record(
            unit_index=33 + cluster_ordinal,
            attempt_index=attempt_index,
            retry_parent_intent_digest=retry_parent_intent_digest,
            maximum_duration_seconds=maximum_duration_seconds,
            actual_elapsed_seconds=elapsed,
            operation_payload=operation_payload,
            metric_statistics=(
                ("whitened_registered_minus_primary_null", observation.whitened_registered_minus_primary_null),
                ("whitened_registered_minus_max_wrong", observation.whitened_registered_minus_max_wrong),
                ("raw_registered_minus_max_wrong", observation.raw_registered_minus_max_wrong),
                ("raw_to_whitened_wrong_margin_improvement", observation.raw_to_whitened_wrong_margin_improvement),
                ("realized_relative_l2", materialization.realized_relative_l2),
            ),
            result_identity_digests=(observation.observation_identity,),
            success=True,
        )

    def create_failed_record(
        self,
        *,
        unit_index: int,
        attempt_index: int,
        retry_parent_intent_digest: str | None,
        maximum_duration_seconds: int,
        actual_elapsed_seconds: float,
        failure_stage: str,
        resource_failure: bool,
        retryable_resource_failure: bool,
    ) -> DevelopmentScientificRecord:
        if not 1 <= unit_index <= 40:
            raise LfWhitenedScoreRunnerError("failed unit is outside frozen roster")
        operation = {
            "failure_stage": failure_stage,
            "result_available": False,
        }
        return self._record(
            unit_index=unit_index,
            attempt_index=attempt_index,
            retry_parent_intent_digest=retry_parent_intent_digest,
            maximum_duration_seconds=maximum_duration_seconds,
            actual_elapsed_seconds=actual_elapsed_seconds,
            operation_payload=operation,
            metric_statistics=(),
            result_identity_digests=(),
            success=False,
            failure_class=(
                "resource_failure"
                if resource_failure
                else "implementation_failure"
            ),
            failure_reason=f"{failure_stage}_failed",
            retry=retryable_resource_failure,
        )

    def replay_screening_decision(
        self,
        verified_evidence: Sequence[
            tuple[DevelopmentScientificRecord, CommittedUnit]
        ],
    ) -> LfWhitenedScreeningDecision:
        evidence = tuple(verified_evidence)
        screening = tuple(
            item for item in evidence if item[0].unit_index >= 33
        )
        if len(screening) != 8 or tuple(
            record.unit_index for record, _ in screening
        ) != tuple(range(33, 41)):
            raise LfWhitenedScoreRunnerError("verified screening coverage is incomplete")
        observations: list[LfWhitenedScreeningObservation] = []
        failures = 0
        asset = self.replay_whitening_asset(evidence)
        for ordinal, (record, marker) in enumerate(screening):
            record.validate()
            if (
                record.protocol_digest != self.protocol_digest
                or record.candidate_config_digest != self.candidate_config_digest
                or marker.protocol_digest != self.protocol_digest
                or marker.revision != self.method_code_revision
            ):
                raise LfWhitenedScoreRunnerError("verified screening binding drifted")
            if record.execution_status != "success":
                failures += 1
                continue
            raw = record.operation_result_payload.get("screening_observation")
            if type(raw) is not dict:
                raise LfWhitenedScoreRunnerError("verified screening observation is missing")
            try:
                observation = LfWhitenedScreeningObservation(
                    **{
                        **raw,
                        "raw_wrong_key_scores": tuple(raw["raw_wrong_key_scores"]),
                        "whitened_wrong_key_scores": tuple(raw["whitened_wrong_key_scores"]),
                    }
                )
            except (KeyError, TypeError) as exc:
                raise LfWhitenedScoreRunnerError("screening observation schema drifted") from exc
            observation.validate()
            if (
                observation.cluster_ordinal != ordinal
                or observation.whitening_asset_digest != asset.whitening_asset_digest
            ):
                raise LfWhitenedScoreRunnerError("screening observation binding drifted")
            observations.append(observation)
        return evaluate_lf_whitened_screening(
            observations,
            integrity_failure_count=failures,
            margin_floor=self.protocol.margin_floor,
        )


__all__ = ["LfWhitenedScoreRunnerError", "LfWhitenedScoreScreeningRunner"]
