"""Create the Phase-B diagnostic-only soft-detector asset bundle once."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from experiments.metrics.lf_whitened_score_screening import (
    fit_semantic_texture_lf_whitening_asset,
    semantic_texture_clean_null_band_energy_sums,
)
from experiments.methods.ceg_wm import (
    CegWmExperimentAdapter,
    serialize_semantic_texture_soft_detector_asset_bundle,
)
from experiments.protocol.semantic_texture_soft_detector_assets import (
    BRANCH_NULL_COUNT,
    BRANCH_NULL_ROLE,
    WHITENING_FIT_COUNT,
    WHITENING_FIT_ROLE,
    SemanticTextureSoftDetectorAssetBundle,
    SemanticTextureSoftDetectorAssetProtocolError,
    load_manifest,
    validate_partition_disjointness,
)
from main import (
    NullScoreRecord,
    SemanticTextureBranchNullCalibration,
    SemanticTextureLfWhiteningAsset,
)
from runtime import InspyrenetSemanticRuntime


class SemanticTextureSoftDetectorAssetPreparationError(RuntimeError):
    """The frozen asset-preparation roster cannot complete safely."""


@dataclass(frozen=True, slots=True)
class SemanticTextureSoftDetectorAssetPreparationConfiguration:
    whitening_fit_manifest_path: str
    branch_null_manifest_path: str
    target_prompt: str
    target_seed: int


def prepare_semantic_texture_soft_detector_assets(
    adapter: CegWmExperimentAdapter,
    configuration: SemanticTextureSoftDetectorAssetPreparationConfiguration,
    *,
    repository_root: str | Path,
    detection_key: str,
    semantic_runtime: InspyrenetSemanticRuntime,
    base_latent_for_seed: Callable[[int], torch.Tensor],
    set_generation_prompt: Callable[[str], None],
) -> SemanticTextureSoftDetectorAssetBundle:
    """Run exactly the frozen 32+32 clean operational diagnostic roster."""

    root = Path(repository_root)
    try:
        whitening_manifest = load_manifest(
            root / configuration.whitening_fit_manifest_path,
            expected_role=WHITENING_FIT_ROLE,
            count=WHITENING_FIT_COUNT,
        )
        branch_manifest = load_manifest(
            root / configuration.branch_null_manifest_path,
            expected_role=BRANCH_NULL_ROLE,
            count=BRANCH_NULL_COUNT,
        )
        validate_partition_disjointness(
            whitening_manifest,
            branch_manifest,
            target_prompt=configuration.target_prompt,
            target_seed=configuration.target_seed,
        )
    except SemanticTextureSoftDetectorAssetProtocolError as exc:
        raise SemanticTextureSoftDetectorAssetPreparationError(
            "Phase-B manifest authority is invalid"
        ) from exc

    energy_rows: list[tuple[float, ...]] = []
    carrier_digest: str | None = None
    for entry in whitening_manifest.entries:
        set_generation_prompt(entry.prompt_text)
        prepared = adapter.prepare_semantic_texture_clean_primary_null(
            base_latent_for_seed(entry.generation_seed),
            detection_key,
            semantic_runtime,
        )
        current_digest = prepared.lf_carrier_config_digest
        if carrier_digest is None:
            carrier_digest = current_digest
        elif carrier_digest != current_digest:
            raise SemanticTextureSoftDetectorAssetPreparationError(
                "Phase-B LF carrier identity drifted"
            )
        energy_rows.append(
            semantic_texture_clean_null_band_energy_sums(
                prepared.runtime_detection.lf_observation.values,
                prepared.routing_result.mask_lf,
            )
        )
    if carrier_digest is None:
        raise SemanticTextureSoftDetectorAssetPreparationError("Phase-B W fit is empty")
    fit = fit_semantic_texture_lf_whitening_asset(
        energy_rows,
        fit_manifest_sha256=whitening_manifest.digest(),
        lf_carrier_config_digest=carrier_digest,
    )
    try:
        whitening_asset = SemanticTextureLfWhiteningAsset.from_canonical_payload(
            fit.canonical_payload,
            whitening_asset_digest=fit.whitening_asset_digest,
        )
        whitening_asset.validate()
    except Exception as exc:
        raise SemanticTextureSoftDetectorAssetPreparationError(
            "Phase-B W construction is invalid"
        ) from exc

    hf_records: list[NullScoreRecord] = []
    lf_records: list[NullScoreRecord] = []
    hf_identity: str | None = None
    lf_identity: str | None = None
    for entry in branch_manifest.entries:
        set_generation_prompt(entry.prompt_text)
        prepared = adapter.prepare_semantic_texture_clean_primary_null(
            base_latent_for_seed(entry.generation_seed),
            detection_key,
            semantic_runtime,
        )
        branches = adapter.observe_semantic_texture_primary_null_branches(
            prepared,
            detection_key,
            whitening_asset,
        )
        if hf_identity is None:
            hf_identity = branches.hf_result.detector_identity
            lf_identity = branches.lf_result.detector_identity
        elif (
            hf_identity != branches.hf_result.detector_identity
            or lf_identity != branches.lf_result.detector_identity
        ):
            raise SemanticTextureSoftDetectorAssetPreparationError(
                "Phase-B branch detector identity drifted"
            )
        hf_records.append(
            NullScoreRecord(
                score=branches.hf_result.hf_score,
                source_cluster_id=entry.source_cluster_id,
                sample_id=entry.image_lineage_digest,
            )
        )
        lf_records.append(
            NullScoreRecord(
                score=branches.lf_result.lf_score,
                source_cluster_id=entry.source_cluster_id,
                sample_id=entry.image_lineage_digest,
            )
        )
    if hf_identity is None or lf_identity is None:
        raise SemanticTextureSoftDetectorAssetPreparationError("Phase-B CDF is empty")
    partition_identity = branch_manifest.digest()
    try:
        hf_null = SemanticTextureBranchNullCalibration(
            branch="hf", detector_identity=hf_identity, partition_identity=partition_identity,
            records=tuple(hf_records),
        )
        lf_null = SemanticTextureBranchNullCalibration(
            branch="lf", detector_identity=lf_identity, partition_identity=partition_identity,
            records=tuple(lf_records),
        )
        return serialize_semantic_texture_soft_detector_asset_bundle(
            whitening_manifest_digest=whitening_manifest.digest(),
            branch_null_manifest_digest=partition_identity,
            whitening_asset=whitening_asset,
            hf_null=hf_null,
            lf_null=lf_null,
        )
    except Exception as exc:
        raise SemanticTextureSoftDetectorAssetPreparationError(
            "Phase-B asset bundle construction is invalid"
        ) from exc


__all__ = [
    "SemanticTextureSoftDetectorAssetPreparationConfiguration",
    "SemanticTextureSoftDetectorAssetPreparationError",
    "prepare_semantic_texture_soft_detector_assets",
]
