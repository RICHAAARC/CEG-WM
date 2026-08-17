"""Neutral replay support for persisted diagnostic records."""

from __future__ import annotations

from math import isfinite
from typing import Mapping, Sequence

from experiments.protocol.internal_splits import AnalysisUnitIdentity
from main import BranchNullCalibration, NullScoreRecord


class DevelopmentInputError(RuntimeError):
    """A replay-derived fit input is unavailable."""


def replay_branch_null_calibration(
    evidence: Sequence[tuple[object, object]],
    *,
    branch: str,
    current_source_cluster_id: str,
    source_cluster_ordinals: Mapping[str, int],
    fold_count: int = 4,
) -> BranchNullCalibration:
    if (
        current_source_cluster_id not in source_cluster_ordinals
        or type(fold_count) is not int
        or fold_count < 2
    ):
        raise DevelopmentInputError("development null cross-fit identity is invalid")
    current_fold = source_cluster_ordinals[current_source_cluster_id] % fold_count
    responsibility = f"{branch}_detector"
    records: list[NullScoreRecord] = []
    detector_identity: str | None = None
    for record, _marker in evidence:
        if record.responsibility_id != responsibility:
            continue
        if record.content_branch_id != "clean_control" or record.execution_status != "success":
            continue
        identity = AnalysisUnitIdentity(**record.analysis_unit_identity)
        ordinal = source_cluster_ordinals.get(identity.source_cluster_id)
        if ordinal is None:
            raise DevelopmentInputError(
                "committed primary-null cluster is outside development manifest"
            )
        if ordinal % fold_count == current_fold:
            continue
        result = record.operation_result_payload
        score = result.get(f"{branch}_score")
        observed_identity = result.get("detector_identity")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not isfinite(float(score)):
            raise DevelopmentInputError("committed primary-null score is invalid")
        if type(observed_identity) is not str or not observed_identity:
            raise DevelopmentInputError("committed detector identity is invalid")
        if detector_identity is not None and detector_identity != observed_identity:
            raise DevelopmentInputError("committed primary-null detector identity drifted")
        detector_identity = observed_identity
        records.append(
            NullScoreRecord(
                float(score),
                identity.source_cluster_id,
                f"{responsibility}_{record.unit_index:04d}",
            )
        )
    if detector_identity is None or len(records) < 2:
        raise DevelopmentInputError("verified COMMITTED primary-null evidence is incomplete")
    return BranchNullCalibration(
        branch=branch,
        detector_identity=detector_identity,
        partition_identity=(
            "development_exploratory_primary_null_cross_fit_fold_"
            + str(current_fold)
        ),
        records=tuple(records),
    )
