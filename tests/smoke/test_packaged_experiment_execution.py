"""Package-contained A3a execution check using no repository or Git state."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from math import exp, log, sqrt
from pathlib import Path

import pytest

from experiments.attacks import GeometricAttackSpec
from experiments.metrics import (
    RectificationMetricCase,
    aggregate_rectification_delta,
    load_metric_registry,
)
from experiments.protocol.internal_splits import AnalysisUnitIdentity
from scripts.experiment_execution.experiment_execution_entrypoint import (
    run_synthetic_wiring,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.smoke
def test_packaged_entrypoint_executes_and_replays_synthetic_paths(
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        (ROOT / "experiment_execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    summary = run_synthetic_wiring(
        package_root=ROOT,
        output_root=tmp_path / "result",
        workspace_root=tmp_path / "workspace",
        committed_revision=manifest["committed_revision"],
        expected_candidate_config_digest=(
            manifest["candidate_config_digest"]
        ),
        expected_execution_config_digest=(
            manifest["execution_config_digest"]
        ),
        expected_input_manifest_digest=(
            manifest["input_manifest_digest"]
        ),
        run_id="packaged-smoke",
    )

    assert summary["run_status"] == "completed"
    assert summary["record_count"] == 5
    assert summary["success_count"] == 1
    assert summary["resource_failure_count"] == 2
    assert summary["scientific_failure_count"] == 1
    assert summary["execution_failure_count"] == 1
    assert summary["scientific_claims_supported"] is False
    assert summary["gpu_executed"] is False
    assert summary["held_out_evaluation_accessed"] is False

    result_root = tmp_path / "result"
    record_path = result_root / summary["record_collection_relative_path"]
    collection = json.loads(record_path.read_text(encoding="utf-8"))
    records = collection["records"]
    by_unit = {}
    for record in records:
        by_unit.setdefault(
            record["analysis_unit_identity"]["unit_id"],
            [],
        ).append(record)

    geometry_success = by_unit["synthetic_wiring_unit_0"]
    assert len(geometry_success) == 1
    assert geometry_success[0]["execution_status"] == "success"
    assert geometry_success[0]["geometry_trace"]["geometry_triggered"]
    assert geometry_success[0]["geometry_trace"][
        "geometry_estimation_identity"
    ]
    assert geometry_success[0]["geometry_trace"]["geometry_reliable"] is True
    assert any(
        value != 0.0
        for value in geometry_success[0]["geometry_trace"][
            "geometry_transform"
        ].values()
    )
    assert geometry_success[0]["provenance_trace"][
        "attack_config_digest"
    ] == GeometricAttackSpec(
        "scale",
        scale_factor=exp(-log(sqrt(2.0)) / 2.0),
    ).attack_config_digest

    resource_lineage = by_unit["synthetic_wiring_unit_1"]
    assert [record["execution_status"] for record in resource_lineage] == [
        "failed",
        "retry",
        "failed",
    ]
    assert [
        record["failure_class"] for record in resource_lineage[:2]
    ] == ["resource_failure", "resource_failure"]
    assert resource_lineage[1]["retry_of_record_id"] == (
        resource_lineage[0]["record_id"]
    )
    assert resource_lineage[2]["retry_of_record_id"] == (
        resource_lineage[1]["record_id"]
    )
    assert resource_lineage[2]["failure_class"] == "scientific_failure"
    assert resource_lineage[2]["geometry_trace"]["geometry_triggered"]

    scientific = resource_lineage[2]
    execution = by_unit["synthetic_wiring_unit_2"][0]
    assert scientific["failure_class"] == "scientific_failure"
    assert scientific["geometry_trace"]["geometry_triggered"]
    assert execution["failure_class"] == "execution_failure"
    assert not execution["geometry_trace"]["geometry_triggered"]

    metric_registry = load_metric_registry(
        ROOT / "configs/experiments/internal_execution_components.json"
    )
    assert all(
        record["provenance_trace"]["metric_set_digest"]
        == metric_registry.registry_digest
        for record in records
    )
    successful = [
        record
        for record in records
        if record["execution_status"] == "success"
        and record["detector_trace"]["rectified_content_score"] is not None
    ]
    metric_cases = tuple(
        RectificationMetricCase(
            analysis_unit_identity=AnalysisUnitIdentity(
                **record["analysis_unit_identity"]
            ),
            split=record["split"],
            raw_detector_identity=record["detector_trace"][
                "raw_detector_identity"
            ],
            rectified_detector_identity=record["detector_trace"][
                "rectified_detector_identity"
            ],
            raw_threshold_identity=record["threshold_trace"][
                "raw_threshold_identity"
            ],
            rectified_threshold_identity=record["threshold_trace"][
                "rectified_threshold_identity"
            ],
            raw_score=record["detector_trace"]["raw_content_score"],
            rectified_score=record["detector_trace"][
                "rectified_content_score"
            ],
        )
        for record in successful
    )
    aggregate = aggregate_rectification_delta(
        metric_cases,
        registry=metric_registry,
    )
    expected_case_results = [
        {
            **asdict(result),
            "record_id": record["record_id"],
            "canonical_record_digest": sha256(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        }
        for record, result in zip(successful, aggregate.cases, strict=True)
    ]
    assert summary["entrypoint_schema_version"] == 2
    assert summary["metric_registry_digest"] == metric_registry.registry_digest
    assert summary["metric_evaluator_identity"] == (
        "experiments.metrics.aggregate_rectification_delta"
    )
    assert summary["metric_aggregate_identity"] == (
        "experiments.metrics.RectificationDeltaAggregate"
    )
    assert summary["metric_case_results"] == expected_case_results
    assert summary["metric_aggregate_values"] == {
        "case_count": len(aggregate.cases),
        "split": aggregate.split,
        "mean_score_delta": aggregate.mean_score_delta,
        "improved_fraction": aggregate.improved_fraction,
        "detector_identity": aggregate.detector_identity,
        "threshold_identity": aggregate.threshold_identity,
    }
    metric_evidence = {
        "metric_aggregate_identity": summary["metric_aggregate_identity"],
        "metric_aggregate_values": summary["metric_aggregate_values"],
        "metric_case_results": summary["metric_case_results"],
        "metric_evaluator_identity": summary["metric_evaluator_identity"],
        "metric_registry_digest": summary["metric_registry_digest"],
    }
    assert summary["metric_evidence_digest"] == sha256(
        json.dumps(
            metric_evidence,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert len(summary["replay_digest"]) == 64
