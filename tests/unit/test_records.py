from __future__ import annotations

import pytest

from cegwm.protocol.records import StageARecord


@pytest.mark.unit
def test_success_record_keeps_public_identity_and_scores() -> None:
    record = StageARecord(
        run_id="stage-a-001",
        unit_id="unit-0001",
        source_cluster_id="cluster-0001",
        arm="lf_only",
        condition="identity",
        code_revision="0123456789abcdef",
        config_digest="a" * 64,
        key_public_digest="b" * 64,
        status="success",
        scores={"registered": 1.25, "wrong_max": 0.10},
        metrics={"paired_rgb8_mse": 0.0001},
    )

    payload = record.to_dict()

    assert payload["status"] == "success"
    assert payload["scores"] == {"registered": 1.25, "wrong_max": 0.10}
    assert "root_key" not in payload
    assert "derived_key" not in payload


@pytest.mark.unit
def test_failed_record_requires_bounded_reason() -> None:
    with pytest.raises(ValueError, match="require failure_reason"):
        StageARecord(
            run_id="stage-a-001",
            unit_id="unit-0002",
            source_cluster_id="cluster-0002",
            arm="lf_only",
            condition="blur_sigma_1",
            code_revision="0123456789abcdef",
            config_digest="a" * 64,
            key_public_digest="b" * 64,
            status="operational_failure",
        )


@pytest.mark.unit
def test_success_record_rejects_failure_reason() -> None:
    with pytest.raises(ValueError, match="cannot carry"):
        StageARecord(
            run_id="stage-a-001",
            unit_id="unit-0003",
            source_cluster_id="cluster-0003",
            arm="hf_only",
            condition="identity",
            code_revision="0123456789abcdef",
            config_digest="a" * 64,
            key_public_digest="b" * 64,
            status="success",
            failure_reason="unexpected",
        )
