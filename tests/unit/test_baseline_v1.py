from __future__ import annotations

from dataclasses import replace

import pytest

from cegwm.baselines import PRIMARY_BASELINES, BaselineObservation, baseline_by_id, validate_observation


def _record(**changes: object) -> BaselineObservation:
    record = BaselineObservation(
        baseline_id="tree_ring",
        source_exact="source-commit",
        adapter_exact="adapter-commit",
        prompt_id="prompt-001",
        seed=7,
        base_latent_commitment="sha256:latent",
        split="test",
        sample_role="watermarked_correct_key",
        attack_id="clean",
        continuous_score=None,
        score_direction=None,
        threshold_provenance=None,
        decision=None,
        quality={"clip": 0.8},
        runtime_seconds=1.0,
        status="not_available",
        failure_reason=None,
        artifact_digests={"status": "sha256:" + "0" * 64},
    )
    return replace(record, **changes)


@pytest.mark.unit
def test_registry_contains_only_the_authorized_four_methods() -> None:
    assert [item.baseline_id for item in PRIMARY_BASELINES] == [
        "tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark"
    ]
    assert all(item.result_status == "not_available" for item in PRIMARY_BASELINES)
    assert all(not item.paper_claim_support for item in PRIMARY_BASELINES)
    assert baseline_by_id("t2smark").sd35_path == "official_run_sd35_native_path"
    with pytest.raises(ValueError, match="out-of-scope"):
        baseline_by_id("image_domain_method")


@pytest.mark.unit
def test_unresolved_methods_cannot_emit_observed_or_placeholder_evidence() -> None:
    assert validate_observation(_record()).as_dict()["baseline_id"] == "tree_ring"
    with pytest.raises(ValueError, match="score direction is unresolved"):
        validate_observation(_record(
            continuous_score=0.25,
            score_direction="higher_is_watermarked",
            threshold_provenance="tree_ring:calibration:future",
            decision=True,
            status="observed",
        ))
    with pytest.raises(ValueError, match="cannot carry detection evidence"):
        validate_observation(_record(continuous_score=0.25))
    with pytest.raises(ValueError, match="source, adapter, and threshold"):
        _record(
            continuous_score=0.25,
            score_direction="higher_is_watermarked",
            threshold_provenance="tree_ring:calibration:future",
            decision=True,
            status="observed",
        ).as_dict()
    with pytest.raises(ValueError, match="validated source and adapter registry"):
        _record(
            source_exact="0" * 40,
            adapter_exact="1" * 40,
            continuous_score=0.25,
            score_direction="higher_is_watermarked",
            threshold_provenance="tree_ring:calibration:future",
            decision=True,
            status="observed",
            artifact_digests={
                "source": "sha256:" + "0" * 64,
                "adapter": "sha256:" + "1" * 64,
                "threshold": "sha256:" + "2" * 64,
            },
        ).as_dict()


@pytest.mark.unit
def test_failed_unit_remains_a_record_with_artifact_identity() -> None:
    failed = _record(
        source_exact=None,
        adapter_exact=None,
        continuous_score=None,
        score_direction=None,
        threshold_provenance=None,
        decision=None,
        status="failed",
        failure_reason="external source not available",
    )
    assert validate_observation(failed).status == "failed"
    with pytest.raises(ValueError, match="artifact_digests"):
        validate_observation(replace(failed, artifact_digests={}))
    with pytest.raises(ValueError, match="sha256"):
        validate_observation(replace(failed, artifact_digests={"log": "not-a-digest"}))
    with pytest.raises(ValueError, match="cannot carry detection evidence"):
        validate_observation(replace(failed, continuous_score=0.25))
