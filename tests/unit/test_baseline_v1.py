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
        continuous_score=0.25,
        score_direction="higher_is_watermarked",
        threshold_provenance="tree_ring:calibration:unresolved",
        decision=True,
        quality={"clip": 0.8},
        runtime_seconds=1.0,
        status="observed",
        failure_reason=None,
        artifact_digests={"image": "sha256:image"},
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
def test_observation_requires_own_identities_and_threshold_provenance() -> None:
    assert validate_observation(_record()).as_dict()["baseline_id"] == "tree_ring"
    with pytest.raises(ValueError, match="threshold provenance"):
        validate_observation(_record(threshold_provenance=None, decision=True))
    with pytest.raises(ValueError, match="source, adapter, and threshold"):
        validate_observation(_record(adapter_exact=None))


@pytest.mark.unit
def test_failed_unit_remains_a_record_with_artifact_identity() -> None:
    failed = _record(
        source_exact=None,
        adapter_exact=None,
        threshold_provenance=None,
        decision=None,
        status="failed",
        failure_reason="external source not available",
    )
    assert validate_observation(failed).status == "failed"
    with pytest.raises(ValueError, match="artifact_digests"):
        validate_observation(replace(failed, artifact_digests={}))
