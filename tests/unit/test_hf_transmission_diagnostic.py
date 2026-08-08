"""CPU contracts for the real HF transport diagnostic execution path."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.metrics.hf_transmission_diagnostic import (
    create_hf_signal_position_observation,
    evaluate_hf_transmission_direction,
)
from experiments.protocol.hf_transmission_diagnostic import (
    SIGNAL_POSITIONS,
    load_hf_transmission_protocol,
)
from experiments.runners.hf_transmission_diagnostic import (
    HfTransmissionDiagnosticRunner,
)
from main import identify_root_key
from runtime import create_runtime_adapter
from tests.unit.test_runtime_content_write_and_vae import FakeContentBackend


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiments/hf_transmission_diagnostic.json"
COMPONENTS = ROOT / "configs/experiments/internal_execution_components.json"
ROOT_KEY = "ceg-wm-hf-transmission-diagnostic-test-key"


@pytest.mark.unit
def test_hf_transmission_protocol_freezes_isolated_eight_cluster_roster() -> None:
    protocol, manifest = load_hf_transmission_protocol(
        PROTOCOL, repository_root=ROOT
    )

    assert protocol.split == "development"
    assert protocol.role_id == "hf_transmission_diagnostic"
    assert len(manifest.entries) == 8
    assert {item.role_id for item in manifest.entries} == {
        "hf_transmission_diagnostic"
    }
    assert len(protocol.unit_roster) == 10
    assert tuple(item.unit_index for item in protocol.unit_roster) == tuple(range(10))
    assert sum(
        item.phase == "development_environment_preflight"
        for item in protocol.unit_roster
    ) == 2
    assert sum(item.responsibility_id == "hf_detector" for item in protocol.unit_roster) == 8


@pytest.mark.unit
def test_hf_transport_runner_separates_latent_diagnostics_from_formal_detector() -> None:
    protocol, manifest = load_hf_transmission_protocol(
        PROTOCOL, repository_root=ROOT
    )
    backend = FakeContentBackend()
    runtime = create_runtime_adapter(backend)
    runtime.initialize("cpu")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS)
    )
    root_public = identify_root_key(ROOT_KEY).root_key_public_digest
    runner = HfTransmissionDiagnosticRunner(
        protocol=protocol,
        manifest=manifest,
        adapter=adapter,
        runtime_adapter=runtime,
        method_code_revision="a" * 40,
        run_id="hf_transmission_diagnostic_test",
        registered_root_key=ROOT_KEY,
        root_key_public_digest=root_public,
        protocol_digest=protocol.digest(),
        execution_intent_authority_digest="b" * 64,
        candidate_config_digest="c" * 64,
    )
    base = torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4).to(torch.float16)

    record = runner.execute_scientific_cluster(
        cluster_ordinal=0,
        base_latent=base,
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
    )

    observations = record.operation_result_payload["signal_positions"]
    assert tuple(item["position_id"] for item in observations) == SIGNAL_POSITIONS
    assert tuple(item["statistic_role"] for item in observations[:3]) == (
        "diagnostic_latent_template_projection",
    ) * 3
    assert observations[-1]["statistic_role"] == "formal_hf_detector_operation"
    assert record.detector_trace["same_image_registered_wrong_reuse"] is True
    assert record.detector_trace["paired_clean_primary_null"] is True
    assert record.threshold_trace["raw_threshold_identity"] is None
    assert record.threshold_trace["rectified_threshold_identity"] is None
    assert record.module_outcome is None
    assert record.candidate_recommendation is None
    runtime.close()


@pytest.mark.unit
def test_hf_transport_directional_rule_is_exactly_seven_of_eight() -> None:
    observations = tuple(
        create_hf_signal_position_observation(
            position_id="rgb_vae_reencoded",
            statistic_role="formal_hf_detector_operation",
            registered_score=1.0 if index < 7 else -1.0,
            wrong_key_score=0.0,
            primary_null_score=0.0,
            registered_observation_digest=f"{index + 1:064x}",
            primary_null_observation_digest=f"{index + 101:064x}",
            registered_statistic_identity="registered_detector",
            wrong_key_statistic_identity="wrong_detector",
            primary_null_statistic_identity="null_detector",
        )
        for index in range(8)
    )

    approved = evaluate_hf_transmission_direction(
        observations, budget_integrity_nonfinite_failure_count=0
    )
    blocked = evaluate_hf_transmission_direction(
        observations, budget_integrity_nonfinite_failure_count=1
    )

    assert approved.registered_minus_wrong_positive_count == 7
    assert approved.registered_minus_null_positive_count == 7
    assert approved.allow_request_for_next_scientific_gate is True
    assert blocked.allow_request_for_next_scientific_gate is False
