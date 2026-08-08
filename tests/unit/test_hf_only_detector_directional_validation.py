"""CPU contracts for HF-only detector directional validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.metrics.hf_only_detector_directional_validation import (
    HfDetectorDirectionalMetricError,
    aggregate_hf_detector_direction,
    create_hf_detector_directional_observation,
)
from experiments.protocol.development_records import DevelopmentOperationalRecord
from experiments.protocol.hf_only_detector_directional_validation import (
    load_hf_only_detector_directional_protocol,
)
from experiments.runners.hf_only_detector_directional_validation import (
    HfOnlyDetectorDirectionalRunner,
)
from main import identify_root_key
from runtime import RuntimeDeviceCapabilities, Sd35RuntimeAdapter, create_runtime_adapter
from scripts.experiment_execution import (
    hf_only_detector_directional_validation_entrypoint as directional_entrypoint,
)
from tests.unit.test_runtime_content_write_and_vae import (
    FakeContentBackend,
    _base_latent,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiments/hf_only_detector_directional_validation.json"
COMPONENTS = ROOT / "configs/experiments/internal_execution_components.json"
ROOT_KEY = "ceg-wm-hf-detector-directional-test-key"


class _DirectionalCudaBackend(FakeContentBackend):
    def __init__(self) -> None:
        callbacks = tuple(tuple(range(20)) for _ in range(20))
        super().__init__(callback_sequences=callbacks)  # type: ignore[arg-type]

    def probe_devices(self) -> RuntimeDeviceCapabilities:
        return RuntimeDeviceCapabilities(cpu_available=True, cuda_device_count=1)

    def set_development_generation_prompts(self, _prompt: str) -> None:
        return None


def _runner() -> tuple[HfOnlyDetectorDirectionalRunner, Sd35RuntimeAdapter]:
    protocol, manifest = load_hf_only_detector_directional_protocol(
        PROTOCOL, repository_root=ROOT
    )
    runtime = create_runtime_adapter(
        FakeContentBackend(
            callback_sequences=tuple(tuple(range(20)) for _ in range(4))  # type: ignore[arg-type]
        )
    )
    runtime.initialize("cpu")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS)
    )
    root_public = identify_root_key(ROOT_KEY).root_key_public_digest
    return (
        HfOnlyDetectorDirectionalRunner(
            protocol=protocol,
            manifest=manifest,
            adapter=adapter,
            runtime_adapter=runtime,
            method_code_revision="a" * 40,
            run_id="hf_detector_directional_test",
            registered_root_key=ROOT_KEY,
            root_key_public_digest=root_public,
            protocol_digest=protocol.digest(),
            execution_intent_authority_digest="b" * 64,
            candidate_config_digest="c" * 64,
        ),
        runtime,
    )


@pytest.mark.unit
def test_hf_detector_directional_protocol_freezes_disjoint_budget_and_controls() -> None:
    protocol, manifest = load_hf_only_detector_directional_protocol(
        PROTOCOL, repository_root=ROOT
    )

    assert protocol.operational_unit_count == 2
    assert protocol.scientific_cluster_count == 32
    assert protocol.initial_gpu_gate_scientific_unit_count == 8
    assert protocol.maximum_total_units == 34
    assert len(manifest.operational_entries) == 2
    assert len(manifest.scientific_entries) == 32
    assert len({item.prompt_digest for item in manifest.entries}) == 34
    assert len({item.generation_seed for item in manifest.entries}) == 34
    assert tuple(item.unit_index for item in protocol.unit_roster) == tuple(range(34))
    assert tuple(
        item.source_cluster_ordinal for item in protocol.unit_roster[2:]
    ) == tuple(range(32))
    assert protocol.wrong_key_roster_size == 4
    assert protocol.practical_margin_floor == 0.001
    assert protocol.minimum_registered_minus_null_success_count == 28
    assert protocol.minimum_registered_minus_wrong_success_count == 28


@pytest.mark.unit
def test_hf_detector_directional_runner_calls_public_blind_detector_for_paired_rgb() -> None:
    runner, runtime = _runner()
    operational = runner.execute_operational_smoke(
        unit_index=0,
        base_latent=_base_latent(),
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
    )
    scientific = runner.execute_scientific_cluster(
        cluster_ordinal=5,
        base_latent=_base_latent(),
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
    )
    runtime.close()

    assert type(operational) is DevelopmentOperationalRecord
    assert operational.counts_as_scientific_coverage is False
    assert operational.scientific_claims_supported is False
    observation = scientific.operation_result_payload["directional_observation"]
    assert observation["wrong_key_index"] == 1
    assert scientific.key_control_trace["wrong_key_roster_size"] == 4
    assert scientific.detector_trace["detector_operation_identity"] == "main.hf_detector"
    assert scientific.detector_trace["same_image_registered_wrong_reuse"] is True
    assert scientific.detector_trace["paired_clean_primary_null"] is True
    assert scientific.detector_trace["reference_image_used"] is False
    assert scientific.detector_trace["embed_record_used"] is False
    assert scientific.detector_trace["private_latent_used_by_detector"] is False
    assert scientific.threshold_trace["raw_threshold_identity"] is None
    assert scientific.module_outcome is None
    assert scientific.candidate_recommendation is None


def _observation(index: int, *, success: bool = True):
    registered = 0.002 if success else -0.002
    return create_hf_detector_directional_observation(
        cluster_ordinal=index,
        wrong_key_index=index % 4,
        registered_score=registered,
        wrong_key_score=0.0,
        primary_null_score=0.0,
        candidate_observation_digest=f"{index + 1:064x}",
        clean_observation_digest=f"{index + 101:064x}",
        registered_detector_identity=f"registered_detector_{index}",
        wrong_key_detector_identity=f"wrong_detector_{index}",
        primary_null_detector_identity=f"null_detector_{index}",
        detector_config_digest="d" * 64,
        observation_protocol="hf_public_image_encoding_v1",
        registered_template_digest="e" * 64,
        wrong_key_template_digest=f"{index + 201:064x}",
        primary_null_template_digest="e" * 64,
        registered_root_key_public_digest="f" * 64,
        wrong_key_root_key_public_digest="f" * 64,
        primary_null_root_key_public_digest="f" * 64,
        materialization_integrity_status="passed",
        realized_relative_l2=0.011,
        content_relative_l2_limit=3 / 250,
        rgb_paired_relative_l2=0.01,
    )


@pytest.mark.unit
def test_hf_detector_directional_gate_uses_floor_exact_bounds_and_full_denominator() -> None:
    approved = aggregate_hf_detector_direction(
        tuple(_observation(index, success=index < 28) for index in range(32)),
        failed_cluster_count=0,
    )
    blocked = aggregate_hf_detector_direction(
        tuple(_observation(index, success=index < 27) for index in range(32)),
        failed_cluster_count=0,
    )

    assert approved.registered_minus_primary_null.practical_success_count == 28
    assert approved.registered_minus_wrong_key.practical_success_count == 28
    assert approved.registered_minus_primary_null.exact_one_sided_confidence_lower_bound > 0.5
    assert approved.registered_minus_wrong_key.exact_one_sided_confidence_lower_bound > 0.5
    assert approved.allow_request_for_next_scientific_gate is True
    assert blocked.allow_request_for_next_scientific_gate is False
    assert approved.registered_minus_primary_null.lower_quartile_nearest_rank_margin == pytest.approx(0.002)
    assert approved.registered_minus_primary_null.threshold_free_paired_ranking_auc == pytest.approx(28 / 32)
    with pytest.raises(HfDetectorDirectionalMetricError, match="coverage is incomplete"):
        aggregate_hf_detector_direction(
            tuple(_observation(index) for index in range(31)),
            failed_cluster_count=0,
        )


@pytest.mark.unit
def test_hf_detector_directional_entrypoint_commits_operational_and_scientific_roster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "directional_package.zip"
    package.write_bytes(b"directional-package")
    backend = _DirectionalCudaBackend()
    monkeypatch.setattr(
        directional_entrypoint,
        "Sd35PipelineBackend",
        lambda **_kwargs: backend,
    )
    initialize_runtime = Sd35RuntimeAdapter.initialize
    monkeypatch.setattr(
        Sd35RuntimeAdapter,
        "initialize",
        lambda self, _device: initialize_runtime(self, "cpu"),
    )
    monkeypatch.setattr(
        directional_entrypoint,
        "_build_or_verify_package",
        lambda *_args: package,
    )
    monkeypatch.setattr(
        directional_entrypoint,
        "_base_latent",
        lambda generation_seed, **_kwargs: (
            _base_latent().to(torch.float32)
            + float(generation_seed % 97) / 10000.0
        ).to(torch.float16),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Test GPU")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _index: 1)

    exit_code, result = (
        directional_entrypoint.execute_hf_only_detector_directional_validation_session(
            repository_root=ROOT,
            expected_revision="a" * 40,
            persistent_root=tmp_path / "persistent",
            cache_root=tmp_path / "cache",
            run_id="hf_detector_directional_entrypoint_test",
            session_id="directional_entrypoint_session",
            environment={"HF_TOKEN": "test-token", "CEG_WM_ROOT_KEY": ROOT_KEY},
            authorized_scientific_unit_count=8,
        )
    )

    assert exit_code == 0
    assert result["termination_reason"] == "authorized_directional_unit_boundary_reached"
    assert result["committed_unit_count"] == 10
    assert result["session_committed_unit_count"] == 10
    assert result["directional_aggregate"] is None
    assert result["authorized_scientific_unit_count"] == 8
    assert result["formal_tau_created"] is False
    assert result["fpr_estimated"] is False
    assert result["candidate_promoted"] is False
    assert len(tuple((tmp_path / "persistent" / "hf_detector_directional_entrypoint_test" / "markers").glob("*.COMMITTED.json"))) == 10
